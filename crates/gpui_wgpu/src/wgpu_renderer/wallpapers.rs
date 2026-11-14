// Added by HexpyGames for Butler (2026), under the Apache License 2.0; see NOTICE.
//! Procedural wallpapers ([`gpui::WallpaperPaint`]): each pass runs the `wallpaper` pipeline into
//! a cached texture of its own size (time-dependent passes re-run when their clock slot changes),
//! and the last texture is composited into the rounded bounds with the blur-composite pipeline.
use std::cell::RefCell;
use std::collections::HashMap;

use gpui::{BackdropFilter, WallpaperImage, WallpaperPaint};
use gpui_render::{
    blur::{FilterCompositeClip, ScissorRectangle},
    shaders::blur::BlurUniforms,
};

use super::WgpuRenderer;

/// The engine noise (lowbias32 hash, butler noise.ts) and the print noise (xorshift, glPrint.ts),
/// 256 x 256 RGBA8 each, bit-exact with the product's textures.
pub(super) fn wallpaper_noise(print: bool) -> Vec<u8> {
    let mut data = vec![0u8; 256 * 256 * 4];
    let mut s: u32 = 0x2f6b1d;
    for (index, byte) in data.iter_mut().enumerate() {
        if print {
            // riso-core.js: s ^= s << 13; s >>>= 0; s ^= s >> 17 (a signed shift); s ^= s << 5
            s ^= s << 13;
            s ^= ((s as i32) >> 17) as u32;
            s ^= s << 5;
            *byte = (s & 255) as u8;
        } else {
            let mut x = index as u32 ^ 0x5eed_b17e;
            x ^= x >> 16;
            x = x.wrapping_mul(0x7feb_352d);
            x ^= x >> 15;
            x = x.wrapping_mul(0x846c_a68b);
            x ^= x >> 16;
            *byte = (x >> 24) as u8;
        }
    }
    data
}

struct Entry {
    generation: u64,
    slot: Option<u32>,
    passes: Vec<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
    image: Option<(u64, wgpu::TextureView)>,
    last_used: u64,
}

#[derive(Default)]
pub(super) struct WallpaperCache {
    noise: Option<[wgpu::TextureView; 2]>,
    entries: HashMap<u64, Entry>,
    images: HashMap<u64, (usize, wgpu::TextureView)>,
    sampler: Option<wgpu::Sampler>,
    tick: u64,
}

const MAX_ENTRIES: usize = 8;

fn noise_texture(device: &wgpu::Device, queue: &wgpu::Queue, print: bool) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("wallpaper_noise"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &wallpaper_noise(print),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(256 * 4),
            rows_per_image: Some(256),
        },
        size,
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

impl WgpuRenderer {
    pub(super) fn draw_wallpaper(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        filter: &BackdropFilter,
        paint: &WallpaperPaint,
        scissor: Option<ScissorRectangle>,
    ) {
        if paint.passes.is_empty() {
            return;
        }
        let format = self.target.format();
        // Collect the views to render into / sample (cloned out, so the cache borrow ends here).
        let slot = paint.clock.map(|clock| clock.slot(self.animation_time_ms));
        let (noise, views, image, sampler, redraw_from) = {
            let resources = self.resources();
            let mut cache = resources.wallpapers.borrow_mut();
            if cache.noise.is_none() {
                cache.noise = Some([
                    noise_texture(&resources.device, &resources.queue, false),
                    noise_texture(&resources.device, &resources.queue, true),
                ]);
            }
            if cache.sampler.is_none() {
                cache.sampler = Some(resources.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("wallpaper_sampler"),
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::MipmapFilterMode::Linear,
                    ..Default::default()
                }));
            }
            cache.tick += 1;
            let tick = cache.tick;
            if !cache.entries.contains_key(&paint.key) && cache.entries.len() >= MAX_ENTRIES {
                if let Some(oldest) = cache
                    .entries
                    .iter()
                    .min_by_key(|(_, e)| e.last_used)
                    .map(|(k, _)| *k)
                {
                    let removed = cache.entries.remove(&oldest).expect("oldest wallpaper");
                    if let Some((id, _)) = removed.image {
                        cache.release_image(id);
                    }
                }
            }
            let previous_image_id = cache
                .entries
                .get(&paint.key)
                .and_then(|entry| entry.image.as_ref().map(|image| image.0));
            let image_changed = previous_image_id != paint.image.as_ref().map(|image| image.id);
            let image = if image_changed {
                let uploaded = match &paint.image {
                    Some(image) => {
                        if let std::collections::hash_map::Entry::Vacant(slot) =
                            cache.images.entry(image.id)
                        {
                            let view = image_texture(&resources.device, &resources.queue, image);
                            slot.insert((0, view));
                        }
                        let cached = cache.images.get_mut(&image.id).expect("uploaded image");
                        cached.0 += 1;
                        Some((image.id, cached.1.clone()))
                    }
                    None => None,
                };
                if let Some(id) = previous_image_id {
                    cache.release_image(id);
                }
                uploaded
            } else {
                None
            };
            let entry = cache.entries.entry(paint.key).or_insert_with(|| Entry {
                generation: u64::MAX,
                slot: None,
                passes: Vec::new(),
                image: None,
                last_used: 0,
            });
            entry.last_used = tick;
            if image_changed {
                entry.image = image;
                entry.generation = u64::MAX;
            }
            let fits = entry.passes.len() == paint.passes.len()
                && entry
                    .passes
                    .iter()
                    .zip(&paint.passes)
                    .all(|(e, p)| e.0 == p.width && e.1 == p.height);
            if !fits {
                entry.passes = paint
                    .passes
                    .iter()
                    .map(|p| {
                        let (texture, view) = super::resources::sampled_texture(
                            &resources.device,
                            format,
                            p.width,
                            p.height,
                        );
                        (p.width, p.height, texture, view)
                    })
                    .collect();
                entry.generation = u64::MAX;
            }
            let redraw_from = if entry.generation != paint.generation {
                0
            } else if entry.slot != slot {
                paint
                    .clock
                    .map_or(paint.passes.len(), |clock| clock.animated_from)
            } else {
                paint.passes.len()
            };
            entry.generation = paint.generation;
            entry.slot = slot;
            let views: Vec<wgpu::TextureView> = entry.passes.iter().map(|e| e.3.clone()).collect();
            let image = entry.image.as_ref().map(|image| image.1.clone());
            (
                cache.noise.clone().expect("noise"),
                views,
                image,
                cache.sampler.clone().expect("sampler"),
                redraw_from,
            )
        };
        for (index, pass) in paint.passes.iter().enumerate().skip(redraw_from) {
            let mut uniforms = BlurUniforms::wallpaper_pass(pass);
            if let Some(clock) = paint.clock.filter(|clock| index >= clock.animated_from) {
                uniforms.wp_frame.w = clock.time_s(slot.expect("wallpaper slot"));
            }
            let noise_view = &noise[pass.noise as usize];
            let previous = if pass.previous {
                &views[index - 1]
            } else {
                noise_view
            };
            let image_view = if pass.image {
                image.as_ref().expect("wallpaper image")
            } else {
                noise_view
            };
            let resources = self.resources();
            let uniform_offset = resources.filter_uniforms.write(&uniforms);
            let bind_group = resources
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("wallpaper_inputs"),
                    layout: &resources.bind_group_layouts.wallpaper,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &resources.filter_uniforms.buffer,
                                offset: 0,
                                size: std::num::NonZeroU64::new(
                                    std::mem::size_of::<BlurUniforms>() as u64,
                                ),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(noise_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&resources.surface_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(previous),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(image_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                });
            let pipeline = &resources.pipelines.wallpaper;
            let mut draw = super::begin_color_render_pass(
                encoder,
                "wallpaper",
                &views[index],
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                None,
            );
            draw.set_pipeline(pipeline);
            draw.set_bind_group(0, &resources.globals_bind_group, &[]);
            draw.set_bind_group(1, &bind_group, &[uniform_offset]);
            draw.draw(0..3, 0..1);
        }
        let last = paint.passes.last().expect("passes");
        let size = [last.width as f32, last.height as f32];
        let uniforms = BlurUniforms::composite(
            filter.bounds,
            filter.content_mask.bounds,
            filter.corner_radii,
            0.0,
            filter.opacity,
            FilterCompositeClip::RoundedBounds,
            size,
            {
                let size = self.target.viewport_size();
                [size.width.0 as f32, size.height.0 as f32]
            },
        )
        .with_wallpaper_texture(size);
        let pipeline = &self.resources().pipelines.blur_composite;
        self.draw_filter_pass(
            encoder,
            "wallpaper_composite",
            pipeline,
            target,
            views.last().expect("views"),
            uniforms,
            wgpu::LoadOp::Load,
            scissor,
        );
    }
}

pub(super) type WallpaperCacheCell = RefCell<WallpaperCache>;

fn image_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &WallpaperImage,
) -> wgpu::TextureView {
    let levels = 32 - image.width.max(image.height).leading_zeros();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("wallpaper_image"),
        size: wgpu::Extent3d {
            width: image.width,
            height: image.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let (mut width, mut height) = (image.width, image.height);
    let mut rgba = image.premultiplied_bottom_up_rgba();
    for level in 0..levels {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        if level + 1 < levels {
            rgba = image_mip(&rgba, width, height);
            width = (width / 2).max(1);
            height = (height / 2).max(1);
        }
    }
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn image_mip(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (w, h) = ((width / 2).max(1), (height / 2).max(1));
    let mut next = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            for channel in 0..4 {
                let mut sum = 0u32;
                let mut count = 0;
                for sy in y * height / h..(y + 1) * height / h {
                    for sx in x * width / w..(x + 1) * width / w {
                        sum += rgba[((sy * width + sx) * 4 + channel) as usize] as u32;
                        count += 1;
                    }
                }
                next.push(((sum + count / 2) / count) as u8);
            }
        }
    }
    next
}

impl WallpaperCache {
    pub(super) fn slot_changed(&self, paint: &WallpaperPaint, now_ms: u32) -> bool {
        paint.clock.is_some_and(|clock| {
            self.entries.get(&paint.key).and_then(|entry| entry.slot) != Some(clock.slot(now_ms))
        })
    }

    fn release_image(&mut self, id: u64) {
        let image = self.images.get_mut(&id).expect("cached image");
        image.0 -= 1;
        if image.0 == 0 {
            self.images.remove(&id);
        }
    }
}

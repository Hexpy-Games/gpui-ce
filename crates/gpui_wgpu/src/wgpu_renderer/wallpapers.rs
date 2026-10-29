//! Procedural wallpapers ([`gpui::WallpaperPaint`]): each pass runs the `wallpaper` pipeline into
//! a cached texture of its own size (the passes re-run only when the paint's generation changes),
//! and the last texture is composited into the rounded bounds with the blur-composite pipeline.
use std::cell::RefCell;
use std::collections::HashMap;

use gpui::{BackdropFilter, WallpaperPaint};
use gpui_render::{blur::FilterCompositeClip, shaders::blur::BlurUniforms};

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
    passes: Vec<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
    last_used: u64,
}

#[derive(Default)]
pub(super) struct WallpaperCache {
    noise: Option<[wgpu::TextureView; 2]>,
    entries: HashMap<u64, Entry>,
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
    ) {
        if paint.passes.is_empty() {
            return;
        }
        let format = self.target.format();
        // Collect the views to render into / sample (cloned out, so the cache borrow ends here).
        let (noise, views, stale) = {
            let resources = self.resources();
            let mut cache = resources.wallpapers.borrow_mut();
            if cache.noise.is_none() {
                cache.noise = Some([
                    noise_texture(&resources.device, &resources.queue, false),
                    noise_texture(&resources.device, &resources.queue, true),
                ]);
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
                    cache.entries.remove(&oldest);
                }
            }
            let entry = cache.entries.entry(paint.key).or_insert_with(|| Entry {
                generation: u64::MAX,
                passes: Vec::new(),
                last_used: 0,
            });
            entry.last_used = tick;
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
            let stale = entry.generation != paint.generation;
            entry.generation = paint.generation;
            let views: Vec<wgpu::TextureView> = entry.passes.iter().map(|e| e.3.clone()).collect();
            (cache.noise.clone().expect("noise"), views, stale)
        };
        if stale {
            for (index, pass) in paint.passes.iter().enumerate() {
                let source = match pass.source {
                    0 => &noise[0],
                    1 => &noise[1],
                    _ => &views[index.saturating_sub(1)],
                };
                let pipeline = &self.resources().pipelines.wallpaper;
                self.draw_filter_pass(
                    encoder,
                    "wallpaper",
                    pipeline,
                    &views[index],
                    source,
                    BlurUniforms::wallpaper_pass(pass),
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    None,
                );
            }
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
            [self.target.width() as f32, self.target.height() as f32],
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
            None,
        );
    }
}

pub(super) type WallpaperCacheCell = RefCell<WallpaperCache>;

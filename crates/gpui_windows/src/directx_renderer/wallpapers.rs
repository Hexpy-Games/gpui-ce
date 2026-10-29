// Added by HexpyGames for Butler (2026), under the Apache License 2.0; see NOTICE.
//! DirectX side of procedural wallpapers ([`gpui::WallpaperPaint`]), mirroring the wgpu path:
//! each pass runs the `wallpaper` shader into a cached color target of its own size (only when
//! the paint's generation changes), and the last target is composited into the rounded bounds.
use std::cell::RefCell;
use std::collections::HashMap;

use super::*;

struct DxEntry {
    generation: u64,
    passes: Vec<(u32, u32, ColorTarget)>,
    last_used: u64,
}

#[derive(Default)]
pub(super) struct DxWallpaperCache {
    noise: Option<[Option<ID3D11ShaderResourceView>; 2]>,
    entries: HashMap<u64, DxEntry>,
    tick: u64,
}

pub(super) type DxWallpaperCacheCell = RefCell<DxWallpaperCache>;

impl DxWallpaperCache {
    /// Drops every GPU object (device lost).
    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }
}

const MAX_ENTRIES: usize = 8;

fn noise_view(device: &ID3D11Device, print: bool) -> Result<Option<ID3D11ShaderResourceView>> {
    let data = gpui_wgpu_free_noise(print);
    let desc = D3D11_TEXTURE2D_DESC {
        Width: 256,
        Height: 256,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_IMMUTABLE,
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let init = D3D11_SUBRESOURCE_DATA {
        pSysMem: data.as_ptr() as *const std::ffi::c_void,
        SysMemPitch: 256 * 4,
        SysMemSlicePitch: 0,
    };
    let mut texture = None;
    unsafe { device.CreateTexture2D(&desc, Some(&init), Some(&mut texture))? };
    let texture = texture.context("noise texture")?;
    let mut view = None;
    unsafe { device.CreateShaderResourceView(&texture, None, Some(&mut view))? };
    Ok(view)
}

/// The engine noise (lowbias32 hash) and the riso print noise (xorshift), as in the product.
fn gpui_wgpu_free_noise(print: bool) -> Vec<u8> {
    let mut data = vec![0u8; 256 * 256 * 4];
    let mut s: u32 = 0x2f6b1d;
    for (index, byte) in data.iter_mut().enumerate() {
        if print {
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

impl DirectXRenderer {
    pub(super) fn dx_draw_wallpaper(
        &self,
        target_rtv: &Option<ID3D11RenderTargetView>,
        filter: &BackdropFilter,
        paint: &WallpaperPaint,
    ) -> Result<()> {
        if paint.passes.is_empty() {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        let (noise, targets, stale) = {
            let mut cache = self.wallpapers.borrow_mut();
            if cache.noise.is_none() {
                cache.noise = Some([
                    noise_view(&devices.device, false)?,
                    noise_view(&devices.device, true)?,
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
            let entry = cache.entries.entry(paint.key).or_insert_with(|| DxEntry {
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
                let mut passes = Vec::with_capacity(paint.passes.len());
                for p in &paint.passes {
                    passes.push((
                        p.width,
                        p.height,
                        create_color_target(&devices.device, p.width, p.height)?,
                    ));
                }
                entry.passes = passes;
                entry.generation = u64::MAX;
            }
            let stale = entry.generation != paint.generation;
            entry.generation = paint.generation;
            let targets: Vec<(
                Option<ID3D11RenderTargetView>,
                Option<ID3D11ShaderResourceView>,
            )> = entry
                .passes
                .iter()
                .map(|e| (e.2.target.clone(), e.2.source.clone()))
                .collect();
            (cache.noise.clone().expect("noise"), targets, stale)
        };
        if stale {
            for (index, pass) in paint.passes.iter().enumerate() {
                let source = match pass.source {
                    0 => &noise[0],
                    1 => &noise[1],
                    _ => &targets[index.saturating_sub(1)].1,
                };
                let viewport = D3D11_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: pass.width as f32,
                    Height: pass.height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                };
                self.dx_blur_pass(
                    &self.pipelines.wallpaper_vertex,
                    &self.pipelines.wallpaper_fragment,
                    &self.pipelines.blur_blend_replace,
                    &targets[index].0,
                    source,
                    BlurUniforms::wallpaper_pass(pass),
                    &viewport,
                    D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                    3,
                    true,
                )?;
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
            [self.width as f32, self.height as f32],
        )
        .with_wallpaper_texture(size);
        self.dx_blur_pass(
            &self.pipelines.blur_composite_vertex,
            &self.pipelines.blur_composite_fragment,
            &self.pipelines.blur_blend_composite,
            target_rtv,
            &targets.last().expect("targets").1,
            uniforms,
            &resources.viewport,
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            4,
            false,
        )
    }
}

// Added by HexpyGames for Butler (2026), under the Apache License 2.0; see NOTICE.
//! Cache of blurred backdrops. An entry holds the half-resolution blur result (and, for a lens,
//! the sharp full-resolution snapshot) of one backdrop filter, keyed by
//! `gpui_render::backdrop_key::backdrop_keys`. A hit copies the textures back so the blur passes
//! and the snapshot are skipped; the composite still runs every frame.
use std::cell::RefCell;
use std::collections::HashMap;

use gpui_render::blur::ScissorRectangle;

/// The cache shared through the renderer resources.
pub(super) type BackdropCacheCell = RefCell<BackdropCache>;

const MAX_ENTRIES: usize = 8;

struct CachedCopy {
    rect: ScissorRectangle,
    texture: wgpu::Texture,
}

struct Entry {
    blurred: Option<CachedCopy>,
    sharp: Option<CachedCopy>,
    last_used: u64,
}

#[derive(Default)]
pub(super) struct BackdropCache {
    entries: HashMap<u64, Entry>,
    tick: u64,
}

fn extent(rect: ScissorRectangle) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: rect.width,
        height: rect.height,
        depth_or_array_layers: 1,
    }
}

fn origin(rect: ScissorRectangle) -> wgpu::Origin3d {
    wgpu::Origin3d {
        x: rect.x,
        y: rect.y,
        z: 0,
    }
}

fn copy_rect(
    encoder: &mut wgpu::CommandEncoder,
    source: &wgpu::Texture,
    source_origin: wgpu::Origin3d,
    destination: &wgpu::Texture,
    destination_origin: wgpu::Origin3d,
    size: wgpu::Extent3d,
) {
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: source,
            mip_level: 0,
            origin: source_origin,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: destination,
            mip_level: 0,
            origin: destination_origin,
            aspect: wgpu::TextureAspect::All,
        },
        size,
    );
}

fn store_copy(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    slot: &mut Option<CachedCopy>,
    source: &wgpu::Texture,
    rect: ScissorRectangle,
) {
    if rect.is_empty() {
        *slot = None;
        return;
    }
    let reusable = slot.as_ref().is_some_and(|copy| {
        copy.rect.width == rect.width
            && copy.rect.height == rect.height
            && copy.texture.format() == source.format()
    });
    if !reusable {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("backdrop_cache"),
            size: extent(rect),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: source.format(),
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        *slot = Some(CachedCopy { rect, texture });
    }
    let Some(copy) = slot.as_mut() else {
        return;
    };
    copy.rect = rect;
    copy_rect(
        encoder,
        source,
        origin(rect),
        &copy.texture,
        wgpu::Origin3d::ZERO,
        extent(rect),
    );
}

impl BackdropCache {
    /// Whether an entry exists for `key`.
    pub(super) fn contains(&self, key: u64) -> bool {
        self.entries.contains_key(&key)
    }

    /// Copies the entry for `key` back into `ping` (half-res) and `snapshot` (full-res) at their
    /// original rects; false when there is no entry or a needed target is missing.
    pub(super) fn restore(
        &mut self,
        key: u64,
        encoder: &mut wgpu::CommandEncoder,
        ping: Option<&wgpu::Texture>,
        snapshot: Option<&wgpu::Texture>,
    ) -> bool {
        let Some(entry) = self.entries.get_mut(&key) else {
            return false;
        };
        if (entry.blurred.is_some() && ping.is_none())
            || (entry.sharp.is_some() && snapshot.is_none())
        {
            return false;
        }
        self.tick += 1;
        entry.last_used = self.tick;
        for (copy, target) in [(&entry.blurred, ping), (&entry.sharp, snapshot)] {
            if let (Some(copy), Some(target)) = (copy, target) {
                copy_rect(
                    encoder,
                    &copy.texture,
                    wgpu::Origin3d::ZERO,
                    target,
                    origin(copy.rect),
                    extent(copy.rect),
                );
            }
        }
        true
    }

    /// Stores copies of `ping[blurred]` and `snapshot[sharp]` under `key`, evicting the least recently used entry past `MAX_ENTRIES`.
    pub(super) fn store(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        key: u64,
        blurred: Option<(&wgpu::Texture, ScissorRectangle)>,
        sharp: Option<(&wgpu::Texture, ScissorRectangle)>,
    ) {
        self.tick += 1;
        let tick = self.tick;
        let entry = self.entries.entry(key).or_insert_with(|| Entry {
            blurred: None,
            sharp: None,
            last_used: tick,
        });
        entry.last_used = tick;
        match blurred {
            Some((texture, rect)) => store_copy(device, encoder, &mut entry.blurred, texture, rect),
            None => entry.blurred = None,
        }
        match sharp {
            Some((texture, rect)) => store_copy(device, encoder, &mut entry.sharp, texture, rect),
            None => entry.sharp = None,
        }
        if self.entries.len() > MAX_ENTRIES {
            let oldest = self
                .entries
                .iter()
                .filter(|(entry_key, _)| **entry_key != key)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(entry_key, _)| *entry_key);
            if let Some(oldest) = oldest {
                self.entries.remove(&oldest);
            }
        }
    }

    /// Drops every entry.
    pub(super) fn clear(&mut self) {
        self.entries.clear();
    }
}

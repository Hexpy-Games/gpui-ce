// Added by HexpyGames for Butler (2026), under the Apache License 2.0; see NOTICE.
//! DirectX side of the backdrop cache: keeps the blurred half-resolution backdrop (and the sharp
//! snapshot for a lens) of a backdrop filter whose sampling region did not change, so the blur
//! passes can be skipped. Entries are keyed by `gpui_render::backdrop_key::backdrop_keys`.
use std::collections::HashMap;

use super::*;

/// A rect-sized copy of part of a render target.
struct StoredRect {
    rect: ScissorRectangle,
    texture: ID3D11Texture2D,
}

struct Entry {
    blurred: Option<StoredRect>,
    sharp: Option<StoredRect>,
    last_used: u64,
}

/// LRU cache of backdrop copies, owned by the blur resources so that it is dropped with them.
#[derive(Default)]
pub(super) struct DxBackdropCache {
    entries: HashMap<u64, Entry>,
    tick: u64,
}

const MAX_ENTRIES: usize = 8;

impl DxBackdropCache {
    /// Copies the entry for `key` back into `ping` (half-res) and `snapshot` (full-res) at their
    /// original rects; false when there is no entry or a needed target is missing.
    pub(super) fn restore(
        &mut self,
        key: u64,
        ctx: &ID3D11DeviceContext,
        ping: Option<&ID3D11Texture2D>,
        snapshot: Option<&ID3D11Texture2D>,
    ) -> bool {
        self.tick += 1;
        let tick = self.tick;
        let Some(entry) = self.entries.get_mut(&key) else {
            return false;
        };
        if (entry.blurred.is_some() && ping.is_none())
            || (entry.sharp.is_some() && snapshot.is_none())
        {
            return false;
        }
        entry.last_used = tick;
        if let (Some(stored), Some(target)) = (&entry.blurred, ping) {
            copy_region(ctx, target, stored.rect.x, stored.rect.y, &stored.texture, &whole(stored.rect));
        }
        if let (Some(stored), Some(target)) = (&entry.sharp, snapshot) {
            copy_region(ctx, target, stored.rect.x, stored.rect.y, &stored.texture, &whole(stored.rect));
        }
        true
    }

    /// Stores copies of `ping[blurred]` and `snapshot[sharp]` under `key`, evicting the least recently used entry past `MAX_ENTRIES`.
    pub(super) fn store(
        &mut self,
        device: &ID3D11Device,
        ctx: &ID3D11DeviceContext,
        key: u64,
        blurred: Option<(&ID3D11Texture2D, ScissorRectangle)>,
        sharp: Option<(&ID3D11Texture2D, ScissorRectangle)>,
    ) {
        self.tick += 1;
        let tick = self.tick;
        let entry = self.entries.entry(key).or_insert(Entry {
            blurred: None,
            sharp: None,
            last_used: tick,
        });
        entry.last_used = tick;
        store_rect(device, ctx, &mut entry.blurred, blurred);
        store_rect(device, ctx, &mut entry.sharp, sharp);

        while self.entries.len() > MAX_ENTRIES {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| *key)
            else {
                break;
            };
            self.entries.remove(&oldest);
        }
    }

    /// Whether an entry for `key` exists.
    pub(super) fn contains(&self, key: u64) -> bool {
        self.entries.contains_key(&key)
    }
}

/// `rect` moved to the origin: the area of an entry texture that holds the copy.
fn whole(rect: ScissorRectangle) -> ScissorRectangle {
    ScissorRectangle {
        x: 0,
        y: 0,
        ..rect
    }
}

/// Replaces `slot` with a copy of `source[rect]`, reusing its texture when the size is unchanged.
fn store_rect(
    device: &ID3D11Device,
    ctx: &ID3D11DeviceContext,
    slot: &mut Option<StoredRect>,
    source: Option<(&ID3D11Texture2D, ScissorRectangle)>,
) {
    let Some((texture, rect)) = source.filter(|(_, rect)| !rect.is_empty()) else {
        *slot = None;
        return;
    };
    let reusable = slot
        .take()
        .filter(|stored| stored.rect.width == rect.width && stored.rect.height == rect.height);
    let target = match reusable {
        Some(stored) => stored.texture,
        None => match create_copy_texture(device, rect.width, rect.height).log_err() {
            Some(texture) => texture,
            None => return,
        },
    };
    copy_region(ctx, &target, 0, 0, texture, &rect);
    *slot = Some(StoredRect {
        rect,
        texture: target,
    });
}

/// Copies `src_rect` of `src` to `dst` at (`x`, `y`).
fn copy_region(
    ctx: &ID3D11DeviceContext,
    dst: &ID3D11Texture2D,
    x: u32,
    y: u32,
    src: &ID3D11Texture2D,
    src_rect: &ScissorRectangle,
) {
    let source_box = D3D11_BOX {
        left: src_rect.x,
        top: src_rect.y,
        front: 0,
        right: src_rect.x + src_rect.width,
        bottom: src_rect.y + src_rect.height,
        back: 1,
    };
    unsafe { ctx.CopySubresourceRegion(dst, 0, x, y, 0, src, 0, Some(&source_box)) };
}

fn create_copy_texture(device: &ID3D11Device, width: u32, height: u32) -> Result<ID3D11Texture2D> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: 0,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let mut texture = None;
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture))? };
    texture.context("backdrop cache texture")
}

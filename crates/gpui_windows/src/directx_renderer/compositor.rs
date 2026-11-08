// Added by HexpyGames for Butler (2026), under the Apache License 2.0; see NOTICE.
//! Looping sprites drawn by DirectComposition instead of the GPU frame. The animations follow the
//! shader clock formulas (`frame_tile_bounds`, `turn_about` in gpui_render), so a sprite looks the
//! same on either path.
use super::*;

/// One sprite's visuals: `clip` is the child of the root visual.
struct Entry {
    sprite: CompositorSprite,
    clip: IDCompositionVisual,
}

/// The child visuals of the window's root visual that draw the scene's [`CompositorSprite`]s.
#[derive(Default)]
pub(super) struct CompositorVisuals {
    entries: Vec<Entry>,
    /// The surface and size of each flipbook grid in use, by (texture, tile id, tint). A tinted
    /// (monochrome) grid is baked with its tint, so the key includes it.
    surfaces: Vec<(
        (AtlasTextureId, u32, Option<Rgba>),
        IDCompositionSurface,
        (u32, u32),
    )>,
}

impl CompositorVisuals {
    /// Makes the visuals draw `sprites`. Returns whether anything changed, in which case the
    /// caller commits.
    pub(super) fn update(
        &mut self,
        composition: &DirectComposition,
        device_context: &ID3D11DeviceContext,
        sprites: &[CompositorSprite],
        atlas: &DirectXAtlas,
    ) -> Result<bool> {
        let matched = self.matches(sprites);
        if matched.len() == self.entries.len() && matched.iter().all(Option::is_some) {
            return Ok(false);
        }

        let now_ms = gpui::animation_time_ms();
        let mut old: Vec<Option<Entry>> = std::mem::take(&mut self.entries)
            .into_iter()
            .map(Some)
            .collect();
        let mut entries = Vec::with_capacity(sprites.len());
        for (sprite, found) in sprites.iter().zip(matched) {
            if let Some(entry) = found.and_then(|index| old[index].take()) {
                entries.push(entry);
            } else if let Some((surface, grid)) =
                self.surface_for(composition, device_context, sprite, atlas)?
            {
                let clip = make_visuals(&composition.comp_device, sprite, &surface, grid, now_ms)?;
                unsafe { composition.comp_visual.AddVisual(&clip, true, None)? };
                entries.push(Entry {
                    sprite: *sprite,
                    clip,
                });
            }
        }
        for entry in old.into_iter().flatten() {
            unsafe { composition.comp_visual.RemoveVisual(&entry.clip)? };
        }

        self.entries = entries;
        let entries = &self.entries;
        self.surfaces.retain(|(key, _, _)| {
            entries.iter().any(|entry| {
                *key == (
                    entry.sprite.tile.texture_id,
                    entry.sprite.tile.tile_id.0,
                    entry.sprite.color.map(hsla_to_rgba),
                )
            })
        });
        Ok(true)
    }

    /// Drops everything without calling DirectComposition: the visuals belong to the device that
    /// was lost.
    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.surfaces.clear();
    }

    /// For each of `sprites`, the existing entry that already draws it.
    fn matches(&self, sprites: &[CompositorSprite]) -> Vec<Option<usize>> {
        let mut taken = vec![false; self.entries.len()];
        sprites
            .iter()
            .map(|sprite| {
                let index = self
                    .entries
                    .iter()
                    .enumerate()
                    .position(|(index, entry)| !taken[index] && entry.sprite == *sprite)?;
                taken[index] = true;
                Some(index)
            })
            .collect()
    }

    fn surface_for(
        &mut self,
        composition: &DirectComposition,
        device_context: &ID3D11DeviceContext,
        sprite: &CompositorSprite,
        atlas: &DirectXAtlas,
    ) -> Result<Option<(IDCompositionSurface, (u32, u32))>> {
        let tint = sprite.color.map(hsla_to_rgba);
        let key = (sprite.tile.texture_id, sprite.tile.tile_id.0, tint);
        if let Some((_, surface, grid)) =
            self.surfaces.iter().find(|(existing, ..)| *existing == key)
        {
            return Ok(Some((surface.clone(), *grid)));
        }

        let Some((size, bytes)) = atlas.frame_grid(&sprite.tile) else {
            return Ok(None);
        };
        let (width, height) = (size.width.0 as u32, size.height.0 as u32);
        let pixel_count = width as usize * height as usize;
        let mut pixels = Vec::with_capacity(pixel_count * 4);
        if let Some(tint) = tint {
            if bytes.len() != pixel_count {
                return Ok(None);
            }
            for &mask in bytes.iter() {
                let alpha = f32::from(mask) / 255.0 * tint.alpha;
                pixels.extend_from_slice(&[
                    (tint.blue * alpha * 255.0).round() as u8,
                    (tint.green * alpha * 255.0).round() as u8,
                    (tint.red * alpha * 255.0).round() as u8,
                    (alpha * 255.0).round() as u8,
                ]);
            }
        } else {
            if bytes.len() != pixel_count * 4 {
                return Ok(None);
            }
            for pixel in bytes.chunks_exact(4) {
                let alpha = u32::from(pixel[3]);
                pixels.extend(
                    pixel[..3]
                        .iter()
                        .map(|&channel| ((u32::from(channel) * alpha + 127) / 255) as u8),
                );
                pixels.push(pixel[3]);
            }
        }

        let surface = unsafe {
            composition.comp_device.CreateSurface(
                width,
                height,
                DXGI_FORMAT_B8G8R8A8_UNORM,
                DXGI_ALPHA_MODE_PREMULTIPLIED,
            )?
        };
        let mut offset = POINT::default();
        let texture: ID3D11Texture2D = unsafe { surface.BeginDraw(None, &mut offset)? };
        unsafe {
            device_context.UpdateSubresource(
                &texture,
                0,
                Some(&D3D11_BOX {
                    left: offset.x as u32,
                    top: offset.y as u32,
                    front: 0,
                    right: offset.x as u32 + width,
                    bottom: offset.y as u32 + height,
                    back: 1,
                }),
                pixels.as_ptr() as _,
                width * 4,
                0,
            );
            surface.EndDraw()?;
        }
        self.surfaces.push((key, surface.clone(), (width, height)));
        Ok(Some((surface, (width, height))))
    }
}

/// The `clip` visual of `sprite` and its children, drawing `surface`, a grid of `grid` size.
fn make_visuals(
    device: &IDCompositionDevice,
    sprite: &CompositorSprite,
    surface: &IDCompositionSurface,
    grid: (u32, u32),
    now_ms: u32,
) -> Result<IDCompositionVisual> {
    let frames = &sprite.frames;
    let (bounds, clip_bounds) = (sprite.bounds, sprite.clip);
    let columns = frames.columns.max(1);
    let count = frames.frame_count.max(1);
    let rows = count.div_ceil(columns);
    let (cell_width, cell_height) = (grid.0 / columns, grid.1 / rows);
    let gutter = FRAME_GUTTER;

    unsafe {
        let clip = device.CreateVisual()?;
        clip.SetOffsetX2(clip_bounds.origin.x.0)?;
        clip.SetOffsetY2(clip_bounds.origin.y.0)?;
        let clip_rect = device.CreateRectangleClip()?;
        clip_rect.SetLeft2(0.0)?;
        clip_rect.SetTop2(0.0)?;
        clip_rect.SetRight2(clip_bounds.size.width.0)?;
        clip_rect.SetBottom2(clip_bounds.size.height.0)?;
        clip.SetClip(&clip_rect)?;
        clip.SetBitmapInterpolationMode(DCOMPOSITION_BITMAP_INTERPOLATION_MODE_LINEAR)?;

        let window = device.CreateVisual()?;
        window.SetOffsetX2(bounds.origin.x.0 - clip_bounds.origin.x.0)?;
        window.SetOffsetY2(bounds.origin.y.0 - clip_bounds.origin.y.0)?;
        let window_rect = device.CreateRectangleClip()?;
        window_rect.SetLeft2(0.0)?;
        window_rect.SetTop2(0.0)?;
        window_rect.SetRight2(bounds.size.width.0)?;
        window_rect.SetBottom2(bounds.size.height.0)?;
        window.SetClip(&window_rect)?;
        if frames.turn_period_ms != 0 {
            let rotate = device.CreateRotateTransform()?;
            rotate.SetCenterX2(bounds.size.width.0 / 2.0)?;
            rotate.SetCenterY2(bounds.size.height.0 / 2.0)?;
            let turn = device.CreateAnimation()?;
            let period = u64::from(frames.turn_period_ms);
            let phase = u64::from(now_ms.wrapping_sub(frames.start_ms)) % period;
            let slope = 360.0 / (period as f32 / 1000.0);
            turn.AddCubic(0.0, 360.0 * phase as f32 / period as f32, slope, 0.0, 0.0)?;
            turn.AddCubic((period - phase) as f64 / 1000.0, 0.0, slope, 0.0, 0.0)?;
            turn.AddRepeat((2 * period - phase) as f64 / 1000.0, period as f64 / 1000.0)?;
            rotate.SetAngle(&turn)?;
            window.SetTransform(&rotate)?;
        }
        clip.AddVisual(&window, true, None)?;

        let content = device.CreateVisual()?;
        content.SetContent(surface)?;
        if frames.frame_count > 1 {
            let offsets = |origin: fn(u32, u32) -> u32, cell: u32| -> Vec<f32> {
                (0..count)
                    .map(|frame| -((origin(frame, columns) * cell + gutter) as f32))
                    .collect()
            };
            let period = u64::from(frames.period_ms);
            let phase = u64::from(now_ms.wrapping_sub(frames.start_ms)) % period;
            let x = stepped(device, &offsets(|f, c| f % c, cell_width), period, phase)?;
            let y = stepped(device, &offsets(|f, c| f / c, cell_height), period, phase)?;
            content.SetOffsetX(&x)?;
            content.SetOffsetY(&y)?;
        } else {
            content.SetOffsetX2(-(gutter as f32))?;
            content.SetOffsetY2(-(gutter as f32))?;
        }
        window.AddVisual(&content, true, None)?;
        Ok(clip)
    }
}

/// A looping animation that holds `values[f]` during frame `f` of a `period_ms` loop, `phase_ms`
/// into it when DWM applies the commit.
fn stepped(
    device: &IDCompositionDevice,
    values: &[f32],
    period_ms: u64,
    phase_ms: u64,
) -> Result<IDCompositionAnimation> {
    let count = values.len() as u64;
    let start = |frame: u64| (frame * period_ms).div_ceil(count);
    let first = (phase_ms * count / period_ms) as usize;
    unsafe {
        let animation = device.CreateAnimation()?;
        animation.AddCubic(0.0, values[first], 0.0, 0.0, 0.0)?;
        for (frame, &value) in values.iter().enumerate().skip(first + 1) {
            let begin = (start(frame as u64) - phase_ms) as f64 / 1000.0;
            animation.AddCubic(begin, value, 0.0, 0.0, 0.0)?;
        }
        for (frame, &value) in values.iter().enumerate() {
            let begin = (period_ms - phase_ms + start(frame as u64)) as f64 / 1000.0;
            animation.AddCubic(begin, value, 0.0, 0.0, 0.0)?;
        }
        animation.AddRepeat(
            (2 * period_ms - phase_ms) as f64 / 1000.0,
            period_ms as f64 / 1000.0,
        )?;
        Ok(animation)
    }
}

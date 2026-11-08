// Added by HexpyGames for Butler (2026), under the Apache License 2.0; see NOTICE.
//! Looping sprites drawn by DirectComposition instead of the GPU frame. The animations follow the
//! shader clock formulas (`frame_tile_bounds`, `turn_about` in gpui_render), so a sprite looks the
//! same on either path.
use super::*;

/// One sprite's visuals: `clip` is the child of the root visual.
struct Entry {
    sprite: CompositorSprite,
    clip: IDCompositionVisual,
    key: Option<u64>,
}

/// The child visuals of the window's root visual that draw the scene's [`CompositorSprite`]s.
#[derive(Default)]
pub(super) struct CompositorVisuals {
    entries: Vec<Entry>,
    /// The surface and size of each flipbook grid in use, by (texture, tile id, tint). A tinted
    /// (monochrome) grid is baked with its tint, so the key includes it.
    surfaces: Vec<(
        (Option<(AtlasTextureId, u32)>, Option<u64>, Option<Rgba>),
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
        keys: &[Option<u64>],
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
        for ((sprite, found), &key) in sprites.iter().zip(matched).zip(keys) {
            if let Some(entry) = found.and_then(|index| old[index].take()) {
                entries.push(entry);
            } else if let Some((surface, grid)) =
                self.surface_for(composition, device_context, sprite, key, atlas)?
            {
                let clip = make_visuals(&composition.comp_device, sprite, &surface, grid, now_ms)?;
                unsafe { composition.comp_visual.AddVisual(&clip, true, None)? };
                entries.push(Entry {
                    sprite: sprite.clone(),
                    clip,
                    key,
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
                    match &entry.sprite.contents {
                        SpriteContents::Atlas(tile) => Some((tile.texture_id, tile.tile_id.0)),
                        SpriteContents::Pixels { .. } => None,
                    },
                    entry.key,
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
        group_key: Option<u64>,
        atlas: &DirectXAtlas,
    ) -> Result<Option<(IDCompositionSurface, (u32, u32))>> {
        let tint = sprite.color.map(hsla_to_rgba);
        let key = (
            match &sprite.contents {
                SpriteContents::Atlas(tile) => Some((tile.texture_id, tile.tile_id.0)),
                SpriteContents::Pixels { .. } => None,
            },
            group_key,
            tint,
        );
        if let Some((_, surface, grid)) =
            self.surfaces.iter().find(|(existing, ..)| *existing == key)
        {
            return Ok(Some((surface.clone(), *grid)));
        }

        let (size, bytes) = match &sprite.contents {
            SpriteContents::Atlas(tile) => {
                let Some(pixels) = atlas.frame_grid(tile) else {
                    return Ok(None);
                };
                pixels
            }
            SpriteContents::Pixels { size, bytes } => (*size, bytes.clone()),
        };
        let (width, height) = (size.width.0 as u32, size.height.0 as u32);
        let pixel_count = width as usize * height as usize;
        let mut pixels = Vec::with_capacity(pixel_count * 4);
        if matches!(&sprite.contents, SpriteContents::Pixels { .. }) {
            pixels.extend_from_slice(&bytes);
        } else if let Some(tint) = tint {
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
    let gutter = if matches!(&sprite.contents, SpriteContents::Atlas(_)) {
        FRAME_GUTTER
    } else {
        0
    };

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
        let motion = &sprite.motion;
        if motion.is_animated() {
            let opacity = motion_animation(device, motion, motion.opacity, now_ms)?;
            let effect = device.CreateEffectGroup()?;
            effect.SetOpacity(&opacity)?;
            window.SetEffect(&effect)?;
            let x = motion_animation(
                device,
                motion,
                (
                    bounds.origin.x.0 - clip_bounds.origin.x.0 + motion.offset.0.x.0,
                    bounds.origin.x.0 - clip_bounds.origin.x.0 + motion.offset.1.x.0,
                ),
                now_ms,
            )?;
            let y = motion_animation(
                device,
                motion,
                (
                    bounds.origin.y.0 - clip_bounds.origin.y.0 + motion.offset.0.y.0,
                    bounds.origin.y.0 - clip_bounds.origin.y.0 + motion.offset.1.y.0,
                ),
                now_ms,
            )?;
            window.SetOffsetX(&x)?;
            window.SetOffsetY(&y)?;
            let scale = device.CreateScaleTransform()?;
            scale.SetCenterX2(bounds.size.width.0 * motion.origin.x)?;
            scale.SetCenterY2(bounds.size.height.0 * motion.origin.y)?;
            let animation = motion_animation(device, motion, motion.scale, now_ms)?;
            scale.SetScaleX(&animation)?;
            scale.SetScaleY(&animation)?;
            window.SetTransform(&scale)?;
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

/// Fits a looping motion curve to DirectComposition cubic segments and aligns its phase.
fn motion_animation(
    device: &IDCompositionDevice,
    motion: &LayerMotion,
    (from, to): (f32, f32),
    now_ms: u32,
) -> Result<IDCompositionAnimation> {
    let duration = f64::from(motion.duration_ms) / 1000.0;
    let delta = to - from;
    let mut segments = Vec::new();
    match motion.easing {
        MotionEasing::Steps(n) => {
            for i in 0..n {
                segments.push((
                    f64::from(i) * duration / f64::from(n),
                    duration / f64::from(n),
                    [from + delta * i as f32 / n as f32, 0.0, 0.0, 0.0],
                ));
            }
        }
        MotionEasing::CubicBezier([x1, y1, x2, y2]) => {
            let [x1, y1, x2, y2] = [x1, y1, x2, y2].map(f64::from);
            // Eased value at linear progress `x`; slopes come from finite differences so a curve
            // with a vertical tangent (x1 or x2 at 0 or 1) still gives finite Hermite segments.
            let ease = |x: f64| {
                let (mut low, mut high) = (0.0, 1.0);
                for _ in 0..32 {
                    let t = (low + high) * 0.5;
                    let u = 1.0 - t;
                    if 3.0 * u * u * t * x1 + 3.0 * u * t * t * x2 + t * t * t < x {
                        low = t;
                    } else {
                        high = t;
                    }
                }
                let t = (low + high) * 0.5;
                let u = 1.0 - t;
                3.0 * u * u * t * y1 + 3.0 * u * t * t * y2 + t * t * t
            };
            let sample = |x: f64| {
                const STEP: f64 = 1.0 / 1024.0;
                let (lo, hi) = ((x - STEP).max(0.0), (x + STEP).min(1.0));
                let slope = (ease(hi) - ease(lo)) / (hi - lo);
                (
                    from + delta * ease(x) as f32,
                    (f64::from(delta) * slope / duration) as f32,
                )
            };
            let h = duration as f32 / 8.0;
            for i in 0..8 {
                let (a, b) = sample(f64::from(i) / 8.0);
                let (end, slope) = sample(f64::from(i + 1) / 8.0);
                let c = (3.0 * (end - a) / h - 2.0 * b - slope) / h;
                let d = (2.0 * (a - end) / h + b + slope) / (h * h);
                segments.push((f64::from(i) * duration / 8.0, duration / 8.0, [a, b, c, d]));
            }
        }
    }
    let mut period = duration;
    if motion.repeat == MotionRepeat::Alternate {
        let reversed: Vec<_> = segments
            .iter()
            .rev()
            .map(|&(begin, length, [a, b, c, d])| {
                let h = length as f32;
                (
                    2.0 * duration - begin - length,
                    length,
                    [
                        a + h * (b + h * (c + h * d)),
                        -b - h * (2.0 * c + 3.0 * h * d),
                        c + 3.0 * h * d,
                        -d,
                    ],
                )
            })
            .collect();
        segments.extend(reversed);
        period *= 2.0;
    }
    let phase = (f64::from(now_ms.wrapping_sub(motion.start_ms)) / 1000.0) % period;
    unsafe {
        let animation = device.CreateAnimation()?;
        for cycle in 0..2 {
            for &(begin, length, [a, b, c, d]) in &segments {
                let time = f64::from(cycle) * period + begin - phase;
                if time + length <= 0.0 {
                    continue;
                }
                let t = (-time).max(0.0) as f32;
                animation.AddCubic(
                    time.max(0.0),
                    a + t * (b + t * (c + t * d)),
                    b + t * (2.0 * c + 3.0 * t * d),
                    c + 3.0 * t * d,
                    d,
                )?;
            }
        }
        animation.AddRepeat(2.0 * period - phase, period)?;
        Ok(animation)
    }
}

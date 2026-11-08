//! What an animation tick redraws when only animated sprites changed.

use super::{
    BackdropFilter, Bounds, DrawOrder, FilterRenderTarget, PaintedBounds, PrimitiveBatch,
    RenderCommand, ScaledPixels, Scene, ScenePlanRequirements, plan::requirements_for,
};
use crate::point;
use smallvec::SmallVec;
use std::ops::Range;

/// What an animation tick redraws when only animated primitives changed.
#[derive(Debug)]
pub enum DamagePlan {
    /// Redraw over the retained frame.
    Partial(DamageRedraw),
    /// Encode the whole scene (still no view work).
    Full,
}

/// A partial redraw.
#[derive(Debug)]
pub struct DamageRedraw {
    /// Scissor in device pixels: whole-pixel edges, not clamped to the target.
    pub rect: Bounds<ScaledPixels>,
    /// The scene's render commands, restricted to what intersects `rect`, in order.
    pub commands: Vec<RenderCommand>,
    /// `scene.backdrop_filters` indices whose backdrop changed and must be re-blurred; every other
    /// backdrop filter in `commands` composites from the blur cache.
    pub reblur: SmallVec<[usize; 4]>,
    /// Resource totals for `commands`.
    pub requirements: ScenePlanRequirements,
}

fn hits(bounds: Bounds<ScaledPixels>, rect: &Bounds<ScaledPixels>) -> bool {
    !bounds.is_empty() && bounds.intersects(rect)
}

/// The sub-range of `range` from its first to its last member painting inside `rect`.
fn trim<T: PaintedBounds>(
    range: &Range<usize>,
    items: &[T],
    rect: &Bounds<ScaledPixels>,
) -> Option<Range<usize>> {
    let first = range
        .clone()
        .find(|&i| hits(items[i].painted_bounds(), rect))?;
    let last = range
        .clone()
        .rev()
        .find(|&i| hits(items[i].painted_bounds(), rect))?;
    Some(first..last + 1)
}

/// Rounds `bounds` out by one pixel to whole pixels.
fn round_out(bounds: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
    let corner = bounds.bottom_right();
    Bounds::from_corners(
        point(
            ScaledPixels((bounds.origin.x.0 - 1.0).floor()),
            ScaledPixels((bounds.origin.y.0 - 1.0).floor()),
        ),
        point(
            ScaledPixels((corner.x.0 + 1.0).ceil()),
            ScaledPixels((corner.y.0 + 1.0).ceil()),
        ),
    )
}

/// The damaged area and the glass whose backdrop it changes, or `None` when a lens is hit.
fn changed_area(scene: &Scene) -> Option<(Bounds<ScaledPixels>, SmallVec<[usize; 4]>)> {
    let mut changed: Vec<(Bounds<ScaledPixels>, DrawOrder)> = Vec::new();
    for sprite in scene
        .monochrome_sprites
        .iter()
        .filter(|sprite| sprite.frames.is_animated())
    {
        changed.push((sprite.painted_bounds(), sprite.order));
    }
    for sprite in scene
        .polychrome_sprites
        .iter()
        .filter(|sprite| sprite.frames.is_animated())
    {
        changed.push((sprite.painted_bounds(), sprite.order));
    }
    changed.retain(|(bounds, _)| !bounds.is_empty());
    let mut rect = changed
        .iter()
        .map(|(bounds, _)| *bounds)
        .reduce(|a, b| a.union(&b))?;

    let mut reblur = SmallVec::<[usize; 4]>::new();
    loop {
        let mut added = false;
        for (index, filter) in scene.backdrop_filters.iter().enumerate() {
            if filter.wallpaper.is_some() || reblur.contains(&index) {
                continue;
            }
            let region = filter.sampling_region();
            if !changed
                .iter()
                .any(|(bounds, order)| *order < filter.order && hits(*bounds, &region))
            {
                continue;
            }
            if filter.lens.is_active() {
                return None;
            }
            reblur.push(index);
            changed.push((BackdropFilter::painted_bounds(filter), filter.order));
            rect = rect.union(&region);
            added = true;
        }
        if !added {
            break;
        }
    }
    Some((round_out(rect), reblur))
}

pub(super) fn plan(scene: &Scene) -> DamagePlan {
    let Some((rect, reblur)) = changed_area(scene) else {
        return DamagePlan::Full;
    };

    let mut commands = Vec::new();
    // Open isolated groups that were kept, and nested groups inside a dropped group.
    let mut depth = 0usize;
    let mut skip = 0usize;
    for command in scene.render_commands() {
        if skip > 0 {
            match command {
                RenderCommand::BeginFilter { .. } => skip += 1,
                RenderCommand::EndFilter { .. } => skip -= 1,
                RenderCommand::Batch(_) => {}
            }
            continue;
        }
        match command {
            RenderCommand::Batch(batch) => {
                let kept = match batch {
                    PrimitiveBatch::Shadows { range, smoothed } => {
                        trim(range, &scene.shadows, &rect).map(|range| PrimitiveBatch::Shadows {
                            range,
                            smoothed: *smoothed,
                        })
                    }
                    PrimitiveBatch::Quads { range, smoothed } => trim(range, &scene.quads, &rect)
                        .map(|range| PrimitiveBatch::Quads {
                            range,
                            smoothed: *smoothed,
                        }),
                    PrimitiveBatch::Underlines(range) => {
                        trim(range, &scene.underlines, &rect).map(PrimitiveBatch::Underlines)
                    }
                    PrimitiveBatch::MonochromeSprites { texture_id, range } => {
                        trim(range, &scene.monochrome_sprites, &rect).map(|range| {
                            PrimitiveBatch::MonochromeSprites {
                                texture_id: *texture_id,
                                range,
                            }
                        })
                    }
                    PrimitiveBatch::SubpixelSprites { texture_id, range } => {
                        trim(range, &scene.subpixel_sprites, &rect).map(|range| {
                            PrimitiveBatch::SubpixelSprites {
                                texture_id: *texture_id,
                                range,
                            }
                        })
                    }
                    PrimitiveBatch::PolychromeSprites {
                        texture_id,
                        range,
                        smoothed,
                    } => trim(range, &scene.polychrome_sprites, &rect).map(|range| {
                        PrimitiveBatch::PolychromeSprites {
                            texture_id: *texture_id,
                            range,
                            smoothed: *smoothed,
                        }
                    }),
                    PrimitiveBatch::Paths { range, .. } => range
                        .clone()
                        .any(|i| hits(scene.paths[i].painted_bounds(), &rect))
                        .then(|| batch.clone()),
                    PrimitiveBatch::Surfaces(range) => {
                        if range
                            .clone()
                            .any(|i| hits(scene.surfaces[i].painted_bounds(), &rect))
                        {
                            return DamagePlan::Full;
                        }
                        None
                    }
                    PrimitiveBatch::BackdropFilters(range) => {
                        // Contiguous runs of members that paint inside the rect.
                        let mut run_start = None;
                        for i in range.clone().chain([range.end]) {
                            let inside = i < range.end
                                && hits(scene.backdrop_filters[i].painted_bounds(), &rect);
                            if inside {
                                let filter = &scene.backdrop_filters[i];
                                if depth > 0 && filter.wallpaper.is_none() {
                                    // Inside an isolated group the backdrop is inherited, so it
                                    // is never cached.
                                    return DamagePlan::Full;
                                }
                                run_start.get_or_insert(i);
                            } else if let Some(start) = run_start.take() {
                                commands.push(RenderCommand::Batch(
                                    PrimitiveBatch::BackdropFilters(start..i),
                                ));
                            }
                        }
                        None
                    }
                    PrimitiveBatch::FilterBoundary(_) => None,
                };
                if let Some(batch) = kept {
                    commands.push(RenderCommand::Batch(batch));
                }
            }
            RenderCommand::BeginFilter {
                boundary_index,
                target: FilterRenderTarget::Isolated(_),
            } => {
                let boundary = &scene.filter_boundaries[*boundary_index];
                let reach = boundary
                    .bounds
                    .dilate(ScaledPixels(3.0 * boundary.max_blur_radius()))
                    .intersect(&boundary.content_mask.bounds);
                if !hits(reach, &rect) {
                    skip = 1;
                } else if boundary.max_blur_radius() > 0.0 || boundary.scale != 1.0 {
                    return DamagePlan::Full;
                } else {
                    depth += 1;
                    commands.push(command.clone());
                }
            }
            RenderCommand::EndFilter {
                target: FilterRenderTarget::Isolated(_),
                ..
            } => {
                depth = depth.saturating_sub(1);
                commands.push(command.clone());
            }
            RenderCommand::BeginFilter { .. } | RenderCommand::EndFilter { .. } => {}
        }
    }

    if commands.is_empty() {
        return DamagePlan::Full;
    }
    let requirements = requirements_for(scene, &commands);
    DamagePlan::Partial(DamageRedraw {
        rect,
        commands,
        reblur,
        requirements,
    })
}

#[cfg(test)]
mod tests {
    // test-category: pure-logic
    use super::*;
    use crate::{
        AtlasTextureId, AtlasTextureKind, AtlasTile, ContentMask, DevicePixels, FrameLoop,
        MonochromeSprite, Quad, ScaledFilter, TileId, bounds, size,
    };

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        bounds(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(w), ScaledPixels(h)),
        )
    }

    fn mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: rect(0.0, 0.0, 400.0, 400.0),
            ..Default::default()
        }
    }

    fn flipbook(bounds: Bounds<ScaledPixels>, frames: FrameLoop) -> MonochromeSprite {
        MonochromeSprite {
            order: 0,
            padding: 0,
            bounds,
            content_mask: mask(),
            color: Default::default(),
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: 0,
                    kind: AtlasTextureKind::Monochrome,
                },
                tile_id: TileId(0),
                padding: 0,
                bounds: crate::Bounds::<DevicePixels>::default(),
            },
            transformation: crate::TransformationMatrix::unit(),
            frames,
        }
    }

    fn eight_frames() -> FrameLoop {
        FrameLoop {
            frame_count: 8,
            columns: 4,
            period_ms: 400,
            ..FrameLoop::NONE
        }
    }

    fn glass(bounds: Bounds<ScaledPixels>) -> BackdropFilter {
        BackdropFilter {
            bounds,
            content_mask: mask(),
            filters: [ScaledFilter::Blur(ScaledPixels(4.0))]
                .into_iter()
                .collect(),
            opacity: 1.0,
            ..Default::default()
        }
    }

    fn assert_close(actual: Bounds<ScaledPixels>, expected: Bounds<ScaledPixels>) {
        for (a, e) in [
            (actual.origin.x.0, expected.origin.x.0),
            (actual.origin.y.0, expected.origin.y.0),
            (actual.size.width.0, expected.size.width.0),
            (actual.size.height.0, expected.size.height.0),
        ] {
            assert!((a - e).abs() < 1e-3, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn frame_loop_sprites_mark_the_scene_animated() {
        let clipped = rect(90.0, 90.0, 20.0, 20.0);
        let mut small_mask = mask();
        small_mask.bounds = rect(0.0, 0.0, 100.0, 100.0);
        let mut sprite = flipbook(clipped, eight_frames());
        sprite.content_mask = small_mask;
        assert_close(sprite.painted_bounds(), rect(90.0, 90.0, 10.0, 10.0));

        let mut scene = Scene::default();
        scene.insert_primitive(sprite);
        scene.finish();
        assert!(scene.has_animations());

        let centered = rect(40.0, 40.0, 20.0, 20.0);
        let turning = flipbook(
            centered,
            FrameLoop {
                turn_period_ms: 1000,
                ..eight_frames()
            },
        );
        let half = 0.5 * (20.0f32 * 20.0 + 20.0 * 20.0).sqrt();
        assert_close(
            turning.painted_bounds(),
            rect(50.0 - half, 50.0 - half, 2.0 * half, 2.0 * half),
        );

        let mut still = Scene::default();
        still.insert_primitive(flipbook(centered, FrameLoop::NONE));
        still.finish();
        assert!(!still.has_animations());
    }

    #[test]
    fn damage_selects_intersecting_work_and_expands_under_changed_glass() {
        let mut scene = Scene::default();
        scene.insert_primitive(Quad {
            bounds: rect(0.0, 0.0, 400.0, 400.0),
            content_mask: mask(),
            ..Default::default()
        });
        scene.insert_primitive(glass(rect(0.0, 0.0, 40.0, 40.0)));
        scene.insert_primitive(flipbook(rect(10.0, 10.0, 10.0, 10.0), eight_frames()));
        scene.insert_primitive(flipbook(rect(100.0, 10.0, 10.0, 10.0), eight_frames()));
        let second_glass = glass(rect(95.0, 5.0, 30.0, 30.0));
        let second_region = second_glass.sampling_region();
        scene.insert_primitive(second_glass);
        scene.insert_primitive(Quad {
            bounds: rect(300.0, 300.0, 10.0, 10.0),
            content_mask: mask(),
            ..Default::default()
        });
        scene.finish();

        let DamagePlan::Partial(damage) = scene.damage_plan() else {
            panic!("expected a partial redraw");
        };
        assert_eq!(damage.reblur.as_slice(), &[1]);
        let contains = |outer: Bounds<ScaledPixels>, inner: Bounds<ScaledPixels>| {
            outer.origin.x <= inner.origin.x
                && outer.origin.y <= inner.origin.y
                && outer.bottom_right().x >= inner.bottom_right().x
                && outer.bottom_right().y >= inner.bottom_right().y
        };
        assert!(contains(damage.rect, second_region));
        assert!(contains(damage.rect, rect(10.0, 10.0, 10.0, 10.0)));

        let mut quads = Vec::new();
        let mut filters = Vec::new();
        let mut sprites = Vec::new();
        for command in &damage.commands {
            match command {
                RenderCommand::Batch(PrimitiveBatch::Quads { range, .. }) => {
                    quads.extend(range.clone())
                }
                RenderCommand::Batch(PrimitiveBatch::BackdropFilters(range)) => {
                    filters.extend(range.clone())
                }
                RenderCommand::Batch(PrimitiveBatch::MonochromeSprites { range, .. }) => {
                    sprites.extend(range.clone())
                }
                _ => {}
            }
        }
        assert_eq!(quads, [0]);
        assert_eq!(filters, [0, 1]);
        assert_eq!(sprites.len(), 2);
    }
}

//! Partial redraws for animation ticks and changes between finished scenes.

use super::{
    BackdropFilter, Bounds, DrawOrder, FilterRenderTarget, PaintOperation, PaintedBounds,
    Primitive, PrimitiveBatch, RenderCommand, ScaledPixels, Scene, ScenePlanRequirements,
    plan::requirements_for,
};
use crate::point;
use smallvec::SmallVec;
use std::ops::Range;

/// What to redraw over a retained frame.
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

pub(super) struct SceneChange {
    pub base: u64,
    pub rect: Option<Bounds<ScaledPixels>>,
}

/// Order-independent identities in recorded paint order, with the reach of each changed span.
pub(super) fn change_rect(previous: &Scene, scene: &Scene) -> Option<Bounds<ScaledPixels>> {
    let old = previous
        .paint_operations
        .iter()
        .map(operation_hash)
        .collect::<Vec<_>>();
    let new = scene
        .paint_operations
        .iter()
        .map(operation_hash)
        .collect::<Vec<_>>();
    let prefix = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
    let suffix = old[prefix..]
        .iter()
        .rev()
        .zip(new[prefix..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let mut rect: Option<Bounds<ScaledPixels>> = None;
    for (scene, end) in [(previous, old.len() - suffix), (scene, new.len() - suffix)] {
        let mut groups = SmallVec::<[(Bounds<ScaledPixels>, bool); 4]>::new();
        for (index, operation) in scene.paint_operations.iter().enumerate() {
            let mut reach = None;
            match operation {
                PaintOperation::Primitive(Primitive::FilterBoundary(boundary)) => {
                    if boundary.is_start {
                        let widened = boundary.motion.is_animated()
                            || boundary.max_blur_radius() > 0.0
                            || boundary.scale != 1.0;
                        let bounds = if boundary.motion.is_animated() {
                            boundary.motion.sweep(
                                boundary.bounds,
                                boundary.scale_origin,
                                scene.scale_factor,
                                boundary.motion_bounds,
                            )
                        } else {
                            boundary.bounds
                        };
                        let group_reach = bounds
                            .dilate(ScaledPixels(3.0 * boundary.max_blur_radius()))
                            .intersect(&boundary.content_mask.bounds);
                        groups.push((group_reach, widened));
                        reach = Some(group_reach);
                    } else {
                        reach = groups.pop().map(|(group_reach, _)| group_reach);
                    }
                }
                PaintOperation::Primitive(Primitive::Surface(_)) => return None,
                PaintOperation::Primitive(primitive) => reach = primitive.painted_bounds(),
                PaintOperation::LiveLayer(layer) => {
                    reach = Some(layer.bounds.intersect(&layer.content_mask.bounds))
                }
                PaintOperation::StartLayer(_) | PaintOperation::EndLayer => {}
            }
            if index >= prefix && index < end {
                reach = groups
                    .iter()
                    .find_map(|(group_reach, widened)| widened.then_some(*group_reach))
                    .or(reach);
                if let Some(reach) = reach.filter(|bounds| !bounds.is_empty()) {
                    rect = Some(rect.map_or(reach, |rect| rect.union(&reach)));
                }
            }
        }
    }
    Some(rect.unwrap_or_default())
}

fn operation_hash(operation: &PaintOperation) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hash = DefaultHasher::new();
    // These scene-buffer values have explicit, padding-free layouts, as checked by scene::abi.
    macro_rules! bytes {
        ($value:expr) => {{
            let value = &$value;
            hash.write(unsafe {
                std::slice::from_raw_parts(
                    std::ptr::from_ref(value).cast::<u8>(),
                    std::mem::size_of_val(value),
                )
            });
        }};
    }
    macro_rules! instance {
        ($value:expr) => {{
            let mut value = *$value;
            value.order = 0;
            bytes!(value);
        }};
    }
    macro_rules! filters {
        ($filters:expr) => {{
            $filters.len().hash(&mut hash);
            for filter in $filters {
                match filter {
                    crate::ScaledFilter::Blur(radius) => bytes!(radius.0),
                }
            }
        }};
    }
    macro_rules! motion {
        ($motion:expr) => {{
            let motion = &$motion;
            motion.start_ms.hash(&mut hash);
            motion.duration_ms.hash(&mut hash);
            std::mem::discriminant(&motion.repeat).hash(&mut hash);
            std::mem::discriminant(&motion.easing).hash(&mut hash);
            match &motion.easing {
                super::MotionEasing::CubicBezier(points) => bytes!(*points),
                super::MotionEasing::Linear { points, len } => {
                    bytes!(*points);
                    len.hash(&mut hash);
                }
                super::MotionEasing::Steps(steps) => steps.hash(&mut hash),
            }
            bytes!(motion.opacity);
            bytes!(motion.offset);
            bytes!(motion.relative_offset);
            motion.clip.is_some().hash(&mut hash);
            if let Some((from, to)) = motion.clip {
                bytes!(from);
                bytes!(to);
            }
            bytes!(motion.visible);
            bytes!(motion.scale);
            bytes!(motion.rotation);
            bytes!(motion.origin);
        }};
    }
    macro_rules! lens {
        ($lens:expr) => {{
            let lens = &$lens;
            lens.saturation.is_some().hash(&mut hash);
            if let Some(saturation) = lens.saturation {
                bytes!(saturation);
            }
            for value in [
                lens.bezel,
                lens.refraction,
                lens.specular,
                lens.dispersion,
                lens.frost,
                lens.lift,
                lens.rim,
                lens.depth,
                lens.light_angle,
                lens.opposite,
                lens.shade,
                lens.fresnel,
                lens.focus,
                lens.ambient,
                lens.halo,
                lens.fold,
                lens.fold_band,
                lens.fold_depth,
                lens.curve,
                lens.sharp_rim,
            ] {
                bytes!(value);
            }
            lens.sharp_full.hash(&mut hash);
            bytes!(lens.tone);
            bytes!(lens.hairline);
            lens.exact.hash(&mut hash);
        }};
    }
    std::mem::discriminant(operation).hash(&mut hash);
    match operation {
        PaintOperation::Primitive(primitive) => {
            std::mem::discriminant(primitive).hash(&mut hash);
            match primitive {
                Primitive::Shadow(value) => instance!(value),
                Primitive::Quad(value) => instance!(value),
                Primitive::Underline(value) => instance!(value),
                Primitive::MonochromeSprite(value) => instance!(value),
                Primitive::SubpixelSprite(value) => instance!(value),
                Primitive::PolychromeSprite(value) => instance!(value),
                Primitive::Path(path) => {
                    bytes!(path.clipped_bounds());
                    bytes!(path.color);
                    path.vertices.len().hash(&mut hash);
                    for vertex in &path.vertices {
                        bytes!(vertex.xy_position);
                        bytes!(vertex.st_position);
                    }
                }
                Primitive::FilterBoundary(boundary) => {
                    bytes!(boundary.bounds);
                    bytes!(boundary.content_mask);
                    bytes!(boundary.corner_radii);
                    bytes!(boundary.corner_smoothing);
                    filters!(&boundary.filters);
                    bytes!(boundary.opacity);
                    bytes!(boundary.scale);
                    bytes!(boundary.scale_origin);
                    motion!(boundary.motion);
                    bytes!(boundary.motion_bounds);
                    boundary.is_start.hash(&mut hash);
                    boundary.inline.hash(&mut hash);
                }
                Primitive::BackdropFilter(filter) => {
                    bytes!(filter.bounds);
                    bytes!(filter.content_mask);
                    bytes!(filter.corner_radii);
                    bytes!(filter.corner_smoothing);
                    filters!(&filter.filters);
                    bytes!(filter.opacity);
                    lens!(filter.lens);
                    filter.appear.is_some().hash(&mut hash);
                    if let Some(appear) = filter.appear {
                        lens!(appear.lens);
                        bytes!(appear.blur);
                        motion!(appear.motion);
                    }
                    filter.wallpaper.is_some().hash(&mut hash);
                    if let Some(wallpaper) = &filter.wallpaper {
                        wallpaper.key.hash(&mut hash);
                        wallpaper.generation.hash(&mut hash);
                        wallpaper.passes.len().hash(&mut hash);
                        for pass in &wallpaper.passes {
                            pass.module.hash(&mut hash);
                            pass.width.hash(&mut hash);
                            pass.height.hash(&mut hash);
                            pass.source.hash(&mut hash);
                            bytes!(pass.uniforms);
                        }
                    }
                }
                Primitive::Surface(_) => {}
            }
        }
        PaintOperation::StartLayer(bounds) => bytes!(*bounds),
        PaintOperation::EndLayer => {}
        PaintOperation::LiveLayer(layer) => {
            layer.key.hash(&mut hash);
            layer.start_ms.hash(&mut hash);
            bytes!(layer.bounds);
            bytes!(layer.content_mask);
        }
    }
    hash.finish()
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
fn changed_area(
    scene: &Scene,
    seed: Option<Bounds<ScaledPixels>>,
) -> Option<(Bounds<ScaledPixels>, SmallVec<[usize; 4]>)> {
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
    for boundary in &scene.filter_boundaries {
        if boundary.is_start && boundary.motion.is_animated() {
            changed.push((
                boundary
                    .motion
                    .sweep(
                        boundary.bounds,
                        boundary.scale_origin,
                        scene.scale_factor,
                        boundary.motion_bounds,
                    )
                    .intersect(&boundary.content_mask.bounds),
                boundary.order,
            ));
        }
    }
    let mut reblur = SmallVec::<[usize; 4]>::new();
    for (index, filter) in scene.backdrop_filters.iter().enumerate() {
        if filter
            .appear
            .is_some_and(|appear| appear.motion.is_animated())
        {
            changed.push((filter.painted_bounds(), filter.order));
            reblur.push(index);
        }
    }
    changed.retain(|(bounds, _)| !bounds.is_empty());
    let mut rect = changed
        .iter()
        .map(|(bounds, _)| *bounds)
        .chain(seed.filter(|rect| !rect.is_empty()))
        .reduce(|a, b| a.union(&b))
        .or(seed)?;

    loop {
        let mut added = false;
        for (index, filter) in scene.backdrop_filters.iter().enumerate() {
            if filter.wallpaper.is_some() || reblur.contains(&index) {
                continue;
            }
            let region = filter.sampling_region();
            if !seed.is_some_and(|rect| hits(rect, &region))
                && !changed
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
    Some((
        if rect.is_empty() {
            rect
        } else {
            round_out(rect)
        },
        reblur,
    ))
}

pub(super) fn plan(scene: &Scene) -> DamagePlan {
    let Some((rect, reblur)) = changed_area(scene, None) else {
        return DamagePlan::Full;
    };
    select_commands(scene, rect, reblur)
}

pub(super) fn plan_over(scene: &Scene, rect: Option<Bounds<ScaledPixels>>) -> DamagePlan {
    let Some((rect, reblur)) = rect.and_then(|rect| changed_area(scene, Some(rect))) else {
        return DamagePlan::Full;
    };
    select_commands(scene, rect, reblur)
}

fn select_commands(
    scene: &Scene,
    rect: Bounds<ScaledPixels>,
    reblur: SmallVec<[usize; 4]>,
) -> DamagePlan {
    let mut commands = Vec::new();
    // Open isolated groups that were kept, and nested groups inside a dropped group.
    let mut depth = 0usize;
    let mut skip = 0usize;
    let mut motion_depth = None;
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
                if motion_depth.is_some() {
                    commands.push(command.clone());
                    continue;
                }
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
                if motion_depth.is_some() {
                    depth += 1;
                    commands.push(command.clone());
                    continue;
                }
                let bounds = if boundary.motion.is_animated() {
                    boundary.motion.sweep(
                        boundary.bounds,
                        boundary.scale_origin,
                        scene.scale_factor,
                        boundary.motion_bounds,
                    )
                } else {
                    boundary.bounds
                };
                let reach = bounds
                    .dilate(ScaledPixels(3.0 * boundary.max_blur_radius()))
                    .intersect(&boundary.content_mask.bounds);
                if !hits(reach, &rect) {
                    skip = 1;
                } else {
                    depth += 1;
                    if boundary.motion.is_animated()
                        || boundary.max_blur_radius() > 0.0
                        || boundary.scale != 1.0
                    {
                        motion_depth = Some(depth);
                    }
                    commands.push(command.clone());
                }
            }
            RenderCommand::EndFilter {
                target: FilterRenderTarget::Isolated(_),
                ..
            } => {
                if motion_depth == Some(depth) {
                    motion_depth = None;
                }
                depth = depth.saturating_sub(1);
                commands.push(command.clone());
            }
            RenderCommand::BeginFilter { .. } | RenderCommand::EndFilter { .. } => {}
        }
    }

    if commands.is_empty() && !rect.is_empty() {
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
        AtlasTextureId, AtlasTextureKind, AtlasTile, BackdropAppear, BackdropLens, ContentMask,
        DevicePixels, FilterBoundary, FrameLoop, LayerMotion, MonochromeSprite, MotionEasing,
        MotionRepeat, Quad, ScaledFilter, TileId, backdrop_appear_parameters, bounds, px, size,
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
    fn scene_change_inside_opacity_group_stays_local() {
        let changed = rect(20.0, 20.0, 10.0, 10.0);
        let far = rect(300.0, 300.0, 10.0, 10.0);
        let build = |color: crate::Hsla, opacity: f32| {
            let mut scene = Scene::default();
            let boundary = FilterBoundary {
                order: 0,
                bounds: changed.union(&far),
                content_mask: mask(),
                corner_radii: Default::default(),
                corner_smoothing: 0.0,
                filters: SmallVec::new(),
                opacity,
                scale: 1.0,
                scale_origin: Default::default(),
                motion_bounds: Bounds::default(),
                motion: LayerMotion::NONE,
                is_start: true,
                inline: false,
            };
            scene.insert_primitive(boundary.clone());
            scene.insert_primitive(Quad {
                bounds: changed,
                content_mask: mask(),
                background: color.into(),
                ..Default::default()
            });
            scene.insert_primitive(Quad {
                bounds: far,
                content_mask: mask(),
                background: crate::white().into(),
                ..Default::default()
            });
            scene.insert_primitive(FilterBoundary {
                is_start: false,
                ..boundary
            });
            scene.insert_primitive(glass(rect(200.0, 200.0, 20.0, 20.0)));
            scene.insert_primitive(BackdropFilter {
                lens: BackdropLens {
                    bezel: 4.0,
                    ..Default::default()
                },
                ..glass(rect(350.0, 350.0, 20.0, 20.0))
            });
            scene
        };
        let mut a = build(crate::red(), 0.5);
        a.finish();
        let mut b = build(crate::blue(), 0.5);
        b.finish_over(&a);
        let Some(DamagePlan::Partial(plan)) = b.damage_plan_over(Some(a.generation())) else {
            panic!("expected a partial redraw");
        };
        assert_eq!(plan.rect.intersect(&changed), changed);
        assert!(!plan.rect.intersects(&far));
        assert!(plan.reblur.is_empty());

        let mut c = build(crate::blue(), 0.6);
        c.finish_over(&b);
        assert!(matches!(
            c.damage_plan_over(Some(b.generation())),
            Some(DamagePlan::Partial(plan))
                if plan.rect.intersect(&changed) == changed && plan.rect.intersect(&far) == far
        ));
    }

    #[test]
    fn backdrop_appearance_animates_lens_and_blur_with_partial_damage() {
        let mut scene = Scene::default();
        scene.insert_primitive(BackdropFilter {
            filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(20.0))],
            lens: BackdropLens {
                refraction: 10.0,
                ..Default::default()
            },
            appear: Some(BackdropAppear {
                lens: BackdropLens::default(),
                blur: 0.0,
                motion: LayerMotion {
                    start_ms: 0,
                    duration_ms: 200,
                    repeat: MotionRepeat::Once,
                    easing: MotionEasing::CubicBezier([0.0, 0.0, 1.0, 1.0]),
                    ..LayerMotion::NONE
                },
            }),
            ..glass(rect(100.0, 100.0, 40.0, 40.0))
        });
        scene.finish();

        assert!(scene.has_animations());
        let DamagePlan::Partial(damage) = scene.damage_plan() else {
            panic!("expected a partial redraw");
        };
        assert_eq!(damage.reblur.as_slice(), &[0]);
        let (lens, blur) = backdrop_appear_parameters(&scene.backdrop_filters[0], 100);
        assert_eq!(blur, 10.0);
        assert_eq!(lens.refraction, 5.0);
        assert!(!scene.animating_at(250));
    }

    #[test]
    fn live_layers_offload_unless_covered() {
        let mut scene = Scene {
            offload_animations: true,
            ..Scene::default()
        };
        for (key, bounds) in [
            (1, rect(0.0, 0.0, 20.0, 20.0)),
            (2, rect(40.0, 0.0, 20.0, 20.0)),
        ] {
            scene.insert_live_layer(crate::LiveLayer {
                key,
                bounds,
                content_mask: mask(),
                order: 0,
                start_ms: 0,
                paint: std::sync::Arc::new(|_, _| {}),
            });
        }
        scene.insert_primitive(Quad {
            bounds: rect(40.0, 0.0, 20.0, 20.0),
            content_mask: mask(),
            background: crate::red().into(),
            ..Default::default()
        });
        scene.finish();
        assert_eq!(scene.live_layers().len(), 1);
        assert_eq!(scene.live_layers()[0].key, 1);
        assert!(scene.covered_live_keys().contains(&2));
        assert!(scene.has_animations());
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

    #[test]
    fn animated_sprites_go_to_the_compositor_only_when_nothing_covers_them() {
        let finished = |offload: bool, covered: bool| {
            let mut scene = Scene::default();
            scene.offload_animations = offload;
            scene.insert_primitive(flipbook(rect(10.0, 10.0, 10.0, 10.0), eight_frames()));
            if covered {
                scene.insert_primitive(Quad {
                    bounds: rect(15.0, 15.0, 20.0, 20.0),
                    content_mask: mask(),
                    ..Default::default()
                });
            }
            scene.finish();
            scene
        };

        let alone = finished(true, false);
        assert_eq!(alone.compositor_sprites().len(), 1);
        assert!(alone.monochrome_sprites.is_empty());
        assert!(!alone.has_animations());

        for scene in [finished(true, true), finished(false, false)] {
            assert!(scene.compositor_sprites().is_empty());
            assert_eq!(scene.monochrome_sprites.len(), 1);
            assert!(scene.has_animations());
        }
    }
    #[test]
    fn uncovered_loop_motion_groups_are_offloaded() {
        let mut scene = Scene {
            offload_animations: true,
            scale_factor: 1.0,
            ..Scene::default()
        };
        let boundary = FilterBoundary {
            order: 0,
            bounds: mask().bounds,
            content_mask: mask(),
            corner_radii: Default::default(),
            corner_smoothing: 0.0,
            filters: SmallVec::new(),
            opacity: 1.0,
            scale: 1.0,
            scale_origin: point(ScaledPixels(10.0), ScaledPixels(10.0)),
            motion_bounds: Bounds::default(),
            motion: LayerMotion {
                duration_ms: 1000,
                repeat: MotionRepeat::Loop,
                offset: (point(px(0.0), px(0.0)), point(px(40.0), px(0.0))),
                ..LayerMotion::NONE
            },
            is_start: true,
            inline: false,
        };
        scene.insert_primitive(boundary.clone());
        for x in [10.0, 30.0] {
            scene.insert_primitive(Quad {
                bounds: rect(x, 10.0, 20.0, 20.0),
                content_mask: mask(),
                ..Default::default()
            });
        }
        scene.insert_primitive(FilterBoundary {
            is_start: false,
            ..boundary.clone()
        });
        scene.finish();
        assert_eq!(scene.compositor_groups().len(), 1);
        assert_eq!(scene.compositor_groups()[0].scene.quads.len(), 2);
        assert!(scene.quads.is_empty());
        assert!(!scene.has_animations());
        scene.insert_primitive(boundary.clone());
        scene.insert_primitive(Quad {
            bounds: rect(100.0, 100.0, 20.0, 20.0),
            content_mask: mask(),
            ..Default::default()
        });
        scene.insert_primitive(FilterBoundary {
            is_start: false,
            ..boundary
        });
        scene.insert_primitive(Quad {
            bounds: rect(110.0, 100.0, 20.0, 20.0),
            content_mask: mask(),
            ..Default::default()
        });
        scene.finish();
        assert_eq!(scene.compositor_groups().len(), 1);
        assert_eq!(scene.filter_boundaries.len(), 2);
    }

    #[test]
    fn motion_group_damage_keeps_the_sweep_and_group_commands() {
        let mut scene = Scene {
            scale_factor: 1.0,
            ..Scene::default()
        };
        scene.insert_primitive(Quad {
            bounds: rect(0.0, 0.0, 400.0, 400.0),
            content_mask: mask(),
            background: crate::white().into(),
            ..Default::default()
        });
        let boundary = FilterBoundary {
            order: 0,
            bounds: mask().bounds,
            content_mask: mask(),
            corner_radii: Default::default(),
            corner_smoothing: 0.0,
            filters: SmallVec::new(),
            opacity: 1.0,
            scale: 1.0,
            scale_origin: point(ScaledPixels(10.0), ScaledPixels(10.0)),
            motion_bounds: Bounds::default(),
            motion: LayerMotion {
                duration_ms: 1000,
                repeat: MotionRepeat::Loop,
                easing: MotionEasing::CubicBezier([0.0, 0.0, 1.0, 1.0]),
                offset: (point(px(0.0), px(0.0)), point(px(40.0), px(0.0))),
                ..LayerMotion::NONE
            },
            is_start: true,
            inline: false,
        };
        scene.insert_primitive(boundary.clone());
        scene.insert_primitive(Quad {
            bounds: rect(10.0, 10.0, 20.0, 20.0),
            content_mask: mask(),
            ..Default::default()
        });
        scene.insert_primitive(FilterBoundary {
            is_start: false,
            ..boundary
        });
        scene.insert_primitive(Quad {
            bounds: rect(300.0, 300.0, 10.0, 10.0),
            content_mask: mask(),
            ..Default::default()
        });
        scene.finish();
        assert!(scene.has_animations());
        let DamagePlan::Partial(damage) = scene.damage_plan() else {
            panic!("expected a partial redraw");
        };
        let swept = rect(10.0, 10.0, 60.0, 20.0);
        assert!(damage.rect.origin.x <= swept.origin.x);
        assert!(damage.rect.origin.y <= swept.origin.y);
        assert!(damage.rect.bottom_right().x >= swept.bottom_right().x);
        assert!(damage.rect.bottom_right().y >= swept.bottom_right().y);
        assert!(
            damage
                .commands
                .iter()
                .any(|command| matches!(command, RenderCommand::BeginFilter { .. }))
        );
        assert!(
            damage
                .commands
                .iter()
                .any(|command| matches!(command, RenderCommand::EndFilter { .. }))
        );
    }

    #[test]
    fn linear_motion_samples_clip_and_visibility() {
        let motion_bounds = rect(10.0, 20.0, 100.0, 80.0);
        let motion = LayerMotion {
            duration_ms: 1000,
            easing: MotionEasing::linear(&[(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]),
            clip: Some((
                crate::Edges {
                    right: px(20.0),
                    ..Default::default()
                },
                crate::Edges::default(),
            )),
            visible: (0.25, 0.5),
            ..LayerMotion::NONE
        };
        let sample = motion.sample(300, 1.0, motion_bounds);
        assert_eq!(sample.opacity, 1.0);
        assert!((sample.clip.unwrap().right().0 - (motion_bounds.right().0 - 8.0)).abs() < 1e-3);
        assert_eq!(motion.sample(600, 1.0, motion_bounds).opacity, 0.0);
        let still = LayerMotion {
            clip: None,
            ..motion
        };
        assert!(still.changes_between(499, 500));

        let motion_bounds = rect(0.0, 0.0, 20.0, 10.0);
        let motion = LayerMotion {
            duration_ms: 1000,
            easing: MotionEasing::linear(&[(0.0, 0.0), (1.0, 1.0)]),
            rotation: (0.0, std::f32::consts::FRAC_PI_2),
            ..LayerMotion::NONE
        };
        assert_eq!(
            motion.sample(500, 1.0, motion_bounds).rotation,
            std::f32::consts::FRAC_PI_4,
        );
        let swept = motion.sweep(motion_bounds, motion_bounds.center(), 1.0, motion_bounds);
        let rotated = rect(5.0, -5.0, 10.0, 20.0);
        assert!(
            swept.origin.x <= rotated.origin.x
                && swept.origin.y <= rotated.origin.y
                && swept.right() >= rotated.right()
                && swept.bottom() >= rotated.bottom()
        );
    }

    #[test]
    fn motion_steps_and_once_determine_frame_changes_and_activity() {
        let mut scene = Scene::default();
        scene.filter_boundaries.push(FilterBoundary {
            order: 0,
            bounds: mask().bounds,
            content_mask: mask(),
            corner_radii: Default::default(),
            corner_smoothing: 0.0,
            filters: SmallVec::new(),
            opacity: 1.0,
            scale: 1.0,
            scale_origin: Default::default(),
            motion_bounds: Bounds::default(),
            motion: LayerMotion {
                duration_ms: 1000,
                repeat: MotionRepeat::Loop,
                easing: MotionEasing::Steps(2),
                opacity: (0.0, 1.0),
                ..LayerMotion::NONE
            },
            is_start: true,
            inline: false,
        });
        assert!(!scene.changes_between(100, 400));
        assert!(scene.changes_between(400, 600));
        scene.filter_boundaries[0].motion = LayerMotion {
            duration_ms: 200,
            ..LayerMotion::NONE
        };
        assert!(scene.animating_at(100));
        assert!(!scene.animating_at(250));

        let mut boundary = scene.filter_boundaries.pop().unwrap();
        boundary.motion = LayerMotion {
            start_ms: crate::animation_time_ms().wrapping_sub(100),
            duration_ms: 1,
            repeat: MotionRepeat::Once,
            opacity: (0.0, 1.0),
            ..LayerMotion::NONE
        };
        scene.insert_primitive(boundary.clone());
        scene.insert_primitive(Quad {
            bounds: rect(10.0, 10.0, 20.0, 20.0),
            content_mask: mask(),
            ..Default::default()
        });
        scene.insert_primitive(FilterBoundary {
            is_start: false,
            ..boundary
        });
        scene.finish();
        assert!(scene.filter_boundaries.is_empty());
        assert_eq!(scene.quads.len(), 1);
        assert_eq!(scene.quads[0].bounds, rect(10.0, 10.0, 20.0, 20.0));
    }
}

//! Cache keys for blurred backdrops: equal keys mean the backdrop pixels a filter samples are
//! unchanged, so its snapshot and blur passes can be restored instead of re-run.

use crate::{path_types::rasterization_vertices, shaders::interface::bytes_of};
use gpui::{
    BackdropFilter, Bounds, Corners, DevicePixels, DrawOrder, MotionEasing, PaintedBounds,
    ScaledPixels, Scene, Size, backdrop_presence,
};
use std::hash::{DefaultHasher, Hash, Hasher};

const TAG_SHADOW: u8 = 0;
const TAG_QUAD: u8 = 1;
const TAG_PATH: u8 = 2;
const TAG_UNDERLINE: u8 = 3;
const TAG_MONOCHROME: u8 = 4;
const TAG_SUBPIXEL: u8 = 5;
const TAG_POLYCHROME: u8 = 6;
const TAG_FILTER: u8 = 7;
const TAG_BOUNDARY: u8 = 8;

/// Identity of a scene's keys at `now_ms`, including its generation, running appear motions,
/// and the sampled steps or active state of motion groups.
pub fn keys_epoch(scene: &Scene, now_ms: u32) -> u64 {
    let running = scene
        .backdrop_filters
        .iter()
        .filter(|filter| filter.appear.is_some_and(|a| a.motion.is_active(now_ms)))
        .count() as u64;
    let mut epoch = scene.generation() ^ (running << 48);
    for boundary in &scene.filter_boundaries {
        if boundary.is_start && boundary.motion.is_animated() {
            let sample = match boundary.motion.easing {
                MotionEasing::Steps(_) => u64::from(boundary.motion.progress(now_ms).to_bits()),
                _ => u64::from(boundary.motion.is_active(now_ms)),
            };
            epoch = epoch.rotate_left(1) ^ sample;
        }
    }
    epoch
}

/// Cache identity of each `scene.backdrop_filters[i]`'s blurred backdrop for a `target`-sized
/// frame at animation time `now_ms`, or `None` when it cannot be reused (it must be blurred this
/// frame).
pub fn backdrop_keys(scene: &Scene, target: Size<DevicePixels>, now_ms: u32) -> Vec<Option<u64>> {
    let mut keys: Vec<Option<u64>> = Vec::with_capacity(scene.backdrop_filters.len());
    for (index, filter) in scene.backdrop_filters.iter().enumerate() {
        let key = filter_key(scene, target, index, filter, &keys, now_ms);
        keys.push(key);
    }
    keys
}

fn filter_key(
    scene: &Scene,
    target: Size<DevicePixels>,
    index: usize,
    filter: &BackdropFilter,
    keys: &[Option<u64>],
    now_ms: u32,
) -> Option<u64> {
    // A running appear changes its blur every tick; it is re-blurred each frame instead. Once it
    // has ended the filter is static, and its settled presence is part of the key.
    if filter.wallpaper.is_some() || filter.appear.is_some_and(|a| a.motion.is_active(now_ms)) {
        return None;
    }
    let region = filter.sampling_region();
    let order = filter.order;
    let touches = |bounds: Bounds<ScaledPixels>| !bounds.is_empty() && bounds.intersects(&region);

    let surfaces_end = scene.surfaces.partition_point(|p| p.order < order);
    if scene.surfaces[..surfaces_end]
        .iter()
        .any(|surface| touches(surface.painted_bounds()))
    {
        return None;
    }
    // An animated sprite changes every tick, so what is behind this filter is never the same.
    let monochrome_end = scene
        .monochrome_sprites
        .partition_point(|p| p.order < order);
    let polychrome_end = scene
        .polychrome_sprites
        .partition_point(|p| p.order < order);
    if scene.monochrome_sprites[..monochrome_end]
        .iter()
        .any(|sprite| sprite.frames.is_animated() && touches(sprite.painted_bounds()))
        || scene.polychrome_sprites[..polychrome_end]
            .iter()
            .any(|sprite| sprite.frames.is_animated() && touches(sprite.painted_bounds()))
    {
        return None;
    }
    if scene.has_live_box_morphs()
        && (scene.quads.iter().enumerate().any(|(index, quad)| {
            quad.order < order && quad.padding != 0 && touches(scene.quad_reach(index))
        }) || scene.shadows.iter().enumerate().any(|(index, shadow)| {
            shadow.order < order && shadow.padding != 0 && touches(scene.shadow_reach(index))
        }))
    {
        return None;
    }
    let filters_end = scene.backdrop_filters[..index].partition_point(|p| p.order < order);
    let mut entries: Vec<(DrawOrder, u8, u64)> = Vec::new();
    for (lower_index, lower) in scene.backdrop_filters[..filters_end].iter().enumerate() {
        if !touches(lower.painted_bounds()) {
            continue;
        }
        let mut hasher = DefaultHasher::new();
        if let Some(paint) = &lower.wallpaper {
            // A clocked wallpaper changes with time, so what is behind this filter is never the same.
            if paint.clock.is_some() {
                return None;
            }
            paint.key.hash(&mut hasher);
            paint.generation.hash(&mut hasher);
        } else {
            let lower_key = keys[lower_index]?;
            if lower.lens.is_active() {
                return None;
            }
            lower_key.hash(&mut hasher);
        }
        hash_bounds(&mut hasher, &lower.bounds);
        hash_bounds(&mut hasher, &lower.content_mask.bounds);
        lower.content_mask.corners.hash(&mut hasher);
        hash_corners(&mut hasher, &lower.corner_radii);
        lower.corner_smoothing.to_bits().hash(&mut hasher);
        lower.opacity.to_bits().hash(&mut hasher);
        entries.push((lower.order, TAG_FILTER, hasher.finish()));
    }

    macro_rules! gather_instances {
        ($slice:expr, $tag:expr, $type:ty) => {{
            let end = $slice.partition_point(|p| p.order < order);
            for p in &$slice[..end] {
                if touches(p.painted_bounds()) {
                    let mut stripped: $type = *p;
                    stripped.order = 0;
                    entries.push((p.order, $tag, hash_bytes(bytes_of(&stripped))));
                }
            }
        }};
    }
    gather_instances!(scene.shadows, TAG_SHADOW, gpui::Shadow);
    gather_instances!(scene.quads, TAG_QUAD, gpui::Quad);
    gather_instances!(scene.underlines, TAG_UNDERLINE, gpui::Underline);
    gather_instances!(
        scene.monochrome_sprites,
        TAG_MONOCHROME,
        gpui::MonochromeSprite
    );
    gather_instances!(scene.subpixel_sprites, TAG_SUBPIXEL, gpui::SubpixelSprite);
    gather_instances!(
        scene.polychrome_sprites,
        TAG_POLYCHROME,
        gpui::PolychromeSprite
    );

    let paths_end = scene.paths.partition_point(|p| p.order < order);
    for path in &scene.paths[..paths_end] {
        if touches(path.painted_bounds()) {
            let mut hasher = DefaultHasher::new();
            for vertex in rasterization_vertices(std::slice::from_ref(path)) {
                hasher.write(bytes_of(&vertex));
            }
            entries.push((path.order, TAG_PATH, hasher.finish()));
        }
    }

    let boundaries_end = scene.filter_boundaries.partition_point(|p| p.order < order);
    for boundary in &scene.filter_boundaries[..boundaries_end] {
        if touches(boundary.bounds.intersect(&boundary.content_mask.bounds)) {
            // Only continuously changing groups block reuse. Other groups key by their sample.
            if !matches!(boundary.motion.easing, MotionEasing::Steps(_))
                && boundary.motion.is_active(now_ms)
            {
                return None;
            }
            let mut hasher = DefaultHasher::new();
            hash_bounds(&mut hasher, &boundary.bounds);
            hash_bounds(&mut hasher, &boundary.content_mask.bounds);
            boundary.content_mask.corners.hash(&mut hasher);
            hash_corners(&mut hasher, &boundary.corner_radii);
            boundary.corner_smoothing.to_bits().hash(&mut hasher);
            boundary.max_blur_radius().to_bits().hash(&mut hasher);
            boundary.opacity.to_bits().hash(&mut hasher);
            boundary.scale.to_bits().hash(&mut hasher);
            boundary.scale_origin.x.0.to_bits().hash(&mut hasher);
            boundary.scale_origin.y.0.to_bits().hash(&mut hasher);
            boundary.motion.progress(now_ms).to_bits().hash(&mut hasher);
            boundary.motion.visible_at(now_ms).hash(&mut hasher);
            boundary.is_start.hash(&mut hasher);
            boundary.inline.hash(&mut hasher);
            entries.push((boundary.order, TAG_BOUNDARY, hasher.finish()));
        }
    }

    entries.sort_by_key(|&(order, tag, _)| (order, tag));
    let mut hasher = DefaultHasher::new();
    target.width.0.hash(&mut hasher);
    target.height.0.hash(&mut hasher);
    hash_bounds(&mut hasher, &filter.bounds);
    filter.max_blur_radius().to_bits().hash(&mut hasher);
    filter.lens.is_active().hash(&mut hasher);
    backdrop_presence(filter, now_ms)
        .to_bits()
        .hash(&mut hasher);
    for (_, tag, hash) in entries {
        tag.hash(&mut hasher);
        hash.hash(&mut hasher);
    }
    Some(hasher.finish())
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write(bytes);
    hasher.finish()
}

fn hash_bounds(hasher: &mut DefaultHasher, bounds: &Bounds<ScaledPixels>) {
    for value in [
        bounds.origin.x.0,
        bounds.origin.y.0,
        bounds.size.width.0,
        bounds.size.height.0,
    ] {
        value.to_bits().hash(hasher);
    }
}

fn hash_corners(hasher: &mut DefaultHasher, corners: &Corners<ScaledPixels>) {
    for value in [
        corners.top_left.0,
        corners.top_right.0,
        corners.bottom_right.0,
        corners.bottom_left.0,
    ] {
        value.to_bits().hash(hasher);
    }
}

#[cfg(test)]
mod tests {
    // test-category: pure-logic
    use super::*;
    use gpui::{
        BackdropFilter, ContentMask, FilterBoundary, LayerMotion, Quad, ScaledFilter, Scene, blue,
        bounds, point, red, size, solid_background,
    };

    fn scaled_bounds(x: f32, y: f32, extent: f32) -> Bounds<ScaledPixels> {
        bounds(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(extent), ScaledPixels(extent)),
        )
    }

    fn key_of_filter(color_a: gpui::Hsla, color_b: gpui::Hsla) -> Option<u64> {
        let mask = ContentMask {
            bounds: scaled_bounds(0.0, 0.0, 1000.0),
            ..Default::default()
        };
        let quad = |x: f32, color: gpui::Hsla| Quad {
            bounds: scaled_bounds(x, x, 50.0),
            content_mask: mask,
            background: solid_background(color),
            ..Default::default()
        };
        let mut scene = Scene::default();
        scene.insert_primitive(quad(110.0, color_a));
        scene.insert_primitive(quad(800.0, color_b));
        scene.insert_primitive(BackdropFilter {
            bounds: scaled_bounds(100.0, 100.0, 100.0),
            content_mask: mask,
            filters: [ScaledFilter::Blur(ScaledPixels(4.0))]
                .into_iter()
                .collect(),
            opacity: 1.0,
            ..Default::default()
        });
        scene.finish();
        let target = size(DevicePixels(1000), DevicePixels(1000));
        let keys = backdrop_keys(&scene, target, 0);
        assert_eq!(keys.len(), 1);
        keys[0]
    }

    #[test]
    fn key_follows_only_content_under_the_region() {
        let base = key_of_filter(red(), red());
        assert!(base.is_some());
        assert_ne!(base, key_of_filter(blue(), red()));
        assert_eq!(base, key_of_filter(red(), blue()));
    }

    #[test]
    fn animated_group_under_the_region_has_no_key() {
        let key_with_motion = |motion| {
            let mask = ContentMask {
                bounds: scaled_bounds(0.0, 0.0, 1000.0),
                ..Default::default()
            };
            let mut scene = Scene::default();
            let boundary = FilterBoundary {
                order: 0,
                bounds: scaled_bounds(110.0, 110.0, 50.0),
                content_mask: mask,
                corner_radii: Default::default(),
                corner_smoothing: 0.0,
                filters: Default::default(),
                opacity: 0.5,
                scale: 1.0,
                scale_origin: Default::default(),
                motion_bounds: Bounds::default(),
                motion,
                is_start: true,
                inline: false,
            };
            scene.insert_primitive(boundary.clone());
            scene.insert_primitive(Quad {
                bounds: scaled_bounds(110.0, 110.0, 50.0),
                content_mask: mask,
                background: solid_background(red()),
                ..Default::default()
            });
            scene.insert_primitive(FilterBoundary {
                is_start: false,
                ..boundary
            });
            scene.insert_primitive(BackdropFilter {
                bounds: scaled_bounds(100.0, 100.0, 100.0),
                content_mask: mask,
                filters: [ScaledFilter::Blur(ScaledPixels(4.0))]
                    .into_iter()
                    .collect(),
                opacity: 1.0,
                ..Default::default()
            });
            scene.finish();
            backdrop_keys(&scene, size(DevicePixels(1000), DevicePixels(1000)), 0)[0]
        };
        assert!(key_with_motion(LayerMotion::NONE).is_some());
        assert!(
            key_with_motion(LayerMotion {
                duration_ms: 1000,
                opacity: (0.0, 1.0),
                ..LayerMotion::NONE
            })
            .is_none()
        );
    }

    #[test]
    fn stepped_group_key_follows_its_step() {
        let mask = ContentMask {
            bounds: scaled_bounds(0.0, 0.0, 1000.0),
            ..Default::default()
        };
        let group_bounds = scaled_bounds(110.0, 110.0, 50.0);
        let [stepped, continuous] =
            [MotionEasing::Steps(4), LayerMotion::NONE.easing].map(|easing| {
                let mut scene = Scene::default();
                let boundary = FilterBoundary {
                    order: 0,
                    bounds: group_bounds,
                    content_mask: mask,
                    corner_radii: Default::default(),
                    corner_smoothing: 0.0,
                    filters: Default::default(),
                    opacity: 1.0,
                    scale: 1.0,
                    scale_origin: Default::default(),
                    motion_bounds: group_bounds,
                    motion: LayerMotion {
                        duration_ms: 400,
                        easing,
                        opacity: (0.0, 1.0),
                        ..LayerMotion::NONE
                    },
                    is_start: true,
                    inline: false,
                };
                scene.insert_primitive(boundary.clone());
                scene.insert_primitive(Quad {
                    bounds: group_bounds,
                    content_mask: mask,
                    background: solid_background(red()),
                    ..Default::default()
                });
                scene.insert_primitive(FilterBoundary {
                    is_start: false,
                    ..boundary
                });
                scene.insert_primitive(BackdropFilter {
                    bounds: scaled_bounds(100.0, 100.0, 100.0),
                    content_mask: mask,
                    filters: [ScaledFilter::Blur(ScaledPixels(4.0))]
                        .into_iter()
                        .collect(),
                    opacity: 1.0,
                    ..Default::default()
                });
                scene.finish();
                scene
            });
        let target = size(DevicePixels(1000), DevicePixels(1000));
        let first = backdrop_keys(&stepped, target, 125)[0];
        let same_step = backdrop_keys(&stepped, target, 150)[0];
        let next_step = backdrop_keys(&stepped, target, 250)[0];
        assert!(first.is_some() && next_step.is_some());
        assert_eq!(first, same_step);
        assert_ne!(first, next_step);
        assert!(backdrop_keys(&continuous, target, 125)[0].is_none());
    }

    #[test]
    fn ended_appear_is_keyed_and_no_longer_damaged() {
        let mask = ContentMask {
            bounds: scaled_bounds(0.0, 0.0, 1000.0),
            ..Default::default()
        };
        let mut scene = Scene::default();
        scene.insert_primitive(BackdropFilter {
            bounds: scaled_bounds(100.0, 100.0, 100.0),
            content_mask: mask,
            filters: [ScaledFilter::Blur(ScaledPixels(4.0))]
                .into_iter()
                .collect(),
            opacity: 1.0,
            appear: Some(gpui::BackdropAppear {
                lens: Default::default(),
                blur: 0.0,
                motion: LayerMotion {
                    duration_ms: 100,
                    opacity: (0.0, 1.0),
                    ..LayerMotion::NONE
                },
            }),
            ..Default::default()
        });
        scene.finish();
        let target = size(DevicePixels(1000), DevicePixels(1000));
        assert!(backdrop_keys(&scene, target, 50)[0].is_none());
        assert!(backdrop_keys(&scene, target, 200)[0].is_some());
        let reblurred = |now, drawn| {
            let plan = scene
                .damage_plan_with_wallpapers(Some(scene.generation()), now, drawn, |_| false)
                .unwrap();
            let gpui::DamagePlan::Partial(plan) = plan.into_owned() else {
                panic!("expected a partial redraw");
            };
            plan.reblur
        };
        assert_eq!(reblurred(50, 40).as_slice(), &[0]);
        assert_eq!(reblurred(200, 50).as_slice(), &[0]);
        assert!(reblurred(300, 200).is_empty());
    }

    #[test]
    fn clocked_wallpaper_under_the_region_has_no_key() {
        let keys_with_clock = |clock| {
            let mask = ContentMask {
                bounds: scaled_bounds(0.0, 0.0, 1000.0),
                ..Default::default()
            };
            let mut scene = Scene::default();
            scene.insert_primitive(BackdropFilter {
                bounds: scaled_bounds(0.0, 0.0, 1000.0),
                content_mask: mask,
                wallpaper: Some(gpui::WallpaperPaint {
                    key: 1,
                    generation: 0,
                    image: None,
                    clock,
                    passes: [gpui::WallpaperPass {
                        module: 0,
                        width: 8,
                        height: 8,
                        noise: 0,
                        previous: false,
                        image: false,
                        uniforms: [[0.0; 4]; 15],
                    }]
                    .into_iter()
                    .collect(),
                }),
                opacity: 1.0,
                ..Default::default()
            });
            scene.insert_primitive(BackdropFilter {
                bounds: scaled_bounds(100.0, 100.0, 100.0),
                content_mask: mask,
                filters: [ScaledFilter::Blur(ScaledPixels(4.0))]
                    .into_iter()
                    .collect(),
                opacity: 1.0,
                ..Default::default()
            });
            scene.finish();
            backdrop_keys(&scene, size(DevicePixels(1000), DevicePixels(1000)), 0)
        };
        let keys = keys_with_clock(None);
        assert!(keys[1].is_some());
        let keys = keys_with_clock(Some(gpui::WallpaperClock {
            start_ms: 0,
            start_s: 0.0,
            fps: 20,
            period_s: 10.0,
            animated_from: 0,
        }));
        assert!(keys[1].is_none());
    }
}

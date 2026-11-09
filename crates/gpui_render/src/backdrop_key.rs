//! Cache keys for blurred backdrops: equal keys mean the backdrop pixels a filter samples are
//! unchanged, so its snapshot and blur passes can be restored instead of re-run.

use crate::{path_types::rasterization_vertices, shaders::interface::bytes_of};
use gpui::{
    BackdropFilter, Bounds, Corners, DevicePixels, DrawOrder, PaintedBounds, ScaledPixels, Scene,
    Size,
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

/// Cache identity of each `scene.backdrop_filters[i]`'s blurred backdrop for a `target`-sized
/// frame, or `None` when it cannot be reused (it must be blurred this frame).
pub fn backdrop_keys(scene: &Scene, target: Size<DevicePixels>) -> Vec<Option<u64>> {
    let mut keys: Vec<Option<u64>> = Vec::with_capacity(scene.backdrop_filters.len());
    for (index, filter) in scene.backdrop_filters.iter().enumerate() {
        let key = filter_key(scene, target, index, filter, &keys);
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
) -> Option<u64> {
    if filter.wallpaper.is_some() {
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
    let filters_end = scene.backdrop_filters[..index].partition_point(|p| p.order < order);
    let mut entries: Vec<(DrawOrder, u8, u64)> = Vec::new();
    for (lower_index, lower) in scene.backdrop_filters[..filters_end].iter().enumerate() {
        if !touches(lower.painted_bounds()) {
            continue;
        }
        let mut hasher = DefaultHasher::new();
        if let Some(paint) = &lower.wallpaper {
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
            let mut hasher = DefaultHasher::new();
            hash_bounds(&mut hasher, &boundary.bounds);
            hash_bounds(&mut hasher, &boundary.content_mask.bounds);
            hash_corners(&mut hasher, &boundary.corner_radii);
            boundary.corner_smoothing.to_bits().hash(&mut hasher);
            boundary.max_blur_radius().to_bits().hash(&mut hasher);
            boundary.opacity.to_bits().hash(&mut hasher);
            boundary.scale.to_bits().hash(&mut hasher);
            boundary.scale_origin.x.0.to_bits().hash(&mut hasher);
            boundary.scale_origin.y.0.to_bits().hash(&mut hasher);
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
        BackdropFilter, ContentMask, Quad, ScaledFilter, Scene, blue, bounds, point, red, size,
        solid_background,
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
        let keys = backdrop_keys(&scene, target);
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
}

//! Animated sprites the platform compositor draws instead of the GPU frame.

use super::{
    DrawOrder, FilterRenderTarget, FrameLoop, PaintOperation, PaintedBounds, Primitive,
    RenderCommand, Scene,
};
use crate::{
    AtlasTile, Bounds, ContentMask, Corners, DevicePixels, Hsla, LayerMotion, MotionRepeat, Point,
    ScaledPixels, Size, TransformationMatrix, point,
};
use smallvec::SmallVec;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::Arc,
};

/// An animated sprite the platform compositor draws instead of the GPU frame.
#[derive(Clone, Debug, PartialEq)]
pub struct CompositorSprite {
    /// The atlas grid or rasterised group pixels.
    pub contents: SpriteContents,
    /// Where the sprite is drawn, in device pixels.
    pub bounds: Bounds<ScaledPixels>,
    /// The visible area, in device pixels (the content mask's bounds).
    pub clip: Bounds<ScaledPixels>,
    /// `Some` for a monochrome sprite (an alpha mask tinted with the colour); `None` for a
    /// polychrome one.
    pub color: Option<Hsla>,
    /// The flipbook and turn clocks.
    pub frames: FrameLoop,
    /// The compositor transform, or identity for a sprite.
    pub motion: LayerMotion,
}

/// Image data for a platform compositor layer.
#[derive(Clone, Debug, PartialEq)]
pub enum SpriteContents {
    /// A flipbook grid in the sprite atlas.
    Atlas(AtlasTile),
    /// Premultiplied BGRA rows, top-down.
    Pixels {
        /// Image dimensions in device pixels.
        size: Size<DevicePixels>,
        /// Tightly packed rows.
        bytes: Arc<[u8]>,
    },
}

/// A painted subtree rasterised once by the platform compositor.
#[expect(missing_docs)]
pub struct CompositorGroup {
    pub scene: Scene,
    pub rect: Bounds<ScaledPixels>,
    pub motion: LayerMotion,
    pub motion_bounds: Bounds<ScaledPixels>,
    pub origin: Point<ScaledPixels>,
    pub key: u64,
    pub content_mask: ContentMask<ScaledPixels>,
}

/// Moves every animated sprite whose pixels do not depend on what is drawn around it out of the
/// scene's sprite lists into `compositor_sprites`. Returns whether any moved.
///
/// Such a sprite has no transformation, sits in no isolated filter group, is outside its content
/// mask's fade, and has nothing at or above its draw order that paints over it or samples it.
pub(super) fn offload(scene: &mut Scene) -> bool {
    let boundaries = &scene.filter_boundaries;
    let isolated: SmallVec<[(DrawOrder, DrawOrder); 4]> = scene
        .render_plan
        .commands
        .iter()
        .filter_map(|command| match command {
            RenderCommand::EndFilter {
                boundary_index,
                closing_boundary_index,
                target: FilterRenderTarget::Isolated(_),
            } => Some((
                boundaries[*boundary_index].order,
                boundaries[*closing_boundary_index].order,
            )),
            _ => None,
        })
        .collect();
    let free = |sprite: *const (),
                order: DrawOrder,
                painted: Bounds<ScaledPixels>,
                mask: &ContentMask<ScaledPixels>| {
        !painted.is_empty()
            && unfaded(&painted, mask)
            && !isolated
                .iter()
                .any(|&(start, end)| start < order && order < end)
            && !covered(scene, sprite, (order, order), &painted)
    };

    let monochrome: Vec<bool> = scene
        .monochrome_sprites
        .iter()
        .map(|sprite| {
            sprite.frames.is_animated()
                && sprite.transformation == TransformationMatrix::unit()
                && free(
                    std::ptr::from_ref(sprite).cast(),
                    sprite.order,
                    sprite.painted_bounds(),
                    &sprite.content_mask,
                )
        })
        .collect();
    let polychrome: Vec<bool> = scene
        .polychrome_sprites
        .iter()
        .map(|sprite| {
            sprite.frames.is_animated()
                && sprite.opacity == 1.0
                && !sprite.grayscale.is_enabled()
                && sprite.corner_radii == Corners::default()
                && free(
                    std::ptr::from_ref(sprite).cast(),
                    sprite.order,
                    sprite.painted_bounds(),
                    &sprite.content_mask,
                )
        })
        .collect();

    let mut groups = Vec::new();
    for pair in boundaries.windows(2) {
        let (start, end) = (&pair[0], &pair[1]);
        if !start.is_start
            || end.is_start
            || !start.motion.is_animated()
            || start.motion.repeat == MotionRepeat::Once
            || start.motion.visible != (0.0, 1.0)
            || isolated
                .iter()
                .any(|&(a, b)| a < start.order && end.order < b)
        {
            continue;
        }
        let mut content = Scene {
            scale_factor: scene.scale_factor,
            ..Scene::default()
        };
        let mut inside = false;
        let mut eligible = true;
        for operation in &scene.paint_operations {
            match operation {
                PaintOperation::Primitive(Primitive::FilterBoundary(boundary)) => {
                    if boundary.order == end.order {
                        break;
                    }
                    if inside {
                        eligible = false;
                        break;
                    }
                    inside = boundary.order == start.order;
                }
                PaintOperation::Primitive(primitive) if inside => {
                    if matches!(
                        primitive,
                        Primitive::Surface(_) | Primitive::BackdropFilter(_)
                    ) || matches!(primitive, Primitive::MonochromeSprite(s) if s.frames.is_animated())
                        || matches!(primitive, Primitive::PolychromeSprite(s) if s.frames.is_animated())
                        || primitive
                            .painted_bounds()
                            .is_some_and(|bounds| !unfaded(&bounds, primitive.content_mask()))
                    {
                        eligible = false;
                        break;
                    }
                    content.insert_primitive(primitive.clone());
                }
                PaintOperation::StartLayer(bounds) if inside => content.push_layer(*bounds),
                PaintOperation::EndLayer if inside => content.pop_layer(),
                _ => {}
            }
        }
        let Some(rect) = content
            .paint_operations
            .iter()
            .filter_map(|operation| match operation {
                PaintOperation::Primitive(primitive) => primitive.painted_bounds(),
                _ => None,
            })
            .reduce(|a, b| a.union(&b))
        else {
            continue;
        };
        let sweep = start.motion.sweep(
            rect,
            start.scale_origin,
            scene.scale_factor,
            start.motion_bounds,
        );
        if !eligible
            || !unfaded(&sweep, &start.content_mask)
            || covered(scene, std::ptr::null(), (start.order, end.order), &sweep)
        {
            continue;
        }
        let rect = Bounds::from_corners(
            point(
                ScaledPixels(rect.left().0.floor()),
                ScaledPixels(rect.top().0.floor()),
            ),
            point(
                ScaledPixels(rect.right().0.ceil()),
                ScaledPixels(rect.bottom().0.ceil()),
            ),
        );
        content.finish();
        let mut hash = DefaultHasher::new();
        format!(
            "{:?}{:?}{:?}{:?}{:?}{:?}{:?}{rect:?}{:?}",
            content.shadows,
            content.quads,
            content.paths,
            content.underlines,
            content.monochrome_sprites,
            content.subpixel_sprites,
            content.polychrome_sprites,
            start.motion
        )
        .hash(&mut hash);
        groups.push((
            (start.order, end.order),
            CompositorGroup {
                scene: content,
                rect,
                motion: start.motion,
                motion_bounds: start.motion_bounds,
                origin: start.scale_origin,
                key: hash.finish(),
                content_mask: start.content_mask,
            },
        ));
    }
    let removed = |order| {
        groups
            .iter()
            .any(|((start, end), _)| *start <= order && order <= *end)
    };
    scene.shadows.retain(|item| !removed(item.order));
    scene.quads.retain(|item| !removed(item.order));
    scene.paths.retain(|item| !removed(item.order));
    scene.underlines.retain(|item| !removed(item.order));
    // Sprite flags refer to the original lists; group sprites are removed below after using them.
    scene.subpixel_sprites.retain(|item| !removed(item.order));
    scene.filter_boundaries.retain(|item| !removed(item.order));
    let groups_moved = !groups.is_empty();

    let before = scene.compositor_sprites.len();
    let mut flags = monochrome.into_iter();
    scene.monochrome_sprites.retain(|sprite| {
        if !flags.next().unwrap_or(false) {
            return true;
        }
        scene.compositor_sprites.push(CompositorSprite {
            contents: SpriteContents::Atlas(sprite.tile),
            bounds: sprite.bounds,
            clip: sprite.content_mask.bounds,
            color: Some(sprite.color.into()),
            frames: sprite.frames,
            motion: LayerMotion::NONE,
        });
        false
    });
    let mut flags = polychrome.into_iter();
    scene.polychrome_sprites.retain(|sprite| {
        if !flags.next().unwrap_or(false) {
            return true;
        }
        scene.compositor_sprites.push(CompositorSprite {
            contents: SpriteContents::Atlas(sprite.tile),
            bounds: sprite.bounds,
            clip: sprite.content_mask.bounds,
            color: None,
            frames: sprite.frames,
            motion: LayerMotion::NONE,
        });
        false
    });
    scene.monochrome_sprites.retain(|item| !removed(item.order));
    scene.polychrome_sprites.retain(|item| !removed(item.order));
    scene
        .compositor_groups
        .extend(groups.into_iter().map(|(_, group)| group));
    groups_moved || scene.compositor_sprites.len() != before
}

/// Whether the visible part of `painted` lies outside the mask's edge fades.
pub(super) fn unfaded(painted: &Bounds<ScaledPixels>, mask: &ContentMask<ScaledPixels>) -> bool {
    let visible = painted.intersect(&mask.bounds);
    let fade = &mask.fade_out;
    visible.left() >= mask.bounds.left() + fade.left
        && visible.top() >= mask.bounds.top() + fade.top
        && visible.right() <= mask.bounds.right() - fade.right
        && visible.bottom() <= mask.bounds.bottom() - fade.bottom
}

/// Whether a primitive other than `sprite` at or after `order` paints inside `painted`, or a
/// backdrop filter there samples from it. Primitives in one paint layer share an order, so an equal
/// order may still draw above.
pub(super) fn covered(
    scene: &Scene,
    sprite: *const (),
    (start_order, end_order): (DrawOrder, DrawOrder),
    painted: &Bounds<ScaledPixels>,
) -> bool {
    let above = |order| order >= start_order && !(start_order < order && order < end_order);
    let hit = |bounds: Bounds<ScaledPixels>| !bounds.is_empty() && bounds.intersects(painted);
    scene
        .shadows
        .iter()
        .any(|item| above(item.order) && hit(item.painted_bounds()))
        || scene
            .quads
            .iter()
            .any(|item| above(item.order) && hit(item.painted_bounds()))
        || scene
            .paths
            .iter()
            .any(|item| above(item.order) && hit(item.painted_bounds()))
        || scene
            .underlines
            .iter()
            .any(|item| above(item.order) && hit(item.painted_bounds()))
        || scene.monochrome_sprites.iter().any(|item| {
            !std::ptr::eq(std::ptr::from_ref(item).cast(), sprite)
                && above(item.order)
                && hit(item.painted_bounds())
        })
        || scene
            .subpixel_sprites
            .iter()
            .any(|item| above(item.order) && hit(item.painted_bounds()))
        || scene.polychrome_sprites.iter().any(|item| {
            !std::ptr::eq(std::ptr::from_ref(item).cast(), sprite)
                && above(item.order)
                && hit(item.painted_bounds())
        })
        || scene
            .surfaces
            .iter()
            .any(|item| above(item.order) && hit(item.painted_bounds()))
        || scene
            .backdrop_filters
            .iter()
            .any(|item| above(item.order) && hit(item.sampling_region()))
}

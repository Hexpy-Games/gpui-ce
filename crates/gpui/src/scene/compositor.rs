//! Animated sprites the platform compositor draws instead of the GPU frame.

use super::{DrawOrder, FilterRenderTarget, FrameLoop, PaintedBounds, RenderCommand, Scene};
use crate::{AtlasTile, Bounds, ContentMask, Corners, Hsla, ScaledPixels, TransformationMatrix};
use smallvec::SmallVec;

/// An animated sprite the platform compositor draws instead of the GPU frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompositorSprite {
    /// The flipbook grid in the atlas.
    pub tile: AtlasTile,
    /// Where the sprite is drawn, in device pixels.
    pub bounds: Bounds<ScaledPixels>,
    /// The visible area, in device pixels (the content mask's bounds).
    pub clip: Bounds<ScaledPixels>,
    /// `Some` for a monochrome sprite (an alpha mask tinted with the colour); `None` for a
    /// polychrome one.
    pub color: Option<Hsla>,
    /// The flipbook and turn clocks.
    pub frames: FrameLoop,
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
            && !covered(scene, sprite, order, &painted)
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

    let before = scene.compositor_sprites.len();
    let mut flags = monochrome.into_iter();
    scene.monochrome_sprites.retain(|sprite| {
        if !flags.next().unwrap_or(false) {
            return true;
        }
        scene.compositor_sprites.push(CompositorSprite {
            tile: sprite.tile,
            bounds: sprite.bounds,
            clip: sprite.content_mask.bounds,
            color: Some(sprite.color.into()),
            frames: sprite.frames,
        });
        false
    });
    let mut flags = polychrome.into_iter();
    scene.polychrome_sprites.retain(|sprite| {
        if !flags.next().unwrap_or(false) {
            return true;
        }
        scene.compositor_sprites.push(CompositorSprite {
            tile: sprite.tile,
            bounds: sprite.bounds,
            clip: sprite.content_mask.bounds,
            color: None,
            frames: sprite.frames,
        });
        false
    });
    scene.compositor_sprites.len() != before
}

/// Whether the visible part of `painted` lies outside the mask's edge fades.
fn unfaded(painted: &Bounds<ScaledPixels>, mask: &ContentMask<ScaledPixels>) -> bool {
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
fn covered(
    scene: &Scene,
    sprite: *const (),
    order: DrawOrder,
    painted: &Bounds<ScaledPixels>,
) -> bool {
    let hit = |bounds: Bounds<ScaledPixels>| !bounds.is_empty() && bounds.intersects(painted);
    scene
        .shadows
        .iter()
        .any(|item| item.order >= order && hit(item.painted_bounds()))
        || scene
            .quads
            .iter()
            .any(|item| item.order >= order && hit(item.painted_bounds()))
        || scene
            .paths
            .iter()
            .any(|item| item.order >= order && hit(item.painted_bounds()))
        || scene
            .underlines
            .iter()
            .any(|item| item.order >= order && hit(item.painted_bounds()))
        || scene.monochrome_sprites.iter().any(|item| {
            !std::ptr::eq(std::ptr::from_ref(item).cast(), sprite)
                && item.order >= order
                && hit(item.painted_bounds())
        })
        || scene
            .subpixel_sprites
            .iter()
            .any(|item| item.order >= order && hit(item.painted_bounds()))
        || scene.polychrome_sprites.iter().any(|item| {
            !std::ptr::eq(std::ptr::from_ref(item).cast(), sprite)
                && item.order >= order
                && hit(item.painted_bounds())
        })
        || scene
            .surfaces
            .iter()
            .any(|item| item.order >= order && hit(item.painted_bounds()))
        || scene
            .backdrop_filters
            .iter()
            .any(|item| item.order >= order && hit(item.sampling_region()))
}

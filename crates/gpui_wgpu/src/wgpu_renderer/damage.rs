// Added by HexpyGames for Butler (2026), under the Apache License 2.0; see NOTICE.
//! Damage redraw: an animation tick that re-encodes only the work intersecting the animated
//! primitives, over the complete frame the previous full encode left in `scene_color`.
use gpui::{Bounds, DamagePlan, DamageRedraw, PrimitiveBatch, RenderCommand, ScaledPixels, Scene};
use gpui_render::blur::ScissorRectangle;

use super::WgpuRenderer;
use super::frame::{FrameEncoder, PreparedTargets, ReadbackCopy, submit, uses_blurred_backdrop};

/// Encodes the scene's damage plan over the retained frame into `frame_view`; `None` when this
/// tick needs a full encode (no retained frame for this scene, a `Full` plan, or a missing cache entry).
pub(super) fn render_damage_to_view(
    renderer: &mut WgpuRenderer,
    scene: &Scene,
    frame_view: &wgpu::TextureView,
    readback: Option<ReadbackCopy<'_>>,
) -> Option<wgpu::SubmissionIndex> {
    let DamagePlan::Partial(plan) = scene.damage_plan() else {
        return None;
    };
    let generation = scene.generation();
    if renderer.retained != Some(generation)
        || renderer.frame_keys.0 != generation
        || renderer.resources.as_ref()?.scene_color_view.is_none()
        || !backdrops_cached(renderer, scene, plan)
    {
        return None;
    }
    let scissor = damage_scissor(renderer, plan.rect)?;

    // The fill needs a ping view, and its uniform slot is not part of the plan's accounting.
    let mut planned = plan.requirements;
    planned.uses_offscreen_target = true;
    planned.uses_blur_target = true;
    let targets = PreparedTargets::prepare(
        renderer,
        &plan.commands,
        planned,
        1,
        frame_view,
        renderer.target.viewport_size(),
    )?;
    let encoder = FrameEncoder::new(
        renderer,
        scene,
        &plan.commands,
        Some(plan),
        Some(scissor),
        None,
        targets,
    );
    submit(renderer, encoder, readback)
}

/// Whether every kept backdrop filter the plan does not re-blur can restore its blur from the cache.
fn backdrops_cached(renderer: &WgpuRenderer, scene: &Scene, plan: &DamageRedraw) -> bool {
    let Some(resources) = renderer.resources.as_ref() else {
        return false;
    };
    let cache = resources.backdrop_cache.borrow();
    plan.commands.iter().all(|command| {
        let RenderCommand::Batch(PrimitiveBatch::BackdropFilters(range)) = command else {
            return true;
        };
        range.clone().all(|index| {
            let filter = &scene.backdrop_filters[index];
            filter.wallpaper.is_some()
                || plan.reblur.contains(&index)
                || !uses_blurred_backdrop(filter)
                || renderer
                    .frame_keys
                    .1
                    .get(index)
                    .copied()
                    .flatten()
                    .is_some_and(|key| cache.contains(key))
        })
    })
}

/// The damage rectangle clamped to the target; `None` when nothing of it lies on the target.
fn damage_scissor(renderer: &WgpuRenderer, rect: Bounds<ScaledPixels>) -> Option<ScissorRectangle> {
    let clamp = |value: f32, limit: u32| value.clamp(0.0, limit as f32);
    let x0 = clamp(rect.origin.x.0.floor(), renderer.target.width());
    let y0 = clamp(rect.origin.y.0.floor(), renderer.target.height());
    let x1 = clamp(
        (rect.origin.x.0 + rect.size.width.0).ceil(),
        renderer.target.width(),
    );
    let y1 = clamp(
        (rect.origin.y.0 + rect.size.height.0).ceil(),
        renderer.target.height(),
    );
    let scissor = ScissorRectangle {
        x: x0 as u32,
        y: y0 as u32,
        width: (x1 - x0).max(0.0) as u32,
        height: (y1 - y0).max(0.0) as u32,
    };
    (!scissor.is_empty()).then_some(scissor)
}

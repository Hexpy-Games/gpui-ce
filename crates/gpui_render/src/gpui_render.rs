//! Shared typed rendering and shader contracts.

pub mod artifacts;
pub mod backdrop_key;
pub mod blur;
mod instances;
pub mod path_types;
pub mod shaders;
mod target_stack;

pub use instances::InstanceRange;
pub use target_stack::TargetStack;

/// Whether a damage scissor covers at most one quarter of the render target.
pub fn damage_fits(
    scissor: blur::ScissorRectangle,
    target: gpui::Size<gpui::DevicePixels>,
) -> bool {
    u64::from(scissor.width) * u64::from(scissor.height) * 4
        <= target.width.0 as u64 * target.height.0 as u64
}

/// Whether the damage area is small or the redraw avoids some backdrop blur work.
pub fn damage_worth_partial(
    scissor: blur::ScissorRectangle,
    viewport: (i32, i32),
    plan_skips_blur: bool,
) -> bool {
    let target = gpui::size(gpui::DevicePixels(viewport.0), gpui::DevicePixels(viewport.1));
    damage_fits(scissor, target) || plan_skips_blur
}

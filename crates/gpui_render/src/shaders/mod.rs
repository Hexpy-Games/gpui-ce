// Modified by HexpyGames for Butler (2026); see NOTICE for the changes.
//! Rust-authored shader sources, split by ABI and rendering domain.
//!
//! Ordinary and smoothed entry points share vertex preparation, but keep distinct stage-output
//! structs so ordinary primitives do not reserve or interpolate corner-smoothing data.

#![allow(dead_code, unused_assignments, unused_imports)]

pub mod common;
pub mod interface;

mod corner_smoothing;
mod emoji;
// The literals are emitted verbatim into WGSL; keep the generated shaders byte-identical.
#[allow(clippy::approx_constant)]
mod filters;
mod linkage;
mod paths;
mod quads;
mod shadows;
mod sprites;

pub use emoji::emoji_rasterization;
pub use filters::{blur, surface};
pub use linkage::base;
pub use paths::{path, path_rasterization};
pub use quads::quad;
pub use shadows::shadow;
pub use sprites::{monochrome_sprite, polychrome_sprite, subpixel_sprite, underline};

mod host;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod blur_validation_tests {
    /// The blur module recurses deeper in its IR builder and Naga's parser than a test
    /// thread's default stack allows in an unoptimized build; validate it on the same
    /// 64 MiB stack the build script uses.
    #[test]
    fn blur_wgsl_validates() {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                super::filters::blur::WGSL_SOURCE
                    .validate()
                    .expect("blur WGSL must validate")
            })
            .expect("spawn blur validation")
            .join()
            .expect("blur validation panicked");
    }
}

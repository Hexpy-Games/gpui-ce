// Modified by HexpyGames for Butler (2026); see NOTICE for the changes.
/// All standard pipelines share one WGPU shader module so drivers reuse parsed common
/// code and compile only each entry point's reachable resources.
#[wgsl_rs::wgsl(skip_validation)]
pub mod base {
    use super::super::blur::*;
    use super::super::monochrome_sprite::*;
    use super::super::path::*;
    use super::super::path_rasterization::*;
    use super::super::polychrome_sprite::*;
    use super::super::quad::*;
    use super::super::shadow::*;
    use super::super::surface::*;
    use super::super::underline::*;
    use super::super::wallpaper::*;
}

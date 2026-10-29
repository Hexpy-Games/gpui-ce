// Modified by HexpyGames for Butler (2026); see NOTICE for the changes.
pub use crate::shaders::{
    blur::BlurUniforms,
    common::{
        BlurCompositeClip as ShaderBlurCompositeClip, Bounds as ShaderBounds,
        Corners as ShaderCorners, DownsampleMode,
    },
};
use gpui::{BackdropLens, Bounds, Corners, ScaledPixels, WallpaperPass};
use wgsl_rs::std::vec2f;

pub const DOWNSAMPLE_FACTOR: u32 = 2;

/// Texture dimension for the downsampled blur passes; allocations must agree.
pub fn downsampled_dimension(dimension: u32) -> u32 {
    dimension.div_ceil(DOWNSAMPLE_FACTOR).max(1)
}
const RADIUS_TO_STANDARD_DEVIATION: f32 = 0.5;
pub const GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS: f32 = 3.0;
pub const MAX_GAUSSIAN_SAMPLES_PER_SIDE: u32 = 32;

#[derive(Clone, Copy)]
pub enum BlurAxis {
    Horizontal,
    Vertical,
}

#[derive(Clone, Copy)]
pub enum FilterCompositeClip {
    RoundedBounds,
    ContentShape,
}

#[derive(Clone, Copy)]
pub struct FilterCompositeParameters {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub corner_smoothing: f32,
    pub blur_radius: f32,
    pub opacity: f32,
    pub clip: FilterCompositeClip,
    pub lens: BackdropLens,
}

#[derive(Clone, Copy)]
pub struct BlurKernel {
    pub standard_deviation: f32,
    pub sample_count: u32,
    pub sample_step: f32,
}

impl BlurKernel {
    pub fn for_radius(blur_radius: f32) -> Option<Self> {
        let standard_deviation = (blur_radius * RADIUS_TO_STANDARD_DEVIATION).max(0.0);
        if standard_deviation <= 0.0 {
            return None;
        }

        let ideal_sample_count = (GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * standard_deviation).ceil();
        let sample_count = (ideal_sample_count as u32).clamp(1, MAX_GAUSSIAN_SAMPLES_PER_SIDE);
        Some(Self {
            standard_deviation,
            sample_count,
            sample_step: (ideal_sample_count / sample_count as f32).max(1.0),
        })
    }
}

impl BlurUniforms {
    pub fn copy(size: [f32; 2]) -> Self {
        with_texture_sizes(empty_uniforms(DownsampleMode::Copy), size, size)
    }

    pub fn downsample(source_size: [f32; 2], target_size: [f32; 2]) -> Self {
        with_texture_sizes(
            empty_uniforms(DownsampleMode::HalfResolution),
            source_size,
            target_size,
        )
    }

    pub fn gaussian(axis: BlurAxis, texture_size: [f32; 2], kernel: BlurKernel) -> Self {
        let direction = match axis {
            BlurAxis::Horizontal => vec2f(1.0 / texture_size[0], 0.0),
            BlurAxis::Vertical => vec2f(0.0, 1.0 / texture_size[1]),
        };
        Self {
            direction,
            standard_deviation: kernel.standard_deviation,
            sample_count: kernel.sample_count,
            sample_step: kernel.sample_step,
            ..with_texture_sizes(
                empty_uniforms(DownsampleMode::Copy),
                texture_size,
                texture_size,
            )
        }
    }

    /// Adds a liquid-glass lens to composite uniforms; a zero bezel leaves them unchanged.
    pub fn with_lens(mut self, lens: BackdropLens) -> Self {
        self.lens_bezel = lens.bezel;
        self.lens_refraction = lens.refraction;
        self.lens_specular = lens.specular;
        self.lens_dispersion = lens.dispersion;
        self.lens_frost = lens.frost;
        self.lens_lift = lens.lift;
        self.lens_rim = lens.rim;
        self.lens_depth = lens.depth;
        self.lens_light_x = lens.light_angle.cos();
        self.lens_light_y = -lens.light_angle.sin();
        self.lens_opposite = lens.opposite;
        self.lens_shade = lens.shade;
        self.lens_fresnel = lens.fresnel;
        self.lens_focus = lens.focus;
        self.lens_ambient = lens.ambient;
        self.lens_halo = lens.halo;
        self.lens_fold = lens.fold;
        self.lens_fold_band = lens.fold_band;
        self.lens_fold_depth = lens.fold_depth;
        self.lens_curve = lens.curve;
        // padding2 flags (no layout change): bit 0 tone curve, bit 1 full-shape sharp pass. With
        // the tone on, its coefficients ride in lens_refraction, lens_depth and lens_shade, which
        // only the v5 bevel (fold off) and the inner shade read; the tone forces both off.
        let tone_on = lens.tone.iter().any(|c| *c != 0.0);
        self.padding2 = u32::from(tone_on) | (u32::from(lens.sharp_full) << 1);
        if tone_on {
            self.lens_refraction = lens.tone[0];
            self.lens_depth = lens.tone[1];
            self.lens_shade = lens.tone[2];
            // The rim thinning and Fresnel are off with the tone on; their slots carry the
            // hairline (see lens_hairline in the shader).
            self.lens_curve = lens.hairline[0];
            self.lens_fresnel = lens.hairline[1];
        }
        self
    }

    /// The sharp-rim pass of a lens (`BackdropLens::sharp_rim`): composites only the fold band,
    /// from the sharp snapshot, over the blurred glass. padding1 carries its weight in percent.
    pub fn with_sharp_rim_pass(mut self, sharp_rim: f32) -> Self {
        self.padding1 = (sharp_rim.clamp(0.0, 1.0) * 100.0).round() as u32;
        self
    }

    /// Uniforms of one wallpaper pass: its own `wp_*` values (filled by the app).
    pub fn wallpaper_pass(pass: &WallpaperPass) -> Self {
        let v = |i: usize| {
            wgsl_rs::std::vec4f(
                pass.uniforms[i][0],
                pass.uniforms[i][1],
                pass.uniforms[i][2],
                pass.uniforms[i][3],
            )
        };
        let mut u = with_texture_sizes(
            empty_uniforms(DownsampleMode::Copy),
            [pass.width as f32, pass.height as f32],
            [pass.width as f32, pass.height as f32],
        );
        u.wp_mode = v(0);
        u.wp_frame = v(1);
        u.wp_env = v(2);
        u.wp_rect = v(3);
        u.wp_css = v(4);
        u.wp_print = v(5);
        u.wp_drop = v(6);
        u.wp_col = v(7);
        u.wp_rect2 = v(8);
        u.wp_p0 = v(9);
        u.wp_p1 = v(10);
        u.wp_p2 = v(11);
        u.wp_p3 = v(12);
        u.wp_p4 = v(13);
        u.wp_p5 = v(14);
        u.wp_mode.x = 1.0;
        u.wp_mode.y = pass.module as f32;
        u
    }

    /// Turns composite uniforms into a wallpaper composite of a `size` texture.
    pub fn with_wallpaper_texture(mut self, size: [f32; 2]) -> Self {
        self.wp_mode = wgsl_rs::std::vec4f(2.0, 0.0, 0.0, 0.0);
        self.wp_frame = wgsl_rs::std::vec4f(size[0], size[1], 0.0, 0.0);
        self
    }

    pub fn composite(
        bounds: Bounds<ScaledPixels>,
        content_mask: Bounds<ScaledPixels>,
        corner_radii: Corners<ScaledPixels>,
        corner_smoothing: f32,
        opacity: f32,
        clip: FilterCompositeClip,
        source_size: [f32; 2],
        target_size: [f32; 2],
    ) -> Self {
        Self {
            bounds: bounds.into(),
            content_mask: content_mask.into(),
            corner_radii: corner_radii.into(),
            corner_smoothing,
            opacity,
            composite_clip: match clip {
                FilterCompositeClip::RoundedBounds => ShaderBlurCompositeClip::RoundedBounds,
                FilterCompositeClip::ContentShape => ShaderBlurCompositeClip::None,
            },
            ..with_texture_sizes(
                empty_uniforms(DownsampleMode::Copy),
                source_size,
                target_size,
            )
        }
    }
}

#[derive(Clone, Copy)]
pub struct ScissorRectangle {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// The full-resolution part of the backdrop a backdrop filter at `bounds` can read, so its
/// snapshot copies only that instead of the whole target: the bounds grown by the blur's reach
/// (rounded out to the half-resolution grid the blur runs on), the frost disc, the v5 bevel's
/// shift cap and how far a fold (with its colour split) can shift a sample past the far side.
pub fn backdrop_snapshot_region(
    bounds: Bounds<ScaledPixels>,
    blur_radius: f32,
    lens: &BackdropLens,
    full_width: u32,
    full_height: u32,
) -> Bounds<ScaledPixels> {
    let short = bounds.size.width.0.min(bounds.size.height.0) * 0.5;
    let fold_reach = if lens.is_active() {
        (lens.fold_depth.max(0.0) * (1.0 + lens.dispersion.abs()) - 1.0).max(0.0) * short
    } else {
        0.0
    };
    let margin = GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * blur_radius
        + lens.sampling_margin()
        + lens.depth.max(0.0)
        + fold_reach
        + 4.0;
    let half = ScissorRectangle::for_blurred_bounds(bounds, margin, full_width, full_height);
    let x0 = (half.x * DOWNSAMPLE_FACTOR).min(full_width);
    let y0 = (half.y * DOWNSAMPLE_FACTOR).min(full_height);
    let x1 = ((half.x + half.width) * DOWNSAMPLE_FACTOR).min(full_width);
    let y1 = ((half.y + half.height) * DOWNSAMPLE_FACTOR).min(full_height);
    Bounds::new(
        gpui::point(ScaledPixels(x0 as f32), ScaledPixels(y0 as f32)),
        gpui::size(
            ScaledPixels((x1 - x0) as f32),
            ScaledPixels((y1 - y0) as f32),
        ),
    )
}

impl ScissorRectangle {
    pub fn for_blurred_bounds(
        bounds: Bounds<ScaledPixels>,
        dilation: f32,
        full_width: u32,
        full_height: u32,
    ) -> Self {
        let downsampled_width = full_width.div_ceil(DOWNSAMPLE_FACTOR).max(1);
        let downsampled_height = full_height.div_ceil(DOWNSAMPLE_FACTOR).max(1);
        let minimum_x = downsampled_coordinate(
            bounds.origin.x.0 - dilation,
            downsampled_width,
            EdgeRounding::OutwardMinimum,
        );
        let minimum_y = downsampled_coordinate(
            bounds.origin.y.0 - dilation,
            downsampled_height,
            EdgeRounding::OutwardMinimum,
        );
        let maximum_x = downsampled_coordinate(
            bounds.origin.x.0 + bounds.size.width.0 + dilation,
            downsampled_width,
            EdgeRounding::OutwardMaximum,
        )
        .max(minimum_x);
        let maximum_y = downsampled_coordinate(
            bounds.origin.y.0 + bounds.size.height.0 + dilation,
            downsampled_height,
            EdgeRounding::OutwardMaximum,
        )
        .max(minimum_y);
        Self {
            x: minimum_x,
            y: minimum_y,
            width: maximum_x - minimum_x,
            height: maximum_y - minimum_y,
        }
    }

    pub fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

#[derive(Clone, Copy)]
enum EdgeRounding {
    OutwardMinimum,
    OutwardMaximum,
}

fn downsampled_coordinate(value: f32, maximum: u32, rounding: EdgeRounding) -> u32 {
    let downsampled = value / DOWNSAMPLE_FACTOR as f32;
    let rounded = match rounding {
        EdgeRounding::OutwardMinimum => downsampled.floor(),
        EdgeRounding::OutwardMaximum => downsampled.ceil(),
    };
    (rounded.max(0.0) as u32).min(maximum)
}

fn with_texture_sizes(
    mut uniforms: BlurUniforms,
    source_size: [f32; 2],
    target_size: [f32; 2],
) -> BlurUniforms {
    uniforms.source_size = vec2f(source_size[0], source_size[1]);
    uniforms.target_size = vec2f(target_size[0], target_size[1]);
    uniforms
}

fn empty_uniforms(downsample_mode: DownsampleMode) -> BlurUniforms {
    let bounds = ShaderBounds {
        origin: vec2f(0.0, 0.0),
        size: vec2f(0.0, 0.0),
    };
    BlurUniforms {
        bounds,
        content_mask: bounds,
        corner_radii: ShaderCorners {
            top_left: 0.0,
            top_right: 0.0,
            bottom_right: 0.0,
            bottom_left: 0.0,
        },
        direction: vec2f(0.0, 0.0),
        standard_deviation: 0.0,
        opacity: 0.0,
        sample_count: 0,
        sample_step: 0.0,
        composite_clip: ShaderBlurCompositeClip::None,
        downsample_mode,
        source_size: vec2f(1.0, 1.0),
        target_size: vec2f(1.0, 1.0),
        corner_smoothing: 0.0,
        lens_bezel: 0.0,
        lens_refraction: 0.0,
        lens_specular: 0.0,
        lens_dispersion: 0.0,
        lens_frost: 0.0,
        lens_lift: 0.0,
        lens_rim: 0.0,
        lens_depth: 0.0,
        lens_light_x: 0.0,
        lens_light_y: 0.0,
        lens_opposite: 0.0,
        lens_shade: 0.0,
        lens_fresnel: 0.0,
        lens_focus: 0.0,
        lens_ambient: 0.0,
        lens_halo: 0.0,
        lens_fold: 0.0,
        lens_fold_band: 0.0,
        lens_fold_depth: 0.0,
        lens_curve: 0.0,
        padding0: 0,
        padding1: 0,
        padding2: 0,
        wp_mode: wgsl_rs::std::vec4f(0.0, 0.0, 0.0, 0.0),
        wp_frame: wgsl_rs::std::vec4f(0.0, 0.0, 0.0, 0.0),
        wp_env: wgsl_rs::std::vec4f(0.0, 0.0, 0.0, 0.0),
        wp_rect: wgsl_rs::std::vec4f(0.0, 0.0, 0.0, 0.0),
        wp_css: wgsl_rs::std::vec4f(0.0, 0.0, 0.0, 0.0),
        wp_print: wgsl_rs::std::vec4f(0.0, 0.0, 0.0, 0.0),
        wp_drop: wgsl_rs::std::vec4f(0.0, 0.0, 0.0, 0.0),
        wp_col: wgsl_rs::std::vec4f(0.0, 0.0, 0.0, 0.0),
        wp_rect2: wgsl_rs::std::vec4f(0.0, 0.0, 0.0, 0.0),
        wp_p0: wgsl_rs::std::vec4f(0.0, 0.0, 0.0, 0.0),
        wp_p1: wgsl_rs::std::vec4f(0.0, 0.0, 0.0, 0.0),
        wp_p2: wgsl_rs::std::vec4f(0.0, 0.0, 0.0, 0.0),
        wp_p3: wgsl_rs::std::vec4f(0.0, 0.0, 0.0, 0.0),
        wp_p4: wgsl_rs::std::vec4f(0.0, 0.0, 0.0, 0.0),
        wp_p5: wgsl_rs::std::vec4f(0.0, 0.0, 0.0, 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::downsampled_dimension;

    #[test]
    fn downsampled_dimensions_cover_odd_edges() {
        assert_eq!(downsampled_dimension(0), 1);
        assert_eq!(downsampled_dimension(1), 1);
        assert_eq!(downsampled_dimension(2), 1);
        assert_eq!(downsampled_dimension(3), 2);
        assert_eq!(downsampled_dimension(33), 17);
    }
}

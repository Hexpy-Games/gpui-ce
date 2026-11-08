//! Device-pixel areas that scene primitives read or change.

use crate::{
    BackdropFilter, BackdropLens, Bounds, MonochromeSprite, PaintSurface, Path, PolychromeSprite,
    Quad, ScaledPixels, Shadow, SubpixelSprite, TransformationMatrix, Underline, point, px,
};

/// Factor by which backdrop blurs downsample the target; region edges round out to it.
pub const BLUR_DOWNSAMPLE_FACTOR: u32 = 2;

/// The renderers cut a gaussian off at this many standard deviations.
const GAUSSIAN_CUTOFF: f32 = 3.0;

/// The device-pixel area a backdrop filter at `bounds` reads: `bounds` grown by the blur reach,
/// the lens frost disc, bevel depth and fold shift, plus 4 px, then rounded out to multiples of
/// `BLUR_DOWNSAMPLE_FACTOR`. It is not clamped to the target.
pub fn backdrop_sampling_region(
    bounds: Bounds<ScaledPixels>,
    blur_radius: f32,
    lens: &BackdropLens,
) -> Bounds<ScaledPixels> {
    let short = bounds.size.width.0.min(bounds.size.height.0) * 0.5;
    let fold_reach = if lens.is_active() {
        (lens.fold_depth.max(0.0) * (1.0 + lens.dispersion.abs()) - 1.0).max(0.0) * short
    } else {
        0.0
    };
    let margin = GAUSSIAN_CUTOFF * blur_radius
        + lens.sampling_margin()
        + lens.depth.max(0.0)
        + fold_reach
        + 4.0;
    let factor = BLUR_DOWNSAMPLE_FACTOR as f32;
    let floor = |value: f32| (value / factor).floor() * factor;
    let ceil = |value: f32| (value / factor).ceil() * factor;
    let x0 = floor(bounds.origin.x.0 - margin);
    let y0 = floor(bounds.origin.y.0 - margin);
    let x1 = ceil(bounds.origin.x.0 + bounds.size.width.0 + margin);
    let y1 = ceil(bounds.origin.y.0 + bounds.size.height.0 + margin);
    Bounds::from_corners(
        point(ScaledPixels(x0), ScaledPixels(y0)),
        point(ScaledPixels(x1), ScaledPixels(y1)),
    )
}

impl BackdropFilter {
    /// [`backdrop_sampling_region`] for this filter.
    pub fn sampling_region(&self) -> Bounds<ScaledPixels> {
        backdrop_sampling_region(self.bounds, self.max_blur_radius(), &self.lens)
    }
}

/// The device-pixel area a scene primitive can change, clipped to its content mask.
pub trait PaintedBounds {
    /// See the trait.
    fn painted_bounds(&self) -> Bounds<ScaledPixels>;
}

fn transformed(
    bounds: &Bounds<ScaledPixels>,
    matrix: &TransformationMatrix,
) -> Bounds<ScaledPixels> {
    let corners = [
        bounds.origin,
        bounds.top_right(),
        bounds.bottom_left(),
        bounds.bottom_right(),
    ]
    .map(|corner| matrix.apply(point(px(corner.x.0), px(corner.y.0))));
    let min_x = corners.iter().map(|c| c.x.0).fold(f32::INFINITY, f32::min);
    let min_y = corners.iter().map(|c| c.y.0).fold(f32::INFINITY, f32::min);
    let max_x = corners
        .iter()
        .map(|c| c.x.0)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_y = corners
        .iter()
        .map(|c| c.y.0)
        .fold(f32::NEG_INFINITY, f32::max);
    Bounds::from_corners(
        point(ScaledPixels(min_x), ScaledPixels(min_y)),
        point(ScaledPixels(max_x), ScaledPixels(max_y)),
    )
}

macro_rules! painted_bounds_is_bounds {
    ($($type:ty),* $(,)?) => {
        $(
            impl PaintedBounds for $type {
                fn painted_bounds(&self) -> Bounds<ScaledPixels> {
                    self.bounds.intersect(&self.content_mask.bounds)
                }
            }
        )*
    };
}

painted_bounds_is_bounds!(
    Quad,
    Path<ScaledPixels>,
    Underline,
    PaintSurface,
    BackdropFilter,
);

impl PaintedBounds for Shadow {
    fn painted_bounds(&self) -> Bounds<ScaledPixels> {
        self.bounds
            .dilate(ScaledPixels(GAUSSIAN_CUTOFF * self.blur_radius.0))
            .intersect(&self.content_mask.bounds)
    }
}

/// The square around the centre of `bounds` that a sprite turning about that centre stays within:
/// its half side is half the bounds' diagonal.
fn turn_square(bounds: &Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
    let half = 0.5 * bounds.size.width.0.hypot(bounds.size.height.0);
    let center = bounds.center();
    Bounds::from_corners(
        point(
            ScaledPixels(center.x.0 - half),
            ScaledPixels(center.y.0 - half),
        ),
        point(
            ScaledPixels(center.x.0 + half),
            ScaledPixels(center.y.0 + half),
        ),
    )
}

impl PaintedBounds for MonochromeSprite {
    fn painted_bounds(&self) -> Bounds<ScaledPixels> {
        let reach = if self.frames.turn_period_ms != 0 {
            turn_square(&self.bounds)
        } else {
            self.bounds
        };
        transformed(&reach, &self.transformation).intersect(&self.content_mask.bounds)
    }
}

impl PaintedBounds for PolychromeSprite {
    fn painted_bounds(&self) -> Bounds<ScaledPixels> {
        let reach = if self.frames.turn_period_ms != 0 {
            turn_square(&self.bounds)
        } else {
            self.bounds
        };
        reach.intersect(&self.content_mask.bounds)
    }
}

impl PaintedBounds for SubpixelSprite {
    fn painted_bounds(&self) -> Bounds<ScaledPixels> {
        transformed(&self.bounds, &self.transformation).intersect(&self.content_mask.bounds)
    }
}

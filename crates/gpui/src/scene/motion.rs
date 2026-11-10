//! Time-driven transforms of retained painted subtrees.

use crate::{
    Bounds, ContentMask, Corners, Edges, FilterBoundary, Hsla, PaintQuad, Pixels, Point, Rgba,
    ScaledPixels, Size, hsla_to_rgba, point,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// How motion continues after its first duration.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum MotionRepeat {
    /// Stop at the end state.
    Once,
    /// Restart from the beginning.
    Loop,
    /// Reverse direction on each iteration.
    Alternate,
}

/// Most points a [`MotionEasing::Linear`] curve holds.
pub const MAX_LINEAR_POINTS: usize = 24;

/// The timing curve applied to linear progress.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum MotionEasing {
    /// CSS cubic-bezier(x1, y1, x2, y2).
    CubicBezier([f32; 4]),
    /// CSS linear() with up to [`MAX_LINEAR_POINTS`] (time, value) points; the design tokens use up to 21.
    Linear {
        /// The (time, value) points in ascending time order.
        points: [(f32, f32); MAX_LINEAR_POINTS],
        /// Number of populated points.
        len: u8,
    },
    /// CSS steps(n, jump-end).
    Steps(u32),
}

impl MotionEasing {
    /// Creates a CSS linear() timing curve with at most [`MAX_LINEAR_POINTS`] points.
    pub fn linear(points: &[(f32, f32)]) -> Self {
        debug_assert!(points.len() <= MAX_LINEAR_POINTS);
        let len = points.len().min(MAX_LINEAR_POINTS);
        let mut stored = [(0.0, 0.0); MAX_LINEAR_POINTS];
        stored[..len].copy_from_slice(&points[..len]);
        Self::Linear {
            points: stored,
            len: len as u8,
        }
    }

    /// Conservative range of eased progress, including overshoot.
    pub(crate) fn range(&self) -> (f32, f32) {
        match self {
            MotionEasing::CubicBezier([_, y1, _, y2]) => {
                (0.0f32.min(*y1).min(*y2), 1.0f32.max(*y1).max(*y2))
            }
            MotionEasing::Steps(_) => (0.0, 1.0),
            MotionEasing::Linear { points, len } => points[..usize::from(*len)]
                .iter()
                .fold((0.0f32, 1.0f32), |(low, high), &(_, y)| {
                    (low.min(y), high.max(y))
                }),
        }
    }
}

/// A layout change animated without running layout on each frame.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub struct LayoutMotion {
    /// Duration of the layout motion in milliseconds.
    pub duration_ms: u32,
    /// Timing curve of the layout motion.
    pub easing: MotionEasing,
}

/// A retained subtree's opacity, translation, scale, and rotation over time.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[expect(missing_docs)]
pub struct LayerMotion {
    pub start_ms: u32,
    pub duration_ms: u32,
    pub repeat: MotionRepeat,
    pub easing: MotionEasing,
    pub opacity: (f32, f32),
    pub offset: (Point<Pixels>, Point<Pixels>),
    pub relative_offset: (Point<f32>, Point<f32>),
    pub clip: Option<(Edges<Pixels>, Edges<Pixels>)>,
    pub clip_radii: Option<(Corners<Pixels>, Corners<Pixels>)>,
    pub visible: (f32, f32),
    pub scale: (f32, f32),
    /// Clockwise rotation in screen-space radians about `origin`.
    pub rotation: (f32, f32),
    pub origin: Point<f32>,
}

/// Composite parameters sampled in device pixels.
#[derive(Clone, Copy, PartialEq, Debug)]
#[expect(missing_docs)]
pub struct LayerMotionSample {
    pub opacity: f32,
    pub offset: Point<ScaledPixels>,
    pub scale: f32,
    pub rotation: f32,
    pub clip: Option<(Bounds<ScaledPixels>, Corners<ScaledPixels>)>,
}

impl LayerMotion {
    /// No motion, with identity composite parameters.
    pub const NONE: Self = Self {
        start_ms: 0,
        duration_ms: 0,
        repeat: MotionRepeat::Once,
        easing: MotionEasing::CubicBezier([0.0, 0.0, 1.0, 1.0]),
        opacity: (1.0, 1.0),
        offset: (
            point(Pixels::ZERO, Pixels::ZERO),
            point(Pixels::ZERO, Pixels::ZERO),
        ),
        relative_offset: (point(0.0, 0.0), point(0.0, 0.0)),
        clip: None,
        clip_radii: None,
        visible: (0.0, 1.0),
        scale: (1.0, 1.0),
        rotation: (0.0, 0.0),
        origin: point(0.5, 0.5),
    };

    /// Whether this declaration animates.
    pub fn is_animated(&self) -> bool {
        self.duration_ms > 0
    }

    /// Whether the motion ends with identity composite parameters and remains visible.
    pub fn rests_at_end(&self) -> bool {
        self.opacity.1 == 1.0
            && self.offset.1 == point(Pixels::ZERO, Pixels::ZERO)
            && self.relative_offset.1 == point(0.0, 0.0)
            && self.scale.1 == 1.0
            && self.rotation.1 == 0.0
            && self.clip.is_none()
            && self.clip_radii.is_none()
            && self.visible.0 <= 1.0
            && self.visible.1 >= 1.0
    }

    fn linear_progress(&self, now_ms: u32) -> f32 {
        if !self.is_animated() {
            return 0.0;
        }
        let elapsed = now_ms.wrapping_sub(self.start_ms);
        let x = if elapsed >= 1 << 31 {
            0.0
        } else {
            elapsed as f32 / self.duration_ms as f32
        };
        match self.repeat {
            MotionRepeat::Once => x.clamp(0.0, 1.0),
            MotionRepeat::Loop => x.fract(),
            MotionRepeat::Alternate => 1.0 - (x % 2.0 - 1.0).abs(),
        }
    }

    fn visible_at(&self, now_ms: u32) -> bool {
        let x = self.linear_progress(now_ms);
        self.visible.0 <= x
            && (x < self.visible.1 || (x <= self.visible.1 && self.visible.1 >= 1.0))
    }

    /// Eased progress at the wrapping animation clock value.
    pub fn progress(&self, now_ms: u32) -> f32 {
        let x = self.linear_progress(now_ms);
        match self.easing {
            MotionEasing::Linear { points, len } => {
                let points = &points[..usize::from(len)];
                let segment = points.windows(2).find(|pair| x <= pair[1].0).unwrap();
                let [(x0, y0), (x1, y1)] = [segment[0], segment[1]];
                y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            }
            MotionEasing::Steps(n) => (x * n as f32).floor() / n as f32,
            MotionEasing::CubicBezier([x1, y1, x2, y2]) => {
                let mut t = x;
                for _ in 0..8 {
                    let inverse = 1.0 - t;
                    let position =
                        3.0 * inverse * inverse * t * x1 + 3.0 * inverse * t * t * x2 + t * t * t;
                    let derivative = 3.0 * inverse * inverse * x1
                        + 6.0 * inverse * t * (x2 - x1)
                        + 3.0 * t * t * (1.0 - x2);
                    if derivative.abs() < 1e-6 {
                        let mut low = 0.0;
                        let mut high = 1.0;
                        for _ in 0..20 {
                            t = (low + high) * 0.5;
                            let inverse = 1.0 - t;
                            let position = 3.0 * inverse * inverse * t * x1
                                + 3.0 * inverse * t * t * x2
                                + t * t * t;
                            if position < x {
                                low = t;
                            } else {
                                high = t;
                            }
                        }
                        break;
                    }
                    t = (t - (position - x) / derivative).clamp(0.0, 1.0);
                }
                let inverse = 1.0 - t;
                3.0 * inverse * inverse * t * y1 + 3.0 * inverse * t * t * y2 + t * t * t
            }
        }
    }

    /// Whether another frame is needed to advance this motion.
    pub fn is_active(&self, now_ms: u32) -> bool {
        self.is_animated()
            && (self.repeat != MotionRepeat::Once
                || now_ms.wrapping_sub(self.start_ms) >= 1 << 31
                || now_ms.wrapping_sub(self.start_ms) < self.duration_ms)
    }

    /// Samples the composite transform without rebuilding the subtree.
    pub fn sample(
        &self,
        now_ms: u32,
        scale_factor: f32,
        motion_bounds: Bounds<ScaledPixels>,
    ) -> LayerMotionSample {
        let p = self.progress(now_ms);
        LayerMotionSample {
            opacity: if self.visible_at(now_ms) {
                self.opacity.0 + (self.opacity.1 - self.opacity.0) * p
            } else {
                0.0
            },
            offset: (self.offset.0 + (self.offset.1 - self.offset.0) * p).scale(scale_factor)
                + point(
                    motion_bounds.size.width
                        * (self.relative_offset.0.x
                            + (self.relative_offset.1.x - self.relative_offset.0.x) * p),
                    motion_bounds.size.height
                        * (self.relative_offset.0.y
                            + (self.relative_offset.1.y - self.relative_offset.0.y) * p),
                ),
            scale: self.scale.0 + (self.scale.1 - self.scale.0) * p,
            rotation: self.rotation.0 + (self.rotation.1 - self.rotation.0) * p,
            clip: self
                .clip
                .or(self
                    .clip_radii
                    .map(|_| (Edges::default(), Edges::default())))
                .map(|(from, to)| {
                    let edges = Edges {
                        top: from.top + (to.top - from.top) * p,
                        right: from.right + (to.right - from.right) * p,
                        bottom: from.bottom + (to.bottom - from.bottom) * p,
                        left: from.left + (to.left - from.left) * p,
                    }
                    .scale(scale_factor);
                    let rect = Bounds::from_corners(
                        motion_bounds.origin + point(edges.left, edges.top),
                        motion_bounds.bottom_right() - point(edges.right, edges.bottom),
                    );
                    let (from, to) = self.clip_radii.unwrap_or_default();
                    let radii = Corners {
                        top_left: from.top_left + (to.top_left - from.top_left) * p,
                        top_right: from.top_right + (to.top_right - from.top_right) * p,
                        bottom_right: from.bottom_right + (to.bottom_right - from.bottom_right) * p,
                        bottom_left: from.bottom_left + (to.bottom_left - from.bottom_left) * p,
                    }
                    .scale(scale_factor)
                    .clamp_radii_for_quad_size(rect.size);
                    (rect, radii)
                }),
        }
    }

    /// Whether the motion changes between two clock values.
    pub fn changes_between(&self, a_ms: u32, b_ms: u32) -> bool {
        if !self.is_animated() || a_ms == b_ms {
            return false;
        }
        if self.opacity.0 == self.opacity.1
            && self.offset.0 == self.offset.1
            && self.relative_offset.0 == self.relative_offset.1
            && self.scale.0 == self.scale.1
            && self.rotation.0 == self.rotation.1
            && self.clip.is_none_or(|(from, to)| from == to)
            && self.clip_radii.is_none_or(|(from, to)| from == to)
        {
            return self.visible_at(a_ms) != self.visible_at(b_ms);
        }
        match self.easing {
            MotionEasing::Steps(n) => {
                (self.progress(a_ms) * n as f32).round() != (self.progress(b_ms) * n as f32).round()
            }
            MotionEasing::CubicBezier(_) | MotionEasing::Linear { .. } => {
                self.is_active(a_ms) || self.is_active(b_ms)
            }
        }
    }

    /// Conservative bounds of every transformed position over this motion.
    pub fn sweep(
        &self,
        bounds: Bounds<ScaledPixels>,
        origin: Point<ScaledPixels>,
        scale_factor: f32,
        motion_bounds: Bounds<ScaledPixels>,
    ) -> Bounds<ScaledPixels> {
        let (low, high) = self.easing.range();
        let mut samples = vec![0.0, 1.0, low, high];
        if self.rotation.0 != self.rotation.1 {
            let delta = self.rotation.1 - self.rotation.0;
            let from = self.rotation.0 + delta * low;
            let to = self.rotation.0 + delta * high;
            let quarter_turn = std::f32::consts::FRAC_PI_2;
            let first = (from.min(to) / quarter_turn).ceil() as i32;
            let last = (from.max(to) / quarter_turn).floor() as i32;
            samples.extend(
                (first..=last).map(|turn| (turn as f32 * quarter_turn - self.rotation.0) / delta),
            );
        }
        samples
            .into_iter()
            .map(|p| {
                let offset = (self.offset.0 + (self.offset.1 - self.offset.0) * p)
                    .scale(scale_factor)
                    + point(
                        motion_bounds.size.width
                            * (self.relative_offset.0.x
                                + (self.relative_offset.1.x - self.relative_offset.0.x) * p),
                        motion_bounds.size.height
                            * (self.relative_offset.0.y
                                + (self.relative_offset.1.y - self.relative_offset.0.y) * p),
                    );
                let scale = self.scale.0 + (self.scale.1 - self.scale.0) * p;
                let rotation = self.rotation.0 + (self.rotation.1 - self.rotation.0) * p;
                let (sin, cos) = rotation.sin_cos();
                [
                    bounds.origin,
                    bounds.top_right(),
                    bounds.bottom_left(),
                    bounds.bottom_right(),
                ]
                .into_iter()
                .map(|corner| {
                    let corner = (corner - origin) * scale;
                    let position = origin
                        + point(
                            corner.x * cos - corner.y * sin,
                            corner.x * sin + corner.y * cos,
                        )
                        + offset;
                    Bounds::from_corners(position, position)
                })
                .reduce(|a, b| a.union(&b))
                .unwrap()
            })
            .reduce(|a, b| a.union(&b))
            .unwrap()
    }
}

/// Geometry deltas for a box painted at its final layout, sampled at presentation time.
#[derive(Clone, Debug)]
pub struct BoxMorph {
    /// Old minus final bounds at each edge.
    pub delta: Edges<ScaledPixels>,
    /// Old minus final corner radii.
    pub radii_delta: Corners<ScaledPixels>,
    /// Start, duration and easing of this once-only morph.
    pub timing: LayerMotion,
}

impl BoxMorph {
    pub(crate) fn sample(
        &self,
        bounds: Bounds<ScaledPixels>,
        radii: Corners<ScaledPixels>,
        now_ms: u32,
    ) -> (Bounds<ScaledPixels>, Corners<ScaledPixels>) {
        let remaining = 1.0 - self.timing.progress(now_ms);
        if remaining == 0.0 {
            return (bounds, radii);
        }
        let bounds = Bounds::from_corners(
            bounds.origin + point(self.delta.left, self.delta.top) * remaining,
            bounds.bottom_right() + point(self.delta.right, self.delta.bottom) * remaining,
        );
        let radii = Corners {
            top_left: (radii.top_left + self.radii_delta.top_left * remaining)
                .max(ScaledPixels(0.0)),
            top_right: (radii.top_right + self.radii_delta.top_right * remaining)
                .max(ScaledPixels(0.0)),
            bottom_right: (radii.bottom_right + self.radii_delta.bottom_right * remaining)
                .max(ScaledPixels(0.0)),
            bottom_left: (radii.bottom_left + self.radii_delta.bottom_left * remaining)
                .max(ScaledPixels(0.0)),
        }
        .clamp_radii_for_quad_size(bounds.size);
        (bounds, radii)
    }

    /// Whether eased progress differs between two presentation times.
    pub fn changes_between(&self, a_ms: u32, b_ms: u32) -> bool {
        self.timing.progress(a_ms) != self.timing.progress(b_ms)
    }

    /// Conservative painted reach over the easing curve, before applying the content mask.
    pub fn reach(&self, painted_bounds: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
        let (low, high) = self.timing.easing.range();
        [0.0, 1.0, low, high]
            .into_iter()
            .map(|p| {
                Bounds::from_corners(
                    painted_bounds.origin + point(self.delta.left, self.delta.top) * (1.0 - p),
                    painted_bounds.bottom_right()
                        + point(self.delta.right, self.delta.bottom) * (1.0 - p),
                )
            })
            .reduce(|a, b| a.union(&b))
            .unwrap()
    }
}

/// Resolves a filter group's static and time-driven composite parameters.
pub fn motion_composite_parameters(
    boundary: &FilterBoundary,
    now_ms: u32,
    scale_factor: f32,
) -> (
    f32,
    f32,
    Point<ScaledPixels>,
    f32,
    Option<(Bounds<ScaledPixels>, Corners<ScaledPixels>)>,
) {
    if boundary.motion.is_animated() {
        let sample = boundary
            .motion
            .sample(now_ms, scale_factor, boundary.motion_bounds);
        (
            boundary.opacity * sample.opacity,
            sample.scale,
            sample.offset,
            sample.rotation,
            sample.clip,
        )
    } else {
        (
            boundary.opacity,
            boundary.scale,
            Point::default(),
            0.0,
            None,
        )
    }
}

/// A canvas-local drawing operation in device pixels.
#[expect(missing_docs)]
pub enum LiveOp {
    Quad {
        bounds: Bounds<ScaledPixels>,
        radii: Corners<ScaledPixels>,
        color: Rgba,
    },
    Fill {
        contours: Vec<Vec<Point<ScaledPixels>>>,
        even_odd: bool,
        color: Rgba,
    },
    Group {
        alpha: f32,
        ops: Vec<LiveOp>,
    },
}

/// A canvas-local display list painted independently of the window's views.
pub struct LiveCanvas {
    /// Drawing operations in device pixels.
    pub ops: Vec<LiveOp>,
    /// Device pixels per logical pixel.
    pub scale_factor: f32,
    /// Logical canvas dimensions.
    pub size: Size<Pixels>,
}

impl LiveCanvas {
    /// The canvas dimensions in logical pixels.
    pub fn size(&self) -> Size<Pixels> {
        self.size
    }

    /// Paints a quad's background fill in canvas-local coordinates. Borders are not supported.
    pub fn paint_quad(&mut self, quad: PaintQuad) {
        self.ops.push(LiveOp::Quad {
            bounds: quad.bounds.scale(self.scale_factor),
            radii: quad.corner_radii.scale(self.scale_factor),
            color: hsla_to_rgba(quad.background.solid.into()),
        });
    }

    /// Fills closed contours in canvas-local coordinates.
    pub fn fill(
        &mut self,
        contours: Vec<Vec<Point<Pixels>>>,
        even_odd: bool,
        color: impl Into<Hsla>,
    ) {
        self.ops.push(LiveOp::Fill {
            contours: contours
                .into_iter()
                .map(|contour| {
                    contour
                        .into_iter()
                        .map(|point| point.scale(self.scale_factor))
                        .collect()
                })
                .collect(),
            even_odd,
            color: hsla_to_rgba(color.into()),
        });
    }

    /// Paints an isolated group composited at the given opacity.
    pub fn group(&mut self, alpha: f32, f: impl FnOnce(&mut LiveCanvas)) {
        let mut canvas = LiveCanvas {
            ops: Vec::new(),
            scale_factor: self.scale_factor,
            size: self.size,
        };
        f(&mut canvas);
        self.ops.push(LiveOp::Group {
            alpha,
            ops: canvas.ops,
        });
    }
}

/// Paints a live canvas at seconds since its layer's start time.
pub type LivePaint = Arc<dyn Fn(f32, &mut LiveCanvas) + Send + Sync>;

/// A procedural scene presented on its own platform surface.
#[derive(Clone)]
#[expect(missing_docs)]
pub struct LiveLayer {
    pub key: u64,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub order: super::DrawOrder,
    pub start_ms: u32,
    pub paint: LivePaint,
}

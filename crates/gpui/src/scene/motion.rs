//! Time-driven transforms of retained painted subtrees.

use crate::{
    Background, Bounds, ContentMask, Edges, FilterBoundary, PaintQuad, Path, Pixels, Point, Quad,
    ScaledPixels, Scene, Size, point,
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
}

/// A layout change animated without running layout on each frame.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub struct LayoutMotion {
    /// Duration of the layout motion in milliseconds.
    pub duration_ms: u32,
    /// Timing curve of the layout motion.
    pub easing: MotionEasing,
}

/// A retained subtree's opacity, translation, and scale over time.
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
    pub visible: (f32, f32),
    pub scale: (f32, f32),
    pub origin: Point<f32>,
}

/// Composite parameters sampled in device pixels.
#[derive(Clone, Copy, PartialEq, Debug)]
#[expect(missing_docs)]
pub struct LayerMotionSample {
    pub opacity: f32,
    pub offset: Point<ScaledPixels>,
    pub scale: f32,
    pub clip: Option<Bounds<ScaledPixels>>,
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
        visible: (0.0, 1.0),
        scale: (1.0, 1.0),
        origin: point(0.5, 0.5),
    };

    /// Whether this declaration animates.
    pub fn is_animated(&self) -> bool {
        self.duration_ms > 0
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
            clip: self.clip.map(|(from, to)| {
                let edges = Edges {
                    top: from.top + (to.top - from.top) * p,
                    right: from.right + (to.right - from.right) * p,
                    bottom: from.bottom + (to.bottom - from.bottom) * p,
                    left: from.left + (to.left - from.left) * p,
                }
                .scale(scale_factor);
                Bounds::from_corners(
                    motion_bounds.origin + point(edges.left, edges.top),
                    motion_bounds.bottom_right() - point(edges.right, edges.bottom),
                )
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
            && self.clip.is_none_or(|(from, to)| from == to)
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
        let (low, high) = match self.easing {
            MotionEasing::CubicBezier([_, y1, _, y2]) => {
                (0.0f32.min(y1).min(y2), 1.0f32.max(y1).max(y2))
            }
            MotionEasing::Steps(_) => (0.0, 1.0),
            MotionEasing::Linear { points, len } => points[..usize::from(len)]
                .iter()
                .fold((0.0f32, 1.0f32), |(low, high), &(_, y)| {
                    (low.min(y), high.max(y))
                }),
        };
        [0.0, 1.0, low, high]
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
                let a = origin + (bounds.origin - origin) * scale + offset;
                let b = origin + (bounds.bottom_right() - origin) * scale + offset;
                Bounds::from_corners(
                    point(a.x.min(b.x), a.y.min(b.y)),
                    point(a.x.max(b.x), a.y.max(b.y)),
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
) -> (f32, f32, Point<ScaledPixels>, Option<Bounds<ScaledPixels>>) {
    if boundary.motion.is_animated() {
        let sample = boundary
            .motion
            .sample(now_ms, scale_factor, boundary.motion_bounds);
        (
            boundary.opacity * sample.opacity,
            sample.scale,
            sample.offset,
            sample.clip,
        )
    } else {
        (boundary.opacity, boundary.scale, Point::default(), None)
    }
}

/// A canvas-local scene painted independently of the window's views.
pub struct LiveCanvas {
    /// The scene being built.
    pub scene: Scene,
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

    /// Paints a quad in canvas-local coordinates.
    pub fn paint_quad(&mut self, quad: PaintQuad) {
        self.scene.insert_primitive(Quad {
            bounds: quad.bounds.scale(self.scale_factor),
            content_mask: ContentMask {
                bounds: Bounds::new(Point::default(), self.size).scale(self.scale_factor),
                ..Default::default()
            },
            background: quad.background,
            border_color: quad.border_color,
            corner_radii: quad.corner_radii.scale(self.scale_factor),
            border_widths: quad.border_widths.scale(self.scale_factor),
            border_style: quad.border_style,
            border_dashed_length: quad.border_dashed_length,
            border_dashed_gap: quad.border_dashed_gap,
            corner_smoothing: quad.corner_smoothing,
            ..Default::default()
        });
    }

    /// Paints a path in canvas-local coordinates.
    pub fn paint_path(&mut self, mut path: Path<Pixels>, color: impl Into<Background>) {
        path.content_mask = ContentMask {
            bounds: Bounds::new(Point::default(), self.size),
            ..Default::default()
        };
        path.color = color.into();
        self.scene.insert_primitive(path.scale(self.scale_factor));
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

//! Time-driven transforms of retained painted subtrees.

use crate::{Bounds, FilterBoundary, Pixels, Point, ScaledPixels, point};
use serde::{Deserialize, Serialize};

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

/// The timing curve applied to linear progress.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum MotionEasing {
    /// CSS cubic-bezier(x1, y1, x2, y2).
    CubicBezier([f32; 4]),
    /// CSS steps(n, jump-end).
    Steps(u32),
}

/// A retained subtree's opacity, translation, and scale over time.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[expect(missing_docs)]
pub struct Motion {
    pub start_ms: u32,
    pub duration_ms: u32,
    pub repeat: MotionRepeat,
    pub easing: MotionEasing,
    pub opacity: (f32, f32),
    pub offset: (Point<Pixels>, Point<Pixels>),
    pub scale: (f32, f32),
    pub origin: Point<f32>,
}

/// Composite parameters sampled in device pixels.
#[derive(Clone, Copy, PartialEq, Debug)]
#[expect(missing_docs)]
pub struct MotionSample {
    pub opacity: f32,
    pub offset: Point<ScaledPixels>,
    pub scale: f32,
}

impl Motion {
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
        scale: (1.0, 1.0),
        origin: point(0.5, 0.5),
    };

    /// Whether this declaration animates.
    pub fn is_animated(&self) -> bool {
        self.duration_ms > 0
    }

    /// Eased progress at the wrapping animation clock value.
    pub fn progress(&self, now_ms: u32) -> f32 {
        if !self.is_animated() {
            return 0.0;
        }
        let elapsed = now_ms.wrapping_sub(self.start_ms);
        let x = if elapsed >= 1 << 31 {
            0.0
        } else {
            elapsed as f32 / self.duration_ms as f32
        };
        let x = match self.repeat {
            MotionRepeat::Once => x.clamp(0.0, 1.0),
            MotionRepeat::Loop => x.fract(),
            MotionRepeat::Alternate => 1.0 - (x % 2.0 - 1.0).abs(),
        };
        match self.easing {
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
    pub fn sample(&self, now_ms: u32, scale_factor: f32) -> MotionSample {
        let p = self.progress(now_ms);
        MotionSample {
            opacity: self.opacity.0 + (self.opacity.1 - self.opacity.0) * p,
            offset: (self.offset.0 + (self.offset.1 - self.offset.0) * p).scale(scale_factor),
            scale: self.scale.0 + (self.scale.1 - self.scale.0) * p,
        }
    }

    /// Whether the motion changes between two clock values.
    pub fn changes_between(&self, a_ms: u32, b_ms: u32) -> bool {
        if !self.is_animated() || a_ms == b_ms {
            return false;
        }
        match self.easing {
            MotionEasing::Steps(n) => {
                (self.progress(a_ms) * n as f32).round() != (self.progress(b_ms) * n as f32).round()
            }
            MotionEasing::CubicBezier(_) => self.is_active(a_ms) || self.is_active(b_ms),
        }
    }

    /// Conservative bounds of every transformed position over this motion.
    pub fn sweep(
        &self,
        bounds: Bounds<ScaledPixels>,
        origin: Point<ScaledPixels>,
        scale_factor: f32,
    ) -> Bounds<ScaledPixels> {
        let (low, high) = match self.easing {
            MotionEasing::CubicBezier([_, y1, _, y2]) => {
                (0.0f32.min(y1).min(y2), 1.0f32.max(y1).max(y2))
            }
            MotionEasing::Steps(_) => (0.0, 1.0),
        };
        [0.0, 1.0, low, high]
            .into_iter()
            .map(|p| {
                let offset =
                    (self.offset.0 + (self.offset.1 - self.offset.0) * p).scale(scale_factor);
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
) -> (f32, f32, Point<ScaledPixels>) {
    if boundary.motion.is_animated() {
        let sample = boundary.motion.sample(now_ms, scale_factor);
        (
            boundary.opacity * sample.opacity,
            sample.scale,
            sample.offset,
        )
    } else {
        (boundary.opacity, boundary.scale, Point::default())
    }
}

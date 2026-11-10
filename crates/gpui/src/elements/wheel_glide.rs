use crate::{App, Global, Pixels, Point, point, px, scene::cubic_bezier};
use web_time::Instant;

/// Enables smoothing of line-based wheel scrolling when installed and true.
pub struct WheelSmoothing(pub bool);
impl Global for WheelSmoothing {}

pub(crate) fn enabled(cx: &App) -> bool {
    cx.try_global::<WheelSmoothing>().is_some_and(|s| s.0)
}

#[derive(Default, Debug)]
pub(crate) struct WheelGlide {
    seg: Option<Segment>,
    applied: Point<Pixels>,
    pub(crate) max: Point<Pixels>,
}

#[derive(Debug)]
struct Segment {
    distance: Point<Pixels>,
    start: Instant,
    secs: f32,
    y1: f32,
}

impl WheelGlide {
    pub(crate) fn add(
        &mut self,
        delta: Point<Pixels>,
        room: (Point<Pixels>, Point<Pixels>),
        now: Instant,
    ) -> bool {
        let remaining = self
            .seg
            .as_ref()
            .map_or(Point::default(), |seg| seg.distance - self.applied);
        let target = remaining + delta;
        let new = point(
            target.x.clamp(-room.0.x, room.1.x),
            target.y.clamp(-room.0.y, room.1.y),
        );
        let change = new - remaining;
        if change.x.abs().max(change.y.abs()) < px(0.01) {
            return false;
        }
        if new == Point::default() {
            // Nothing is left to travel: the glide ends where the content is now.
            self.cancel();
            return false;
        }

        let horizontal = new.x.abs() > new.y.abs();
        let distance = if horizontal { new.x.0 } else { new.y.0 };
        let velocity = self.seg.as_ref().map_or(0.0, |seg| {
            let x = ((now - seg.start).as_secs_f32() / seg.secs).min(1.0);
            let d = if horizontal {
                seg.distance.x.0
            } else {
                seg.distance.y.0
            };
            slope(seg.y1, x) * d / seg.secs
        });
        let mut duration = secs(distance);
        if velocity * distance > 0.0 {
            duration = duration.min(2.5 * distance / velocity);
        }
        let initial_slope = (velocity * duration / distance).clamp(-1000.0, 1000.0);
        self.seg = Some(Segment {
            distance: new,
            start: now,
            secs: duration,
            y1: 0.42 * initial_slope,
        });
        self.applied = Point::default();
        true
    }

    pub(crate) fn step(&mut self, now: Instant) -> Option<(Point<Pixels>, bool)> {
        let seg = self.seg.as_ref()?;
        let t = if seg.secs < 0.01 {
            1.0
        } else {
            ((now - seg.start).as_secs_f32() / seg.secs).min(1.0)
        };
        let want = seg.distance * ease(seg.y1, t);
        let slice = want - self.applied;
        self.applied = want;
        let active = t < 1.0;
        if !active {
            self.cancel();
        }
        Some((slice, active))
    }

    pub(crate) fn cancel(&mut self) {
        self.seg = None;
    }
}

fn secs(d: f32) -> f32 {
    (12.0 - (d.abs() - 120.0) * (6.0 / 360.0)).clamp(6.0, 12.0) / 60.0
}

fn ease(y1: f32, x: f32) -> f32 {
    cubic_bezier(0.42, y1, 0.58, 1.0, x)
}

fn slope(y1: f32, x: f32) -> f32 {
    (ease(y1, x + 1e-3) - ease(y1, x - 1e-3)) / 2e-3
}

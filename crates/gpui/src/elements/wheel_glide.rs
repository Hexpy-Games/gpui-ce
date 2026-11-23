use crate::{App, Global, Pixels, Point, point, px, scene::cubic_bezier};
use std::time::Duration;
use web_time::Instant;

/// Longest stretch of unplayed motion a new notch may replay at once: one 30 Hz frame,
/// so a long stall does not jump the content.
const MAX_CATCH_UP: Duration = Duration::from_millis(33);

/// Enables smoothing of line-based wheel scrolling when installed and true.
pub struct WheelSmoothing(pub bool);
impl Global for WheelSmoothing {}

pub(crate) fn enabled(cx: &App) -> bool {
    cx.try_global::<WheelSmoothing>().is_some_and(|s| s.0)
}

#[derive(Default, Debug)]
pub(crate) struct WheelGlide {
    seg: Option<Segment>,
    /// Distance of the segment already handed out by `step`, as of `applied_at`.
    applied: Point<Pixels>,
    /// The instant `applied` was last brought up to date (segment start or last step).
    applied_at: Option<Instant>,
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
    pub(crate) fn remaining(&self) -> Point<Pixels> {
        self.seg
            .as_ref()
            .map_or(Point::default(), |seg| seg.distance - self.applied)
    }

    /// Adds a wheel notch. A running glide continues from the instant it last played
    /// (the last frame), not from the event time, so the motion it owed between that
    /// frame and the event is not postponed into a short next frame.
    pub(crate) fn add(
        &mut self,
        delta: Point<Pixels>,
        room: (Point<Pixels>, Point<Pixels>),
        now: Instant,
    ) -> bool {
        let remaining = self.remaining();
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

        let base = self.applied_at.map_or(now, |at| {
            let earliest = now.checked_sub(MAX_CATCH_UP).unwrap_or(now);
            at.max(earliest).min(now)
        });
        let horizontal = new.x.abs() > new.y.abs();
        let distance = if horizontal { new.x.0 } else { new.y.0 };
        let velocity = self.seg.as_ref().map_or(0.0, |seg| {
            let x = (base.saturating_duration_since(seg.start).as_secs_f32() / seg.secs).min(1.0);
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
            start: base,
            secs: duration,
            y1: 0.42 * initial_slope,
        });
        self.applied = Point::default();
        self.applied_at = Some(base);
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
        self.applied_at = Some(now);
        let active = t < 1.0;
        if !active {
            self.cancel();
        }
        Some((slice, active))
    }

    pub(crate) fn cancel(&mut self) {
        self.seg = None;
        self.applied_at = None;
    }
}

fn secs(d: f32) -> f32 {
    (12.0 - (d.abs() - 120.0) * (6.0 / 360.0)).clamp(6.0, 12.0) / 60.0
}

fn ease(y1: f32, x: f32) -> f32 {
    cubic_bezier(0.42, y1, 0.58, 1.0, x)
}

fn slope(y1: f32, x: f32) -> f32 {
    // One-sided at the ends of the curve, where `ease` clamps its input.
    let lo = (x - 1e-3).max(0.0);
    let hi = (x + 1e-3).min(1.0);
    (ease(y1, hi) - ease(y1, lo)) / (hi - lo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Plays notches of -60 px landing `offsets_ms` after each of frames 0, 7, 14 and 21
    /// and checks the glide stays continuous and plays exactly the summed deltas.
    fn check_continuity(offsets_ms: &[u64]) {
        let base = Instant::now();
        let frame = Duration::from_secs_f64(1.0 / 60.0);
        let room = (
            point(px(10_000.), px(10_000.)),
            point(px(10_000.), px(10_000.)),
        );
        let delta = point(px(0.), px(-60.));
        let notch_frames = [0usize, 7, 14, 21];
        let mut glide = WheelGlide::default();
        let mut steps: Vec<f32> = Vec::new();
        let mut total = 0.0;
        let mut ended_at = None;
        for k in 0..160usize {
            let at = base + frame * k as u32;
            let mut step = 0.0;
            if let Some((slice, active)) = glide.step(at) {
                step = slice.y.0;
                total += step;
                if !active {
                    ended_at = Some(k);
                }
            }
            steps.push(step);
            if notch_frames.contains(&k) {
                for &ms in offsets_ms {
                    assert!(glide.add(delta, room, at + Duration::from_millis(ms)));
                }
            }
            if ended_at.is_some() {
                break;
            }
        }
        for &k in &notch_frames[1..] {
            let ratio = steps[k + 1] / steps[k];
            assert!(
                ratio >= 0.7,
                "frame after notch at {k} moved {ratio:.2} of the one before"
            );
        }
        let last = *notch_frames.last().unwrap();
        let ended = ended_at.expect("glide must end");
        assert!(
            ended <= last + 30,
            "glide ended {} frames after the last notch",
            ended - last
        );
        let want = -60.0 * (notch_frames.len() * offsets_ms.len()) as f32;
        assert!((total - want).abs() < 0.01, "played {total}, wanted {want}");
    }

    // test-category: pure-logic
    #[test]
    fn a_notch_mid_frame_continues_the_glide_without_a_hitch() {
        check_continuity(&[10]);
    }

    // test-category: pure-logic
    #[test]
    fn two_notches_in_one_frame_keep_the_glide_continuous() {
        check_continuity(&[4, 10]);
    }
}

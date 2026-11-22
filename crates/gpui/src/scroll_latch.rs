//! Wheel-gesture latching, as Chromium does it: a gesture moves the one scroller picked at its
//! start, and its momentum stays with it.
use crate::{IsZero, Pixels, Point, ScrollDelta, ScrollWheelEvent, TouchPhase, px};
use smallvec::SmallVec;
use std::{
    rc::Rc,
    time::{Duration, Instant},
};

/// A phase-less wheel idle this long starts a new gesture (Chromium's
/// `kDefaultMouseWheelLatchingTransaction`).
const WHEEL_TRANSACTION: Duration = Duration::from_millis(500);
/// Less than this from an edge counts as at the edge.
const EDGE: Pixels = px(0.5);

/// Identifies a scroller across frames: the address of its persistent scroll state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScrollLatchKey(usize);

impl ScrollLatchKey {
    pub(crate) fn of<T>(state: &Rc<T>) -> Self {
        Self(Rc::as_ptr(state).cast::<()>().addr())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Gesture {
    /// Phase-less wheel events; a pause ends the gesture.
    #[default]
    Wheel,
    /// Between a touch's `Started` and its `Ended`.
    Contact,
    /// After `Ended`: momentum may follow.
    Released,
}

/// One per window. Each wheel event: `begin`, scrollers `offer` in the capture phase,
/// `resolve`, then only the latched scroller moves in the bubble phase.
#[derive(Default)]
pub(crate) struct ScrollLatch {
    target: Option<ScrollLatchKey>,
    gesture: Gesture,
    last_event: Option<Instant>,
    momentum: bool,
    has_delta: bool,
    /// (scroller, can move directly, can move through the vertical-to-horizontal remap),
    /// outermost first.
    offers: SmallVec<[(ScrollLatchKey, bool, bool); 4]>,
}

impl ScrollLatch {
    /// Before the capture phase: ends the gesture this event does not continue.
    pub(crate) fn begin(&mut self, event: &ScrollWheelEvent, now: Instant) {
        self.offers.clear();
        self.momentum = event.momentum;
        self.has_delta = match event.delta {
            ScrollDelta::Pixels(delta) => !delta.x.is_zero() || !delta.y.is_zero(),
            ScrollDelta::Lines(delta) => delta.x != 0. || delta.y != 0.,
        };
        if !event.momentum {
            match event.touch_phase {
                TouchPhase::Started => {
                    self.target = None;
                    self.gesture = Gesture::Contact;
                }
                TouchPhase::Ended | TouchPhase::Cancelled => self.gesture = Gesture::Released,
                TouchPhase::Moved => match self.gesture {
                    Gesture::Contact => {}
                    Gesture::Released => {
                        self.target = None;
                        self.gesture = Gesture::Wheel;
                    }
                    Gesture::Wheel => {
                        if self
                            .last_event
                            .is_none_or(|last| now.duration_since(last) > WHEEL_TRANSACTION)
                        {
                            self.target = None;
                        }
                    }
                },
            }
        }
        self.last_event = Some(now);
    }

    /// Capture phase: a scroller under the pointer says whether this event's delta can move it.
    pub(crate) fn offer(&mut self, key: ScrollLatchKey, direct: bool, remapped: bool) {
        self.offers.push((key, direct, remapped));
    }

    /// Between capture and bubble: keeps the latched scroller (in wheel mode only while it is
    /// under the pointer); otherwise latches the innermost that can move directly, else
    /// through its remap, else the outermost. Momentum and zero deltas never latch.
    pub(crate) fn resolve(&mut self) {
        if self.momentum {
            return;
        }
        if self.gesture == Gesture::Wheel
            && self
                .target
                .is_some_and(|target| !self.offers.iter().any(|(key, ..)| *key == target))
        {
            self.target = None;
        }
        if self.target.is_none() && self.has_delta {
            self.target = (self.offers.iter().rev().find(|offer| offer.1))
                .or_else(|| self.offers.iter().rev().find(|offer| offer.2))
                .or_else(|| self.offers.first())
                .map(|offer| offer.0);
        }
    }

    pub(crate) fn is_latched(&self, key: ScrollLatchKey) -> bool {
        self.target == Some(key)
    }
}

/// Whether a scroller at `offset` (per axis 0 at the start, `-max` at the end) can move by
/// `delta` on some axis.
pub(crate) fn has_room(offset: Point<Pixels>, max: Point<Pixels>, delta: Point<Pixels>) -> bool {
    let axis = |offset: Pixels, max: Pixels, delta: Pixels| {
        let offset = offset.clamp(-max, Pixels::ZERO);
        (delta > Pixels::ZERO && offset < -EDGE) || (delta < Pixels::ZERO && offset > -max + EDGE)
    };
    axis(offset.x, max.x, delta.x) || axis(offset.y, max.y, delta.y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point;

    // test-category: pure-logic
    #[test]
    fn a_gesture_stays_on_the_scroller_it_started_on() {
        let (outer, inner) = (ScrollLatchKey(1), ScrollLatchKey(2));
        let t0 = Instant::now();
        let mut latch = ScrollLatch::default();
        // One event: phase, momentum, ms since t0, dy, offers (outer first, as capture runs).
        let mut step = |phase, momentum, ms, dy: f32, offers: &[(ScrollLatchKey, bool, bool)]| {
            latch.begin(
                &ScrollWheelEvent {
                    delta: ScrollDelta::Pixels(point(px(0.), px(dy))),
                    touch_phase: phase,
                    momentum,
                    ..Default::default()
                },
                t0 + Duration::from_millis(ms),
            );
            for &(key, direct, remapped) in offers {
                latch.offer(key, direct, remapped);
            }
            latch.resolve();
            latch.target
        };
        let both = [(outer, true, true), (inner, true, true)];
        let inner_stuck = [(outer, true, true), (inner, false, false)];
        use TouchPhase::*;
        // A touch with no movement yet latches nothing; the first move takes the inner scroller.
        assert_eq!(step(Started, false, 0, 0., &both), None);
        assert_eq!(step(Moved, false, 8, -5., &both), Some(inner));
        // At its edge the gesture and its momentum stay on it.
        assert_eq!(step(Moved, false, 16, -50., &inner_stuck), Some(inner));
        assert_eq!(step(Ended, false, 24, 0., &inner_stuck), Some(inner));
        assert_eq!(step(Moved, true, 40, -80., &inner_stuck), Some(inner));
        // A new touch: the inner one cannot move, the outer one takes it.
        assert_eq!(step(Started, false, 60, -5., &inner_stuck), Some(outer));
        // Nothing can move: the outermost; a reversal in the same gesture stays on it.
        let stuck = [(outer, false, false), (inner, false, false)];
        assert_eq!(step(Started, false, 100, -5., &stuck), Some(outer));
        assert_eq!(step(Moved, false, 110, 5., &both), Some(outer));
        // Momentum never latches.
        assert_eq!(step(Started, false, 200, 0., &both), None);
        assert_eq!(step(Ended, false, 210, 0., &both), None);
        assert_eq!(step(Moved, true, 220, -80., &both), None);
        // A phase-less wheel after a trackpad gesture is a new gesture; within 500 ms it stays.
        assert_eq!(step(Moved, false, 230, -40., &both), Some(inner));
        assert_eq!(step(Moved, false, 630, -40., &inner_stuck), Some(inner));
        // Idle over 500 ms: a new gesture.
        assert_eq!(step(Moved, false, 1200, -40., &inner_stuck), Some(outer));
        // A wheel whose pointer leaves the latched scroller: a new latch.
        let other = ScrollLatchKey(3);
        assert_eq!(
            step(Moved, false, 1300, -40., &[(other, true, true)]),
            Some(other)
        );
        // (rev) A touch keeps its scroller wherever the pointer goes.
        assert_eq!(step(Started, false, 1350, -5., &both), Some(inner));
        assert_eq!(
            step(Moved, false, 1360, -5., &[(other, true, true)]),
            Some(inner)
        );
        // The vertical-to-horizontal remap is only a fallback.
        assert_eq!(
            step(
                Started,
                false,
                1400,
                -5.,
                &[(outer, true, true), (inner, false, true)]
            ),
            Some(outer)
        );
        assert_eq!(
            step(
                Started,
                false,
                1500,
                -5.,
                &[(outer, false, false), (inner, false, true)]
            ),
            Some(inner)
        );
    }
}

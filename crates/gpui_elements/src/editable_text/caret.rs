#[cfg(test)]
use gpui::{AppContext, TestAppContext};

use gpui::{Context, Entity, EventEmitter, Subscription};
use smallvec::SmallVec;
use std::time::Duration;

/// Default interval for caret blinking (500ms).
pub const BLINK_INTERVAL_500MS: Duration = Duration::from_millis(500);

/// Events emitted that the [`Caret`] listens to.
pub enum CaretNotify {
    /// The caret should pause blinking in response to a user-action
    PauseBlinking,
}

/// Controls caret visibility and blinking; text layout owns position and geometry.
/// Blinking is disabled by default.
#[derive(Default)]
pub struct Caret {
    /// The frequency at which the caret blinks
    interval: Duration,
    /// Animation clock value of the last focus gain or user activity.
    blink_start_ms: u32,
    /// Whether the caret's EditableText element is currently focused.
    /// Caret is only eligible to be blinking if currently focused.
    has_focus: bool,
    #[allow(dead_code)]
    subscriptions: SmallVec<[Subscription; 2]>,
}

impl Caret {
    /// Returns the duration of the current blink interval
    pub fn blink_interval(&self) -> Duration {
        self.interval
    }

    /// Sets the blinking interval of the caret.
    pub fn set_blink_interval(&mut self, interval: Duration) {
        self.interval = interval;
    }

    /// Sets the blinking interval of the caret.
    pub fn with_blink_interval(mut self, interval: Duration) -> Self {
        self.set_blink_interval(interval);
        self
    }

    /// Sets the blinking interval of the caret to the global "default".
    /// The true default of the caret is "do not blink".
    pub fn with_blink_interval_500ms(self) -> Self {
        self.with_blink_interval(BLINK_INTERVAL_500MS)
    }

    /// Listens for CaretNotify events on an entity (e.g. [`EditableTextState`]).
    pub fn subscribe_to<E>(&mut self, emitter: &Entity<E>, cx: &mut Context<Self>)
    where
        E: EventEmitter<CaretNotify>,
    {
        let handle = cx.subscribe(emitter, |state, _emitter, event, cx| match event {
            CaretNotify::PauseBlinking => {
                if state.interval.is_zero() || !state.has_focus {
                    return;
                }

                // Leave the caret visible for one complete interval after user activity.
                state.blink_start_ms = gpui::animation_time_ms();
                cx.notify();
            }
        });
        self.subscriptions.push(handle);
    }

    /// Processes updates during prepaint and returns whether the caret is focused.
    pub(super) fn update_focus(&mut self, is_focused: bool, cx: &mut Context<Self>) -> bool {
        if self.has_focus != is_focused {
            self.has_focus = is_focused;
            if is_focused {
                self.blink_start_ms = gpui::animation_time_ms();
            }
            cx.notify();
        }

        is_focused
    }

    pub(super) fn blink(&self) -> gpui::LayerMotion {
        if self.interval.is_zero() {
            gpui::LayerMotion::NONE
        } else {
            gpui::LayerMotion {
                start_ms: self.blink_start_ms,
                duration_ms: 2 * self.interval.as_millis() as u32,
                repeat: gpui::MotionRepeat::Loop,
                easing: gpui::MotionEasing::linear(&[
                    (0.0, 0.0),
                    (0.5, 0.0),
                    (0.5, 1.0),
                    (1.0, 1.0),
                ]),
                opacity: (1.0, 0.0),
                ..gpui::LayerMotion::NONE
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CaretEventEmitter;
    impl EventEmitter<CaretNotify> for CaretEventEmitter {}

    #[gpui::test]
    fn visibility_tracks_focus_activity_and_blur(cx: &mut TestAppContext) {
        let interval = Duration::from_millis(10);
        let emitter = cx.new(|_| CaretEventEmitter);
        let caret = cx.new(|cx| {
            let mut caret = Caret::default().with_blink_interval(interval);
            caret.subscribe_to(&emitter, cx);
            caret
        });

        caret.update(cx, |caret, cx| assert!(caret.update_focus(true, cx)));

        let blink = cx.read(|cx| caret.read(cx).blink());
        assert!(blink.is_animated());
        assert_eq!(blink.duration_ms, 2 * interval.as_millis() as u32);

        std::thread::sleep(interval);
        emitter.update(cx, |_, cx| cx.emit(CaretNotify::PauseBlinking));
        cx.run_until_parked();
        assert!(cx.read(|cx| caret.read(cx).blink().start_ms) > blink.start_ms);

        caret.update(cx, |caret, cx| assert!(!caret.update_focus(false, cx)));
    }
}

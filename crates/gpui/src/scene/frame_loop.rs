//! Looping flipbook sprites and the animation clock the GPU reads.

use scheduler::Instant;
use std::{
    cell::Cell,
    sync::{
        LazyLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

/// Border, in device pixels, around each flipbook cell.
pub const FRAME_GUTTER: u32 = 1;

/// A looping flipbook a sprite carries. Its atlas tile is a row-major grid of `frame_count`
/// equal cells, each with a `FRAME_GUTTER` transparent border. The GPU picks the cell from the
/// animation clock every frame, so the scene does not change while it plays.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct FrameLoop {
    /// Frames in the loop; 0 means a still sprite drawn from the whole tile.
    pub frame_count: u32,
    /// Grid columns; rows = ceil(frame_count / columns).
    pub columns: u32,
    /// Milliseconds per loop.
    pub period_ms: u32,
    /// Clock value (ms) at which frame 0 shows, so a loop starts where it is mounted.
    pub start_ms: u32,
    /// Milliseconds per clockwise turn of the whole sprite about its bounds' centre; 0 means no turn.
    pub turn_period_ms: u32,
    /// Keeps the stride a multiple of 8.
    pub padding: u32,
}

impl FrameLoop {
    /// A still sprite.
    pub const NONE: Self = Self {
        frame_count: 0,
        columns: 0,
        period_ms: 0,
        start_ms: 0,
        turn_period_ms: 0,
        padding: 0,
    };

    /// Whether the sprite changes over time.
    pub fn is_animated(&self) -> bool {
        self.frame_count > 1 || self.turn_period_ms != 0
    }
}

/// A flipbook for painting APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameSpec {
    /// Frames rasterized for one loop.
    pub frame_count: usize,
    /// Loop length.
    pub period: Duration,
    /// Clock value (from [`animation_time_ms`]) at which frame 0 shows.
    pub start_ms: u32,
    /// One clockwise turn of the whole sprite per this duration, if any.
    pub turn: Option<Duration>,
}

static ANIMATION_EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

thread_local! {
    /// The running frame's stamp on the animation clock, in nanoseconds, while a frame runs on
    /// this thread.
    static FRAME_NS: Cell<Option<u64>> = const { Cell::new(None) };
}

/// The latest animation time handed out, so the clock never goes backwards.
static HIGH_WATER_NS: AtomicU64 = AtomicU64::new(0);

/// The refresh period of the last frame that had a vsync, 0 when none had.
static VSYNC_PERIOD_NS: AtomicU64 = AtomicU64::new(0);

/// Nanoseconds on the animation clock's epoch: what [`FrameVsync`] is measured on.
pub fn animation_clock_ns() -> u64 {
    ANIMATION_EPOCH.elapsed().as_nanos() as u64
}

/// Hands out `ns`, or the latest time already handed out if that is later.
fn hand_out(ns: u64) -> u64 {
    ns.max(HIGH_WATER_NS.fetch_max(ns, Ordering::Relaxed))
}

/// The display's current vsync, as a platform's frame source reports it.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct FrameVsync {
    /// When the vsync happened, on [`animation_clock_ns`]'s epoch.
    pub at_ns: u64,
    /// The display's refresh period.
    pub period_ns: u64,
}

/// One frame on the animation clock. While it lives, [`animation_time_ms`] on this thread
/// reads one value: the display's vsync when the platform reports it, else the wall clock at
/// the frame's start, never earlier than any time handed out before.
pub struct AnimationFrame {
    previous: Option<u64>,
}

impl AnimationFrame {
    /// Starts a frame stamped at `vsync`, or now.
    pub fn begin(vsync: Option<FrameVsync>) -> Self {
        if let Some(vsync) = vsync {
            VSYNC_PERIOD_NS.store(vsync.period_ns, Ordering::Relaxed);
        }
        let stamp = hand_out(vsync.map_or_else(animation_clock_ns, |vsync| vsync.at_ns));
        Self {
            previous: FRAME_NS.replace(Some(stamp)),
        }
    }
}

impl Drop for AnimationFrame {
    fn drop(&mut self) {
        FRAME_NS.set(self.previous);
    }
}

/// The renderer's single live-layer gate stays below a 60 Hz period with room for tick jitter
/// and above a 120 Hz period, following the display up to 60 Hz and every other tick above it.
pub const LIVE_FRAME_MIN_MS: u32 = 12;

/// Whether the renderer may draw another live-layer frame.
pub fn live_frame_due(last_ms: u32, now_ms: u32) -> bool {
    now_ms.wrapping_sub(last_ms) >= LIVE_FRAME_MIN_MS
}

#[cfg(any(test, feature = "test-support"))]
thread_local! {
    static PINNED_MS: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// Pins [`animation_time_ms`] on this thread to `ms` until it is set back to `None`; for deterministic frame captures.
#[cfg(any(test, feature = "test-support"))]
pub fn set_animation_clock(ms: Option<u32>) {
    PINNED_MS.set(ms);
}

/// The animation clock in milliseconds, wrapping at 2^32 (about 49.7 days): the running frame's
/// stamp during a frame (see [`AnimationFrame`]), else now (less one refresh period while frames
/// follow the display); it never goes backwards.
pub fn animation_time_ms() -> u32 {
    #[cfg(any(test, feature = "test-support"))]
    if let Some(ms) = PINNED_MS.get() {
        return ms;
    }

    // Between frames: now, less one refresh period when frames follow the display. The
    // display's latest vsync is never older than that, so the next frame's stamp is not held
    // back by a time handed out in between.
    let ns = FRAME_NS.get().unwrap_or_else(|| {
        hand_out(animation_clock_ns().saturating_sub(VSYNC_PERIOD_NS.load(Ordering::Relaxed)))
    });
    (ns / 1_000_000) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    // test-category: pure-logic
    #[test]
    fn a_frame_reads_one_time_that_never_goes_back() {
        let before = animation_time_ms();
        let frame = AnimationFrame::begin(Some(FrameVsync {
            at_ns: 0,
            period_ns: 16_666_667,
        }));
        let stamp = animation_time_ms();
        assert!(stamp >= before);
        std::thread::sleep(std::time::Duration::from_millis(3));
        assert_eq!(animation_time_ms(), stamp);
        drop(frame);
        assert!(animation_time_ms() >= stamp);
    }

    #[test]
    fn live_layers_follow_the_display_up_to_60_hz() {
        let mut last = 0;
        for i in 0..120 {
            let now = (i * 1000 / 60) as u32 + u32::from(i % 3 == 0);
            let draw = live_frame_due(last, now);
            if draw {
                last = now;
            }
            assert_eq!(draw, i > 0);
        }

        let mut last = 0;
        for i in 0..240 {
            let now = (i * 1000 / 120) as u32;
            let draw = live_frame_due(last, now);
            if draw {
                last = now;
            }
            assert_eq!(draw, i > 0 && i % 2 == 0);
        }
    }

    #[test]
    fn pinned_animation_clock_is_seen_until_released() {
        set_animation_clock(Some(1234));
        assert_eq!(animation_time_ms(), 1234);
        assert_eq!(animation_time_ms(), 1234);
        set_animation_clock(None);
    }
}

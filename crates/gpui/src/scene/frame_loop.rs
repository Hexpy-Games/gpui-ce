//! Looping flipbook sprites and the animation clock the GPU reads.

use scheduler::Instant;
use std::{sync::LazyLock, time::Duration};

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

/// Milliseconds since the process's first call, wrapping at 2^32 (about 49.7 days).
pub fn animation_time_ms() -> u32 {
    (ANIMATION_EPOCH.elapsed().as_millis() as u64) as u32
}

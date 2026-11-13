use std::{
    sync::{
        LazyLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use gpui_util::ResultExt;
use windows::Win32::{
    Foundation::HWND,
    Graphics::Dwm::{DWM_TIMING_INFO, DwmFlush, DwmGetCompositionTimingInfo},
    System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency},
};

/// The latest DWM vblank on gpui's animation clock; written by the vsync thread. Zero is unknown.
static LATEST_VSYNC_NS: AtomicU64 = AtomicU64::new(0);
/// The latest DWM refresh period; written by the vsync thread. Zero is unknown.
static LATEST_PERIOD_NS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn latest_vsync() -> Option<gpui::FrameVsync> {
    let at_ns = LATEST_VSYNC_NS.load(Ordering::Relaxed);
    (at_ns != 0).then(|| gpui::FrameVsync {
        at_ns,
        period_ns: LATEST_PERIOD_NS.load(Ordering::Relaxed),
    })
}

static QPC_TICKS_PER_SECOND: LazyLock<u64> = LazyLock::new(|| {
    let mut frequency = 0;
    // On systems that run Windows XP or later, the function will always succeed and
    // will thus never return zero.
    unsafe { QueryPerformanceFrequency(&mut frequency).unwrap() };
    frequency as u64
});

const VSYNC_INTERVAL_THRESHOLD: Duration = Duration::from_millis(1);
const DEFAULT_VSYNC_INTERVAL: Duration = Duration::from_micros(16_666); // ~60Hz

pub(crate) struct VSyncProvider {
    interval: Duration,
    f: Box<dyn Fn() -> bool>,
}

impl VSyncProvider {
    pub(crate) fn new() -> Self {
        let interval = get_dwm_interval()
            .context("Failed to get DWM interval")
            .log_err()
            .unwrap_or(DEFAULT_VSYNC_INTERVAL);
        let f = Box::new(|| unsafe { DwmFlush().is_ok() });
        Self { interval, f }
    }

    pub(crate) fn wait_for_vsync(&self) {
        let vsync_start = Instant::now();
        let wait_succeeded = (self.f)();
        let elapsed = vsync_start.elapsed();
        // DwmFlush and DCompositionWaitForCompositorClock returns very early
        // instead of waiting until vblank when the monitor goes to sleep or is
        // unplugged (nothing to present due to desktop occlusion). We use 1ms as
        // a threshold for the duration of the wait functions and fallback to
        // Sleep() if it returns before that. This could happen during normal
        // operation for the first call after the vsync thread becomes non-idle,
        // but it shouldn't happen often.
        if !wait_succeeded || elapsed < VSYNC_INTERVAL_THRESHOLD {
            LATEST_VSYNC_NS.store(0, Ordering::Relaxed);
            log::trace!("VSyncProvider::wait_for_vsync() took less time than expected");
            std::thread::sleep(self.interval);
        } else {
            record_vsync();
        }
    }
}

fn get_dwm_interval() -> Result<Duration> {
    Ok(interval_of(&timing_info()?))
}

fn timing_info() -> Result<DWM_TIMING_INFO> {
    let mut timing_info = DWM_TIMING_INFO {
        cbSize: std::mem::size_of::<DWM_TIMING_INFO>() as u32,
        ..Default::default()
    };
    unsafe { DwmGetCompositionTimingInfo(HWND::default(), &mut timing_info) }?;
    Ok(timing_info)
}

fn interval_of(info: &DWM_TIMING_INFO) -> Duration {
    let interval = retrieve_duration(info.qpcRefreshPeriod, *QPC_TICKS_PER_SECOND);
    // Check for interval values that are impossibly low. A 29 microsecond
    // interval was seen (from a qpcRefreshPeriod of 60).
    if interval < VSYNC_INTERVAL_THRESHOLD {
        retrieve_duration(
            info.rateRefresh.uiDenominator as u64,
            info.rateRefresh.uiNumerator as u64,
        )
    } else {
        interval
    }
}

/// A QPC vblank mapped onto the animation clock: now on that clock, plus how far the vblank
/// lies from now in QPC time.
fn vsync_on_animation_clock(
    qpc_vblank: u64,
    qpc_now: u64,
    ticks_per_second: u64,
    clock_now_ns: u64,
) -> u64 {
    let offset_ns =
        (qpc_vblank as i128 - qpc_now as i128) * 1_000_000_000 / ticks_per_second as i128;
    (clock_now_ns as i128 + offset_ns).max(1) as u64
}

fn record_vsync() {
    let Ok(info) = timing_info() else {
        LATEST_VSYNC_NS.store(0, Ordering::Relaxed);
        return;
    };
    let mut qpc_now = 0;
    unsafe { QueryPerformanceCounter(&mut qpc_now).unwrap() };
    let clock_now = gpui::animation_clock_ns();
    LATEST_VSYNC_NS.store(
        vsync_on_animation_clock(
            info.qpcVBlank,
            qpc_now as u64,
            *QPC_TICKS_PER_SECOND,
            clock_now,
        ),
        Ordering::Relaxed,
    );
    LATEST_PERIOD_NS.store(interval_of(&info).as_nanos() as u64, Ordering::Relaxed);
}

#[inline]
fn retrieve_duration(counts: u64, ticks_per_second: u64) -> Duration {
    let ticks_per_microsecond = ticks_per_second / 1_000_000;
    Duration::from_micros(counts / ticks_per_microsecond)
}

#[cfg(test)]
mod tests {
    use super::vsync_on_animation_clock;

    // test-category: pure-logic
    #[test]
    fn vsync_maps_qpc_offset_onto_clock() {
        let ticks_per_second = 10_000_000;
        let qpc_now = 1_000_000;
        assert_eq!(
            vsync_on_animation_clock(qpc_now - 40_000, qpc_now, ticks_per_second, 1_000_000_000),
            996_000_000
        );
        assert_eq!(
            vsync_on_animation_clock(qpc_now + 10_000, qpc_now, ticks_per_second, 1_000_000_000),
            1_001_000_000
        );
        assert_eq!(
            vsync_on_animation_clock(qpc_now - 40_000, qpc_now, ticks_per_second, 0),
            1
        );
    }
}

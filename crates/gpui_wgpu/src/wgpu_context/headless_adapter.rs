// Added by HexpyGames for Butler (2026), under the Apache License 2.0; see NOTICE.
//! The Direct3D 12 adapter of a headless context on Windows: the system default adapter, which
//! other processes creating a device without naming an adapter use too. Shared textures and
//! fences only open on the adapter that created them.
use anyhow::{Context as _, Result};
use gpui_util::ResultExt as _;
use windows::Win32::{
    Foundation::LUID,
    Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1},
};

/// Overrides the headless adapter on Windows with an adapter LUID, as 16 hex digits (the high
/// part, then the low part), such as a producer's `DXGI_ADAPTER_DESC1::AdapterLuid`.
pub const HEADLESS_ADAPTER_LUID_ENV: &str = "GPUI_HEADLESS_ADAPTER_LUID";

/// The Direct3D 12 adapter whose LUID is the override, or else the system default adapter's;
/// `None` (after logging) when none matches, so the caller falls back to its own choice.
pub(super) fn headless_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    let luid = match std::env::var(HEADLESS_ADAPTER_LUID_ENV) {
        Ok(value) => u64::from_str_radix(value.trim().trim_start_matches("0x"), 16)
            .with_context(|| format!("parsing {HEADLESS_ADAPTER_LUID_ENV}={value:?}"))
            .log_err()?,
        Err(_) => default_adapter_luid().log_err()?,
    };
    let adapters = gpui::block_on(instance.enumerate_adapters(wgpu::Backends::DX12));
    let adapter = adapters
        .into_iter()
        .find(|adapter| adapter_luid(adapter) == Some(luid));
    if adapter.is_none() {
        log::warn!("no Direct3D 12 adapter has LUID {luid:016x}; using the low-power adapter");
    }
    adapter
}

/// The LUID of DXGI adapter 0, the system default adapter.
pub fn default_adapter_luid() -> Result<u64> {
    // SAFETY: plain factory creation and queries; the objects are released on drop.
    let desc = unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1()?;
        factory.EnumAdapters1(0)?.GetDesc1()
    }
    .context("reading the default DXGI adapter")?;
    Ok(luid_value(desc.AdapterLuid))
}

/// The LUID of a Direct3D 12 adapter, or `None` for another backend.
pub fn adapter_luid(adapter: &wgpu::Adapter) -> Option<u64> {
    // SAFETY: the raw adapter is only queried while the guard is held.
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Dx12>() }?;
    // SAFETY: the raw adapter is a live COM object.
    let desc = unsafe { hal.raw_adapter().GetDesc1() }.ok()?;
    Some(luid_value(desc.AdapterLuid))
}

fn luid_value(luid: LUID) -> u64 {
    (u64::from(luid.HighPart as u32) << 32) | u64::from(luid.LowPart)
}

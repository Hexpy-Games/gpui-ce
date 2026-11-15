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
/// part, then the low part) with an optional `0x` prefix, such as a producer's
/// `DXGI_ADAPTER_DESC1::AdapterLuid`.
pub const HEADLESS_ADAPTER_LUID_ENV: &str = "GPUI_HEADLESS_ADAPTER_LUID";

/// The Direct3D 12 adapter named by the override, or else the system default adapter (DXGI
/// adapter 0). `None` when wgpu does not expose that adapter, so the caller falls back to its
/// own choice.
///
/// wgpu hides some adapters, notably the non-software "Microsoft Basic Render Driver" that is
/// adapter 0 on hosts without a GPU (it exposes the software WARP adapter instead, under another
/// LUID). No wgpu adapter then shares a device with a producer that names no adapter, so such a
/// producer must create its device on the consumer's adapter by LUID instead.
pub(super) fn headless_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    let adapters = gpui::block_on(instance.enumerate_adapters(wgpu::Backends::DX12));
    let find = |luid: u64| {
        adapters
            .iter()
            .find(|adapter| adapter_luid(adapter) == Some(luid))
            .cloned()
    };
    if let Ok(value) = std::env::var(HEADLESS_ADAPTER_LUID_ENV) {
        match parse_luid(&value) {
            Some(luid) => match find(luid) {
                Some(adapter) => return Some(adapter),
                None => log::error!(
                    "no Direct3D 12 adapter has the LUID {HEADLESS_ADAPTER_LUID_ENV}={value:?} \
                     names; using the default adapter"
                ),
            },
            None => log::error!(
                "{HEADLESS_ADAPTER_LUID_ENV}={value:?} is not 16 hex digits; using the default \
                 adapter"
            ),
        }
    }
    let luid = default_adapter_luid().log_err()?;
    let adapter = find(luid);
    if adapter.is_none() {
        log::warn!(
            "wgpu does not expose the default adapter (LUID {luid:016x}); using the low-power \
             adapter, so shared-texture producers must use its LUID"
        );
    }
    adapter
}

/// Parses an optional `0x` or `0X` prefix and exactly 16 hex digits.
fn parse_luid(value: &str) -> Option<u64> {
    let digits = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
        .unwrap_or(value);
    if digits.len() != 16 || !digits.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(digits, 16).ok()
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

#[cfg(test)]
mod tests {
    use super::parse_luid;

    #[test]
    fn adapter_luids_parse_strictly() {
        assert_eq!(parse_luid("0000000000027c90"), Some(0x27c90));
        assert_eq!(parse_luid("0x0000000000027C90"), Some(0x27c90));
        assert_eq!(parse_luid("0X0000000000027c90"), Some(0x27c90));
        for invalid in [
            "27c90",
            "0x0x0000000000027c90",
            "000000000027c90 ",
            "",
            "+000000000027c90",
        ] {
            assert_eq!(parse_luid(invalid), None, "{invalid:?}");
        }
    }
}

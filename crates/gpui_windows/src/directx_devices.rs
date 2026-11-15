use anyhow::{Context, Result};
use gpui_util::ResultExt;
use itertools::Itertools;
use windows::Win32::{
    Foundation::{HMODULE, LUID},
    Graphics::{
        Direct3D::{
            D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_11_1,
        },
        Direct3D11::{
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_DEBUG, D3D11_SDK_VERSION,
            D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext,
        },
        Dxgi::{
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_DEBUG, DXGI_CREATE_FACTORY_FLAGS,
            DXGI_ERROR_NOT_FOUND, IDXGIAdapter1, IDXGIFactory6,
        },
    },
};
use windows::core::Interface;

pub(crate) fn try_to_recover_from_device_lost<T>(mut f: impl FnMut() -> Result<T>) -> Result<T> {
    (0..5)
        .map(|i| {
            if i > 0 {
                // Add a small delay before retrying
                std::thread::sleep(std::time::Duration::from_millis(100 + i * 10));
            }
            f()
        })
        .find_or_last(Result::is_ok)
        .unwrap()
        .context("DirectXRenderer failed to recover from lost device after multiple attempts")
}

/// The Direct3D 11 device every window renders with; the platform creates it once and shares
/// it with all windows (and recreates it after a device loss).
///
/// Adapter choice: the adapters in DXGI's default enumeration order (`EnumAdapters`, not
/// `EnumAdapterByGpuPreference`), the first one that creates a feature level 11.0+ device.
/// Adapter 0 is the adapter Windows assigns this process: normally the one driving the primary
/// display, unless a per-app GPU preference (Settings > System > Display > Graphics) reorders
/// it. So on a hybrid-GPU machine the renderer may run on either GPU; consumers learn which
/// from [`adapter_luid`] (`Window::gpu_adapter_luid`) rather than assuming one.
#[derive(Clone)]
pub(crate) struct DirectXDevices {
    pub(crate) adapter: IDXGIAdapter1,
    pub(crate) dxgi_factory: IDXGIFactory6,
    pub(crate) device: ID3D11Device,
    pub(crate) device_context: ID3D11DeviceContext,
}

impl DirectXDevices {
    pub(crate) fn new() -> Result<Self> {
        let debug_layer_available = check_debug_layer_available();
        let dxgi_factory =
            get_dxgi_factory(debug_layer_available).context("Creating DXGI factory")?;
        let (adapter, device, device_context, feature_level) =
            get_adapter(&dxgi_factory, debug_layer_available).context("Getting DXGI adapter")?;
        match feature_level {
            D3D_FEATURE_LEVEL_11_1 => {
                log::info!("Created device with Direct3D 11.1 feature level.")
            }
            D3D_FEATURE_LEVEL_11_0 => {
                log::info!("Created device with Direct3D 11.0 feature level.")
            }
            _ => unreachable!(),
        }

        Ok(Self {
            adapter,
            dxgi_factory,
            device,
            device_context,
        })
    }
}

/// The adapter's LUID as one integer: `HighPart` in the upper 32 bits, `LowPart` in the lower.
pub(crate) fn adapter_luid(adapter: &IDXGIAdapter1) -> Result<u64> {
    // SAFETY: GetDesc1 only writes the returned descriptor.
    let desc = unsafe { adapter.GetDesc1() }.context("Reading the DXGI adapter description")?;
    Ok(luid_to_u64(desc.AdapterLuid))
}

pub(crate) fn luid_to_u64(luid: LUID) -> u64 {
    (u64::from(luid.HighPart as u32) << 32) | u64::from(luid.LowPart)
}

#[inline]
fn check_debug_layer_available() -> bool {
    #[cfg(debug_assertions)]
    {
        use windows::Win32::Graphics::Dxgi::{DXGIGetDebugInterface1, IDXGIInfoQueue};

        unsafe { DXGIGetDebugInterface1::<IDXGIInfoQueue>(0) }
            .log_err()
            .is_some()
    }
    #[cfg(not(debug_assertions))]
    {
        false
    }
}

#[inline]
fn get_dxgi_factory(debug_layer_available: bool) -> Result<IDXGIFactory6> {
    let factory_flag = if debug_layer_available {
        DXGI_CREATE_FACTORY_DEBUG
    } else {
        #[cfg(debug_assertions)]
        log::warn!(
            "Failed to get DXGI debug interface. DirectX debugging features will be disabled."
        );
        DXGI_CREATE_FACTORY_FLAGS::default()
    };
    unsafe { Ok(CreateDXGIFactory2(factory_flag)?) }
}

#[inline]
fn get_adapter(
    dxgi_factory: &IDXGIFactory6,
    debug_layer_available: bool,
) -> Result<(
    IDXGIAdapter1,
    ID3D11Device,
    ID3D11DeviceContext,
    D3D_FEATURE_LEVEL,
)> {
    for adapter_index in 0.. {
        let adapter: IDXGIAdapter1 = match unsafe { dxgi_factory.EnumAdapters(adapter_index) } {
            Ok(adapter) => adapter.cast()?,
            Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
            Err(error) => return Err(error.into()),
        };
        if let Ok(desc) = unsafe { adapter.GetDesc1() } {
            let gpu_name = String::from_utf16_lossy(&desc.Description)
                .trim_matches(char::from(0))
                .to_string();
            log::info!(
                "Using GPU: {} (DXGI adapter {}, LUID {:016x})",
                gpu_name,
                adapter_index,
                luid_to_u64(desc.AdapterLuid)
            );
        }
        // Check to see whether the adapter supports Direct3D 11 and create
        // the device if it does.
        let mut context: Option<ID3D11DeviceContext> = None;
        let mut feature_level = D3D_FEATURE_LEVEL::default();
        if let Some(device) = get_device(
            &adapter,
            Some(&mut context),
            Some(&mut feature_level),
            debug_layer_available,
        )
        .log_err()
        {
            let context = context.context("D3D11CreateDevice returned no device context")?;
            return Ok((adapter, device, context, feature_level));
        }
    }

    anyhow::bail!("No compatible Direct3D 11 adapter found")
}

#[inline]
fn get_device(
    adapter: &IDXGIAdapter1,
    context: Option<*mut Option<ID3D11DeviceContext>>,
    feature_level: Option<*mut D3D_FEATURE_LEVEL>,
    debug_layer_available: bool,
) -> Result<ID3D11Device> {
    let mut device: Option<ID3D11Device> = None;
    let device_flags = if debug_layer_available {
        D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_DEBUG
    } else {
        D3D11_CREATE_DEVICE_BGRA_SUPPORT
    };
    unsafe {
        D3D11CreateDevice(
            adapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            HMODULE::default(),
            device_flags,
            // The generated shader corpus is Shader Model 5.0. Restrict device creation to
            // feature levels that can execute it instead of accepting FL10.1 and failing later
            // while constructing the renderer's first pipeline.
            Some(&[D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            feature_level,
            context,
        )?;
    }
    let device = device.context("D3D11CreateDevice returned no device")?;
    // FL11.0/11.1 guarantees the compute/raw-buffer capabilities required by the SM5 DXBC
    // artifacts, so no legacy D3D10.x optional-feature probe is needed here.
    Ok(device)
}

#[cfg(test)]
mod tests {
    use super::{DirectXDevices, adapter_luid, luid_to_u64};
    use anyhow::Result;
    use windows::Win32::{
        Foundation::LUID,
        Graphics::Dxgi::{DXGI_ERROR_NOT_FOUND, IDXGIDevice},
    };
    use windows::core::Interface as _;

    #[test]
    fn luid_packs_high_part_above_low_part() {
        let luid = LUID {
            LowPart: 0x0001_a14e,
            HighPart: 0,
        };
        assert_eq!(luid_to_u64(luid), 0x0001_a14e);
        let luid = LUID {
            LowPart: 0xdead_beef,
            HighPart: -1,
        };
        assert_eq!(luid_to_u64(luid), 0xffff_ffff_dead_beef);
    }

    /// The reported LUID is the adapter the D3D11 device actually runs on, as DXGI reports it
    /// for the device itself. Needs no window, so it also runs in session 0.
    #[test]
    fn reported_luid_is_the_device_adapter() -> Result<()> {
        let devices = DirectXDevices::new()?;
        let reported = adapter_luid(&devices.adapter)?;
        let dxgi_device: IDXGIDevice = devices.device.cast()?;
        // SAFETY: plain COM getters on live objects.
        let device_desc = unsafe { dxgi_device.GetAdapter()?.GetDesc()? };
        assert_eq!(reported, luid_to_u64(device_desc.AdapterLuid));

        // Log every adapter and the pick, so a run shows how this machine resolves.
        for index in 0.. {
            // SAFETY: plain COM enumeration on a live factory.
            let adapter = match unsafe { devices.dxgi_factory.EnumAdapters1(index) } {
                Ok(adapter) => adapter,
                Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(error) => return Err(error.into()),
            };
            // SAFETY: GetDesc1 only writes the returned descriptor.
            let desc = unsafe { adapter.GetDesc1()? };
            let luid = luid_to_u64(desc.AdapterLuid);
            let name = String::from_utf16_lossy(&desc.Description);
            eprintln!(
                "DXGI adapter {index}: {} LUID {luid:016x}{}",
                name.trim_matches(char::from(0)),
                if luid == reported { " <- renderer" } else { "" },
            );
        }
        Ok(())
    }
}

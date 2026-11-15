// Added by HexpyGames for Butler (2026), under the Apache License 2.0; see NOTICE.
use super::{WgpuRenderer, d3d12};
use anyhow::{Context as _, Result};
use gpui::{
    Bounds, ContentMask, DevicePixels, PaintSurface, Quad, ScaledPixels, Scene, Size,
    WindowsSharedTextureFrame, point, size,
};
use std::os::windows::io::{AsRawHandle as _, FromRawHandle as _, OwnedHandle};
use std::sync::Arc;
use std::time::{Duration, Instant};
use windows_061::{
    Win32::{
        Foundation::{GENERIC_ALL, HANDLE, HMODULE, LUID},
        Graphics::{
            Direct3D::{D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1},
            Direct3D11::*,
            Dxgi::{
                Common::{
                    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM,
                    DXGI_SAMPLE_DESC,
                },
                CreateDXGIFactory1, IDXGIAdapter, IDXGIFactory4, IDXGIResource1,
            },
        },
    },
    core::Interface as _,
};

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// A producer on its own D3D11 device, on wgpu's adapter, with a pool of two shared textures and
/// two shared fences, like a browser's accelerated off-screen paint. Frame `n` uses texture
/// `n % 2`. A third shared fence, the gate, lets the test hold the producer's GPU work back.
struct Producer {
    context: ID3D11DeviceContext4,
    targets: [ID3D11RenderTargetView; 2],
    fence: ID3D11Fence,
    release: ID3D11Fence,
    gate: ID3D11Fence,
    // [texture 0, texture 1, acquire fence, release fence, gate]
    handles: [Arc<OwnedHandle>; 5],
    size: Size<DevicePixels>,
}

impl Producer {
    fn new(renderer: &WgpuRenderer, format: DXGI_FORMAT) -> Result<Self> {
        // SAFETY: wgpu's device is live for the call.
        let luid = unsafe { d3d12(renderer)?.0.GetAdapterLuid() };
        // SAFETY: plain factory creation.
        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory1() }?;
        // SAFETY: `luid` names an adapter of this machine.
        let adapter: IDXGIAdapter = unsafe {
            factory.EnumAdapterByLuid(LUID {
                LowPart: luid.LowPart,
                HighPart: luid.HighPart,
            })
        }?;
        let (mut device, mut context) = (None, None);
        // SAFETY: the adapter is live and the out-pointers are valid for the call.
        unsafe {
            D3D11CreateDevice(
                &adapter,
                D3D_DRIVER_TYPE_UNKNOWN,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                Some(&[D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0]),
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        }?;
        let device: ID3D11Device = device.context("no producer device")?;
        let context: ID3D11DeviceContext4 = context.context("no producer context")?.cast()?;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: 40,
            Height: 20,
            MipLevels: 1,
            ArraySize: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: (D3D11_RESOURCE_MISC_SHARED.0 | D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0)
                as u32,
        };
        let shared_texture = || -> Result<(ID3D11RenderTargetView, Arc<OwnedHandle>)> {
            let mut texture = None;
            // SAFETY: `desc` is valid and `texture` is a valid out-pointer.
            unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture)) }?;
            let texture = texture.context("no producer texture")?;
            let mut target = None;
            // SAFETY: `texture` is a live render-target texture; `target` is a valid out-pointer.
            unsafe { device.CreateRenderTargetView(&texture, None, Some(&mut target)) }?;
            let target = target.context("no producer target")?;
            // Content no frame names yet, so a frame sampled before its paint shows red.
            // SAFETY: the view is live.
            unsafe { context.ClearRenderTargetView(&target, &RED) };
            let resource = texture.cast::<IDXGIResource1>()?;
            // SAFETY: the texture was created shareable; the new handle is owned at once.
            let handle = unsafe { owned(resource.CreateSharedHandle(None, GENERIC_ALL.0, None)?) };
            Ok((target, handle))
        };
        let ((target0, texture0), (target1, texture1)) = (shared_texture()?, shared_texture()?);
        let device5 = device.cast::<ID3D11Device5>()?;
        let shared_fence = || -> Result<(ID3D11Fence, Arc<OwnedHandle>)> {
            let mut fence: Option<ID3D11Fence> = None;
            // SAFETY: `fence` is a valid out-pointer.
            unsafe { device5.CreateFence(0, D3D11_FENCE_FLAG_SHARED, &mut fence) }?;
            let fence = fence.context("no producer fence")?;
            // SAFETY: the fence was created shareable; the new handle is owned at once.
            let handle = unsafe { owned(fence.CreateSharedHandle(None, GENERIC_ALL.0, None)?) };
            Ok((fence, handle))
        };
        let (fence, fence_handle) = shared_fence()?;
        let (release, release_handle) = shared_fence()?;
        let (gate, gate_handle) = shared_fence()?;
        Ok(Self {
            context,
            targets: [target0, target1],
            fence,
            release,
            gate,
            handles: [
                texture0,
                texture1,
                fence_handle,
                release_handle,
                gate_handle,
            ],
            size: size(DevicePixels(40), DevicePixels(20)),
        })
    }

    /// Paints `color` into texture `fence_value % 2` and signals `fence_value` on the acquire
    /// fence. With `gate`, the producer's GPU first waits for the gate fence to reach it.
    fn paint(
        &self,
        color: [f32; 4],
        fence_value: u64,
        gate: Option<u64>,
    ) -> Result<WindowsSharedTextureFrame> {
        let index = (fence_value % 2) as usize;
        // SAFETY: every object is live and owned by this producer.
        unsafe {
            if let Some(gate) = gate {
                self.context.Wait(&self.gate, gate)?;
            }
            self.context
                .ClearRenderTargetView(&self.targets[index], &color);
            self.context.Signal(&self.fence, fence_value)?;
            self.context.Flush();
        }
        Ok(self.frame(index, fence_value, self.size))
    }

    fn frame(
        &self,
        index: usize,
        value: u64,
        size: Size<DevicePixels>,
    ) -> WindowsSharedTextureFrame {
        let texture = self.handles[index].clone();
        let (fence, release) = (self.handles[2].clone(), self.handles[3].clone());
        WindowsSharedTextureFrame::new(texture, fence, value, release, size)
    }

    /// The release fence's completed value once it reaches `value`, or after two seconds. The
    /// release signal is queued right after the submission `render_to_image` waits for.
    fn released(&self, value: u64) -> u64 {
        // SAFETY: the fence is live and owned by this producer.
        let completed = || unsafe { self.release.GetCompletedValue() };
        for _ in 0..200 {
            if completed() >= value {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        completed()
    }

    /// The gate fence opened on wgpu's D3D12 device, so the CPU can signal it.
    fn gate_on_d3d12(
        &self,
        renderer: &WgpuRenderer,
    ) -> Result<windows::Win32::Graphics::Direct3D12::ID3D12Fence> {
        let device = d3d12(renderer)?.0;
        let mut fence = None;
        // SAFETY: the handle is owned by this producer and open; `fence` is a valid out-pointer.
        unsafe {
            device.OpenSharedHandle(
                windows::Win32::Foundation::HANDLE(self.handles[4].as_raw_handle()),
                &raw mut fence,
            )
        }?;
        fence.context("D3D12 returned no gate fence")
    }
}

/// # Safety
/// `handle` must be an open handle the caller owns.
unsafe fn owned(handle: HANDLE) -> Arc<OwnedHandle> {
    // SAFETY: guaranteed by the caller.
    Arc::new(unsafe { OwnedHandle::from_raw_handle(handle.0) })
}

fn scaled(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
    Bounds::new(
        point(ScaledPixels(x), ScaledPixels(y)),
        size(ScaledPixels(width), ScaledPixels(height)),
    )
}

fn surface_scene(frame: Option<WindowsSharedTextureFrame>) -> Scene {
    let mask = ContentMask {
        bounds: scaled(0.0, 0.0, 64.0, 40.0),
        ..Default::default()
    };
    let mut scene = Scene::default();
    if let Some(frame) = frame {
        scene.insert_primitive(PaintSurface {
            order: 0,
            bounds: scaled(10.0, 10.0, 40.0, 20.0),
            content_mask: mask,
            source: frame.into(),
        });
    }
    scene.insert_primitive(Quad {
        bounds: scaled(54.0, 2.0, 8.0, 8.0),
        content_mask: mask,
        background: gpui::solid_background(gpui::rgb(0x00ff00)),
        ..Default::default()
    });
    scene.finish();
    scene
}

fn expect_pixel(image: &image::RgbaImage, name: &str, x: u32, y: u32, wanted: [u8; 3]) {
    let [r, g, b, _] = image.get_pixel(x, y).0;
    let close = |actual: u8, wanted: u8| actual.abs_diff(wanted) <= 8;
    assert!(
        close(r, wanted[0]) && close(g, wanted[1]) && close(b, wanted[2]),
        "{name} at ({x},{y}) rendered ({r},{g},{b}), expected {wanted:?}"
    );
}

/// Cached shared surfaces, and how many of them are drawable.
fn shared_state(renderer: &WgpuRenderer) -> (usize, usize) {
    let cache = renderer.resources().surface_cache.borrow();
    let surfaces = &cache.shared.surfaces;
    let drawable = surfaces.values().filter(|surface| !surface.failed).count();
    (surfaces.len(), drawable)
}

#[test]
fn shared_textures_render_release_skip_and_evict() -> Result<()> {
    // Shows wgpu's D3D12 debug-layer messages, when the layer is installed.
    let _ = env_logger::builder().is_test(true).try_init();
    let context = crate::WgpuContext::new_headless(None)?;
    let info = context.adapter.get_info();
    eprintln!("headless wgpu backend: {:?} ({})", info.backend, info.name);
    if info.backend != wgpu::Backend::Dx12 {
        // CI sets this so the test cannot pass there without exercising the import.
        anyhow::ensure!(
            std::env::var_os("GPUI_REQUIRE_DX12").is_none(),
            "GPUI_REQUIRE_DX12 is set but wgpu picked {:?}",
            info.backend
        );
        eprintln!("skipping: shared textures are imported only on Direct3D 12");
        return Ok(());
    }
    // Producers that create their device without naming an adapter use the default adapter, so
    // headless uses it too whenever wgpu exposes it (it hides the non-software Basic Render
    // Driver that is adapter 0 on hosts without a GPU). The producer below follows wgpu's
    // adapter by LUID either way.
    let headless_luid = crate::adapter_luid(&context.adapter).context("no adapter LUID")?;
    let default_luid = crate::default_adapter_luid()?;
    eprintln!("headless adapter LUID {headless_luid:016x}, default {default_luid:016x}");
    let exposed = gpui::block_on(context.instance.enumerate_adapters(wgpu::Backends::DX12))
        .iter()
        .any(|adapter| crate::adapter_luid(adapter) == Some(default_luid));
    if std::env::var_os(crate::HEADLESS_ADAPTER_LUID_ENV).is_some() {
        eprintln!(
            "not checking the default adapter: {}",
            crate::HEADLESS_ADAPTER_LUID_ENV
        );
    } else if !exposed {
        eprintln!("not checking the default adapter: wgpu does not expose it");
    } else {
        assert_eq!(
            headless_luid, default_luid,
            "headless must use the default adapter"
        );
    }
    let mut renderer =
        WgpuRenderer::new_headless(&context, size(DevicePixels(64), DevicePixels(40)))?;
    let producer = Producer::new(&renderer, DXGI_FORMAT_B8G8R8A8_UNORM)?;

    let image = renderer.render_to_image(&surface_scene(Some(producer.paint(BLUE, 1, None)?)))?;
    expect_pixel(&image, "first frame top-left", 11, 11, [0, 0, 255]);
    expect_pixel(&image, "first frame bottom-right", 48, 28, [0, 0, 255]);
    expect_pixel(&image, "left of surface", 8, 20, [0, 0, 0]);
    expect_pixel(&image, "quad", 58, 6, [0, 255, 0]);
    assert_eq!(shared_state(&renderer), (1, 1));
    assert_eq!(producer.released(1), 1, "the renderer must release frame 1");

    // The second texture holds red until the producer's GPU, held by the gate, paints it blue
    // and signals the acquire fence. The CPU opens the gate only after a delay, so the frame
    // shows blue only if the renderer's queue waited for the acquire fence.
    let gate = producer.gate_on_d3d12(&renderer)?;
    let frame = producer.paint(BLUE, 2, Some(1))?;
    let opener = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        let opened_at = Instant::now();
        // SAFETY: the fence is live; D3D12 fences are free-threaded.
        unsafe { gate.Signal(1) }.map(|()| opened_at)
    });
    let image = renderer.render_to_image(&surface_scene(Some(frame)))?;
    let rendered_at = Instant::now();
    let opened_at = opener.join().expect("gate thread panicked")?;
    expect_pixel(&image, "gated frame", 30, 20, [0, 0, 255]);
    assert!(
        rendered_at > opened_at,
        "the render returned before the gate opened"
    );
    // The pool's first texture stays imported while its release fence is in use.
    assert_eq!(shared_state(&renderer), (2, 2));
    assert_eq!(producer.released(2), 2, "the renderer must release frame 2");

    // A frame whose size disagrees with its texture is skipped on its own, yet released.
    producer.paint(RED, 3, None)?;
    let wrong_size = producer.frame(1, 3, size(DevicePixels(4), DevicePixels(4)));
    let image = renderer.render_to_image(&surface_scene(Some(wrong_size)))?;
    expect_pixel(&image, "size-mismatch frame", 30, 20, [0, 0, 0]);
    expect_pixel(&image, "quad beside it", 58, 6, [0, 255, 0]);
    assert_eq!(producer.released(3), 3, "a size-mismatch frame is released");
    let image = renderer.render_to_image(&surface_scene(Some(producer.paint(RED, 4, None)?)))?;
    expect_pixel(&image, "frame after the mismatch", 30, 20, [255, 0, 0]);
    assert_eq!(shared_state(&renderer), (2, 2));
    assert_eq!(producer.released(4), 4, "the renderer must release frame 4");

    // A surface that leaves the scene is evicted with its whole pool.
    let image = renderer.render_to_image(&surface_scene(None))?;
    expect_pixel(&image, "removed surface", 30, 20, [0, 0, 0]);
    assert_eq!(shared_state(&renderer), (0, 0));

    // A texture the renderer cannot sample is skipped and released; the frame still draws.
    let wrong_format = Producer::new(&renderer, DXGI_FORMAT_R8G8B8A8_UNORM)?;
    for value in 1..=2 {
        let frame = wrong_format.paint(RED, value, None)?;
        let image = renderer.render_to_image(&surface_scene(Some(frame)))?;
        expect_pixel(&image, "skipped surface", 30, 20, [0, 0, 0]);
        expect_pixel(&image, "quad beside skipped surface", 58, 6, [0, 255, 0]);
        assert_eq!(shared_state(&renderer), (value as usize, 0));
        assert_eq!(
            wrong_format.released(value),
            value,
            "a skipped frame is released"
        );
    }
    Ok(())
}

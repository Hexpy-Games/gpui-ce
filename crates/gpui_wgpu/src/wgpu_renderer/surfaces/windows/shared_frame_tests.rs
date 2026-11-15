// Added by HexpyGames for Butler (2026), under the Apache License 2.0; see NOTICE.
use super::{WgpuRenderer, d3d12};
use anyhow::{Context as _, Result};
use gpui::{
    Bounds, ContentMask, DevicePixels, PaintSurface, Quad, ScaledPixels, Scene, Size,
    WindowsSharedTextureFrame, point, size,
};
use std::os::windows::io::{FromRawHandle as _, OwnedHandle};
use std::sync::Arc;
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

/// A producer on its own D3D11 device, on wgpu's adapter, sharing a texture and two fences like
/// a browser's accelerated off-screen paint does.
struct Producer {
    context: ID3D11DeviceContext4,
    target: ID3D11RenderTargetView,
    fence: ID3D11Fence,
    release: ID3D11Fence,
    handles: [Arc<OwnedHandle>; 3],
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
        let mut texture = None;
        // SAFETY: `desc` is valid and `texture` is a valid out-pointer.
        unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture)) }?;
        let texture = texture.context("no producer texture")?;
        let mut target = None;
        // SAFETY: `texture` is a live render-target texture; `target` is a valid out-pointer.
        unsafe { device.CreateRenderTargetView(&texture, None, Some(&mut target)) }?;
        let device5 = device.cast::<ID3D11Device5>()?;
        let shared_fence = || -> Result<ID3D11Fence> {
            let mut fence = None;
            // SAFETY: `fence` is a valid out-pointer.
            unsafe { device5.CreateFence(0, D3D11_FENCE_FLAG_SHARED, &mut fence) }?;
            fence.context("no producer fence")
        };
        let (fence, release) = (shared_fence()?, shared_fence()?);
        let resource = texture.cast::<IDXGIResource1>()?;
        // SAFETY: the objects were created shareable; each call returns a new NT handle that is
        // owned at once.
        let handles = unsafe {
            [
                owned(resource.CreateSharedHandle(None, GENERIC_ALL.0, None)?),
                owned(fence.CreateSharedHandle(None, GENERIC_ALL.0, None)?),
                owned(release.CreateSharedHandle(None, GENERIC_ALL.0, None)?),
            ]
        };
        Ok(Self {
            context: context.context("no producer context")?.cast()?,
            target: target.context("no producer target")?,
            fence,
            release,
            handles,
            size: size(DevicePixels(40), DevicePixels(20)),
        })
    }

    /// Paints `color`, signals `fence_value` on the acquire fence, and returns the frame.
    fn paint(&self, color: [f32; 4], fence_value: u64) -> Result<WindowsSharedTextureFrame> {
        // SAFETY: every object is live and owned by this producer.
        unsafe {
            self.context.ClearRenderTargetView(&self.target, &color);
            self.context.Signal(&self.fence, fence_value)?;
            self.context.Flush();
        }
        let [texture, fence, release] = self.handles.clone();
        Ok(WindowsSharedTextureFrame::new(
            texture,
            fence,
            fence_value,
            release,
            self.size,
        ))
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
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        completed()
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
    let context = crate::WgpuContext::new_headless(None)?;
    let mut renderer =
        WgpuRenderer::new_headless(&context, size(DevicePixels(64), DevicePixels(40)))?;
    let producer = Producer::new(&renderer, DXGI_FORMAT_B8G8R8A8_UNORM)
        .context("the shared-texture path needs wgpu on Direct3D 12")?;

    let image = renderer.render_to_image(&surface_scene(Some(producer.paint(RED, 1)?)))?;
    expect_pixel(&image, "first frame top-left", 11, 11, [255, 0, 0]);
    expect_pixel(&image, "first frame bottom-right", 48, 28, [255, 0, 0]);
    expect_pixel(&image, "left of surface", 8, 20, [0, 0, 0]);
    expect_pixel(&image, "quad", 58, 6, [0, 255, 0]);
    assert_eq!(shared_state(&renderer), (1, 1));
    assert_eq!(producer.released(1), 1, "the renderer must release frame 1");

    // Same texture handle, higher fence value: the cached texture is sampled again.
    let image = renderer.render_to_image(&surface_scene(Some(producer.paint(BLUE, 2)?)))?;
    expect_pixel(&image, "second frame", 30, 20, [0, 0, 255]);
    assert_eq!(shared_state(&renderer), (1, 1));
    assert_eq!(producer.released(2), 2, "the renderer must release frame 2");

    // A frame whose size disagrees with its texture is skipped on its own, yet released.
    producer.paint(RED, 3)?;
    let [texture, fence, release] = producer.handles.clone();
    let tiny = size(DevicePixels(4), DevicePixels(4));
    let wrong_size = WindowsSharedTextureFrame::new(texture, fence, 3, release, tiny);
    let image = renderer.render_to_image(&surface_scene(Some(wrong_size)))?;
    expect_pixel(&image, "size-mismatch frame", 30, 20, [0, 0, 0]);
    expect_pixel(
        &image,
        "quad beside size-mismatch frame",
        58,
        6,
        [0, 255, 0],
    );
    assert_eq!(
        producer.released(3),
        3,
        "a size-mismatch frame must be released"
    );
    let image = renderer.render_to_image(&surface_scene(Some(producer.paint(BLUE, 4)?)))?;
    expect_pixel(&image, "frame after the mismatch", 30, 20, [0, 0, 255]);
    assert_eq!(producer.released(4), 4, "the renderer must release frame 4");

    // A surface that leaves the scene is evicted.
    let image = renderer.render_to_image(&surface_scene(None))?;
    expect_pixel(&image, "removed surface", 30, 20, [0, 0, 0]);
    assert_eq!(shared_state(&renderer), (0, 0));

    // A texture the renderer cannot sample is skipped and released; the frame still draws.
    let wrong_format = Producer::new(&renderer, DXGI_FORMAT_R8G8B8A8_UNORM)?;
    for value in 1..=2 {
        let frame = wrong_format.paint(RED, value)?;
        let image = renderer.render_to_image(&surface_scene(Some(frame)))?;
        expect_pixel(&image, "skipped surface", 30, 20, [0, 0, 0]);
        expect_pixel(&image, "quad beside skipped surface", 58, 6, [0, 255, 0]);
        assert_eq!(shared_state(&renderer), (1, 0));
        assert_eq!(
            wrong_format.released(value),
            value,
            "a skipped frame is released"
        );
    }
    Ok(())
}

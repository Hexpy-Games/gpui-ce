use crate::{DevicePixels, Size};
use std::os::windows::io::{AsHandle as _, AsRawHandle as _, BorrowedHandle, OwnedHandle};
use std::sync::Arc;
use windows_061::Win32::Graphics::Direct3D11::ID3D11Texture2D;

/// A Windows Graphics Capture frame backed by its native D3D11 texture.
#[derive(Clone)]
pub struct WindowsScreenCaptureFrame {
    texture: Arc<ID3D11Texture2D>,
    size: Size<DevicePixels>,
    display_time: u64,
}

impl WindowsScreenCaptureFrame {
    #[cfg(feature = "screen-capture")]
    pub(crate) fn new(
        texture: ID3D11Texture2D,
        size: Size<DevicePixels>,
        display_time: u64,
    ) -> Self {
        Self {
            texture: Arc::new(texture),
            size,
            display_time,
        }
    }

    /// Returns the native texture containing this frame.
    pub fn texture(&self) -> &ID3D11Texture2D {
        &self.texture
    }

    /// Returns the frame dimensions in device pixels.
    pub fn size(&self) -> Size<DevicePixels> {
        self.size
    }

    /// Returns the capture timestamp in 100-nanosecond units.
    pub fn display_time(&self) -> u64 {
        self.display_time
    }
}

impl std::fmt::Debug for WindowsScreenCaptureFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WindowsScreenCaptureFrame")
            .field("size", &self.size)
            .field("display_time", &self.display_time)
            .finish_non_exhaustive()
    }
}

/// A frame that another Direct3D 11 device rendered into a shared texture, such as a browser's
/// accelerated off-screen paint. Only the Direct3D 11 renderer imports it.
///
/// The producer creates the texture with `D3D11_RESOURCE_MISC_SHARED_NTHANDLE` and no keyed
/// mutex, as a single-sample, single-slice `DXGI_FORMAT_B8G8R8A8_UNORM` or
/// `DXGI_FORMAT_B8G8R8A8_TYPELESS` texture of premultiplied-alpha pixels (only mip 0 is shown).
/// It also creates two shared `ID3D11Fence`s:
///
/// - the acquire fence: after writing the frame the producer signals it with `fence_value` and
///   flushes; the renderer's GPU waits for that value before sampling the texture;
/// - the release fence: after each draw that samples the frame the renderer's GPU signals it
///   with `fence_value`.
///
/// A release fence serves one surface in one window, and its frames must be shown in increasing
/// fence-value order. The producer may write a texture last used for frame K again only once the
/// release fence's completed value is greater than K: the renderer has then drawn a newer frame,
/// so frame K is no longer in the scene. A producer therefore needs a pool of at least two
/// textures. A frame is released when it is drawn; a surface left entirely outside a damage
/// redraw is not drawn, so its frame is released when the surface is next drawn.
///
/// A surface the renderer cannot import (for example a texture of another format, or handles
/// from another adapter) is skipped and logged once; the renderer still signals its release
/// fence when it can open it, and retries the import when a frame brings other fence handles.
/// If the release fence itself cannot be used, frames are never released, so a producer should
/// treat a release that does not arrive within its own timeout as a failure and fall back, for
/// example by recreating its textures and fences.
///
/// A frame is identified by its texture handle and fence value, so a new frame must use a new
/// texture handle or a higher fence value, otherwise damage tracking treats it as unchanged.
/// The handles close when the last clone of the frame drops; renderers keep a clone while they
/// cache the opened objects, so a handle value cannot be reused for another object meanwhile.
#[derive(Clone)]
pub struct WindowsSharedTextureFrame {
    texture: Arc<OwnedHandle>,
    fence: Arc<OwnedHandle>,
    fence_value: u64,
    release: Arc<OwnedHandle>,
    size: Size<DevicePixels>,
}

impl WindowsSharedTextureFrame {
    /// Creates a frame from NT handles to a shared texture, its acquire fence and its release
    /// fence (see the type documentation for the protocol).
    ///
    /// Pass an `Arc` to reuse one handle across frames (the fences, or a texture from the
    /// producer's pool) so renderers open it only once. A renderer that cannot import the
    /// handles logs the failure once and skips the surface.
    pub fn new(
        texture: impl Into<Arc<OwnedHandle>>,
        fence: impl Into<Arc<OwnedHandle>>,
        fence_value: u64,
        release: impl Into<Arc<OwnedHandle>>,
        size: Size<DevicePixels>,
    ) -> Self {
        Self {
            texture: texture.into(),
            fence: fence.into(),
            fence_value,
            release: release.into(),
            size,
        }
    }

    /// Returns the NT handle of the shared texture.
    pub fn texture_handle(&self) -> BorrowedHandle<'_> {
        self.texture.as_handle()
    }

    /// Returns the NT handle of the acquire fence the producer signals.
    pub fn fence_handle(&self) -> BorrowedHandle<'_> {
        self.fence.as_handle()
    }

    /// Returns the value the producer signals on the acquire fence once the frame is written;
    /// the renderer signals the same value on the release fence after sampling it.
    pub fn fence_value(&self) -> u64 {
        self.fence_value
    }

    /// Returns the NT handle of the release fence the renderer signals.
    pub fn release_handle(&self) -> BorrowedHandle<'_> {
        self.release.as_handle()
    }

    /// Returns the frame dimensions in device pixels.
    pub fn size(&self) -> Size<DevicePixels> {
        self.size
    }
}

impl std::fmt::Debug for WindowsSharedTextureFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WindowsSharedTextureFrame")
            .field("texture", &self.texture.as_raw_handle())
            .field("fence", &self.fence.as_raw_handle())
            .field("fence_value", &self.fence_value)
            .field("release", &self.release.as_raw_handle())
            .field("size", &self.size)
            .finish()
    }
}

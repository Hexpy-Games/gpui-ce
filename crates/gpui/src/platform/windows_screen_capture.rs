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
/// accelerated off-screen paint.
///
/// The producer must create the texture with `D3D11_RESOURCE_MISC_SHARED_NTHANDLE` (no keyed
/// mutex) in `DXGI_FORMAT_B8G8R8A8_UNORM` or `DXGI_FORMAT_B8G8R8A8_TYPELESS`, holding
/// premultiplied-alpha pixels, and share it with a shared `ID3D11Fence`. After writing the frame
/// it signals the fence with `fence_value` and flushes; the renderer makes the GPU wait for that
/// value before it samples the texture. The producer must not write the texture again until the
/// frame is no longer painted.
///
/// A frame is identified by its texture handle and fence value: a new frame must use a new
/// texture handle or a higher fence value, otherwise damage tracking treats it as unchanged.
/// The handles close when the last clone of the frame drops; renderers keep a clone while they
/// cache the opened texture, so a handle value cannot be reused for another texture meanwhile.
#[derive(Clone)]
pub struct WindowsSharedTextureFrame {
    texture: Arc<OwnedHandle>,
    fence: Arc<OwnedHandle>,
    fence_value: u64,
    size: Size<DevicePixels>,
}

impl WindowsSharedTextureFrame {
    /// Creates a frame from NT handles to a shared texture and a shared fence.
    ///
    /// Pass an `Arc` to reuse one handle across frames (for example a fence that every frame
    /// signals, or a texture from a producer's pool) so renderers open it only once. Invalid or
    /// mismatched handles are reported as a draw error by the renderer.
    pub fn new(
        texture: impl Into<Arc<OwnedHandle>>,
        fence: impl Into<Arc<OwnedHandle>>,
        fence_value: u64,
        size: Size<DevicePixels>,
    ) -> Self {
        Self {
            texture: texture.into(),
            fence: fence.into(),
            fence_value,
            size,
        }
    }

    /// Returns the NT handle of the shared texture.
    pub fn texture_handle(&self) -> BorrowedHandle<'_> {
        self.texture.as_handle()
    }

    /// Returns the NT handle of the shared fence.
    pub fn fence_handle(&self) -> BorrowedHandle<'_> {
        self.fence.as_handle()
    }

    /// Returns the fence value the producer signals once the frame is written.
    pub fn fence_value(&self) -> u64 {
        self.fence_value
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
            .field("fence_value", &self.fence_value)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

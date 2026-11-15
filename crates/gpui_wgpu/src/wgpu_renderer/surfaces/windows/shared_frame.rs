// Added by HexpyGames for Butler (2026), under the Apache License 2.0; see NOTICE.
//! Draws `WindowsSharedTextureFrame`s: textures another Direct3D 11 device shares through NT
//! handles, opened on wgpu's Direct3D 12 device. Mirrors the Direct3D 11 renderer's contract
//! (see `WindowsSharedTextureFrame`): the queue waits on the acquire fence before the
//! submission that samples a frame and signals the release fence after it.
use super::{SurfaceBinding, WgpuRenderer, frame};
use anyhow::{Context as _, Result};
use collections::FxHashMap;
use gpui::{DevicePixels, PaintSurface, Size, WindowsSharedTextureFrame};
use gpui_render::shaders::common::SurfaceColorFormat;
use smallvec::SmallVec;
use std::os::windows::io::{AsRawHandle as _, BorrowedHandle};
use windows::Win32::{
    Foundation::{CompareObjectHandles, HANDLE},
    Graphics::{
        Direct3D12::{
            D3D12_RESOURCE_DIMENSION_TEXTURE2D, ID3D12CommandQueue, ID3D12Device, ID3D12Fence,
            ID3D12Resource,
        },
        Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_TYPELESS, DXGI_FORMAT_B8G8R8A8_UNORM},
    },
};

/// The shared surfaces of the window's scene and the release signals of the frame being encoded.
#[derive(Default)]
pub(super) struct SharedSurfaces {
    // Keyed by texture handle.
    surfaces: FxHashMap<usize, SharedSurface>,
    release_fences: ReleaseFences,
    // (texture key, release fence, value) to signal once the frame is submitted.
    pending_releases: Vec<(usize, ID3D12Fence, u64)>,
    // Set while rasterizing a scene other than the window's own, which can only be older.
    pub(super) detached: bool,
}

impl SharedSurfaces {
    /// Evicts the textures and fences of surfaces that left the window's scene.
    pub(super) fn retain(&mut self, surfaces: &[PaintSurface]) {
        let active = shared_frames(surfaces)
            .map(texture_key)
            .collect::<SmallVec<[usize; 4]>>();
        self.surfaces.retain(|key, _| active.contains(key));
        self.release_fences.retain(surfaces);
    }

    pub(super) fn draw(
        &mut self,
        renderer: &WgpuRenderer,
        surface: &PaintSurface,
        frame: &WindowsSharedTextureFrame,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let Some(frame) = self.release_fences.newest(frame, self.detached) else {
            // Only an older scene holds a fence the window has not drawn.
            return Ok(());
        };
        let key = texture_key(&frame);
        let shared = self
            .surfaces
            .entry(key)
            .or_insert_with(|| SharedSurface::new(&frame));
        if shared.acquire(renderer, &frame)
            && let Some(texture) = shared.texture.as_mut()
        {
            let binding = &mut texture.binding;
            renderer.draw_surface_binding(surface, SurfaceColorFormat::Rgba, binding, pass)?;
        }
        // A skipped frame is released too: nothing samples it, so the producer may have it back.
        if let Some(release) = &shared.release {
            self.pending_releases
                .push((key, release.clone(), shared.frame.fence_value()));
        }
        Ok(())
    }

    /// Signals the release fences of the frame just submitted (or abandoned), so the signals
    /// follow every draw that sampled the frames on the queue.
    pub(super) fn signal_releases(&mut self, renderer: &WgpuRenderer) {
        if self.pending_releases.is_empty() {
            return;
        }
        let queue = d3d12(renderer).ok().map(|(_, queue)| queue);
        for (key, fence, value) in self.pending_releases.drain(..) {
            let result = queue
                .as_ref()
                .context("wgpu is not running on Direct3D 12")
                .and_then(|queue| {
                    // SAFETY: the queue and the fence are live COM objects.
                    unsafe { queue.Signal(&fence, value) }
                        .context("signaling the shared surface's release fence")
                });
            if let (Err(error), Some(shared)) = (result, self.surfaces.get_mut(&key)) {
                shared.release = None;
                shared.settle::<()>(Err(error));
            }
        }
    }
}

/// The newest frame drawn with one release fence, which a producer may pass under several
/// handles to the same kernel object (for example a duplicated handle per frame).
struct ReleaseFenceRecord {
    // A frame for each release handle of the window's scene that refers to this fence. Holding
    // the frames keeps the handles open, so their values identify only this fence.
    handles: SmallVec<[WindowsSharedTextureFrame; 1]>,
    newest: WindowsSharedTextureFrame,
}

impl ReleaseFenceRecord {
    fn has_handle(&self, key: usize) -> bool {
        self.handles.iter().any(|frame| release_key(frame) == key)
    }

    fn is_fence_of(&self, frame: &WindowsSharedTextureFrame) -> bool {
        // SAFETY: both frames own their handles and keep them open for this call.
        unsafe {
            CompareObjectHandles(
                HANDLE(self.newest.release_handle().as_raw_handle()),
                HANDLE(frame.release_handle().as_raw_handle()),
            )
        }
        .as_bool()
    }
}

/// The newest frame drawn with each release fence of the window's scene, so the release fence
/// never moves backwards (the Direct3D 11 renderer's `ReleaseFences`).
#[derive(Default)]
struct ReleaseFences {
    records: Vec<ReleaseFenceRecord>,
}

impl ReleaseFences {
    fn retain(&mut self, surfaces: &[PaintSurface]) {
        let active = shared_frames(surfaces).collect::<SmallVec<[_; 4]>>();
        // A duplicated handle joins the record of its kernel object; a handle from a second
        // `CreateSharedHandle` cannot be matched, so its old record is dropped.
        for frame in &active {
            let key = release_key(frame);
            if !self.records.iter().any(|record| record.has_handle(key))
                && let Some(record) = self.records.iter_mut().find(|r| r.is_fence_of(frame))
            {
                record.handles.push((*frame).clone());
            }
        }
        self.records.retain_mut(|record| {
            record
                .handles
                .retain(|held| active.iter().any(|f| release_key(f) == release_key(held)));
            !record.handles.is_empty()
        });
    }

    /// Returns the frame to draw for `frame`: itself, or the newest frame already drawn with the
    /// same release fence when `frame` comes from a stale scene. `None` skips the surface: a
    /// detached scene holding a fence the window has not drawn.
    fn newest(
        &mut self,
        frame: &WindowsSharedTextureFrame,
        detached: bool,
    ) -> Option<WindowsSharedTextureFrame> {
        let key = release_key(frame);
        let found = self
            .records
            .iter()
            .position(|record| record.has_handle(key))
            .or_else(|| self.records.iter().position(|r| r.is_fence_of(frame)));
        let Some(index) = found else {
            if detached {
                return None;
            }
            self.records.push(ReleaseFenceRecord {
                handles: smallvec::smallvec![frame.clone()],
                newest: frame.clone(),
            });
            return Some(frame.clone());
        };
        let record = &mut self.records[index];
        if !detached {
            // Holding the latest frame per handle lets older textures of the producer close.
            match record.handles.iter_mut().find(|h| release_key(h) == key) {
                Some(held) => *held = frame.clone(),
                None => record.handles.push(frame.clone()),
            }
        }
        if record.newest.fence_value() < frame.fence_value() {
            record.newest = frame.clone();
        }
        Some(record.newest.clone())
    }
}

/// A shared texture of the window's scene and its fences, opened on wgpu's D3D12 device.
struct SharedSurface {
    // The latest frame drawn from this texture. Holding it keeps the handles open, so their
    // values cannot be reused for other objects while they key the caches.
    frame: WindowsSharedTextureFrame,
    texture: Option<ImportedTexture>,
    acquire: Option<ID3D12Fence>,
    release: Option<ID3D12Fence>,
    // Importing or using the current handles failed: logged once, and the surface is skipped
    // until a frame brings other fence handles.
    failed: bool,
    // The last frame's size disagreed with the texture and was logged.
    size_mismatch_logged: bool,
}

struct ImportedTexture {
    binding: SurfaceBinding,
    size: Size<DevicePixels>,
    _texture: wgpu::Texture,
}

impl SharedSurface {
    fn new(frame: &WindowsSharedTextureFrame) -> Self {
        Self {
            frame: frame.clone(),
            texture: None,
            acquire: None,
            release: None,
            failed: false,
            size_mismatch_logged: false,
        }
    }

    /// Imports what `frame` needs and makes the queue wait until the producer finished writing
    /// it. Returns whether the texture can be sampled.
    fn acquire(&mut self, renderer: &WgpuRenderer, frame: &WindowsSharedTextureFrame) -> bool {
        if self.frame.fence_handle().as_raw_handle() != frame.fence_handle().as_raw_handle() {
            self.acquire = None;
            self.failed = false;
        }
        if self.frame.release_handle().as_raw_handle() != frame.release_handle().as_raw_handle() {
            self.release = None;
            self.failed = false;
        }
        self.frame = frame.clone();
        if self.failed {
            return false;
        }
        let imported = d3d12(renderer).and_then(|d3d12| {
            let acquire = self.import(renderer, &d3d12.0)?;
            Ok((d3d12.1, acquire))
        });
        let Some((queue, acquire)) = self.settle(imported) else {
            return false;
        };
        // The size is per-frame data: a frame that disagrees with the texture is skipped on its
        // own and the next frame is tried normally.
        let size = self.texture.as_ref().map(|texture| texture.size);
        if size != Some(self.frame.size()) {
            if !self.size_mismatch_logged {
                log::error!(
                    "skipping shared surface frame {:?}: the texture is {size:?}",
                    self.frame
                );
            }
            self.size_mismatch_logged = true;
            return false;
        }
        self.size_mismatch_logged = false;
        // SAFETY: the queue and the fence are live COM objects. The wait is queued before the
        // submission that samples the texture, which follows on the same queue.
        let waited = unsafe { queue.Wait(&acquire, self.frame.fence_value()) }
            .context("waiting for the shared surface's acquire fence");
        self.settle(waited).is_some()
    }

    /// Opens the handles of the current frame that are not open yet; returns the acquire fence.
    fn import(&mut self, renderer: &WgpuRenderer, device: &ID3D12Device) -> Result<ID3D12Fence> {
        // The release fence comes first so that a surface whose texture fails is still released.
        if self.release.is_none() {
            self.release = Some(open_fence(device, self.frame.release_handle())?);
        }
        if self.texture.is_none() {
            self.texture = Some(import_texture(renderer, device, &self.frame)?);
        }
        match &self.acquire {
            Some(acquire) => Ok(acquire.clone()),
            None => Ok(self
                .acquire
                .insert(open_fence(device, self.frame.fence_handle())?)
                .clone()),
        }
    }

    fn settle<T>(&mut self, result: Result<T>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(error) => {
                if !self.failed {
                    log::error!("skipping shared surface {:?}: {error:#}", self.frame);
                }
                self.failed = true;
                None
            }
        }
    }
}

/// wgpu's raw D3D12 device and queue, or an error when wgpu runs on another backend.
fn d3d12(renderer: &WgpuRenderer) -> Result<(ID3D12Device, ID3D12CommandQueue)> {
    // SAFETY: the raw objects are only cloned (AddRef) here; the guard drops at the end.
    let device = unsafe { renderer.resources().device.as_hal::<wgpu::hal::api::Dx12>() }
        .context("wgpu is not running on Direct3D 12")?;
    Ok((device.raw_device().clone(), device.raw_queue().clone()))
}

fn import_texture(
    renderer: &WgpuRenderer,
    device: &ID3D12Device,
    frame: &WindowsSharedTextureFrame,
) -> Result<ImportedTexture> {
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: the frame owns the handle and keeps it open for this call; `resource` is a valid
    // out-pointer.
    unsafe {
        device.OpenSharedHandle(
            HANDLE(frame.texture_handle().as_raw_handle()),
            &raw mut resource,
        )
    }
    .context("opening shared surface texture")?;
    let resource = resource.context("D3D12 returned no shared surface texture")?;
    // SAFETY: the resource is a live COM object.
    let desc = unsafe { resource.GetDesc() };
    anyhow::ensure!(
        desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
            && matches!(
                desc.Format,
                DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_TYPELESS
            ),
        "shared surface texture has unsupported format {:?}",
        desc.Format
    );
    anyhow::ensure!(
        desc.DepthOrArraySize == 1 && desc.SampleDesc.Count == 1,
        "shared surface texture has {} slices and {} samples, expected one of each",
        desc.DepthOrArraySize,
        desc.SampleDesc.Count
    );
    let width = u32::try_from(desc.Width).context("shared surface texture is too wide")?;
    let extent = wgpu::Extent3d {
        width,
        height: desc.Height,
        depth_or_array_layers: 1,
    };
    let mip_level_count = u32::from(desc.MipLevels).max(1);
    // SAFETY: the resource is a single-sample, single-slice 2D BGRA texture of `extent` and
    // `mip_level_count` mips, as checked above.
    let hal_texture = unsafe {
        wgpu::hal::dx12::Device::texture_from_raw(
            resource,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureDimension::D2,
            extent,
            mip_level_count,
            1,
        )
    };
    let device = &renderer.resources().device;
    // SAFETY: the hal texture was created from a resource of this device, and the descriptor
    // matches it; it is only sampled.
    let texture = unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Dx12>(
            hal_texture,
            &wgpu::TextureDescriptor {
                label: Some("windows_shared_texture"),
                size: extent,
                mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Bgra8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )
    };
    // Only mip 0 is shown.
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        mip_level_count: Some(1),
        ..Default::default()
    });
    Ok(ImportedTexture {
        binding: SurfaceBinding::new(renderer, view.clone(), view),
        size: Size {
            width: DevicePixels(extent.width as i32),
            height: DevicePixels(extent.height as i32),
        },
        _texture: texture,
    })
}

fn open_fence(device: &ID3D12Device, handle: BorrowedHandle<'_>) -> Result<ID3D12Fence> {
    let mut fence: Option<ID3D12Fence> = None;
    // SAFETY: the borrowed handle is open for this call; `fence` is a valid out-pointer.
    unsafe { device.OpenSharedHandle(HANDLE(handle.as_raw_handle()), &raw mut fence) }
        .context("opening shared surface fence")?;
    fence.context("D3D12 returned no shared surface fence")
}

fn shared_frames(surfaces: &[PaintSurface]) -> impl Iterator<Item = &WindowsSharedTextureFrame> {
    surfaces.iter().filter_map(|surface| match &surface.source {
        gpui::SurfaceSource::WindowsSharedTexture(frame) => Some(frame),
        _ => None,
    })
}

fn texture_key(frame: &WindowsSharedTextureFrame) -> usize {
    frame.texture_handle().as_raw_handle() as usize
}

fn release_key(frame: &WindowsSharedTextureFrame) -> usize {
    frame.release_handle().as_raw_handle() as usize
}

#[cfg(all(test, feature = "test-support"))]
#[path = "shared_frame_tests.rs"]
mod tests;

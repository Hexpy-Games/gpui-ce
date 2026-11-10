// Modified by HexpyGames for Butler (2026); see NOTICE for the changes.
use crate::{CompositorGpuHint, WgpuAtlas, WgpuContext, WgpuContextHandle, WgpuDeviceRequirements};
use gpui::{Bounds, DevicePixels, GpuSpecs, ScaledPixels, Scene, Size};

#[cfg(test)]
use gpui::BackdropFilter;
#[cfg(all(test, feature = "test-support", not(target_family = "wasm")))]
use gpui::{ColorExt, Quad, Underline, blue, red, white};
#[cfg(test)]
use gpui_render::blur::{BlurKernel, MAX_GAUSSIAN_SAMPLES_PER_SIDE};
#[cfg(test)]
use gpui_render::shaders::interface as shader_interface;
#[cfg(all(test, feature = "test-support", not(target_family = "wasm")))]
use smallvec::{SmallVec, smallvec};

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

mod backdrop_cache;
mod buffers;
mod damage;
mod drawing;
mod filters;
mod frame;
#[cfg(all(feature = "test-support", not(target_family = "wasm")))]
mod headless;
mod path_types;
mod pipelines;
mod platform;
mod resources;
mod settings;
mod surfaces;
mod target;
mod wallpapers;

#[cfg(all(feature = "test-support", not(target_family = "wasm")))]
pub use headless::WgpuHeadlessRenderer;
use pipelines::WgpuPipelines as ShaderPipelines;
use resources::{GlobalBufferLayout, ResourceMetadata, WgpuResources};
use settings::RenderingParameters;
pub use settings::{FontRasterizationSettings, SubpixelOrder, WgpuSurfaceConfig};
use target::RenderTarget;

/// Shared GPU context reference, used to coordinate device recovery across multiple windows.
pub type GpuContext = Rc<RefCell<Option<WgpuContext>>>;

struct GpuFaultState {
    pending_error: Arc<Mutex<Option<String>>>,
    consecutive_failed_frames: u32,
    device_lost: Arc<std::sync::atomic::AtomicBool>,
    #[cfg(not(target_family = "wasm"))]
    recovery_not_before: Option<std::time::Instant>,
}

pub struct WgpuRenderer {
    /// Shared GPU context for device recovery coordination (unused on WASM).
    #[allow(dead_code)]
    context: Option<GpuContext>,
    /// Compositor GPU hint for adapter selection (unused on WASM).
    #[allow(dead_code)]
    compositor_gpu: Option<CompositorGpuHint>,
    /// Application-requested extra wgpu features/limits, stored for device recovery.
    #[allow(dead_code)]
    extra_requirements: Option<WgpuDeviceRequirements>,
    resources: Option<WgpuResources>,
    target: RenderTarget,
    atlas: Arc<WgpuAtlas>,
    globals: GlobalBufferLayout,
    rendering_params: RenderingParameters,
    subpixel_order: SubpixelOrder,
    dual_source_blending: bool,
    adapter_info: wgpu::AdapterInfo,
    uploaded_globals: Option<frame::GlobalUniformState>,
    faults: GpuFaultState,
    /// The animation clock the next frame's shaders read, in milliseconds.
    animation_time_ms: u32,
    /// The scene generation whose complete frame `scene_color` holds, if any.
    retained: Option<u64>,
    retained_atlas_epoch: u64,
    /// The backdrop-cache keys of the last full frame, with that frame's scene generation.
    frame_keys: (u64, Vec<Option<u64>>),
}

impl WgpuRenderer {
    fn resources(&self) -> &WgpuResources {
        self.resources
            .as_ref()
            .expect("GPU resources not available")
    }

    fn resources_mut(&mut self) -> &mut WgpuResources {
        self.resources
            .as_mut()
            .expect("GPU resources not available")
    }

    fn new_internal(
        gpu_context: Option<GpuContext>,
        context: &WgpuContext,
        surface: Option<wgpu::Surface<'static>>,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
        extra_requirements: Option<WgpuDeviceRequirements>,
        atlas: Arc<WgpuAtlas>,
    ) -> anyhow::Result<Self> {
        let target =
            RenderTarget::new(&context.adapter, &context.device, surface.as_ref(), config)?;
        if let Some(surface) = surface.as_ref() {
            surface.configure(&context.device, target.configuration());
        }

        let dual_source_blending = context.supports_dual_source_blending();
        let rendering_params = RenderingParameters::new(&context.adapter, target.format());
        let adapter_info = context.adapter.get_info();
        let (resources, metadata) = WgpuResources::new(
            context,
            surface,
            target.configuration(),
            &rendering_params,
            dual_source_blending,
        )?;
        let ResourceMetadata {
            globals,
            last_error,
        } = metadata;

        Ok(Self {
            context: gpu_context,
            compositor_gpu,
            extra_requirements,
            resources: Some(resources),
            target,
            atlas,
            globals,
            rendering_params,
            subpixel_order: SubpixelOrder::RedGreenBlue,
            dual_source_blending,
            adapter_info,
            uploaded_globals: None,
            animation_time_ms: 0,
            retained: None,
            retained_atlas_epoch: 0,
            frame_keys: (0, Vec::new()),
            faults: GpuFaultState {
                pending_error: last_error,
                consecutive_failed_frames: 0,
                device_lost: context.device_lost_flag(),
                #[cfg(not(target_family = "wasm"))]
                recovery_not_before: None,
            },
        })
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        if !self.target.resize(size) {
            return;
        }
        self.retained = None;
        let config = self.target.configuration().clone();
        let Some(resources) = self.resources.as_mut() else {
            return;
        };
        resources.invalidate_intermediate_textures();
        if let Some(surface) = resources.surface.as_ref() {
            surface.configure(&resources.device, &config);
        }
    }

    /// Selects the physical LCD component order used by subpixel glyph correction.
    pub fn set_subpixel_order(&mut self, order: SubpixelOrder) {
        self.subpixel_order = order;
    }

    /// Updates platform-specific text rasterization parameters.
    pub fn set_font_rasterization_settings(&mut self, settings: FontRasterizationSettings) {
        self.retained = None;
        self.rendering_params.font_rasterization = settings;
        self.subpixel_order = settings.subpixel_order;
    }

    /// Compatibility wrapper for callers that still report the layout as a boolean.
    pub fn set_subpixel_layout(&mut self, is_bgr: bool) {
        self.set_subpixel_order(if is_bgr {
            SubpixelOrder::BlueGreenRed
        } else {
            SubpixelOrder::RedGreenBlue
        });
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        let scene_blend_state = pipelines::scene_blend_state(self.target.alpha_mode());
        if !self.target.set_transparent(transparent) {
            return;
        }
        self.retained = None;
        let config = self.target.configuration().clone();
        if let Some(resources) = self.resources.as_ref() {
            if let Some(surface) = resources.surface.as_ref() {
                surface.configure(&resources.device, &config);
            }
        }
        if pipelines::scene_blend_state(self.target.alpha_mode()) != scene_blend_state {
            self.rebuild_pipelines();
        }
    }

    pub(super) fn rebuild_pipelines(&mut self) {
        let config = self.target.configuration();
        let Some(resources) = self.resources.as_mut() else {
            return;
        };
        resources.pipelines = ShaderPipelines::new(
            &resources.device,
            &resources.bind_group_layouts,
            config.format,
            config.alpha_mode,
            self.rendering_params.path_sample_count,
            self.dual_source_blending,
            resources.renderer_tier,
        );
    }

    #[allow(dead_code)]
    pub fn viewport_size(&self) -> Size<DevicePixels> {
        self.target.viewport_size()
    }

    pub fn sprite_atlas(&self) -> &Arc<WgpuAtlas> {
        &self.atlas
    }

    pub fn supports_dual_source_blending(&self) -> bool {
        self.dual_source_blending
    }

    pub fn gpu_context(&self) -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        let resources = self.resources();
        (resources.device.clone(), resources.queue.clone())
    }

    pub fn gpu_context_info(&self) -> Option<WgpuContextHandle> {
        if let Some(context) = self.context.as_ref().and_then(|context| {
            context
                .borrow()
                .as_ref()
                .map(|context| context.handle(self.target.format()))
        }) {
            return Some(context);
        }
        self.resources.as_ref().map(|resources| {
            WgpuContextHandle::from_resources(
                resources.device.clone(),
                resources.queue.clone(),
                self.target.format(),
                self.adapter_info.clone(),
                match self.adapter_info.backend {
                    wgpu::Backend::BrowserWebGpu => crate::WgpuBackend::BrowserWebGpu,
                    wgpu::Backend::Gl => crate::WgpuBackend::Gl,
                    backend => crate::WgpuBackend::Native(backend),
                },
                self.faults.device_lost.clone(),
            )
        })
    }

    pub fn gpu_specs(&self) -> GpuSpecs {
        GpuSpecs {
            is_software_emulated: self.adapter_info.device_type == wgpu::DeviceType::Cpu,
            device_name: self.adapter_info.name.clone(),
            driver_name: self.adapter_info.driver.clone(),
            driver_info: self.adapter_info.driver_info.clone(),
        }
    }

    pub fn max_texture_size(&self) -> u32 {
        self.target.maximum_dimension()
    }

    /// Sets the animation clock the next frames read; the headless override of the wall clock.
    #[cfg(feature = "test-support")]
    pub fn set_animation_time_ms(&mut self, ms: u32) {
        self.animation_time_ms = ms;
    }

    /// Rasterises a scene region as tightly packed, premultiplied BGRA rows.
    pub fn rasterize(
        &mut self,
        scene: &Scene,
        rect: Bounds<ScaledPixels>,
    ) -> Option<(Size<DevicePixels>, Arc<[u8]>)> {
        let (width, height) = (rect.size.width.0 as u32, rect.size.height.0 as u32);
        let size = gpui::size(DevicePixels(width as i32), DevicePixels(height as i32));
        let device = self.resources().device.clone();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("motion_group"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target.format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let row_bytes = width * 4;
        let padded = row_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion_group_readback"),
            size: u64::from(padded) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let clear = self.target.clear_color;
        self.target.clear_color = wgpu::Color::TRANSPARENT;
        let rendered = frame::render_offscreen(self, scene, &view, size);
        self.target.clear_color = clear;
        if !rendered {
            return None;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let submission = self.resources().queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .ok()?;
        receiver.recv().ok()?.ok()?;
        let mapped = buffer.slice(..).get_mapped_range();
        let mut bytes = Vec::with_capacity(row_bytes as usize * height as usize);
        for row in mapped.chunks_exact(padded as usize) {
            bytes.extend_from_slice(&row[..row_bytes as usize]);
        }
        drop(mapped);
        buffer.unmap();
        if self.target.format() == wgpu::TextureFormat::Rgba8Unorm {
            for pixel in bytes.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        Some((
            gpui::size(DevicePixels(width as i32), DevicePixels(height as i32)),
            bytes.into(),
        ))
    }

    /// Encodes and submits a complete scene into an arbitrary color target.
    /// Swapchain drawing, window capture, and headless rendering all use this path so
    /// platform integrations can't diverge in batching, filters, or pipeline state.
    fn render_to_view(&mut self, scene: &Scene, frame_view: &wgpu::TextureView) -> bool {
        frame::render_to_view(self, scene, frame_view, None, true).is_some()
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    fn render_to_view_with_readback(
        &mut self,
        scene: &Scene,
        frame_view: &wgpu::TextureView,
        readback: frame::ReadbackCopy<'_>,
    ) -> Option<wgpu::SubmissionIndex> {
        frame::render_to_view(self, scene, frame_view, Some(readback), true)
    }
}

fn begin_color_render_pass<'encoder>(
    encoder: &'encoder mut wgpu::CommandEncoder,
    label: &'encoder str,
    target: &'encoder wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    scissor: Option<gpui_render::blur::ScissorRectangle>,
) -> wgpu::RenderPass<'encoder> {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        ..Default::default()
    });
    if let Some(scissor) = scissor {
        pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
    }
    pass
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgpu_renderer::filters::{FILTER_UNIFORMS_PER_COMPOSITE, FrameUniformRequirements};

    // Native shader backends can differ slightly when quantizing transcendental math to UNORM8.
    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    const MAX_CROSS_BACKEND_CHANNEL_DELTA: u8 = 2;

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    fn dashed_border_scene(dash_length: f32, dash_gap: f32) -> Scene {
        let full_bounds = test_bounds(0.0, 0.0, 40.0, 20.0);
        let quad_bounds = |left| test_bounds(left, 4.0, 12.0, 12.0);
        let mut scene = Scene::default();

        for (bounds, corner_smoothing) in [(quad_bounds(4.0), 0.0), (quad_bounds(24.0), 0.6)] {
            scene.insert_primitive(Quad {
                bounds,
                content_mask: gpui::ContentMask {
                    bounds: full_bounds,
                    ..Default::default()
                },
                background: gpui::solid_background(gpui::hsla(0.05, 0.8, 0.45, 1.0)),
                border_style: gpui::BorderStyle::Dashed,
                border_dashed_length: dash_length,
                border_dashed_gap: dash_gap,
                border_color: gpui::hsla(0.6, 0.9, 0.7, 1.0).into(),
                corner_radii: gpui::Corners::all(ScaledPixels(4.0)),
                border_widths: gpui::Edges::all(ScaledPixels(2.0)),
                corner_smoothing,
                ..Default::default()
            });
        }

        scene.finish();

        scene
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    fn images_differ_in_region(
        first: &image::RgbaImage,
        second: &image::RgbaImage,
        left: u32,
        right: u32,
    ) -> bool {
        (4..16).any(|y| (left..right).any(|x| first.get_pixel(x, y) != second.get_pixel(x, y)))
    }

    #[test]
    fn rust_storage_types_match_shader_strides() {
        // Shader generation and Naga parsing need the build script's 64 MiB stack in unoptimized builds.
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                fn module_layout(source: &wgsl_rs::Source) -> (naga::Module, naga::proc::Layouter) {
                    let source = source.wgsl_source().expect("shader must generate WGSL");
                    let module = naga::front::wgsl::parse_str(&format!(
                        "enable dual_source_blending;\n{source}"
                    ))
                    .expect("generated shader must parse as WGSL");
                    let mut layouter = naga::proc::Layouter::default();
                    layouter
                        .update(module.to_ctx())
                        .expect("generated shader types must have valid layouts");
                    (module, layouter)
                }

                let (base, base_layout) = module_layout(&gpui_render::shaders::base::WGSL_SOURCE);
                let (subpixel, subpixel_layout) =
                    module_layout(&gpui_render::shaders::subpixel_sprite::WGSL_SOURCE);

                for abi in shader_interface::SCENE_STORAGE_ABI
                    .iter()
                    .chain(path_types::STORAGE_ABI)
                {
                    let base_handle = base
                        .types
                        .iter()
                        .find(|(_, ty)| ty.name.as_deref() == Some(abi.wgsl_type))
                        .map(|(handle, _)| handle);
                    let shader_stride = if let Some(handle) = base_handle {
                        base_layout[handle].to_stride()
                    } else {
                        let (handle, _) = subpixel
                            .types
                            .iter()
                            .find(|(_, ty)| ty.name.as_deref() == Some(abi.wgsl_type))
                            .unwrap_or_else(|| panic!("missing WGSL type {}", abi.wgsl_type));
                        subpixel_layout[handle].to_stride()
                    };
                    assert_eq!(
                        shader_stride as usize, abi.rust_stride,
                        "Rust and WGSL layouts differ for {}",
                        abi.wgsl_type,
                    );
                }
            })
            .expect("spawn shader stride validation")
            .join()
            .expect("shader stride validation panicked");
    }

    #[test]
    fn large_filter_scenes_reserve_unique_uniform_slots() {
        let mut scene = Scene::default();
        scene
            .backdrop_filters
            .resize_with(100, BackdropFilter::default);
        scene.finish();

        assert_eq!(
            frame::FrameRequirements::for_commands(
                scene.render_commands(),
                scene.render_plan().requirements(),
                super::buffers::InstanceTransport::StorageBuffer
            )
            .uniforms,
            FrameUniformRequirements {
                filter_count: FILTER_UNIFORMS_PER_COMPOSITE * 100 + 1,
                surface_count: 0,
            }
        );
    }

    #[test]
    fn large_blur_radii_expand_tap_spacing_instead_of_truncating_the_kernel() {
        let kernel = BlurKernel::for_radius(100.0).expect("positive radii have a blur kernel");
        assert_eq!(kernel.standard_deviation, 50.0);
        assert_eq!(kernel.sample_count, MAX_GAUSSIAN_SAMPLES_PER_SIDE);
        assert!(kernel.sample_step > 4.0);
    }

    #[cfg(any(target_os = "macos", target_os = "windows"))]
    #[test]
    fn desktop_blending_preserves_source_over_alpha() {
        let straight = pipelines::desktop_scene_blend_state(wgpu::CompositeAlphaMode::Opaque);
        assert_eq!(straight.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(
            straight.alpha.dst_factor,
            wgpu::BlendFactor::OneMinusSrcAlpha
        );

        let premultiplied =
            pipelines::desktop_scene_blend_state(wgpu::CompositeAlphaMode::PreMultiplied);
        assert_eq!(premultiplied.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(
            premultiplied.alpha.dst_factor,
            wgpu::BlendFactor::OneMinusSrcAlpha
        );
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    #[test]
    fn underline_opacity_is_applied_once() -> anyhow::Result<()> {
        let context = WgpuContext::new_headless(None)?;
        let size = gpui::size(DevicePixels(3), DevicePixels(3));
        let mut renderer = WgpuRenderer::new_headless(&context, size)?;
        let bounds = test_bounds(0.0, 0.0, 3.0, 3.0);
        let mut scene = Scene::default();
        // Forces a non-zero first-instance index in the mixed-type arena.
        scene.insert_primitive(Quad::default());
        scene.insert_primitive(Underline {
            order: 0,
            padding: 0,
            bounds,
            content_mask: gpui::ContentMask {
                bounds,
                ..Default::default()
            },
            color: gpui::hsla(0.0, 1.0, 0.5, 0.5).into(),
            thickness: ScaledPixels(1.0),
            wavy: false.into(),
        });
        scene.finish();

        let image = renderer.render_to_image(&scene)?;
        let center = image.get_pixel(1, 1).0;
        assert_eq!(
            center,
            [128, 0, 0, 255],
            "must match the retired Metal renderer's mixed primitive baseline"
        );
        Ok(())
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    #[test]
    fn configurable_dashes_reach_both_wgpu_quad_pipelines() -> anyhow::Result<()> {
        let context = WgpuContext::new_headless(None)?;
        let size = gpui::size(DevicePixels(40), DevicePixels(20));
        let mut renderer = WgpuRenderer::new_headless(&context, size)?;
        let default_image = renderer.render_to_image(&dashed_border_scene(2.0, 1.0))?;
        let custom_image = renderer.render_to_image(&dashed_border_scene(4.0, 0.5))?;

        assert!(
            images_differ_in_region(&default_image, &custom_image, 4, 16),
            "custom dash length and gap must change the ordinary quad pipeline"
        );
        assert!(
            images_differ_in_region(&default_image, &custom_image, 24, 36),
            "custom dash length and gap must change the smoothed quad pipeline"
        );

        Ok(())
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    #[test]
    fn quad_backgrounds_match_legacy_metal_pixels() -> anyhow::Result<()> {
        const LEGACY: &[u8] = &[
            255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 0, 255, 38, 110, 217, 255, 255, 0, 0, 255, 255,
            0, 0, 255, 38, 110, 217, 255, 0, 0, 0, 255, 10, 34, 4, 255, 57, 196, 22, 255, 193, 114,
            152, 255, 223, 155, 105, 255, 57, 196, 22, 255, 10, 34, 4, 255, 165, 60, 182, 255, 193,
            114, 151, 255,
        ];
        let bounds = |x, y| test_bounds(x, y, 2.0, 2.0);
        let mut scene = Scene::default();
        for (order, (bounds, background)) in [
            (
                bounds(0.0, 0.0),
                gpui::solid_background(gpui::hsla(0.0, 1.0, 0.5, 1.0)),
            ),
            (
                bounds(2.0, 0.0),
                gpui::checkerboard(gpui::hsla(0.6, 0.7, 0.5, 1.0), 1.0),
            ),
            (
                bounds(0.0, 2.0),
                gpui::pattern_slash(gpui::hsla(0.3, 0.8, 0.5, 1.0), 1.0, 1.0),
            ),
            (
                bounds(2.0, 2.0),
                gpui::linear_gradient(
                    45.0,
                    gpui::linear_color_stop(gpui::hsla(0.8, 0.9, 0.4, 1.0), 0.0),
                    gpui::linear_color_stop(gpui::hsla(0.1, 0.8, 0.6, 1.0), 1.0),
                ),
            ),
        ]
        .into_iter()
        .enumerate()
        {
            scene.insert_primitive(Quad {
                order: order as u32,
                bounds,
                content_mask: gpui::ContentMask {
                    bounds,
                    ..Default::default()
                },
                background,
                ..Default::default()
            });
        }
        scene.finish();
        let context = WgpuContext::new_headless(None)?;
        let mut renderer =
            WgpuRenderer::new_headless(&context, gpui::size(DevicePixels(4), DevicePixels(4)))?;
        let actual = renderer.render_to_image(&scene)?;
        for (index, (actual, expected)) in actual.as_raw().iter().zip(LEGACY).enumerate() {
            assert!(
                actual.abs_diff(*expected) <= MAX_CROSS_BACKEND_CHANNEL_DELTA,
                "legacy mismatch at pixel ({}, {}), channel {}: got {}, expected {}",
                (index / 4) % 4,
                (index / 4) / 4,
                index % 4,
                actual,
                expected,
            );
        }
        assert_eq!(actual.as_raw().len(), LEGACY.len());
        Ok(())
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    #[test]
    fn quad_border_backgrounds_use_the_fill_coordinate_space() -> anyhow::Result<()> {
        let context = WgpuContext::new_headless(None)?;
        let size = gpui::size(DevicePixels(12), DevicePixels(12));
        let bounds = test_bounds(0.0, 0.0, 12.0, 12.0);
        let render = |quad| -> anyhow::Result<image::RgbaImage> {
            let mut scene = Scene::default();
            scene.insert_primitive(quad);
            scene.finish();
            let mut renderer = WgpuRenderer::new_headless(&context, size)?;
            renderer.render_to_image(&scene)
        };
        let backgrounds = [
            gpui::solid_background(gpui::hsla(0.0, 1.0, 0.5, 1.0)),
            gpui::checkerboard(gpui::hsla(0.6, 0.7, 0.5, 1.0), 2.0),
            gpui::pattern_slash(gpui::hsla(0.3, 0.8, 0.5, 1.0), 2.0, 2.0),
            gpui::linear_gradient(
                90.0,
                gpui::linear_color_stop(gpui::hsla(0.8, 0.9, 0.4, 1.0), 0.0),
                gpui::linear_color_stop(gpui::hsla(0.1, 0.8, 0.6, 1.0), 1.0),
            ),
        ];
        let border_samples = [
            (1, 1),
            (6, 1),
            (10, 1),
            (1, 6),
            (10, 6),
            (1, 10),
            (6, 10),
            (10, 10),
        ];

        for background in backgrounds {
            let filled = render(Quad {
                bounds,
                content_mask: gpui::ContentMask {
                    bounds,
                    ..Default::default()
                },
                background,
                ..Default::default()
            })?;
            let bordered = render(Quad {
                bounds,
                content_mask: gpui::ContentMask {
                    bounds,
                    ..Default::default()
                },
                border_color: background,
                border_widths: gpui::Edges::all(ScaledPixels(4.0)),
                ..Default::default()
            })?;

            for (x, y) in border_samples {
                let border_pixel = bordered.get_pixel(x, y).0;
                let fill_pixel = filled.get_pixel(x, y).0;
                for (channel, (border, fill)) in
                    border_pixel.iter().zip(fill_pixel.iter()).enumerate()
                {
                    assert!(
                        border.abs_diff(*fill) <= MAX_CROSS_BACKEND_CHANNEL_DELTA,
                        "border background must sample like a fill at ({x}, {y}), channel {channel}, for {background:?}: got {border_pixel:?}, expected {fill_pixel:?}"
                    );
                }
            }
            assert_eq!(
                bordered.get_pixel(6, 6).0,
                [0, 0, 0, 255],
                "border background must not paint the quad interior for {background:?}"
            );
        }
        Ok(())
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    #[test]
    fn cached_filter_bindings_accept_frame_local_dynamic_offsets() -> anyhow::Result<()> {
        let context = WgpuContext::new_headless(None)?;
        let size = gpui::size(DevicePixels(4), DevicePixels(4));
        let mut renderer = WgpuRenderer::new_headless(&context, size)?;
        let bounds = test_bounds(0.0, 0.0, 4.0, 4.0);
        let mut scene = Scene::default();
        scene.insert_primitive(BackdropFilter {
            bounds,
            content_mask: gpui::ContentMask {
                bounds,
                ..Default::default()
            },
            filters: smallvec::smallvec![gpui::ScaledFilter::Blur(ScaledPixels(1.0))],
            opacity: 1.0,
            ..BackdropFilter::default()
        });
        scene.finish();

        renderer.render_to_image(&scene)?;
        renderer.render_to_image(&scene)?;
        Ok(())
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    #[test]
    fn odd_sized_blur_preserves_edge_symmetry() -> anyhow::Result<()> {
        let context = WgpuContext::new_headless(None)?;
        let size = gpui::size(DevicePixels(5), DevicePixels(5));
        let mut renderer = WgpuRenderer::new_headless(&context, size)?;
        let full_bounds = test_bounds(0.0, 0.0, 5.0, 5.0);
        let center_bounds = test_bounds(2.0, 2.0, 1.0, 1.0);
        let mut scene = Scene::default();
        scene.insert_primitive(Quad {
            order: 0,
            bounds: full_bounds,
            content_mask: gpui::ContentMask {
                bounds: full_bounds,
                ..Default::default()
            },
            background: gpui::solid_background(gpui::hsla(0.0, 0.0, 0.0, 1.0)),
            ..Default::default()
        });
        scene.insert_primitive(Quad {
            order: 1,
            bounds: center_bounds,
            content_mask: gpui::ContentMask {
                bounds: center_bounds,
                ..Default::default()
            },
            background: gpui::solid_background(gpui::hsla(0.0, 0.0, 1.0, 1.0)),
            ..Default::default()
        });
        scene.insert_primitive(BackdropFilter {
            order: 2,
            bounds: full_bounds,
            content_mask: gpui::ContentMask {
                bounds: full_bounds,
                ..Default::default()
            },
            filters: smallvec::smallvec![gpui::ScaledFilter::Blur(ScaledPixels(2.0))],
            opacity: 1.0,
            ..Default::default()
        });
        scene.finish();

        let image = renderer.render_to_image(&scene)?;
        for y in 0..5 {
            for x in 0..5 {
                let actual = image.get_pixel(x, y).0;
                let horizontal = image.get_pixel(4 - x, y).0;
                let vertical = image.get_pixel(x, 4 - y).0;
                for channel in 0..4 {
                    assert!(
                        actual[channel].abs_diff(horizontal[channel]) <= 1,
                        "horizontal blur asymmetry at ({x}, {y}), channel {channel}: {actual:?} vs {horizontal:?}"
                    );
                    assert!(
                        actual[channel].abs_diff(vertical[channel]) <= 1,
                        "vertical blur asymmetry at ({x}, {y}), channel {channel}: {actual:?} vs {vertical:?}"
                    );
                }
            }
        }
        assert!(
            image.get_pixel(1, 2).0[0] > 0,
            "blur must spread from the center pixel"
        );
        Ok(())
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    #[test]
    fn alternating_instance_batches_fit_the_exact_upload_arena() -> anyhow::Result<()> {
        let context = WgpuContext::new_headless(None)?;
        let size = gpui::size(DevicePixels(1), DevicePixels(1));
        let mut renderer = WgpuRenderer::new_headless(&context, size)?;
        let bounds = test_bounds(0.0, 0.0, 1.0, 1.0);
        let mut scene = Scene::default();
        for index in 0..512 {
            scene.insert_primitive(Quad {
                order: index * 2,
                bounds,
                content_mask: gpui::ContentMask {
                    bounds,
                    ..Default::default()
                },
                background: gpui::solid_background(gpui::hsla(0.0, 0.0, 0.0, 1.0)),
                ..Default::default()
            });
            scene.insert_primitive(Underline {
                order: index * 2 + 1,
                padding: 0,
                bounds,
                content_mask: gpui::ContentMask {
                    bounds,
                    ..Default::default()
                },
                color: gpui::hsla(0.0, 0.0, 1.0, 1.0).into(),
                thickness: ScaledPixels(1.0),
                wavy: false.into(),
            });
        }
        scene.finish();
        assert_eq!(
            scene
                .render_commands()
                .iter()
                .filter(|command| matches!(command, gpui::RenderCommand::Batch(_)))
                .count(),
            1024
        );

        renderer.render_to_image(&scene)?;
        Ok(())
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    fn test_bounds(left: f32, top: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            gpui::point(ScaledPixels(left), ScaledPixels(top)),
            gpui::size(ScaledPixels(width), ScaledPixels(height)),
        )
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    fn test_quad(
        bounds: Bounds<ScaledPixels>,
        mask: Bounds<ScaledPixels>,
        color: gpui::Hsla,
    ) -> Quad {
        Quad {
            bounds,
            content_mask: gpui::ContentMask {
                bounds: mask,
                ..Default::default()
            },
            background: gpui::solid_background(color),
            ..Default::default()
        }
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    fn test_group(
        scene: &mut Scene,
        bounds: Bounds<ScaledPixels>,
        opacity: f32,
        filters: SmallVec<[gpui::ScaledFilter; 4]>,
        paint: impl FnOnce(&mut Scene),
    ) {
        let boundary = gpui::FilterBoundary {
            order: 0,
            bounds,
            content_mask: gpui::ContentMask {
                bounds,
                ..Default::default()
            },
            corner_radii: gpui::Corners::default(),
            corner_smoothing: 0.0,
            filters,
            opacity,
            scale: 1.0,
            scale_origin: gpui::Point::default(),
            motion: gpui::LayerMotion::NONE,
            motion_bounds: Default::default(),
            is_start: true,
            inline: false,
        };
        scene.insert_primitive(boundary.clone());
        paint(scene);
        scene.insert_primitive(gpui::FilterBoundary {
            is_start: false,
            ..boundary
        });
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    fn test_renderers() -> anyhow::Result<Vec<Box<dyn gpui::PlatformHeadlessRenderer>>> {
        let mut renderers: Vec<Box<dyn gpui::PlatformHeadlessRenderer>> =
            vec![Box::new(WgpuHeadlessRenderer::new()?)];

        if let Some(native) = gpui_platform::current_headless_renderer() {
            renderers.push(native);
        }

        Ok(renderers)
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    #[test]
    fn rasterized_group_matches_its_region() -> anyhow::Result<()> {
        use gpui::PlatformHeadlessRenderer;

        let viewport = test_bounds(0.0, 0.0, 96.0, 80.0);
        let bounds = test_bounds(17.0, 13.0, 40.0, 32.0);
        let boundary = gpui::FilterBoundary {
            order: 0,
            bounds,
            content_mask: gpui::ContentMask {
                bounds: viewport,
                ..Default::default()
            },
            corner_radii: Default::default(),
            corner_smoothing: 0.0,
            filters: smallvec![],
            opacity: 1.0,
            scale: 1.0,
            scale_origin: bounds.center(),
            motion: gpui::LayerMotion {
                duration_ms: 1000,
                repeat: gpui::MotionRepeat::Loop,
                opacity: (0.5, 1.0),
                ..gpui::LayerMotion::NONE
            },
            motion_bounds: bounds,
            is_start: true,
            inline: false,
        };
        let quads = [
            Quad {
                border_widths: gpui::Edges::all(ScaledPixels(3.0)),
                border_color: gpui::hsla(0.05, 0.8, 0.5, 0.8).into(),
                corner_radii: gpui::Corners::all(ScaledPixels(6.0)),
                ..test_quad(bounds, viewport, gpui::hsla(0.6, 0.7, 0.4, 0.6))
            },
            test_quad(
                test_bounds(27.0, 21.0, 14.0, 12.0),
                viewport,
                gpui::hsla(0.3, 0.8, 0.6, 0.7),
            ),
        ];
        let mut grouped = Scene::default();
        grouped.offload_animations = true;
        grouped.scale_factor = 1.0;
        let mut plain = Scene::default();
        grouped.insert_primitive(boundary.clone());
        for quad in quads {
            grouped.insert_primitive(quad);
            plain.insert_primitive(quad);
        }
        grouped.insert_primitive(gpui::FilterBoundary {
            is_start: false,
            ..boundary
        });
        grouped.finish();
        plain.finish();

        let group = &grouped.compositor_groups()[0];
        let target_size = gpui::size(DevicePixels(96), DevicePixels(80));
        let context = WgpuContext::new_headless(None)?;
        let mut renderer = WgpuRenderer::new_headless(&context, target_size)?;
        let (size, raster) = renderer
            .rasterize(&group.scene, group.rect)
            .ok_or_else(|| anyhow::anyhow!("group rasterization failed"))?;
        assert_eq!(size.width.0 as f32, group.rect.size.width.0);
        assert_eq!(size.height.0 as f32, group.rect.size.height.0);

        let mut renderer = WgpuHeadlessRenderer::new()?;
        let image = renderer.render_scene_to_image(&plain, target_size)?;
        let crop = image::imageops::crop_imm(
            &image,
            group.rect.origin.x.0 as u32,
            group.rect.origin.y.0 as u32,
            size.width.0 as u32,
            size.height.0 as u32,
        )
        .to_image();
        for (bgra, rgba) in raster.chunks_exact(4).zip(crop.pixels()) {
            let rgb = [bgra[2], bgra[1], bgra[0]];
            for (actual, expected) in rgb.into_iter().zip(rgba.0) {
                assert!(actual.abs_diff(expected) <= 1, "{rgb:?} != {rgba:?}");
            }
        }
        Ok(())
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    #[test]
    fn motion_rounded_clip_follows_source_offset() -> anyhow::Result<()> {
        use gpui::PlatformHeadlessRenderer;

        let viewport = test_bounds(0.0, 0.0, 96.0, 80.0);
        let motion_bounds = test_bounds(12.0, 14.0, 48.0, 40.0);
        let now = 500;
        let motion = gpui::LayerMotion {
            duration_ms: 1000,
            easing: gpui::MotionEasing::linear(&[(0.0, 0.0), (1.0, 1.0)]),
            offset: (
                gpui::point(gpui::px(8.0), gpui::px(4.0)),
                gpui::point(gpui::px(16.0), gpui::px(12.0)),
            ),
            clip: Some((
                gpui::Edges {
                    right: gpui::px(20.0),
                    ..Default::default()
                },
                gpui::Edges::default(),
            )),
            clip_radii: Some((gpui::Corners::default(), gpui::Corners::all(gpui::px(16.0)))),
            ..gpui::LayerMotion::NONE
        };
        let boundary = gpui::FilterBoundary {
            order: 0,
            bounds: viewport,
            content_mask: gpui::ContentMask {
                bounds: viewport,
                ..Default::default()
            },
            corner_radii: gpui::Corners::default(),
            corner_smoothing: 0.0,
            filters: smallvec![],
            opacity: 1.0,
            scale: 1.0,
            scale_origin: motion_bounds.center(),
            motion,
            motion_bounds,
            is_start: true,
            inline: false,
        };
        let mut grouped = Scene::default();
        grouped.scale_factor = 1.0;
        grouped.insert_primitive(boundary.clone());
        grouped.insert_primitive(test_quad(motion_bounds, viewport, white()));
        grouped.insert_primitive(gpui::FilterBoundary {
            is_start: false,
            ..boundary
        });
        grouped.finish();

        let sample = motion.sample(now, 1.0, motion_bounds);
        let (mut clip, radii) = sample.clip.unwrap();
        clip.origin += sample.offset;
        let mut reference = Scene::default();
        reference.insert_primitive(Quad {
            corner_radii: radii,
            border_color: gpui::transparent_black().into(),
            ..test_quad(clip, viewport, white())
        });
        reference.finish();

        let target_size = gpui::size(DevicePixels(96), DevicePixels(80));
        let mut renderer = WgpuHeadlessRenderer::new()?;
        renderer.set_animation_time_ms(now);
        let actual = renderer.render_scene_to_image(&grouped, target_size)?;
        let expected = renderer.render_scene_to_image(&reference, target_size)?;
        for (x, y, expected) in expected.enumerate_pixels() {
            let actual = actual.get_pixel(x, y);
            let tolerance = if expected[0] > 0 && expected[0] < 255 {
                2
            } else {
                0
            };
            assert!(
                actual
                    .0
                    .iter()
                    .zip(expected.0)
                    .all(|(a, b)| a.abs_diff(b) <= tolerance),
                "pixel ({x}, {y}): {actual:?}, expected {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    #[test]
    fn box_morph_matches_plain_geometry() -> anyhow::Result<()> {
        use gpui::PlatformHeadlessRenderer;

        let viewport = test_bounds(0.0, 0.0, 128.0, 112.0);
        let bounds = test_bounds(40.0, 32.0, 64.0, 56.0);
        let start = gpui::animation_time_ms();
        let morph = std::sync::Arc::new(gpui::BoxMorph {
            delta: gpui::Edges {
                left: ScaledPixels(-24.0),
                top: ScaledPixels(-16.0),
                right: ScaledPixels(-40.0),
                bottom: ScaledPixels(-32.0),
            },
            radii_delta: gpui::Corners::all(ScaledPixels(-8.0)),
            timing: gpui::LayerMotion {
                start_ms: start,
                duration_ms: 1000,
                easing: gpui::MotionEasing::linear(&[(0.0, 0.0), (1.0, 1.0)]),
                ..gpui::LayerMotion::NONE
            },
        });
        let quad = Quad {
            corner_radii: gpui::Corners::all(ScaledPixels(12.0)),
            border_widths: gpui::Edges::all(ScaledPixels(3.0)),
            border_color: red().into(),
            ..test_quad(bounds, viewport, white())
        };
        let shadow = gpui::Shadow {
            order: 0,
            blur_radius: ScaledPixels(3.0),
            bounds: (bounds + gpui::point(ScaledPixels(4.0), ScaledPixels(3.0)))
                .dilate(ScaledPixels(2.0)),
            corner_radii: quad.corner_radii,
            content_mask: quad.content_mask,
            color: gpui::black().opacity(0.5).into(),
            element_bounds: bounds,
            element_corner_radii: quad.corner_radii,
            inset: false.into(),
            corner_smoothing: 0.0,
            clip_outside: true.into(),
            padding: 0,
        };
        let mut morphed = Scene::default();
        morphed.insert_primitive(gpui::PaintOperation::Morphed(shadow.into(), morph.clone()));
        morphed.insert_primitive(gpui::PaintOperation::Morphed(quad.into(), morph));
        morphed.finish();

        let middle = test_bounds(28.0, 24.0, 56.0, 48.0);
        let radii = gpui::Corners::all(ScaledPixels(8.0));
        let mut reference = Scene::default();
        reference.insert_primitive(gpui::Shadow {
            bounds: (middle + gpui::point(ScaledPixels(4.0), ScaledPixels(3.0)))
                .dilate(ScaledPixels(2.0)),
            corner_radii: radii,
            element_bounds: middle,
            element_corner_radii: radii,
            ..shadow
        });
        reference.insert_primitive(Quad {
            bounds: middle,
            corner_radii: radii,
            ..quad
        });
        reference.finish();

        let target = gpui::size(DevicePixels(128), DevicePixels(112));
        let mut renderer = WgpuHeadlessRenderer::new()?;
        renderer.set_animation_time_ms(start.wrapping_add(500));
        let actual = renderer.render_scene_to_image(&morphed, target)?;
        let expected = renderer.render_scene_to_image(&reference, target)?;
        for (x, y, expected) in expected.enumerate_pixels() {
            let actual = actual.get_pixel(x, y);
            let tolerance = if expected[0] > 0 && expected[0] < 255 {
                2
            } else {
                0
            };
            assert!(
                actual
                    .0
                    .iter()
                    .zip(expected.0)
                    .all(|(a, b)| a.abs_diff(b) <= tolerance),
                "pixel ({x}, {y}): {actual:?}, expected {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    #[test]
    fn opacity_groups_preserve_overlap_alpha_and_deep_nesting() -> anyhow::Result<()> {
        let bounds = test_bounds(0.0, 0.0, 16.0, 8.0);
        let target_size = gpui::size(DevicePixels(16), DevicePixels(8));
        let green = gpui::hsla(1.0 / 3.0, 1.0, 0.5, 1.0);
        let mut slider = Scene::default();
        test_group(&mut slider, bounds, 0.5, smallvec![], |scene| {
            scene.insert_primitive(test_quad(test_bounds(0.0, 3.0, 8.0, 2.0), bounds, red()));
            scene.insert_primitive(test_quad(test_bounds(4.0, 1.0, 4.0, 6.0), bounds, white()));
        });

        let mut translucent = Scene::default();
        translucent.insert_primitive(test_quad(bounds, bounds, green));
        test_group(&mut translucent, bounds, 0.5, smallvec![], |scene| {
            scene.insert_primitive(test_quad(bounds, bounds, red().opacity(0.5)));
            scene.insert_primitive(test_quad(bounds, bounds, blue().opacity(0.5)));
        });

        let mut nested = Scene::default();
        test_group(&mut nested, bounds, 0.5, smallvec![], |scene| {
            scene.insert_primitive(test_quad(bounds, bounds, red()));
            test_group(scene, bounds, 0.5, smallvec![], |scene| {
                scene.insert_primitive(test_quad(bounds, bounds, blue()));
                test_group(scene, bounds, 0.5, smallvec![], |scene| {
                    scene.insert_primitive(test_quad(bounds, bounds, white()));
                });
            });
        });
        nested.insert_primitive(test_quad(test_bounds(12.0, 0.0, 4.0, 8.0), bounds, green));

        for scene in [&mut slider, &mut translucent, &mut nested] {
            scene.finish();
        }

        let cases = [
            (
                "overlap",
                &slider,
                &[
                    (6, 4, [128, 128, 128, 255]),
                    (1, 4, [128, 0, 0, 255]),
                    (6, 1, [128, 128, 128, 255]),
                    (8, 4, [0, 0, 0, 255]),
                ][..],
            ),
            (
                "intrinsic alpha",
                &translucent,
                &[(6, 4, [32, 159, 64, 255])][..],
            ),
            (
                "nesting",
                &nested,
                &[(6, 4, [96, 32, 64, 255]), (14, 4, [0, 255, 0, 255])][..],
            ),
        ];

        for (renderer_idx, renderer) in test_renderers()?.iter_mut().enumerate() {
            for (name, scene, samples) in cases {
                let image = renderer.render_scene_to_image(scene, target_size)?;

                for &(x, y, expected) in samples {
                    let actual = image.get_pixel(x, y).0;
                    assert!(
                        actual
                            .iter()
                            .zip(expected)
                            .all(|(actual, expected)| actual.abs_diff(expected) <= 2),
                        "renderer {renderer_idx}, {name}, pixel ({x}, {y}): {actual:?}, expected {expected:?}"
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(all(feature = "test-support", not(target_family = "wasm")))]
    #[test]
    fn opacity_groups_preserve_blur_and_inherited_backdrops() -> anyhow::Result<()> {
        let bounds = test_bounds(0.0, 0.0, 16.0, 8.0);
        let target_size = gpui::size(DevicePixels(16), DevicePixels(8));
        let green = gpui::hsla(1.0 / 3.0, 1.0, 0.5, 1.0);
        let blur = smallvec![gpui::ScaledFilter::Blur(ScaledPixels(2.0))];
        let backdrop = BackdropFilter {
            bounds,
            content_mask: gpui::ContentMask {
                bounds,
                ..Default::default()
            },
            filters: blur.clone(),
            opacity: 1.0,
            ..Default::default()
        };
        let paint_background = |scene: &mut Scene| {
            scene.insert_primitive(test_quad(bounds, bounds, red()));
            scene.insert_primitive(test_quad(test_bounds(8.0, 0.0, 8.0, 8.0), bounds, blue()));
        };

        let mut original = Scene::default();
        paint_background(&mut original);
        original.finish();

        for (renderer_idx, renderer) in test_renderers()?.iter_mut().enumerate() {
            let background = renderer.render_scene_to_image(&original, target_size)?;

            for (name, depth, with_local_quad, content_blur) in [
                ("inherited backdrop", 2, false, false),
                ("local backdrop", 1, true, false),
                ("content blur", 1, false, true),
            ] {
                let paint = |scene: &mut Scene, blur_enabled: bool| {
                    if with_local_quad {
                        scene.insert_primitive(test_quad(
                            test_bounds(6.0, 0.0, 4.0, 8.0),
                            bounds,
                            green,
                        ));
                    }

                    if content_blur {
                        let filters = if blur_enabled {
                            blur.clone()
                        } else {
                            smallvec![]
                        };
                        test_group(scene, bounds, 1.0, filters, |scene| {
                            scene.insert_primitive(test_quad(
                                test_bounds(6.0, 2.0, 4.0, 4.0),
                                bounds,
                                white(),
                            ));
                        });
                    } else if blur_enabled {
                        scene.insert_primitive(backdrop.clone());
                    }
                };

                let mut reference = Scene::default();
                let mut grouped = Scene::default();
                let mut unfiltered = Scene::default();

                if !content_blur {
                    paint_background(&mut reference);
                    paint_background(&mut grouped);
                    paint_background(&mut unfiltered);
                }

                paint(&mut reference, true);
                paint(&mut unfiltered, false);
                test_group(&mut grouped, bounds, 0.5, smallvec![], |scene| {
                    if depth == 2 {
                        test_group(scene, bounds, 0.5, smallvec![], |scene| paint(scene, true));
                    } else {
                        paint(scene, true);
                    }
                });

                reference.finish();
                grouped.finish();
                unfiltered.finish();

                let reference = renderer.render_scene_to_image(&reference, target_size)?;
                let actual = renderer.render_scene_to_image(&grouped, target_size)?;
                let unfiltered = renderer.render_scene_to_image(&unfiltered, target_size)?;
                let opacity = 0.5_f32.powi(depth);

                assert!(
                    reference
                        .pixels()
                        .zip(unfiltered.pixels())
                        .any(|(blurred, plain)| {
                            (0..3).any(|channel_idx| {
                                blurred[channel_idx].abs_diff(plain[channel_idx]) > 8
                            })
                        }),
                    "renderer {renderer_idx}, {name}: blur must visibly change the unfiltered scene"
                );

                if content_blur {
                    let falloff = reference.get_pixel(5, 4).0;
                    assert!(
                        falloff[0] > 8 && falloff[1] > 8 && falloff[2] > 8,
                        "renderer {renderer_idx}: content blur must spread outside the white quad: {falloff:?}"
                    );
                }

                for (pixel_idx, ((actual, reference), background)) in actual
                    .pixels()
                    .zip(reference.pixels())
                    .zip(background.pixels())
                    .enumerate()
                {
                    for channel_idx in 0..3 {
                        let original = if content_blur {
                            0.0
                        } else {
                            background[channel_idx] as f32
                        };
                        let expected = (reference[channel_idx] as f32 * opacity
                            + original * (1.0 - opacity))
                            .round() as u8;
                        assert!(
                            actual[channel_idx].abs_diff(expected) <= 3,
                            "renderer {renderer_idx}, {name}, pixel {pixel_idx}, channel {channel_idx}: {}, expected {expected}",
                            actual[channel_idx]
                        );
                    }
                }
            }
        }

        Ok(())
    }
}

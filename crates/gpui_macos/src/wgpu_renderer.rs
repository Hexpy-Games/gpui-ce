//! The WGPU renderer behind the Metal renderer's interface, for the `wgpu`
//! feature. Applications can then share the window's device through
//! `gpui_wgpu::WgpuContextHandle` and composite their own textures.

mod compositor;

use std::{collections::HashMap, sync::Arc};

use compositor::CompositorLayers;
use foreign_types::ForeignType as _;
use gpui::{
    Bounds, CompositorSprite, DevicePixels, FrameLoop, GpuSpecs, LiveCanvas, ScaledPixels, Scene,
    Size, SpriteContents, animation_time_ms, point, px,
};
use gpui_apple::metal_renderer::new_window_layer;
use gpui_wgpu::{
    GpuContext, WgpuAtlas, WgpuContextHandle, WgpuDeviceRequirements, WgpuRenderer,
    WgpuSurfaceConfig, wgpu,
};
use metal::{CAMetalLayer, MetalLayer, MetalLayerRef};
use objc2::{
    class, msg_send,
    rc::Retained,
    runtime::{AnyObject, Bool},
};
use objc2_foundation::{NSPoint, NSRect, NSSize};

struct LiveSurface {
    surface: wgpu::Surface<'static>,
    layer: MetalLayer,
    clip: compositor::MainThread<Retained<AnyObject>>,
    size: Size<DevicePixels>,
}

/// The device every window shares, and what the application requires of it.
#[derive(Clone, Default)]
pub struct Context {
    gpu: GpuContext,
    requirements: Option<WgpuDeviceRequirements>,
}

impl Context {
    pub fn set_requirements(&mut self, requirements: WgpuDeviceRequirements) {
        self.requirements = Some(requirements);
    }
}

pub type Renderer = MacWgpuRenderer;

pub fn new_renderer(
    context: Context,
    bounds: gpui::Size<f32>,
    transparent: bool,
) -> anyhow::Result<Renderer> {
    MacWgpuRenderer::new(context, bounds, transparent)
}

pub struct MacWgpuRenderer {
    renderer: WgpuRenderer,
    layer: MetalLayer,
    compositor: CompositorLayers,
    live: HashMap<u64, LiveSurface>,
    last_live_frame_ms: u32,
    gpu: GpuContext,
    group_pixels: HashMap<u64, (Size<DevicePixels>, Arc<[u8]>)>,
    presents_with_transaction: bool,
}

impl MacWgpuRenderer {
    fn new(context: Context, bounds: gpui::Size<f32>, transparent: bool) -> anyhow::Result<Self> {
        let layer = new_window_layer(transparent);
        let config = WgpuSurfaceConfig {
            // The view resizes this to device pixels once it knows its scale factor.
            size: Size {
                width: DevicePixels(bounds.width.max(1.0) as i32),
                height: DevicePixels(bounds.height.max(1.0) as i32),
            },
            transparent,
            preferred_present_mode: None,
        };
        let gpu = context.gpu.clone();
        let renderer =
            WgpuRenderer::new_for_metal_layer(context.gpu, &layer, config, context.requirements)?;
        Ok(Self {
            renderer,
            layer,
            compositor: CompositorLayers::default(),
            live: HashMap::new(),
            last_live_frame_ms: animation_time_ms().wrapping_sub(16),
            gpu,
            group_pixels: HashMap::new(),
            presents_with_transaction: false,
        })
    }

    pub fn layer(&self) -> Option<&MetalLayerRef> {
        Some(&self.layer)
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer.as_ptr()
    }

    pub fn sprite_atlas(&self) -> &Arc<WgpuAtlas> {
        self.renderer.sprite_atlas()
    }

    /// Whether windows draw the scene's compositor sprites with Core Animation.
    pub fn supports_compositor_sprites(&self) -> bool {
        true
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        self.layer
            .set_presents_with_transaction(presents_with_transaction);
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        self.renderer.update_drawable_size(size);
    }

    /// The surface's alpha mode sets the layer's opacity whenever WGPU
    /// configures it, so it must not be set on the layer directly.
    pub fn update_transparency(&mut self, transparent: bool) {
        self.renderer.update_transparency(transparent);
    }

    pub fn destroy(&mut self) {
        for live in self.live.values() {
            let _: () = unsafe { msg_send![&*live.clip.0, removeFromSuperlayer] };
        }
        self.live.clear();
        self.renderer.destroy();
    }

    /// Draws `scene`, recovering the device first if it was lost. Returns
    /// whether the next frame must render the scene again, uncached.
    pub fn draw(&mut self, scene: &Scene) -> bool {
        if self.renderer.device_lost() {
            if let Err(error) = self.renderer.recover_metal_layer(&self.layer) {
                log::warn!("GPU recovery failed, will retry on next frame: {error}");
            }
            return true;
        }
        // A frame that adds or removes a compositor sprite appears together with its layer.
        let root = self.layer.as_ptr() as *mut AnyObject;
        self.group_pixels.retain(|key, _| {
            scene
                .compositor_groups()
                .iter()
                .any(|group| group.key == *key)
        });
        let mut sprites = scene.compositor_sprites().to_vec();
        for group in scene.compositor_groups() {
            let pixels = match self.group_pixels.get(&group.key) {
                Some(pixels) => pixels.clone(),
                None => {
                    let Some(pixels) = self.renderer.rasterize(&group.scene, group.rect) else {
                        continue;
                    };
                    self.group_pixels.insert(group.key, pixels.clone());
                    pixels
                }
            };
            let mut motion = group.motion;
            motion.origin = point(
                (group.origin.x.0 - group.rect.origin.x.0) / group.rect.size.width.0,
                (group.origin.y.0 - group.rect.origin.y.0) / group.rect.size.height.0,
            );
            let offset = |offset: gpui::Point<gpui::Pixels>, relative: gpui::Point<f32>| {
                offset
                    + point(
                        gpui::px(
                            group.motion_bounds.size.width.0 * relative.x / scene.scale_factor,
                        ),
                        gpui::px(
                            group.motion_bounds.size.height.0 * relative.y / scene.scale_factor,
                        ),
                    )
            };
            motion.offset = (
                offset(motion.offset.0, motion.relative_offset.0),
                offset(motion.offset.1, motion.relative_offset.1),
            );
            motion.relative_offset = (point(0.0, 0.0), point(0.0, 0.0));
            motion.clip = motion.clip.map(|(from, to)| {
                let insets = |edges: gpui::Edges<gpui::Pixels>| gpui::Edges {
                    left: gpui::px(
                        (group.motion_bounds.left().0 - group.rect.left().0) / scene.scale_factor,
                    ) + edges.left,
                    top: gpui::px(
                        (group.motion_bounds.top().0 - group.rect.top().0) / scene.scale_factor,
                    ) + edges.top,
                    right: gpui::px(
                        (group.rect.right().0 - group.motion_bounds.right().0) / scene.scale_factor,
                    ) + edges.right,
                    bottom: gpui::px(
                        (group.rect.bottom().0 - group.motion_bounds.bottom().0)
                            / scene.scale_factor,
                    ) + edges.bottom,
                };
                (insets(from), insets(to))
            });
            sprites.push(CompositorSprite {
                contents: SpriteContents::Pixels {
                    size: pixels.0,
                    bytes: pixels.1,
                },
                bounds: group.rect,
                clip: group.content_mask.bounds,
                color: None,
                frames: FrameLoop::NONE,
                motion,
            });
        }
        let sprites_changed = self.compositor.changed(root, &sprites);
        let layers_changed =
            sprites_changed || !scene.live_layers().is_empty() || !self.live.is_empty();
        if layers_changed {
            self.layer.set_presents_with_transaction(true);
            let _: () = unsafe { msg_send![class!(CATransaction), begin] };
            let _: () = unsafe { msg_send![class!(CATransaction), setDisableActions: Bool::YES] };
            if let Err(error) = self.sync_live_layers(scene) {
                log::error!("syncing live layers: {error}");
            }
        }
        self.renderer.draw(scene);
        self.draw_live_layers(scene);
        if sprites_changed {
            self.compositor
                .update(root, &sprites, self.renderer.sprite_atlas());
        }
        if layers_changed {
            let _: () = unsafe { msg_send![class!(CATransaction), commit] };
            self.layer
                .set_presents_with_transaction(self.presents_with_transaction);
        }
        self.renderer.needs_redraw()
    }

    fn sync_live_layers(&mut self, scene: &Scene) -> anyhow::Result<()> {
        let root = self.layer.as_ptr() as *mut AnyObject;
        let root_bounds: NSRect = unsafe { msg_send![root, bounds] };
        let scale: f64 = unsafe { msg_send![root, contentsScale] };
        let flipped: Bool = unsafe { msg_send![root, isGeometryFlipped] };
        let gpu = self.gpu.borrow();
        let gpu = gpu.as_ref().unwrap();
        self.live.retain(|key, live| {
            if scene.live_layers().iter().any(|layer| layer.key == *key) {
                true
            } else {
                let _: () = unsafe { msg_send![&*live.clip.0, removeFromSuperlayer] };
                false
            }
        });
        // Live surfaces share the window renderer pipelines, so they take its format and alpha.
        let (format, alpha_mode) = self.renderer.surface_format();
        for layer in scene.live_layers() {
            let size = layer
                .bounds
                .size
                .map(|value| DevicePixels(value.0.ceil() as i32));
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width.0 as u32,
                height: size.height.0 as u32,
                present_mode: wgpu::PresentMode::Fifo,
                desired_maximum_frame_latency: 2,
                alpha_mode,
                view_formats: Vec::new(),
            };
            if let std::collections::hash_map::Entry::Vacant(entry) = self.live.entry(layer.key) {
                let metal = new_window_layer(true);
                metal.set_presents_with_transaction(false);
                // SAFETY: the surface retains the layer, and LiveSurface also owns it.
                let surface = unsafe {
                    gpu.instance.create_surface_unsafe(
                        wgpu::SurfaceTargetUnsafe::CoreAnimationLayer(metal.as_ptr().cast()),
                    )?
                };
                surface.configure(&gpu.device, &config);
                let clip: Retained<AnyObject> = unsafe { msg_send![class!(CALayer), new] };
                let _: () = unsafe { msg_send![&*clip, setMasksToBounds: Bool::YES] };
                let _: () =
                    unsafe { msg_send![&*clip, addSublayer: metal.as_ptr() as *mut AnyObject] };
                let _: () = unsafe { msg_send![root, addSublayer: &*clip] };
                entry.insert(LiveSurface {
                    surface,
                    layer: metal,
                    clip: compositor::MainThread(clip),
                    size,
                });
            }
            let live = self.live.get_mut(&layer.key).unwrap();
            if live.size != size {
                live.surface.configure(&gpu.device, &config);
                live.size = size;
            }
            let points = |value: f32| f64::from(value) / scale;
            let clip = layer.content_mask.bounds;
            let clip_height = points(clip.size.height.0);
            let clip_y = if flipped.as_bool() {
                points(clip.origin.y.0)
            } else {
                root_bounds.size.height - points(clip.origin.y.0) - clip_height
            };
            let _: () = unsafe {
                msg_send![&*live.clip.0, setFrame: NSRect::new(
                NSPoint::new(points(clip.origin.x.0), clip_y),
                NSSize::new(points(clip.size.width.0), clip_height))]
            };
            let metal = live.layer.as_ptr() as *mut AnyObject;
            let _: () = unsafe { msg_send![metal, setContentsScale: scale] };
            let _: () = unsafe {
                msg_send![metal, setFrame: NSRect::new(
                NSPoint::new(points(layer.bounds.origin.x.0 - clip.origin.x.0),
                    clip_height - points(layer.bounds.origin.y.0 - clip.origin.y.0) - points(layer.bounds.size.height.0)),
                NSSize::new(points(layer.bounds.size.width.0), points(layer.bounds.size.height.0)))]
            };
        }
        Ok(())
    }

    pub fn draw_live_layers(&mut self, scene: &Scene) {
        let now = animation_time_ms();
        if now.wrapping_sub(self.last_live_frame_ms) < 16 {
            return;
        }
        self.last_live_frame_ms = now;
        for layer in scene.live_layers() {
            let Some(live) = self.live.get(&layer.key) else {
                continue;
            };
            let mut canvas = LiveCanvas {
                scene: Scene::default(),
                scale_factor: scene.scale_factor,
                size: layer
                    .bounds
                    .size
                    .map(|value| px(value.0 / scene.scale_factor)),
            };
            canvas.scene.scale_factor = scene.scale_factor;
            (layer.paint)(
                now.wrapping_sub(layer.start_ms) as f32 / 1000.0,
                &mut canvas,
            );
            canvas.scene.finish();
            let (wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture)) =
                live.surface.get_current_texture()
            else {
                continue;
            };
            let view = texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            if self.renderer.render_sized(&canvas.scene, &view, live.size) {
                texture.present();
            }
        }
    }

    pub fn rasterize(
        &mut self,
        scene: &Scene,
        rect: Bounds<ScaledPixels>,
    ) -> Option<(Size<DevicePixels>, Arc<[u8]>)> {
        self.renderer.rasterize(scene, rect)
    }

    /// Presents an animation tick of the scene last passed to [`Self::draw`]. Returns whether the
    /// next frame must render the scene again, uncached.
    pub fn draw_animation_frame(&mut self, scene: &Scene) -> bool {
        if self.renderer.device_lost() {
            if let Err(error) = self.renderer.recover_metal_layer(&self.layer) {
                log::warn!("GPU recovery failed, will retry on next frame: {error}");
            }
            return true;
        }
        self.renderer.draw_animation_frame(scene);
        self.renderer.needs_redraw()
    }

    pub fn gpu_specs(&self) -> GpuSpecs {
        self.renderer.gpu_specs()
    }

    pub fn gpu_context(&self) -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        self.renderer.gpu_context()
    }

    pub fn device_lost(&self) -> bool {
        self.renderer.device_lost()
    }

    pub fn gpu_context_info(&self) -> Option<WgpuContextHandle> {
        self.renderer.gpu_context_info()
    }

    #[cfg(feature = "test-support")]
    pub fn render_to_image(&mut self, scene: &Scene) -> anyhow::Result<image::RgbaImage> {
        self.renderer.render_to_image(scene)
    }

    // WGPU's offscreen rendering needs gpui_wgpu's test-support, which the
    // `test-support` feature enables.
    #[cfg(all(test, not(feature = "test-support")))]
    pub fn render_to_image(&mut self, _scene: &Scene) -> anyhow::Result<image::RgbaImage> {
        anyhow::bail!("rendering to an image needs the test-support feature")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transparency_survives_resizing() {
        let mut renderer =
            MacWgpuRenderer::new(Context::default(), gpui::size(64.0, 64.0), false).unwrap();
        assert!(renderer.layer.is_opaque());

        renderer.update_transparency(true);
        assert!(!renderer.layer.is_opaque());
        renderer.update_drawable_size(gpui::size(DevicePixels(128), DevicePixels(96)));
        assert!(!renderer.layer.is_opaque());

        renderer.update_transparency(false);
        renderer.update_drawable_size(gpui::size(DevicePixels(64), DevicePixels(64)));
        assert!(renderer.layer.is_opaque());
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn renders_scenes_to_images() {
        let mut renderer =
            MacWgpuRenderer::new(Context::default(), gpui::size(32.0, 16.0), false).unwrap();
        // A red quad over the left half.
        let bounds = |width| gpui::Bounds {
            origin: gpui::point(gpui::ScaledPixels(0.0), gpui::ScaledPixels(0.0)),
            size: gpui::size(gpui::ScaledPixels(width), gpui::ScaledPixels(16.0)),
        };
        let mut scene = Scene::default();
        scene.insert_primitive(gpui::Quad {
            bounds: bounds(16.0),
            content_mask: gpui::ContentMask {
                bounds: bounds(32.0),
            },
            background: gpui::solid_background(gpui::red()),
            ..Default::default()
        });
        scene.finish();

        let image = renderer.render_to_image(&scene).unwrap();

        assert_eq!(image.dimensions(), (32, 16));
        assert_eq!(image.get_pixel(8, 8).0, [255, 0, 0, 255]);
        assert_ne!(image.get_pixel(24, 8).0, [255, 0, 0, 255]);
    }
}

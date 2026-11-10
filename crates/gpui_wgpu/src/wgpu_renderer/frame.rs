// Modified by HexpyGames for Butler (2026); see NOTICE for the changes.
use super::{
    WgpuRenderer, begin_color_render_pass,
    buffers::{InstanceTransport, InstanceUpload},
    filters::{FILTER_UNIFORMS_PER_COMPOSITE, FrameUniformRequirements},
    path_types,
};
use gpui::{
    BackdropFilter, Bounds, DamageRedraw, DevicePixels, FilterRenderTarget, MonochromeSprite,
    Point, PolychromeSprite, PrimitiveBatch, Quad, RenderCommand, Scene, ScenePlanRequirements,
    Shadow, Size, SubpixelSprite, Underline, backdrop_appear_parameters,
};
use gpui_render::TargetStack;
use gpui_render::blur::{
    BlurKernel, FilterCompositeClip, FilterCompositeParameters,
    GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS, ScissorRectangle,
};
use gpui_render::shaders::{
    blur::BlurUniforms,
    common::{FontRasterizationUniforms, GlobalUniforms, ShaderBool},
    interface as shader_interface,
};

/// Submits a complete scene. Animation fallbacks leave `release_unused` disabled.
pub(super) fn render_to_view(
    renderer: &mut WgpuRenderer,
    scene: &Scene,
    frame_view: &wgpu::TextureView,
    readback: Option<ReadbackCopy<'_>>,
    release_unused: bool,
) -> Option<wgpu::SubmissionIndex> {
    let mut planned = *scene.render_plan().requirements();
    planned.uses_offscreen_target |= scene.has_animations();
    let commands = scene.render_commands();
    let Some(targets) = PreparedTargets::prepare(
        renderer,
        commands,
        planned,
        0,
        frame_view,
        renderer.target.viewport_size(),
    ) else {
        renderer.retained = None;
        return None;
    };
    // Prune once against the complete scene. Doing this inside each draw batch would evict a
    // texture that reappears after an intervening primitive and recreate its platform view.
    renderer.retain_surface_cache(&scene.surfaces);

    let encoder = FrameEncoder::new(renderer, scene, commands, None, None, targets);
    let offscreen = encoder.offscreen.is_some();
    let keys = encoder.backdrop_keys.clone();
    let submission = submit(renderer, encoder, readback);
    renderer.retained = (submission.is_some() && offscreen).then(|| scene.generation());
    renderer.frame_keys = (scene.generation(), keys);
    renderer.retained_atlas_epoch = renderer.atlas.removal_epoch();
    if submission.is_some() && release_unused {
        renderer.resources_mut().release_unused(&planned);
    }
    submission
}

/// Renders `scene` into `view` for an offscreen copy, invalidating any reused scene target.
pub(super) fn render_offscreen(
    renderer: &mut WgpuRenderer,
    scene: &Scene,
    view: &wgpu::TextureView,
    size: Size<DevicePixels>,
) -> bool {
    let planned = *scene.render_plan().requirements();
    let commands = scene.render_commands();
    let Some(targets) = PreparedTargets::prepare(renderer, commands, planned, 0, view, size) else {
        return false;
    };
    if targets.offscreen.is_some() {
        renderer.retained = None;
    }
    let encoder = FrameEncoder::new(renderer, scene, commands, None, None, targets);
    submit(renderer, encoder, None).is_some()
}

/// Encodes and submits `encoder`; `None` when encoding failed (nothing is submitted).
pub(super) fn submit(
    renderer: &WgpuRenderer,
    encoder: FrameEncoder<'_>,
    readback: Option<ReadbackCopy<'_>>,
) -> Option<wgpu::SubmissionIndex> {
    match encoder.encode(readback) {
        Ok(command_buffer) => Some(renderer.resources().queue.submit([command_buffer])),
        Err(DrawError::ExternalSurface) => None,
        Err(DrawError::CapacityPlanningInvariant) => {
            log::error!("frame storage exceeded its precomputed capacity");
            None
        }
        Err(DrawError::MissingIntermediateTarget) => {
            log::error!("frame preparation did not create a required intermediate target");
            None
        }
    }
}

pub(super) struct ReadbackCopy<'a> {
    pub(super) texture: &'a wgpu::Texture,
    pub(super) buffer: &'a wgpu::Buffer,
    pub(super) bytes_per_row: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

pub(super) struct PreparedTargets {
    active: wgpu::TextureView,
    presentation: wgpu::TextureView,
    offscreen: Option<wgpu::TextureView>,
    instances: InstanceUpload,
}

impl PreparedTargets {
    /// Sizes and uploads for `commands`; `extra_filter_uniforms` reserves slots the encoder
    /// writes beyond what `planned` accounts for. `planned.uses_offscreen_target` selects the
    /// offscreen target.
    pub(super) fn prepare(
        renderer: &mut WgpuRenderer,
        commands: &[RenderCommand],
        planned: ScenePlanRequirements,
        extra_filter_uniforms: u64,
        frame_view: &wgpu::TextureView,
        size: Size<DevicePixels>,
    ) -> Option<Self> {
        if !begin_frame(renderer) {
            return None;
        }
        let mut requirements = {
            let transport = renderer.resources().instances.transport();
            FrameRequirements::for_commands(commands, &planned, transport)
        };
        requirements.uniforms.filter_count += extra_filter_uniforms;
        let device = renderer.resources().device.clone();
        let resources = renderer.resources_mut();
        if !resources.instances.ensure_capacity(
            &device,
            &resources.bind_group_layouts,
            requirements.storage_bytes,
            requirements.instance_batches,
        ) {
            return None;
        }
        let instances = {
            let resources = renderer.resources();
            resources.instances.begin_upload(
                &resources.queue,
                requirements.storage_bytes,
                requirements.instance_batches,
            )?
        };
        if !renderer.ensure_uniform_capacity(requirements.uniforms) {
            return None;
        }

        if requirements.planned.uses_path_target {
            renderer.ensure_path_textures(size);
        }
        if requirements.planned.uses_offscreen_target {
            renderer.ensure_filter_textures(&requirements.planned);
        }
        write_shader_globals(renderer, size);

        if requirements.planned.uses_offscreen_target {
            let resources = renderer.resources();
            let offscreen = resources
                .scene_color_view
                .as_ref()
                .expect("blur texture preparation must create a scene target")
                .clone();
            Some(Self {
                active: offscreen.clone(),
                presentation: frame_view.clone(),
                offscreen: Some(offscreen),
                instances,
            })
        } else {
            Some(Self {
                active: frame_view.clone(),
                presentation: frame_view.clone(),
                offscreen: None,
                instances,
            })
        }
    }
}

fn begin_frame(renderer: &mut WgpuRenderer) -> bool {
    let Some(error) = renderer.faults.pending_error.lock().unwrap().take() else {
        renderer.faults.consecutive_failed_frames = 0;
        renderer.atlas.before_frame();
        return true;
    };

    renderer.retained = None;
    renderer.faults.consecutive_failed_frames += 1;
    log::error!(
        "GPU error during frame (failure {} of 10): {error}",
        renderer.faults.consecutive_failed_frames
    );
    if renderer.faults.consecutive_failed_frames > 10 {
        panic!("too many consecutive GPU errors; last error: {error}");
    }
    if renderer.faults.consecutive_failed_frames > 5 {
        if let Some(resources) = renderer.resources.as_mut() {
            resources.invalidate_intermediate_textures();
        }
        renderer.atlas.clear();
        renderer.target.request_redraw();
        renderer.faults.consecutive_failed_frames = 0;
        return false;
    }

    renderer.atlas.before_frame();
    true
}

#[derive(Clone, Copy, PartialEq)]
pub(super) struct GlobalUniformState {
    globals: GlobalUniforms,
    path_globals: GlobalUniforms,
    font_rasterization: FontRasterizationUniforms,
}

pub(super) fn write_shader_globals(renderer: &mut WgpuRenderer, size: Size<DevicePixels>) {
    let font = renderer.rendering_params.font_rasterization;
    let font_rasterization = FontRasterizationUniforms {
        gamma_ratios: wgsl_rs::std::vec4f(
            font.gamma_ratios[0],
            font.gamma_ratios[1],
            font.gamma_ratios[2],
            font.gamma_ratios[3],
        ),
        grayscale_enhanced_contrast: font.grayscale_enhanced_contrast,
        subpixel_enhanced_contrast: font.subpixel_enhanced_contrast,
        uses_blue_green_red_subpixel_order: ShaderBool::from(
            renderer.subpixel_order == super::SubpixelOrder::BlueGreenRed,
        ),
        padding: 0,
    };
    let globals = GlobalUniforms {
        viewport_size: wgsl_rs::std::vec2f(size.width.0 as f32, size.height.0 as f32),
        premultiplied_alpha: ShaderBool::from(
            renderer.target.alpha_mode() == wgpu::CompositeAlphaMode::PreMultiplied,
        ),
        animation_time_ms: renderer.animation_time_ms,
    };
    let path_globals = GlobalUniforms {
        premultiplied_alpha: ShaderBool::Disabled,
        ..globals
    };
    let state = GlobalUniformState {
        globals,
        path_globals,
        font_rasterization,
    };
    if renderer.uploaded_globals == Some(state) {
        return;
    }

    let resources = renderer.resources();
    let globals_size = std::mem::size_of::<GlobalUniforms>();
    let font_size = std::mem::size_of::<FontRasterizationUniforms>();
    let upload_size = renderer.globals.font_offset + font_size as u64;
    let mut upload = resources
        .queue
        .write_buffer_with(
            &resources.globals_buffer,
            0,
            std::num::NonZeroU64::new(upload_size).expect("global uniforms are non-empty"),
        )
        .expect("global uniform upload must fit its buffer");
    upload.slice(..).fill(0);
    upload
        .slice(..globals_size)
        .copy_from_slice(shader_interface::bytes_of(&globals));
    upload
        .slice(
            renderer.globals.path_offset as usize
                ..renderer.globals.path_offset as usize + globals_size,
        )
        .copy_from_slice(shader_interface::bytes_of(&path_globals));
    upload
        .slice(
            renderer.globals.font_offset as usize
                ..renderer.globals.font_offset as usize + font_size,
        )
        .copy_from_slice(shader_interface::bytes_of(&font_rasterization));
    drop(upload);
    renderer.uploaded_globals = Some(state);
}

#[derive(Clone, Copy, Default)]
pub(super) struct FrameRequirements {
    storage_bytes: u64,
    /// Instance batches this frame; one downlevel range-uniform slot per batch.
    instance_batches: u64,
    pub(super) uniforms: FrameUniformRequirements,
    planned: ScenePlanRequirements,
}

impl FrameRequirements {
    pub(super) fn for_commands(
        commands: &[RenderCommand],
        planned: &ScenePlanRequirements,
        transport: InstanceTransport,
    ) -> Self {
        let mut storage_bytes = 0_u64;
        let mut instance_batches = 0_u64;
        let mut reserve = |element_size: usize, count: usize| {
            if count > 0 {
                let stride = element_size as u64;
                storage_bytes = storage_bytes.next_multiple_of(transport.batch_alignment(stride));
                storage_bytes = storage_bytes.saturating_add(stride.saturating_mul(count as u64));
                instance_batches += 1;
            }
        };

        for command in commands {
            let RenderCommand::Batch(batch) = command else {
                continue;
            };
            match batch {
                PrimitiveBatch::Shadows { range, .. } => {
                    reserve(std::mem::size_of::<Shadow>(), range.len())
                }
                PrimitiveBatch::Quads { range, .. } => {
                    reserve(std::mem::size_of::<Quad>(), range.len())
                }
                PrimitiveBatch::Paths {
                    rasterization_vertex_count,
                    sprite_count,
                    ..
                } if *rasterization_vertex_count > 0 => {
                    reserve(
                        std::mem::size_of::<path_types::PathRasterizationVertex>(),
                        *rasterization_vertex_count,
                    );
                    reserve(std::mem::size_of::<path_types::PathSprite>(), *sprite_count);
                }
                PrimitiveBatch::Underlines(range) => {
                    reserve(std::mem::size_of::<Underline>(), range.len())
                }
                PrimitiveBatch::MonochromeSprites { range, .. } => {
                    reserve(std::mem::size_of::<MonochromeSprite>(), range.len())
                }
                PrimitiveBatch::SubpixelSprites { range, .. } => {
                    reserve(std::mem::size_of::<SubpixelSprite>(), range.len())
                }
                PrimitiveBatch::PolychromeSprites { range, .. } => {
                    reserve(std::mem::size_of::<PolychromeSprite>(), range.len())
                }
                PrimitiveBatch::Paths { .. }
                | PrimitiveBatch::Surfaces(_)
                | PrimitiveBatch::BackdropFilters(_)
                | PrimitiveBatch::FilterBoundary(_) => {}
            }
        }
        debug_assert_eq!(instance_batches as usize, planned.instance_batch_count);

        Self {
            storage_bytes,
            instance_batches,
            uniforms: FrameUniformRequirements {
                filter_count: FILTER_UNIFORMS_PER_COMPOSITE
                    * (planned.backdrop_filter_count + planned.isolated_filter_count) as u64
                    + u64::from(planned.uses_offscreen_target)
                    + planned.backdrop_lens_count as u64
                    + if planned.isolated_target_count > 0 {
                        (planned.backdrop_filter_count * (planned.isolated_target_count + 1)) as u64
                    } else {
                        0
                    },
                surface_count: planned.surface_count as u64,
            },
            planned: *planned,
        }
    }
}

pub(super) struct FrameEncoder<'a> {
    renderer: &'a WgpuRenderer,
    scene: &'a Scene,
    /// The commands to encode: the whole scene, or a damage plan's restricted list.
    commands: &'a [RenderCommand],
    /// Set for a damage redraw over the retained frame.
    damage: Option<&'a DamageRedraw>,
    /// The damage rectangle every pass that writes the scene target is limited to.
    scissor: Option<ScissorRectangle>,
    encoder: wgpu::CommandEncoder,
    targets: TargetStack<wgpu::TextureView>,
    offscreen: Option<wgpu::TextureView>,
    presentation: wgpu::TextureView,
    instances: InstanceUpload,
    backdrop_keys: Vec<Option<u64>>,
}

/// Whether compositing `filter` reads a blurred backdrop that the cache can supply.
pub(super) fn uses_blurred_backdrop(filter: &BackdropFilter) -> bool {
    BlurKernel::for_radius(filter.max_blur_radius()).is_some() || filter.lens.is_active()
}

impl<'a> FrameEncoder<'a> {
    pub(super) fn new(
        renderer: &'a WgpuRenderer,
        scene: &'a Scene,
        commands: &'a [RenderCommand],
        damage: Option<&'a DamageRedraw>,
        scissor: Option<ScissorRectangle>,
        targets: PreparedTargets,
    ) -> Self {
        let encoder =
            renderer
                .resources()
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gpui_frame"),
                });
        let backdrop_keys = if damage.is_some() && renderer.frame_keys.0 == scene.generation() {
            // Reuse keys only for the same generation.
            renderer.frame_keys.1.clone()
        } else if scene.backdrop_filters.is_empty() {
            Vec::new()
        } else {
            gpui_render::backdrop_key::backdrop_keys(scene, renderer.target.viewport_size())
        };
        Self {
            renderer,
            scene,
            commands,
            damage,
            scissor,
            encoder,
            targets: TargetStack::new(targets.active),
            offscreen: targets.offscreen,
            presentation: targets.presentation,
            instances: targets.instances,
            backdrop_keys,
        }
    }

    pub(super) fn encode(
        mut self,
        readback: Option<ReadbackCopy<'_>>,
    ) -> Result<wgpu::CommandBuffer, DrawError> {
        let result = self.encode_commands();
        if result.is_ok() {
            if let Some(offscreen) = &self.offscreen {
                self.renderer
                    .blit_to_frame(&mut self.encoder, offscreen, &self.presentation);
            }
            if let Some(readback) = readback {
                self.encoder.copy_texture_to_buffer(
                    readback.texture.as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: readback.buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(readback.bytes_per_row),
                            rows_per_image: Some(readback.height),
                        },
                    },
                    wgpu::Extent3d {
                        width: readback.width,
                        height: readback.height,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
        self.instances.finish(&mut self.encoder);
        self.renderer.resources().finish_frame_uploads();
        let command_buffer = self.encoder.finish();
        result.map(|()| command_buffer)
    }

    fn encode_commands(&mut self) -> DrawResult {
        let mut scissor = self.scissor;
        let mut motion_target = None;
        let damage = self.damage;
        let mut pending_load = if let Some(scissor) = scissor.filter(|_| damage.is_some()) {
            // A clear ignores the scissor, so the damage rectangle is overwritten with the
            // frame's clear colour by a scissored draw instead.
            let renderer = self.renderer;
            let ping = renderer
                .resources()
                .blur_ping_view
                .clone()
                .ok_or(DrawError::MissingIntermediateTarget)?;
            let clear = renderer.target.clear_color();
            renderer.draw_filter_pass(
                &mut self.encoder,
                "damage_clear",
                &renderer.resources().pipelines.blur_downsample,
                self.targets.current(),
                &ping,
                BlurUniforms::fill([
                    clear.r as f32,
                    clear.g as f32,
                    clear.b as f32,
                    clear.a as f32,
                ]),
                wgpu::LoadOp::Load,
                Some(scissor),
            );
            wgpu::LoadOp::Load
        } else {
            wgpu::LoadOp::Clear(self.renderer.target.clear_color())
        };
        let mut pass = None;
        let mut parent_loads = smallvec::SmallVec::<[_; gpui::MAX_FILTER_GROUP_DEPTH]>::new();

        for command in self.commands {
            match command {
                RenderCommand::Batch(PrimitiveBatch::Paths {
                    range,
                    rasterization_vertex_count,
                    ..
                }) => {
                    if *rasterization_vertex_count == 0 {
                        continue;
                    }
                    let paths = &self.scene.paths[range.clone()];
                    drop(pass.take());
                    self.renderer.draw_paths_to_intermediate(
                        &mut self.encoder,
                        paths,
                        &mut self.instances,
                    )?;
                    let pass = self.scene_pass(&mut pass, &mut pending_load, scissor);
                    self.renderer
                        .draw_paths_from_intermediate(paths, &mut self.instances, pass)?;
                }
                RenderCommand::Batch(PrimitiveBatch::BackdropFilters(range)) => {
                    self.scene_pass(&mut pass, &mut pending_load, scissor);
                    drop(pass.take());
                    for index in range.clone() {
                        let filter = &self.scene.backdrop_filters[index];
                        if let Some(paint) = &filter.wallpaper {
                            self.renderer.draw_wallpaper(
                                &mut self.encoder,
                                self.targets.current(),
                                filter,
                                paint,
                                scissor,
                            );
                            continue;
                        }
                        let (lens, radius) =
                            backdrop_appear_parameters(filter, self.renderer.animation_time_ms);
                        let width = self.renderer.target.width();
                        let height = self.renderer.target.height();
                        let region = gpui_render::blur::backdrop_snapshot_region(
                            filter.bounds,
                            radius,
                            &lens,
                            width,
                            height,
                        );
                        let reblur = damage.is_some_and(|plan| plan.reblur.contains(&index));
                        let key = if self.targets.inherits_backdrop() {
                            None
                        } else {
                            self.backdrop_keys.get(index).copied().flatten()
                        };
                        let hit = !reblur
                            && key.is_some_and(|key| {
                                let resources = self.renderer.resources();
                                resources.backdrop_cache.borrow_mut().restore(
                                    key,
                                    &mut self.encoder,
                                    resources.blur_ping_texture.as_ref(),
                                    resources.backdrop_snapshot_texture.as_ref(),
                                )
                            });
                        if damage.is_some() && !reblur && !hit && uses_blurred_backdrop(filter) {
                            // The plan promised a cache entry for every glass it does not re-blur.
                            return Err(DrawError::MissingIntermediateTarget);
                        }
                        let snapshot = if hit {
                            if lens.is_active() {
                                self.renderer.resources().backdrop_snapshot_view.clone()
                            } else {
                                None
                            }
                        } else if self.targets.inherits_backdrop() {
                            // A lens reads the sharp full-resolution backdrop, so it needs a copy
                            // even when the backdrop is the target it composites into.
                            Some(self.renderer.snapshot_backdrop(
                                &mut self.encoder,
                                self.targets.backdrop_layers(),
                                region,
                            ))
                        } else if lens.is_active() {
                            Some(self.renderer.snapshot_backdrop(
                                &mut self.encoder,
                                std::iter::once(self.targets.current()),
                                region,
                            ))
                        } else {
                            None
                        };
                        let source = snapshot.as_ref().unwrap_or_else(|| self.targets.current());
                        self.renderer.blur_and_composite(
                            &mut self.encoder,
                            source,
                            self.targets.current(),
                            FilterCompositeParameters {
                                bounds: filter.bounds,
                                content_mask: filter.content_mask.bounds,
                                corner_radii: filter.corner_radii,
                                corner_smoothing: filter.corner_smoothing,
                                blur_radius: radius,
                                opacity: filter.opacity,
                                scale: 1.0,
                                scale_origin: Point::default(),
                                composite_offset: Point::default(),
                                composite_rotation: 0.0,
                                clip: FilterCompositeClip::RoundedBounds,
                                motion_clip: Bounds::default(),
                                lens,
                            },
                            hit,
                            wgpu::LoadOp::Load,
                            scissor,
                        );
                        if let (false, Some(key)) = (hit, key) {
                            let resources = self.renderer.resources();
                            let blurred = BlurKernel::for_radius(radius)
                                .and_then(|_| resources.blur_ping_texture.as_ref())
                                .map(|ping| {
                                    (
                                        ping,
                                        ScissorRectangle::for_blurred_bounds(
                                            filter.bounds,
                                            GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * radius,
                                            width,
                                            height,
                                        ),
                                    )
                                });
                            let sharp = if lens.is_active() {
                                resources.backdrop_snapshot_texture.as_ref().map(|texture| {
                                    (
                                        texture,
                                        ScissorRectangle {
                                            x: region.origin.x.0 as u32,
                                            y: region.origin.y.0 as u32,
                                            width: region.size.width.0 as u32,
                                            height: region.size.height.0 as u32,
                                        },
                                    )
                                })
                            } else {
                                None
                            };
                            resources.backdrop_cache.borrow_mut().store(
                                &resources.device,
                                &mut self.encoder,
                                key,
                                blurred,
                                sharp,
                            );
                        }
                    }
                }
                RenderCommand::Batch(PrimitiveBatch::FilterBoundary(_)) => {
                    unreachable!("filter boundaries must be compiled into render commands")
                }
                RenderCommand::Batch(batch) => {
                    let pass = self.scene_pass(&mut pass, &mut pending_load, scissor);
                    encode_inline_batch(
                        self.renderer,
                        self.scene,
                        batch,
                        &mut self.instances,
                        pass,
                    )?;
                }
                RenderCommand::BeginFilter {
                    boundary_index,
                    target: FilterRenderTarget::Isolated(index),
                } => {
                    drop(pass.take());
                    parent_loads.push(pending_load);
                    let target =
                        self.renderer.resources().filter_group_views[index.as_usize()].clone();
                    if motion_target.is_none()
                        && (self.scene.filter_boundaries[*boundary_index]
                            .motion
                            .is_animated()
                            || self.scene.filter_boundaries[*boundary_index].max_blur_radius()
                                > 0.0
                            || self.scene.filter_boundaries[*boundary_index].scale != 1.0)
                    {
                        motion_target = Some(*index);
                        scissor = None;
                    }
                    self.targets.enter(
                        target,
                        self.scene.filter_boundaries[*boundary_index]
                            .filters
                            .is_empty(),
                    );
                    pending_load = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);
                }
                RenderCommand::EndFilter {
                    boundary_index,
                    target: FilterRenderTarget::Isolated(index),
                    ..
                } => {
                    if matches!(pending_load, wgpu::LoadOp::Clear(_)) {
                        self.scene_pass(&mut pass, &mut pending_load, scissor);
                    }
                    drop(pass.take());
                    let (filtered, parent) = self.targets.exit();
                    pending_load = parent_loads.pop().expect("filter parent load must exist");
                    if motion_target == Some(*index) {
                        motion_target = None;
                        scissor = self.scissor;
                    }
                    let boundary = &self.scene.filter_boundaries[*boundary_index];
                    let (opacity, scale, composite_offset, composite_rotation, clip) =
                        gpui::motion_composite_parameters(
                            boundary,
                            self.renderer.animation_time_ms,
                            self.scene.scale_factor,
                        );
                    self.renderer.blur_and_composite(
                        &mut self.encoder,
                        &filtered,
                        parent,
                        FilterCompositeParameters {
                            bounds: boundary.bounds,
                            content_mask: boundary.content_mask.bounds,
                            corner_radii: clip.map_or(boundary.corner_radii, |(_, radii)| radii),
                            corner_smoothing: boundary.corner_smoothing,
                            blur_radius: boundary.max_blur_radius(),
                            opacity,
                            scale,
                            composite_offset,
                            composite_rotation,
                            scale_origin: boundary.scale_origin,
                            clip: clip.map_or(FilterCompositeClip::ContentShape, |_| {
                                FilterCompositeClip::MotionRounded
                            }),
                            motion_clip: clip.map_or(Bounds::default(), |(rect, _)| rect),
                            lens: Default::default(),
                        },
                        false,
                        pending_load,
                        scissor,
                    );
                    pending_load = wgpu::LoadOp::Load;
                }
                RenderCommand::BeginFilter {
                    target: FilterRenderTarget::Inline,
                    ..
                }
                | RenderCommand::EndFilter {
                    target: FilterRenderTarget::Inline,
                    ..
                } => {}
            }
        }
        if matches!(pending_load, wgpu::LoadOp::Clear(_)) {
            self.scene_pass(&mut pass, &mut pending_load, scissor);
        }
        drop(pass);
        self.targets.assert_balanced();
        Ok(())
    }

    fn scene_pass<'pass>(
        &mut self,
        pass: &'pass mut Option<wgpu::RenderPass<'static>>,
        pending_load: &mut wgpu::LoadOp<wgpu::Color>,
        scissor: Option<ScissorRectangle>,
    ) -> &'pass mut wgpu::RenderPass<'static> {
        pass.get_or_insert_with(|| {
            let pass = begin_scene_render_pass(
                self.renderer,
                &mut self.encoder,
                "scene_pass",
                self.targets.current(),
                *pending_load,
                scissor,
            )
            .forget_lifetime();
            *pending_load = wgpu::LoadOp::Load;
            pass
        })
    }
}

fn begin_scene_render_pass<'a>(
    renderer: &'a WgpuRenderer,
    encoder: &'a mut wgpu::CommandEncoder,
    label: &'a str,
    target: &'a wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    scissor: Option<ScissorRectangle>,
) -> wgpu::RenderPass<'a> {
    let mut pass = begin_color_render_pass(encoder, label, target, load, scissor);
    pass.set_bind_group(
        shader_interface::GLOBAL_BIND_GROUP,
        &renderer.resources().globals_bind_group,
        &[],
    );
    pass
}

#[derive(Debug)]
pub(super) enum DrawError {
    CapacityPlanningInvariant,
    ExternalSurface,
    MissingIntermediateTarget,
}

pub(super) type DrawResult = Result<(), DrawError>;

fn encode_inline_batch(
    renderer: &WgpuRenderer,
    scene: &Scene,
    batch: &PrimitiveBatch,
    instances: &mut InstanceUpload,
    pass: &mut wgpu::RenderPass<'_>,
) -> DrawResult {
    match batch {
        PrimitiveBatch::Quads { range, smoothed } => {
            renderer.draw_quads(&scene.quads[range.clone()], *smoothed, instances, pass)
        }
        PrimitiveBatch::Shadows { range, smoothed } => {
            renderer.draw_shadows(&scene.shadows[range.clone()], *smoothed, instances, pass)
        }
        PrimitiveBatch::Underlines(range) => {
            renderer.draw_underlines(&scene.underlines[range.clone()], instances, pass)
        }
        PrimitiveBatch::MonochromeSprites { texture_id, range } => renderer
            .draw_monochrome_sprites(
                &scene.monochrome_sprites[range.clone()],
                *texture_id,
                instances,
                pass,
            ),
        PrimitiveBatch::SubpixelSprites { texture_id, range } => renderer.draw_subpixel_sprites(
            &scene.subpixel_sprites[range.clone()],
            *texture_id,
            instances,
            pass,
        ),
        PrimitiveBatch::PolychromeSprites {
            texture_id,
            range,
            smoothed,
        } => renderer.draw_polychrome_sprites(
            &scene.polychrome_sprites[range.clone()],
            *texture_id,
            *smoothed,
            instances,
            pass,
        ),
        PrimitiveBatch::Surfaces(range) => {
            renderer.draw_surfaces(&scene.surfaces[range.clone()], pass)
        }
        PrimitiveBatch::Paths { .. }
        | PrimitiveBatch::BackdropFilters(_)
        | PrimitiveBatch::FilterBoundary(_) => {
            unreachable!("pass-interrupting batches are handled by FrameEncoder")
        }
    }
}

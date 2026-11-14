// Modified by HexpyGames for Butler (2026); see NOTICE for the changes.
use gpui::{Bounds, Corners, Point, ScaledPixels};
use gpui_render::shaders::interface as shader_interface;
use gpui_render::{
    blur::{
        BlurAxis, BlurKernel, FilterCompositeClip, FilterCompositeParameters,
        GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS, ScissorRectangle, downsampled_dimension,
    },
    shaders::blur::BlurUniforms,
};
use smallvec::SmallVec;

use super::{WgpuRenderer, begin_color_render_pass, pipelines};

// downsample, two blur passes, the composite and a lens's sharp-rim pass
pub(super) const FILTER_UNIFORMS_PER_COMPOSITE: u64 = 5;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct FrameUniformRequirements {
    pub(super) filter_count: u64,
    pub(super) surface_count: u64,
}

const _: () = assert!(std::mem::size_of::<BlurUniforms>() == 496);

impl WgpuRenderer {
    pub(super) fn draw_filter_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &pipelines::WgpuRenderPipeline,
        target: &wgpu::TextureView,
        source: &wgpu::TextureView,
        uniforms: BlurUniforms,
        load: wgpu::LoadOp<wgpu::Color>,
        scissor: Option<ScissorRectangle>,
    ) {
        let mut pass = begin_color_render_pass(encoder, label, target, load, scissor);
        self.draw_filter(&mut pass, pipeline, source, uniforms);
    }

    pub(super) fn draw_filter(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        pipeline: &pipelines::WgpuRenderPipeline,
        source: &wgpu::TextureView,
        uniforms: BlurUniforms,
    ) {
        let resources = self.resources();
        let uniform_offset = resources.filter_uniforms.write(&uniforms);
        let bind_group = resources.blur_bind_group(source);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(
            shader_interface::GLOBAL_BIND_GROUP,
            &resources.globals_bind_group,
            &[],
        );
        pass.set_bind_group(
            shader_interface::DATA_BIND_GROUP,
            &bind_group,
            &[uniform_offset],
        );

        pass.draw(0..pipeline.fixed_vertex_count(), 0..1);
    }

    /// Prepares composites with one pass per blur stage for all blurred items.
    pub(super) fn prepare_composites(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        parameters: &[FilterCompositeParameters],
        blurred_ready: bool,
    ) -> SmallVec<[Option<wgpu::TextureView>; 4]> {
        debug_assert!(
            parameters
                .windows(2)
                .all(|pair| std::mem::discriminant(&pair[0].clip)
                    == std::mem::discriminant(&pair[1].clip))
        );
        let mut blurred = SmallVec::<[_; 4]>::new();
        let prepared = parameters
            .iter()
            .map(|parameters| {
                if parameters.blur_radius <= 0.0
                    && !parameters.lens.is_active()
                    && matches!(parameters.clip, FilterCompositeClip::RoundedBounds)
                {
                    return None;
                }
                let Some(kernel) = BlurKernel::for_radius(parameters.blur_radius) else {
                    return Some(source.clone());
                };
                let scissor = ScissorRectangle::for_blurred_bounds(
                    parameters.bounds,
                    GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * parameters.blur_radius,
                    self.target.width(),
                    self.target.height(),
                );
                if scissor.is_empty() {
                    return None;
                }
                let resources = self.resources();
                let (Some(horizontal), Some(_)) =
                    (&resources.blur_ping_view, &resources.blur_pong_view)
                else {
                    return None;
                };
                blurred.push((scissor, kernel));
                Some(horizontal.clone())
            })
            .collect();
        if !blurred_ready && !blurred.is_empty() {
            let load = match parameters[0].clip {
                FilterCompositeClip::RoundedBounds => wgpu::LoadOp::Load,
                FilterCompositeClip::ContentShape | FilterCompositeClip::MotionRounded => {
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                }
            };
            self.prepare_blurs(encoder, source, &blurred, load);
        }
        prepared
    }

    fn prepare_blurs(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        items: &[(ScissorRectangle, BlurKernel)],
        load: wgpu::LoadOp<wgpu::Color>,
    ) {
        let resources = self.resources();
        let horizontal = resources
            .blur_ping_view
            .as_ref()
            .expect("blur target was prepared");
        let vertical = resources
            .blur_pong_view
            .as_ref()
            .expect("blur target was prepared");
        let full_size = [self.target.width() as f32, self.target.height() as f32];
        let blur_size = [
            downsampled_dimension(self.target.width()) as f32,
            downsampled_dimension(self.target.height()) as f32,
        ];
        for (label, pipeline, target, source, axis) in [
            (
                "blur_downsample",
                &resources.pipelines.blur_downsample,
                horizontal,
                source,
                None,
            ),
            (
                "blur_horizontal",
                &resources.pipelines.blur,
                vertical,
                horizontal,
                Some(BlurAxis::Horizontal),
            ),
            (
                "blur_vertical",
                &resources.pipelines.blur,
                horizontal,
                vertical,
                Some(BlurAxis::Vertical),
            ),
        ] {
            let mut pass = begin_color_render_pass(encoder, label, target, load, None);
            for (scissor, kernel) in items {
                pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
                let uniforms = axis.map_or_else(
                    || BlurUniforms::downsample(full_size, blur_size),
                    |axis| BlurUniforms::gaussian(axis, blur_size, *kernel),
                );
                self.draw_filter(&mut pass, pipeline, source, uniforms);
            }
        }
    }

    /// Draws a prepared composite, then a lens's sharp rim, into `pass`, an open pass on
    /// the composite's target.
    pub(super) fn draw_composite(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        prepared: &wgpu::TextureView,
        source: &wgpu::TextureView,
        parameters: FilterCompositeParameters,
    ) {
        let full_size = [self.target.width() as f32, self.target.height() as f32];
        let (uniforms, pipeline) = self.composite_draw(parameters, full_size, 0.0);
        self.draw_filter(pass, pipeline, prepared, uniforms);
        if BlurKernel::for_radius(parameters.blur_radius).is_some()
            && parameters.lens.is_active()
            && parameters.lens.sharp_rim > 0.0
        {
            let (uniforms, pipeline) =
                self.composite_draw(parameters, full_size, parameters.lens.sharp_rim);
            self.draw_filter(pass, pipeline, source, uniforms);
        }
    }

    fn composite_draw(
        &self,
        parameters: FilterCompositeParameters,
        source_size: [f32; 2],
        sharp_rim: f32,
    ) -> (BlurUniforms, &pipelines::WgpuRenderPipeline) {
        let clips_to_bounds = matches!(parameters.clip, FilterCompositeClip::RoundedBounds);
        let composite_bounds = if clips_to_bounds {
            parameters.bounds
        } else {
            parameters.bounds.dilate(ScaledPixels(
                GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * parameters.blur_radius,
            ))
        };
        let mut uniforms = BlurUniforms::composite(
            composite_bounds,
            parameters.content_mask,
            parameters.corner_radii,
            parameters.corner_smoothing,
            parameters.opacity,
            parameters.clip,
            source_size,
            {
                let size = self.target.viewport_size();
                [size.width.0 as f32, size.height.0 as f32]
            },
        )
        .with_lens(parameters.lens)
        .with_composite_scale(parameters.scale, parameters.scale_origin)
        .with_sharp_rim_pass(sharp_rim);
        uniforms.composite_offset = wgsl_rs::std::vec2f(
            parameters.composite_offset.x.0,
            parameters.composite_offset.y.0,
        );
        uniforms.composite_rotation = parameters.composite_rotation;
        uniforms.motion_clip = parameters.motion_clip.into();
        let resources = self.resources();
        let pipeline = if uniforms.corner_smoothing > 0.0 {
            &resources.pipelines.smoothed_blur_composite
        } else {
            &resources.pipelines.blur_composite
        };

        (uniforms, pipeline)
    }

    /// Copies `region` of the backdrop (the part one filter reads, from
    /// `backdrop_snapshot_region`) into the snapshot texture: the first layer overwrites the
    /// region (a scissored copy, no clear), later layers composite over it. The rest of the
    /// texture keeps stale texels that are never read, so each lens costs in proportion to its
    /// own size instead of a whole-target clear and copy per lens.
    pub(super) fn snapshot_backdrop<'a>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        sources: impl Iterator<Item = &'a wgpu::TextureView>,
        region: Bounds<ScaledPixels>,
    ) -> wgpu::TextureView {
        let target = self
            .resources()
            .backdrop_snapshot_view
            .as_ref()
            .expect("backdrop snapshot was prepared")
            .clone();
        let full_size = [self.target.width() as f32, self.target.height() as f32];
        let bounds = region;
        let scissor = ScissorRectangle {
            x: region.origin.x.0 as u32,
            y: region.origin.y.0 as u32,
            width: region.size.width.0 as u32,
            height: region.size.height.0 as u32,
        };
        if scissor.is_empty() {
            return target;
        }

        for (index, source) in sources.enumerate() {
            if index == 0 {
                self.draw_filter_pass(
                    encoder,
                    "copy_backdrop_snapshot",
                    &self.resources().pipelines.blur_downsample,
                    &target,
                    source,
                    BlurUniforms::copy(full_size),
                    wgpu::LoadOp::Load,
                    Some(scissor),
                );
                continue;
            }
            let (uniforms, pipeline) = self.composite_draw(
                FilterCompositeParameters {
                    bounds,
                    content_mask: bounds,
                    corner_radii: Corners::default(),
                    corner_smoothing: 0.0,
                    blur_radius: 0.0,
                    opacity: 1.0,
                    scale: 1.0,
                    scale_origin: Point::default(),
                    composite_offset: Point::default(),
                    composite_rotation: 0.0,
                    clip: FilterCompositeClip::ContentShape,
                    motion_clip: Bounds::default(),
                    lens: Default::default(),
                },
                full_size,
                0.0,
            );
            self.draw_filter_pass(
                encoder,
                "blur_composite",
                pipeline,
                &target,
                source,
                uniforms,
                wgpu::LoadOp::Load,
                None,
            );
        }

        target
    }

    /// Copies all pane sampling regions into the snapshot in one render pass.
    pub(super) fn snapshot_backdrops(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        regions: &[Bounds<ScaledPixels>],
    ) -> wgpu::TextureView {
        let resources = self.resources();
        let target = resources
            .backdrop_snapshot_view
            .as_ref()
            .expect("backdrop snapshot was prepared")
            .clone();
        let full_size = [self.target.width() as f32, self.target.height() as f32];
        let mut pass = begin_color_render_pass(
            encoder,
            "copy_backdrop_snapshot",
            &target,
            wgpu::LoadOp::Load,
            None,
        );
        for region in regions {
            if region.is_empty() {
                continue;
            }
            pass.set_scissor_rect(
                region.origin.x.0 as u32,
                region.origin.y.0 as u32,
                region.size.width.0 as u32,
                region.size.height.0 as u32,
            );
            self.draw_filter(
                &mut pass,
                &resources.pipelines.blur_downsample,
                source,
                BlurUniforms::copy(full_size),
            );
        }
        drop(pass);
        target
    }

    pub(super) fn blit_to_frame(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        frame_view: &wgpu::TextureView,
    ) {
        let size = [self.target.width() as f32, self.target.height() as f32];

        self.draw_filter_pass(
            encoder,
            "scene_blit",
            &self.resources().pipelines.blur_downsample,
            frame_view,
            source,
            BlurUniforms::copy(size),
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            None,
        );
    }
}

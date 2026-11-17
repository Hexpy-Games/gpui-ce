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
        viewport: [u32; 2],
    ) {
        let mut pass = begin_color_render_pass(encoder, label, target, load, scissor, viewport);
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

    /// Runs the blur passes of a filter composite (unless `blurred_ready`) and returns the
    /// texture the composite reads with its content extent; `None` when the composite draws
    /// nothing. `frame_size` is the extent of the frame being encoded.
    pub(super) fn prepare_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        parameters: FilterCompositeParameters,
        blurred_ready: bool,
        [full_width, full_height]: [u32; 2],
    ) -> Option<(wgpu::TextureView, [f32; 2])> {
        if parameters.blur_radius <= 0.0
            && !parameters.lens.is_active()
            && matches!(parameters.clip, FilterCompositeClip::RoundedBounds)
        {
            return None;
        }

        let full_size = [full_width as f32, full_height as f32];
        let Some(kernel) = BlurKernel::for_radius(parameters.blur_radius) else {
            return Some((source.clone(), full_size));
        };
        let blur_extent = [
            downsampled_dimension(full_width),
            downsampled_dimension(full_height),
        ];
        let blur_size = [blur_extent[0] as f32, blur_extent[1] as f32];
        let dilation = GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * parameters.blur_radius;
        let scissor = ScissorRectangle::for_blurred_bounds(
            parameters.bounds,
            dilation,
            full_width,
            full_height,
        );
        if scissor.is_empty() {
            return None;
        }
        let load = match parameters.clip {
            FilterCompositeClip::RoundedBounds => wgpu::LoadOp::Load,
            FilterCompositeClip::ContentShape | FilterCompositeClip::MotionRounded => {
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
            }
        };

        let (horizontal_target, vertical_target) = {
            let resources = self.resources();
            match (
                resources.blur_ping_view.as_ref(),
                resources.blur_pong_view.as_ref(),
            ) {
                (Some(horizontal), Some(vertical)) => (horizontal.clone(), vertical.clone()),
                _ => return None,
            }
        };

        if !blurred_ready {
            self.draw_filter_pass(
                encoder,
                "blur_downsample",
                &self.resources().pipelines.blur_downsample,
                &horizontal_target,
                source,
                BlurUniforms::downsample(full_size, blur_size).with_source_extent(full_size),
                // No clear for backdrops: each pass writes its whole scissor, and the composite only
                // reads texels the passes wrote (the bounds lie 3 sigma inside the scissor), so
                // whatever an earlier lens left outside it is never sampled; a full clear per pass
                // per lens cost more than the passes. Content blurs composite out to the scissor's
                // edge, so they keep the clear (transparent outside).
                load,
                Some(scissor),
                blur_extent,
            );
            self.draw_filter_pass(
                encoder,
                "blur_horizontal",
                &self.resources().pipelines.blur,
                &vertical_target,
                &horizontal_target,
                BlurUniforms::gaussian(BlurAxis::Horizontal, blur_size, kernel)
                    .with_source_extent(blur_size),
                load,
                Some(scissor),
                blur_extent,
            );
            self.draw_filter_pass(
                encoder,
                "blur_vertical",
                &self.resources().pipelines.blur,
                &horizontal_target,
                &vertical_target,
                BlurUniforms::gaussian(BlurAxis::Vertical, blur_size, kernel)
                    .with_source_extent(blur_size),
                load,
                Some(scissor),
                blur_extent,
            );
        }

        Some((horizontal_target, blur_size))
    }

    /// Draws a prepared composite, then a lens's sharp rim, into `pass`, an open pass on
    /// the composite's target.
    pub(super) fn draw_composite(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        (prepared, prepared_extent): &(wgpu::TextureView, [f32; 2]),
        source: &wgpu::TextureView,
        parameters: FilterCompositeParameters,
        frame_size: [u32; 2],
    ) {
        let full_size = [frame_size[0] as f32, frame_size[1] as f32];
        let (uniforms, pipeline) =
            self.composite_draw(parameters, full_size, *prepared_extent, 0.0);
        self.draw_filter(pass, pipeline, prepared, uniforms);
        if BlurKernel::for_radius(parameters.blur_radius).is_some()
            && parameters.lens.is_active()
            && parameters.lens.sharp_rim > 0.0
        {
            let (uniforms, pipeline) =
                self.composite_draw(parameters, full_size, full_size, parameters.lens.sharp_rim);
            self.draw_filter(pass, pipeline, source, uniforms);
        }
    }

    fn composite_draw(
        &self,
        parameters: FilterCompositeParameters,
        frame_size: [f32; 2],
        source_extent: [f32; 2],
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
            frame_size,
            frame_size,
        )
        .with_lens(parameters.lens)
        .with_composite_scale(parameters.scale, parameters.scale_origin)
        .with_sharp_rim_pass(sharp_rim)
        .with_source_extent(source_extent);
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
        full_extent: [u32; 2],
    ) -> wgpu::TextureView {
        let target = self
            .resources()
            .backdrop_snapshot_view
            .as_ref()
            .expect("backdrop snapshot was prepared")
            .clone();
        let full_size = [full_extent[0] as f32, full_extent[1] as f32];
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
                    BlurUniforms::copy(full_size).with_source_extent(full_size),
                    wgpu::LoadOp::Load,
                    Some(scissor),
                    full_extent,
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
                full_extent,
            );
        }

        target
    }

    pub(super) fn blit_to_frame(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        frame_view: &wgpu::TextureView,
        frame_size: [u32; 2],
    ) {
        let size = [frame_size[0] as f32, frame_size[1] as f32];

        self.draw_filter_pass(
            encoder,
            "scene_blit",
            &self.resources().pipelines.blur_downsample,
            frame_view,
            source,
            BlurUniforms::copy(size).with_source_extent(size),
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            None,
            frame_size,
        );
    }
}

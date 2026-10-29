use gpui::{Bounds, Corners, ScaledPixels};
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

const _: () = assert!(std::mem::size_of::<BlurUniforms>() == 432);

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
        let resources = self.resources();
        let uniform_offset = resources.filter_uniforms.write(&uniforms);
        let bind_group = resources.blur_bind_group(source);
        let mut pass = begin_color_render_pass(encoder, label, target, load);
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

        if let Some(scissor) = scissor {
            pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
        }

        pass.draw(0..pipeline.fixed_vertex_count(), 0..1);
    }

    pub(super) fn blur_and_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        target: &wgpu::TextureView,
        parameters: FilterCompositeParameters,
    ) {
        if parameters.blur_radius <= 0.0
            && !parameters.lens.is_active()
            && matches!(parameters.clip, FilterCompositeClip::RoundedBounds)
        {
            return;
        }

        let full_size = [self.target.width() as f32, self.target.height() as f32];

        let Some(kernel) = BlurKernel::for_radius(parameters.blur_radius) else {
            self.composite_texture(encoder, source, target, parameters, full_size);

            return;
        };
        let full_width = self.target.width();
        let full_height = self.target.height();
        let blur_size = [
            downsampled_dimension(full_width) as f32,
            downsampled_dimension(full_height) as f32,
        ];
        let dilation = GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * parameters.blur_radius;
        let scissor = ScissorRectangle::for_blurred_bounds(
            parameters.bounds,
            dilation,
            full_width,
            full_height,
        );
        if scissor.is_empty() {
            return;
        }
        let load = match parameters.clip {
            FilterCompositeClip::RoundedBounds => wgpu::LoadOp::Load,
            FilterCompositeClip::ContentShape => wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        };

        let (horizontal_target, vertical_target) = {
            let resources = self.resources();
            match (
                resources.blur_ping_view.as_ref(),
                resources.blur_pong_view.as_ref(),
            ) {
                (Some(horizontal), Some(vertical)) => (horizontal.clone(), vertical.clone()),
                _ => return,
            }
        };

        self.draw_filter_pass(
            encoder,
            "blur_downsample",
            &self.resources().pipelines.blur_downsample,
            &horizontal_target,
            source,
            BlurUniforms::downsample([full_width as f32, full_height as f32], blur_size),
            // No clear for backdrops: each pass writes its whole scissor, and the composite only
            // reads texels the passes wrote (the bounds lie 3 sigma inside the scissor), so
            // whatever an earlier lens left outside it is never sampled; a full clear per pass
            // per lens cost more than the passes. Content blurs composite out to the scissor's
            // edge, so they keep the clear (transparent outside).
            load,
            Some(scissor),
        );
        self.draw_filter_pass(
            encoder,
            "blur_horizontal",
            &self.resources().pipelines.blur,
            &vertical_target,
            &horizontal_target,
            BlurUniforms::gaussian(BlurAxis::Horizontal, blur_size, kernel),
            load,
            Some(scissor),
        );
        self.draw_filter_pass(
            encoder,
            "blur_vertical",
            &self.resources().pipelines.blur,
            &horizontal_target,
            &vertical_target,
            BlurUniforms::gaussian(BlurAxis::Vertical, blur_size, kernel),
            load,
            Some(scissor),
        );

        self.composite_texture(encoder, &horizontal_target, target, parameters, blur_size);
        if parameters.lens.is_active() && parameters.lens.sharp_rim > 0.0 {
            self.composite_texture_with(
                encoder,
                source,
                target,
                parameters,
                full_size,
                parameters.lens.sharp_rim,
            );
        }
    }

    fn composite_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        target: &wgpu::TextureView,
        parameters: FilterCompositeParameters,
        source_size: [f32; 2],
    ) {
        self.composite_texture_with(encoder, source, target, parameters, source_size, 0.0);
    }

    /// `sharp_rim` > 0: the lens's sharp-rim pass (see `BlurUniforms::with_sharp_rim_pass`).
    fn composite_texture_with(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        target: &wgpu::TextureView,
        parameters: FilterCompositeParameters,
        source_size: [f32; 2],
        sharp_rim: f32,
    ) {
        let clips_to_bounds = matches!(parameters.clip, FilterCompositeClip::RoundedBounds);
        let composite_bounds = if clips_to_bounds {
            parameters.bounds
        } else {
            parameters.bounds.dilate(ScaledPixels(
                GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * parameters.blur_radius,
            ))
        };
        let uniforms = BlurUniforms::composite(
            composite_bounds,
            parameters.content_mask,
            parameters.corner_radii,
            parameters.corner_smoothing,
            parameters.opacity,
            parameters.clip,
            source_size,
            [self.target.width() as f32, self.target.height() as f32],
        )
        .with_lens(parameters.lens)
        .with_sharp_rim_pass(sharp_rim);
        let resources = self.resources();
        let pipeline = if uniforms.corner_smoothing > 0.0 {
            &resources.pipelines.smoothed_blur_composite
        } else {
            &resources.pipelines.blur_composite
        };

        self.draw_filter_pass(
            encoder,
            "blur_composite",
            pipeline,
            target,
            source,
            uniforms,
            wgpu::LoadOp::Load,
            None,
        );
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
            self.composite_texture(
                encoder,
                source,
                &target,
                FilterCompositeParameters {
                    bounds,
                    content_mask: bounds,
                    corner_radii: Corners::default(),
                    corner_smoothing: 0.0,
                    blur_radius: 0.0,
                    opacity: 1.0,
                    clip: FilterCompositeClip::ContentShape,
                    lens: Default::default(),
                },
                full_size,
            );
        }

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

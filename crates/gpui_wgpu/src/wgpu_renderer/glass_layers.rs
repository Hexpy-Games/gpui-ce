//! Preparation of sibling glass panes before their scene-pass composites.
use super::frame::FrameEncoder;
use gpui::{BackdropFilter, BackdropLens, Bounds, Point, backdrop_appear_parameters};
use gpui_render::{
    blur::{FilterCompositeClip, FilterCompositeParameters, backdrop_snapshot_region},
    glass_layer::GlassLayer,
};
use smallvec::SmallVec;

pub(super) struct LayerPanes {
    pub(super) panes: SmallVec<[(usize, Option<wgpu::TextureView>, FilterCompositeParameters); 4]>,
    pub(super) snapshot: wgpu::TextureView,
}

impl FrameEncoder<'_> {
    pub(super) fn prepare_layer(&mut self, layer: &GlassLayer) {
        let parameters: SmallVec<[_; 4]> = layer
            .panes
            .iter()
            .map(|index| {
                let filter = &self.scene.backdrop_filters[*index];
                let (lens, radius) =
                    backdrop_appear_parameters(filter, self.renderer.animation_time_ms);
                backdrop_parameters(filter, lens, radius)
            })
            .collect();
        let regions: SmallVec<[_; 4]> = parameters
            .iter()
            .map(|parameters| {
                backdrop_snapshot_region(
                    parameters.bounds,
                    parameters.blur_radius,
                    &parameters.lens,
                    self.renderer.target.width(),
                    self.renderer.target.height(),
                )
            })
            .collect();
        let snapshot =
            self.renderer
                .snapshot_backdrops(&mut self.encoder, self.targets.current(), &regions);
        let prepared =
            self.renderer
                .prepare_composites(&mut self.encoder, &snapshot, &parameters, false);
        self.layer_panes = Some(LayerPanes {
            panes: layer
                .panes
                .iter()
                .copied()
                .zip(prepared)
                .zip(parameters)
                .map(|((index, view), parameters)| (index, view, parameters))
                .collect(),
            snapshot,
        });
    }
}

pub(super) fn backdrop_parameters(
    filter: &BackdropFilter,
    lens: BackdropLens,
    radius: f32,
) -> FilterCompositeParameters {
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
    }
}

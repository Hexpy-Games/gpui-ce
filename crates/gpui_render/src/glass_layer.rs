//! Sibling backdrop filters that sample and composite as one glass layer.
use crate::blur::{
    BlurKernel, GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS, ScissorRectangle, backdrop_snapshot_region,
};
use gpui::{Bounds, PaintedBounds, PrimitiveBatch, RenderCommand, ScaledPixels, Scene};
use smallvec::SmallVec;

/// Sibling backdrop filters prepared and composited as one layer.
pub struct GlassLayer {
    /// `scene.backdrop_filters` indices of the panes, in paint order (two or more).
    pub panes: SmallVec<[usize; 4]>,
    /// Command indices of later panes' drop shadows, painted before the first pane.
    pub shadows: SmallVec<[usize; 2]>,
    /// Everything the layer paints or samples: pane regions and moved shadows.
    pub bounds: Bounds<ScaledPixels>,
}

struct Pane {
    index: usize,
    bounds: Bounds<ScaledPixels>,
    region: Bounds<ScaledPixels>,
    radius: f32,
    scissor: Option<ScissorRectangle>,
}

#[derive(Default)]
struct OpenLayer {
    panes: Vec<Pane>,
    foreign: Vec<Bounds<ScaledPixels>>,
    pending: Vec<(usize, usize, Vec<Bounds<ScaledPixels>>)>,
    claimed: SmallVec<[usize; 2]>,
}

/// The glass layers of `commands` at `now_ms` for a `width` x `height` target.
pub fn glass_layers(
    scene: &Scene,
    commands: &[RenderCommand],
    now_ms: u32,
    width: u32,
    height: u32,
) -> Vec<GlassLayer> {
    let mut layers = Vec::new();
    let mut open = OpenLayer::default();
    let mut depth = 0;
    for (command_index, command) in commands.iter().enumerate() {
        match command {
            RenderCommand::BeginFilter { .. } => {
                close(&mut open, &mut layers);
                depth += 1;
            }
            RenderCommand::EndFilter { .. } => {
                close(&mut open, &mut layers);
                depth -= 1;
            }
            RenderCommand::Batch(PrimitiveBatch::BackdropFilters(range)) if depth == 0 => {
                for index in range.clone() {
                    let filter = &scene.backdrop_filters[index];
                    if filter.wallpaper.is_some() {
                        close(&mut open, &mut layers);
                        continue;
                    }
                    let pane = pane(scene, index, now_ms, width, height);
                    if !joins(&open, &pane) {
                        close(&mut open, &mut layers);
                    }
                    for (index, _, shadows) in &open.pending {
                        if shadows.iter().all(|shadow| inside(pane.bounds, *shadow))
                            && !open.claimed.contains(index)
                        {
                            open.claimed.push(*index);
                        }
                    }
                    open.panes.push(pane);
                }
            }
            RenderCommand::Batch(PrimitiveBatch::Shadows { range, .. })
                if !open.panes.is_empty() =>
            {
                let shadows: Vec<_> = scene.shadows[range.clone()]
                    .iter()
                    .map(PaintedBounds::painted_bounds)
                    .collect();
                if !shadows.iter().all(|bounds| on_pane(&open, *bounds)) {
                    open.pending
                        .push((command_index, open.foreign.len(), shadows));
                }
            }
            RenderCommand::Batch(batch) if !open.panes.is_empty() => {
                let foreign = batch_bounds(scene, batch)
                    .into_iter()
                    .filter(|bounds| !bounds.is_empty() && !on_pane(&open, *bounds))
                    .reduce(|a, b| a.union(&b));
                if let Some(bounds) = foreign {
                    open.foreign.push(bounds);
                }
            }
            _ => {}
        }
    }
    close(&mut open, &mut layers);
    layers
}

fn pane(scene: &Scene, index: usize, now_ms: u32, width: u32, height: u32) -> Pane {
    let filter = &scene.backdrop_filters[index];
    let (lens, radius) = gpui::backdrop_appear_parameters(filter, now_ms);
    Pane {
        index,
        bounds: filter.painted_bounds(),
        radius,
        region: backdrop_snapshot_region(filter.bounds, radius, &lens, width, height),
        scissor: BlurKernel::for_radius(radius).map(|_| {
            ScissorRectangle::for_blurred_bounds(
                filter.bounds,
                GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * radius,
                width,
                height,
            )
        }),
    }
}

fn inside(inner: Bounds<ScaledPixels>, outer: Bounds<ScaledPixels>) -> bool {
    inner.origin.x >= outer.origin.x
        && inner.origin.y >= outer.origin.y
        && inner.right() <= outer.right()
        && inner.bottom() <= outer.bottom()
}

fn on_pane(open: &OpenLayer, bounds: Bounds<ScaledPixels>) -> bool {
    open.panes.iter().any(|pane| inside(bounds, pane.bounds))
}

fn joins(open: &OpenLayer, pane: &Pane) -> bool {
    if open
        .panes
        .iter()
        .any(|other| pane.bounds.intersects(&other.bounds))
        || open
            .foreign
            .iter()
            .any(|bounds| pane.region.intersects(bounds))
    {
        return false;
    }
    for (_, before, shadows) in &open.pending {
        if shadows.iter().all(|shadow| inside(pane.bounds, *shadow)) {
            if shadows.iter().any(|shadow| {
                open.foreign[..*before]
                    .iter()
                    .any(|foreign| shadow.intersects(foreign))
            }) {
                return false;
            }
        } else if shadows.iter().any(|shadow| shadow.intersects(&pane.region)) {
            return false;
        }
    }
    open.panes
        .iter()
        .all(|other| match (pane.scissor, other.scissor) {
            (Some(a), Some(b)) => {
                pane.radius.to_bits() == other.radius.to_bits()
                    || a.x >= b.x + b.width
                    || b.x >= a.x + a.width
                    || a.y >= b.y + b.height
                    || b.y >= a.y + a.height
            }
            _ => true,
        })
}

fn close(open: &mut OpenLayer, layers: &mut Vec<GlassLayer>) {
    let mut closed = std::mem::take(open);
    if closed.panes.len() < 2 {
        return;
    }
    closed.claimed.sort_unstable();
    let mut bounds = closed.panes[0].region;
    for pane in &closed.panes[1..] {
        bounds = bounds.union(&pane.region);
    }
    for (index, _, shadows) in &closed.pending {
        if closed.claimed.contains(index) {
            for shadow in shadows {
                bounds = bounds.union(shadow);
            }
        }
    }
    layers.push(GlassLayer {
        panes: closed.panes.iter().map(|pane| pane.index).collect(),
        shadows: closed.claimed,
        bounds,
    });
}

fn batch_bounds(scene: &Scene, batch: &PrimitiveBatch) -> Vec<Bounds<ScaledPixels>> {
    fn painted<T: PaintedBounds>(items: &[T]) -> Vec<Bounds<ScaledPixels>> {
        items.iter().map(PaintedBounds::painted_bounds).collect()
    }
    match batch {
        PrimitiveBatch::Quads { range, .. } => painted(&scene.quads[range.clone()]),
        PrimitiveBatch::Paths { range, .. } => painted(&scene.paths[range.clone()]),
        PrimitiveBatch::Underlines(range) => painted(&scene.underlines[range.clone()]),
        PrimitiveBatch::MonochromeSprites { range, .. } => {
            painted(&scene.monochrome_sprites[range.clone()])
        }
        PrimitiveBatch::SubpixelSprites { range, .. } => {
            painted(&scene.subpixel_sprites[range.clone()])
        }
        PrimitiveBatch::PolychromeSprites { range, .. } => {
            painted(&scene.polychrome_sprites[range.clone()])
        }
        PrimitiveBatch::Surfaces(range) => painted(&scene.surfaces[range.clone()]),
        _ => Vec::new(),
    }
}

/// Moves every layer's shadows just before its first pane, splitting backdrop batches as needed.
pub fn layered_commands(commands: &[RenderCommand], layers: &[GlassLayer]) -> Vec<RenderCommand> {
    let mut result = Vec::new();
    for (index, command) in commands.iter().enumerate() {
        if layers.iter().any(|layer| layer.shadows.contains(&index)) {
            continue;
        }
        if let RenderCommand::Batch(PrimitiveBatch::BackdropFilters(range)) = command {
            let mut start = range.start;
            for pane in range.clone() {
                if let Some(layer) = layers.iter().find(|layer| layer.panes[0] == pane) {
                    if start < pane {
                        result.push(RenderCommand::Batch(PrimitiveBatch::BackdropFilters(
                            start..pane,
                        )));
                    }
                    result.extend(layer.shadows.iter().map(|index| commands[*index].clone()));
                    start = pane;
                }
            }
            if start < range.end {
                result.push(RenderCommand::Batch(PrimitiveBatch::BackdropFilters(
                    start..range.end,
                )));
            }
        } else {
            result.push(command.clone());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    // test-category: pure-logic
    use super::*;
    use gpui::{
        BackdropFilter, ContentMask, Quad, ScaledFilter, Shadow, bounds, point, size,
        solid_background,
    };

    #[test]
    fn sibling_panes_share_a_layer_until_one_overlaps() {
        let rect = |x, y, w, h| {
            bounds(
                point(ScaledPixels(x), ScaledPixels(y)),
                size(ScaledPixels(w), ScaledPixels(h)),
            )
        };
        let mask = ContentMask {
            bounds: rect(0.0, 0.0, 1000.0, 1000.0),
            ..Default::default()
        };
        let pane = |bounds| BackdropFilter {
            bounds,
            content_mask: mask,
            filters: [ScaledFilter::Blur(ScaledPixels(8.0))]
                .into_iter()
                .collect(),
            opacity: 1.0,
            ..Default::default()
        };
        let mut scene = Scene::default();
        scene.insert_primitive(Quad {
            bounds: mask.bounds,
            content_mask: mask,
            background: solid_background(gpui::black()),
            ..Default::default()
        });
        scene.insert_primitive(pane(rect(100.0, 100.0, 200.0, 100.0)));
        scene.insert_primitive(Quad {
            bounds: rect(120.0, 120.0, 20.0, 20.0),
            content_mask: mask,
            background: solid_background(gpui::white()),
            ..Default::default()
        });
        scene.insert_primitive(Shadow {
            bounds: rect(320.0, 100.0, 200.0, 100.0),
            content_mask: mask,
            blur_radius: ScaledPixels(10.0),
            order: 0,
            corner_radii: Default::default(),
            color: gpui::black().into(),
            element_bounds: rect(320.0, 100.0, 200.0, 100.0),
            element_corner_radii: Default::default(),
            inset: gpui::ShaderBool::Disabled,
            corner_smoothing: 0.0,
            clip_outside: gpui::ShaderBool::Disabled,
            padding: 0,
        });
        scene.insert_primitive(pane(rect(320.0, 100.0, 200.0, 100.0)));
        scene.insert_primitive(pane(rect(400.0, 150.0, 200.0, 100.0)));
        scene.finish();
        let commands = scene.render_commands();
        let shadow = commands
            .iter()
            .position(|command| {
                matches!(
                    command,
                    RenderCommand::Batch(PrimitiveBatch::Shadows { .. })
                )
            })
            .unwrap();
        let layers = glass_layers(&scene, commands, 0, 1000, 1000);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].panes.as_slice(), &[0, 1]);
        assert_eq!(layers[0].shadows.as_slice(), &[shadow]);
        let reordered = layered_commands(commands, &layers);
        let shadow_position = reordered
            .iter()
            .position(|command| command == &commands[shadow])
            .unwrap();
        let pane_position = reordered
            .iter()
            .position(|command| {
                matches!(
                    command,
                    RenderCommand::Batch(PrimitiveBatch::BackdropFilters(_))
                )
            })
            .unwrap();
        assert!(shadow_position < pane_position);
    }
}

// Added by HexpyGames for Butler (2026), under the Apache License 2.0; see NOTICE.
//! A frame drawn into intermediates a larger frame left behind must equal a fresh render.
#![cfg(feature = "test-support")]

use gpui::{
    BackdropFilter, BackdropLens, Background, Bounds, ContentMask, DevicePixels, FilterBoundary,
    LayerMotion, PlatformHeadlessRenderer, Point, Quad, ScaledFilter, ScaledPixels, Scene, Size,
    linear_color_stop, linear_gradient, solid_background,
};
use gpui_ce_wgpu::WgpuHeadlessRenderer;
use smallvec::smallvec;

fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
    Bounds {
        origin: Point {
            x: ScaledPixels(x),
            y: ScaledPixels(y),
        },
        size: Size {
            width: ScaledPixels(w),
            height: ScaledPixels(h),
        },
    }
}

fn color(hex: u32) -> gpui::Hsla {
    gpui::rgb_to_hsla(gpui::rgb(hex))
}

/// A full-frame background, a lens glass touching the right and bottom edges (its blur taps
/// cross the content edge), and a plain edge glass inside an opacity group.
fn scene(target: Size<DevicePixels>, background: Background) -> Scene {
    let (w, h) = (target.width.0 as f32, target.height.0 as f32);
    let frame = bounds(0.0, 0.0, w, h);
    let mask = ContentMask {
        bounds: frame,
        ..Default::default()
    };
    let glass = |glass_bounds, lens| BackdropFilter {
        order: 0,
        bounds: glass_bounds,
        content_mask: mask,
        corner_radii: Default::default(),
        corner_smoothing: 0.0,
        filters: smallvec![ScaledFilter::Blur(ScaledPixels(4.0))],
        opacity: 1.0,
        lens,
        appear: None,
        wallpaper: None,
    };
    let group = FilterBoundary {
        order: 0,
        bounds: frame,
        content_mask: mask,
        corner_radii: Default::default(),
        corner_smoothing: 0.0,
        filters: smallvec![],
        opacity: 0.5,
        scale: 1.0,
        scale_origin: Point::default(),
        motion: LayerMotion::NONE,
        motion_bounds: frame,
        is_start: true,
        inline: false,
    };

    let mut scene = Scene::default();
    scene.insert_primitive(Quad {
        bounds: frame,
        content_mask: mask,
        background,
        ..Default::default()
    });
    // The lens of `gpui::scene::damage`'s partial-redraw test.
    let lens = BackdropLens {
        bezel: 4.0,
        ..Default::default()
    };
    scene.insert_primitive(glass(bounds(w - 100.0, h - 60.0, 100.0, 60.0), lens));
    scene.insert_primitive(group.clone());
    scene.insert_primitive(Quad {
        bounds: bounds(20.0, 20.0, 80.0, 60.0),
        content_mask: mask,
        background: solid_background(color(0xffffff)),
        ..Default::default()
    });
    scene.insert_primitive(glass(
        bounds(w - 60.0, h - 100.0, 60.0, 100.0),
        BackdropLens::default(),
    ));
    scene.insert_primitive(FilterBoundary {
        is_start: false,
        ..group
    });
    scene.finish();
    scene
}

// test-category: pure-logic
#[test]
fn smaller_frame_on_reused_targets_matches_fresh_render() {
    let large = Size {
        width: DevicePixels(320),
        height: DevicePixels(200),
    };
    // Odd sizes leave a half-resolution texel half covered at the right and bottom edges.
    let small = Size {
        width: DevicePixels(301),
        height: DevicePixels(181),
    };
    // Grows past the larger frame: the intermediates are reallocated in 256 px buckets.
    let grown = Size {
        width: DevicePixels(517),
        height: DevicePixels(301),
    };
    let gradient = || {
        linear_gradient(
            90.0,
            linear_color_stop(color(0xd0402a), 0.0),
            linear_color_stop(color(0x2a60d0), 1.0),
        )
    };

    // The larger red frame leaves red in every intermediate beyond the smaller frame's edge.
    let mut reused = WgpuHeadlessRenderer::new().expect("headless renderer");
    reused
        .render_scene_to_image(&scene(large, solid_background(color(0xff0000))), large)
        .expect("larger frame");
    for size in [small, grown] {
        let actual = reused
            .render_scene_to_image(&scene(size, gradient()), size)
            .expect("frame on reused targets");

        let mut fresh = WgpuHeadlessRenderer::new().expect("headless renderer");
        let expected = fresh
            .render_scene_to_image(&scene(size, gradient()), size)
            .expect("frame on fresh targets");

        assert!(actual.as_raw() == expected.as_raw(), "{size:?} differs");
    }
}

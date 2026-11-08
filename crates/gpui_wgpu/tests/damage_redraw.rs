// Added by HexpyGames for Butler (2026), under the Apache License 2.0; see NOTICE.
//! An animation tick drawn as a damage redraw must equal a full redraw of the same scene.
#![cfg(feature = "test-support")]

use gpui::{
    AtlasKey, AtlasTile, BackdropFilter, Bounds, ContentMask, DevicePixels, FrameLoop, Hsla,
    ImageId, MonochromeSprite, PlatformHeadlessRenderer, Point, PolychromeSprite, Quad,
    RenderImageParams, RenderSvgParams, ScaledFilter, ScaledPixels, Scene, Size, checkerboard,
    linear_color_stop, linear_gradient, solid_background,
};
use gpui_ce_wgpu::WgpuHeadlessRenderer;
use smallvec::smallvec;
use std::borrow::Cow;

const TARGET: Size<DevicePixels> = Size {
    width: DevicePixels(128),
    height: DevicePixels(48),
};

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

fn mask() -> ContentMask<ScaledPixels> {
    ContentMask {
        bounds: bounds(0.0, 0.0, 128.0, 48.0),
        ..Default::default()
    }
}

fn color(hex: u32) -> Hsla {
    gpui::rgb_to_hsla(gpui::rgb(hex))
}

/// A 2x2 grid of `cell`-sized cells (a 1 px transparent gutter inside each); `pixel(frame, x, y)`
/// gives the bytes of the pixel at `(x, y)` of the cell's content.
fn flipbook_tile(
    renderer: &WgpuHeadlessRenderer,
    key: AtlasKey,
    content: usize,
    bytes_per_pixel: usize,
    pixel: impl Fn(usize, usize, usize) -> Vec<u8>,
) -> AtlasTile {
    let cell = content + 2;
    let side = cell * 2;
    let mut bytes = vec![0u8; side * side * bytes_per_pixel];
    for frame in 0..4 {
        let (column, row) = (frame % 2, frame / 2);
        for y in 0..content {
            for x in 0..content {
                let index = (row * cell + y + 1) * side + column * cell + x + 1;
                let offset = index * bytes_per_pixel;
                bytes[offset..offset + bytes_per_pixel].copy_from_slice(&pixel(frame, x, y));
            }
        }
    }
    renderer
        .sprite_atlas()
        .get_or_insert_with(&key, &mut || {
            Ok(Some((
                Size {
                    width: DevicePixels(side as i32),
                    height: DevicePixels(side as i32),
                },
                Cow::Owned(bytes.clone()),
            )))
        })
        .expect("atlas insert must succeed")
        .expect("atlas insert must produce a tile")
}

fn glass(glass_bounds: Bounds<ScaledPixels>) -> BackdropFilter {
    BackdropFilter {
        order: 0,
        bounds: glass_bounds,
        content_mask: mask(),
        corner_radii: Default::default(),
        corner_smoothing: 0.0,
        filters: smallvec![ScaledFilter::Blur(ScaledPixels(4.0))],
        opacity: 1.0,
        lens: Default::default(),
        wallpaper: None,
    }
}

fn animated_scene(renderer: &WgpuHeadlessRenderer) -> Scene {
    // Frame k of the mask lights the vertical stripe x in [4k, 4k + 4).
    let mono = flipbook_tile(
        renderer,
        AtlasKey::Svg(RenderSvgParams {
            path: "damage-mono".into(),
            size: Size {
                width: DevicePixels(16),
                height: DevicePixels(16),
            },
        }),
        16,
        1,
        |frame, x, _| vec![if x / 4 == frame { 255 } else { 0 }],
    );
    // Frame k is one opaque BGRA colour.
    let poly = flipbook_tile(
        renderer,
        AtlasKey::Image(RenderImageParams {
            image_id: ImageId(7),
            frame_index: 0,
        }),
        12,
        4,
        |frame, _, _| match frame {
            0 => vec![0, 0, 255, 255],
            1 => vec![0, 255, 0, 255],
            2 => vec![255, 0, 0, 255],
            _ => vec![0, 255, 255, 255],
        },
    );
    let flipbook = |columns| FrameLoop {
        frame_count: 4,
        columns,
        period_ms: 400,
        start_ms: 0,
        ..FrameLoop::NONE
    };

    let mut scene = Scene::default();
    scene.insert_primitive(Quad {
        bounds: bounds(0.0, 0.0, 128.0, 48.0),
        content_mask: mask(),
        background: linear_gradient(
            90.0,
            linear_color_stop(color(0xd0402a), 0.0),
            linear_color_stop(color(0x2a60d0), 1.0),
        ),
        ..Default::default()
    });
    scene.insert_primitive(Quad {
        bounds: bounds(8.0, 8.0, 40.0, 32.0),
        content_mask: mask(),
        background: checkerboard(color(0xffffff), 3.0),
        ..Default::default()
    });
    scene.insert_primitive(glass(bounds(8.0, 8.0, 40.0, 32.0)));
    scene.insert_primitive(MonochromeSprite {
        order: 0,
        padding: 0,
        bounds: bounds(16.0, 16.0, 16.0, 16.0),
        content_mask: mask(),
        color: color(0xffffff).into(),
        tile: mono,
        transformation: Default::default(),
        frames: flipbook(2),
    });
    scene.insert_primitive(PolychromeSprite {
        order: 0,
        grayscale: gpui::ShaderBool::Disabled,
        opacity: 1.0,
        corner_smoothing: 0.0,
        bounds: bounds(84.0, 14.0, 12.0, 12.0),
        content_mask: mask(),
        corner_radii: Default::default(),
        tile: poly,
        frames: flipbook(2),
    });
    scene.insert_primitive(glass(bounds(78.0, 8.0, 40.0, 32.0)));
    scene.insert_primitive(Quad {
        bounds: bounds(300.0, 300.0, 10.0, 10.0),
        content_mask: mask(),
        background: solid_background(color(0x00ff00)),
        ..Default::default()
    });
    scene.finish();
    scene
}

// test-category: pure-logic
#[test]
fn damage_redraw_matches_full_redraw() {
    let mut renderer = WgpuHeadlessRenderer::new().expect("headless renderer");
    let scene = animated_scene(&renderer);

    let mut first_frames = Vec::new();
    for (previous, now) in [(0, 125), (125, 250)] {
        renderer.set_animation_time_ms(previous);
        let before = renderer
            .render_scene_to_image(&scene, TARGET)
            .expect("full frame");
        renderer.set_animation_time_ms(now);
        let damage = renderer
            .render_animation_frame_to_image(&scene, TARGET)
            .expect("the tick must be a damage redraw");
        let full = renderer
            .render_scene_to_image(&scene, TARGET)
            .expect("full frame");
        assert_eq!(
            damage.as_raw(),
            full.as_raw(),
            "damage redraw at {now} ms differs from the full redraw"
        );
        if first_frames.is_empty() {
            first_frames = vec![before, full];
        }
    }

    let [frame0, frame1] = &first_frames[..] else {
        unreachable!("the first pair stores both images");
    };
    let differs = (16..32)
        .any(|y| (16..32).any(|x| frame0.get_pixel(x, y) != frame1.get_pixel(x, y)));
    assert!(differs, "the first sprite must change between frames 0 and 1");
}

#![cfg(feature = "test-support")]
// test-category: pure-logic

use gpui::{
    BackdropFilter, Bounds, ContentMask, DevicePixels, PlatformHeadlessRenderer, ScaledPixels,
    Scene, WallpaperPaint, WallpaperPass, point, size,
};
use gpui_ce_wgpu::WgpuHeadlessRenderer;
use smallvec::smallvec;

#[test]
fn intro_resting_print_clears_card_and_keeps_edge_ink() {
    let mut renderer = WgpuHeadlessRenderer::new().expect("headless renderer");
    let target = size(DevicePixels(1200), DevicePixels(900));
    let css = [1200.0_f32, 900.0_f32];
    let density = [(css[0] / 4.0).ceil() as u32, (css[1] / 4.0).ceil() as u32];
    let card = [390.0, 180.0, 420.0, 344.0];
    let mut density_uniforms = [[0.0; 4]; 15];
    density_uniforms[1] = [density[0] as f32, density[1] as f32, 1.0, 2.9];
    density_uniforms[4] = [css[0], css[1], 0.0, 0.0];
    density_uniforms[9] = card;
    density_uniforms[10] = [
        600.0,
        414.0,
        (0.4 * css[1]).clamp(140.0, 240.0),
        0.55 * css[0],
    ];
    density_uniforms[11] = [1.9, 1.0, 1.0, 1.0];
    density_uniforms[12] = [1.0, 1.0, 1.2, 1.0];
    density_uniforms[13] = [0.0, 600.0, 236.0, 64.0];

    let mut print_uniforms = [[0.0; 4]; 15];
    print_uniforms[1] = [css[0], css[1], 1.0, 2.9];
    print_uniforms[4] = [css[0], css[1], 0.0, 0.0];
    print_uniforms[5] = [0.0, density[0] as f32, density[1] as f32, 0.0];
    // The default day edition (paper = 0) uses RGB (250, 250, 248).
    print_uniforms[9] = [0.0, 0.0, 0.0, 0.0];

    let bounds = Bounds::new(
        point(ScaledPixels(0.0), ScaledPixels(0.0)),
        size(ScaledPixels(css[0]), ScaledPixels(css[1])),
    );
    let mut scene = Scene::default();
    scene.insert_primitive(BackdropFilter {
        order: 0,
        bounds,
        content_mask: ContentMask {
            bounds,
            ..Default::default()
        },
        corner_radii: Default::default(),
        corner_smoothing: 0.0,
        filters: smallvec![],
        opacity: 1.0,
        lens: Default::default(),
        appear: None,
        wallpaper: Some(WallpaperPaint {
            key: 1,
            generation: 0,
            passes: smallvec![
                WallpaperPass {
                    module: 6,
                    width: density[0],
                    height: density[1],
                    source: 1,
                    uniforms: density_uniforms,
                },
                WallpaperPass {
                    module: 5,
                    width: target.width.0 as u32,
                    height: target.height.0 as u32,
                    source: 2,
                    uniforms: print_uniforms,
                },
            ],
        }),
    });
    scene.finish();
    let image = renderer
        .render_scene_to_image(&scene, target)
        .expect("render intro resting print");

    let paper = [250_u8, 250, 248];
    let mut band_pixels = 0;
    let mut band_ink = 0;
    let mut far_pixels = 0;
    let mut far_ink = 0;
    for (x, y, pixel) in image.enumerate_pixels() {
        let p = [x as f32 + 0.5, y as f32 + 0.5];
        if p[0] >= card[0] + 40.0
            && p[0] <= card[0] + card[2] - 40.0
            && p[1] >= card[1] + 40.0
            && p[1] <= card[1] + card[3] - 40.0
        {
            assert!(
                pixel.0[..3]
                    .iter()
                    .zip(paper)
                    .all(|(&actual, expected)| actual.abs_diff(expected) <= 6),
                "card interior at ({x}, {y}): {pixel:?}"
            );
        }
        let bq = [
            (p[0] - (card[0] + card[2] * 0.5)).abs() - card[2] * 0.5,
            (p[1] - (card[1] + card[3] * 0.5)).abs() - card[3] * 0.5,
        ];
        let sd_c = bq[0].max(0.0).hypot(bq[1].max(0.0)) + bq[0].max(bq[1]).min(0.0);
        let inked = pixel.0[..3].iter().any(|&channel| channel < 230);
        if (0.0..=40.0).contains(&sd_c) {
            band_pixels += 1;
            band_ink += usize::from(inked);
        }
        if sd_c > 420.0 {
            far_pixels += 1;
            far_ink += usize::from(inked);
        }
    }
    let band_fraction = band_ink as f64 / band_pixels as f64;
    let far_fraction = far_ink as f64 / far_pixels as f64;
    assert!(
        band_fraction < 0.005,
        "card band ink fraction: {band_fraction}"
    );
    assert!(far_fraction > 0.02, "far ink fraction: {far_fraction}");
    assert!(
        far_fraction > band_fraction,
        "far ink must exceed card band ink"
    );

    // Change the generation so both passes run again instead of reusing the cached print.
    scene.backdrop_filters[0]
        .wallpaper
        .as_mut()
        .unwrap()
        .generation += 1;
    let repeated = renderer
        .render_scene_to_image(&scene, target)
        .expect("render intro resting print again");
    assert_eq!(image.as_raw(), repeated.as_raw());
}

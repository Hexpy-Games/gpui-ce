#![cfg(feature = "test-support")]
// test-category: pure-logic

use gpui::{
    BackdropFilter, Bounds, ContentMask, DevicePixels, PlatformHeadlessRenderer, ScaledPixels,
    Scene, WallpaperImage, WallpaperPaint, WallpaperPass, point, size,
};
use gpui_ce_wgpu::WgpuHeadlessRenderer;
use smallvec::smallvec;

#[test]
fn intro_resting_print_clears_card_and_keeps_edge_ink() {
    let _ = env_logger::builder().is_test(true).filter_level(log::LevelFilter::Warn).try_init();
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
            clock: None,
            image: None,
            passes: smallvec![
                WallpaperPass {
                    module: 6,
                    width: density[0],
                    height: density[1],
                    noise: 1,
                    previous: false,
                    image: false,
                    uniforms: density_uniforms,
                },
                WallpaperPass {
                    module: 5,
                    width: target.width.0 as u32,
                    height: target.height.0 as u32,
                    noise: 1,
                    previous: true,
                    image: false,
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

// Manifest-order parameter packing: colors occupy complete vec4s, scalar values fill xyzw.
// Entries 4/5 share riso-print defaults; 6 uses introParams.ts at two seconds (no params).
const DEFAULT_PARAMS: [[[f32; 4]; 6]; 18] = [
    // 1 bloom
    [
        [0.19607843, 0.25882353, 0.3019608, 0.0],
        [0.33333333, 0.3647059, 0.4862745, 0.0],
        [0.28235294, 0.36078431, 0.4392157, 0.0],
        [0.41568627, 0.49019608, 0.6039216, 0.0],
        [0.3254902, 0.4392157, 0.5529412, 0.0],
        [0.2627451, 0.3019608, 0.4392157, 0.0],
    ],
    // 2 dusk
    [
        [0.45, 0.6, 0.5, 0.6],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    // 3 shoreline
    [
        [0.0, 1.0, 0.5, 0.6],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    // 4 riso-print
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    // 5 riso-print
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    // 6 intro: card, ring/scale, flow/bloom/residue/cardOn, draw/dry/sweep/clear, mark, pointer.
    [
        [20.8, 8.0, 22.4, 15.2],
        [32.0, 18.4, 1.4e+02, 35.2],
        [1.8495947, 1.0, 0.8999236, 0.451617824],
        [1.0, 1.0, 1.2, 1.0],
        [1.0, 32.0, 11.1712045, 71.32644],
        [0.0, 0.0, 0.0, 0.0],
    ],
    // 7 silk
    [
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    // 8 riso-flow
    [
        [0.0, 9.0, 2.0, 3.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    // 9 lamina
    [
        [1.0, 1.5, 1.0, 0.35],
        [0.0, 0.9, 0.35, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    // 10 diatom
    [
        [0.0, 0.0, 0.0, 0.97],
        [1.08, 0.45, 0.117, 0.35],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    // 11 diatom
    [
        [0.0, 0.0, 0.0, 0.97],
        [1.08, 0.45, 0.117, 0.35],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    // 12 photo-clouds
    [
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    // 13 photo-daisies
    [
        [1.0, 1.0, 0.1, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    // 14 stipple
    [
        [0.0, 0.15, 0.0, 0.5],
        [0.55, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    // 15 stipple
    [
        [0.0, 0.15, 0.0, 0.5],
        [0.55, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    // 16 grain
    [
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    // 17 image
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    // 18 cherry-blossom
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
];

#[test]
fn every_module_renders_a_finite_varied_still() {
    let _ = env_logger::builder().is_test(true).filter_level(log::LevelFilter::Warn).try_init();
    let mut renderer = WgpuHeadlessRenderer::new().expect("headless renderer");
    let target = size(DevicePixels(64), DevicePixels(40));
    let rgba: Vec<u8> = (0..8)
        .flat_map(|y| (0..16).flat_map(move |x| [x * 17, y * 36, (x + y) * 11, 255]))
        .collect();
    let image = WallpaperImage {
        id: 1,
        width: 16,
        height: 8,
        rgba: rgba.into(),
    };
    for module in 1..=18 {
        let sequence: &[u32] = match module {
            5 => &[4, 5],
            11 => &[10, 11],
            15 => &[17, 14, 15],
            12 => &[17, 12],
            13 => &[17, 13],
            14 => &[17, 14],
            16 => &[17, 16],
            _ => std::slice::from_ref(&module),
        };
        let passes = sequence
            .iter()
            .map(|&id| {
                let mut uniforms = [[0.0; 4]; 15];
                uniforms[1] = [
                    64.0,
                    40.0,
                    1.0,
                    if id == 10 || id == 14 { 0.0 } else { 2.0 },
                ];
                uniforms[2] = [0.0, 0.5, 0.0, 2.0];
                uniforms[4] = [64.0, 40.0, 0.0, 0.0];
                uniforms[5] = [0.0, 64.0, 40.0, 0.0];
                if id == 17 {
                    uniforms[6] = [0.8, 1.0, 0.1, 0.0];
                }
                uniforms[7][3] = f32::from(u8::from((12..=17).contains(&id)));
                uniforms[9..].copy_from_slice(&DEFAULT_PARAMS[id as usize - 1]);
                WallpaperPass {
                    module: id,
                    width: 64,
                    height: 40,
                    noise: u32::from((4..=6).contains(&id)),
                    previous: matches!(id, 5 | 11..=16),
                    image: id == 17,
                    uniforms,
                }
            })
            .collect();
        let bounds = Bounds::new(
            point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size(ScaledPixels(64.0), ScaledPixels(40.0)),
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
                key: module as u64,
                generation: 0,
                clock: None,
                image: Some(image.clone()),
                passes,
            }),
        });
        scene.finish();
        let frame = renderer
            .render_scene_to_image(&scene, target)
            .unwrap_or_else(|error| panic!("module {module}: {error}"));
        assert!(
            frame
                .pixels()
                .all(|pixel| pixel.0.iter().all(|&v| f32::from(v).is_finite())),
            "module {module}: non-finite pixel"
        );
        let first = frame.get_pixel(0, 0);
        assert!(
            frame.pixels().any(|pixel| pixel != first),
            "module {module}: flat frame"
        );
    }
}

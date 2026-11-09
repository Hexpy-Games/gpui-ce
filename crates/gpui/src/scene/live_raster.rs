//! CPU rasterisation of live canvas display lists.

use super::LiveOp;
use resvg::tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, PixmapPaint, Transform};

/// Rasterises device-pixel operations into premultiplied BGRA rows.
pub fn rasterize_live(ops: &[LiveOp], width: u32, height: u32) -> Vec<u8> {
    let mut stack = vec![(ops.iter(), Pixmap::new(width, height).unwrap(), 1.0)];
    let mut pixels = loop {
        let (ops, pixmap, _) = stack.last_mut().unwrap();
        if let Some(op) = ops.next() {
            let mut path = PathBuilder::new();
            let (color, rule) = match op {
                LiveOp::Quad {
                    bounds,
                    radii,
                    color,
                } => {
                    let (x, y) = (bounds.origin.x.0, bounds.origin.y.0);
                    let (right, bottom) = (bounds.right().0, bounds.bottom().0);
                    let (tl, tr, br, bl) = (
                        radii.top_left.0,
                        radii.top_right.0,
                        radii.bottom_right.0,
                        radii.bottom_left.0,
                    );
                    // Cubic approximation of each quarter circle.
                    let k = 0.552_284_8;
                    path.move_to(x + tl, y);
                    path.line_to(right - tr, y);
                    path.cubic_to(
                        right - tr + k * tr,
                        y,
                        right,
                        y + tr - k * tr,
                        right,
                        y + tr,
                    );
                    path.line_to(right, bottom - br);
                    path.cubic_to(
                        right,
                        bottom - br + k * br,
                        right - br + k * br,
                        bottom,
                        right - br,
                        bottom,
                    );
                    path.line_to(x + bl, bottom);
                    path.cubic_to(
                        x + bl - k * bl,
                        bottom,
                        x,
                        bottom - bl + k * bl,
                        x,
                        bottom - bl,
                    );
                    path.line_to(x, y + tl);
                    path.cubic_to(x, y + tl - k * tl, x + tl - k * tl, y, x + tl, y);
                    path.close();
                    (color, FillRule::Winding)
                }
                LiveOp::Fill {
                    contours,
                    even_odd,
                    color,
                } => {
                    for contour in contours {
                        path.move_to(contour[0].x.0, contour[0].y.0);
                        for point in &contour[1..] {
                            path.line_to(point.x.0, point.y.0);
                        }
                        path.close();
                    }
                    (
                        color,
                        if *even_odd {
                            FillRule::EvenOdd
                        } else {
                            FillRule::Winding
                        },
                    )
                }
                LiveOp::Group { alpha, ops } => {
                    stack.push((ops.iter(), Pixmap::new(width, height).unwrap(), *alpha));
                    continue;
                }
            };
            let paint = Paint {
                shader: resvg::tiny_skia::Shader::SolidColor(
                    Color::from_rgba(color.red, color.green, color.blue, color.alpha).unwrap(),
                ),
                anti_alias: true,
                ..Paint::default()
            };
            pixmap.fill_path(
                &path.finish().unwrap(),
                &paint,
                rule,
                Transform::identity(),
                None,
            );
        } else {
            let (_, pixmap, alpha) = stack.pop().unwrap();
            if let Some((_, parent, _)) = stack.last_mut() {
                parent.draw_pixmap(
                    0,
                    0,
                    pixmap.as_ref(),
                    &PixmapPaint {
                        opacity: alpha,
                        ..PixmapPaint::default()
                    },
                    Transform::identity(),
                    None,
                );
            } else {
                break pixmap.take();
            }
        }
    };
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
    pixels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bounds, Corners, ScaledPixels, hsla_to_rgba, point, red, size};

    // test-category: pure-logic
    #[test]
    fn isolated_group_opacity_and_bgra() {
        let ops = [LiveOp::Group {
            alpha: 0.5,
            ops: [0.0, 2.0]
                .into_iter()
                .map(|origin| LiveOp::Quad {
                    bounds: Bounds::new(
                        point(ScaledPixels(origin), ScaledPixels(origin)),
                        size(ScaledPixels(6.0), ScaledPixels(6.0)),
                    ),
                    radii: Corners::default(),
                    color: hsla_to_rgba(red()),
                })
                .collect(),
        }];
        let bytes = rasterize_live(&ops, 8, 8);
        let pixel = &bytes[(3 * 8 + 3) * 4..(3 * 8 + 3) * 4 + 4];
        assert!((f32::from(pixel[3]) / 255.0 - 0.5).abs() <= 0.01);
        assert_eq!(pixel[0], 0);
        assert_eq!(pixel[1], 0);
        assert_eq!(pixel[2], pixel[3]);
    }
}

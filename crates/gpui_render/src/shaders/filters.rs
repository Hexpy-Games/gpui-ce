// Modified by HexpyGames for Butler (2026); see NOTICE for the changes.
#[wgsl_rs::wgsl]
pub mod surface {
    use super::super::common::*;
    use wgsl_rs::std::*;

    #[repr(C)]
    #[derive(Clone, Copy, Wgsl)]
    pub struct SurfaceUniforms {
        pub bounds: Bounds,
        pub content_mask: ContentMask,
        pub color_format: SurfaceColorFormat,
        pub opacity: f32,
        pub padding0: u32,
        pub padding1: u32,
        pub padding2: u32,
        pub padding3: u32,
        pub padding4: u32,
        pub padding5: u32,
    }
    uniform!(group(1), binding(0), SURFACE_LOCALS: SurfaceUniforms);
    texture!(group(1), binding(1), SURFACE_TEXTURE: Texture2D<f32>);
    texture!(group(1), binding(2), SURFACE_CHROMA_TEXTURE: Texture2D<f32>);
    sampler!(group(1), binding(3), SURFACE_SAMPLER: Sampler);

    pub const YCBCR_TO_LINEAR_RGB: Mat4x4f = mat4x4f(
        vec4f(1.0000, 1.0000, 1.0000, 0.0),
        vec4f(0.0000, -0.3441, 1.7720, 0.0),
        vec4f(1.4020, -0.7141, 0.0000, 0.0),
        vec4f(-0.7010, 0.5291, -0.8860, 1.0),
    );

    pub fn sample_yuv_surface(texture_position: Vec2f) -> Vec4f {
        let luma = texture_sample_level(SURFACE_TEXTURE, SURFACE_SAMPLER, texture_position, 0.0).x;
        let chroma = texture_sample_level(
            SURFACE_CHROMA_TEXTURE,
            SURFACE_SAMPLER,
            texture_position,
            0.0,
        )
        .xy();
        YCBCR_TO_LINEAR_RGB * vec4f(luma, chroma.x, chroma.y, 1.0)
    }

    #[derive(Wgsl)]
    pub struct SurfaceVarying {
        #[builtin(position)]
        pub position: Vec4f,
        #[location(0)]
        pub texture_position: Vec2f,
        #[location(3)]
        pub clip_distances: Vec4f,
    }

    #[vertex]
    pub fn vertex_surface(#[builtin(vertex_index)] vertex_id: u32) -> SurfaceVarying {
        let vertex = rectangle_vertex(vertex_id, get!(SURFACE_LOCALS).bounds);
        SurfaceVarying {
            position: vertex.clip_position,
            texture_position: vertex.unit_position,
            clip_distances: clip_distances(
                vertex.viewport_position,
                get!(SURFACE_LOCALS).content_mask.bounds,
            ),
        }
    }

    #[fragment]
    pub fn fragment_surface(input: SurfaceVarying) -> Vec4f {
        if is_clipped(input.clip_distances) {
            return transparent();
        }
        let locals = get!(SURFACE_LOCALS);
        let fade = ContentMask::alpha(locals.content_mask, input.position.xy());
        if locals.color_format == SurfaceColorFormat::Yuv {
            return sample_yuv_surface(input.texture_position) * locals.opacity * fade;
        }
        texture_sample_level(
            SURFACE_TEXTURE,
            SURFACE_SAMPLER,
            input.texture_position,
            0.0,
        ) * locals.opacity
            * fade
    }
}

#[wgsl_rs::wgsl]
pub mod blur {
    use super::super::common::*;
    use super::super::corner_smoothing::*;
    use wgsl_rs::std::*;

    #[repr(C)]
    #[derive(Clone, Copy, Wgsl)]
    pub struct BlurUniforms {
        pub bounds: Bounds,
        pub content_mask: Bounds,
        pub corner_radii: Corners,
        pub direction: Vec2f,
        pub standard_deviation: f32,
        pub opacity: f32,
        pub sample_count: u32,
        pub sample_step: f32,
        pub composite_clip: BlurCompositeClip,
        pub downsample_mode: DownsampleMode,
        pub source_size: Vec2f,
        pub target_size: Vec2f,
        pub corner_smoothing: f32,
        pub lens_bezel: f32,
        pub lens_refraction: f32,
        pub lens_specular: f32,
        pub lens_dispersion: f32,
        pub lens_frost: f32,
        pub lens_lift: f32,
        pub lens_rim: f32,
        pub lens_depth: f32,
        pub lens_light_x: f32,
        pub lens_light_y: f32,
        pub lens_opposite: f32,
        pub lens_shade: f32,
        pub lens_fresnel: f32,
        pub lens_focus: f32,
        pub lens_ambient: f32,
        pub lens_halo: f32,
        pub lens_fold: f32,
        pub lens_fold_band: f32,
        pub lens_fold_depth: f32,
        pub lens_curve: f32,
        pub padding0: u32,
        pub padding1: u32,
        pub padding2: u32,
        pub wp_mode: Vec4f,
        pub wp_frame: Vec4f,
        pub wp_env: Vec4f,
        pub wp_rect: Vec4f,
        pub wp_css: Vec4f,
        pub wp_print: Vec4f,
        pub wp_drop: Vec4f,
        pub wp_col: Vec4f,
        pub wp_rect2: Vec4f,
        pub wp_p0: Vec4f,
        pub wp_p1: Vec4f,
        pub wp_p2: Vec4f,
        pub wp_p3: Vec4f,
        pub wp_p4: Vec4f,
        pub wp_p5: Vec4f,
    }
    uniform!(group(1), binding(0), BLUR_LOCALS: BlurUniforms);

    /// Tangential pull toward the centre inside the bevel, as a share of the refraction: lines
    /// crossing a straight edge curve through the bevel (ybouane's centre term, kept to the
    /// bevel and to the along-edge direction so the middle stays 1:1).
    pub const LENS_BEND: f32 = 0.18;
    /// Decay length of the rim glow, in core widths (fitted to iOS: 2.25).
    pub const LENS_GLOW_LENGTH: f32 = 2.25;
    /// Half width of the curvature stencil, in device px.
    pub const LENS_CURVATURE_STENCIL: f32 = 4.0;
    texture!(group(1), binding(1), BLUR_TEXTURE: Texture2D<f32>);
    sampler!(group(1), binding(2), BLUR_SAMPLER: Sampler);

    #[derive(Clone, Copy, Wgsl)]
    pub struct BlurCompositeVertexData {
        pub position: Vec4f,
        pub texture_coordinates: Vec2f,
        pub clip_distances: Vec4f,
    }

    pub fn prepare_blur_composite_vertex(vertex_id: u32) -> BlurCompositeVertexData {
        // A lens draws 2 px past its silhouette: the outer half of the edge's anti-aliasing and the
        // one-pixel hairline outside it (native glass) need pixels whose centres lie outside.
        let b = get!(BLUR_LOCALS).bounds;
        let grow = select(0.0, 2.0, get!(BLUR_LOCALS).lens_bezel > 0.0);
        let unit_position = rectangle_corner(vertex_id);
        let viewport_position =
            b.origin - vec2f(grow, grow) + unit_position * (b.size + vec2f(2.0 * grow, 2.0 * grow));
        BlurCompositeVertexData {
            position: viewport_to_clip_position(viewport_position),
            texture_coordinates: unit_position,
            clip_distances: clip_distances(viewport_position, get!(BLUR_LOCALS).content_mask),
        }
    }

    pub fn sample_composite_source(position: Vec2f) -> Vec4f {
        texture_sample_level(
            BLUR_TEXTURE,
            BLUR_SAMPLER,
            position / get!(BLUR_LOCALS).target_size,
            0.0,
        )
    }

    /// Slope of a circular bevel of radius `bezel` at depth `d` inside the edge, from the height
    /// field h(d) = sqrt(d * (2 * bezel - d)) (LiquidGlass by ybouane), tapered by
    /// (1 - d / bezel) so it reaches zero smoothly where the bevel meets the flat middle; that
    /// keeps the middle 1:1 with no magnified ring or seam. Steep (unbounded) at the rim.
    pub fn lens_slope(d: f32, bezel: f32) -> f32 {
        let inside = clamp(d, 0.0, bezel);
        let rest = bezel - inside;
        rest / sqrt(max(inside * (2.0 * bezel - inside), 0.0001)) * (rest / bezel)
    }

    /// The sharp backdrop at `position`, or a small 13-tap disc around it when `frost` > 0.
    pub fn lens_source(position: Vec2f, frost: f32) -> Vec4f {
        let center = sample_composite_source(position);
        if frost <= 0.0 {
            return center;
        }
        let outer = frost;
        let inner = frost * 0.5;
        let outer_ring = sample_composite_source(position + vec2f(outer, 0.0))
            + sample_composite_source(position + vec2f(outer * 0.5, outer * 0.866))
            + sample_composite_source(position + vec2f(-outer * 0.5, outer * 0.866))
            + sample_composite_source(position + vec2f(-outer, 0.0))
            + sample_composite_source(position + vec2f(-outer * 0.5, -outer * 0.866))
            + sample_composite_source(position + vec2f(outer * 0.5, -outer * 0.866));
        let inner_ring = sample_composite_source(position + vec2f(inner * 0.866, inner * 0.5))
            + sample_composite_source(position + vec2f(0.0, inner))
            + sample_composite_source(position + vec2f(-inner * 0.866, inner * 0.5))
            + sample_composite_source(position + vec2f(-inner * 0.866, -inner * 0.5))
            + sample_composite_source(position + vec2f(0.0, -inner))
            + sample_composite_source(position + vec2f(inner * 0.866, -inner * 0.5));
        (center * 2.0 + inner_ring * 1.25 + outer_ring * 0.75) / 14.0
    }

    pub fn lens_signed_distance(position: Vec2f) -> f32 {
        let locals = get!(BLUR_LOCALS);
        rounded_rectangle_signed_distance(position, locals.bounds, locals.corner_radii)
    }

    /// The refraction's depth field and its outward direction, (depth, direction.x, direction.y).
    /// The distance field's inner contours lose their rounding deeper than the corner radius r
    /// and meet the straight edges' contours on 45-degree medial-axis creases, where the inward
    /// direction turns by 90 degrees at once: the seam inside a tall composer's corners. Here
    /// the contours keep the radius r while they shrink (their corner centres move inward
    /// diagonally, |q0 + t| = r), until they are the stadium of the shorter side, from where
    /// the field is the stadium's exact distance. Every contour is a C1 rounded rectangle and the
    /// field is C1 across its zones, so direction and depth turn continuously everywhere except
    /// on the stadium's centre line, where the fold has run out (depth = half the short side).
    /// Equal to the distance on straight edges, pills and circles; in a corner the depth runs
    /// slower than the distance (by 1/sqrt(2) on the diagonal near the rim), which widens the
    /// fold band round the corner a little instead of creasing it.
    pub fn lens_field(position: Vec2f) -> Vec3f {
        let locals = get!(BLUR_LOCALS);
        let half = Bounds::half_size(locals.bounds);
        let local = position - Bounds::center(locals.bounds);
        let side = vec2f(
            select(1.0, -1.0, local.x < 0.0),
            select(1.0, -1.0, local.y < 0.0),
        );
        let p = abs(local);
        let short = max(min(half.x, half.y), 0.0001);
        let r = clamp(pick_corner_radius(local, locals.corner_radii), 0.0, short);
        let q0 = p - (half - vec2f(r, r));
        let gap = q0.x - q0.y;
        let straight = abs(gap) > r;
        let corner_depth = (sqrt(max(2.0 * r * r - gap * gap, 0.0)) - (q0.x + q0.y)) * 0.5;
        let corner_dir = q0 + vec2f(corner_depth, corner_depth);
        let straight_dir = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), gap > 0.0);
        let shrink_depth = select(corner_depth, r - max(q0.x, q0.y), straight);
        let shrink_dir = select(
            corner_dir / max(length(corner_dir), 0.0001),
            straight_dir,
            straight,
        );
        let segment = max(half - vec2f(short, short), vec2f(0.0, 0.0));
        let to_segment = max(p - segment, vec2f(0.0, 0.0));
        let stadium_depth = short - length(to_segment);
        let stadium_dir = to_segment / max(length(to_segment), 2.0);
        let deep = shrink_depth > short - r;
        let depth = select(shrink_depth, stadium_depth, deep);
        let dir = select(shrink_dir, stadium_dir, deep);
        vec3f(depth, dir.x * side.x, dir.y * side.y)
    }

    /// The backdrop seen through the lens. Each sample moves inward along the edge normal by
    /// `lens_refraction` times the bevel slope, softly capped at `lens_depth` (s / (1 + s / cap)),
    /// so the backdrop near the rim is squashed toward it, while the middle maps 1:1. A smaller
    /// pull along the edge toward the centre (`LENS_BEND`) curves lines crossing the bevel. With the cap the mapping folds only within about a pixel
    /// of the rim. Red shifts a little more and blue a little less (dispersion). The steep edge
    /// is whitened by Fresnel and lit by a directional specular core (no uniform stroke); the
    /// middle can be lifted toward white or black under content.
    pub fn lens_tap(position: Vec2f) -> Vec4f {
        let locals = get!(BLUR_LOCALS);
        let distance = lens_signed_distance(position);
        let step_x = vec2f(1.0, 0.0);
        let step_y = vec2f(0.0, 1.0);
        let east = lens_signed_distance(position + step_x);
        let west = lens_signed_distance(position - step_x);
        let south = lens_signed_distance(position + step_y);
        let north = lens_signed_distance(position - step_y);
        let gradient = vec2f(east - west, south - north);
        // Curvature of the edge here (the Laplacian of a distance field): 0 on straight edges,
        // 1 / radius on the rounded corners and ends. A circular corner's curvature jumps where
        // it meets the straight edge, so it is taken over a wider (4px) stencil, which ramps it
        // over about 8px instead of leaving a seam in everything that depends on it.
        let wide = LENS_CURVATURE_STENCIL;
        let curvature = max(
            (lens_signed_distance(position + vec2f(wide, 0.0))
                + lens_signed_distance(position - vec2f(wide, 0.0))
                + lens_signed_distance(position + vec2f(0.0, wide))
                + lens_signed_distance(position - vec2f(0.0, wide))
                - 4.0 * distance)
                / (wide * wide),
            0.0,
        );
        let normal = gradient / max(length(gradient), 0.0001);
        let depth = max(-distance, 0.0);
        // The refraction follows the continuous field (lens_field); the rim light keeps the exact
        // distance and normal.
        let field = lens_field(position);
        let fold_depth_at = max(field.x, 0.0);
        let fold_normal = vec2f(field.y, field.z);
        let half = Bounds::half_size(locals.bounds);
        let bezel = max(min(locals.lens_bezel, min(half.x, half.y)), 1.0);
        let t = saturate(fold_depth_at / bezel);
        let slope = lens_slope(fold_depth_at, bezel);
        let edge_slope = lens_slope(depth, bezel);
        let cap = max(locals.lens_depth, 0.0);
        let pull = max(locals.lens_refraction, 0.0) * slope;
        let bevel_shift = pull / (1.0 + pull / max(cap, 0.0001));
        // Fold (fitted to iOS Liquid Glass against its clean wallpaper): the inward shift is
        // S (1 - d / B)^p with B and S relative to the shape's half size. The rim samples S deep,
        // and the shift falls faster than the depth grows, so the band next to the rim shows a
        // mirrored, squeezed image of the backdrop further in (about 3x at the rim); it turns
        // into a magnified zone and reaches 1:1 at B. It follows the SDF normal, so it applies
        // on every edge alike (top, bottom, the rounded ends, corners).
        let size = min(half.x, half.y);
        let fold = max(locals.lens_fold, 0.0);
        let fold_band = max(locals.lens_fold_band * size, 1.0);
        let fold_rest = max(1.0 - fold_depth_at / fold_band, 0.00001);
        let fold_shift = max(locals.lens_fold_depth, 0.0) * size * pow(fold_rest, max(fold, 0.01));
        let folding = fold > 0.0;
        let shift_length = select(bevel_shift, fold_shift, folding);
        let spread = abs(locals.lens_dispersion);
        let toward = (Bounds::center(locals.bounds) - position) / max(half, vec2f(1.0, 1.0));
        let along = toward - fold_normal * dot(toward, fold_normal);
        let bend_amount = select(LENS_BEND, 0.0, folding);
        let bend = along * (max(locals.lens_refraction, 0.0) * bend_amount * (1.0 - t) * (1.0 - t));
        let shift = fold_normal * (-shift_length) + bend;

        // Minification filter: the fold squeezes the backdrop by |dD/dd| across the edge (up to
        // about 3x next to the rim) and, on curved edges, by the radius ratio along it, so each
        // tap gathers a disc as wide as its share of that footprint instead of one bilinear
        // sample (which would skip texels and alias the squeezed band).
        let fold_slope = max(locals.lens_fold_depth, 0.0) * size * fold / fold_band
            * pow(fold_rest, max(fold - 1.0, 0.0));
        let across = abs(1.0 - select(0.0, fold_slope, folding));
        let along_ratio = max(1.0 - curvature * fold_depth_at, 0.05)
            / max(1.0 - curvature * (fold_depth_at + shift_length), 0.05);
        let footprint = max(across, along_ratio);
        let frost = max(locals.lens_frost, 0.35 * max(footprint - 1.0, 0.0));
        let red = lens_source(position + shift * (1.0 + spread), frost);
        let green = lens_source(position + shift, frost);
        let blue = lens_source(position + shift * (1.0 - spread), frost);

        // A uniform lift toward white (or black): iOS maps the backdrop to about 0.87 bg + 0.13
        // white everywhere inside, from the rim band to the centre.
        let lift = clamp(locals.lens_lift, -1.0, 1.0);
        let lift_target = select(0.0, green.w, lift > 0.0);
        let keep_lift = 1.0 - abs(lift);
        let r0 = red.x * keep_lift + lift_target * abs(lift);
        let g0 = green.y * keep_lift + lift_target * abs(lift);
        let b0 = blue.z * keep_lift + lift_target * abs(lift);
        // Native fill (padding2 tone flag, see BlurUniforms::with_lens): a tone curve on the
        // sample instead of the lift.
        let toned = lens_tone_on();
        let r = select(r0, lens_tone(red.x, red.w), toned);
        let g = select(g0, lens_tone(green.y, green.w), toned);
        let b = select(b0, lens_tone(blue.z, blue.w), toned);

        // Edge light (Liquid Glass Studio's two-sided glare, MIT, iyinchao; ybouane's Fresnel):
        // the 2D edge normal against the light direction gives a key lobe on the lit side and
        // a weaker opposite lobe, so brightness varies continuously round the perimeter. It
        // lights a thin Gaussian core at the rim with a softer glow inside it; Fresnel whitens
        // the steep bevel; the side away from the light is shaded a few pixels in. A negative
        // ambient leaves a thin dark rim on the unlit sides, as in macOS's Dock.
        let facing = dot(normal, vec2f(locals.lens_light_x, locals.lens_light_y));
        let focus = max(locals.lens_focus, 0.05);
        let key = pow(max(facing, 0.00001), focus);
        let back = pow(max(-facing, 0.00001), focus) * max(locals.lens_opposite, 0.0);
        let lobe = key + back;
        let ambient = clamp(locals.lens_ambient, -1.0, 1.0);
        // The rim line thins (and its glow fades) on curved edges, as measured on iOS: about half
        // as wide on the Dock's corners as on its straight top and bottom.
        // With the tone on, lens_curve and lens_fresnel carry the hairline (see lens_hairline).
        let thin = select(
            1.0 / (1.0 + curvature * max(locals.lens_curve, 0.0)),
            1.0,
            toned,
        );
        let width = max(locals.lens_rim, 0.25) * thin;
        // The core box-filtered over this tap's quarter-pixel footprint (an erf difference), so a
        // sub-pixel rim line dims smoothly instead of point-sampling into a staircase.
        let core = lens_core(depth, width);
        let glow = exp(-depth / (LENS_GLOW_LENGTH * width)) * max(locals.lens_halo, 0.0) * thin;
        let grazing = 1.0 - 1.0 / sqrt(1.0 + edge_slope * edge_slope);
        let fresnel = select(
            grazing * grazing * grazing * grazing * max(locals.lens_fresnel, 0.0),
            0.0,
            toned,
        );
        // A positive ambient lights the rim all round; a negative one leaves a thin dark line
        // where the light lobe is weak (iOS's vertical ends), without dimming the lit rim.
        let lit = max(locals.lens_specular, 0.0) * (lobe + max(ambient, 0.0)) * (core + glow);
        let highlight = saturate(lit + fresnel);
        let rim_dark = saturate(max(-ambient, 0.0) * (1.0 - saturate(lobe)) * core);
        let shade_in = 1.0 - exp(-depth / max(width * 1.5, 0.5));
        let shade_out = exp(-depth / max(bezel * 0.3, 1.0));
        let shade_amount = select(max(locals.lens_shade, 0.0), 0.0, toned);
        let shade = saturate(shade_amount * max(-facing, 0.0) * shade_in * shade_out * 2.0);
        let dim = (1.0 - shade) * (1.0 - rim_dark);
        // With the tone curve on (native glass) the rim light adds to the toned sample (it is
        // measured as a near-constant step over black, grey and white), otherwise it mixes
        // toward white.
        let keep = select(1.0 - highlight, 1.0, toned);
        let alpha = select(green.w * keep + highlight, green.w, toned);
        vec4f(
            r * dim * keep + highlight,
            g * dim * keep + highlight,
            b * dim * keep + highlight,
            alpha,
        )
    }

    /// Native glass's hairline (tone mode only): black in the one-pixel ring just outside the
    /// silhouette, side strength (lens_curve) on left and right edges falling to the top/bottom
    /// strength (lens_fresnel) with the edge normal's vertical share; box-filtered over the pixel.
    pub fn lens_hairline(position: Vec2f) -> Vec4f {
        let locals = get!(BLUR_LOCALS);
        if !lens_tone_on() || (locals.lens_curve <= 0.0 && locals.lens_fresnel <= 0.0) {
            return transparent();
        }
        let sd = lens_signed_distance(position);
        let ring = saturate(min(sd + 0.5, 1.0) - max(sd - 0.5, 0.0));
        if ring <= 0.0 {
            return transparent();
        }
        let gx = lens_signed_distance(position + vec2f(0.5, 0.0))
            - lens_signed_distance(position - vec2f(0.5, 0.0));
        let gy = lens_signed_distance(position + vec2f(0.0, 0.5))
            - lens_signed_distance(position - vec2f(0.0, 0.5));
        let n = vec2f(gx, gy) / max(length(vec2f(gx, gy)), 0.0001);
        let side = max(locals.lens_curve, 0.0);
        let top = max(locals.lens_fresnel, 0.0);
        let alpha = saturate((top + (side - top) * n.x * n.x) * ring);
        vec4f(0.0, 0.0, 0.0, alpha)
    }

    /// Whether the tone curve is on (padding2 = 1 or 3; see BlurUniforms::with_lens).
    pub fn lens_tone_on() -> bool {
        let flags = get!(BLUR_LOCALS).padding2;
        flags == 1u32 || flags == 3u32
    }

    /// Whether the sharp pass covers the whole shape (padding2 = 2 or 3).
    pub fn lens_sharp_full() -> bool {
        get!(BLUR_LOCALS).padding2 >= 2u32
    }

    /// The tone curve a + b x + c x^2 on one premultiplied channel (coefficients in
    /// lens_refraction, lens_depth, lens_shade while the tone is on).
    pub fn lens_tone(x: f32, alpha: f32) -> f32 {
        let locals = get!(BLUR_LOCALS);
        let a = max(alpha, 0.0001);
        let v = saturate(x / a);
        let out = locals.lens_refraction + locals.lens_depth * v + locals.lens_shade * v * v;
        saturate(out) * alpha
    }

    /// erf(x), Abramowitz-Stegun style approximation through tanh (|error| < 0.002).
    pub fn lens_erf(x: f32) -> f32 {
        let y = x * (1.128379 + 0.1009 * x * x);
        1.0 - 2.0 / (exp(2.0 * clamp(y, -15.0, 15.0)) + 1.0)
    }

    /// exp(-(d / w)^2) averaged over d +- 0.25px.
    pub fn lens_core(depth: f32, width: f32) -> f32 {
        let half = 0.25;
        let scale = width * 0.886227 / (2.0 * half);
        saturate(scale * (lens_erf((depth + half) / width) - lens_erf((depth - half) / width)))
    }

    /// One lens tap, weighted 1 inside the shape and 0 outside (the composite's coverage
    /// already covers the edge pixel's outside part).
    pub fn lens_weighted_tap(position: Vec2f) -> Vec4f {
        let inside = select(0.0, 1.0, lens_signed_distance(position) < 0.0);
        lens_tap(position) * inside
    }

    /// The lens, box-filtered over the pixel: four rotated-grid taps (RGSS) each run the whole
    /// mapping and lighting, so the mirrored, up to 3x squeezed fold band and the
    /// sub-pixel-wide rim line are averaged over the pixel instead of point-sampled at its centre
    /// (which stair-steps along curves, worst at a scale factor of 1). Taps outside the shape are
    /// left out; the composite multiplies by the edge coverage.
    pub fn lens_color(position: Vec2f) -> Vec4f {
        // Deep in a flat sheet's middle the lens is the identity mapping with only the uniform
        // lift (no fold shift, bevel slope, rim core, glow, shade or minification filter reach
        // there), so one plain sample gives the same colour as the four full taps.
        let locals = get!(BLUR_LOCALS);
        let half = Bounds::half_size(locals.bounds);
        let size = min(half.x, half.y);
        let band = select(
            0.0,
            max(locals.lens_fold_band * size, 1.0),
            locals.lens_fold > 0.0,
        );
        let reach = max(band, max(locals.lens_bezel, 1.0))
            + LENS_GLOW_LENGTH * 6.0 * max(locals.lens_rim, 0.25)
            + max(locals.lens_bezel * 0.3, 1.0) * 6.0
            + 4.0;
        if locals.lens_frost <= 0.0 && lens_field(position).x > reach {
            let c = sample_composite_source(position);
            if lens_tone_on() {
                return vec4f(
                    lens_tone(c.x, c.w),
                    lens_tone(c.y, c.w),
                    lens_tone(c.z, c.w),
                    c.w,
                );
            }
            let lift = clamp(locals.lens_lift, -1.0, 1.0);
            let lift_to = select(0.0, c.w, lift > 0.0) * abs(lift);
            let keep = 1.0 - abs(lift);
            return vec4f(
                c.x * keep + lift_to,
                c.y * keep + lift_to,
                c.z * keep + lift_to,
                c.w,
            );
        }
        let a = lens_weighted_tap(position + vec2f(0.125, 0.375));
        let b = lens_weighted_tap(position + vec2f(-0.375, 0.125));
        let c = lens_weighted_tap(position + vec2f(-0.125, -0.375));
        let d = lens_weighted_tap(position + vec2f(0.375, -0.125));
        let inside = select(
            0.0,
            1.0,
            lens_signed_distance(position + vec2f(0.125, 0.375)) < 0.0,
        ) + select(
            0.0,
            1.0,
            lens_signed_distance(position + vec2f(-0.375, 0.125)) < 0.0,
        ) + select(
            0.0,
            1.0,
            lens_signed_distance(position + vec2f(-0.125, -0.375)) < 0.0,
        ) + select(
            0.0,
            1.0,
            lens_signed_distance(position + vec2f(0.375, -0.125)) < 0.0,
        );
        let sum = a + b + c + d;
        select(lens_tap(position), sum / max(inside, 1.0), inside > 0.0)
    }

    /// A wallpaper texture (wp_mode.x = 2) composited into the bounds: bilinear, clamped to its
    /// edge texels; wp_frame.xy is the texture size.
    pub fn wallpaper_composite_color(position: Vec2f, factor: f32) -> Vec4f {
        let u = get!(BLUR_LOCALS);
        let size = u.wp_frame.xy();
        let uv = (position - u.bounds.origin) / u.bounds.size;
        let lo = vec2f(0.5, 0.5) / size;
        let tc = clamp(uv, lo, vec2f(1.0, 1.0) - lo);
        let c = texture_sample_level(BLUR_TEXTURE, BLUR_SAMPLER, tc, 0.0);
        vec4f(c.x * factor, c.y * factor, c.z * factor, factor)
    }

    pub fn blur_composite_color(position: Vec2f, coverage: f32) -> Vec4f {
        let factor = coverage * get!(BLUR_LOCALS).opacity;
        if get!(BLUR_LOCALS).wp_mode.x > 1.5 {
            return wallpaper_composite_color(position, factor);
        }
        if get!(BLUR_LOCALS).lens_bezel > 0.0 {
            // Sharp-rim pass (padding1 = weight in percent): the lens again, from the sharp
            // snapshot, only in the fold band, fading out toward the band's inner end, over the
            // blurred glass the first pass composited.
            let rim_pass = get!(BLUR_LOCALS).padding1 as f32 / 100.0;
            if rim_pass > 0.0 && lens_sharp_full() {
                return lens_color(position) * (factor * rim_pass);
            }
            if rim_pass > 0.0 {
                let half = Bounds::half_size(get!(BLUR_LOCALS).bounds);
                let band = max(get!(BLUR_LOCALS).lens_fold_band * min(half.x, half.y), 1.0);
                let rest = 1.0 - max(lens_field(position).x, 0.0) / band;
                if rest <= 0.0 {
                    return transparent();
                }
                return lens_color(position) * (factor * rim_pass * saturate(rest * 2.0));
            }
            // Main pass: the glass where the shape covers the pixel, and the hairline in the
            // one-device-pixel ring just outside it (float geometry, so a growing shape moves
            // smoothly instead of in whole-pixel steps).
            let ring = lens_hairline(position) * (1.0 - coverage) * get!(BLUR_LOCALS).opacity;
            if coverage <= 0.0 {
                return ring;
            }
            return lens_color(position) * factor + ring;
        }
        let blurred = sample_composite_source(position);
        vec4f(
            blurred.x * factor,
            blurred.y * factor,
            blurred.z * factor,
            blurred.w * factor,
        )
    }

    #[derive(Wgsl)]
    pub struct BlurVarying {
        #[builtin(position)]
        pub position: Vec4f,
        #[location(0)]
        pub texture_coordinates: Vec2f,
        #[location(3)]
        pub clip_distances: Vec4f,
    }

    #[vertex]
    pub fn vertex_blur_fullscreen(#[builtin(vertex_index)] vertex_id: u32) -> BlurVarying {
        let vertex = fullscreen_vertex(vertex_id);
        BlurVarying {
            position: vertex.clip_position,
            texture_coordinates: vertex.texture_coordinates,
            clip_distances: unclipped_distances(),
        }
    }

    #[fragment]
    pub fn fragment_blur_downsample(input: BlurVarying) -> Vec4f {
        if get!(BLUR_LOCALS).downsample_mode == DownsampleMode::HalfResolution {
            let destination = floor(input.position.xy());
            let source_coordinates = min(
                (destination + 0.5) / get!(BLUR_LOCALS).target_size,
                (get!(BLUR_LOCALS).source_size - 0.5) / get!(BLUR_LOCALS).source_size,
            );
            return texture_sample_level(BLUR_TEXTURE, BLUR_SAMPLER, source_coordinates, 0.0);
        }
        texture_sample_level(BLUR_TEXTURE, BLUR_SAMPLER, input.texture_coordinates, 0.0)
    }

    pub fn sample_blur_texture(texture_coordinates: Vec2f) -> Vec4f {
        texture_sample_level(BLUR_TEXTURE, BLUR_SAMPLER, texture_coordinates, 0.0)
    }

    pub fn gaussian_blur(texture_coordinates: Vec2f) -> Vec4f {
        let uniforms = get!(BLUR_LOCALS);
        let center_weight = gaussian(0.0, uniforms.standard_deviation);
        let mut weighted_color = sample_blur_texture(texture_coordinates) * center_weight;
        let mut weight_sum = center_weight;
        let mut sample_index = 1u32;

        while sample_index <= uniforms.sample_count {
            let distance = sample_index as f32 * uniforms.sample_step;
            let weight = gaussian(distance, uniforms.standard_deviation);
            let coordinate_offset = uniforms.direction * distance;
            let symmetric_pair = sample_blur_texture(texture_coordinates - coordinate_offset)
                + sample_blur_texture(texture_coordinates + coordinate_offset);
            weighted_color = weighted_color + symmetric_pair * weight;
            weight_sum += 2.0 * weight;
            sample_index += 1u32;
        }
        weighted_color / max(weight_sum, MIN_NORMALIZED_WEIGHT)
    }

    #[fragment]
    pub fn fragment_blur(input: BlurVarying) -> Vec4f {
        gaussian_blur(input.texture_coordinates)
    }

    #[vertex]
    pub fn vertex_blur_composite(#[builtin(vertex_index)] vertex_id: u32) -> BlurVarying {
        let vertex = prepare_blur_composite_vertex(vertex_id);
        BlurVarying {
            position: vertex.position,
            texture_coordinates: vertex.texture_coordinates,
            clip_distances: vertex.clip_distances,
        }
    }

    #[fragment]
    pub fn fragment_blur_composite(input: BlurVarying) -> Vec4f {
        if is_clipped(input.clip_distances) {
            return transparent();
        }
        let coverage = select(
            1.0,
            antialiased_coverage(rounded_rectangle_signed_distance(
                input.position.xy(),
                get!(BLUR_LOCALS).bounds,
                get!(BLUR_LOCALS).corner_radii,
            )),
            get!(BLUR_LOCALS).composite_clip == BlurCompositeClip::RoundedBounds,
        );
        blur_composite_color(input.position.xy(), coverage)
    }

    #[derive(Wgsl)]
    pub struct SmoothedBlurVarying {
        #[builtin(position)]
        pub position: Vec4f,
        #[location(0)]
        pub texture_coordinates: Vec2f,
        #[location(1)]
        #[interpolate(flat)]
        pub horizontal_corner_reaches: Vec4f,
        #[location(2)]
        #[interpolate(flat)]
        pub vertical_corner_reaches: Vec4f,
        #[location(3)]
        pub clip_distances: Vec4f,
        #[location(4)]
        #[interpolate(flat)]
        pub smoothing_factors: Vec4f,
        #[location(5)]
        #[interpolate(flat)]
        pub superellipse_power: f32,
    }

    // ---------------------------------------------------------------------------------------
    // Butler product wallpapers, ported from packages/butler-app/client/ui/src/libs/design-system/
    // blocks/Wallpaper/modules (butler.bloom, butler.dusk, butler.shoreline, butler.riso-print
    // density.frag + print.frag) line for line. The engine contract maps as follows:
    //   gl_FragCoord.xy -> wp_frag_coord() (bottom-left origin, like GL)
    //   u_resolution, u_pixelRatio, u_time       -> wp_frame (xy, z, w)
    //   u_dark, u_dayPhase, u_seed               -> wp_env (x, y, z)
    //   u_contentRect                            -> wp_rect
    //   u_css, u_flow, u_lift                    -> wp_css (xy, z, w)
    //   u_pulse                                  -> wp_print.x
    //   u_drop, u_col, u_contentRect2            -> wp_drop, wp_col, wp_rect2
    //   module params (p_*)                      -> wp_p0 .. wp_p5
    //   u_noiseTexture / u_density               -> BLUR_TEXTURE, sampled by hand: GL's LINEAR +
    //                                               REPEAT (noise) or CLAMP_TO_EDGE (density)
    // ---------------------------------------------------------------------------------------

    pub fn wp_frag_coord(position: Vec2f) -> Vec2f {
        vec2f(position.x, get!(BLUR_LOCALS).wp_frame.y - position.y)
    }

    pub fn wp_mod(x: f32, y: f32) -> f32 {
        x - y * floor(x / y)
    }

    /// texelFetch(u_noiseTexture, c & 255, 0)
    pub fn wp_texel(x: i32, y: i32) -> Vec4f {
        texture_load(BLUR_TEXTURE, vec2i(x & 255, y & 255), 0i32)
    }

    /// texture(u_noiseTexture, uv) with LINEAR filtering and REPEAT wrapping on the 256 x 256 noise.
    pub fn wp_noise(uv: Vec2f) -> Vec4f {
        let t = uv * 256.0 - 0.5;
        let i = floor(t);
        let f = t - i;
        let ix = i.x as i32;
        let iy = i.y as i32;
        let a = wp_texel(ix, iy);
        let b = wp_texel(ix + 1, iy);
        let c = wp_texel(ix, iy + 1);
        let d = wp_texel(ix + 1, iy + 1);
        let top = a + (b - a) * f.x;
        let bottom = c + (d - c) * f.x;
        top + (bottom - top) * f.y
    }

    /// texture(u_density, uv) with LINEAR filtering and CLAMP_TO_EDGE (uv with v down, like ours).
    pub fn wp_density(uv: Vec2f) -> Vec3f {
        let size = get!(BLUR_LOCALS).wp_print.yz();
        let t = uv * size - 0.5;
        let i = floor(t);
        let f = t - i;
        let mx = size.x as i32 - 1;
        let my = size.y as i32 - 1;
        let x0 = clamp(i.x as i32, 0, mx);
        let x1 = clamp(i.x as i32 + 1, 0, mx);
        let y0 = clamp(i.y as i32, 0, my);
        let y1 = clamp(i.y as i32 + 1, 0, my);
        let a = texture_load(BLUR_TEXTURE, vec2i(x0, y0), 0i32);
        let b = texture_load(BLUR_TEXTURE, vec2i(x1, y0), 0i32);
        let c = texture_load(BLUR_TEXTURE, vec2i(x0, y1), 0i32);
        let d = texture_load(BLUR_TEXTURE, vec2i(x1, y1), 0i32);
        let top = a + (b - a) * f.x;
        let bottom = c + (d - c) * f.x;
        (top + (bottom - top) * f.y).xyz()
    }

    pub fn wp_lum(c: Vec3f) -> f32 {
        dot(c, vec3f(0.2126, 0.7152, 0.0722))
    }

    // ------------------------------------- butler.bloom -------------------------------------

    pub fn wp_bloom_blob(p: Vec2f, c: Vec2f, radius: f32, scale: Vec2f, aspect: f32) -> f32 {
        let mut d = p - c;
        d.x = d.x * aspect;
        d = d * scale;
        1.0 - smoothstep(0.0, radius, length(d))
    }

    pub fn wp_bloom_flow(q: Vec2f, t: f32) -> Vec2f {
        let p = q + vec2f(sin(q.y * 3.7 + t * 0.7), cos(q.x * 3.1 - t * 0.55)) * 0.045;
        p + vec2f(
            sin((p.x + p.y) * 4.2 - t * 0.48),
            cos((p.x - p.y) * 3.8 + t * 0.42),
        ) * 0.025
    }

    pub fn wp_bloom(fc: Vec2f) -> Vec4f {
        let u = get!(BLUR_LOCALS);
        let res = u.wp_frame.xy();
        let t = u.wp_frame.w;
        let dark = u.wp_env.x;
        let asp = res.x / res.y;
        let p = wp_bloom_flow(fc / res, t);
        let c1 = vec2f(0.22 + sin(t * 0.31) * 0.5, 0.22 + cos(t * 0.37) * 0.42);
        let c2 = vec2f(0.78 + cos(t * 0.42) * 0.46, 0.32 + sin(t * 0.29) * 0.36);
        let c3 = vec2f(
            0.35 + sin(t * 0.23 + 2.0) * 0.52,
            0.82 + cos(t * 0.35 + 1.2) * 0.35,
        );
        let c4 = vec2f(
            0.9 + cos(t * 0.36 + 3.0) * 0.44,
            0.76 + sin(t * 0.44 + 2.1) * 0.38,
        );
        let c5 = vec2f(
            0.52 + sin(t * 0.19 + 4.8) * 0.5,
            0.52 + cos(t * 0.25 + 3.4) * 0.3,
        );
        let c6 = vec2f(
            0.5 + cos(t * 0.28 + 1.7) * 0.55,
            0.18 + sin(t * 0.21 + 2.8) * 0.42,
        );
        let a1 = wp_bloom_blob(p, c1, 0.74, vec2f(1.55, 0.74), asp);
        let a2 = wp_bloom_blob(p, c2, 0.7, vec2f(1.22, 0.86), asp);
        let a3 = wp_bloom_blob(p, c3, 0.76, vec2f(1.48, 0.78), asp);
        let a4 = wp_bloom_blob(p, c4, 0.66, vec2f(1.1, 0.92), asp);
        let a5 = wp_bloom_blob(p, c5, 0.86, vec2f(1.8, 0.58), asp);
        let a6 = wp_bloom_blob(p, c6, 0.72, vec2f(1.35, 0.72), asp);
        let sum = a1 + a2 + a3 + a4 + a5 + a6;
        let liquid = (u.wp_p0.xyz() * a1
            + u.wp_p1.xyz() * a2
            + u.wp_p2.xyz() * a3
            + u.wp_p3.xyz() * a4
            + u.wp_p4.xyz() * (a5 * 0.8)
            + u.wp_p5.xyz() * (a6 * 0.85))
            / max(a1 + a2 + a3 + a4 + a5 * 0.8 + a6 * 0.85, 0.001);
        let lift = wp_bloom_blob(
            p,
            vec2f(0.43 + sin(t * 0.34) * 0.28, 0.34 + cos(t * 0.27) * 0.2),
            0.42,
            vec2f(1.8, 0.62),
            asp,
        ) + wp_bloom_blob(
            p,
            vec2f(0.6 + cos(t * 0.22) * 0.25, 0.72 + sin(t * 0.31) * 0.18),
            0.38,
            vec2f(1.55, 0.68),
            asp,
        );
        let veil = smoothstep(0.32, 1.8, sum) * (1.0 - smoothstep(0.4, 1.2, lift) * 0.36);
        let base = mix(
            vec3f(1.0, 1.0, 1.0),
            vec3f(0.04, 0.045, 0.07),
            vec3f(dark, dark, dark),
        );
        let wash = mix(
            vec3f(0.965, 0.955, 1.0),
            vec3f(0.09, 0.08, 0.13),
            vec3f(dark, dark, dark),
        );
        let k = veil * mix(0.46, 0.38, dark);
        let color = base + (liquid - base) * k;
        let color2 = color + (wash - color) * 0.008;
        vec4f(color2.x, color2.y, color2.z, 1.0)
    }

    // ------------------------------------- butler.dusk --------------------------------------

    pub const WP_DUSK_PERIOD: f32 = 7680.0;
    pub const WP_TAU: f32 = 6.2831853;

    pub fn wp_lin(c: Vec3f) -> Vec3f {
        c * c
    }

    /// sin(TAU * (fract(k * u_time / PERIOD) + ph)): k whole cycles per period.
    pub fn wp_cyc(k: f32, ph: f32, period: f32) -> f32 {
        sin(WP_TAU * (fract(k * get!(BLUR_LOCALS).wp_frame.w / period) + ph))
    }

    /// Four C1 value-noise fields, one bilinear fetch.
    pub fn wp_vn4(q: Vec2f) -> Vec4f {
        let i = floor(q);
        let f0 = q - i;
        let f = f0 * f0 * (vec2f(3.0, 3.0) - f0 * 2.0);
        wp_noise((i + f + 0.5) * (1.0 / 256.0))
    }

    /// One candidate city light per world cell (cx, cy), offset by the seed (sx, sy).
    pub fn wp_city_light(
        cx: i32,
        cy: i32,
        sx: i32,
        sy: i32,
        prob: f32,
        fly: f32,
        xo: f32,
        px: f32,
        g: f32,
        shim_a: f32,
    ) -> Vec3f {
        let res_y = get!(BLUR_LOCALS).wp_frame.y;
        let cs = vec2f(0.025, 160.0 / 3584.0);
        let h = wp_texel(cx + sx, cy + sy);
        if h.x >= prob {
            return vec3f(0.0, 0.0, 0.0);
        }
        let lw = (vec2f(cx as f32, cy as f32) + 0.2 + vec2f(h.y, h.z) * 0.6) * cs;
        let zl = lw.y - fly;
        let dp = vec2f((lw.x - xo) / (zl * 1.3) - px, 0.32 / zl - 0.018 - g) * res_y;
        let r = (0.5 + 0.6 * h.w) * mix(0.65, 1.0, smoothstep(0.06, 0.35, g));
        let e = dot(dp, dp) / (r * r);
        let intensity = (0.25 + 0.75 * h.w * h.w)
            * (1.0 + shim_a * wp_cyc(2400.0 + floor(h.y * 700.0), h.z, WP_DUSK_PERIOD));
        let lc = select(
            vec3f(1.0, 0.5, 0.17),
            vec3f(0.78, 0.86, 1.0),
            fract(h.w * 7.31) < 0.18,
        );
        lc * (intensity * (exp(-e) + 0.07 * exp(-0.12 * e)))
    }

    /// Soft stop blend: halfway between linear and smoothstep.
    pub fn wp_sb(a: f32, b: f32, x: f32) -> f32 {
        let u = clamp((x - a) / (b - a), 0.0, 1.0);
        u * (0.5 + u * (1.5 - u))
    }

    pub fn wp_mix3(a: Vec3f, b: Vec3f, t: f32) -> Vec3f {
        a + (b - a) * t
    }

    pub fn wp_dusk_sky(s: f32, d: f32, l: f32) -> Vec3f {
        let mut col = vec3f(0.0, 0.0, 0.0);
        if d < -0.0045 {
        } else if s < 0.077 {
            col = wp_mix3(vec3f(0.80, 0.25, 0.13), vec3f(0.97, 0.54, 0.44), l);
            col = wp_mix3(
                col,
                wp_mix3(vec3f(0.85, 0.31, 0.15), vec3f(0.99, 0.64, 0.46), l),
                wp_sb(0.0, 0.013, s),
            );
            col = wp_mix3(
                col,
                wp_mix3(vec3f(0.87, 0.48, 0.20), vec3f(1.0, 0.77, 0.54), l),
                wp_sb(0.013, 0.034, s),
            );
            col = wp_mix3(
                col,
                wp_mix3(vec3f(0.87, 0.62, 0.38), vec3f(0.98, 0.85, 0.70), l),
                wp_sb(0.034, 0.055, s),
            );
            col = wp_mix3(
                col,
                wp_mix3(vec3f(0.84, 0.68, 0.55), vec3f(0.95, 0.86, 0.80), l),
                wp_sb(0.055, 0.077, s),
            );
        } else if s < 0.27 {
            col = wp_mix3(vec3f(0.84, 0.68, 0.55), vec3f(0.95, 0.86, 0.80), l);
            col = wp_mix3(
                col,
                wp_mix3(vec3f(0.72, 0.60, 0.56), vec3f(0.88, 0.85, 0.86), l),
                wp_sb(0.077, 0.11, s),
            );
            col = wp_mix3(
                col,
                wp_mix3(vec3f(0.63, 0.56, 0.61), vec3f(0.80, 0.84, 0.89), l),
                wp_sb(0.11, 0.13, s),
            );
            col = wp_mix3(
                col,
                wp_mix3(vec3f(0.57, 0.57, 0.68), vec3f(0.71, 0.84, 0.91), l),
                wp_sb(0.13, 0.16, s),
            );
            col = wp_mix3(
                col,
                wp_mix3(vec3f(0.46, 0.56, 0.72), vec3f(0.61, 0.81, 0.91), l),
                wp_sb(0.16, 0.21, s),
            );
            col = wp_mix3(
                col,
                wp_mix3(vec3f(0.40, 0.57, 0.74), vec3f(0.52, 0.76, 0.91), l),
                wp_sb(0.21, 0.27, s),
            );
        } else {
            col = wp_mix3(vec3f(0.40, 0.57, 0.74), vec3f(0.52, 0.76, 0.91), l);
            col = wp_mix3(
                col,
                wp_mix3(vec3f(0.28, 0.49, 0.70), vec3f(0.44, 0.68, 0.88), l),
                wp_sb(0.27, 0.345, s),
            );
            col = wp_mix3(
                col,
                wp_mix3(vec3f(0.17, 0.36, 0.54), vec3f(0.38, 0.61, 0.85), l),
                wp_sb(0.345, 0.4, s),
            );
            col = wp_mix3(
                col,
                wp_mix3(vec3f(0.11, 0.23, 0.37), vec3f(0.33, 0.55, 0.81), l),
                wp_sb(0.4, 0.45, s),
            );
            col = wp_mix3(
                col,
                wp_mix3(vec3f(0.075, 0.135, 0.22), vec3f(0.28, 0.48, 0.75), l),
                wp_sb(0.45, 0.53, s),
            );
        }
        col
    }

    /// The ground below the limb (returns rgb, and the cloud noise gn in the alpha).
    pub fn wp_dusk_ground(p: Vec2f, d: f32, l: f32) -> Vec4f {
        let u = get!(BLUR_LOCALS);
        let res = u.wp_frame.xy();
        let time = u.wp_frame.w;
        let seed = u.wp_env.z;
        let g = max(-d, 0.0);
        let mut ground = wp_mix3(
            wp_lin(wp_mix3(
                vec3f(0.072, 0.062, 0.055),
                vec3f(0.07, 0.078, 0.12),
                l,
            )),
            wp_lin(wp_mix3(
                vec3f(0.20, 0.12, 0.11),
                vec3f(0.15, 0.155, 0.235),
                l,
            )),
            exp(-g / 0.085),
        );
        let mut gn = 0.0;
        if d < 0.0 {
            let z = 0.32 / (g + 0.018);
            let xo = 61.0 * fract(seed * 7.13);
            let x = p.x * z * 1.3 + xo;
            let fly = fract(time / WP_DUSK_PERIOD * 2.0) * 160.0;
            let wg = z + fly;
            let wcl = z + fract(time / WP_DUSK_PERIOD * 3.0) * 160.0;
            let far = smoothstep(0.004, 0.05, g);

            let cu = vec2f(x + 25.0, wcl) * 1.6;
            gn = 0.65 * wp_vn4(cu).x + 0.35 * wp_vn4(cu * 2.0 + 17.0).y;
            let cloud = smoothstep(0.5, 0.8, gn) * far;
            ground = ground * (1.0 + (gn - 0.5) * mix(0.4, 0.26, l) * far);
            let rim = smoothstep(0.46, 0.6, gn) * (1.0 - smoothstep(0.6, 0.8, gn));
            ground = ground
                + wp_lin(vec3f(0.46, 0.56, 0.74))
                    * (l * rim * 0.045 * (0.35 + 0.65 * exp(-g / 0.12)) * far);

            let tn = wp_vn4(vec2f(x + 60.0, wg) * 4.8);
            let town = smoothstep(0.62, 0.9, tn.x) * (0.55 + 0.45 * tn.y);
            let road = smoothstep(0.02, 0.0, abs(tn.w - 0.5)) * smoothstep(0.4, 0.62, tn.z);
            let prob = (0.012 + 0.75 * town + 0.5 * road) * mix(1.0, 0.4, l);

            let cs = vec2f(0.025, 160.0 / 3584.0);
            let wc = vec2f(x, wg) / cs;
            let ci = floor(wc);
            let fy = wc.y - ci.y;
            let o = select(1, -1, fy < 0.5);
            let sx = (seed * 911.0) as i32;
            let sy = (seed * 353.0) as i32;
            let near = smoothstep(0.035, 0.09, g);
            let shim_a = 0.06 + 0.2 * exp(-g / 0.08);
            let mut lights = vec3f(0.0, 0.0, 0.0);
            if near > 0.0 {
                let cx = ci.x as i32;
                let cy = ci.y as i32;
                lights = wp_city_light(cx, cy, sx, sy, prob, fly, xo, p.x, g, shim_a);
                if min(fy, 1.0 - fy) * cs.y * res.y * 0.32 / (z * z) < 3.5 {
                    lights =
                        lights + wp_city_light(cx, cy + o, sx, sy, prob, fly, xo, p.x, g, shim_a);
                }
            }
            let sodium = vec3f(1.0, 0.5, 0.17);
            ground = ground + lights * (mix(1.6, 0.75, l) * near * (1.0 - 0.85 * cloud));
            ground = ground
                + sodium
                    * ((0.7 * town + 0.25 * road)
                        * mix(0.012, 0.004, l)
                        * (1.0 - 0.6 * near)
                        * far
                        * (1.0 - 0.5 * cloud));
            ground = ground + sodium * (town * 0.004 * cloud * (1.0 - l));
        }
        vec4f(ground.x, ground.y, ground.z, gn)
    }

    pub fn wp_dusk(fc: Vec2f) -> Vec4f {
        let u = get!(BLUR_LOCALS);
        let res = u.wp_frame.xy();
        let time = u.wp_frame.w;
        let l = select(1.0, 0.0, u.wp_env.x > 0.5);
        let cr = u.wp_rect;

        let mut hy = 0.45 * res.y;
        if cr.z > 0.0 {
            hy = clamp(cr.y - 0.035 * res.y, 0.38 * res.y, 0.5 * res.y);
        }
        let p = (fc - vec2f(0.5 * res.x, hy)) / res.y;

        let rp = 1.0 / mix(0.03, 0.26, clamp(u.wp_p0.x, 0.0, 1.0));
        let d = length(p + vec2f(0.0, rp)) - rp;
        let sky_h = max(1.0 - hy / res.y, 0.3);
        let s0 = max(d, 0.0) * 0.55 / sky_h;
        let s = mix(s0, min(s0, 0.27) + max(s0 - 0.27, 0.0) * 0.82, l);

        let sky = wp_lin(wp_dusk_sky(s, d, l));
        let gr = wp_dusk_ground(p, d, l);
        let ground = gr.xyz();
        let gn = gr.w;
        let g = max(-d, 0.0);

        let mut col = wp_mix3(ground, sky, smoothstep(-0.0045, 0.0045, d));

        let ad = abs(d);
        if ad < 0.25 {
            let lp = time / WP_DUSK_PERIOD;
            let sun_x = 0.14 * wp_cyc(3.0, 0.0, WP_DUSK_PERIOD) - 0.04;
            let az = exp(-(p.x - sun_x) * (p.x - sun_x) * 1.1);
            let breath = 1.0
                + 0.07 * wp_cyc(800.0, 0.0, WP_DUSK_PERIOD)
                + 0.035 * wp_cyc(1213.0, 0.3, WP_DUSK_PERIOD);
            let big_g = (0.4 + clamp(u.wp_p0.y, 0.0, 1.0)) * breath * (0.8 + 0.25 * az);
            let up = max(d, 0.0);
            let dn = max(-d, 0.0);
            let mut sh = 0.5;
            let mut sh2 = 0.5;
            if up < 0.1 && dn < 0.03 {
                sh = wp_vn4(vec2f(p.x * 7.0 + 30.0, 256.0 * 6.0 * lp)).x;
                sh2 = wp_vn4(vec2f(p.x * 19.0 + 90.0, 256.0 * 11.0 * lp + 50.0)).y;
            }
            let wz = exp(-up / 0.05 - dn / 0.009);
            let core = exp(-up / 0.018 - dn / 0.004);
            col = col
                * mix(
                    1.0,
                    big_g * (1.0 + core * (0.2 * sh + 0.1 * sh2 - 0.15)),
                    wz,
                );
            let c_core = wp_lin(wp_mix3(vec3f(0.95, 0.40, 0.16), vec3f(1.0, 0.76, 0.52), l));
            col = col + c_core * (big_g * exp(-up / 0.006 - dn / 0.005) * 0.22 * (0.85 + 0.3 * sh));
            col = col + c_core * (big_g * 0.16 * gn * gn * exp(-g / 0.02));
            col = col
                + wp_lin(wp_mix3(vec3f(1.0, 0.36, 0.16), vec3f(1.0, 0.64, 0.52), l))
                    * (big_g
                        * mix(1.0, 0.7, l)
                        * (0.11 * exp(-ad / 0.022)
                            + 0.035 * exp(-ad / 0.09) * smoothstep(0.25, 0.1, ad)));
        }

        // grade
        let mut c = sqrt(max(col, vec3f(0.0, 0.0, 0.0)));
        let t = max(c - 0.78, vec3f(0.0, 0.0, 0.0));
        c = min(c, vec3f(0.78, 0.78, 0.78)) + t / (t / 0.2 + 1.0);
        c = wp_mix3(c, c * c * (vec3f(3.0, 3.0, 3.0) - c * 2.0), 0.16);
        let mut y = wp_lum(c);
        let warm = clamp(u.wp_p0.z, 0.0, 1.0) - 0.5;
        let w_s = smoothstep(0.55, 0.12, y) * smoothstep(0.0, 0.12, y);
        let w_h = smoothstep(0.45, 0.95, y);
        c = c
            + vec3f(-0.030, 0.006, 0.030) * (w_s * (1.0 - warm))
            + vec3f(0.034, 0.012, -0.030) * (w_h * (1.0 + warm) * mix(1.0, 0.45, l));
        c = c * (vec3f(-0.035, 0.0, 0.035) * l + 1.0);
        c = c * (vec3f(0.07, 0.01, -0.07) * warm + 1.0);
        let om = clamp((c.x - c.z) * 2.2, 0.0, 1.0) * clamp((c.x - c.y) * 3.5, 0.0, 1.0);
        y = wp_lum(c);
        c = wp_mix3(vec3f(y, y, y), c, mix(0.88, 1.08, om));
        let lift = wp_mix3(vec3f(0.050, 0.040, 0.034), vec3f(0.040, 0.046, 0.060), l);
        c = lift + (vec3f(1.0, 1.0, 1.0) - lift) * c;
        let vq = (fc / res - 0.5) * vec2f(res.x / res.y, 1.0);
        c = c * mix(mix(0.72, 0.8, l), 1.0, smoothstep(1.05, 0.3, length(vq)));

        let mut head = 0.0;
        if cr.z > 0.0 {
            let lo = vec2f(cr.x, cr.y + cr.w * 0.55);
            let hi = cr.xy() + cr.zw();
            let dd = abs(fc - (lo + hi) * 0.5) - (hi - lo) * 0.5;
            let sd = length(max(dd, vec2f(0.0, 0.0))) + min(max(dd.x, dd.y), 0.0);
            head = 1.0 - smoothstep(-0.04 * res.y, mix(0.14, 0.2, l) * res.y, sd);
            if l < 0.5 {
                c = c * (1.0 - 0.16 * head);
            } else {
                c = wp_mix3(c, vec3f(0.72, 0.85, 0.96), 0.35 * head);
            }
        }

        // dust specks and hairline scratches, sky only
        let mut dust = 0.0;
        if d > 0.0 {
            let cell = floor(fc / 36.0);
            let h = wp_texel(cell.x as i32 + 13, cell.y as i32 + 101);
            if h.x > 0.972 {
                let pos = cell * 36.0 + 5.0 + vec2f(h.y, h.z) * 26.0;
                let r = 0.45 + 1.1 * h.w * h.w;
                let fade = smoothstep(
                    0.1,
                    0.9,
                    0.5 + 0.5 * wp_cyc(40.0 + floor(h.y * 60.0), h.w, WP_DUSK_PERIOD),
                );
                dust = smoothstep(r + 0.9, r - 0.3, length(fc - pos)) * fade * (0.35 + 0.4 * h.z);
            } else if h.x < 0.0035 {
                let an = h.y * 3.1416;
                let dir = vec2f(cos(an), sin(an));
                let q = fc - (cell * 36.0 + 18.0);
                let along = dot(q, dir);
                let across = dot(q, vec2f(-dir.y, dir.x)) - 0.004 * along * along;
                let hl = 8.0 + 9.0 * h.z;
                dust = smoothstep(0.85, 0.15, abs(across))
                    * smoothstep(hl, hl - 5.0, abs(along))
                    * 0.22;
            }
        }
        c = wp_mix3(c, vec3f(0.86, 0.82, 0.76), dust * (1.0 - 0.7 * head));

        // grain: heavy, fine, new every frame
        let fr = (time * 24.0) as u32;
        let r3 = wp_texel(
            fc.x as i32 + (fr * 73u32) as i32,
            fc.y as i32 + (fr * 151u32) as i32,
        )
        .xyz();
        y = wp_lum(c);
        let amp = 0.12 * clamp(u.wp_p0.w, 0.0, 1.0) * (0.42 + 0.58 * 4.0 * y * (1.0 - y));
        let gm = r3.x + r3.y - 1.0;
        c = c + (vec3f(gm, gm, gm) + vec3f(0.12, 0.0, -0.12) * (r3.z - 0.5)) * amp;
        vec4f(c.x, c.y, c.z, 1.0)
    }

    // ----------------------------------- butler.shoreline -----------------------------------

    pub const WP_SHORE_PERIOD: f32 = 120.0;
    pub const WP_CLOUD_PERIOD: f32 = 240.0;
    pub const WP_ZOOM: f32 = 2.6;

    pub fn wp_vn(q: Vec2f) -> f32 {
        let i = floor(q);
        let f0 = q - i;
        let f = f0 * f0 * (vec2f(3.0, 3.0) - f0 * 2.0);
        wp_noise((i + f + 0.5) / 256.0).x
    }

    /// Value noise periodic in y with integer period k.
    pub fn wp_pn(q: Vec2f, k: f32) -> f32 {
        let i = floor(q);
        let f0 = q - i;
        let f = f0 * f0 * (vec2f(3.0, 3.0) - f0 * 2.0);
        let y0 = wp_mod(i.y, k);
        let y1 = wp_mod(i.y + 1.0, k);
        let a = wp_noise(vec2f(i.x + f.x + 0.5, y0 + 0.5) / 256.0).y;
        let b = wp_noise(vec2f(i.x + f.x + 0.5, y1 + 0.5) / 256.0).y;
        mix(a, b, f.y)
    }

    pub fn wp_hash4(n: f32) -> Vec4f {
        wp_texel(wp_mod(n, 256.0) as i32, 97)
    }

    /// Nearest wall of a jittered cell grid: x = distance to the wall, y = a value shared by the wall.
    pub fn wp_cell_edge(q: Vec2f) -> Vec2f {
        let i = floor(q);
        let f = q - i;
        let ibx = i.x as i32;
        let iby = i.y as i32;
        let mut d1 = 9.0;
        let mut d2 = 9.0;
        let mut o1 = vec2f(0.0, 0.0);
        let mut o2 = vec2f(0.0, 0.0);
        for yi in 0..3 {
            for xi in 0..3 {
                let x = xi - 1;
                let y = yi - 1;
                let o = wp_texel(ibx + x, iby + y).xy();
                let r = vec2f(x as f32, y as f32) + 0.1 + o * 0.8 - f;
                let d = dot(r, r);
                if d < d1 {
                    d2 = d1;
                    o2 = o1;
                    d1 = d;
                    o1 = o;
                } else if d < d2 {
                    d2 = d;
                    o2 = o;
                }
            }
        }
        vec2f(
            sqrt(d2) - sqrt(d1),
            fract((o1.x + o2.x) * 5.37 + (o1.y + o2.y) * 3.71),
        )
    }

    pub fn wp_foam_at(q: Vec2f, scale: f32, cdensity: f32, front: f32) -> f32 {
        let res_y = get!(BLUR_LOCALS).wp_frame.y;
        let qq = vec2f(q.x * 0.75, q.y) * scale;
        let nz = wp_vn4(qq * 0.55 + 11.0);
        let ce = wp_cell_edge(qq + (nz.xy() - 0.5) * 1.0);
        let cc = clamp(cdensity, 0.0, 1.0);
        let wdt = (0.02 + 0.95 * cc * cc) * (0.3 + 1.4 * nz.z);
        let aa = scale * WP_ZOOM / res_y * 1.9;
        let mut core = 1.0 - smoothstep(wdt - aa, wdt + aa, ce.x);
        let pop = clamp(0.8 - 1.2 * cc + 0.35 * (nz.w - 0.5), 0.0, 0.97);
        core = core * smoothstep(pop - 0.04, pop + 0.04, ce.y);
        let lace = core * mix(0.55, 1.0, cc) * (0.7 + 0.3 * nz.z);
        let f = max(
            lace * smoothstep(0.05, 0.1, cdensity),
            (0.1 + 0.2 * nz.x) * cc,
        );
        f * smoothstep(0.0, 0.02, front - 0.07 * nz.w)
    }

    #[derive(Clone, Copy, Wgsl)]
    pub struct WpWaterPalette {
        pub absorb: Vec3f,
        pub scat: Vec3f,
        pub dry: Vec3f,
        pub wet: Vec3f,
    }

    pub fn wp_shore_palette(w: i32) -> WpWaterPalette {
        if w == 1 {
            return WpWaterPalette {
                absorb: vec3f(0.95, 0.42, 0.50),
                scat: vec3f(0.05, 0.19, 0.22),
                dry: vec3f(0.80, 0.75, 0.66),
                wet: vec3f(0.55, 0.49, 0.40),
            };
        }
        if w == 2 {
            return WpWaterPalette {
                absorb: vec3f(0.80, 0.52, 0.46),
                scat: vec3f(0.07, 0.14, 0.19),
                dry: vec3f(0.76, 0.74, 0.70),
                wet: vec3f(0.47, 0.46, 0.43),
            };
        }
        WpWaterPalette {
            absorb: vec3f(1.15, 0.26, 0.17),
            scat: vec3f(0.02, 0.19, 0.33),
            dry: vec3f(0.86, 0.80, 0.70),
            wet: vec3f(0.68, 0.58, 0.45),
        }
    }

    pub fn wp_filmic(x: Vec3f) -> Vec3f {
        clamp(
            (x * (x * 2.51 + 0.03)) / (x * (x * 2.43 + 0.59) + 0.14),
            vec3f(0.0, 0.0, 0.0),
            vec3f(1.0, 1.0, 1.0),
        )
    }

    /// The lighting state shoreline's main() sets once (Lnight, Ltint, Lexpo, LglC, LhaC) plus
    /// the glint geometry it derives with it.
    #[derive(Clone, Copy, Wgsl)]
    pub struct WpShoreLight {
        pub night: f32,
        pub tint: Vec3f,
        pub expo: f32,
        pub glint: Vec3f,
        pub halation: Vec3f,
        pub gdir: Vec2f,
        pub gc: Vec2f,
        pub g_len: f32,
        pub g_wid: f32,
        pub gl_i: f32,
        pub stretch: f32,
        pub relief: f32,
        pub lo: f32,
        pub path_k: f32,
        pub sh_w: f32,
    }

    pub fn wp_shore_light(asp: f32) -> WpShoreLight {
        let u = get!(BLUR_LOCALS);
        let dark = u.wp_env.x > 0.5;
        let mut light = WpShoreLight {
            night: 0.0,
            tint: vec3f(1.0, 1.0, 1.0),
            expo: 0.95,
            glint: vec3f(1.0, 0.94, 0.84) * 0.95,
            halation: vec3f(1.0, 0.42, 0.2) * 0.95,
            gdir: vec2f(0.0, -1.0),
            gc: vec2f(select(0.66, 0.64, dark) * asp, 0.2),
            g_len: 6.0,
            g_wid: 0.15,
            gl_i: 1.0,
            stretch: 1.0,
            relief: 1.0,
            lo: 0.0,
            path_k: 0.0,
            sh_w: 0.05,
        };
        if u.wp_p1.y > 0.5 {
            let ph = u.wp_env.y;
            let el = sin(WP_TAU * (ph - 0.25));
            let dw = smoothstep(-0.14, 0.06, el);
            let warm = dw * (1.0 - smoothstep(0.05, 0.42, el));
            let bh = exp(-(el + 0.05) * (el + 0.05) / 0.005);
            light.night = 1.0 - dw;
            light.tint = wp_mix3(
                vec3f(0.64, 0.83, 1.2),
                wp_mix3(vec3f(0.98, 1.0, 1.04), vec3f(1.16, 0.95, 0.72), warm),
                dw,
            ) * wp_mix3(vec3f(1.0, 1.0, 1.0), vec3f(1.04, 0.9, 1.1), 0.6 * bh);
            light.expo = exp(mix(
                log(0.16),
                log(mix(0.55, 0.95, smoothstep(0.0, 0.5, el))),
                dw,
            ));
            let h = select((fract(ph + 0.5) - 0.5) / 0.25, (ph - 0.5) / 0.25, el >= 0.0);
            let a = clamp(h, -1.2, 1.2) * 1.5707963;
            let elev = abs(el);
            light.gdir = vec2f(-sin(a), -cos(a));
            light.gc = vec2f(
                asp * (0.5 - 0.42 * sin(a)),
                0.2 - 0.06 * cos(a) * (1.0 - elev),
            );
            light.g_len = 0.12 + 0.42 * (1.0 - elev);
            light.g_wid = 0.09 + 0.06 * elev;
            light.gl_i = smoothstep(0.0, 0.1, elev);
            light.stretch = mix(1.8, 1.0, elev);
            light.relief = dot(vec2f(0.34, 0.94), light.gdir) * mix(2.2, 1.0, elev);
            light.lo = 1.0 - elev;
            light.path_k = 0.015 + 0.03 * light.lo;
            light.sh_w = 0.25;
            let sun_g = wp_mix3(vec3f(1.0, 0.94, 0.84), vec3f(1.05, 0.68, 0.4), warm) * 1.05;
            light.glint = wp_mix3(vec3f(0.45, 0.58, 0.88), sun_g, dw) * light.gl_i;
            light.halation = wp_mix3(
                vec3f(0.12, 0.16, 0.26),
                vec3f(1.0, 0.42 + 0.1 * (1.0 - warm), 0.2) * 0.95,
                dw,
            ) * light.gl_i;
        } else if dark {
            light.night = 1.0;
            light.tint = vec3f(0.64, 0.83, 1.2);
            light.expo = 0.16;
            light.glint = vec3f(0.45, 0.58, 0.88);
            light.halation = vec3f(0.12, 0.16, 0.26);
            light.gc.x = 0.64 * asp;
        }
        light
    }

    pub fn wp_cloud_at(ps: Vec2f, gdir: Vec2f, stretch: f32, night: f32) -> f32 {
        let u = get!(BLUR_LOCALS);
        let gn = vec2f(-gdir.y, gdir.x);
        let mut cq = (gdir * (dot(ps, gdir) / stretch) + gn * dot(ps, gn)) * vec2f(2.0, 2.6);
        let ang = WP_TAU * u.wp_frame.w / WP_CLOUD_PERIOD;
        cq = cq - vec2f(cos(ang), 0.45 * sin(ang));
        mix(u.wp_p1.x, 1.0, night)
            * smoothstep(0.5, 0.92, wp_vn(cq + vec2f(0.0, u.wp_env.z * 50.0) + 40.0))
    }

    pub fn wp_shore_finish(
        col0: Vec3f,
        fo: f32,
        s: f32,
        gl: f32,
        ha: f32,
        sheen: f32,
        light: WpShoreLight,
        fc: Vec2f,
    ) -> Vec3f {
        let u = get!(BLUR_LOCALS);
        let res = u.wp_frame.xy();
        let nt = light.night;
        let col = wp_mix3(
            col0,
            wp_mix3(vec3f(0.80, 0.85, 0.86), vec3f(0.97, 0.965, 0.95), fo),
            fo * 0.95,
        );
        let mut lc = col * col;
        if nt > 0.0 {
            lc = lc
                * mix(
                    1.0,
                    mix(0.55, 1.0, smoothstep(-0.02, 0.0, s)) * (1.0 + 0.6 * sheen),
                    nt,
                );
            let ly = wp_lum(lc);
            lc = wp_mix3(vec3f(ly, ly, ly), lc, mix(1.0, 0.42, nt));
        }
        lc = lc * light.tint * light.expo;
        lc = lc + light.glint * gl + light.halation * ha;
        let mut y = sqrt(wp_filmic(lc));
        let mut l = wp_lum(y);
        let sh_t = wp_mix3(vec3f(-0.035, 0.01, 0.03), vec3f(-0.02, 0.01, 0.03), nt);
        let hi_t = wp_mix3(vec3f(0.04, 0.012, -0.035), vec3f(0.01, 0.008, 0.0), nt);
        y = y + sh_t * ((1.0 - l) * (1.0 - l)) + hi_t * (l * l);
        y = wp_mix3(
            vec3f(l, l, l),
            y,
            mix(0.88, 0.82, nt) * min(0.72 + 0.7 * l, 1.0),
        );
        y = wp_mix3(vec3f(0.012, 0.018, 0.024), vec3f(0.016, 0.02, 0.028), nt) + y * 0.975;
        let v = (fc / res - 0.5) * vec2f(1.0, 0.9);
        y = y * (1.0 - 0.32 * dot(v, v));
        let fr = floor(u.wp_frame.w * 24.0);
        let ox = wp_mod(fr * 71.0, 256.0) as i32;
        let oy = wp_mod(fr * 137.0, 256.0) as i32;
        let g = wp_texel(fc.x as i32 + ox, fc.y as i32 + oy);
        l = wp_lum(y);
        y = y + vec3f(1.0, 1.0, 1.0)
            * ((g.x + g.y - 1.0)
                * mix(0.038, 0.026, nt)
                * (0.35 + 0.65 * (1.0 - abs(2.0 * l - 1.0))));
        clamp(y, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0))
    }

    pub fn wp_shore(fc: Vec2f) -> Vec4f {
        let u = get!(BLUR_LOCALS);
        let res = u.wp_frame.xy();
        let ps = fc / res.y;
        let asp = res.x / res.y;
        let t = u.wp_frame.w;
        let seed = u.wp_env.z;
        let light = wp_shore_light(asp);

        let sx = seed * 37.0;
        let px = WP_ZOOM / res.y;

        let anchor = select(0.55, u.wp_rect.y / res.y, u.wp_rect.w > 1.0);
        let base = clamp(anchor - 0.07 + (u.wp_p0.z - 0.5) * 0.36, 0.22, 0.8);
        let p = vec2f(ps.x - 0.5 * res.x / res.y, ps.y - base) * WP_ZOOM;
        let kx = 2.1;
        let ph = 0.8 + seed * 3.0;
        let xn = wp_vn4(vec2f(p.x * 1.6 + sx, 3.5));
        let ys = 0.045 * sin(p.x * kx + ph) + 0.03 * (xn.x - 0.5);
        let slope = 0.045 * kx * cos(p.x * kx + ph);
        let s = (ys - p.y) / sqrt(1.0 + slope * slope);
        let x = p.x;

        let pal = wp_shore_palette(u.wp_p0.x as i32);
        let sky = vec3f(0.70, 0.80, 0.88);

        let sn = wp_vn4(p * vec2f(9.0, 12.0) + sx);
        let ripple = sin((p.y * 0.94 + p.x * 0.34) * 330.0 + 7.0 * sn.z + 3.0 * sn.w);
        let mut sand = pal.dry * (1.0 + 0.018 * light.relief * ripple * sn.w);
        if s < -0.125 {
            let shaded =
                sand * (1.0 - 0.24 * wp_cloud_at(ps, light.gdir, light.stretch, light.night));
            let out = wp_shore_finish(shaded, 0.0, s, 0.0, 0.0, 0.0, light, fc);
            return vec4f(out.x, out.y, out.z, 1.0);
        }

        let n = clamp(floor(18.0 * u.wp_p0.y + 0.5), 8.0, 30.0);
        let big_t = WP_SHORE_PERIOD / n;
        let lam = 0.13;
        let tw = t / big_t;
        let eta = s / lam + tw;
        let sb = 0.17 + 0.05 * xn.y;
        let mut w = 0.30 * (wp_pn(vec2f(x * 2.4 + sx, eta), n) - 0.5)
            + 0.10 * (wp_vn4(vec2f(x * 9.0 + sx, 60.0)).z - 0.5);
        w = w * lam;
        let uu = s + w;
        let phi = uu / lam + tw;
        let fr = fract(phi);
        let j = floor(phi);
        let bn = wp_vn4(vec2f(x * 3.0 + sx, s * 4.0 + 20.0));
        let d = max(s, 0.0) * (0.85 + 0.5 * bn.x);

        let scroll = t / WP_SHORE_PERIOD * 256.0;
        let r1 = wp_vn4(vec2f(x * 130.0 + scroll * 0.5, s * 150.0));
        let bed = wp_mix3(pal.wet, pal.dry, 0.7) * (0.92 + 0.08 * (bn.y - 0.5));
        let tr = exp(-pal.absorb * (d * 13.0 + 0.25 * smoothstep(0.0, 0.02, d)));
        let mut wc = bed * tr + pal.scat * (vec3f(1.0, 1.0, 1.0) - tr) * (0.94 + 0.12 * bn.y);
        let outside = smoothstep(sb - 0.03, sb + 0.05, uu);
        let steep = mix(0.35, 1.0, smoothstep(sb + 0.3, sb, uu));
        let crest = 0.5 + 0.5 * cos(6.2831853 * (fr - 0.12)) + 0.5 * exp(-(1.0 - fr) * 14.0);
        wc = wc * (1.0 + 0.08 * (crest - 0.6) * outside * steep);
        let deepish = smoothstep(0.01, 0.12, d);
        wc = wp_mix3(wc, sky, (0.03 + 0.05 * r1.z * r1.z) * deepish);
        let dq = ps - light.gc;
        let al = dot(dq, light.gdir) / light.g_len;
        let ac = dot(dq, vec2f(-light.gdir.y, light.gdir.x)) / light.g_wid;
        let r2 = al * al + ac * ac;
        let q1 = max(1.0 - r2 * 0.25, 0.0);
        let q2 = max(1.0 - light.sh_w * r2 * 0.25, 0.0);
        let sun = q1 * q1 * q1 * q1 * smoothstep(0.05, 0.35, d);
        let sheen = q2 * q2 * q2 * q2 * smoothstep(-0.02, 0.2, s) * light.gl_i;
        let mut gl_e = 0.0;
        let mut ha_e = 0.0;
        if sun > 0.01 {
            let gfc = fc * vec2f(0.5, 0.85);
            let mut g = wp_vn(gfc + vec2f(2.0 * scroll, scroll)) * (0.25 + 0.75 * r1.y);
            g = g * (0.75 + 0.5 * r1.w);
            gl_e = sun * sun * smoothstep(0.78 - 0.18 * light.lo, 0.92 - 0.1 * light.lo, g) * 2.6;
            ha_e = sun * sun * (smoothstep(0.5, 0.8, g) * 0.14 + 0.025);
            gl_e = gl_e + sun * sun * light.path_k;
        }

        let amt = u.wp_p0.w;
        let mut foam = 0.0;
        let mut turb = 0.0;
        if uu < sb + 0.03 && s > -0.01 {
            let a = fr;
            let hh = wp_hash4(wp_mod(j, n) * 7.0 + 3.0);
            let size = 0.45 + 0.55 * hh.w;
            let sj = sb * (0.72 + 0.4 * size);
            let brk = smoothstep(sj + 0.01, sj - 0.025, uu);
            let hz = wp_vn4(vec2f(
                x * 6.0 + hh.z * 20.0,
                uu * 16.0 + a * 1.5 + hh.z * 30.0,
            ));
            let mut cfoam = (exp(-a * 2.6) + 0.22 * exp(-a * 1.0))
                * size
                * (0.55 + 0.6 * hz.y)
                * (0.6 + 0.8 * amt)
                * brk;
            let pch = smoothstep(0.2, 0.75, hz.x);
            cfoam = cfoam
                * mix(
                    0.85 + 0.3 * pch,
                    0.35 + 0.95 * pch,
                    smoothstep(0.05, 0.4, a),
                );
            turb = clamp(
                cfoam * 1.3 + 0.25 * brk * size * (0.5 + 0.5 * hz.z),
                0.0,
                1.0,
            );
            if cfoam > 0.05 {
                let q = vec2f(x, uu + 0.2 * lam * a) + vec2f(17.3, 5.1) * hh.z;
                foam = wp_foam_at(q, 80.0, cfoam, a);
            }
        }
        let milky = wp_mix3(pal.scat, vec3f(0.62, 0.76, 0.72), 0.55);
        wc = wp_mix3(wc, milky, 0.42 * turb);

        let mut col = wc;
        let mut fo = foam;
        if s < 0.004 {
            let hw = -0.1 - 0.015 * wp_vn(vec2f(x * 6.0 + sx, 90.0));
            let wet_amt =
                smoothstep(hw, hw + 0.06, s) * (0.8 + 0.2 * smoothstep(hw + 0.01, -0.01, s));
            sand = wp_mix3(sand, pal.wet, wet_amt * 0.75);
            let phi0 = w / lam + tw;
            let tau = fract(phi0);
            let hh = wp_hash4(wp_mod(floor(phi0), n) * 7.0 + 3.0);
            let size = 0.45 + 0.55 * hh.w;
            let rr = 0.07
                * (0.5 + 0.5 * size)
                * (0.6 + 0.7 * wp_vn4(vec2f(x * 3.0 + hh.y * 50.0 + sx, 80.0)).x)
                * (0.85 + 0.3 * wp_vn4(vec2f(x * 12.0 + hh.y * 20.0, 81.0)).y);
            let z = clamp(tau / 0.92, 0.0, 1.0);
            let zz = pow(z, 0.7);
            let e = -rr * 4.0 * zz * (1.0 - zz);
            let up = 1.0 - smoothstep(0.4, 0.6, zz);
            let cov = smoothstep(e - px, e + px, s);
            let h = clamp((s - e) / max(-e, 0.0001), 0.0, 1.0);
            let mut gloss = wet_amt * (0.08 + 0.1 * sn.y);
            let trail = step(-rr, s) * (1.0 - cov) * (1.0 - up) * (1.0 - z);
            gloss = gloss + 0.22 * trail;
            let mut sc = wp_mix3(sand * (1.0 - 0.12 * trail), sky, gloss);
            let film = wp_mix3(
                wp_mix3(pal.wet * 0.92, sky, 0.22),
                wc,
                0.1 + 0.85 * h * sqrt(h),
            );
            sc = wp_mix3(sc, film, cov);
            let mut sf = 0.0;
            let cs = (0.3 * exp(-tau * 2.5) + exp(-(s - e) / 0.006) * mix(0.45, 1.0, up))
                * (0.5 + 0.8 * amt);
            if cov > 0.0 && cs > 0.05 {
                let q = vec2f(x, s - 0.6 * e) + vec2f(13.0, 3.0) * hh.y;
                sf = wp_foam_at(q, 90.0, cs, 1.0) * cov * smoothstep(0.0, 0.04, z);
            }
            let fs = max(fo * step(0.0, s), sf);
            let k0 = smoothstep(0.0, 0.004, s);
            col = wp_mix3(sc, wc, k0);
            fo = mix(fs, fo, k0);
        }
        let wet0 = smoothstep(0.0, 0.01, s);
        let cloud = wp_cloud_at(ps, light.gdir, light.stretch, light.night);
        col = col * (1.0 - 0.24 * cloud);
        let cg = wet0 * (1.0 - 0.85 * cloud);
        let out = wp_shore_finish(col, fo, s, gl_e * cg, ha_e * cg, sheen, light, fc);
        vec4f(out.x, out.y, out.z, 1.0)
    }

    // ------------------------------ butler.riso-print (density) ------------------------------

    pub fn wp_col_sd(p: Vec2f, col: Vec4f) -> f32 {
        let sx = max(col.x - p.x, p.x - col.y);
        let t = col.z - p.y;
        select(sx, max(sx, t), t > 0.0)
    }

    pub fn wp_col_cap(sd: f32, k: f32) -> f32 {
        mix(
            0.25 * smoothstep(-90.0 * k, 0.0, sd),
            1.0,
            smoothstep(0.0, 120.0 * k, sd),
        )
    }

    pub fn wp_k_scale(r: Vec2f) -> f32 {
        clamp(sqrt(r.x * r.y) / 1006.0, 0.45, 1.2)
    }

    pub fn wp_riso_blob(p: Vec2f, c: Vec2f, r: f32, warp: f32, k: f32) -> f32 {
        1.0 - smoothstep(r - 150.0 * k, r + 150.0 * k, length(p - c) + warp)
    }

    pub fn wp_nz(x: Vec2f) -> Vec4f {
        let i = floor(x);
        let f0 = x - i;
        let f = f0 * f0 * (vec2f(3.0, 3.0) - f0 * 2.0);
        wp_noise((i + f + 0.5) / 256.0)
    }

    pub fn wp_fbm4(p0: Vec2f) -> Vec4f {
        let mut s = vec4f(0.0, 0.0, 0.0, 0.0);
        let mut a = 0.5;
        let mut p = p0;
        for _i in 0..4 {
            s = s + wp_nz(p) * a;
            // mat2(1.6, 1.2, -1.2, 1.6) * p (GLSL column-major)
            p = vec2f(1.6 * p.x - 1.2 * p.y, 1.2 * p.x + 1.6 * p.y) + vec2f(17.3, 9.1);
            a = a * 0.5;
        }
        s / 0.9375
    }

    pub fn wp_masses(
        p: Vec2f,
        r: Vec2f,
        w: Vec4f,
        ft: f32,
        comp: f32,
        col: Vec4f,
        seed: Vec2f,
    ) -> f32 {
        let k = wp_k_scale(r);
        let hh = r.y;
        let ww = r.x;
        let ss = k * 1006.0;
        let warp = (w.x - 0.5) * 160.0 * k;
        let dr = vec2f(sin(ft * 0.9), cos(ft * 0.7)) * (28.0 * k);
        let drx = vec2f(dr.y, dr.x);
        let mut m = 0.0;
        if comp < 0.5 {
            m = max(
                m,
                wp_riso_blob(p, vec2f(0.02 * ww, hh - 0.02 * ss) + dr, 0.41 * ss, warp, k),
            );
            m = max(
                m,
                wp_riso_blob(
                    p,
                    vec2f(ww - 0.01 * ss, 0.054 * ss) - dr,
                    0.30 * ss,
                    warp,
                    k,
                ),
            );
            m = max(
                m,
                0.75 * wp_riso_blob(p, vec2f(ww, hh - 0.13 * ss) + drx, 0.14 * ss, warp, k),
            );
        } else if comp < 1.5 {
            m = max(m, wp_riso_blob(p, vec2f(0.0, hh) + dr, 0.32 * ss, warp, k));
            m = max(m, wp_riso_blob(p, vec2f(ww, 0.0) - dr, 0.27 * ss, warp, k));
            m = max(
                m,
                0.85 * wp_riso_blob(p, vec2f(0.0, 0.0) + drx, 0.16 * ss, warp, k),
            );
            m = max(
                m,
                0.85 * wp_riso_blob(p, vec2f(ww, hh) - drx, 0.20 * ss, warp, k),
            );
        } else {
            m = max(
                m,
                wp_riso_blob(p, vec2f(0.02 * ww, hh + 0.02 * ss) + dr, 0.47 * ss, warp, k),
            );
            m = max(
                m,
                wp_riso_blob(
                    p,
                    vec2f(ww - 0.02 * ss, hh + 0.054 * ss) - dr,
                    0.36 * ss,
                    warp,
                    k,
                ),
            );
        }
        let side = step(0.5 * (col.x + col.y), p.x);
        let rr =
            0.65 * wp_nz(vec2f(
                p.y / hh * 2.6 + ft * 0.06,
                side * 13.7 + seed.x + 3.1,
            ))
            .x + 0.35 * wp_nz(vec2f(p.y / hh * 5.3 - ft * 0.04, side * 5.1 + seed.y)).y;
        let r2 = wp_nz(vec2f(p.y / hh * 1.9 - ft * 0.05, side * 7.9 + seed.y + 1.7)).z;
        let reach = (-125.0 + 65.0 * smoothstep(0.28, 0.72, rr)) * k;
        let ramp = (380.0 + 180.0 * smoothstep(0.25, 0.75, r2)) * k;
        m * clamp((wp_col_sd(p, col) - reach) / ramp, 0.0, 1.0)
    }

    /// A content rect in drawing-buffer px (bottom-left origin) as CSS px (top-left origin).
    pub fn wp_riso_css_rect(r: Vec4f, css: Vec2f, res: Vec2f) -> Vec4f {
        let s = css / res;
        select(
            vec4f(-9999.0, -9999.0, 0.0, 0.0),
            vec4f(r.x * s.x, css.y - (r.y + r.w) * s.y, r.z * s.x, r.w * s.y),
            r.z > 0.0 && r.w > 0.0,
        )
    }

    pub fn wp_riso_weights() -> Vec3f {
        let u = get!(BLUR_LOCALS);
        let b = select(
            vec3f(1.0, 0.86, 0.9),
            vec3f(0.42, 0.5, 1.05),
            u.wp_p0.x > 0.5,
        );
        let dp = u.wp_env.y;
        b * (vec3f(
            cos(6.2831853 * (dp - 0.33)),
            cos(6.2831853 * (dp - 0.6)),
            cos(6.2831853 * (dp - 0.85)),
        ) * 0.16
            + 0.84)
    }

    pub fn wp_riso_density(fc: Vec2f) -> Vec4f {
        let u = get!(BLUR_LOCALS);
        let dres = u.wp_frame.xy();
        let r = u.wp_css.xy();
        let seed = vec2f(13.7, 41.3) + vec2f(3.7, 2.3) * u.wp_p0.w;
        let ft = u.wp_css.z;
        let mis = u.wp_css.w;
        let night = select(0.0, 1.0, u.wp_p0.y > 0.5);
        let comp = u.wp_p0.z;
        let drop = u.wp_drop;
        let keep_rect = wp_riso_css_rect(u.wp_rect, r, dres);
        let keep2_rect = wp_riso_css_rect(u.wp_rect2, r, dres);
        let col = u.wp_col;
        let uw = wp_riso_weights();

        let p = vec2f(fc.x, dres.y - fc.y) / dres * r;
        let sc = max(420.0, 0.5 * max(r.x, r.y));
        let q = (p - r * 0.5) / sc;
        let w = wp_fbm4(q * 1.25 + seed + vec2f(ft * 0.32, -ft * 0.21));
        let base = q * 2.05 + (w.xy() - 0.5) * 2.5 + vec2f(seed.y, seed.x) * 0.7;
        let n = wp_fbm4(base + vec2f(1.7, 9.2) + vec2f(0.16, -0.11) * ft);
        let mut d = smoothstep(vec3f(0.32, 0.34, 0.33), vec3f(0.88, 0.90, 0.87), n.xyz())
            * (0.9 + 0.42 * w.w);
        let cc = keep_rect.xy() + keep_rect.zw() * 0.5;
        let bq = abs(p - cc) - keep_rect.zw() * 0.5;
        let sd_c = length(max(bq, vec2f(0.0, 0.0))) + min(max(bq.x, bq.y), 0.0);
        let mut keep = smoothstep(
            -30.0,
            440.0,
            sd_c + (w.z - 0.5) * 260.0 + (n.w - 0.5) * 240.0,
        );
        let e = abs(p - r * 0.5) / (r * 0.5);
        let rim = smoothstep(0.25, 1.1, length(e) * 0.85 + (n.w - 0.5) * 0.6);
        let c2 = keep2_rect.xy() + keep2_rect.zw() * 0.5;
        let b2 = abs(p - c2) - keep2_rect.zw() * 0.5;
        let sd2 = length(max(b2, vec2f(0.0, 0.0))) + min(max(b2.x, b2.y), 0.0);
        let keep2 = mix(0.3, 1.0, smoothstep(-10.0, 160.0, sd2 + (w.y - 0.5) * 80.0));
        let sdc = wp_col_sd(p, col);
        let kk = wp_k_scale(r);
        let colm = mix(1.0, wp_col_cap(sdc, kk), col.w);
        keep = mix(keep, wp_masses(p, r, w, ft, comp, col, seed), col.w);
        d = d
            * (keep * keep2 * mix(0.3, 1.0, rim) * mix(0.86, 1.0, night) * (1.0 - 0.08 * mis))
            * uw;
        let cap = wp_col_cap(sdc, kk);
        d = wp_mix3(d, min(d, vec3f(cap, cap, cap)), col.w);
        d = d * mix(
            1.0,
            mix(0.15, 1.0, smoothstep(-40.0 * kk, 80.0 * kk, sdc)),
            col.w,
        );
        if drop.w > 0.0 {
            let pg = drop.z;
            let rd = 14.0 + 160.0 * (1.0 - pow(1.0 - pg, 3.0));
            let rr = length(p - drop.xy())
                + (w.x - 0.5) * 80.0 * smoothstep(0.0, 0.35, pg)
                + (n.y - 0.5) * 30.0;
            let fade =
                drop.w * (1.0 - smoothstep(0.45, 1.0, pg)) * smoothstep(0.3, 0.8, keep) * colm;
            let core = 1.0 - smoothstep(rd * 0.35, rd, rr);
            let ring =
                smoothstep(rd * 0.45, rd * 0.85, rr) * (1.0 - smoothstep(rd * 0.85, rd * 1.2, rr));
            d.z = max(d.z, core * 0.78 * fade);
            d.x = max(d.x, ring * 0.45 * fade * smoothstep(0.15, 0.4, pg));
            d.y = max(d.y, ring * 0.32 * fade * smoothstep(0.2, 0.5, pg));
        }
        let dc = clamp(d, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0));
        vec4f(dc.x, dc.y, dc.z, 1.0)
    }

    // ------------------------------- butler.riso-print (print) -------------------------------

    pub fn wp_col_gate(p: Vec2f, col: Vec4f) -> f32 {
        let sx = max(col.x - p.x, p.x - col.y);
        let t = col.z - p.y;
        let sd = select(sx, max(sx, t), t > 0.0);
        mix(1.0, smoothstep(-130.0, -90.0, sd), col.w)
    }

    pub fn wp_hash(p: Vec2f) -> f32 {
        let mut p3 = fract(vec3f(p.x, p.y, p.x) * 0.1031);
        p3 = p3 + dot(p3, vec3f(p3.y, p3.z, p3.x) + 33.33);
        fract((p3.x + p3.y) * p3.z)
    }

    pub fn wp_den(p: Vec2f) -> Vec3f {
        let css = get!(BLUR_LOCALS).wp_css.xy();
        wp_density(vec2f(p.x / css.x, p.y / css.y))
    }

    pub fn wp_dot(p: Vec2f, ang: f32, cell: f32, d: f32, pr: f32, gain: f32) -> f32 {
        let c = cos(ang);
        let s = sin(ang);
        // mat2(c, -s, s, c) * p / cell (GLSL column-major)
        let q = vec2f(c * p.x + s * p.y, -s * p.x + c * p.y) / cell;
        let g = fract(q) - 0.5;
        let r = sqrt(d * 0.3183) + 0.15 * smoothstep(0.55, 1.0, d) + 0.02 + gain * 0.07;
        let aa = 1.1 / (cell * pr) + 0.022;
        (1.0 - smoothstep(r - aa, r + aa, length(g))) * smoothstep(0.006, 0.06, d)
    }

    pub fn wp_riso_print(fc: Vec2f) -> Vec4f {
        let u = get!(BLUR_LOCALS);
        let res = u.wp_frame.xy();
        let pr = u.wp_frame.z;
        let mis = u.wp_css.w;
        let gain = u.wp_print.x;
        let night = u.wp_p0.y > 0.5;
        let bleed = 0.16;
        let col = u.wp_col;
        let paper0 = select(vec3f(250.0, 250.0, 248.0), vec3f(25.0, 28.0, 36.0), night) / 255.0;
        let ua = select(vec3f(0.0, 120.0, 191.0), vec3f(96.0, 140.0, 255.0), night) / 255.0;
        let ub = select(vec3f(255.0, 72.0, 176.0), vec3f(255.0, 112.0, 190.0), night) / 255.0;
        let uc = select(vec3f(96.0, 56.0, 156.0), vec3f(150.0, 96.0, 236.0), night) / 255.0;

        let p = vec2f(fc.x, res.y - fc.y) / pr;
        let d = wp_den(p) * 0.36
            + (wp_den(p + vec2f(6.0, 2.5))
                + wp_den(p + vec2f(-2.5, 6.0))
                + wp_den(p + vec2f(-6.0, -2.5))
                + wp_den(p + vec2f(2.5, -6.0)))
                * 0.16;
        let h = (wp_den(p + vec2f(22.0, 9.0))
            + wp_den(p + vec2f(-9.0, 22.0))
            + wp_den(p + vec2f(-22.0, -9.0))
            + wp_den(p + vec2f(9.0, -22.0)))
            * 0.25;
        let fib = 0.55 * wp_hash(floor(p * 0.5) + 3.0)
            + 0.45 * wp_hash(floor(p * vec2f(0.22, 0.9)) + 11.0);
        let wash = smoothstep(vec3f(0.02, 0.02, 0.02), vec3f(0.6, 0.6, 0.6), max(h, d))
            * (bleed * (0.45 + 0.55 * fib));
        let mo = 0.6 + 3.0 * mis;
        let m_a = vec2f(1.6, -1.0) * mo;
        let m_b = vec2f(-1.3, 1.2) * mo;
        let mut cov = vec3f(
            wp_dot(p + m_a, 0.2618, 6.5, d.x, pr, gain),
            wp_dot(p + m_b, 1.309, 6.5, d.y, pr, gain),
            wp_dot(p, 0.7854, 6.5, d.z, pr, gain),
        );
        cov = cov * smoothstep(0.0, 0.12, wp_hash(floor(p * 0.75) + 7.0) + 0.06);
        cov = max(cov, wash) * wp_col_gate(p, col);
        let tooth = wp_hash(floor(p)) - 0.5;
        let paper = paper0 * (1.0 + select(0.03, 0.1, night) * tooth);
        let one = vec3f(1.0, 1.0, 1.0);
        let mut outc = vec3f(0.0, 0.0, 0.0);
        if !night {
            let k = cov * 0.9;
            outc = paper * wp_mix3(one, ua, k.x) * wp_mix3(one, ub, k.y) * wp_mix3(one, uc, k.z);
        } else {
            let m = wp_mix3(one, ua, cov.x) * wp_mix3(one, ub, cov.y) * wp_mix3(one, uc, cov.z);
            let un = 1.0 - (1.0 - cov.x) * (1.0 - cov.y) * (1.0 - cov.z);
            outc = wp_mix3(paper, m * 0.86, un * 0.9);
        }
        vec4f(outc.x, outc.y, outc.z, 1.0)
    }

    /// One wallpaper pass into its offscreen target: wp_mode.y picks the module (1 bloom,
    /// 2 dusk, 3 shoreline, 4 riso print density pass, 5 riso print print pass).
    #[fragment]
    pub fn fragment_wallpaper(input: BlurVarying) -> Vec4f {
        let fc = wp_frag_coord(input.position.xy());
        let which = get!(BLUR_LOCALS).wp_mode.y;
        if which < 1.5 {
            return wp_bloom(fc);
        }
        if which < 2.5 {
            return wp_dusk(fc);
        }
        if which < 3.5 {
            return wp_shore(fc);
        }
        if which < 4.5 {
            return wp_riso_density(fc);
        }
        wp_riso_print(fc)
    }

    #[vertex]
    pub fn vertex_smoothed_blur_composite(
        #[builtin(vertex_index)] vertex_id: u32,
    ) -> SmoothedBlurVarying {
        let vertex = prepare_blur_composite_vertex(vertex_id);
        let prepared = prepare_corners(
            get!(BLUR_LOCALS).bounds.size,
            get!(BLUR_LOCALS).corner_radii,
            get!(BLUR_LOCALS).corner_smoothing,
            true,
        );
        SmoothedBlurVarying {
            position: vertex.position,
            texture_coordinates: vertex.texture_coordinates,
            horizontal_corner_reaches: prepared.horizontal_reaches,
            vertical_corner_reaches: prepared.vertical_reaches,
            clip_distances: vertex.clip_distances,
            smoothing_factors: prepared.smoothing_factors,
            superellipse_power: prepared.superellipse_power,
        }
    }

    #[fragment]
    pub fn fragment_smoothed_blur_composite(input: SmoothedBlurVarying) -> Vec4f {
        if is_clipped(input.clip_distances) {
            return transparent();
        }
        let coverage = select(
            1.0,
            antialiased_coverage(prepared_corner_signed_distance(
                input.position.xy(),
                get!(BLUR_LOCALS).bounds,
                get!(BLUR_LOCALS).corner_radii,
                get!(BLUR_LOCALS).corner_smoothing,
                PreparedCorners {
                    horizontal_reaches: input.horizontal_corner_reaches,
                    vertical_reaches: input.vertical_corner_reaches,
                    smoothing_factors: input.smoothing_factors,
                    superellipse_power: input.superellipse_power,
                },
            )),
            get!(BLUR_LOCALS).composite_clip == BlurCompositeClip::RoundedBounds,
        );
        blur_composite_color(input.position.xy(), coverage)
    }
}

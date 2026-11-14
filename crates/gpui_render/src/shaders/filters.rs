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

#[wgsl_rs::wgsl(skip_validation)]
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
        pub lens_saturation: f32,
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
        pub composite_scale: f32,
        pub composite_scale_padding: f32,
        pub composite_scale_origin: Vec2f,
        pub composite_offset: Vec2f,
        pub composite_offset_padding: Vec2f,
        pub composite_rotation: f32,
        pub composite_rotation_padding0: f32,
        pub composite_rotation_padding1: f32,
        pub composite_rotation_padding2: f32,
        pub motion_clip: Bounds,
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
        let source_position =
            b.origin - vec2f(grow, grow) + unit_position * (b.size + vec2f(2.0 * grow, 2.0 * grow));
        let scale = get!(BLUR_LOCALS).composite_scale;
        let origin = get!(BLUR_LOCALS).composite_scale_origin;
        let offset = get!(BLUR_LOCALS).composite_offset;
        let rotation = get!(BLUR_LOCALS).composite_rotation;
        let scaled_position = (source_position - origin) * scale;
        let viewport_position = origin
            + vec2f(
                cos(rotation) * scaled_position.x - sin(rotation) * scaled_position.y,
                sin(rotation) * scaled_position.x + cos(rotation) * scaled_position.y,
            )
            + offset;
        let mut content_mask = get!(BLUR_LOCALS).content_mask;
        if get!(BLUR_LOCALS).composite_scale != 1.0 {
            content_mask.origin = origin + (content_mask.origin - origin) * scale;
            content_mask.size = content_mask.size * scale;
        }
        BlurCompositeVertexData {
            position: viewport_to_clip_position(viewport_position),
            texture_coordinates: unit_position,
            clip_distances: clip_distances(viewport_position, content_mask),
        }
    }

    pub fn sample_composite_source(position: Vec2f) -> Vec4f {
        texture_sample_level(
            BLUR_TEXTURE,
            BLUR_SAMPLER,
            position / get!(BLUR_LOCALS).source_size,
            0.0,
        )
    }

    pub fn blur_composite_source_position(destination_position: Vec2f) -> Vec2f {
        let locals = get!(BLUR_LOCALS);
        let position =
            destination_position - locals.composite_offset - locals.composite_scale_origin;
        let rotation = locals.composite_rotation;
        locals.composite_scale_origin
            + vec2f(
                cos(rotation) * position.x + sin(rotation) * position.y,
                -sin(rotation) * position.x + cos(rotation) * position.y,
            ) / locals.composite_scale
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
        let mut r = select(r0, lens_tone(red.x, red.w), toned);
        let mut g = select(g0, lens_tone(green.y, green.w), toned);
        let mut b = select(b0, lens_tone(blue.z, blue.w), toned);
        if locals.lens_saturation != 1.0 {
            let saturated = lens_saturate(r, g, b, locals.lens_saturation);
            r = saturated.x;
            g = saturated.y;
            b = saturated.z;
        }

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

    /// CSS Filter Effects saturation matrix, clamped to the normalized color range.
    pub fn lens_saturate(r: f32, g: f32, b: f32, s: f32) -> Vec3f {
        clamp(
            vec3f(
                (0.213 + 0.787 * s) * r + (0.715 - 0.715 * s) * g + (0.072 - 0.072 * s) * b,
                (0.213 - 0.213 * s) * r + (0.715 + 0.285 * s) * g + (0.072 - 0.072 * s) * b,
                (0.213 - 0.213 * s) * r + (0.715 - 0.715 * s) * g + (0.072 + 0.928 * s) * b,
            ),
            vec3f(0.0, 0.0, 0.0),
            vec3f(1.0, 1.0, 1.0),
        )
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
            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            if lens_tone_on() {
                r = lens_tone(c.x, c.w);
                g = lens_tone(c.y, c.w);
                b = lens_tone(c.z, c.w);
            } else {
                let lift = clamp(locals.lens_lift, -1.0, 1.0);
                let lift_to = select(0.0, c.w, lift > 0.0) * abs(lift);
                let keep = 1.0 - abs(lift);
                r = c.x * keep + lift_to;
                g = c.y * keep + lift_to;
                b = c.z * keep + lift_to;
            }
            if locals.lens_saturation != 1.0 {
                let saturated = lens_saturate(r, g, b, locals.lens_saturation);
                r = saturated.x;
                g = saturated.y;
                b = saturated.z;
            }
            return vec4f(r, g, b, c.w);
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
    /// Composites the pass output as premultiplied colour, scaling its own alpha too.
    pub fn wallpaper_composite_color(position: Vec2f, factor: f32) -> Vec4f {
        let u = get!(BLUR_LOCALS);
        let size = u.wp_frame.xy();
        let uv = (position - u.bounds.origin) / u.bounds.size;
        let lo = vec2f(0.5, 0.5) / size;
        let tc = clamp(uv, lo, vec2f(1.0, 1.0) - lo);
        let c = texture_sample_level(BLUR_TEXTURE, BLUR_SAMPLER, tc, 0.0);
        vec4f(c.x * factor, c.y * factor, c.z * factor, c.w * factor)
    }

    pub fn blur_composite_color(position: Vec2f, coverage: f32) -> Vec4f {
        let factor = coverage * get!(BLUR_LOCALS).opacity;
        if get!(BLUR_LOCALS).wp_mode.x > 1.5 {
            return wallpaper_composite_color(position, factor);
        }
        if get!(BLUR_LOCALS).lens_bezel > 0.0 || get!(BLUR_LOCALS).lens_saturation != 1.0 {
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
        if get!(BLUR_LOCALS).downsample_mode == DownsampleMode::Fill {
            return get!(BLUR_LOCALS).wp_mode;
        }
        if get!(BLUR_LOCALS).downsample_mode == DownsampleMode::HalfResolution {
            let destination = floor(input.position.xy());
            let source_coordinates = min(
                (destination + 0.5) / get!(BLUR_LOCALS).target_size,
                (get!(BLUR_LOCALS).source_size - 0.5) / get!(BLUR_LOCALS).source_size,
            );
            return texture_sample_level(BLUR_TEXTURE, BLUR_SAMPLER, source_coordinates, 0.0);
        }
        texture_sample_level(
            BLUR_TEXTURE,
            BLUR_SAMPLER,
            input.position.xy() / get!(BLUR_LOCALS).source_size,
            0.0,
        )
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
        let mut coverage = select(
            1.0,
            antialiased_coverage(rounded_rectangle_signed_distance(
                input.position.xy(),
                get!(BLUR_LOCALS).bounds,
                get!(BLUR_LOCALS).corner_radii,
            )),
            get!(BLUR_LOCALS).composite_clip == BlurCompositeClip::RoundedBounds,
        );
        if get!(BLUR_LOCALS).composite_clip == BlurCompositeClip::MotionRounded {
            coverage = coverage
                * antialiased_coverage(rounded_rectangle_signed_distance(
                    blur_composite_source_position(input.position.xy()),
                    get!(BLUR_LOCALS).motion_clip,
                    get!(BLUR_LOCALS).corner_radii,
                ));
        }
        blur_composite_color(
            blur_composite_source_position(input.position.xy()),
            coverage,
        )
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
        let mut coverage = select(
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
        if get!(BLUR_LOCALS).composite_clip == BlurCompositeClip::MotionRounded {
            coverage = coverage
                * antialiased_coverage(rounded_rectangle_signed_distance(
                    blur_composite_source_position(input.position.xy()),
                    get!(BLUR_LOCALS).motion_clip,
                    get!(BLUR_LOCALS).corner_radii,
                ));
        }
        blur_composite_color(
            blur_composite_source_position(input.position.xy()),
            coverage,
        )
    }
}

// WGSL needs explicit pointer dereferences and has no mem::swap; retain GLSL literals.
#[allow(
    clippy::excessive_precision,
    clippy::explicit_auto_deref,
    clippy::manual_swap
)]
#[wgsl_rs::wgsl(skip_validation)]
pub mod wallpaper {
    use super::super::common::*;
    use super::blur::*;
    use wgsl_rs::std::*;

    texture!(group(1), binding(3), WP_PREVIOUS: Texture2D<f32>);
    texture!(group(1), binding(4), WP_IMAGE: Texture2D<f32>);
    sampler!(group(1), binding(5), WP_SAMPLER: Sampler);

    pub fn wp_bitand2i(a: Vec2i, b: Vec2i) -> Vec2i {
        vec2i(a.x & b.x, a.y & b.y)
    }
    pub fn wp_bitand4u(a: Vec4u, b: Vec4u) -> Vec4u {
        vec4u(a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w)
    }
    pub fn wp_shr2i(a: Vec2i, b: Vec2i) -> Vec2i {
        vec2i(a.x >> b.x as u32, a.y >> b.y as u32)
    }
    pub fn wp_previous(uv: Vec2f) -> Vec4f {
        texture_sample_level(WP_PREVIOUS, WP_SAMPLER, vec2f(uv.x, 1.0 - uv.y), 0.0)
    }
    pub fn wp_previous_texel(p: Vec2i) -> Vec4f {
        let height = get!(BLUR_LOCALS).wp_frame.y as i32;
        texture_load(WP_PREVIOUS, vec2i(p.x, height - 1 - p.y), 0i32)
    }
    pub fn wp_image_sample(uv: Vec2f) -> Vec4f {
        texture_sample(WP_IMAGE, WP_SAMPLER, uv)
    }
    pub fn wp_image_lod(uv: Vec2f, lod: f32) -> Vec4f {
        texture_sample_level(WP_IMAGE, WP_SAMPLER, uv, lod)
    }
    // ---------------------------------------------------------------------------------------
    // Butler product wallpapers, ported from packages/butler-app/client/ui/src/libs/design-system/
    // blocks/Wallpaper/modules (butler.bloom, butler.dusk, butler.shoreline, butler.riso-print
    // density.frag + print.frag) line for line. The engine contract maps as follows:
    //   gl_FragCoord.xy -> wp_frag_coord() (bottom-left origin, like GL)
    //   u_resolution, u_pixelRatio, u_time       -> wp_frame (xy, z, w)
    //   u_dark, u_dayPhase, u_seed, u_imageAspectRatio -> wp_env (x, y, z, w)
    //   u_contentRect                            -> wp_rect
    //   u_css, u_flow, u_lift                    -> wp_css (xy, z, w)
    //   u_pulse                                  -> wp_print.x
    //   u_drop, u_col, u_contentRect2            -> wp_drop, wp_col, wp_rect2
    //   u_hasImage                              -> wp_col.w
    //   image_fit / image_blur                   -> wp_drop / wp_col.xyz
    //   module params (p_*)                      -> wp_p0 .. wp_p5 in manifest order:
    //     each color / palette color starts a whole vec4 (xyz); numbers, enums (option index),
    //     booleans (0/1) fill components in order; after a color start the next vec4.
    //   host wraps u_time in seconds; u_contentRect2 -> wp_rect2
    //   u_noiseTexture / u_density / u_base / u_image -> noise / previous / previous / image
    //     (filter modules read the previous drawing-buffer-sized prefit as u_image).
    //   u_noiseTexture / u_density               -> BLUR_TEXTURE / WP_PREVIOUS, sampled by hand: GL's LINEAR +
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
        let a = texture_load(WP_PREVIOUS, vec2i(x0, y0), 0i32);
        let b = texture_load(WP_PREVIOUS, vec2i(x1, y0), 0i32);
        let c = texture_load(WP_PREVIOUS, vec2i(x0, y1), 0i32);
        let d = texture_load(WP_PREVIOUS, vec2i(x1, y1), 0i32);
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
        glint_k: f32,
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
        lc = lc + light.glint * glint_k + light.halation * ha;
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
        let paper0 = select(vec3f(250.0, 250.0, 248.0), vec3f(22.0, 23.0, 25.0), night) / 255.0;
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

    // --------------------------------- butler.riso-intro ------------------------------------

    pub fn wp_intro_vn(p: Vec2f) -> f32 {
        let i = floor(p);
        let f0 = fract(p);
        let f = f0 * f0 * (vec2f(3.0, 3.0) - 2.0 * f0);
        mix(
            mix(wp_hash(i), wp_hash(i + vec2f(1.0, 0.0)), f.x),
            mix(
                wp_hash(i + vec2f(0.0, 1.0)),
                wp_hash(i + vec2f(1.0, 1.0)),
                f.x,
            ),
            f.y,
        )
    }

    pub fn wp_intro_fbm(p0: Vec2f) -> f32 {
        let mut s = 0.0;
        let mut a = 0.5;
        let mut p = p0;
        for _i in 0..4 {
            s = s + a * wp_intro_vn(p);
            // mat2(1.6, 1.2, -1.2, 1.6) * p (GLSL column-major)
            p = vec2f(1.6 * p.x - 1.2 * p.y, 1.2 * p.x + 1.6 * p.y);
            a = a * 0.5;
        }
        s
    }

    pub fn wp_intro_ang(d: Vec2f) -> f32 {
        fract(atan2(d.x, -d.y) / WP_TAU + 1.0)
    }

    // ---------------------------------------------------------------------------------------
    // Intro density, ported line for line from Wallpaper/intro/introShaders.ts INTRO_DENSITY.
    // Distances are CSS px with a top-left origin, converted from GL's bottom-left fc:
    //   u_resolution, u_pixelRatio, u_time -> wp_frame (res.x, res.y, ratio, t seconds)
    //   u_css                             -> wp_css.xy
    //   uCard                             -> wp_p0 (x, y, w, h)
    //   uRing, uOrigin, uScale             -> wp_p1 (ring.x, ring.y, size, scale);
    //                                         uOrigin is also ring.xy
    //   uFlow, uBloom, uResidue, uCardOn   -> wp_p2
    //   uDraw, uDry, uSweep, uClear        -> wp_p3
    //   uMarkA, uMark                     -> wp_p4 (alpha, mark.x, mark.y, mark.z)
    //   uPtr                              -> wp_p5 (x, y, push x, push y)
    // ---------------------------------------------------------------------------------------
    pub fn wp_intro_density(fc: Vec2f) -> Vec4f {
        let u = get!(BLUR_LOCALS);
        let res = u.wp_frame.xy();
        let card = u.wp_p0;
        let ring = u.wp_p1.xyz();
        let scale = u.wp_p1.w;
        let flow = u.wp_p2.x;
        let bloom = u.wp_p2.y;
        let residue = u.wp_p2.z;
        let card_on = u.wp_p2.w;
        let draw = u.wp_p3.x;
        let dry = u.wp_p3.y;
        let sweep = u.wp_p3.z;
        let clear = u.wp_p3.w;
        let mark_a = u.wp_p4.x;
        let mark = vec3f(u.wp_p4.y, u.wp_p4.z, u.wp_p4.w);
        let pointer = u.wp_p5;

        let p = vec2f(fc.x, res.y - fc.y) / res * u.wp_css.xy();
        let pd = p - pointer.xy();
        let fall = exp(-dot(pd, pd) / 12100.0);
        let sp = p - pointer.zw() * fall * 34.0;
        // The pen drags the field clockwise and pulls it onto the passed stroke.
        let rp = ring.z * (435.0 / 1200.0);
        let rd = sp - ring.xy();
        let rl = length(rd) + 1e-3;
        let ra = wp_intro_ang(rd);
        let seam = mix(smoothstep(0.0, 0.07, ra), 1.0, dry);
        let passed = smoothstep(ra - 0.02, ra + 0.1, sweep) * seam;
        let dl = (rl - rp) / rp;
        let g = passed * exp(-dl * dl / 1.6) * 1.7 * (1.0 - 0.55 * residue);
        let sw = -0.85 * draw * exp(-dl * dl / 3.2);
        let cs = cos(sw);
        let sn = sin(sw);
        let dir = rd / rl;
        // mat2(cs, sn, -sn, cs) * (rd / rl) (GLSL column-major)
        let rdir = vec2f(cs * dir.x - sn * dir.y, sn * dir.x + cs * dir.y);
        let th = draw * WP_TAU;
        let tip = ring.xy() + vec2f(sin(th), -cos(th)) * rp;
        let pen = smoothstep(0.0, 0.04, draw) * (1.0 - smoothstep(0.94, 1.0, draw));
        let tv = sp - tip;
        let tf = exp(-dot(tv, tv) / (rp * rp * 1.6));
        let gp = ring.xy() + rdir * (rp + (rl - rp) * (1.0 + g)) + tv * tf * pen * 1.2;
        let q = (gp - ring.xy()) / scale;
        let grow = 0.16 + 1.05 * bloom;
        let qa = q / grow;
        let ft = flow;
        let w = vec2f(
            wp_intro_fbm(qa * 1.4 + vec2f(3.1, ft * 0.45)),
            wp_intro_fbm(qa * 1.4 + vec2f(5.2 - ft * 0.35, 1.3)),
        );
        let rr = length(qa) + (w.x - 0.5) * 0.6;
        let env = 1.0 - smoothstep(0.5, 1.05, rr);
        let base = qa * 2.1 + (w - 0.5) * 2.6;
        let n = vec3f(
            wp_intro_fbm(base + vec2f(1.7, 9.2) + ft * vec2f(0.18, -0.1)),
            wp_intro_fbm(base * 1.08 + vec2f(8.3, 2.8) + ft * vec2f(-0.12, 0.16)),
            wp_intro_fbm(base * 0.94 + vec2f(4.1, 6.6) + ft * vec2f(0.05, 0.14)),
        );
        let mut d = smoothstep(vec3f(0.41, 0.43, 0.44), vec3f(0.86, 0.86, 0.86), n) * env;
        d = d * vec3f(1.0, 0.92, 0.85);
        // Dries to a residue that keeps to the edges, clear of the card.
        let cc = card.xy() + card.zw() * 0.5;
        let bq = abs(p - cc) - card.zw() * 0.5;
        let sd_c = length(max(bq, vec2f(0.0, 0.0))) + min(max(bq.x, bq.y), 0.0);
        let edge = smoothstep(60.0, 360.0, sd_c + (w.y - 0.5) * 120.0);
        d = d * mix(1.0, 0.62 * edge, residue);
        d = d * mix(1.0, smoothstep(16.0, 120.0, sd_c), card_on);
        // Ink is used up along the written stroke and inside the ring, traveling with the mark.
        let md = p - mark.xy();
        let ml = length(md);
        let rm = mark.z * (435.0 / 1200.0);
        let wm = mark.z * (74.0 / 1200.0);
        let ma = wp_intro_ang(md);
        let used =
            smoothstep(ma + 0.03, ma + 0.16, sweep) * mix(smoothstep(0.0, 0.07, ma), 1.0, dry);
        let halo = 1.0 - smoothstep(wm * 0.5, wm * 0.5 + rm * 0.6, abs(ml - rm));
        let inner = (1.0 - smoothstep(rm * 0.78, rm * 1.02, ml)) * clear;
        d = d * (1.0 - 0.95 * max(used * halo, inner) * mark_a);
        let dc = clamp(d, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0));
        vec4f(dc.x, dc.y, dc.z, 1.0)
    }

    // butler.silk/shader.frag
    pub fn wp_silk_silk_noise(p: Vec2f) -> f32 {
        let q: Vec2f = 2.71828 * sin(2.71828 * p);
        return fract(q.x * q.y * (1.0 + p.x));
    }

    pub fn wp_silk_rotate_silk(p: Vec2f, a: f32) -> Vec2f {
        let c: f32 = cos(a);
        let s: f32 = sin(a);
        return mat2x2f(vec2f(c, -s), vec2f(s, c)) * p;
    }

    pub fn wp_silk(fc: Vec2f) -> Vec4f {
        let mut result = vec4f(0.0, 0.0, 0.0, 0.0);
        let t: f32 = get!(BLUR_LOCALS).wp_frame.w;
        let rnd: f32 = wp_silk_silk_noise(fc);
        let uv: Vec2f =
            (fc / get!(BLUR_LOCALS).wp_frame.xy() - vec2f(0.5, 0.5)) * 0.9 + vec2f(0.5, 0.5);
        let mut tex: Vec2f = wp_silk_rotate_silk(uv, 2.7708) * 1.06;
        let phase: f32 = t * 0.64;
        tex.y = tex.y + 0.03 * sin(3.0 * tex.x - phase);
        tex.x = tex.x + 0.015 * sin(2.0 * tex.y + phase * 0.5);
        let fold: f32 = 0.94
            + 0.06
                * sin(
                    2.5 * (tex.x + tex.y + cos(1.5 * tex.x + 2.5 * tex.y) + 0.02 * phase)
                        + sin(10.0 * (tex.x + tex.y - 0.1 * phase)),
                );
        let drift: f32 = 0.92 + 0.08 * sin(1.2 * (tex.x - tex.y) + phase * 0.18);
        let highlight: f32 = smoothstep(0.78, 1.0, fold);
        let shade: Vec3f = mix(
            vec3f(0.72, 0.72, 0.74),
            vec3f(0.34, 0.35, 0.38),
            vec3f(
                get!(BLUR_LOCALS).wp_env.x,
                get!(BLUR_LOCALS).wp_env.x,
                get!(BLUR_LOCALS).wp_env.x,
            ),
        );
        let shine: Vec3f = mix(
            vec3f(1.0, 1.0, 1.0),
            vec3f(0.56, 0.58, 0.62),
            vec3f(
                get!(BLUR_LOCALS).wp_env.x,
                get!(BLUR_LOCALS).wp_env.x,
                get!(BLUR_LOCALS).wp_env.x,
            ),
        );
        let shadow: f32 = 1.0 - smoothstep(0.88, 0.99, fold * drift);
        let mut color: Vec3f = mix(
            get!(BLUR_LOCALS).wp_p0.xyz(),
            shade,
            vec3f(shadow * 0.42, shadow * 0.42, shadow * 0.42),
        );
        color = mix(
            color,
            shine,
            vec3f(highlight * 0.18, highlight * 0.18, highlight * 0.18),
        );
        color = mix(
            color,
            get!(BLUR_LOCALS).wp_p0.xyz(),
            vec3f(0.14, 0.14, 0.14),
        );
        color = color - vec3f(rnd * 0.006, rnd * 0.006, rnd * 0.006);
        result = vec4f(color.x, color.y, color.z, 1.0);
        result
    }

    // butler.riso-flow/shader.frag
    pub const WP_RISO_FLOW_TAU: f32 = 6.2831853;

    pub fn wp_riso_flow_nz(x: Vec2f) -> Vec4f {
        let i: Vec2f = floor(x);
        let mut f: Vec2f = fract(x);
        f = f * f * (vec2f(3.0, 3.0) - 2.0 * f);
        return wp_noise((i + f + vec2f(0.5, 0.5)) / 256.0);
    }

    pub fn wp_riso_flow_nr(x: Vec2f) -> Vec4f {
        return wp_noise(x / 256.0);
    }

    pub fn wp_riso_flow_atan_up(y: f32, x: f32) -> f32 {
        let ax: f32 = abs(x);
        let a: f32 = min(ax, y) / max(max(ax, y), 1e-6);
        let s: f32 = a * a;
        let mut t: f32 = ((-0.0464965 * s + 0.1593142) * s - 0.3276228) * s * a + a;
        t = select(t, 1.5707963 - t, y > ax);
        return select(t, 3.1415927 - t, x < 0.0);
    }

    pub fn wp_riso_flow_wav(x: Vec3f) -> Vec3f {
        let t: Vec3f = fract(x * 0.1591549) * 2.0 - vec3f(1.0, 1.0, 1.0);
        return -4.0 * t * (vec3f(1.0, 1.0, 1.0) - abs(t));
    }

    pub fn wp_riso_flow_screen_dot(q: Vec2f, d: f32, rag: f32, aa: f32, jit: Vec2f) -> f32 {
        let g: Vec2f = fract(q) - vec2f(0.5, 0.5) + jit;
        let r: f32 = sqrt(d * 0.3183) + 0.17 * smoothstep(0.5, 1.0, d) + 0.02;
        return (1.0 - smoothstep(r - aa, r + aa, length(g) + rag)) * smoothstep(0.01, 0.05, d);
    }

    pub fn wp_riso_flow_ink_set(
        k: i32,
        a: ptr!(function, Vec3f),
        b: ptr!(function, Vec3f),
        c: ptr!(function, Vec3f),
    ) {
        if k == 1 {
            *a = vec3f(1.0, 0.424, 0.184);
            *b = vec3f(0.239, 0.333, 0.533);
            *c = vec3f(1.0, 0.710, 0.067);
        } else {
            if k == 2 {
                *a = vec3f(1.0, 0.282, 0.690);
                *b = vec3f(0.0, 0.514, 0.541);
                *c = vec3f(1.0, 0.710, 0.067);
            } else {
                if k == 3 {
                    *a = vec3f(1.0, 0.282, 0.690);
                    *b = vec3f(0.369, 0.784, 0.898);
                    *c = vec3f(1.0, 0.910, 0.0);
                } else {
                    *a = vec3f(1.0, 0.282, 0.690);
                    *b = vec3f(0.196, 0.333, 0.643);
                    *c = vec3f(1.0, 0.910, 0.0);
                }
            }
        }
    }

    pub fn wp_riso_flow(fc: Vec2f) -> Vec4f {
        let mut result = vec4f(0.0, 0.0, 0.0, 0.0);
        let res: Vec2f = get!(BLUR_LOCALS).wp_frame.xy();
        let fc: Vec2f = fc;
        let pr: f32 = max(get!(BLUR_LOCALS).wp_frame.z, 0.25);
        let portrait: bool = res.x < res.y;
        let p: Vec2f = (fc - 0.5 * res) / res.y;
        let ph: f32 = WP_RISO_FLOW_TAU * get!(BLUR_LOCALS).wp_frame.w / 240.0
            * floor(get!(BLUR_LOCALS).wp_p0.z + 0.5);
        let cyc: Vec2f = vec2f(cos(ph), sin(ph));
        let fx_capital: Vec3f = select(
            vec3f(-0.4, -0.12, -0.37 * res.x / res.y),
            vec3f(-0.07, 0.06, -0.16),
            portrait,
        );
        let fy_capital: Vec3f = select(
            vec3f(-0.65, -0.66, -0.64),
            vec3f(-0.63, -0.64, -0.62),
            portrait,
        );
        let w: Vec4f = wp_riso_flow_nz(
            p * 1.05
                + cyc * 0.7
                + vec2f(17.0, 3.0)
                + vec2f(
                    get!(BLUR_LOCALS).wp_env.z * 40.0,
                    get!(BLUR_LOCALS).wp_env.z * 40.0,
                ),
        );
        let dx: Vec3f = vec3f(p.x, p.x, p.x) - fx_capital;
        let dy: Vec3f = vec3f(p.y, p.y, p.y) - fy_capital;
        let r: Vec3f = sqrt(dx * dx + dy * dy);
        let th: Vec3f = vec3f(
            wp_riso_flow_atan_up(dy.x, dx.x),
            wp_riso_flow_atan_up(dy.y, dx.y),
            wp_riso_flow_atan_up(dy.z, dx.z),
        ) + vec3f((w.x - 0.5) * 0.2, (w.x - 0.5) * 0.2, (w.x - 0.5) * 0.2);
        let crest: Vec3f = vec3f(0.98, 0.76, 1.1)
            + vec3f(
                select(0.0, 0.2, portrait),
                select(0.0, 0.2, portrait),
                select(0.0, 0.2, portrait),
            )
            - 0.26 * r
            + 0.1
                * wp_riso_flow_wav(
                    vec3f(4.2, 3.6, 5.0) * r - vec3f(5.0, 4.0, 5.0) * ph + vec3f(0.0, 2.1, 4.2),
                )
            + 0.05 * wp_riso_flow_wav(vec3f(1.0, -1.0, 1.0) * ph + vec3f(1.3, 0.2, 3.9));
        let w_upper: Vec3f = vec3f(0.5, 0.52, 0.56);
        let mis: Vec3f = (dx * vec3f(-0.35, 0.7, 0.95) - dy * vec3f(0.8, -0.6, 0.15))
            * (vec3f(
                get!(BLUR_LOCALS).wp_p0.w * pr,
                get!(BLUR_LOCALS).wp_p0.w * pr,
                get!(BLUR_LOCALS).wp_p0.w * pr,
            ) / (res.y * r * r));
        let v: Vec3f = (crest - th + mis) / w_upper;
        let c: Vec3f = clamp(
            vec3f(1.0, 1.0, 1.0) - v,
            vec3f(0.0, 0.0, 0.0),
            vec3f(1.0, 1.0, 1.0),
        );
        let mut d: Vec3f =
            smoothstep(vec3f(-0.03, -0.03, -0.03), vec3f(0.08, 0.08, 0.08), v) * c * sqrt(c);
        let pl: Vec3f = fract(v * 3.0 + vec3f(0.15, 0.15, 0.15));
        d = d * (vec3f(0.8, 0.8, 0.8) + 0.8 * pl * (vec3f(1.0, 1.0, 1.0) - pl));
        d = d * (vec3f(0.72, 0.72, 0.72) + 0.36 * vec3f(w.z, w.w, 1.0 - w.z));
        d = d
            * (vec3f(1.0, 0.9, 1.0)
                * (vec3f(1.0, 1.0, 1.0)
                    - smoothstep(vec3f(0.6, 0.6, 0.6), vec3f(1.95, 2.05, 1.5), r)));
        let mut cr: Vec4f = get!(BLUR_LOCALS).wp_rect;
        if cr.z < 1.0 {
            cr = vec4f(res.x * 0.22, res.y * 0.45, res.x * 0.56, res.y * 0.47);
        }
        let hh: f32 = min(cr.w * 0.42, 190.0 * pr);
        let hq: Vec2f = abs(fc - vec2f(cr.x + cr.z * 0.5, cr.y + cr.w - hh * 0.5))
            - vec2f(cr.z * 0.5, hh * 0.5)
            + vec2f(24.0 * pr, 24.0 * pr);
        let sd_h: f32 = max(hq.x, hq.y) - 24.0 * pr;
        d = clamp(
            d * mix(0.05, 1.0, smoothstep(-10.0 * pr, 160.0 * pr, sd_h)),
            vec3f(0.0, 0.0, 0.0),
            vec3f(1.0, 1.0, 1.0),
        );
        let fine: Vec4f = wp_riso_flow_nr(fc * 0.55 / pr);
        let fib: f32 = smoothstep(
            0.84,
            0.98,
            wp_riso_flow_nr(
                mat2x2f(vec2f(0.825, 0.565), vec2f(-0.565, 0.825)) * fc * vec2f(0.7, 0.12) / pr,
            )
            .x,
        );
        let tooth: f32 = fine.w - 0.5;
        let paper: Vec3f = select(
            vec3f(0.098, 0.110, 0.141) * (1.0 + 0.12 * tooth)
                + vec3f(0.007 * fib, 0.007 * fib, 0.007 * fib),
            vec3f(0.953, 0.937, 0.902) * (1.0 + 0.03 * tooth)
                + vec3f(0.015 * fib, 0.015 * fib, 0.015 * fib),
            get!(BLUR_LOCALS).wp_env.x < 0.5,
        );
        let mut col: Vec3f = paper;
        if max(d.x, max(d.y, d.z)) > 0.008 {
            let cell: f32 = get!(BLUR_LOCALS).wp_p0.y * pr;
            let q: Vec2f = fc / cell;
            let blot: Vec4f = wp_riso_flow_nr(q * 1.43 + vec2f(11.0, 11.0));
            let mid: Vec4f = wp_riso_flow_nr(fc / (pr * 26.0) + vec2f(101.0, 101.0));
            let aa: f32 = 1.1 / cell;
            let rag: Vec3f = (fine.xyz() - vec3f(0.5, 0.5, 0.5)) * 0.13
                + (blot.xyz() - vec3f(0.5, 0.5, 0.5)) * 0.15;
            let jit: Vec2f = (vec2f(blot.w, mid.w) - vec2f(0.5, 0.5)) * vec2f(0.14, 0.1);
            let mut cov: Vec3f = vec3f(0.0, 0.0, 0.0);
            if d.x > 0.008 {
                cov.x = wp_riso_flow_screen_dot(
                    mat2x2f(vec2f(0.2588, 0.9659), vec2f(-0.9659, 0.2588)) * q,
                    d.x,
                    rag.x,
                    aa,
                    jit,
                );
            }
            if d.y > 0.008 {
                cov.y = wp_riso_flow_screen_dot(
                    mat2x2f(vec2f(0.9659, 0.2588), vec2f(-0.2588, 0.9659)) * q,
                    d.y,
                    rag.y,
                    aa,
                    jit.yx(),
                );
            }
            if d.z > 0.008 {
                cov.z = wp_riso_flow_screen_dot(
                    mat2x2f(vec2f(0.7071, 0.7071), vec2f(-0.7071, 0.7071)) * q,
                    d.z,
                    rag.z,
                    aa,
                    -jit,
                );
            }
            cov = cov * smoothstep(0.05, 0.16, fine.w + 0.1 * mid.w);
            let film: Vec3f = (vec3f(0.78, 0.78, 0.78) + 0.22 * mid.xyz()) * (0.86 + 0.14 * fine.w);
            let mut a_upper: Vec3f = vec3f(0.0, 0.0, 0.0);
            let mut b_upper: Vec3f = vec3f(0.0, 0.0, 0.0);
            let mut c_upper: Vec3f = vec3f(0.0, 0.0, 0.0);
            wp_riso_flow_ink_set(
                get!(BLUR_LOCALS).wp_p0.x as i32,
                &mut a_upper,
                &mut b_upper,
                &mut c_upper,
            );
            if get!(BLUR_LOCALS).wp_env.x < 0.5 {
                let k: Vec3f = cov * film * 0.95;
                col = paper
                    * mix(vec3f(1.0, 1.0, 1.0), a_upper, vec3f(k.x, k.x, k.x))
                    * mix(vec3f(1.0, 1.0, 1.0), b_upper, vec3f(k.y, k.y, k.y))
                    * mix(vec3f(1.0, 1.0, 1.0), c_upper, vec3f(k.z, k.z, k.z));
            } else {
                let m_upper: Vec3f = mix(vec3f(1.0, 1.0, 1.0), a_upper, vec3f(cov.x, cov.x, cov.x))
                    * mix(vec3f(1.0, 1.0, 1.0), b_upper, vec3f(cov.y, cov.y, cov.y))
                    * mix(vec3f(1.0, 1.0, 1.0), c_upper, vec3f(cov.z, cov.z, cov.z));
                let u_upper: f32 = 1.0 - (1.0 - cov.x) * (1.0 - cov.y) * (1.0 - cov.z);
                let f: f32 = dot(cov, film) / max(cov.x + cov.y + cov.z, 1e-3);
                col = mix(
                    paper,
                    m_upper * (0.5 + 0.4 * f),
                    vec3f(u_upper * 0.94, u_upper * 0.94, u_upper * 0.94),
                );
            }
        }
        result = vec4f(col.x, col.y, col.z, 1.0);
        result
    }

    // butler.lamina/shader.frag
    #[derive(Clone, Copy, Wgsl)]
    pub struct WpLaminaLeaf {
        pub tip: Vec2f,
        pub dir: Vec2f,
        pub w_upper: f32,
        pub taper: f32,
        pub costa: f32,
        pub z: f32,
        pub seed: u32,
        pub tilt: f32,
        pub band: f32,
    }

    #[derive(Clone, Copy, Wgsl)]
    pub struct WpLaminaLout {
        pub t_upper: Vec3f,
        pub wall: f32,
        pub chl: f32,
        pub mean: f32,
        pub glow: f32,
        pub fringe: f32,
    }

    #[derive(Clone, Copy, Wgsl)]
    pub struct WpLaminaContext {
        pub g_k: f32,
        pub g_breath: f32,
        pub g_rack_sign: f32,
        pub g_phase: f32,
        pub g_focus: f32,
        pub g_bmax: f32,
        pub g_head_c: Vec2f,
        pub g_head_b: Vec2f,
        pub g_rect_c: Vec2f,
        pub g_rect_b: Vec2f,
        pub g_c_as: Vec2f,
    }

    pub const WP_LAMINA_TAU: f32 = 6.28318530718;

    pub const WP_LAMINA_PERIOD: f32 = 300.0;

    pub const WP_LAMINA_FIELD: Vec3f = vec3f(0.855, 0.8469, 0.8963);

    pub const WP_LAMINA_T_CHL_A: Vec3f = vec3f(0.357, 0.4325, 0.0504);

    pub const WP_LAMINA_T_CHL_B: Vec3f = vec3f(0.2526, 0.3343, 0.0089);

    pub const WP_LAMINA_T_WALL: Vec3f = vec3f(1.058, 1.088, 0.866);

    pub const WP_LAMINA_T_COSTA: Vec3f = vec3f(0.1195, 0.1736, 0.0153);

    pub const WP_LAMINA_T_TIP: Vec3f = vec3f(0.8114, 0.8454, 0.6811);

    pub const WP_LAMINA_T_GAP: Vec3f = vec3f(0.74, 0.80, 0.42);

    pub const WP_LAMINA_CL: f32 = 20.0;

    pub const WP_LAMINA_CW: f32 = 13.5;

    pub const WP_LAMINA_SHEAR: f32 = 0.36;

    pub const WP_LAMINA_RC: f32 = 2.5;

    pub const WP_LAMINA_HT: f32 = 0.62;

    pub const WP_LAMINA_CA_K: f32 = 0.0021;

    pub const WP_LAMINA_B_LO: f32 = 4.4;

    pub const WP_LAMINA_B_HI: f32 = 5.7;

    pub fn wp_lamina_lb32(x: u32) -> u32 {
        let mut x = x;
        x = x ^ (x >> 16u32);
        x = x * 0x7feb352du32;
        x = x ^ (x >> 15u32);
        x = x * 0x846ca68bu32;
        x = x ^ (x >> 16u32);
        return x;
    }

    pub fn wp_lamina_hcell(c: Vec2i, s: u32) -> u32 {
        return wp_lamina_lb32(((c.x as u32 * 0x9E3779B1u32) ^ (c.y as u32 * 0x85EBCA77u32)) ^ s);
    }

    pub fn wp_lamina_u4(h: u32) -> Vec4f {
        return vec4f(
            (wp_bitand4u(
                vec4u(h, h >> 8u32, h >> 16u32, h >> 24u32),
                vec4u(255u32, 255u32, 255u32, 255u32),
            ))
            .x as f32,
            (wp_bitand4u(
                vec4u(h, h >> 8u32, h >> 16u32, h >> 24u32),
                vec4u(255u32, 255u32, 255u32, 255u32),
            ))
            .y as f32,
            (wp_bitand4u(
                vec4u(h, h >> 8u32, h >> 16u32, h >> 24u32),
                vec4u(255u32, 255u32, 255u32, 255u32),
            ))
            .z as f32,
            (wp_bitand4u(
                vec4u(h, h >> 8u32, h >> 16u32, h >> 24u32),
                vec4u(255u32, 255u32, 255u32, 255u32),
            ))
            .w as f32,
        ) * (1.0 / 255.0);
    }

    pub fn wp_lamina_nz(x: Vec2f) -> f32 {
        let i: Vec2f = floor(x);
        let mut f: Vec2f = fract(x);
        f = f * f * (vec2f(3.0, 3.0) - 2.0 * f);
        return wp_noise((i + f + vec2f(0.5, 0.5)) / 256.0).x;
    }

    pub fn wp_lamina_nz2(x: Vec2f) -> Vec2f {
        let i: Vec2f = floor(x);
        let mut f: Vec2f = fract(x);
        f = f * f * (vec2f(3.0, 3.0) - 2.0 * f);
        return wp_noise((i + f + vec2f(0.5, 0.5)) / 256.0).yz();
    }

    pub fn wp_lamina_sq(x: f32) -> f32 {
        return x * x;
    }

    pub fn wp_lamina_sq4(x: Vec4f) -> Vec4f {
        return x * x;
    }

    pub fn wp_lamina_sd_r_box(p: Vec2f, b: Vec2f, r: f32) -> f32 {
        let d: Vec2f = abs(p) - b + vec2f(r, r);
        return length(max(d, vec2f(0.0, 0.0))) + min(max(d.x, d.y), 0.0) - r;
    }

    pub fn wp_lamina_smin(a: f32, b: f32, k: f32) -> f32 {
        let h: f32 = max(k - abs(a - b), 0.0) / k;
        return min(a, b) - h * h * k * 0.25;
    }

    pub fn wp_lamina_to_srgb(c: Vec3f) -> Vec3f {
        let mut c = c;
        c = max(c, vec3f(0.0, 0.0, 0.0));
        return mix(
            c * 12.92,
            1.055 * pow(c, vec3f(1.0 / 2.4, 1.0 / 2.4, 1.0 / 2.4)) - vec3f(0.055, 0.055, 0.055),
            step(vec3f(0.0031308, 0.0031308, 0.0031308), c),
        );
    }

    pub fn wp_lamina_half_w(l_upper: WpLaminaLeaf, a: f32) -> f32 {
        let x: f32 = clamp(a / l_upper.taper, 0.0, 1.0);
        return l_upper.w_upper
            * x
            * (2.0 - x)
            * (1.0 + 0.03 * sin(a * 0.013 + (l_upper.seed & 63u32) as f32))
            + max(a - l_upper.taper, 0.0) * 0.05;
    }

    pub fn wp_lamina_squeeze(l_upper: WpLaminaLeaf, hw: f32) -> f32 {
        return l_upper.w_upper / max(hw, 0.65 * l_upper.w_upper);
    }

    pub fn wp_lamina_costa_w(l_upper: WpLaminaLeaf, a: f32) -> f32 {
        return l_upper.costa * smoothstep(0.06 * l_upper.taper, 0.7 * l_upper.taper, a);
    }

    pub fn wp_lamina_sab(x: f32) -> f32 {
        return sqrt(x * x + 64.0);
    }

    pub fn wp_lamina_cell_pt(id: Vec2i, h: u32) -> Vec2f {
        let r: Vec4f = wp_lamina_u4(h);
        return vec2f(
            (id.x as f32 + (r.x - 0.5) * 0.36) * WP_LAMINA_CL,
            (id.y as f32 + 0.5 * (id.x & 1) as f32 + (r.y - 0.5) * 0.3) * WP_LAMINA_CW,
        );
    }

    pub fn wp_lamina_dens_screen(ctx: ptr!(function, WpLaminaContext), q: Vec2f) -> f32 {
        let h: Vec2f = abs(q - (*ctx).g_head_c) - (*ctx).g_head_b;
        let r: Vec2f = abs(q - (*ctx).g_rect_c) - (*ctx).g_rect_b;
        let d_h: f32 = max(h.x, h.y);
        let d_r: f32 = max(r.x, r.y);
        return smoothstep(0.0, 170.0, d_h) * mix(0.5, 1.0, smoothstep(-30.0, 90.0, d_r));
    }

    pub fn wp_lamina_line_cov(d: f32, ht: f32, b: f32, aa: f32) -> f32 {
        let e: f32 = aa + b;
        return smoothstep(e, -e, d - 0.5 * b) * (ht + aa) / (ht + aa + b);
    }

    pub fn wp_lamina_line_cov3(d: Vec3f, ht: f32, b: f32, aa: f32) -> Vec3f {
        let e: f32 = aa + b;
        return smoothstep(
            vec3f(e, e, e),
            vec3f(-e, -e, -e),
            d - vec3f(0.5 * b, 0.5 * b, 0.5 * b),
        ) * (ht + aa)
            / (ht + aa + b);
    }

    pub fn wp_lamina_disc_cov(
        ctx: ptr!(function, WpLaminaContext),
        sd: f32,
        rad: f32,
        b: f32,
        aa: f32,
        rim: ptr!(function, f32),
    ) -> f32 {
        let e: f32 = aa + 0.3 * b;
        let body: f32 = smoothstep(e, -e, sd - b);
        let r_upper: f32 = rad + b;
        *rim = smoothstep(e + 0.7, 0.0, abs(sd - b + 0.7));
        let bokeh: f32 = 1.0 + (*ctx).g_rack_sign * *rim * smoothstep(0.4, 2.0, b);
        return body * (rad * rad) / (r_upper * r_upper) * bokeh;
    }

    pub fn wp_lamina_cell_rnd(id: Vec2i, salt: i32) -> Vec4f {
        return texture_load(
            BLUR_TEXTURE,
            wp_bitand2i(id + vec2i(salt, salt * 7), vec2i(255, 255)),
            0i32,
        );
    }

    pub fn wp_lamina_cell_at(id: Vec2i, r: Vec4f) -> Vec2f {
        return vec2f(
            (id.x as f32 + (r.x - 0.5) * 0.36) * WP_LAMINA_CL,
            (id.y as f32 + 0.5 * (id.x & 1) as f32 + (r.y - 0.5) * 0.3) * WP_LAMINA_CW,
        );
    }

    pub fn wp_lamina_fast_atan(y: f32, x: f32) -> f32 {
        let ax: f32 = abs(x);
        let ay: f32 = abs(y);
        let m: f32 = min(ax, ay) / max(max(ax, ay), 1e-9);
        let s2: f32 = m * m;
        let mut r: f32 = ((-0.0464964749 * s2 + 0.15931422) * s2 - 0.327622764) * s2 * m + m;
        r = select(r, 1.57079637 - r, ay > ax);
        r = select(r, 3.14159274 - r, x < 0.0);
        return select(r, -r, y < 0.0);
    }

    pub fn wp_lamina_mix_l(
        ctx: ptr!(function, WpLaminaContext),
        a: WpLaminaLout,
        s: WpLaminaLout,
        b: f32,
    ) -> WpLaminaLout {
        let k: f32 = smoothstep(WP_LAMINA_B_LO, WP_LAMINA_B_HI, b * (*ctx).g_k);
        return WpLaminaLout {
            t_upper: mix(a.t_upper, s.t_upper, vec3f(k, k, k)),
            wall: mix(a.wall, s.wall, k),
            chl: mix(a.chl, s.chl, k),
            mean: mix(a.mean, s.mean, k),
            glow: mix(a.glow, s.glow, k),
            fringe: mix(a.fringe, s.fringe, k),
        };
    }

    pub fn wp_lamina_layer_soft(
        ctx: ptr!(function, WpLaminaContext),
        l_upper: WpLaminaLeaf,
        p: Vec2f,
        ac: Vec2f,
        hw: f32,
        b: f32,
        e: f32,
        aa: f32,
        g_m: Vec2f,
        w_band: f32,
    ) -> WpLaminaLout {
        let mut o: WpLaminaLout = WpLaminaLout {
            t_upper: vec3f(1.0, 1.0, 1.0),
            wall: 0.0,
            chl: 0.0,
            mean: 0.0,
            glow: 1.0,
            fringe: 0.0,
        };
        let dm: f32 = min(hw - abs(ac.y), ac.x);
        let dens: f32 = wp_lamina_dens_screen(ctx, p);
        let s: f32 = wp_lamina_squeeze(l_upper, hw);
        let x_upper: Vec2f = vec2f(ac.x + WP_LAMINA_SHEAR * wp_lamina_sab(ac.y * s), ac.y * s);
        let mott: f32 = wp_lamina_nz(
            x_upper / vec2f(WP_LAMINA_CL, WP_LAMINA_CW)
                + vec2f(
                    (l_upper.seed & 63u32) as f32 * 3.1,
                    (l_upper.seed & 63u32) as f32 * 3.1,
                ),
        );
        let fill: f32 =
            0.42 * get!(BLUR_LOCALS).wp_p1.y * mix(0.35, 1.0, dens) * (0.7 + 0.6 * mott);
        let mut tt_capital: Vec3f = mix(
            WP_LAMINA_T_GAP,
            mix(WP_LAMINA_T_CHL_A, WP_LAMINA_T_CHL_B, vec3f(0.4, 0.4, 0.4)),
            vec3f(fill, fill, fill),
        );
        tt_capital = mix(
            tt_capital,
            pow(WP_LAMINA_T_COSTA, vec3f(0.6, 0.6, 0.6)),
            vec3f(w_band * 0.8, w_band * 0.8, w_band * 0.8),
        );
        o.t_upper = tt_capital;
        o.chl = fill * 0.75 + 0.1 * w_band;
        o.mean = mix(fill * 0.8, 0.5, w_band);
        if dm < e + 2.0 * b + 3.0 {
            let ca_m: f32 = dot((*ctx).g_c_as, g_m) / (*ctx).g_k;
            let ci: Vec3f = smoothstep(
                vec3f(-e, -e, -e),
                vec3f(e, e, e),
                vec3f(dm, dm, dm) + vec3f(ca_m, 0.0, -ca_m),
            );
            let wl: f32 = 0.6 * wp_lamina_line_cov(abs(dm - 1.25) - 1.25, 1.25, b, aa);
            o.t_upper = mix(
                mix(vec3f(1.0, 1.0, 1.0), tt_capital, ci),
                WP_LAMINA_T_WALL,
                vec3f(
                    wl * (1.0 - ci.y * 0.5),
                    wl * (1.0 - ci.y * 0.5),
                    wl * (1.0 - ci.y * 0.5),
                ),
            );
            o.wall = wl;
            o.chl = o.chl * ci.y;
            o.mean = o.mean * ci.y;
            o.glow = ci.y;
            o.fringe = exp(-wp_lamina_sq((-dm - 0.9 - 0.5 * b) / (0.9 + b))) * 0.9 / (0.9 + b);
        }
        return o;
    }

    pub fn wp_lamina_layer(
        ctx: ptr!(function, WpLaminaContext),
        l_upper: WpLaminaLeaf,
        p: Vec2f,
        ac: Vec2f,
        hw: f32,
        blur_s: f32,
        aa: f32,
        t: f32,
        _dark: bool,
    ) -> WpLaminaLout {
        let mut o: WpLaminaLout = WpLaminaLout {
            t_upper: vec3f(1.0, 1.0, 1.0),
            wall: 0.0,
            chl: 0.0,
            mean: 0.0,
            glow: 0.0,
            fringe: 0.0,
        };
        let nrm: Vec2f = vec2f(-l_upper.dir.y, l_upper.dir.x);
        let _out0: f32 = abs(ac.y) - hw;
        let salt: i32 = (l_upper.seed & 255u32) as i32;
        let mut zl: f32 = l_upper.z + l_upper.tilt * (hw - abs(ac.y) - l_upper.band);
        let g_m: Vec2f = -sign(ac.y) * nrm;
        let c_band: f32 = wp_lamina_costa_w(l_upper, max(ac.x, 0.0));
        if abs((*ctx).g_focus - zl) < 1.1 {
            zl = zl
                + (0.24
                    * (wp_lamina_nz(
                        ac / 280.0
                            + vec2f(
                                (l_upper.seed & 511u32) as f32 * 0.71,
                                (l_upper.seed & 511u32) as f32 * 0.71,
                            ),
                    ) - 0.5)
                    + 0.06 * smoothstep(0.6, 1.0, abs(ac.y) / max(hw, 1.0)));
        }
        let dz: f32 = abs((*ctx).g_focus - zl);
        let b: f32 = (blur_s + (*ctx).g_bmax * dz / (1.0 + 1.2 * dz)) / (*ctx).g_k;
        let e: f32 = aa + b;
        let w_band: f32 = smoothstep(e, -e, abs(ac.y) - c_band);
        if b * (*ctx).g_k >= WP_LAMINA_B_HI {
            return wp_lamina_layer_soft(ctx, l_upper, p, ac, hw, b, e, aa, g_m, w_band);
        }
        let ca_m: f32 = dot((*ctx).g_c_as, g_m) / (*ctx).g_k;
        let s: f32 = wp_lamina_squeeze(l_upper, hw);
        let mut x_upper: Vec2f = vec2f(0.0, ac.y * s);
        x_upper.x = ac.x + WP_LAMINA_SHEAR * wp_lamina_sab(x_upper.y);
        x_upper = x_upper
            + (wp_lamina_nz2(
                x_upper * vec2f(0.5 / WP_LAMINA_CL, 0.5 / WP_LAMINA_CW)
                    + vec2f(
                        (l_upper.seed & 255u32) as f32,
                        (l_upper.seed & 255u32) as f32,
                    ),
            ) - vec2f(0.5, 0.5))
                * vec2f(WP_LAMINA_CL, WP_LAMINA_CW)
                * 0.3;
        let g: Vec2f = x_upper / vec2f(WP_LAMINA_CL, WP_LAMINA_CW);
        let ic: i32 = floor(g.x + 0.5) as i32;
        let sx: i32 = select(-1, 1, g.x >= ic as f32);
        let jc: i32 = floor(g.y - 0.5 * (ic & 1) as f32 + 0.5) as i32;
        let mut d1: f32 = 1e9;
        let mut d2: f32 = 1e9;
        let mut d3: f32 = 1e9;
        for k in 0..6 {
            let id: Vec2i = vec2i(select(ic + sx, ic, k < 3), jc - 1 + k - select(3, 0, k < 3));
            let p_upper: Vec2f = wp_lamina_cell_at(id, wp_lamina_cell_rnd(id, salt));
            let dd: f32 = bitcast_f32(
                (bitcast_u32(dot(p_upper - x_upper, p_upper - x_upper) + 1e-3) & 0xFFFFFFF8u32)
                    | k as u32,
            );
            d3 = min(d3, max(d2, dd));
            d2 = min(d2, max(d1, dd));
            d1 = min(d1, dd);
        }
        let k1: i32 = (bitcast_u32(d1) & 7u32) as i32;
        let k2: i32 = (bitcast_u32(d2) & 7u32) as i32;
        let mb: Vec2i = vec2i(
            select(ic + sx, ic, k1 < 3),
            jc - 1 + k1 - select(3, 0, k1 < 3),
        );
        let m2: Vec2i = vec2i(
            select(ic + sx, ic, k2 < 3),
            jc - 1 + k2 - select(3, 0, k2 < 3),
        );
        let r1v: Vec4f = wp_lamina_cell_rnd(mb, salt);
        let r2v: Vec4f = wp_lamina_cell_rnd(m2, salt);
        let pb_capital: Vec2f = wp_lamina_cell_at(mb, r1v);
        let p2_upper: Vec2f = wp_lamina_cell_at(m2, r2v);
        let hc: Vec4f = wp_lamina_cell_rnd(mb, salt + 53);
        let c_b: f32 = c_band * s;
        let c1: i32 = select(1, 2, abs(pb_capital.y) < c_b);
        let m_upper: f32 = hw * s;
        let al: f32 = clamp(
            (x_upper.x - pb_capital.x) / (0.55 * WP_LAMINA_CL),
            -1.0,
            1.0,
        );
        let tooth: f32 = select(0.0, 0.38 * (1.0 - abs(al + 0.35)), hc.w < 0.09);
        let bulge: f32 = WP_LAMINA_CW
            * (0.13 * (1.0 - al * al) + max(tooth, 0.0))
            * step(m_upper - 1.4 * WP_LAMINA_CW, abs(pb_capital.y))
            * step(abs(pb_capital.y), m_upper + 0.5 * WP_LAMINA_CW)
            * (1.0 - smoothstep(0.8, 3.0, b));
        let dm: f32 = min((m_upper + bulge - abs(x_upper.y)) / s, ac.x);
        let hm_upper: f32 = 1.25;
        let dm3: Vec3f = vec3f(dm, dm, dm) + vec3f(ca_m, 0.0, -ca_m);
        let ab: f32 = pb_capital.x - WP_LAMINA_SHEAR * wp_lamina_sab(pb_capital.y);
        let acb: Vec2f = vec2f(ab, pb_capital.y / s);
        let dens: f32 = wp_lamina_dens_screen(
            ctx,
            l_upper.tip + (*ctx).g_k * (acb.x * l_upper.dir + acb.y * nrm),
        );
        let tavg_capital: Vec3f = mix(
            WP_LAMINA_T_GAP,
            mix(WP_LAMINA_T_CHL_A, WP_LAMINA_T_CHL_B, vec3f(0.4, 0.4, 0.4)),
            vec3f(
                0.42 * get!(BLUR_LOCALS).wp_p1.y * mix(0.35, 1.0, dens),
                0.42 * get!(BLUR_LOCALS).wp_p1.y * mix(0.35, 1.0, dens),
                0.42 * get!(BLUR_LOCALS).wp_p1.y * mix(0.35, 1.0, dens),
            ),
        );
        let ci: Vec3f = smoothstep(vec3f(-e, -e, -e), vec3f(e, e, e), dm3);
        if dm < 0.0 {
            let wc: Vec3f = wp_lamina_line_cov3(-dm3, hm_upper, b, aa) * 0.6;
            o.t_upper = mix(
                mix(vec3f(1.0, 1.0, 1.0), tavg_capital, ci),
                WP_LAMINA_T_WALL,
                wc,
            );
            o.wall = wc.y;
            o.chl = 0.3 * ci.y * get!(BLUR_LOCALS).wp_p1.y;
            o.glow = ci.y;
            o.fringe = exp(-wp_lamina_sq((-dm - 0.9 - 0.5 * b) / (0.9 + b))) * 0.9 / (0.9 + b);
            {
                if b * (*ctx).g_k > WP_LAMINA_B_LO {
                    let soft = wp_lamina_layer_soft(ctx, l_upper, p, ac, hw, b, e, aa, g_m, w_band);
                    o = wp_lamina_mix_l(ctx, o, soft, b);
                }
                return o;
            }
        }
        let n2: Vec2f = normalize(p2_upper - pb_capital);
        let b2: f32 = dot(0.5 * (p2_upper + pb_capital) - x_upper, n2);
        let c2: i32 = select(1, 2, abs(p2_upper.y) < c_b);
        let mut d_c: f32 = select(1e9, b2, c2 != c1);
        let w2: f32 = select(
            b2 - WP_LAMINA_HT
                * (0.6 + 0.8 * fract((r1v.z + r2v.z) * 7.31))
                * select(1.0, 1.4, c1 != c2),
            1e9,
            c1 == 2 && c2 == 2,
        );
        let mut d_w: f32 = wp_lamina_smin(abs(dm - hm_upper) - hm_upper, w2, 3.2);
        let mut n_w: Vec2f = n2;
        if sqrt(d3) - sqrt(d1) < 2.0 * (3.2 + 1.3 + d_w) {
            let k3: i32 = (bitcast_u32(d3) & 7u32) as i32;
            let m3: Vec2i = vec2i(
                select(ic + sx, ic, k3 < 3),
                jc - 1 + k3 - select(3, 0, k3 < 3),
            );
            let r3v: Vec4f = wp_lamina_cell_rnd(m3, salt);
            let p3_upper: Vec2f = wp_lamina_cell_at(m3, r3v);
            let n3: Vec2f = normalize(p3_upper - pb_capital);
            let b3: f32 = dot(0.5 * (p3_upper + pb_capital) - x_upper, n3);
            let c3: i32 = select(1, 2, abs(p3_upper.y) < c_b);
            if c3 != c1 {
                d_c = min(d_c, b3);
            }
            let w3: f32 = select(
                b3 - WP_LAMINA_HT
                    * (0.6 + 0.8 * fract((r1v.z + r3v.z) * 7.31))
                    * select(1.0, 1.4, c1 != c3),
                1e9,
                c1 == 2 && c3 == 2,
            );
            n_w = select(n2, n3, w3 < w2);
            d_w = wp_lamina_smin(d_w, w3, 3.2);
        }
        let gws: Vec2f = select(
            g_m * -sign(dm - hm_upper),
            -(n_w.x * l_upper.dir + (WP_LAMINA_SHEAR * sign(x_upper.y) * n_w.x + s * n_w.y) * nrm),
            d_w < abs(dm - hm_upper) - hm_upper - 0.01,
        );
        let ca_w: f32 = dot((*ctx).g_c_as, gws) / (*ctx).g_k;
        let mesh_k: f32 = 1.0 - smoothstep(3.4, WP_LAMINA_B_HI, b * (*ctx).g_k);
        let wc: Vec3f = wp_lamina_line_cov3(
            vec3f(d_w, d_w, d_w) + vec3f(ca_w, 0.0, -ca_w),
            WP_LAMINA_HT,
            b,
            aa,
        ) * mesh_k;
        let flank: f32 = exp(-wp_lamina_sq((d_w - 0.55 - 0.5 * b) / (0.55 + b))) * 0.55
            / (0.55 + b)
            * (*ctx).g_phase
            * mesh_k;
        let mut w_c: f32 = smoothstep(0.0, aa, d_c) * 0.5;
        w_c = select(0.5 - w_c, 0.5 + w_c, c1 == 2);
        w_c = mix(w_c, w_band, smoothstep(0.4, 2.5, b));
        let tcos_capital: Vec3f = pow(WP_LAMINA_T_COSTA, vec3f(0.6, 0.6, 0.6));
        let _chl_c: f32 = 0.1;
        let mut tlam_capital: Vec3f = vec3f(1.0, 1.0, 1.0);
        let _chl_l: f32 = 0.0;
        let mut mean_c: f32 = 0.0;
        if w_c < 0.999 {
            let tip_t: f32 = smoothstep(0.03, 0.17, acb.x / l_upper.taper);
            let filled: bool = hc.x < tip_t * (0.2 + 0.8 * dens) + 0.02;
            let cover: f32 = get!(BLUR_LOCALS).wp_p1.y * (0.78 + 0.22 * hc.y) * mix(0.6, 1.0, dens);
            mean_c = select(0.0, cover * 0.62, filled);
            let depth: f32 = d_w;
            let cluster: f32 =
                wp_lamina_nz(acb / 170.0 + vec2f((l_upper.seed & 1023u32) as f32 * 0.37, 3.1));
            let streams: f32 = step(0.32, cluster) * step(hc.z, 0.9);
            let under: f32 = wp_lamina_nz(x_upper / 5.5 + 60.0 * hc.zw());
            let mut tgap_capital: Vec3f = mix(
                WP_LAMINA_T_GAP,
                vec3f(0.86, 0.9, 0.62),
                vec3f(0.35 * streams, 0.35 * streams, 0.35 * streams),
            );
            tgap_capital = mix(
                tgap_capital,
                mix(WP_LAMINA_T_CHL_A, WP_LAMINA_T_GAP, vec3f(0.3, 0.3, 0.3)),
                vec3f(
                    min(mean_c * (0.3 + 0.3 * under), 0.92) * (1.0 - 0.6 * streams),
                    min(mean_c * (0.3 + 0.3 * under), 0.92) * (1.0 - 0.6 * streams),
                    min(mean_c * (0.3 + 0.3 * under), 0.92) * (1.0 - 0.6 * streams),
                ),
            );
            let det: f32 = select(
                1.0 - smoothstep(2.8, WP_LAMINA_B_HI, b * (*ctx).g_k),
                0.0,
                c1 == 2,
            );
            let mut cov: f32 = 0.0;
            let mut shade: f32 = 0.5;
            let mut rimv: f32 = 0.0;
            let mut cen: f32 = 0.0;
            let mut cov3: Vec3f = vec3f(0.0, 0.0, 0.0);
            if filled && det > 0.0 {
                let lc: Vec2f = x_upper - pb_capital;
                let a: f32 = 11.0;
                let bb: f32 = 0.5 * WP_LAMINA_CW;
                let ak: f32 = a - WP_LAMINA_HT - (0.3 + WP_LAMINA_RC);
                let bk: f32 = bb - WP_LAMINA_HT - (0.3 + WP_LAMINA_RC);
                let eck: f32 = (ak - bk) / (ak + bk);
                let nk_capital: f32 = 5.0;
                let th: f32 = wp_lamina_fast_atan(lc.y * ak, lc.x * bk);
                let uu: f32 = (th - eck * sin(2.0 * th) * 0.5) / WP_LAMINA_TAU;
                let dir: f32 = select(-1.0, 1.0, ((mb.x + mb.y) & 1) == 0);
                let lk_capital: f32 = 3.14159 * (ak + bk) * (1.0 + 0.25 * eck * eck);
                let rev: f32 = floor(
                    get!(BLUR_LOCALS).wp_p0.y / (*ctx).g_k * WP_LAMINA_PERIOD / lk_capital + 0.5,
                );
                let ph: f32 =
                    rev * nk_capital * (t / WP_LAMINA_PERIOD) * dir * streams + hc.w * nk_capital;
                let cvk: f32 = cover * (1.0 - 0.2 * streams) + 0.15;
                let us: f32 = uu * nk_capital - ph;
                let s0: f32 = floor(us + 0.5);
                let s1: f32 = s0 + select(-1.0, 1.0, us > s0);
                let soft: f32 = 0.6;
                let dep: f32 = smoothstep(-0.8, 0.6, depth);
                (*ctx).g_rack_sign = 0.12;
                let mut wsum: f32 = 0.0;
                let mut rsum: f32 = 0.0;
                let mut csum: f32 = 0.0;
                let mut ssum: f32 = 0.0;
                let mut miss: f32 = 1.0;
                for o2 in 0..2 {
                    let sl: f32 = select(s1, s0, o2 == 0);
                    let si: i32 = wp_mod(sl, nk_capital) as i32;
                    let r: Vec4f = wp_lamina_cell_rnd(mb * vec2i(7, 3) + vec2i(si, 101), salt);
                    let uk: f32 = (sl + ph + (r.y - 0.5) * 0.16) / nk_capital;
                    let thk: f32 = WP_LAMINA_TAU * uk + 0.5 * eck * sin(2.0 * WP_LAMINA_TAU * uk);
                    let pk_capital: Vec2f =
                        vec2f(ak * cos(thk), bk * sin(thk)) * (1.0 + (r.z - 0.5) * 0.08);
                    let d: Vec2f = lc - pk_capital;
                    let rad: f32 = WP_LAMINA_RC * (0.9 + 0.12 * r.w);
                    let sd: f32 = length(d) - rad;
                    let mut rm: f32 = 0.0;
                    let c: f32 = wp_lamina_disc_cov(ctx, sd, rad, b + soft, aa, &mut rm)
                        * dep
                        * step(r.x, cvk);
                    miss = miss * (1.0 - clamp(c, 0.0, 1.0));
                    wsum = wsum + c;
                    ssum = ssum + c * r.y * r.y;
                    rsum = rsum + c * rm;
                    csum = csum + c * smoothstep(0.0, -0.8 * rad, sd);
                }
                cov = 1.0 - miss;
                cov3 = vec3f(cov, cov, cov);
                let iw: f32 = 1.0 / max(wsum, 1e-3);
                shade = mix(0.5, ssum * iw, min(wsum * 4.0, 1.0));
                rimv = rsum * iw * cov;
                cen = csum * iw * cov;
            }
            let prof: f32 =
                smoothstep(-0.5, 3.5, depth) * (1.0 - 0.3 * smoothstep(3.0, 6.0, depth));
            let melt: f32 = smoothstep(-0.5 * b, 0.5 + 2.2 * b, depth) * mesh_k;
            let mut tchl_capital: Vec3f = mix(
                WP_LAMINA_T_CHL_A,
                WP_LAMINA_T_CHL_B,
                vec3f(shade, shade, shade),
            );
            tchl_capital = mix(
                tchl_capital,
                tchl_capital * tchl_capital * 0.9,
                vec3f(0.18 * rimv * det, 0.18 * rimv * det, 0.18 * rimv * det),
            );
            tchl_capital =
                tchl_capital * (vec3f(1.0, 1.0, 1.0) + 0.2 * cen * det * vec3f(1.0, 1.0, 1.4));
            let tsharp_capital: Vec3f = mix(
                tgap_capital,
                tchl_capital,
                clamp(cov3, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0)),
            );
            let tsoft_capital: Vec3f = mix(
                tgap_capital,
                mix(WP_LAMINA_T_CHL_A, WP_LAMINA_T_CHL_B, vec3f(0.4, 0.4, 0.4)),
                vec3f(
                    mean_c * prof * 1.1,
                    mean_c * prof * 1.1,
                    mean_c * prof * 1.1,
                ),
            );
            tlam_capital = select(
                mix(WP_LAMINA_T_TIP, WP_LAMINA_T_GAP, vec3f(0.2, 0.2, 0.2)),
                mix(tsoft_capital, tsharp_capital, vec3f(det, det, det)),
                filled,
            );
            tlam_capital = mix(tavg_capital, tlam_capital, vec3f(melt, melt, melt));
        }
        let mut tin_capital: Vec3f = mix(tlam_capital, tcos_capital, vec3f(w_c, w_c, w_c));
        tin_capital = tin_capital * (vec3f(1.0, 1.0, 1.0) - 0.42 * flank * vec3f(1.0, 0.96, 0.8));
        tin_capital = mix(tin_capital, WP_LAMINA_T_WALL, wc);
        o.t_upper = mix(vec3f(1.0, 1.0, 1.0), tin_capital, ci);
        o.glow = ci.y;
        {
            if b * (*ctx).g_k > WP_LAMINA_B_LO {
                let soft = wp_lamina_layer_soft(ctx, l_upper, p, ac, hw, b, e, aa, g_m, w_band);
                o = wp_lamina_mix_l(ctx, o, soft, b);
            }
            return o;
        }
    }

    pub fn wp_lamina_brown(t: f32, o: Vec2f) -> Vec2f {
        let a: f32 = WP_LAMINA_TAU * t / WP_LAMINA_PERIOD;
        let c: Vec2f = vec2f(cos(a), sin(a));
        return (wp_noise((o + c * 9.0) / 256.0).xy() - vec2f(0.5, 0.5)) * 4.0
            + (wp_noise((o + vec2f(73.0, 73.0) + c * 40.0) / 256.0).xy() - vec2f(0.5, 0.5)) * 2.4
            + (wp_noise((o + vec2f(151.0, 151.0) + c * 110.0) / 256.0).xy() - vec2f(0.5, 0.5))
                * 1.3;
    }

    pub fn wp_lamina(fc: Vec2f) -> Vec4f {
        let mut result = vec4f(0.0, 0.0, 0.0, 0.0);
        let mut state = WpLaminaContext {
            g_k: 0.0,
            g_breath: 0.0,
            g_rack_sign: 0.45,
            g_phase: 0.0,
            g_focus: 0.0,
            g_bmax: 0.0,
            g_head_c: vec2f(0.0, 0.0),
            g_head_b: vec2f(0.0, 0.0),
            g_rect_c: vec2f(0.0, 0.0),
            g_rect_b: vec2f(0.0, 0.0),
            g_c_as: vec2f(0.0, 0.0),
        };
        let ctx = &mut state;
        let pr: f32 = max(get!(BLUR_LOCALS).wp_frame.z, 0.25);
        let r_upper: Vec2f = get!(BLUR_LOCALS).wp_frame.xy() / pr;
        let q: Vec2f = fc / pr;
        let t: f32 = get!(BLUR_LOCALS).wp_frame.w;
        let dark: bool = get!(BLUR_LOCALS).wp_env.x > 0.5;
        let mag: f32 = clamp(get!(BLUR_LOCALS).wp_p0.x, 0.5, 3.0);
        (*ctx).g_k = 3.2 * mag;
        let sd0: f32 =
            fract(get!(BLUR_LOCALS).wp_env.z * 7.31 + get!(BLUR_LOCALS).wp_p1.w * 0.6180339);
        let sd1: f32 = fract(sd0 * 13.7 + 0.29);
        let seed_u: u32 = get!(BLUR_LOCALS).wp_p1.w as u32 * 7919u32
            + (get!(BLUR_LOCALS).wp_env.z * 1000.0) as u32;
        let mut cr: Vec4f = get!(BLUR_LOCALS).wp_rect / pr;
        if cr.z < 1.0 || cr.w < 1.0 {
            cr = vec4f(
                r_upper.x * 0.2,
                r_upper.y * 0.42,
                r_upper.x * 0.6,
                r_upper.y * 0.5,
            );
        }
        let head_h: f32 = min(cr.w * 0.45, max(150.0, cr.w * 0.36));
        (*ctx).g_head_b = vec2f(cr.z, head_h) * 0.5;
        (*ctx).g_head_c = vec2f(cr.x, cr.y + cr.w - head_h) + (*ctx).g_head_b;
        let wave: f32 = 0.5 - 0.5 * cos(WP_LAMINA_TAU * t * 15.0 / WP_LAMINA_PERIOD);
        let f: f32 = clamp(
            get!(BLUR_LOCALS).wp_p1.x + select(0.0, 0.42 * wave, get!(BLUR_LOCALS).wp_p0.z > 0.5),
            0.0,
            1.0,
        );
        (*ctx).g_breath = 1.0 + 0.012 * (f - 0.5);
        let drift: Vec2f = vec2f(
            7.0 * sin(WP_LAMINA_TAU * 2.0 * t / WP_LAMINA_PERIOD),
            4.5 * sin(WP_LAMINA_TAU * 5.0 * t / WP_LAMINA_PERIOD + 1.3),
        );
        let sp: Vec2f = 0.5 * r_upper + (q - 0.5 * r_upper) / (*ctx).g_breath + drift;
        let cond: f32 = clamp(get!(BLUR_LOCALS).wp_p0.w, 0.0, 1.0);
        (*ctx).g_bmax = mix(15.0, 10.0, cond);
        let m0: Vec2f = vec2f(-0.05 * r_upper.x, cr.y - 105.0 + (sd0 - 0.5) * 24.0);
        let m1: Vec2f = vec2f(1.05 * r_upper.x, cr.y - 65.0 + (sd1 - 0.5) * 24.0);
        let md: Vec2f = normalize(m1 - m0);
        let inw: Vec2f = vec2f(md.y, -md.x);
        let wa_upper: f32 = 1.05 * max(r_upper.y, 0.55 * r_upper.x);
        let ta_upper: f32 = 2.6 * wa_upper;
        let tip_a: Vec2f = mix(m0, m1, vec2f(0.5, 0.5)) + inw * wa_upper + md * ta_upper * 1.35;
        let a_upper: WpLaminaLeaf = WpLaminaLeaf {
            tip: tip_a,
            dir: -md,
            w_upper: wa_upper / (*ctx).g_k,
            taper: ta_upper / (*ctx).g_k,
            costa: 0.16 * wa_upper / (*ctx).g_k,
            z: 0.0,
            seed: seed_u * 3u32 + 17u32,
            tilt: 1.0 / 50.0,
            band: 44.0,
        };
        let v_a: Vec2f = (sp - tip_a) / (*ctx).g_k;
        let ac_a: Vec2f = vec2f(
            dot(v_a, a_upper.dir),
            dot(v_a, vec2f(-a_upper.dir.y, a_upper.dir.x)),
        );
        let reach_a: f32 = 6.0 + 2.5 * (1.3 + 10.0 + 3.0 + 2.9 * (*ctx).g_bmax) / (*ctx).g_k;
        let hw_a: f32 = wp_lamina_half_w(a_upper, max(ac_a.x, 0.0));
        let near_a: bool = ac_a.x > -reach_a && abs(ac_a.y) - hw_a < reach_a;
        let ocn: Vec2f = (q - r_upper * vec2f(0.46, 0.55)) / max(r_upper.x, r_upper.y);
        let illum: f32 = 1.012 - 0.13 * dot(ocn, ocn);
        let fr: u32 = wp_lamina_lb32(floor(t * 30.0) as u32 + 0x51u32);
        let gp: Vec2f = floor(fc) + vec2f((fr & 255u32) as f32, ((fr >> 8u32) & 255u32) as f32);
        let gn: Vec2f = wp_noise((gp + vec2f(0.5, 0.5)) / 256.0).xy();
        let gc: f32 = wp_noise((gp + vec2f(38.0, 92.0)) / 256.0).z;
        let film: f32 = (gn.x + gn.y - 1.0 + 1.4 * (gc - 0.5)) * 2.2;
        let mut la: WpLaminaLout = WpLaminaLout {
            t_upper: vec3f(1.0, 1.0, 1.0),
            wall: 0.0,
            chl: 0.0,
            mean: 0.0,
            glow: 0.0,
            fringe: 0.0,
        };
        let mut keep: f32 = 1.0;
        if near_a {
            let aa: f32 = 0.6 / (pr * (*ctx).g_k * (*ctx).g_breath);
            (*ctx).g_c_as = (q - r_upper * vec2f(0.46, 0.55)) * WP_LAMINA_CA_K;
            (*ctx).g_phase = mix(0.55, 1.5, cond);
            (*ctx).g_focus = f;
            (*ctx).g_rect_b = cr.zw() * 0.5;
            (*ctx).g_rect_c = cr.xy() + (*ctx).g_rect_b;
            let d_h: f32 = wp_lamina_sd_r_box(q - (*ctx).g_head_c, (*ctx).g_head_b, 18.0);
            let zone_blur: f32 = (1.0 - smoothstep(-20.0, 120.0, d_h)) * 10.0;
            let blur_s: f32 = 1.3 + zone_blur + 12.0 * dot(ocn, ocn);
            keep = mix(0.3, 1.0, smoothstep(-10.0, 140.0, d_h));
            la = wp_lamina_layer(ctx, a_upper, sp, ac_a, hw_a, blur_s, aa, t, dark);
        }
        let mut p_t: f32 = 0.0;
        let mut p_halo: f32 = 0.0;
        {
            let gs_upper: f32 = 150.0;
            let cid: Vec2f = floor(sp / gs_upper);
            let r: Vec4f = texture_load(
                BLUR_TEXTURE,
                wp_bitand2i(
                    vec2i(cid.x as i32, cid.y as i32) + vec2i((seed_u & 127u32) as i32, 37),
                    vec2i(255, 255),
                ),
                0i32,
            );
            if r.x < 0.1 {
                let base: Vec2f = (cid + vec2f(0.27, 0.27) + 0.46 * r.yz()) * gs_upper;
                let z: f32 = select(r.w * 1.4, r.w * 0.5 - 0.1, r.w < 0.55);
                let bz: f32 = abs(f - z) * (*ctx).g_bmax * 1.1 + 0.3;
                let rad: f32 = 1.6 + 2.6 * fract(r.w * 7.3);
                let mut d: Vec2f = sp - base;
                let lim: f32 = rad + bz + 12.0;
                if dot(d, d) < lim * lim
                    && wp_lamina_sd_r_box(
                        base - (*ctx).g_head_c,
                        (*ctx).g_head_b + vec2f(40.0, 40.0),
                        18.0,
                    ) > 0.0
                {
                    d = d - wp_lamina_brown(t, r.yz() * 256.0);
                    let mut rm: f32 = 0.0;
                    (*ctx).g_rack_sign = 0.45;
                    let c: f32 =
                        wp_lamina_disc_cov(ctx, length(d) - rad, rad, bz, 0.6 / pr, &mut rm);
                    p_t = c;
                    p_halo = exp(-wp_lamina_sq((length(d) - rad - 1.2 - bz) / (1.0 + bz))) * 1.0
                        / (1.0 + bz);
                }
            }
        }
        let mut dust: f32 = 0.0;
        let mut dust_rim: f32 = 0.0;
        {
            let c1: Vec2f = r_upper * vec2f(0.07 + 0.1 * sd1, 0.12 + 0.18 * sd0);
            let d1: f32 = length(sp - c1);
            let mut x: Vec2f = max(
                vec2f(1.0, 1.0)
                    - vec2f((d1 - 55.0) / 6.4, d1 / 6.0) * vec2f((d1 - 55.0) / 6.4, d1 / 6.0),
                vec2f(0.0, 0.0),
            );
            x = x * x;
            dust = clamp((62.0 - d1) / 8.0, 0.0, 1.0);
            dust_rim = x.x + 0.5 * x.y;
        }
        let mut col: Vec3f = vec3f(0.0, 0.0, 0.0);
        if !dark {
            let mut t_upper: Vec3f = mix(vec3f(1.0, 1.0, 1.0), la.t_upper, vec3f(keep, keep, keep));
            if la.fringe + p_t + p_halo + dust + dust_rim > 0.0 {
                let lat: f32 = clamp(length(ocn) * 2.2, 0.0, 1.0);
                t_upper = t_upper
                    * mix(
                        vec3f(1.0, 1.0, 1.0),
                        vec3f(0.93, 0.86, 0.99),
                        vec3f(
                            la.fringe * keep * (0.55 + 0.45 * lat),
                            la.fringe * keep * (0.55 + 0.45 * lat),
                            la.fringe * keep * (0.55 + 0.45 * lat),
                        ),
                    );
                t_upper = t_upper * (vec3f(1.0, 1.0, 1.0) - 0.62 * p_t * vec3f(1.0, 1.03, 1.1));
                t_upper = t_upper * (1.0 + 0.09 * p_halo);
                t_upper = t_upper
                    * (vec3f(1.0 - 0.016 * dust, 1.0 - 0.016 * dust, 1.0 - 0.016 * dust)
                        - 0.022 * dust_rim * vec3f(0.9, 1.0, 1.25));
            }
            col = 1.055
                * pow(
                    max(WP_LAMINA_FIELD * illum * t_upper, vec3f(1e-4, 1e-4, 1e-4)),
                    vec3f(1.0 / 2.4, 1.0 / 2.4, 1.0 / 2.4),
                )
                - vec3f(0.055, 0.055, 0.055);
        } else {
            let mut lit: Vec3f = max(
                la.t_upper - vec3f(1.0 - la.glow, 1.0 - la.glow, 1.0 - la.glow),
                vec3f(0.0, 0.0, 0.0),
            );
            lit = lit * (vec3f(1.0, 1.0, 1.0) - 0.62 * p_t * vec3f(1.0, 1.03, 1.1));
            let cv: f32 = la.glow * keep;
            let ly: f32 = dot(lit, vec3f(0.3, 0.55, 0.15));
            lit = max(
                mix(vec3f(ly, ly, ly), lit, vec3f(1.7, 1.7, 1.7)),
                vec3f(0.0, 0.0, 0.0),
            );
            col = vec3f(0.0042, 0.0042, 0.0056) * illum * (1.0 - cv)
                + 0.3 * WP_LAMINA_FIELD * illum * lit * vec3f(0.85, 1.05, 0.72) * keep;
            col = col + vec3f(0.0055, 0.0052, 0.0075) * la.fringe * keep;
            col = col
                + (vec3f(0.010, 0.011, 0.010) * (p_t * (1.0 - cv) + 0.4 * p_halo)
                    + vec3f(0.0012 * dust_rim, 0.0012 * dust_rim, 0.0012 * dust_rim));
            col = 1.055
                * pow(
                    max(col, vec3f(0.0031308, 0.0031308, 0.0031308)),
                    vec3f(1.0 / 2.4, 1.0 / 2.4, 1.0 / 2.4),
                )
                - vec3f(0.055, 0.055, 0.055);
        }
        let lum: f32 = dot(col, vec3f(0.3, 0.55, 0.15));
        let g_amp: f32 = max(
            get!(BLUR_LOCALS).wp_p1.z
                * 0.043
                * (0.45 + 2.2 * lum * (1.0 - lum))
                * select(1.0, 0.8, dark),
            0.6 / 255.0,
        );
        col = col + film * g_amp * vec3f(1.0, 0.98, 1.03);
        result = vec4f(col.x, col.y, col.z, 1.0);
        result
    }

    // butler.diatom/shader.frag
    #[derive(Clone, Copy, Wgsl)]
    pub struct WpDiatomPx {
        pub t_upper: Vec3f,
        pub g_upper: Vec3f,
        pub h_upper: f32,
        pub pig: f32,
    }

    #[derive(Clone, Copy, Wgsl)]
    pub struct WpDiatomGeo {
        pub d_p: f32,
        pub rp: f32,
        pub fill: f32,
        pub present: f32,
        pub d_co: f32,
        pub hw_co: f32,
        pub d_ru: f32,
        pub hw_ru: f32,
        pub rung_on: f32,
        pub hyal: f32,
        pub rim_w: f32,
        pub cell_v: f32,
    }

    #[derive(Clone, Copy, Wgsl)]
    pub struct WpDiatomContext {
        pub g_aa: f32,
        pub g_dof: f32,
        pub g_ca: f32,
        pub g_k: f32,
    }

    pub const WP_DIATOM_TAU: f32 = 6.28318530718;

    pub const WP_DIATOM_PI: f32 = 3.14159265359;

    pub fn wp_diatom_fsin(x: f32) -> f32 {
        let mut x = x;
        x = x * 0.15915494;
        x = x - floor(x + 0.5);
        let y: f32 = x * (8.0 - 16.0 * abs(x));
        return 0.225 * (y * abs(y) - y) + y;
    }

    pub fn wp_diatom_fcos(x: f32) -> f32 {
        return wp_diatom_fsin(x + 1.5707963);
    }

    pub fn wp_diatom_fsin3(x: Vec3f) -> Vec3f {
        return vec3f(
            wp_diatom_fsin(x.x),
            wp_diatom_fsin(x.y),
            wp_diatom_fsin(x.z),
        );
    }

    pub fn wp_diatom_fcos3(x: Vec3f) -> Vec3f {
        return vec3f(
            wp_diatom_fcos(x.x),
            wp_diatom_fcos(x.y),
            wp_diatom_fcos(x.z),
        );
    }

    pub fn wp_diatom_sq(x: f32) -> f32 {
        return x * x;
    }

    pub fn wp_diatom_sat(x: f32) -> f32 {
        return clamp(x, 0.0, 1.0);
    }

    pub fn wp_diatom_hash11(p: f32) -> f32 {
        let mut p = p;
        p = fract(p * 0.1031);
        p = p * (p + 33.33);
        p = p * (p + p);
        return fract(p);
    }

    pub fn wp_diatom_hash12(p: Vec2f) -> f32 {
        let mut p3: Vec3f = fract(
            vec3f(
                vec3f(p.x, p.y, p.x).x,
                vec3f(p.x, p.y, p.x).y,
                vec3f(p.x, p.y, p.x).z,
            ) * 0.1031,
        );
        p3 = p3
            + vec3f(
                dot(p3, p3.yzx() + vec3f(33.33, 33.33, 33.33)),
                dot(p3, p3.yzx() + vec3f(33.33, 33.33, 33.33)),
                dot(p3, p3.yzx() + vec3f(33.33, 33.33, 33.33)),
            );
        return fract((p3.x + p3.y) * p3.z);
    }

    pub fn wp_diatom_hash22(p: Vec2f) -> Vec2f {
        let mut p3: Vec3f = fract(
            vec3f(
                vec3f(p.x, p.y, p.x).x,
                vec3f(p.x, p.y, p.x).y,
                vec3f(p.x, p.y, p.x).z,
            ) * vec3f(0.1031, 0.1030, 0.0973),
        );
        p3 = p3
            + vec3f(
                dot(p3, p3.yzx() + vec3f(33.33, 33.33, 33.33)),
                dot(p3, p3.yzx() + vec3f(33.33, 33.33, 33.33)),
                dot(p3, p3.yzx() + vec3f(33.33, 33.33, 33.33)),
            );
        return fract((p3.xx() + p3.yz()) * p3.zy());
    }

    pub fn wp_diatom_hash32(p: Vec2f) -> Vec3f {
        let mut p3: Vec3f = fract(
            vec3f(
                vec3f(p.x, p.y, p.x).x,
                vec3f(p.x, p.y, p.x).y,
                vec3f(p.x, p.y, p.x).z,
            ) * vec3f(0.1031, 0.1030, 0.0973),
        );
        p3 = p3
            + vec3f(
                dot(p3, p3.yxz() + vec3f(33.33, 33.33, 33.33)),
                dot(p3, p3.yxz() + vec3f(33.33, 33.33, 33.33)),
                dot(p3, p3.yxz() + vec3f(33.33, 33.33, 33.33)),
            );
        return fract((p3.xxy() + p3.yzz()) * p3.zyx());
    }

    pub fn wp_diatom_vnoise(x: Vec2f) -> f32 {
        let i: Vec2f = floor(x);
        let mut f: Vec2f = fract(x);
        f = f * f * (vec2f(3.0, 3.0) - 2.0 * f);
        return wp_noise((i + f + vec2f(0.5, 0.5)) / 256.0).x;
    }

    pub fn wp_diatom_rot(a: f32) -> Mat2x2f {
        let c: f32 = wp_diatom_fcos(a);
        let s: f32 = wp_diatom_fsin(a);
        return mat2x2f(vec2f(c, s), vec2f(-s, c));
    }

    pub fn wp_diatom_sd_round_box(p: Vec2f, b: Vec2f, r: f32) -> f32 {
        let q: Vec2f = abs(p) - b + vec2f(r, r);
        return length(max(q, vec2f(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
    }

    pub fn wp_diatom_tri(x: f32) -> f32 {
        return abs(fract(x) - 0.5);
    }

    pub fn wp_diatom_contrast_c(z: Vec3f) -> Vec3f {
        return wp_diatom_fcos3(1.35 * z) * exp(-0.1 * z * z);
    }

    pub fn wp_diatom_becke_b(z: Vec3f) -> Vec3f {
        return wp_diatom_fsin3(1.35 * z) * exp(-0.22 * z * z);
    }

    pub fn wp_diatom_optics_cb(
        ctx: ptr!(function, WpDiatomContext),
        z: f32,
        c_upper: ptr!(function, Vec3f),
        b_upper: ptr!(function, Vec3f),
        b0_upper: ptr!(function, f32),
    ) {
        let s: f32 = wp_diatom_fsin(1.35 * z);
        let c: f32 = wp_diatom_fcos(1.35 * z);
        let e1: f32 = exp(-0.1 * z * z);
        let e2: f32 = exp(-0.22 * z * z);
        let off: Vec3f = vec3f((*ctx).g_ca, -(*ctx).g_ca, (*ctx).g_ca * 0.7);
        *c_upper = vec3f(c * e1, c * e1, c * e1) + (-1.35 * s - 0.2 * z * c) * e1 * off;
        *b0_upper = s * e2;
        *b_upper = vec3f(*b0_upper, *b0_upper, *b0_upper) + (1.35 * c - 0.44 * z * s) * e2 * off;
    }

    pub fn wp_diatom_becke_ca(ctx: ptr!(function, WpDiatomContext), z: f32) -> Vec3f {
        let s: f32 = wp_diatom_fsin(1.35 * z);
        let c: f32 = wp_diatom_fcos(1.35 * z);
        let e2: f32 = exp(-0.22 * z * z);
        return vec3f(s * e2, s * e2, s * e2)
            + (1.35 * c - 0.44 * z * s) * e2 * vec3f((*ctx).g_ca, -(*ctx).g_ca, (*ctx).g_ca * 0.7);
    }

    pub fn wp_diatom_costa_cand(
        k0: f32,
        k0i: i32,
        i: i32,
        r: f32,
        a: f32,
        sd: f32,
        a_l: ptr!(function, f32),
        j_l: ptr!(function, f32),
        lv_l: ptr!(function, f32),
        st_l: ptr!(function, f32),
        a_r: ptr!(function, f32),
        lv_r: ptr!(function, f32),
        st_r: ptr!(function, f32),
    ) {
        let ji: i32 = (k0i + i) & 63;
        let jm: f32 = ji as f32;
        let lv: f32 = select(select(2.0, 1.0, (ji & 1) == 0), 0.0, (ji & 3) == 0);
        let hj: f32 = wp_diatom_hash11(jm * 7.31 + sd * 13.0);
        let st: f32 = select(
            select(0.55 + 0.22 * fract(hj * 9.1), 0.33 + 0.13 * hj, lv < 1.5),
            0.18,
            lv < 0.5,
        );
        if r >= st {
            let pos: f32 =
                k0 + i as f32 + (hj - 0.5) * 0.3 + 0.14 * wp_diatom_fsin(r * 7.0 + hj * 6.28);
            if pos <= a && pos > *a_l {
                *a_l = pos;
                *j_l = jm;
                *lv_l = lv;
                *st_l = st;
            }
            if pos > a && pos < *a_r {
                *a_r = pos;
                *lv_r = lv;
                *st_r = st;
            }
        }
    }

    pub fn wp_diatom_geo_radial(q: Vec2f, r_upper: f32, sd: f32, th: f32) -> WpDiatomGeo {
        let mut g: WpDiatomGeo = WpDiatomGeo {
            d_p: 0.0,
            rp: 0.0,
            fill: 0.0,
            present: 0.0,
            d_co: 0.0,
            hw_co: 0.0,
            d_ru: 0.0,
            hw_ru: 0.0,
            rung_on: 0.0,
            hyal: 0.0,
            rim_w: 0.0,
            cell_v: 0.0,
        };
        g.d_p = 1e3;
        g.rp = 1.0;
        g.fill = 0.4;
        g.present = 0.0;
        g.d_co = 1e3;
        g.hw_co = 1.0;
        g.d_ru = 1e3;
        g.hw_ru = 1.0;
        g.rung_on = 0.0;
        g.hyal = 0.0;
        g.cell_v = 0.0;
        g.rim_w = 0.046 * r_upper;
        let r: f32 = length(q) / r_upper;
        let sp: f32 = 0.0192 * r_upper;
        if r < 0.185 {
            let row: f32 = select(1.0, 0.0, r < 0.142);
            let nn: f32 = select(26.0, 19.0, row < 0.5);
            let rc: f32 = select(0.162, 0.124, row < 0.5);
            let ang: f32 = (th / WP_DIATOM_TAU + 0.5) * nn + row * 0.37;
            let cid: f32 = floor(ang);
            let hc: Vec3f = wp_diatom_hash32(vec2f(cid, row) + vec2f(sd * 5.0, sd * 5.0));
            let mut dxy: Vec2f = vec2f(
                (fract(ang) - 0.5) * WP_DIATOM_TAU / nn * rc * r_upper,
                (r - rc) * r_upper,
            );
            dxy = dxy + (hc.xy() - vec2f(0.5, 0.5)) * 0.2 * sp;
            g.rp = select(0.0096, 0.0112, row < 0.5) * r_upper * (0.88 + 0.24 * hc.z);
            g.d_p = mix(length(dxy), max(abs(dxy.x), abs(dxy.y)), 0.2);
            g.fill = 3.1 * g.rp * g.rp / (WP_DIATOM_TAU / nn * rc * r_upper * 0.038 * r_upper);
            g.present = step(0.05, fract(hc.z * 7.3));
            g.hyal = 1.0 - smoothstep(0.099, 0.107, r);
            return g;
        }
        let a: f32 = (th / WP_DIATOM_TAU + 0.5) * 64.0;
        let k0: f32 = floor(a);
        let mut a_l: f32 = -1e4;
        let mut a_r: f32 = 1e4;
        let mut j_l: f32 = 0.0;
        let mut lv_l: f32 = 0.0;
        let mut lv_r: f32 = 0.0;
        let mut st_l: f32 = 0.0;
        let mut st_r: f32 = 0.0;
        let k0i: i32 = k0 as i32;
        wp_diatom_costa_cand(
            k0, k0i, -1, r, a, sd, &mut a_l, &mut j_l, &mut lv_l, &mut st_l, &mut a_r, &mut lv_r,
            &mut st_r,
        );
        wp_diatom_costa_cand(
            k0, k0i, 0, r, a, sd, &mut a_l, &mut j_l, &mut lv_l, &mut st_l, &mut a_r, &mut lv_r,
            &mut st_r,
        );
        wp_diatom_costa_cand(
            k0, k0i, 1, r, a, sd, &mut a_l, &mut j_l, &mut lv_l, &mut st_l, &mut a_r, &mut lv_r,
            &mut st_r,
        );
        wp_diatom_costa_cand(
            k0, k0i, 2, r, a, sd, &mut a_l, &mut j_l, &mut lv_l, &mut st_l, &mut a_r, &mut lv_r,
            &mut st_r,
        );
        if r < 0.77 {
            wp_diatom_costa_cand(
                k0, k0i, -2, r, a, sd, &mut a_l, &mut j_l, &mut lv_l, &mut st_l, &mut a_r,
                &mut lv_r, &mut st_r,
            );
            wp_diatom_costa_cand(
                k0, k0i, 3, r, a, sd, &mut a_l, &mut j_l, &mut lv_l, &mut st_l, &mut a_r,
                &mut lv_r, &mut st_r,
            );
            if r < 0.46 {
                wp_diatom_costa_cand(
                    k0, k0i, -4, r, a, sd, &mut a_l, &mut j_l, &mut lv_l, &mut st_l, &mut a_r,
                    &mut lv_r, &mut st_r,
                );
                wp_diatom_costa_cand(
                    k0, k0i, -3, r, a, sd, &mut a_l, &mut j_l, &mut lv_l, &mut st_l, &mut a_r,
                    &mut lv_r, &mut st_r,
                );
                wp_diatom_costa_cand(
                    k0, k0i, 4, r, a, sd, &mut a_l, &mut j_l, &mut lv_l, &mut st_l, &mut a_r,
                    &mut lv_r, &mut st_r,
                );
                wp_diatom_costa_cand(
                    k0, k0i, 5, r, a, sd, &mut a_l, &mut j_l, &mut lv_l, &mut st_l, &mut a_r,
                    &mut lv_r, &mut st_r,
                );
            }
        }
        let arc_px: f32 = WP_DIATOM_TAU / 64.0 * r * r_upper;
        let hw_l: f32 = r_upper
            * select(select(0.0043, 0.0052, lv_l < 1.5), 0.0064, lv_l < 0.5)
            * smoothstep(st_l, st_l + 0.035, r);
        let hw_r: f32 = r_upper
            * select(select(0.0043, 0.0052, lv_r < 1.5), 0.0064, lv_r < 0.5)
            * smoothstep(st_r, st_r + 0.035, r);
        let d_l: f32 = (a - a_l) * arc_px;
        let d_r: f32 = (a_r - a) * arc_px;
        let e_l: f32 = d_l - hw_l;
        let e_r: f32 = d_r - hw_r;
        g.d_co = min(e_l, e_r);
        g.hw_co = select(hw_r, hw_l, e_l < e_r);
        let f: f32 = (a - a_l) / (a_r - a_l);
        let wpx: f32 = (a_r - a_l) * arc_px;
        let hs: f32 = wp_diatom_hash11(j_l * 3.7 + 11.0 + sd * 7.0);
        let rs: f32 = 0.04 + 0.016 * hs;
        let r0: f32 = (r - 0.18) / rs;
        let wob: f32 = r0 * 1.9 + hs * 17.0;
        let rho: f32 = r0 + hs * 5.0 - 0.24 * f * (1.0 - f) * 4.0
            + 0.1 * wp_diatom_fsin(wob)
            + 0.06 * wp_diatom_fsin(f * 5.3 + wob * 0.37);
        let drho: f32 = 1.0 + 0.19 * wp_diatom_fcos(wob);
        let n: f32 = floor(rho);
        let gg: f32 = fract(rho);
        g.hw_ru = 0.0032 * r_upper;
        g.d_ru = min(gg, 1.0 - gg) * rs * r_upper / drho - g.hw_ru;
        let hcell: f32 = wp_diatom_hash12(vec2f(j_l, n) + vec2f(sd * 3.0, sd * 3.0));
        g.rung_on = step(0.13, hcell);
        g.cell_v = fract(hcell * 7.7) - 0.5;
        let cell_r: f32 = rs * r_upper / drho;
        let in_w: f32 = max(wpx - hw_l - hw_r, 1.0);
        let in_r: f32 = max(cell_r - 2.0 * g.hw_ru, 1.0);
        let n_a: f32 = max(1.0, floor(in_w / sp + 0.2));
        let n_r: f32 = max(1.0, floor(in_r / sp + 0.3));
        let uu: f32 = clamp((f * wpx - hw_l) / in_w, 0.0, 0.999) * n_a;
        let vv: f32 = clamp((gg * cell_r - g.hw_ru) / in_r, 0.0, 0.999) * n_r;
        let cell_px: Vec2f = vec2f(in_w / n_a, in_r / n_r);
        let mut dxy: Vec2f = (fract(vec2f(uu, vv)) - vec2f(0.5, 0.5)) * cell_px;
        let hp: Vec3f = wp_diatom_hash32(
            vec2f(
                j_l * 97.0 + n * 3.1,
                floor(uu) * 13.0 + floor(vv) + n * 71.0,
            ) + vec2f(sd * 17.0, sd * 17.0),
        );
        let jit: Vec2f = (hp.xy() - vec2f(0.5, 0.5)) * 0.08 * sp;
        dxy = dxy + jit;
        g.rp = min(
            min(cell_px.x, cell_px.y) * (0.35 + 0.08 * hp.z),
            0.5 * min(cell_px.x, cell_px.y) - max(abs(jit.x), abs(jit.y)) - 0.6,
        );
        g.d_p = mix(length(dxy), max(abs(dxy.x), abs(dxy.y)), 0.12);
        g.fill = 3.3 * g.rp * g.rp / (cell_px.x * cell_px.y);
        g.present = step(0.025, fract(hp.z * 11.7));
        let outer: f32 = smoothstep(0.83, 0.9, r);
        if outer > 0.0 {
            let nz: f32 = wp_diatom_vnoise(q / (sp * 0.9) + vec2f(sd * 11.0, sd * 11.0)) - 0.5;
            g.d_p = mix(g.d_p, g.d_p * (1.0 + 0.5 * nz) + 0.8 * sp * nz, outer);
            g.rp = g.rp * (1.0 + 0.25 * outer);
            g.cell_v = g.cell_v + outer * nz * 1.6;
            g.present = g.present * step(0.12 * outer, fract(hp.z * 5.1));
        }
        return g;
    }

    pub fn wp_diatom_hex_nearest(p: Vec2f, id: ptr!(function, Vec2f)) -> Vec2f {
        let s: Vec2f = vec2f(1.0, 1.7320508);
        let hc: Vec4f = floor(
            vec4f(p.x, p.y, (p - vec2f(0.5, 1.0)).x, (p - vec2f(0.5, 1.0)).y)
                / vec4f(s.x, s.y, s.x, s.y),
        ) + vec4f(0.5, 0.5, 0.5, 0.5);
        let h: Vec4f = vec4f(
            (p - hc.xy() * s).x,
            (p - hc.xy() * s).y,
            (p - (hc.zw() + vec2f(0.5, 0.5)) * s).x,
            (p - (hc.zw() + vec2f(0.5, 0.5)) * s).y,
        );
        if dot(h.xy(), h.xy()) < dot(h.zw(), h.zw()) {
            *id = hc.xy();
            return h.xy();
        }
        *id = hc.zw() + vec2f(0.5, 0.5);
        return h.zw();
    }

    pub fn wp_diatom_geo_areolate(q: Vec2f, r_upper: f32, sd: f32) -> WpDiatomGeo {
        let mut g: WpDiatomGeo = WpDiatomGeo {
            d_p: 0.0,
            rp: 0.0,
            fill: 0.0,
            present: 0.0,
            d_co: 0.0,
            hw_co: 0.0,
            d_ru: 0.0,
            hw_ru: 0.0,
            rung_on: 0.0,
            hyal: 0.0,
            rim_w: 0.0,
            cell_v: 0.0,
        };
        g.d_co = 1e3;
        g.hw_co = 1.0;
        g.d_ru = 1e3;
        g.hw_ru = 1.0;
        g.rung_on = 0.0;
        g.hyal = 0.0;
        g.cell_v = 0.0;
        g.rim_w = 0.034 * r_upper;
        let r: f32 = length(q) / r_upper;
        let sp: f32 = 0.024 * r_upper;
        let k: f32 = 1.0 + 0.85 * r * r;
        let l_upper: Vec2f = wp_diatom_rot(0.32 * r) * q * k / sp;
        let mut id: Vec2f = vec2f(0.0, 0.0);
        let mut h: Vec2f = wp_diatom_hex_nearest(l_upper, &mut id);
        let pitch: f32 = sp / (k + 1.7 * r * r);
        let hp: Vec3f = wp_diatom_hash32(id + vec2f(sd * 23.0, sd * 23.0));
        h = h + (hp.xy() - vec2f(0.5, 0.5)) * 0.1;
        g.d_p = length(h) * pitch;
        g.rp = pitch * (0.34 + 0.07 * hp.z);
        g.fill = 3.63 * wp_diatom_sq(g.rp / pitch);
        g.present = step(0.003, fract(hp.z * 13.1));
        if r < 0.075 {
            let th: f32 = atan2(q.y, q.x);
            let ang: f32 = (th / WP_DIATOM_TAU + 0.5) * 7.0;
            let rc: f32 = 0.048;
            let dxy: Vec2f = vec2f(
                (fract(ang) - 0.5) * WP_DIATOM_TAU / 7.0 * rc * r_upper,
                (r - rc) * r_upper,
            );
            g.rp = 0.0145 * r_upper;
            g.d_p = length(dxy);
            g.fill = 0.45;
            g.present = 1.0;
            g.hyal = 1.0 - smoothstep(0.022, 0.028, r);
        }
        return g;
    }

    pub fn wp_diatom_chip_field(r: f32, th: f32, th_b: f32) -> f32 {
        let mut dth: f32 = th_b - th;
        dth = dth - WP_DIATOM_TAU * floor(dth / WP_DIATOM_TAU + 0.5);
        let jag: f32 = 0.022 * wp_diatom_tri(dth * 7.3 + 0.3)
            + 0.03 * (wp_diatom_vnoise(vec2f(dth * 23.0, 3.1)) - 0.5)
            + 0.012 * (wp_diatom_vnoise(vec2f(dth * 71.0, 7.7)) - 0.5);
        let slope: f32 = select(0.82, 1.25, dth > 0.0);
        let thr: f32 = 0.57 + slope * abs(dth) + jag;
        return (r - thr) / sqrt(1.0 + wp_diatom_sq(slope / max(r, 0.3)));
    }

    pub fn wp_diatom_valve_shade(
        ctx: ptr!(function, WpDiatomContext),
        q: Vec2f,
        r_upper: f32,
        z_p: f32,
        sd: f32,
        radial: bool,
        mode_upper: i32,
        th: f32,
    ) -> WpDiatomPx {
        let mut o: WpDiatomPx = WpDiatomPx {
            t_upper: vec3f(0.0, 0.0, 0.0),
            g_upper: vec3f(0.0, 0.0, 0.0),
            h_upper: 0.0,
            pig: 0.0,
        };
        o.t_upper = vec3f(1.0, 1.0, 1.0);
        o.g_upper = vec3f(0.0, 0.0, 0.0);
        o.h_upper = 0.0;
        o.pig = 0.0;
        let r: f32 = length(q) / r_upper;
        let e: f32 = (1.0 - r) * r_upper;
        if e < -10.0 {
            return o;
        }
        let mut g: WpDiatomGeo = WpDiatomGeo {
            d_p: 0.0,
            rp: 0.0,
            fill: 0.0,
            present: 0.0,
            d_co: 0.0,
            hw_co: 0.0,
            d_ru: 0.0,
            hw_ru: 0.0,
            rung_on: 0.0,
            hyal: 0.0,
            rim_w: 0.0,
            cell_v: 0.0,
        };
        if radial {
            g = wp_diatom_geo_radial(q, r_upper, sd, th);
        } else {
            g = wp_diatom_geo_areolate(q, r_upper, sd);
        }
        let aa: f32 = (*ctx).g_aa;
        let sc: f32 = r_upper / 468.0;
        let sp: f32 = select(0.02, 0.0192, radial) * r_upper;
        let zs: f32 = z_p * (*ctx).g_dof;
        let mut cc_capital: Vec3f = vec3f(0.0, 0.0, 0.0);
        let mut bc_capital: Vec3f = vec3f(0.0, 0.0, 0.0);
        let mut b0_upper: f32 = 0.0;
        wp_diatom_optics_cb(ctx, zs, &mut cc_capital, &mut bc_capital, &mut b0_upper);
        let blur_p: f32 = sp * (0.06 + 0.14 * abs(zs));
        let att_p: f32 = exp(-2.2 * wp_diatom_sq(blur_p / sp));
        let z_c: f32 = (z_p + 0.75) * (*ctx).g_dof;
        let blur_c: f32 = (0.4 + 1.5 * abs(z_c)) * sc;
        let bc_c_capital: Vec3f = wp_diatom_becke_ca(ctx, z_c);
        let blur_e: f32 = (0.5 + 2.4 * abs(zs)) * sc;
        let in_v: f32 = smoothstep(-aa, aa, e);
        let pore_zone: f32 = (1.0 - g.hyal) * smoothstep(g.rim_w - 1.0, g.rim_w + 2.5, e);
        let se: f32 = min(blur_p, 0.55 * g.rp) + aa;
        let pm: f32 = mix(0.3, 1.0, g.present) * (1.0 - smoothstep(g.rp - se, g.rp + se, g.d_p));
        let ring: f32 = exp(-wp_diatom_sq((g.d_p - g.rp) / (0.28 * g.rp + se)));
        let speck: f32 = wp_diatom_vnoise(q / (sp * 1.7) + vec2f(sd * 9.0, sd * 9.0)) - 0.5;
        let mut t_upper: Vec3f = vec3f(
            0.57 - 0.09 * g.cell_v,
            0.57 - 0.09 * g.cell_v,
            0.57 - 0.09 * g.cell_v,
        );
        t_upper = t_upper
            + pore_zone
                * (0.86 * (*ctx).g_k * att_p * cc_capital * (pm - g.fill)
                    + 0.12 * att_p * bc_capital * (ring - 0.28));
        t_upper = t_upper
            + vec3f(
                pore_zone * 0.16 * speck * (1.0 - att_p),
                pore_zone * 0.16 * speck * (1.0 - att_p),
                pore_zone * 0.16 * speck * (1.0 - att_p),
            );
        t_upper = min(t_upper, vec3f(1.02, 1.02, 1.02));
        let mott: f32 = select(
            0.0,
            wp_diatom_vnoise(q / (r_upper * 0.016) + vec2f(sd * 40.0, sd * 40.0)) - 0.5,
            g.hyal > 0.0,
        );
        t_upper = mix(
            t_upper,
            vec3f(
                0.86 + 0.14 * mott * (0.4 + 0.6 * att_p),
                0.86 + 0.14 * mott * (0.4 + 0.6 * att_p),
                0.86 + 0.14 * mott * (0.4 + 0.6 * att_p),
            ),
            vec3f(g.hyal, g.hyal, g.hyal),
        );
        let ed_c: f32 = blur_c + aa;
        let costa: f32 = smoothstep(ed_c, -ed_c, g.d_co) * g.hw_co / (g.hw_co + 0.6 * blur_c);
        let rung: f32 =
            g.rung_on * smoothstep(ed_c, -ed_c, g.d_ru) * g.hw_ru / (g.hw_ru + 0.6 * blur_c);
        let c_fr: f32 = exp(-wp_diatom_sq((g.d_co - blur_c - 1.2) / (1.0 + blur_c)));
        t_upper = t_upper * (1.0 - 0.92 * costa - 0.72 * rung * (1.0 - costa));
        t_upper = t_upper + 0.12 * bc_c_capital * c_fr;
        let rim_b: f32 = smoothstep(g.rim_w + 1.0, g.rim_w - 1.5, e);
        let mut slit: f32 = 0.0;
        if rim_b > 0.0 {
            let n_t: f32 = floor(WP_DIATOM_TAU * r_upper / (0.0105 * r_upper));
            let tk: f32 = fract((th / WP_DIATOM_TAU + 0.5) * n_t);
            slit = (1.0 - smoothstep(0.16, 0.3, abs(tk - 0.5)))
                * smoothstep(0.12, 0.32, e / g.rim_w)
                * smoothstep(0.95, 0.7, e / g.rim_w);
        }
        let rim_t: f32 = 0.34 + 0.38 * slit * att_p * (0.5 + 0.5 * cc_capital.y);
        t_upper = mix(
            t_upper,
            vec3f(rim_t, rim_t, rim_t),
            vec3f(rim_b, rim_b, rim_b),
        );
        let wall: f32 = exp(-wp_diatom_sq(e / (1.1 * sc + blur_e)));
        t_upper = t_upper * (1.0 - 0.55 * wall);
        let halo: f32 = exp(-wp_diatom_sq((e + 2.5 * sc + blur_e) / (1.6 * sc + blur_e)));
        if mode_upper != 1 {
            o.t_upper = mix(
                vec3f(1.0, 1.0, 1.0) + 0.16 * bc_capital * halo,
                t_upper,
                vec3f(in_v, in_v, in_v),
            );
        }
        if mode_upper != 1 {
            o.h_upper =
                in_v * (1.0 + 0.35 * rim_b) - 0.55 * pm * pore_zone + 0.9 * costa + 0.45 * rung;
            return o;
        }
        let mut g_upper: Vec3f = vec3f(0.05, 0.05, 0.05);
        let cm_capital: f32 = dot(cc_capital, vec3f(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0));
        g_upper = g_upper
            + pore_zone
                * (vec3f(
                    0.07 + 0.04 * g.cell_v,
                    0.07 + 0.04 * g.cell_v,
                    0.07 + 0.04 * g.cell_v,
                ) + (*ctx).g_k
                    * att_p
                    * (0.62
                        * ring
                        * max(
                            mix(
                                vec3f(cm_capital, cm_capital, cm_capital),
                                cc_capital,
                                vec3f(0.5, 0.5, 0.5),
                            ),
                            vec3f(0.0, 0.0, 0.0),
                        )
                        + vec3f(
                            0.5 * pm * max(-cm_capital, 0.0),
                            0.5 * pm * max(-cm_capital, 0.0),
                            0.5 * pm * max(-cm_capital, 0.0),
                        ))
                    + vec3f(
                        0.07 * speck * (1.0 - att_p),
                        0.07 * speck * (1.0 - att_p),
                        0.07 * speck * (1.0 - att_p),
                    ));
        g_upper = mix(
            g_upper,
            vec3f(0.05 + 0.05 * mott, 0.05 + 0.05 * mott, 0.05 + 0.05 * mott),
            vec3f(g.hyal, g.hyal, g.hyal),
        );
        g_upper = g_upper
            + (vec3f(0.46 * costa, 0.46 * costa, 0.46 * costa)
                + 0.32 * c_fr * (vec3f(0.6, 0.6, 0.6) + 0.4 * bc_c_capital) * (1.0 - costa)
                + vec3f(0.24 * rung, 0.24 * rung, 0.24 * rung));
        g_upper = mix(
            g_upper,
            vec3f(
                0.26 + 0.45 * slit * att_p,
                0.26 + 0.45 * slit * att_p,
                0.26 + 0.45 * slit * att_p,
            ),
            vec3f(rim_b, rim_b, rim_b),
        );
        g_upper = g_upper + vec3f(0.85 * wall, 0.85 * wall, 0.85 * wall);
        o.g_upper =
            g_upper * in_v + 0.35 * halo * max(bc_capital, vec3f(0.0, 0.0, 0.0)) * (1.0 - in_v);
        return o;
    }

    pub fn wp_diatom(fc: Vec2f) -> Vec4f {
        let mut result = vec4f(0.0, 0.0, 0.0, 0.0);
        let mut state = WpDiatomContext {
            g_aa: 0.0,
            g_dof: 0.0,
            g_ca: 0.0,
            g_k: 0.0,
        };
        let ctx = &mut state;
        let dpr: f32 = max(get!(BLUR_LOCALS).wp_frame.z, 0.25);
        let res: Vec2f = get!(BLUR_LOCALS).wp_frame.xy() / dpr;
        let p: Vec2f = fc / dpr;
        (*ctx).g_aa = 0.7 / dpr;
        let sd: f32 = get!(BLUR_LOCALS).wp_p1.z;
        let mode: i32 = get!(BLUR_LOCALS).wp_p0.y as i32;
        let radial: bool = get!(BLUR_LOCALS).wp_p0.z as i32 == 0;
        (*ctx).g_dof = mix(0.7, 1.45, get!(BLUR_LOCALS).wp_p1.y);
        (*ctx).g_k = mix(1.18, 0.84, get!(BLUR_LOCALS).wp_p1.y);
        (*ctx).g_ca = 0.03;
        let mut cr: Vec4f = get!(BLUR_LOCALS).wp_rect / dpr;
        if cr.z < 1.0 || cr.w < 1.0 {
            let w: f32 = min(760.0, res.x - 40.0);
            let hh: f32 = min(420.0, res.y * 0.5);
            cr = vec4f((res.x - w) * 0.5, res.y - 56.0 - hh, w, hh);
        }
        let head_h: f32 = min(cr.w * 0.36, 150.0);
        let _comp_top: f32 = select(130.0, 112.0, res.x < 560.0);
        let _h0: f32 = wp_diatom_hash11(sd * 17.31 + 0.13);
        let h1: f32 = wp_diatom_hash11(sd * 29.7 + 0.71);
        let h2: f32 = wp_diatom_hash11(sd * 41.3 + 0.37);
        let h3: f32 = wp_diatom_hash11(sd * 53.9 + 0.91);
        let side: f32 = select(-1.0, 1.0, get!(BLUR_LOCALS).wp_p0.x as i32 == 0);
        let r0_upper: f32 = get!(BLUR_LOCALS).wp_p1.x * min(0.8 * res.y, 0.92 * res.x + 60.0);
        let mut vc: Vec2f = vec2f(0.0, 0.0);
        let portrait: f32 = step(res.x, res.y);
        let inset: f32 = mix(0.06 + 0.1 * h1, 0.02 + 0.06 * h1, portrait);
        vc.x = select(inset * r0_upper, res.x - inset * r0_upper, side > 0.0);
        let y_hi: f32 = cr.y + cr.w - head_h - 34.0 - r0_upper;
        vc.y = max(
            min(0.03 * r0_upper + 0.05 * r0_upper * h2, y_hi),
            -0.35 * r0_upper,
        );
        let mut acc: WpDiatomPx = WpDiatomPx {
            t_upper: vec3f(0.0, 0.0, 0.0),
            g_upper: vec3f(0.0, 0.0, 0.0),
            h_upper: 0.0,
            pig: 0.0,
        };
        acc.t_upper = vec3f(1.0, 1.0, 1.0);
        acc.g_upper = vec3f(0.0, 0.0, 0.0);
        acc.h_upper = 0.0;
        acc.pig = 0.0;
        {
            let vrot: f32 = h3 * WP_DIATOM_TAU;
            let cv: f32 = wp_diatom_fcos(vrot);
            let sv: f32 = wp_diatom_fsin(vrot);
            let rv: Mat2x2f = mat2x2f(vec2f(cv, -sv), vec2f(sv, cv));
            let mut q: Vec2f = rv * (p - vc);
            q.y = q.y / 0.968;
            let rmax_capital: f32 = r0_upper * 1.0125;
            let bd_s: Vec2f = wp_diatom_rot((h2 - 0.5) * 0.6) * normalize(vec2f(-side, 0.3));
            let bd: Vec2f = rv * bd_s;
            let bp: Vec2f = vec2f(-bd.y, bd.x);
            let p0_upper: Vec2f = bd * 0.8;
            let fsg: f32 = select(-1.0, 1.0, h1 < 0.5);
            let rf: Mat2x2f = mat2x2f(
                vec2f(wp_diatom_fcos(0.24), -fsg * wp_diatom_fsin(0.24)),
                vec2f(fsg * wp_diatom_fsin(0.24), wp_diatom_fcos(0.24)),
            );
            let qfm: Vec2f = rf * (q / rmax_capital - p0_upper - bd * 0.085 - bp * 0.03) + p0_upper;
            if dot(q, q) < wp_diatom_sq(1.035 * rmax_capital) || dot(qfm, qfm) < wp_diatom_sq(1.05)
            {
                let focus: f32 = get!(BLUR_LOCALS).wp_p0.w;
                let rv_capital: f32 = r0_upper * (1.0 + 0.012 * focus / 1.65);
                let qn: Vec2f = q / rv_capital;
                let qf: Vec2f = rf * (qn - p0_upper - bd * 0.085 - bp * 0.03) + p0_upper;
                let r_m: f32 = length(qn);
                let r_f: f32 = length(qf);
                let th_b: f32 = atan2(bd.y, bd.x);
                let th_m: f32 = atan2(qn.y, qn.x);
                let c_m: f32 = select(
                    1.0,
                    select(
                        -1.0,
                        wp_diatom_chip_field(r_m, th_m, th_b),
                        dot(qn, bd) > 0.54 * r_m,
                    ),
                    r_m < 1.03,
                );
                let in_main: bool = r_m < 1.03 && c_m < 0.006;
                let mut th_f: f32 = 0.0;
                let mut c_f: f32 = -1.0;
                if !in_main && r_f < 1.03 {
                    th_f = atan2(qf.y, qf.x);
                    c_f = wp_diatom_chip_field(r_f, th_f, th_b);
                }
                let in_frag: bool = !in_main && r_f < 1.03 && c_f > -0.006;
                if in_main || in_frag {
                    let tdir: Vec2f =
                        rv * normalize(res * vec2f(0.5, 0.55) - vc + vec2f(1e-3, 1e-3));
                    let qq: Vec2f = select(qf, qn, in_main);
                    let cc: f32 = select(c_f, c_m, in_main);
                    let th: f32 = select(th_f, th_m, in_main);
                    let r: f32 = select(r_f, r_m, in_main);
                    let depth: f32 = 1.5 * (1.0 - r * r) - 0.75
                        + 1.35 * dot(qq, tdir)
                        + select(0.95, 0.0, in_main);
                    let z_p: f32 = focus - depth;
                    let zs0: f32 = z_p * (*ctx).g_dof;
                    let b0_upper: f32 = wp_diatom_fsin(1.35 * zs0) * exp(-0.22 * zs0 * zs0);
                    let mut v: WpDiatomPx = WpDiatomPx {
                        t_upper: vec3f(0.0, 0.0, 0.0),
                        g_upper: vec3f(0.0, 0.0, 0.0),
                        h_upper: 0.0,
                        pig: 0.0,
                    };
                    if mode == 0 {
                        v = wp_diatom_valve_shade(
                            ctx,
                            qq * rv_capital,
                            rv_capital,
                            z_p,
                            sd,
                            radial,
                            0,
                            th,
                        );
                    } else {
                        if mode == 1 {
                            v = wp_diatom_valve_shade(
                                ctx,
                                qq * rv_capital,
                                rv_capital,
                                z_p,
                                sd,
                                radial,
                                1,
                                th,
                            );
                        } else {
                            v = wp_diatom_valve_shade(
                                ctx,
                                qq * rv_capital,
                                rv_capital,
                                z_p,
                                sd,
                                radial,
                                2,
                                th,
                            );
                        }
                    }
                    let eb: f32 = select(cc, -cc, in_main) * rv_capital;
                    let blur_e: f32 = (0.5 + 2.4 * abs(z_p * (*ctx).g_dof)) * rv_capital / 468.0;
                    let keep: f32 = smoothstep(-(*ctx).g_aa, (*ctx).g_aa, eb) * step(r, 1.0);
                    let bw: f32 = exp(-wp_diatom_sq(eb / (1.2 + blur_e)));
                    v.t_upper = mix(
                        vec3f(1.0, 1.0, 1.0)
                            + vec3f(
                                0.1 * b0_upper
                                    * exp(-wp_diatom_sq((eb + 2.5 + blur_e) / (1.6 + blur_e))),
                                0.1 * b0_upper
                                    * exp(-wp_diatom_sq((eb + 2.5 + blur_e) / (1.6 + blur_e))),
                                0.1 * b0_upper
                                    * exp(-wp_diatom_sq((eb + 2.5 + blur_e) / (1.6 + blur_e))),
                            ),
                        v.t_upper,
                        vec3f(keep, keep, keep),
                    );
                    v.t_upper = v.t_upper * (1.0 - 0.5 * bw * step(r, 1.0));
                    v.g_upper = v.g_upper * keep
                        + vec3f(
                            0.9 * bw * step(r, 1.0),
                            0.9 * bw * step(r, 1.0),
                            0.9 * bw * step(r, 1.0),
                        );
                    v.h_upper = v.h_upper * keep;
                    acc.t_upper = acc.t_upper * v.t_upper;
                    acc.g_upper = acc.g_upper + v.g_upper;
                    acc.h_upper = acc.h_upper + v.h_upper;
                }
                let c0: Vec2f = wp_diatom_rot(-0.12) * bd * 0.6;
                let cdir: Vec2f = normalize(-bd + bp * 0.3);
                let w: Vec2f = qn - c0;
                let wm: Vec2f = w - cdir * 0.15;
                if in_main && dot(wm, wm) < 0.031 {
                    let s: f32 = clamp(dot(w, cdir), 0.0, 0.3);
                    let dcr: f32 = length(
                        w - cdir * s
                            + bp * (0.012 * wp_diatom_tri(s * 11.0)
                                + 0.004 * wp_diatom_tri(s * 37.0)),
                    ) * rv_capital;
                    let crack: f32 = exp(-wp_diatom_sq(dcr / (0.7 + (*ctx).g_aa)))
                        * step(r_m, 0.97)
                        * smoothstep(0.3, 0.16, s);
                    acc.t_upper = acc.t_upper * (1.0 - 0.45 * crack);
                    acc.g_upper = acc.g_upper + vec3f(0.5 * crack, 0.5 * crack, 0.5 * crack);
                }
            }
        }
        let mut dust: f32 = 0.0;
        let mut dust_rim: f32 = 0.0;
        {
            let hh: Vec4f = texture_load(
                BLUR_TEXTURE,
                wp_bitand2i(vec2i((sd * 613.0) as i32, 203), vec2i(255, 255)),
                0i32,
            );
            let c1: Vec2f = res * vec2f(0.08 + 0.84 * hh.x, 0.12 + 0.76 * hh.y);
            let c2: Vec2f = res * vec2f(0.08 + 0.84 * hh.z, 0.12 + 0.76 * hh.w);
            let s9: f32 = res.y / 900.0;
            let rr: Vec2f = (vec2f(30.0, 52.0) + 30.0 * hh.xz()) * s9;
            let d: Vec2f = vec2f(length(p - c1), length(p - c2));
            let x: Vec2f = max(
                vec2f(1.0, 1.0) - (d - rr + vec2f(2.0, 2.0)) * (d - rr + vec2f(2.0, 2.0)) / 36.0,
                vec2f(0.0, 0.0),
            );
            let y: Vec2f = max(
                vec2f(1.0, 1.0) - (d - rr * 0.55) * (d - rr * 0.55) / 64.0,
                vec2f(0.0, 0.0),
            );
            dust = dot(
                clamp(
                    (rr + vec2f(5.0, 5.0) - d) / 10.0,
                    vec2f(0.0, 0.0),
                    vec2f(1.0, 1.0),
                ),
                vec2f(1.0, 1.0),
            );
            dust_rim = dot(x * x + 0.25 * y * y, vec2f(1.0, 1.0));
        }
        let uvn: Vec2f = (p - res * vec2f(0.42, 0.6)) / res.y;
        let illum: f32 = 1.0 - 0.065 * smoothstep(0.1, 1.15, length(uvn * vec2f(0.85, 1.0)));
        let grain: f32 = (wp_noise(fc / 256.0).z - 0.5) / 255.0;
        let mut col: Vec3f = vec3f(0.0, 0.0, 0.0);
        if mode == 1 {
            let bg: Vec3f =
                vec3f(0.010, 0.012, 0.018) + vec3f(0.004, 0.006, 0.012) * (1.0 - length(uvn));
            let tint: Vec3f = vec3f(0.72, 0.84, 1.0);
            let mut g_upper: Vec3f = acc.g_upper * (*ctx).g_k;
            g_upper = g_upper + vec3f(0.012, 0.015, 0.022) * (dust * 0.5 + dust_rim);
            col = bg + g_upper * tint * mix(1.0, 0.78, get!(BLUR_LOCALS).wp_env.x * 0.0);
            col = col + acc.pig * vec3f(0.30, 0.20, 0.05) * 0.8;
            col = vec3f(1.0, 1.0, 1.0) - exp(-col * 1.25);
            if get!(BLUR_LOCALS).wp_env.x < 0.5 {
                let paper: Vec3f = vec3f(0.953, 0.957, 0.962) * illum;
                col = paper * (vec3f(1.0, 1.0, 1.0) - 0.9 * (col - bg) * vec3f(1.05, 1.0, 0.92));
            }
        } else {
            if mode == 2 {
                let slate: Vec3f = mix(
                    vec3f(0.435, 0.514, 0.651),
                    vec3f(0.19, 0.225, 0.29),
                    vec3f(
                        get!(BLUR_LOCALS).wp_env.x,
                        get!(BLUR_LOCALS).wp_env.x,
                        get!(BLUR_LOCALS).wp_env.x,
                    ),
                );
                let hs_capital: f32 = acc.h_upper * 1.1;
                let rel: f32 = (dpdx(hs_capital) - dpdy(hs_capital)) * 0.7071 * dpr;
                col = slate
                    * illum
                    * (1.0 + 1.25 * rel)
                    * mix(vec3f(1.0, 1.0, 1.0), acc.t_upper, vec3f(0.22, 0.22, 0.22));
                col = mix(
                    col,
                    vec3f(0.69, 0.54, 0.165)
                        * (0.75 + 0.5 * illum)
                        * mix(1.0, 0.72, get!(BLUR_LOCALS).wp_env.x)
                        * (1.0 + 0.8 * rel),
                    vec3f(acc.pig * 0.8, acc.pig * 0.8, acc.pig * 0.8),
                );
                col = col * (1.0 - 0.03 * dust + 0.02 * dust_rim);
            } else {
                let mut field: Vec3f = mix(
                    vec3f(0.965, 0.965, 0.957),
                    vec3f(0.145, 0.146, 0.15),
                    vec3f(
                        get!(BLUR_LOCALS).wp_env.x,
                        get!(BLUR_LOCALS).wp_env.x,
                        get!(BLUR_LOCALS).wp_env.x,
                    ),
                ) * illum;
                field = field * (vec3f(1.0, 1.0, 1.0) + 0.012 * vec3f(-uvn.x, 0.0, uvn.x));
                let mut t_upper: Vec3f = acc.t_upper;
                t_upper = t_upper * (1.0 - 0.028 * dust + 0.018 * dust_rim);
                col = field * t_upper;
                col = col
                    * mix(
                        vec3f(1.0, 1.0, 1.0),
                        vec3f(0.93, 0.8, 0.5),
                        vec3f(acc.pig * 0.7, acc.pig * 0.7, acc.pig * 0.7),
                    );
            }
        }
        col = col + vec3f(grain, grain, grain);
        result = vec4f(
            clamp(col, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0)).x,
            clamp(col, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0)).y,
            clamp(col, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0)).z,
            1.0,
        );
        result
    }

    // butler.diatom/overlay.frag
    #[derive(Clone, Copy, Wgsl)]
    pub struct WpDiatomOverlayContext {
        pub g_aa: f32,
        pub g_dof: f32,
        pub g_k: f32,
        pub a_t: Vec3f,
        pub a_g: f32,
        pub a_h: f32,
        pub sh_th_b_capital: f32,
    }

    pub const WP_DIATOM_OVERLAY_TAU: f32 = 6.28318530718;

    pub const WP_DIATOM_OVERLAY_T_PERIOD: f32 = 240.0;

    pub fn wp_diatom_overlay_fsin(x: f32) -> f32 {
        let mut x = x;
        x = x * 0.15915494;
        x = x - floor(x + 0.5);
        let y: f32 = x * (8.0 - 16.0 * abs(x));
        return 0.225 * (y * abs(y) - y) + y;
    }

    pub fn wp_diatom_overlay_fcos(x: f32) -> f32 {
        return wp_diatom_overlay_fsin(x + 1.5707963);
    }

    pub fn wp_diatom_overlay_sq(x: f32) -> f32 {
        return x * x;
    }

    pub fn wp_diatom_overlay_sat(x: f32) -> f32 {
        return clamp(x, 0.0, 1.0);
    }

    pub fn wp_diatom_overlay_hash11(p: f32) -> f32 {
        let mut p = p;
        p = fract(p * 0.1031);
        p = p * (p + 33.33);
        p = p * (p + p);
        return fract(p);
    }

    pub fn wp_diatom_overlay_rot(a: f32) -> Mat2x2f {
        let c: f32 = wp_diatom_overlay_fcos(a);
        let s: f32 = wp_diatom_overlay_fsin(a);
        return mat2x2f(vec2f(c, s), vec2f(-s, c));
    }

    pub fn wp_diatom_overlay_sd_round_box(p: Vec2f, b: Vec2f, r: f32) -> f32 {
        let q: Vec2f = abs(p) - b + vec2f(r, r);
        return length(max(q, vec2f(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
    }

    pub fn wp_diatom_overlay_tri(x: f32) -> f32 {
        return abs(fract(x) - 0.5);
    }

    pub fn wp_diatom_overlay_lb32(x: u32) -> u32 {
        let mut x = x;
        x = x ^ (x >> 16u32);
        x = x * 0x7feb352du32;
        x = x ^ (x >> 15u32);
        x = x * 0x846ca68bu32;
        x = x ^ (x >> 16u32);
        return x;
    }

    pub fn wp_diatom_overlay_hex_nearest(p: Vec2f) -> Vec2f {
        let s: Vec2f = vec2f(1.0, 1.7320508);
        let hc: Vec4f = floor(
            vec4f(p.x, p.y, (p - vec2f(0.5, 1.0)).x, (p - vec2f(0.5, 1.0)).y)
                / vec4f(s.x, s.y, s.x, s.y),
        ) + vec4f(0.5, 0.5, 0.5, 0.5);
        let h: Vec4f = vec4f(
            (p - hc.xy() * s).x,
            (p - hc.xy() * s).y,
            (p - (hc.zw() + vec2f(0.5, 0.5)) * s).x,
            (p - (hc.zw() + vec2f(0.5, 0.5)) * s).y,
        );
        return select(h.zw(), h.xy(), dot(h.xy(), h.xy()) < dot(h.zw(), h.zw()));
    }

    pub fn wp_diatom_overlay_shard(
        ctx: ptr!(function, WpDiatomOverlayContext),
        p: Vec2f,
        c: Vec2f,
        size: f32,
        ang: f32,
        depth: f32,
        focus: f32,
        pitch: f32,
        hs: f32,
    ) {
        let u: Vec2f = wp_diatom_overlay_rot(-ang) * (p - c);
        let el: f32 = 1.25 + 0.6 * hs;
        let uq: Vec2f = vec2f(u.x / el, u.y);
        let mut d_p: f32 = -1e3;
        let mut v0: Vec2f = vec2f(0.0, 0.0);
        for k in 0..6 {
            let kk: f32 = (k % 5) as f32;
            let a: f32 = (k as f32 + 0.6 * (wp_diatom_overlay_hash11(hs * 7.1 + kk) - 0.5))
                * WP_DIATOM_OVERLAY_TAU
                / 5.0;
            let rr: f32 = size * (0.5 + 0.6 * wp_diatom_overlay_hash11(hs * 3.3 + kk * 1.7));
            let v: Vec2f = rr * vec2f(wp_diatom_overlay_fcos(a), wp_diatom_overlay_fsin(a));
            if k > 0 {
                let n: Vec2f = normalize(vec2f(v.y - v0.y, v0.x - v.x));
                d_p = max(d_p, dot(uq - v0, n));
            }
            v0 = v;
        }
        d_p = d_p + 0.9 * wp_diatom_overlay_tri(dot(u, vec2f(0.37, 0.93)) * 0.21 + hs);
        let zs: f32 = (focus - depth) * (*ctx).g_dof;
        let sc: f32 = pitch / 15.0;
        let c_upper: f32 = wp_diatom_overlay_fcos(1.35 * zs) * exp(-0.1 * zs * zs);
        let b_upper: f32 = wp_diatom_overlay_fsin(1.35 * zs) * exp(-0.22 * zs * zs);
        let blur_p: f32 = pitch * (0.06 + 0.14 * abs(zs));
        let att_p: f32 = exp(-2.2 * wp_diatom_overlay_sq(blur_p / pitch));
        let blur_e: f32 = (0.5 + 2.4 * abs(zs)) * sc;
        let e: f32 = -d_p;
        let in_v: f32 = smoothstep(-(*ctx).g_aa - 0.5 * blur_e, (*ctx).g_aa + 0.5 * blur_e, e);
        let h: Vec2f = wp_diatom_overlay_hex_nearest(u / pitch + vec2f(hs * 9.0, hs * 9.0)) * pitch;
        let rp: f32 = 0.34 * pitch;
        let se: f32 = min(blur_p, 0.55 * rp) + (*ctx).g_aa;
        let d_a: f32 = length(h);
        let pm: f32 = 1.0 - smoothstep(rp - se, rp + se, d_a);
        let ring: f32 = exp(-wp_diatom_overlay_sq((d_a - rp) / (0.28 * rp + se)));
        let hw: f32 = 0.27 * pitch;
        let blur_c: f32 = (0.4 + 1.5 * abs(zs + 0.75 * (*ctx).g_dof)) * sc;
        let costa: f32 = smoothstep(
            blur_c + (*ctx).g_aa,
            -blur_c - (*ctx).g_aa,
            abs(u.y - size * (hs - 0.5) * 0.6) - hw,
        ) * hw
            / (hw + 0.6 * blur_c);
        let mut tin_capital: f32 = 0.6
            + 0.8 * (*ctx).g_k * att_p * c_upper * (pm - 0.38)
            + 0.1 * att_p * b_upper * (ring - 0.28);
        tin_capital = tin_capital * (1.0 - 0.85 * costa);
        let wall: f32 = exp(-wp_diatom_overlay_sq(e / (1.1 * sc + blur_e)));
        let halo: f32 = exp(-wp_diatom_overlay_sq(
            (e + 2.5 * sc + blur_e) / (1.6 * sc + blur_e),
        ));
        let t_upper: f32 = mix(
            1.0 + 0.03 * b_upper * halo,
            tin_capital * (1.0 - 0.38 * wall),
            in_v,
        );
        (*ctx).a_t = (*ctx).a_t * t_upper;
        (*ctx).a_g = (*ctx).a_g
            + (in_v
                * (0.07
                    + (*ctx).g_k
                        * att_p
                        * (0.55 * ring * max(c_upper, 0.0) + 0.45 * pm * max(-c_upper, 0.0))
                    + 0.45 * costa)
                + 0.6 * wall * in_v
                + 0.2 * halo * max(b_upper, 0.0) * (1.0 - in_v));
        (*ctx).a_h = (*ctx).a_h + in_v * (1.0 - 0.55 * pm + 0.9 * costa);
    }

    pub fn wp_diatom_overlay_shard_slot(
        ctx: ptr!(function, WpDiatomOverlayContext),
        p: Vec2f,
        fi: f32,
        dth: f32,
        rho: f32,
        size_k: f32,
        dz: f32,
        vc: Vec2f,
        rv_capital: f32,
        r0_upper: f32,
        side: f32,
        sd: f32,
        tau: f32,
        focus: f32,
        res: Vec2f,
        cr: Vec4f,
    ) {
        let hs: f32 = wp_diatom_overlay_hash11(sd * 71.3 + fi * 13.7);
        let pitch: f32 = 0.0192 * r0_upper;
        let size: f32 = pitch * size_k;
        let depth: f32 = focus + dz;
        let a: f32 = dth + 0.06 * (hs - 0.5);
        let c_a: Vec2f = vc
            + rv_capital
                * rho
                * vec2f(
                    wp_diatom_overlay_fcos((*ctx).sh_th_b_capital - side * a),
                    wp_diatom_overlay_fsin((*ctx).sh_th_b_capital - side * a),
                );
        let c_b: Vec2f = vc
            + rv_capital
                * rho
                * vec2f(
                    wp_diatom_overlay_fcos((*ctx).sh_th_b_capital + side * a),
                    wp_diatom_overlay_fsin((*ctx).sh_th_b_capital + side * a),
                );
        let cr_c: Vec2f = cr.xy() + 0.5 * cr.zw();
        let cr_b: Vec2f = 0.5 * cr.zw() + vec2f(36.0, 36.0) + vec2f(size, size);
        let m: Vec2f = 0.5 * res - vec2f(size, size);
        let ok_a: f32 = step(0.0, wp_diatom_overlay_sd_round_box(c_a - cr_c, cr_b, 22.0))
            * step(
                max(
                    abs(c_a.x - 0.5 * res.x) - m.x,
                    abs(c_a.y - 0.5 * res.y) - m.y,
                ),
                0.0,
            );
        let ok_b: f32 = step(0.0, wp_diatom_overlay_sd_round_box(c_b - cr_c, cr_b, 22.0))
            * step(
                max(
                    abs(c_b.x - 0.5 * res.x) - m.x,
                    abs(c_b.y - 0.5 * res.y) - m.y,
                ),
                0.0,
            );
        let c: Vec2f = select(c_b, c_a, ok_a > 0.0)
            + vec2f(
                wp_diatom_overlay_fsin(tau + fi * 2.1),
                0.8 * wp_diatom_overlay_fsin(2.0 * tau + fi * 1.3),
            ) * (5.0 + 3.0 * hs);
        let reach: f32 = 1.25 * size + 3.0 * (0.35 + 2.4 * abs(dz * (*ctx).g_dof)) + 8.0;
        if ok_a + ok_b > 0.0 && dot(p - c, p - c) < reach * reach {
            wp_diatom_overlay_shard(
                ctx,
                p,
                c,
                size,
                hs * WP_DIATOM_OVERLAY_TAU + 0.06 * wp_diatom_overlay_fsin(tau + fi),
                depth,
                focus,
                pitch,
                hs,
            );
        }
    }

    pub fn wp_diatom_overlay(fc: Vec2f) -> Vec4f {
        let mut result = vec4f(0.0, 0.0, 0.0, 0.0);
        let mut state = WpDiatomOverlayContext {
            g_aa: 0.0,
            g_dof: 0.0,
            g_k: 0.0,
            a_t: vec3f(1.0, 1.0, 1.0),
            a_g: 0.0,
            a_h: 0.0,
            sh_th_b_capital: 0.0,
        };
        let ctx = &mut state;
        let dpr: f32 = max(get!(BLUR_LOCALS).wp_frame.z, 0.25);
        let res: Vec2f = get!(BLUR_LOCALS).wp_frame.xy() / dpr;
        let p: Vec2f = fc / dpr;
        (*ctx).g_aa = 0.7 / dpr;
        let t: f32 = wp_mod(get!(BLUR_LOCALS).wp_frame.w, WP_DIATOM_OVERLAY_T_PERIOD);
        let tau: f32 = WP_DIATOM_OVERLAY_TAU * t / WP_DIATOM_OVERLAY_T_PERIOD;
        let sd: f32 = get!(BLUR_LOCALS).wp_p1.z;
        let mode: i32 = get!(BLUR_LOCALS).wp_p0.y as i32;
        (*ctx).g_dof = mix(0.7, 1.45, get!(BLUR_LOCALS).wp_p1.y);
        (*ctx).g_k = mix(1.18, 0.84, get!(BLUR_LOCALS).wp_p1.y);
        let focus: f32 = get!(BLUR_LOCALS).wp_p0.w;
        let mut cr: Vec4f = get!(BLUR_LOCALS).wp_rect / dpr;
        if cr.z < 1.0 || cr.w < 1.0 {
            let w: f32 = min(760.0, res.x - 40.0);
            let hh: f32 = min(420.0, res.y * 0.5);
            cr = vec4f((res.x - w) * 0.5, res.y - 56.0 - hh, w, hh);
        }
        let head_h: f32 = min(cr.w * 0.36, 150.0);
        let h1: f32 = wp_diatom_overlay_hash11(sd * 29.7 + 0.71);
        let h2: f32 = wp_diatom_overlay_hash11(sd * 41.3 + 0.37);
        let side: f32 = select(-1.0, 1.0, get!(BLUR_LOCALS).wp_p0.x as i32 == 0);
        let r0_upper: f32 = get!(BLUR_LOCALS).wp_p1.x * min(0.8 * res.y, 0.92 * res.x + 60.0);
        let mut vc: Vec2f = vec2f(0.0, 0.0);
        let portrait: f32 = step(res.x, res.y);
        let inset: f32 = mix(0.06 + 0.1 * h1, 0.02 + 0.06 * h1, portrait);
        vc.x = select(inset * r0_upper, res.x - inset * r0_upper, side > 0.0);
        let y_hi: f32 = cr.y + cr.w - head_h - 34.0 - r0_upper;
        vc.y = max(
            min(0.03 * r0_upper + 0.05 * r0_upper * h2, y_hi),
            -0.35 * r0_upper,
        );
        let rv_capital: f32 = r0_upper * (1.0 + 0.012 * focus / 1.65);
        let head_c: Vec2f = vec2f(cr.x + cr.z * 0.5, cr.y + cr.w - head_h * 0.5);
        let head_b: Vec2f = vec2f(cr.z * 0.5, head_h * 0.5) + vec2f(14.0, 14.0);
        {
            let cs: f32 = 58.0 * r0_upper / 468.0 + 18.0;
            let cid: Vec2f = floor(p / cs);
            let tc: Vec2i = wp_bitand2i(
                vec2i(cid.x as i32, cid.y as i32) + vec2i((sd * 997.0) as i32, 71),
                vec2i(255, 255),
            );
            let hd: Vec4f = texture_load(BLUR_TEXTURE, tc, 0i32);
            let dv0: f32 = length((cid + vec2f(0.5, 0.5)) * cs - vc) / rv_capital;
            let dm: f32 = 0.33 * cs / rv_capital;
            let dx: f32 = max(max(dv0 - dm - 1.08, 1.08 - dv0 - dm), 0.0);
            if hd.x
                < 0.04 + 0.5 * exp(-wp_diatom_overlay_sq(dx / 0.22)) + 0.12 * step(dv0 - dm, 1.0)
            {
                let j: Vec2f = texture_load(
                    BLUR_TEXTURE,
                    wp_bitand2i(tc + vec2i(0, 131), vec2i(255, 255)),
                    0i32,
                )
                .xy()
                    - vec2f(0.5, 0.5);
                let depth: f32 = (fract(hd.z * 7.0) - 0.5) * 4.4;
                let size: f32 =
                    mix(0.9, 4.2, wp_diatom_overlay_sq(hd.y)) * (0.6 + 0.4 * r0_upper / 468.0);
                let z: f32 = (focus - depth) * (*ctx).g_dof;
                let bl: f32 = 0.35 + 1.9 * abs(z);
                let lift: f32 = 0.55 + 0.45 * (depth + 2.2) / 4.4;
                let creep: Vec2f = vec2f(
                    wp_diatom_overlay_fsin(tau),
                    0.7 * wp_diatom_overlay_fsin(2.0 * tau + 1.3),
                ) * (0.06 * cs)
                    * lift
                    + vec2f(
                        wp_diatom_overlay_fsin(2.0 * tau + 6.28 * hd.w),
                        wp_diatom_overlay_fcos(3.0 * tau + 6.28 * hd.z),
                    ) * (0.012 * cs);
                let cc: Vec2f = (cid + vec2f(0.5, 0.5) + j * 0.2) * cs + creep;
                let reach: f32 = 2.8 * size + max(3.0 * (0.9 + bl), 1.7 * bl + 5.2) + 2.0;
                if dot(p - cc, p - cc) < reach * reach {
                    let dv: f32 = length(cc - vc) / rv_capital;
                    let sd_head: f32 = wp_diatom_overlay_sd_round_box(p - head_c, head_b, 22.0);
                    let dens: f32 = (0.04
                        + 0.5 * exp(-wp_diatom_overlay_sq((dv - 1.08) / 0.22))
                        + 0.12 * step(dv, 1.0))
                        * smoothstep(0.0, 30.0, sd_head);
                    if hd.x < dens {
                        let rel: Vec2f = wp_diatom_overlay_rot(
                            hd.z * 6.28 + 0.05 * wp_diatom_overlay_fsin(tau + hd.w * 6.28),
                        ) * (p - cc);
                        let typ: f32 = fract(hd.y * 5.3);
                        let dsh: f32 = select(
                            select(
                                length(vec2f(max(abs(rel.x) - size * 1.3, 0.0), rel.y))
                                    - size * 0.45,
                                wp_diatom_overlay_sd_round_box(
                                    rel,
                                    vec2f(size * 1.9, size * 0.75),
                                    size * 0.3,
                                ) + 0.5 * size * wp_diatom_overlay_tri(rel.x * 0.4),
                                typ < 0.85,
                            ),
                            length(rel) - size,
                            typ < 0.55,
                        );
                        let sz: f32 = select(size * 1.2, size, typ < 0.55);
                        let mut cov: f32 = smoothstep(bl + (*ctx).g_aa, -bl - (*ctx).g_aa, dsh)
                            * wp_diatom_overlay_sat(sz / (sz + 0.45 * bl));
                        let dcen: f32 = dsh + sz;
                        let rad_d: f32 = sz + 1.3 * bl;
                        let dfw: f32 = wp_diatom_overlay_sat((bl - 1.0) / 2.5);
                        let disc: f32 = smoothstep(rad_d + 1.2, rad_d - 1.2, dcen)
                            * wp_diatom_overlay_sat(1.6 * sz * sz / (rad_d * rad_d) + 0.08);
                        cov = mix(cov, disc, dfw);
                        let rim: f32 =
                            exp(-wp_diatom_overlay_sq((dcen - rad_d) / (0.9 + 0.1 * bl)));
                        let bz: f32 = wp_diatom_overlay_fsin(1.35 * z) * exp(-0.12 * z * z);
                        let fc: Vec2f = abs(fract(p / cs) - vec2f(0.5, 0.5)) * cs;
                        let win: f32 = smoothstep(0.5 * cs, 0.5 * cs - 6.0, max(fc.x, fc.y));
                        (*ctx).a_t = (*ctx).a_t
                            * mix(
                                1.0,
                                1.0 - 0.55 * cov
                                    + 0.1
                                        * bz
                                        * rim
                                        * dfw
                                        * wp_diatom_overlay_sat(4.0 * sz / rad_d),
                                win,
                            );
                        (*ctx).a_g = (*ctx).a_g
                            + win
                                * (0.7
                                    * exp(-wp_diatom_overlay_sq(dsh / (0.9 + bl)))
                                    * wp_diatom_overlay_sat(sz / (sz + 0.6 * bl))
                                    + 0.1 * cov);
                        (*ctx).a_h = (*ctx).a_h + win * 1.2 * cov;
                    }
                }
            }
        }
        {
            let bd_s: Vec2f =
                wp_diatom_overlay_rot((h2 - 0.5) * 0.6) * normalize(vec2f(-side, 0.3));
            (*ctx).sh_th_b_capital = atan2(bd_s.y, bd_s.x);
            wp_diatom_overlay_shard_slot(
                ctx, p, 0.0, 0.34, 1.11, 2.4, -0.15, vc, rv_capital, r0_upper, side, sd, tau,
                focus, res, cr,
            );
            wp_diatom_overlay_shard_slot(
                ctx, p, 1.0, 0.17, 1.36, 1.7, -1.1, vc, rv_capital, r0_upper, side, sd, tau, focus,
                res, cr,
            );
            wp_diatom_overlay_shard_slot(
                ctx, p, 2.0, 0.03, 1.62, 2.0, 0.55, vc, rv_capital, r0_upper, side, sd, tau, focus,
                res, cr,
            );
        }
        let base: Vec3f = wp_previous_texel(vec2i(fc.x as i32, fc.y as i32)).xyz();
        let mut col: Vec3f = base;
        let uvn: Vec2f = (p - res * vec2f(0.42, 0.6)) / res.y;
        if mode == 1 {
            let k: Vec3f = vec3f(1.0, 1.0, 1.0)
                - exp(-1.25 * (*ctx).a_g * (*ctx).g_k * vec3f(0.72, 0.84, 1.0));
            if get!(BLUR_LOCALS).wp_env.x > 0.5 {
                col = vec3f(1.0, 1.0, 1.0)
                    - (vec3f(1.0, 1.0, 1.0) - base) * (vec3f(1.0, 1.0, 1.0) - k);
            } else {
                let illum: f32 =
                    1.0 - 0.065 * smoothstep(0.1, 1.15, length(uvn * vec2f(0.85, 1.0)));
                let paper: Vec3f = vec3f(0.953, 0.957, 0.962) * illum;
                let w: Vec3f = vec3f(1.05, 1.0, 0.92);
                let bg: Vec3f =
                    vec3f(0.010, 0.012, 0.018) + vec3f(0.004, 0.006, 0.012) * (1.0 - length(uvn));
                let c_tm: Vec3f = bg + (vec3f(1.0, 1.0, 1.0) - base / paper) / (0.9 * w);
                col = base - paper * 0.9 * w * (vec3f(1.0, 1.0, 1.0) - c_tm) * k;
            }
        } else {
            if mode == 2 {
                let hs_capital: f32 = (*ctx).a_h * 1.1;
                let rel: f32 = (dpdx(hs_capital) - dpdy(hs_capital)) * 0.7071 * dpr;
                col = base
                    * (1.0 + 1.25 * rel)
                    * mix(vec3f(1.0, 1.0, 1.0), (*ctx).a_t, vec3f(0.22, 0.22, 0.22));
            } else {
                col = base * (*ctx).a_t;
            }
        }
        let fr: u32 = wp_diatom_overlay_lb32(floor(t * 30.0) as u32 + 0x51u32);
        let gp: Vec2f = floor(fc) + vec2f((fr & 255u32) as f32, ((fr >> 8u32) & 255u32) as f32);
        let gn: Vec2f = wp_noise((gp + vec2f(0.5, 0.5)) / 256.0).xy();
        let gc: f32 = wp_noise((gp + vec2f(38.0, 92.0)) / 256.0).z;
        let film: f32 = (gn.x + gn.y - 1.0 + 1.4 * (gc - 0.5)) * 2.2;
        let lum: f32 = dot(col, vec3f(0.3, 0.55, 0.15));
        let g_amp: f32 = max(
            get!(BLUR_LOCALS).wp_p1.w
                * 0.043
                * (0.45 + 2.2 * lum * (1.0 - lum))
                * select(1.0, 0.8, get!(BLUR_LOCALS).wp_env.x > 0.5),
            0.6 / 255.0,
        );
        col = col + film * g_amp * vec3f(1.0, 0.98, 1.03);
        result = vec4f(
            clamp(col, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0)).x,
            clamp(col, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0)).y,
            clamp(col, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0)).z,
            1.0,
        );
        result
    }

    // butler.photo-clouds/shader.frag
    pub const WP_PHOTO_CLOUDS_T: f32 = 120.0;

    pub const WP_PHOTO_CLOUDS_TAU: f32 = 6.2831853;

    pub const WP_PHOTO_CLOUDS_LUMA: Vec3f = vec3f(0.2126, 0.7152, 0.0722);

    pub const WP_PHOTO_CLOUDS_PHOTO_W: f32 = 2000.0;

    pub const WP_PHOTO_CLOUDS_PHOTO_ASPECT: f32 = 2000.0 / 1334.0;

    pub const WP_PHOTO_CLOUDS_BIRD: Vec2f = vec2f(1006.0 / 2000.0, 1.0 - 674.0 / 1334.0);

    pub const WP_PHOTO_CLOUDS_BIRD_COL: Vec3f = vec3f(0.085, 0.035, 0.035);

    pub fn wp_photo_clouds_vn(p: Vec2f) -> f32 {
        let i: Vec2f = floor(p);
        let mut f: Vec2f = fract(p);
        f = f * f * (vec2f(3.0, 3.0) - 2.0 * f);
        return wp_noise((i + f + vec2f(0.5, 0.5)) / 256.0).x;
    }

    pub fn wp_photo_clouds_vn2(p: Vec2f) -> Vec2f {
        let i: Vec2f = floor(p);
        let mut f: Vec2f = fract(p);
        f = f * f * (vec2f(3.0, 3.0) - 2.0 * f);
        return wp_noise((i + f + vec2f(0.5, 0.5)) / 256.0).yz();
    }

    pub fn wp_photo_clouds_img(fc: Vec2f) -> Vec3f {
        return wp_previous(fc / get!(BLUR_LOCALS).wp_frame.xy()).xyz();
    }

    pub fn wp_photo_clouds_headline_veil(col: Vec3f, fc: Vec2f, strength: f32) -> Vec3f {
        let pr: f32 = max(get!(BLUR_LOCALS).wp_frame.z, 1e-3);
        let p: Vec2f = fc / pr;
        let r_upper: Vec2f = get!(BLUR_LOCALS).wp_frame.xy() / pr;
        let mut cr: Vec4f = get!(BLUR_LOCALS).wp_rect / pr;
        if cr.z < 1.0 || cr.w < 1.0 {
            cr = vec4f(
                r_upper.x * 0.2,
                r_upper.y * 0.42,
                r_upper.x * 0.6,
                r_upper.y * 0.5,
            );
        }
        let head_h: f32 = min(cr.w * 0.45, max(150.0, cr.w * 0.36));
        let hb: Vec2f = vec2f(cr.z, head_h) * 0.5;
        let hc: Vec2f = vec2f(cr.x, cr.y + cr.w - head_h) + hb;
        let q: Vec2f = abs(p - hc) - hb + vec2f(28.0, 28.0);
        let d: f32 = length(max(q, vec2f(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - 28.0;
        let m: f32 = (1.0 - smoothstep(-40.0, 120.0, d)) * strength;
        if m < 0.003 {
            return col;
        }
        let o: f32 = 5.0 * pr;
        let b: Vec3f = (col
            + wp_photo_clouds_img(fc + vec2f(o, o))
            + wp_photo_clouds_img(fc + vec2f(-o, o))
            + wp_photo_clouds_img(fc + vec2f(o, -o))
            + wp_photo_clouds_img(fc - vec2f(o, o)))
            * 0.2;
        let l_upper: f32 = dot(b, WP_PHOTO_CLOUDS_LUMA);
        let mut veil: Vec3f = vec3f(0.0, 0.0, 0.0);
        let mut a: f32 = 0.0;
        if get!(BLUR_LOCALS).wp_env.x > 0.5 {
            veil = b * 0.3;
            a = clamp(
                (l_upper - 0.36) / max(l_upper - dot(veil, WP_PHOTO_CLOUDS_LUMA), 0.05),
                0.0,
                0.5,
            ) + 0.08;
        } else {
            veil = vec3f(1.0, 1.0, 1.0) - (vec3f(1.0, 1.0, 1.0) - b) * 0.22;
            a = clamp(
                (0.6 - l_upper) / max(dot(veil, WP_PHOTO_CLOUDS_LUMA) - l_upper, 0.05),
                0.0,
                0.5,
            ) + 0.08;
        }
        return mix(
            col,
            mix(mix(col, b, vec3f(0.7, 0.7, 0.7)), veil, vec3f(a, a, a)),
            vec3f(min(m, 1.0), min(m, 1.0), min(m, 1.0)),
        );
    }

    pub fn wp_photo_clouds(fc: Vec2f) -> Vec4f {
        let mut result = vec4f(0.0, 0.0, 0.0, 0.0);
        let fc: Vec2f = fc;
        if get!(BLUR_LOCALS).wp_col.w < 0.5 {
            result = vec4f(
                mix(
                    vec3f(0.93, 0.9, 0.88),
                    vec3f(0.12, 0.12, 0.14),
                    vec3f(
                        get!(BLUR_LOCALS).wp_env.x,
                        get!(BLUR_LOCALS).wp_env.x,
                        get!(BLUR_LOCALS).wp_env.x,
                    ),
                )
                .x,
                mix(
                    vec3f(0.93, 0.9, 0.88),
                    vec3f(0.12, 0.12, 0.14),
                    vec3f(
                        get!(BLUR_LOCALS).wp_env.x,
                        get!(BLUR_LOCALS).wp_env.x,
                        get!(BLUR_LOCALS).wp_env.x,
                    ),
                )
                .y,
                mix(
                    vec3f(0.93, 0.9, 0.88),
                    vec3f(0.12, 0.12, 0.14),
                    vec3f(
                        get!(BLUR_LOCALS).wp_env.x,
                        get!(BLUR_LOCALS).wp_env.x,
                        get!(BLUR_LOCALS).wp_env.x,
                    ),
                )
                .z,
                1.0,
            );
            return result;
        }
        let pr: f32 = max(get!(BLUR_LOCALS).wp_frame.z, 1e-3);
        let p: Vec2f = fc / pr;
        let t: f32 = get!(BLUR_LOCALS).wp_frame.w;
        let ang: f32 = WP_PHOTO_CLOUDS_TAU * t / WP_PHOTO_CLOUDS_T;
        let r: f32 = WP_PHOTO_CLOUDS_PHOTO_ASPECT
            / (get!(BLUR_LOCALS).wp_frame.xy().x / get!(BLUR_LOCALS).wp_frame.xy().y);
        let disp: Vec2f = select(vec2f(r, 1.0), vec2f(1.0, 1.0 / r), r < 1.0);
        let s: f32 = disp.x * get!(BLUR_LOCALS).wp_frame.xy().x / WP_PHOTO_CLOUDS_PHOTO_W;
        let bo: Vec2f = ((WP_PHOTO_CLOUDS_BIRD - vec2f(0.5, 0.5)) * disp + vec2f(0.5, 0.5))
            * get!(BLUR_LOCALS).wp_frame.xy();
        let c0: Vec3f = wp_photo_clouds_img(fc);
        let cm: Vec3f = (c0 * 2.0
            + wp_photo_clouds_img(fc + vec2f(9.0, 5.0) * pr)
            + wp_photo_clouds_img(fc - vec2f(5.0, 9.0) * pr))
            * 0.25;
        let cloud: f32 = smoothstep(-0.22, -0.05, cm.x - cm.z);
        let keep: f32 = smoothstep(16.0, 42.0, length((fc - bo) / s));
        let evo: Vec2f = 2.2 * vec2f(cos(ang), sin(ang));
        let f_upper: Vec2f =
            (wp_photo_clouds_vn2(p / 170.0 + evo) - vec2f(0.5, 0.5)) * 2.6 + vec2f(0.55, 0.08);
        let p_upper: f32 = 8.0;
        let ph: f32 = fract(t / p_upper + wp_photo_clouds_vn(p / 320.0 + vec2f(40.0, 40.0)));
        let ph2: f32 = fract(ph + 0.5);
        let amp: Vec2f = f_upper * 17.0 * pr * get!(BLUR_LOCALS).wp_p0.x * cloud * keep;
        let a0: Vec3f = wp_photo_clouds_img(fc - amp * (ph - 0.5));
        let a1: Vec3f = wp_photo_clouds_img(fc - amp * (ph2 - 0.5));
        let mut col: Vec3f = mix(
            a0,
            a1,
            vec3f(
                abs(ph - 0.5) * 2.0,
                abs(ph - 0.5) * 2.0,
                abs(ph - 0.5) * 2.0,
            ),
        );
        let lum: f32 = dot(col, WP_PHOTO_CLOUDS_LUMA);
        let glow: f32 = wp_photo_clouds_vn(
            p / 260.0 + 1.6 * vec2f(cos(ang * 2.0), sin(ang * 2.0)) + vec2f(90.0, 90.0),
        ) - 0.5;
        let lit: f32 = cloud * smoothstep(0.25, 0.75, lum);
        col = col * (1.0 + 0.16 * glow * lit * get!(BLUR_LOCALS).wp_p0.y);
        let warm: f32 = sin(ang + 1.2);
        col = col
            * mix(
                vec3f(1.0, 1.0, 1.0),
                mix(
                    vec3f(0.975, 0.995, 1.03),
                    vec3f(1.035, 1.0, 0.955),
                    vec3f(0.5 + 0.5 * warm, 0.5 + 0.5 * warm, 0.5 + 0.5 * warm),
                ),
                vec3f(
                    lit * get!(BLUR_LOCALS).wp_p0.y,
                    lit * get!(BLUR_LOCALS).wp_p0.y,
                    lit * get!(BLUR_LOCALS).wp_p0.y,
                ),
            );
        if get!(BLUR_LOCALS).wp_p0.z > 0.5 {
            let d0: Vec2f = (fc - bo) / s;
            let hole: f32 =
                (1.0 - smoothstep(6.0, 10.0, abs(d0.x))) * (1.0 - smoothstep(5.0, 8.5, abs(d0.y)));
            if hole > 0.0 {
                col = mix(
                    col,
                    (wp_photo_clouds_img(fc + vec2f(-16.0, 0.0) * s)
                        + wp_photo_clouds_img(fc + vec2f(-12.0, -12.0) * s))
                        * 0.5,
                    vec3f(hole, hole, hole),
                );
            }
            let lo: f32 = (0.5 - 0.5 / disp.x) * WP_PHOTO_CLOUDS_PHOTO_W;
            let hi: f32 = (0.5 + 0.5 / disp.x) * WP_PHOTO_CLOUDS_PHOTO_W;
            let ax: f32 = min(380.0, (hi - lo) * 0.42);
            let cx: f32 = clamp(1330.0, lo + ax + 20.0, hi - ax - 20.0);
            let mut env: f32 = 0.0;
            for k in 0..4 {
                let b: Vec2f = select(
                    select(
                        select(vec2f(91.0, 0.9), vec2f(64.5, 1.7), k == 2),
                        vec2f(33.0, 0.8),
                        k == 1,
                    ),
                    vec2f(4.5, 1.5),
                    k == 0,
                );
                let dt: f32 = t - b.x;
                env = env + smoothstep(0.0, 0.12, dt) * (1.0 - smoothstep(b.y - 0.2, b.y, dt));
            }
            let mut p0_upper: Vec2f = vec2f(0.0, 0.0);
            let mut p1_upper: Vec2f = vec2f(0.0, 0.0);
            for k in 0..2 {
                let u: f32 = (t + k as f32 * 0.25) / WP_PHOTO_CLOUDS_T;
                let th: f32 =
                    WP_PHOTO_CLOUDS_TAU * u + 0.3 * sin(WP_PHOTO_CLOUDS_TAU * u * 2.0 + 3.14159);
                let mut p_upper: Vec2f = vec2f(
                    cx - ax * cos(th) + 35.0 * cos(2.0 * th),
                    860.0 - 115.0 * sin(th) + 30.0 * sin(2.0 * th + 0.7),
                );
                p_upper.y = p_upper.y + (5.0 * sin(ang * 17.0) - 4.0 * env);
                if k == 0 {
                    p0_upper = p_upper;
                } else {
                    p1_upper = p_upper;
                }
            }
            let v: Vec2f = p1_upper - p0_upper;
            let fwd: f32 = abs(v.x) / max(length(v), 1e-3);
            let sx: f32 = sign(v.x + 1e-4) * mix(0.28, 1.0, smoothstep(0.0, 0.6, fwd));
            let tilt: f32 = clamp(atan2(-v.y, abs(v.x)) * 0.7, -0.45, 0.45)
                * sign(v.x + 1e-4)
                * smoothstep(0.0, 0.5, fwd);
            let depth: f32 = 1.0 - 0.12 * (860.0 - p0_upper.y) / 115.0;
            let sy: f32 = 1.0 - 0.6 * env * (0.5 + 0.5 * sin(WP_PHOTO_CLOUDS_TAU * 4.5 * t));
            let pos: Vec2f = ((vec2f(
                p0_upper.x / WP_PHOTO_CLOUDS_PHOTO_W,
                1.0 - p0_upper.y / (WP_PHOTO_CLOUDS_PHOTO_W / WP_PHOTO_CLOUDS_PHOTO_ASPECT),
            ) - vec2f(0.5, 0.5))
                * disp
                + vec2f(0.5, 0.5))
                * get!(BLUR_LOCALS).wp_frame.xy();
            let dl: Vec2f = (fc - pos) / s;
            let cs: f32 = cos(tilt);
            let sn: f32 = sin(tilt);
            let lb: Vec2f = mat2x2f(vec2f(cs, -sn), vec2f(sn, cs)) * dl / (vec2f(sx, sy) * depth);
            if abs(lb.x) < 9.0 && abs(lb.y) < 7.0 {
                let src: Vec2f = bo + lb * s;
                let sc: Vec3f = wp_photo_clouds_img(src);
                let bg: Vec3f = (wp_photo_clouds_img(src + vec2f(-16.0, 0.0) * s)
                    + wp_photo_clouds_img(src + vec2f(-12.0, -12.0) * s))
                    * 0.5;
                let lb_capital: f32 = dot(bg, WP_PHOTO_CLOUDS_LUMA);
                let mut al: f32 = smoothstep(
                    0.35,
                    0.85,
                    (lb_capital - dot(sc, WP_PHOTO_CLOUDS_LUMA)) / max(lb_capital - 0.04, 0.05),
                );
                al = al
                    * ((1.0 - smoothstep(7.0, 9.0, abs(lb.x)))
                        * (1.0 - smoothstep(5.0, 7.0, abs(lb.y))));
                col = mix(col, WP_PHOTO_CLOUDS_BIRD_COL, vec3f(al, al, al));
            }
        }
        col = wp_photo_clouds_headline_veil(col, fc, get!(BLUR_LOCALS).wp_p0.w);
        result = vec4f(
            clamp(col, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0)).x,
            clamp(col, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0)).y,
            clamp(col, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0)).z,
            1.0,
        );
        result
    }

    // butler.photo-daisies/shader.frag
    pub const WP_PHOTO_DAISIES_T: f32 = 120.0;

    pub const WP_PHOTO_DAISIES_TAU: f32 = 6.2831853;

    pub const WP_PHOTO_DAISIES_LUMA: Vec3f = vec3f(0.2126, 0.7152, 0.0722);

    pub const WP_PHOTO_DAISIES_PHOTO_ASPECT: f32 = 2000.0 / 1266.0;

    pub fn wp_photo_daisies_vn(p: Vec2f) -> f32 {
        let i: Vec2f = floor(p);
        let mut f: Vec2f = fract(p);
        f = f * f * (vec2f(3.0, 3.0) - 2.0 * f);
        return wp_noise((i + f + vec2f(0.5, 0.5)) / 256.0).x;
    }

    pub fn wp_photo_daisies_img(fc: Vec2f) -> Vec3f {
        return wp_previous(fc / get!(BLUR_LOCALS).wp_frame.xy()).xyz();
    }

    pub fn wp_photo_daisies_headline_veil(col: Vec3f, fc: Vec2f, strength: f32) -> Vec3f {
        let pr: f32 = max(get!(BLUR_LOCALS).wp_frame.z, 1e-3);
        let p: Vec2f = fc / pr;
        let r_upper: Vec2f = get!(BLUR_LOCALS).wp_frame.xy() / pr;
        let mut cr: Vec4f = get!(BLUR_LOCALS).wp_rect / pr;
        if cr.z < 1.0 || cr.w < 1.0 {
            cr = vec4f(
                r_upper.x * 0.2,
                r_upper.y * 0.42,
                r_upper.x * 0.6,
                r_upper.y * 0.5,
            );
        }
        let head_h: f32 = min(cr.w * 0.45, max(150.0, cr.w * 0.36));
        let hb: Vec2f = vec2f(cr.z, head_h) * 0.5;
        let hc: Vec2f = vec2f(cr.x, cr.y + cr.w - head_h) + hb;
        let q: Vec2f = abs(p - hc) - hb + vec2f(28.0, 28.0);
        let d: f32 = length(max(q, vec2f(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - 28.0;
        let m: f32 = (1.0 - smoothstep(-40.0, 120.0, d)) * strength;
        if m < 0.003 {
            return col;
        }
        let o: f32 = 5.0 * pr;
        let b: Vec3f = (col
            + wp_photo_daisies_img(fc + vec2f(o, o))
            + wp_photo_daisies_img(fc + vec2f(-o, o))
            + wp_photo_daisies_img(fc + vec2f(o, -o))
            + wp_photo_daisies_img(fc - vec2f(o, o)))
            * 0.2;
        let l_upper: f32 = dot(b, WP_PHOTO_DAISIES_LUMA);
        let mut veil: Vec3f = vec3f(0.0, 0.0, 0.0);
        let mut a: f32 = 0.0;
        if get!(BLUR_LOCALS).wp_env.x > 0.5 {
            veil = b * 0.3;
            a = clamp(
                (l_upper - 0.36) / max(l_upper - dot(veil, WP_PHOTO_DAISIES_LUMA), 0.05),
                0.0,
                0.5,
            ) + 0.08;
        } else {
            veil = vec3f(1.0, 1.0, 1.0) - (vec3f(1.0, 1.0, 1.0) - b) * 0.22;
            a = clamp(
                (0.6 - l_upper) / max(dot(veil, WP_PHOTO_DAISIES_LUMA) - l_upper, 0.05),
                0.0,
                0.5,
            ) + 0.08;
        }
        return mix(
            col,
            mix(mix(col, b, vec3f(0.7, 0.7, 0.7)), veil, vec3f(a, a, a)),
            vec3f(min(m, 1.0), min(m, 1.0), min(m, 1.0)),
        );
    }

    pub fn wp_photo_daisies(fc: Vec2f) -> Vec4f {
        let mut result = vec4f(0.0, 0.0, 0.0, 0.0);
        let fc: Vec2f = fc;
        if get!(BLUR_LOCALS).wp_col.w < 0.5 {
            result = vec4f(
                mix(
                    vec3f(0.93, 0.93, 0.9),
                    vec3f(0.12, 0.13, 0.12),
                    vec3f(
                        get!(BLUR_LOCALS).wp_env.x,
                        get!(BLUR_LOCALS).wp_env.x,
                        get!(BLUR_LOCALS).wp_env.x,
                    ),
                )
                .x,
                mix(
                    vec3f(0.93, 0.93, 0.9),
                    vec3f(0.12, 0.13, 0.12),
                    vec3f(
                        get!(BLUR_LOCALS).wp_env.x,
                        get!(BLUR_LOCALS).wp_env.x,
                        get!(BLUR_LOCALS).wp_env.x,
                    ),
                )
                .y,
                mix(
                    vec3f(0.93, 0.93, 0.9),
                    vec3f(0.12, 0.13, 0.12),
                    vec3f(
                        get!(BLUR_LOCALS).wp_env.x,
                        get!(BLUR_LOCALS).wp_env.x,
                        get!(BLUR_LOCALS).wp_env.x,
                    ),
                )
                .z,
                1.0,
            );
            return result;
        }
        let pr: f32 = max(get!(BLUR_LOCALS).wp_frame.z, 1e-3);
        let p: Vec2f = fc / pr;
        let t: f32 = get!(BLUR_LOCALS).wp_frame.w;
        let ang: f32 = WP_PHOTO_DAISIES_TAU * t / WP_PHOTO_DAISIES_T;
        let r: f32 = WP_PHOTO_DAISIES_PHOTO_ASPECT
            / (get!(BLUR_LOCALS).wp_frame.xy().x / get!(BLUR_LOCALS).wp_frame.xy().y);
        let disp: Vec2f = select(vec2f(r, 1.0), vec2f(1.0, 1.0 / r), r < 1.0);
        let pu: Vec2f =
            (fc / get!(BLUR_LOCALS).wp_frame.xy() - vec2f(0.5, 0.5)) / disp + vec2f(0.5, 0.5);
        let y_top: f32 = 1.0 - pu.y;
        let in_focus: f32 = smoothstep(0.28, 0.5, y_top)
            * (1.0 - smoothstep(0.62, 0.85, pu.x) * (1.0 - smoothstep(0.7, 0.9, y_top)));
        let oof: f32 = 1.0 - in_focus;
        let gust: f32 = 0.65 + 0.35 * sin(ang * 3.0 + 1.0);
        let wave: f32 = dot(p, vec2f(0.0088, 0.0031)) - ang * 24.0
            + wp_photo_daisies_vn(p / 380.0 + vec2f(17.0, 17.0)) * 2.5;
        let sway: Vec2f = vec2f(sin(wave), 0.35 * sin(wave * 0.7 + 1.3))
            * 3.6
            * pr
            * gust
            * oof
            * get!(BLUR_LOCALS).wp_p0.z;
        let mut col: Vec3f = wp_photo_daisies_img(fc + sway);
        let lum: f32 = dot(col, WP_PHOTO_DAISIES_LUMA);
        let n1: f32 = wp_photo_daisies_vn(p / 95.0 + 3.0 * vec2f(cos(ang), sin(ang)));
        let n2: f32 = wp_photo_daisies_vn(
            p / 42.0
                + 4.0 * vec2f(cos(-ang * 2.0 + 2.0), sin(-ang * 2.0 + 2.0))
                + vec2f(60.0, 60.0),
        );
        let leaf: f32 = smoothstep(0.36, 0.74, n1 * 0.62 + n2 * 0.38);
        let warmth: f32 = smoothstep(0.15, 0.7, lum);
        let sun: f32 = (leaf - 0.4) * 0.16;
        col = col * (1.0 + sun * (0.35 + 0.65 * warmth) * get!(BLUR_LOCALS).wp_p0.x);
        col = col + vec3f(0.035, 0.025, 0.0) * leaf * warmth * get!(BLUR_LOCALS).wp_p0.x;
        let hl: f32 = smoothstep(0.6, 0.9, lum) * (0.25 + 0.75 * oof);
        let breath: f32 = sin(ang * 20.0
            + wp_photo_daisies_vn(p / 38.0 + vec2f(200.0, 200.0)) * WP_PHOTO_DAISIES_TAU * 2.0);
        col = col * (1.0 + 0.085 * hl * breath * get!(BLUR_LOCALS).wp_p0.y);
        col = wp_photo_daisies_headline_veil(col, fc, get!(BLUR_LOCALS).wp_p0.w);
        result = vec4f(
            clamp(col, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0)).x,
            clamp(col, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0)).y,
            clamp(col, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0)).z,
            1.0,
        );
        result
    }

    // butler.stipple/shader.frag
    pub fn wp_stipple_img(fc: Vec2f) -> Vec3f {
        return wp_previous(fc / get!(BLUR_LOCALS).wp_frame.xy()).xyz();
    }

    pub fn wp_stipple_tone(c: Vec3f) -> f32 {
        return dot(c, vec3f(0.2126, 0.7152, 0.0722))
            + 0.12 * (max(c.x, max(c.y, c.z)) - min(c.x, min(c.y, c.z)));
    }

    pub fn wp_stipple_nz(p: Vec2f) -> Vec4f {
        let i: Vec2f = floor(p);
        let mut f: Vec2f = fract(p);
        f = f * f * (vec2f(3.0, 3.0) - 2.0 * f);
        return wp_noise((i + f + vec2f(0.5, 0.5)) / 256.0);
    }

    pub fn wp_stipple_head_mask(fc: Vec2f) -> f32 {
        let pr: f32 = max(get!(BLUR_LOCALS).wp_frame.z, 1e-3);
        let p: Vec2f = fc / pr;
        let r_upper: Vec2f = get!(BLUR_LOCALS).wp_frame.xy() / pr;
        let mut cr: Vec4f = get!(BLUR_LOCALS).wp_rect / pr;
        if cr.z < 1.0 || cr.w < 1.0 {
            cr = vec4f(
                r_upper.x * 0.2,
                r_upper.y * 0.42,
                r_upper.x * 0.6,
                r_upper.y * 0.5,
            );
        }
        let head_h: f32 = min(cr.w * 0.45, max(150.0, cr.w * 0.36));
        let hb: Vec2f = vec2f(cr.z, head_h) * 0.5;
        let hc: Vec2f = vec2f(cr.x, cr.y + cr.w - head_h) + hb;
        let q: Vec2f = abs(p - hc) - hb + vec2f(28.0, 28.0);
        let d: f32 = length(max(q, vec2f(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - 28.0;
        return 1.0 - smoothstep(-30.0, 130.0, d);
    }

    pub fn wp_stipple(fc: Vec2f) -> Vec4f {
        let mut result = vec4f(0.0, 0.0, 0.0, 0.0);
        let fc: Vec2f = fc;
        let pr: f32 = max(get!(BLUR_LOCALS).wp_frame.z, 1e-3);
        let uv: Vec2f = fc / get!(BLUR_LOCALS).wp_frame.xy();
        let mut tn: f32 = 0.0;
        if get!(BLUR_LOCALS).wp_col.w > 0.5 {
            let o: f32 = 1.3 * pr;
            let r: f32 = 9.0 * pr;
            let c0: f32 = wp_stipple_tone(wp_stipple_img(fc));
            let s: f32 = wp_stipple_tone(wp_stipple_img(fc + vec2f(o, o * 0.4)))
                + wp_stipple_tone(wp_stipple_img(fc + vec2f(-o * 0.4, o)))
                + wp_stipple_tone(wp_stipple_img(fc - vec2f(o, o * 0.4)))
                + wp_stipple_tone(wp_stipple_img(fc + vec2f(o * 0.4, -o)));
            let mut l_upper: f32 = mix(c0, s * 0.25, 0.6);
            let s_upper: f32 = wp_stipple_tone(wp_stipple_img(fc + vec2f(r, r * 0.5)))
                + wp_stipple_tone(wp_stipple_img(fc + vec2f(-r * 0.5, r)))
                + wp_stipple_tone(wp_stipple_img(fc - vec2f(r, r * 0.5)))
                + wp_stipple_tone(wp_stipple_img(fc + vec2f(r * 0.5, -r)));
            l_upper = l_upper + (l_upper - s_upper * 0.25) * 0.75;
            let mut m: f32 = 0.0;
            let mut m2: f32 = 0.0;
            for j in 0..4 {
                for i in 0..4 {
                    let v: f32 = wp_stipple_tone(
                        wp_previous((vec2f(i as f32, j as f32) + vec2f(0.5, 0.5)) / 4.0).xyz(),
                    );
                    m = m + v;
                    m2 = m2 + v * v;
                }
            }
            m = m / 16.0;
            let sd: f32 = clamp(sqrt(max(m2 / 16.0 - m * m, 0.0)), 0.1, 0.3);
            let lo: f32 = clamp(m - 1.3 * sd, 0.0, 0.7);
            let hi: f32 = clamp(m + 1.7 * sd, lo + 0.3, 1.0);
            tn = clamp((l_upper - lo) / (hi - lo), 0.0, 1.0);
        } else {
            tn = 0.2 + 0.4 * smoothstep(0.0, 1.0, uv.y);
        }
        tn = mix(tn, tn * tn * (3.0 - 2.0 * tn), 0.7);
        let c: Vec2f = (uv - vec2f(0.5, 0.5)) * 2.0;
        let vd: f32 = mix(length(c) * 0.7071, max(abs(c.x), abs(c.y)), 0.5);
        let r0: f32 = mix(1.05, 0.15, get!(BLUR_LOCALS).wp_p1.x);
        let vig: f32 = get!(BLUR_LOCALS).wp_p0.w * smoothstep(r0, r0 + 0.75, vd);
        let lay: f32 = 0.86 + 0.14 * wp_stipple_nz(fc / pr / 170.0 + vec2f(40.5, 11.5)).w;
        result = vec4f(tn, vig, wp_stipple_head_mask(fc), lay);
        result
    }

    // butler.stipple/overlay.frag
    pub const WP_STIPPLE_OVERLAY_T: f32 = 120.0;

    pub const WP_STIPPLE_OVERLAY_TAU: f32 = 6.2831853;

    pub fn wp_stipple_overlay_nz(p: Vec2f) -> Vec4f {
        let i: Vec2f = floor(p);
        let mut f: Vec2f = fract(p);
        f = f * f * (vec2f(3.0, 3.0) - 2.0 * f);
        return wp_noise((i + f + vec2f(0.5, 0.5)) / 256.0);
    }

    pub fn wp_stipple_overlay_pair(
        k: i32,
        dark: bool,
        ink: ptr!(function, Vec3f),
        paper: ptr!(function, Vec3f),
    ) {
        if k == 3 {
            *ink = vec3f(0.925, 0.898, 0.835) * select(1.0, 0.9, dark);
            *paper = vec3f(0.075, 0.086, 0.196);
            return;
        }
        if !dark {
            if k == 1 {
                *ink = vec3f(0.890, 0.243, 0.153);
                *paper = vec3f(0.957, 0.922, 0.851);
            } else {
                if k == 2 {
                    *ink = vec3f(0.102, 0.314, 0.212);
                    *paper = vec3f(0.929, 0.906, 0.851);
                } else {
                    *ink = vec3f(0.169, 0.169, 0.961);
                    *paper = vec3f(0.965, 0.961, 0.945);
                }
            }
        } else {
            if k == 1 {
                *ink = vec3f(0.93, 0.56, 0.46);
                *paper = vec3f(0.149, 0.051, 0.035);
            } else {
                if k == 2 {
                    *ink = vec3f(0.58, 0.77, 0.65);
                    *paper = vec3f(0.035, 0.110, 0.078);
                } else {
                    *ink = vec3f(0.56, 0.56, 0.95);
                    *paper = vec3f(0.047, 0.047, 0.205);
                }
            }
        }
    }

    pub fn wp_stipple_overlay(fc: Vec2f) -> Vec4f {
        let mut result = vec4f(0.0, 0.0, 0.0, 0.0);
        let fc: Vec2f = fc;
        let ip: Vec2i = vec2i(fc.x as i32, fc.y as i32);
        let base: Vec4f = wp_previous_texel(ip);
        let dark: bool = get!(BLUR_LOCALS).wp_env.x > 0.5;
        let mut ink: Vec3f = vec3f(0.0, 0.0, 0.0);
        let mut paper: Vec3f = vec3f(0.0, 0.0, 0.0);
        wp_stipple_overlay_pair(get!(BLUR_LOCALS).wp_p0.x as i32, dark, &mut ink, &mut paper);
        let light_ink: bool = dark || get!(BLUR_LOCALS).wp_p0.x as i32 == 3;
        let y: f32 = dot(ink, vec3f(0.2126, 0.7152, 0.0722));
        ink = mix(
            mix(
                ink,
                vec3f(y, y, y),
                vec3f(
                    0.45 * get!(BLUR_LOCALS).wp_p0.y,
                    0.45 * get!(BLUR_LOCALS).wp_p0.y,
                    0.45 * get!(BLUR_LOCALS).wp_p0.y,
                ),
            ),
            paper,
            vec3f(
                0.55 * get!(BLUR_LOCALS).wp_p0.y,
                0.55 * get!(BLUR_LOCALS).wp_p0.y,
                0.55 * get!(BLUR_LOCALS).wp_p0.y,
            ),
        );
        let inv: bool = get!(BLUR_LOCALS).wp_p0.z > 0.5;
        if inv {
            let k: Vec3f = ink;
            ink = paper;
            paper = k;
        }
        let ink_light: bool = light_ink != inv;
        let pr: f32 = max(get!(BLUR_LOCALS).wp_frame.z, 1e-3);
        let uv: Vec2f = fc / get!(BLUR_LOCALS).wp_frame.xy();
        let t: f32 = get!(BLUR_LOCALS).wp_frame.w;
        let ang: f32 = WP_STIPPLE_OVERLAY_TAU * t / WP_STIPPLE_OVERLAY_T;
        let mut tn: f32 = base.x;
        let sx: f32 = uv.x
            + (uv.y - 0.5) * 0.5 * get!(BLUR_LOCALS).wp_frame.xy().y
                / get!(BLUR_LOCALS).wp_frame.xy().x;
        let band: f32 = exp(-pow(
            (sx - mix(-0.55, 1.55, fract(t / (WP_STIPPLE_OVERLAY_T / 2.0)))) / 0.17,
            2.0,
        ));
        tn = clamp(
            tn + band * 0.17 * get!(BLUR_LOCALS).wp_p1.w * smoothstep(0.05, 0.7, tn),
            0.0,
            1.0,
        );
        let mut cov: f32 = select(1.0 - tn, tn, light_ink);
        cov = pow(cov, exp2(-0.9 * get!(BLUR_LOCALS).wp_p1.y));
        cov = cov + 0.12 * cov * (1.0 - cov);
        let b_upper: Vec4f = wp_stipple_overlay_nz(fc / pr / 170.0 + vec2f(40.5, 11.5));
        let br: f32 = (b_upper.x - 0.5) * cos(ang * 3.0) + (b_upper.y - 0.5) * sin(ang * 3.0);
        cov = cov + br * 0.45 * get!(BLUR_LOCALS).wp_p1.z * cov * (1.0 - cov);
        cov = cov * (1.0 - base.y);
        let flood: bool = ink_light != dark;
        cov = mix(
            cov,
            select(0.0, 1.0, flood),
            select(0.78, 0.9, flood) * base.z,
        );
        cov = clamp(cov, 0.0, 1.0) * 0.92;
        let tile: Vec2i = wp_shr2i(ip, vec2i(8, 8));
        let f_upper: Vec4f = texture_load(
            BLUR_TEXTURE,
            wp_bitand2i(
                ip + vec2i(tile.x * 73 + tile.y * 151, tile.x * 197 + tile.y * 89),
                vec2i(255, 255),
            ),
            0i32,
        );
        let a1: f32 = ang * 7.0 + b_upper.z * WP_STIPPLE_OVERLAY_TAU;
        let n: f32 = (f_upper.x - 0.5) * cos(a1) + (f_upper.y - 0.5) * sin(a1);
        let aa: f32 = 0.07;
        let thr: f32 = clamp(0.5 + n * 1.15, aa, 1.0 - aa);
        let ink_a: f32 = smoothstep(thr - aa, thr + aa, cov) * base.w;
        let pap: Vec3f = paper * (1.0 + (f_upper.z - 0.5) * select(0.08, 0.035, ink_light));
        result = vec4f(
            clamp(
                mix(pap, ink, vec3f(ink_a, ink_a, ink_a)),
                vec3f(0.0, 0.0, 0.0),
                vec3f(1.0, 1.0, 1.0),
            )
            .x,
            clamp(
                mix(pap, ink, vec3f(ink_a, ink_a, ink_a)),
                vec3f(0.0, 0.0, 0.0),
                vec3f(1.0, 1.0, 1.0),
            )
            .y,
            clamp(
                mix(pap, ink, vec3f(ink_a, ink_a, ink_a)),
                vec3f(0.0, 0.0, 0.0),
                vec3f(1.0, 1.0, 1.0),
            )
            .z,
            1.0,
        );
        result
    }

    // butler.grain/shader.frag
    pub const WP_GRAIN_LUMA: Vec3f = vec3f(0.2126, 0.7152, 0.0722);

    pub fn wp_grain(fc: Vec2f) -> Vec4f {
        let mut result = vec4f(0.0, 0.0, 0.0, 0.0);
        let mut color: Vec3f = wp_previous(fc / get!(BLUR_LOCALS).wp_frame.xy()).xyz();
        let luma: f32 = dot(color, WP_GRAIN_LUMA);
        color = mix(
            color,
            vec3f(luma, luma, luma),
            vec3f(
                get!(BLUR_LOCALS).wp_p0.y,
                get!(BLUR_LOCALS).wp_p0.y,
                get!(BLUR_LOCALS).wp_p0.y,
            ),
        );
        let cell: Vec2f = fc / (get!(BLUR_LOCALS).wp_frame.z * 1.5);
        let fine: f32 = wp_noise(
            (cell
                + vec2f(
                    get!(BLUR_LOCALS).wp_env.z * 256.0,
                    get!(BLUR_LOCALS).wp_env.z * 256.0,
                ))
                / 256.0,
        )
        .x - 0.5;
        let coarse: f32 = wp_noise((cell * 0.5 + vec2f(97.0, 31.0)) / 256.0).y - 0.5;
        let midtones: f32 = 1.0 - pow(abs(luma - 0.5) * 2.0, 2.0);
        let grain: f32 = (fine * 0.7 + coarse * 0.3)
            * get!(BLUR_LOCALS).wp_p0.x
            * 0.4
            * (0.35 + 0.65 * midtones);
        result = vec4f(
            (clamp(
                color + vec3f(grain, grain, grain),
                vec3f(0.0, 0.0, 0.0),
                vec3f(1.0, 1.0, 1.0),
            ))
            .x,
            (clamp(
                color + vec3f(grain, grain, grain),
                vec3f(0.0, 0.0, 0.0),
                vec3f(1.0, 1.0, 1.0),
            ))
            .y,
            (clamp(
                color + vec3f(grain, grain, grain),
                vec3f(0.0, 0.0, 0.0),
                vec3f(1.0, 1.0, 1.0),
            ))
            .z,
            1.0,
        );
        result
    }

    // butler.image/shader.frag
    pub fn wp_image_image_tap(uv: Vec2f, offset: Vec2f) -> Vec4f {
        return wp_image_lod(
            uv + offset * get!(BLUR_LOCALS).wp_col.xyz().xy(),
            get!(BLUR_LOCALS).wp_col.xyz().z,
        );
    }

    pub fn wp_image(fc: Vec2f) -> Vec4f {
        let mut result = vec4f(0.0, 0.0, 0.0, 0.0);
        let field: Vec3f = mix(
            vec3f(0.96, 0.96, 0.965),
            vec3f(0.102, 0.106, 0.118),
            vec3f(
                get!(BLUR_LOCALS).wp_env.x,
                get!(BLUR_LOCALS).wp_env.x,
                get!(BLUR_LOCALS).wp_env.x,
            ),
        );
        let uv: Vec2f = fc / get!(BLUR_LOCALS).wp_frame.xy() * get!(BLUR_LOCALS).wp_drop.xy()
            + get!(BLUR_LOCALS).wp_drop.zw();
        let mut image = vec4f(0.0, 0.0, 0.0, 0.0);
        if get!(BLUR_LOCALS).wp_col.x == 0.0 && get!(BLUR_LOCALS).wp_col.y == 0.0 {
            image = wp_image_sample(uv);
        } else {
            image = (wp_image_image_tap(uv, vec2f(0.0, 0.0)) * 2.0
                + wp_image_image_tap(uv, vec2f(1.0, 0.5))
                + wp_image_image_tap(uv, vec2f(-0.5, 1.0))
                + wp_image_image_tap(uv, vec2f(-1.0, -0.5))
                + wp_image_image_tap(uv, vec2f(0.5, -1.0)))
                / 6.0;
        }
        let inside: Vec2f = step(vec2f(0.0, 0.0), uv) * step(uv, vec2f(1.0, 1.0));
        let color: Vec3f = image.xyz() + field * (1.0 - image.w);
        result = vec4f(
            (mix(
                field,
                color,
                vec3f(
                    inside.x * inside.y * get!(BLUR_LOCALS).wp_col.w,
                    inside.x * inside.y * get!(BLUR_LOCALS).wp_col.w,
                    inside.x * inside.y * get!(BLUR_LOCALS).wp_col.w,
                ),
            ))
            .x,
            (mix(
                field,
                color,
                vec3f(
                    inside.x * inside.y * get!(BLUR_LOCALS).wp_col.w,
                    inside.x * inside.y * get!(BLUR_LOCALS).wp_col.w,
                    inside.x * inside.y * get!(BLUR_LOCALS).wp_col.w,
                ),
            ))
            .y,
            (mix(
                field,
                color,
                vec3f(
                    inside.x * inside.y * get!(BLUR_LOCALS).wp_col.w,
                    inside.x * inside.y * get!(BLUR_LOCALS).wp_col.w,
                    inside.x * inside.y * get!(BLUR_LOCALS).wp_col.w,
                ),
            ))
            .z,
            1.0,
        );
        result
    }

    // butler.cherry-blossom/shader.frag
    #[derive(Clone, Copy, Wgsl)]
    pub struct WpCherryBlossomContext {
        pub w_upper: f32,
        pub h_upper: f32,
        pub aa_upper: f32,
        pub er_upper: f32,
        pub el_upper: f32,
        pub ix_upper: f32,
        pub iy_upper: f32,
        pub tb_upper: f32,
        pub dark_upper: bool,
        pub right_part_upper: bool,
        pub tail_upper: f32,
    }

    pub const WP_CHERRY_BLOSSOM_T: f32 = 240.0;

    pub const WP_CHERRY_BLOSSOM_TAU: f32 = 6.2831853;

    pub const WP_CHERRY_BLOSSOM_WREF: f32 = 349.0;

    pub const WP_CHERRY_BLOSSOM_TOP: f32 = 12.0;

    pub const WP_CHERRY_BLOSSOM_SIDE: f32 = 15.0;

    pub fn wp_cherry_blossom_hash(x: i32, row: i32) -> Vec4f {
        return texture_load(BLUR_TEXTURE, vec2i(x & 255, row & 255), 0i32);
    }

    pub fn wp_cherry_blossom_hash2(cell: Vec2f, row: i32) -> Vec4f {
        return wp_cherry_blossom_hash(
            wp_mod(cell.x, 64.0) as i32 + 64 * wp_mod(cell.y + 64.0, 4.0) as i32,
            row,
        );
    }

    pub fn wp_cherry_blossom_vn(p: Vec2f) -> f32 {
        let i: Vec2f = floor(p);
        let mut f: Vec2f = fract(p);
        f = f * f * (vec2f(3.0, 3.0) - 2.0 * f);
        return wp_noise((i + f + vec2f(0.5, 0.5)) / 256.0).z;
    }

    pub fn wp_cherry_blossom_over(dst: Vec4f, src: Vec4f) -> Vec4f {
        return src + dst * (1.0 - src.w);
    }

    pub fn wp_cherry_blossom_right_env(
        ctx: ptr!(function, WpCherryBlossomContext),
        rc: f32,
    ) -> f32 {
        return 2.0 + max((*ctx).iy_upper - 2.0, 0.0) * exp(-rc / 29.0);
    }

    pub fn wp_cherry_blossom_left_env(lc: f32) -> f32 {
        return 2.5 + 10.0 * exp(-lc / 25.0);
    }

    pub fn wp_cherry_blossom_mass(
        ctx: ptr!(function, WpCherryBlossomContext),
        rc: f32,
        lc: f32,
        y: f32,
    ) -> Vec3f {
        let mut right: f32 = 0.0;
        let mut left: f32 = 0.0;
        let mut t_r: f32 = 1.0;
        let mut t_l: f32 = 1.0;
        if rc < (*ctx).er_upper + (*ctx).tail_upper {
            t_r = smoothstep(
                (*ctx).er_upper * 0.7,
                (*ctx).er_upper + (*ctx).tail_upper,
                rc,
            );
            let sc: f32 = 0.86
                + 0.22 * wp_cherry_blossom_vn(vec2f(rc * 0.09, 5.0))
                + 0.12 * wp_cherry_blossom_vn(vec2f(rc * 0.27, 6.0));
            right = wp_cherry_blossom_right_env(ctx, rc) * sc * (1.0 - t_r);
        }
        if !(*ctx).right_part_upper && lc < (*ctx).el_upper + (*ctx).tail_upper {
            t_l = smoothstep(
                (*ctx).el_upper * 0.6,
                (*ctx).el_upper + (*ctx).tail_upper,
                lc,
            );
            let sc: f32 = 0.86
                + 0.22 * wp_cherry_blossom_vn(vec2f(lc * 0.09, 7.0))
                + 0.12 * wp_cherry_blossom_vn(vec2f(lc * 0.27, 8.0));
            left = wp_cherry_blossom_left_env(lc) * sc * (1.0 - t_l);
        }
        let depth: f32 = max(right, left);
        if depth < 0.3 || y > depth {
            return vec3f(0.0, 0.0, 0.0);
        }
        let s: f32 = y / depth;
        let taper: f32 = select(t_l, t_r, right >= left);
        return vec3f(
            1.0,
            clamp(1.0 - 0.8 * s, 0.0, 1.0),
            max(0.6 * smoothstep(0.6, 1.0, s), taper),
        );
    }

    pub fn wp_cherry_blossom_tone(l: f32, n: f32) -> Vec3f {
        let k: f32 = clamp(l * 3.2 + (n - 0.5) * 1.1, 0.0, 3.0);
        let deep: Vec3f = vec3f(0.84, 0.43, 0.58);
        let rose: Vec3f = vec3f(0.92, 0.57, 0.70);
        let mid: Vec3f = vec3f(0.97, 0.73, 0.82);
        let pale: Vec3f = vec3f(1.0, 0.91, 0.94);
        if k < 1.0 {
            return mix(deep, rose, vec3f(step(0.6, k), step(0.6, k), step(0.6, k)));
        }
        if k < 2.0 {
            return mix(rose, mid, vec3f(step(1.5, k), step(1.5, k), step(1.5, k)));
        }
        return mix(mid, pale, vec3f(step(2.5, k), step(2.5, k), step(2.5, k)));
    }

    pub fn wp_cherry_blossom_deep_tone(l: f32, n: f32) -> Vec3f {
        let k: f32 = clamp(l * 3.2 + (n - 0.5) * 1.1, 0.0, 3.0);
        let a: Vec3f = vec3f(0.30, 0.09, 0.18);
        let b: Vec3f = vec3f(0.38, 0.13, 0.24);
        let c: Vec3f = vec3f(0.46, 0.17, 0.30);
        let d: Vec3f = vec3f(0.53, 0.21, 0.35);
        if k < 1.0 {
            return mix(a, b, vec3f(step(0.6, k), step(0.6, k), step(0.6, k)));
        }
        if k < 2.0 {
            return mix(b, c, vec3f(step(1.5, k), step(1.5, k), step(1.5, k)));
        }
        return mix(c, d, vec3f(step(2.5, k), step(2.5, k), step(2.5, k)));
    }

    pub fn wp_cherry_blossom_corner_weight(
        ctx: ptr!(function, WpCherryBlossomContext),
        ctr: Vec2f,
    ) -> f32 {
        let o: f32 = length(vec2f(
            ((*ctx).w_upper - ctr.x) / (*ctx).ix_upper,
            ctr.y / (*ctx).iy_upper,
        ));
        return 1.0 - smoothstep(1.25, 2.4, o);
    }

    pub fn wp_cherry_blossom_in_text_zone(
        ctx: ptr!(function, WpCherryBlossomContext),
        p: Vec2f,
    ) -> bool {
        return p.y >= WP_CHERRY_BLOSSOM_TOP - 1.0
            && (*ctx).w_upper - p.x >= WP_CHERRY_BLOSSOM_SIDE - 1.0
            && p.y <= (*ctx).tb_upper;
    }

    pub fn wp_cherry_blossom_dabs(
        ctx: ptr!(function, WpCherryBlossomContext),
        p: Vec2f,
        _px: Vec2f,
        cell: f32,
        r_bias: f32,
        lift: f32,
        row: i32,
    ) -> Vec4f {
        let gi: Vec2f = floor(p / cell);
        let mut col: Vec4f = vec4f(0.0, 0.0, 0.0, 0.0);
        #[wgsl_allow(non_literal_loop_bounds)]
        for y in -1..=1 {
            #[wgsl_allow(non_literal_loop_bounds)]
            for x in -1..=1 {
                let g: Vec2f = gi + vec2f(x as f32, y as f32);
                let h: Vec4f = wp_cherry_blossom_hash2(g, row);
                let ctr: Vec2f = (g + vec2f(0.2, 0.2) + 0.6 * h.xy()) * cell;
                let m: Vec3f = wp_cherry_blossom_mass(ctx, (*ctx).w_upper - ctr.x, ctr.x, ctr.y);
                if m.x < 0.5 {
                    continue;
                }
                if h.w < m.z * 0.75 {
                    continue;
                }
                let ang: f32 = h.z * WP_CHERRY_BLOSSOM_TAU;
                let mut q: Vec2f = p - ctr;
                q = mat2x2f(vec2f(cos(ang), sin(ang)), vec2f(-sin(ang), cos(ang))) * q;
                let r: f32 = cell * r_bias * (0.62 + 0.3 * h.w) * (1.0 - 0.3 * m.z);
                let d: f32 = length(q / vec2f(1.0, 0.72)) - r;
                let a: f32 = 1.0 - smoothstep(-(*ctx).aa_upper, (*ctx).aa_upper, d);
                if a <= 0.0 {
                    continue;
                }
                let l: f32 = clamp(m.y + lift, 0.0, 1.0);
                let mut c: Vec3f = wp_cherry_blossom_tone(l, h.y);
                if (*ctx).dark_upper {
                    c = mix(
                        c,
                        wp_cherry_blossom_deep_tone(l, h.y),
                        vec3f(
                            wp_cherry_blossom_corner_weight(ctx, ctr),
                            wp_cherry_blossom_corner_weight(ctx, ctr),
                            wp_cherry_blossom_corner_weight(ctx, ctr),
                        ),
                    );
                }
                col = wp_cherry_blossom_over(col, vec4f((c * a).x, (c * a).y, (c * a).z, a));
            }
        }
        return col;
    }

    pub fn wp_cherry_blossom_segment(p: Vec2f, a: Vec2f, b: Vec2f) -> f32 {
        let pa: Vec2f = p - a;
        let ba: Vec2f = b - a;
        let h: f32 = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        return length(pa - ba * h);
    }

    pub fn wp_cherry_blossom_branches(
        ctx: ptr!(function, WpCherryBlossomContext),
        p: Vec2f,
    ) -> Vec4f {
        let mut col: Vec4f = vec4f(0.0, 0.0, 0.0, 0.0);
        for k in 0..5 {
            let h: Vec4f = wp_cherry_blossom_hash(k, 151);
            let a: Vec2f = vec2f((*ctx).w_upper - 2.0 - 30.0 * h.x, 2.0 + 5.0 * h.y);
            let b: Vec2f = a - vec2f(30.0 + 60.0 * h.z, -3.0 * (h.w - 0.4));
            let aa: f32 = 1.0
                - smoothstep(
                    0.7 - (*ctx).aa_upper,
                    0.7 + (*ctx).aa_upper,
                    wp_cherry_blossom_segment(p, a, b),
                );
            col = wp_cherry_blossom_over(
                col,
                vec4f(
                    (vec3f(0.33, 0.24, 0.28) * aa).x,
                    (vec3f(0.33, 0.24, 0.28) * aa).y,
                    (vec3f(0.33, 0.24, 0.28) * aa).z,
                    aa,
                ),
            );
        }
        return col;
    }

    pub fn wp_cherry_blossom_petals(
        ctx: ptr!(function, WpCherryBlossomContext),
        css_fc: Vec2f,
        p: Vec2f,
        under: f32,
        t: f32,
        cell: f32,
        size: f32,
        fall: i32,
        dens: f32,
        row: i32,
    ) -> Vec4f {
        let bar: bool = (*ctx).h_upper > 60.0 && p.y > (*ctx).tb_upper;
        let tz: bool = wp_cherry_blossom_in_text_zone(ctx, p) || bar;
        let vy: f32 = fall as f32 * cell * 8.0 / WP_CHERRY_BLOSSOM_T;
        let vx: f32 = (fall / 2 + 1) as f32 * cell * 16.0 / WP_CHERRY_BLOSSOM_T;
        let pp: Vec2f = css_fc + vec2f(vx * t, vy * t);
        let id: Vec2f = floor(pp / cell);
        let f: Vec2f = pp - id * cell;
        let hx: i32 = wp_mod(id.x, 16.0) as i32 + 16 * wp_mod(id.y, 8.0) as i32;
        let h: Vec4f = wp_cherry_blossom_hash(hx, row);
        if h.w > dens * mix(0.3, 1.0, under) * select(1.0, 0.25, tz) {
            return vec4f(0.0, 0.0, 0.0, 0.0);
        }
        let g: Vec4f = wp_cherry_blossom_hash(hx + 128, row);
        let phase: f32 = h.z * WP_CHERRY_BLOSSOM_TAU;
        let ctr: Vec2f = cell * (vec2f(0.3, 0.3) + 0.4 * h.xy())
            + vec2f(
                sin(t * WP_CHERRY_BLOSSOM_TAU * 36.0 / WP_CHERRY_BLOSSOM_T + phase),
                cos(t * WP_CHERRY_BLOSSOM_TAU * 22.0 / WP_CHERRY_BLOSSOM_T + phase),
            ) * cell
                * 0.12;
        let mut q: Vec2f = f - ctr;
        let spin: f32 =
            phase + t * WP_CHERRY_BLOSSOM_TAU * select(9.0, -9.0, g.x < 0.5) / WP_CHERRY_BLOSSOM_T;
        let tumble: f32 = 0.25
            + 0.75
                * abs(cos(t * WP_CHERRY_BLOSSOM_TAU * (24.0 + floor(g.y * 10.0))
                    / WP_CHERRY_BLOSSOM_T
                    + phase));
        q = mat2x2f(vec2f(cos(spin), sin(spin)), vec2f(-sin(spin), cos(spin))) * q;
        q.x = q.x / tumble;
        let s: f32 = size * (0.6 + 0.8 * g.z);
        let notch: f32 = length(q - vec2f(0.0, s * 1.02)) - s * 0.28;
        let d: f32 = max(length(vec2f(q.x / 0.62, q.y)) - s, -notch) * tumble;
        let a: f32 = (1.0 - smoothstep(-(*ctx).aa_upper * 0.6, (*ctx).aa_upper * 0.6, d))
            * select(select(0.92, 0.3, tz), 0.15, bar);
        let col: Vec3f = mix(
            vec3f(0.93, 0.62, 0.74),
            vec3f(0.99, 0.86, 0.90),
            vec3f(tumble, tumble, tumble),
        );
        return vec4f((col * a).x, (col * a).y, (col * a).z, a);
    }

    pub fn wp_cherry_blossom(fc: Vec2f) -> Vec4f {
        let mut result = vec4f(0.0, 0.0, 0.0, 0.0);
        let mut state = WpCherryBlossomContext {
            w_upper: 0.0,
            h_upper: 0.0,
            aa_upper: 0.0,
            er_upper: 0.0,
            el_upper: 0.0,
            ix_upper: 0.0,
            iy_upper: 0.0,
            tb_upper: 0.0,
            dark_upper: false,
            right_part_upper: false,
            tail_upper: 0.0,
        };
        let ctx = &mut state;
        let res: Vec2f = get!(BLUR_LOCALS).wp_frame.xy() / get!(BLUR_LOCALS).wp_frame.z;
        let css_fc: Vec2f = fc / get!(BLUR_LOCALS).wp_frame.z;
        let p: Vec2f = vec2f(css_fc.x, res.y - css_fc.y);
        (*ctx).w_upper = res.x;
        (*ctx).h_upper = res.y;
        (*ctx).dark_upper = get!(BLUR_LOCALS).wp_env.x > 0.5;
        let t: f32 = get!(BLUR_LOCALS).wp_frame.w;
        (*ctx).aa_upper = 0.8 / get!(BLUR_LOCALS).wp_frame.z;
        (*ctx).er_upper = WP_CHERRY_BLOSSOM_WREF * 0.5;
        (*ctx).el_upper = WP_CHERRY_BLOSSOM_WREF * 0.2;
        (*ctx).ix_upper = 50.0;
        (*ctx).tb_upper = select((*ctx).h_upper, (*ctx).h_upper - 50.0, (*ctx).h_upper > 60.0);
        (*ctx).iy_upper = min(56.0, (*ctx).tb_upper);
        let rc: f32 = (*ctx).w_upper - p.x;
        let mut col: Vec4f = vec4f(0.0, 0.0, 0.0, 0.0);
        if p.y < (*ctx).iy_upper + 10.0 {
            let split: f32 = 0.5 * ((*ctx).el_upper + ((*ctx).w_upper - (*ctx).er_upper));
            let mut q: Vec2f = p;
            (*ctx).tail_upper = clamp((*ctx).w_upper - WP_CHERRY_BLOSSOM_WREF, 0.0, 48.0);
            (*ctx).right_part_upper = p.x >= split;
            if (*ctx).right_part_upper {
                q.x = q.x - ((*ctx).w_upper - WP_CHERRY_BLOSSOM_WREF);
                (*ctx).w_upper = WP_CHERRY_BLOSSOM_WREF;
            }
            let back: Vec4f =
                wp_cherry_blossom_dabs(ctx, q + vec2f(1.7, 1.3), q, 4.5, 1.25, -0.25, 31);
            let front: Vec4f = wp_cherry_blossom_dabs(ctx, q, q, 3.5, 1.1, 0.3, 47);
            let wood: Vec4f = wp_cherry_blossom_branches(ctx, q)
                * select(0.0, 0.85, back.w > 0.2 || front.w > 0.2)
                * select(1.0, 0.0, wp_cherry_blossom_in_text_zone(ctx, q));
            (*ctx).w_upper = res.x;
            col = wp_cherry_blossom_over(col, back);
            col = wp_cherry_blossom_over(col, wood);
            col = wp_cherry_blossom_over(col, front);
        }
        let env_r: f32 = select(
            0.0,
            wp_cherry_blossom_right_env(ctx, rc)
                * (1.0
                    - smoothstep(
                        (*ctx).er_upper * 0.7,
                        (*ctx).er_upper + (*ctx).tail_upper,
                        rc,
                    )),
            rc < (*ctx).er_upper + (*ctx).tail_upper,
        );
        let env_l: f32 = select(
            0.0,
            wp_cherry_blossom_left_env(p.x)
                * (1.0
                    - smoothstep(
                        (*ctx).el_upper * 0.6,
                        (*ctx).el_upper + (*ctx).tail_upper,
                        p.x,
                    )),
            p.x < (*ctx).el_upper + (*ctx).tail_upper,
        );
        let lim: f32 = min(max(env_r, env_l) * 1.1 + 5.0, (*ctx).iy_upper + 3.0);
        let frame: f32 = 1.0 - smoothstep(lim - 2.0, lim, p.y);
        col = col * frame;
        let under: f32 = max(
            1.0 - smoothstep((*ctx).er_upper * 0.6, (*ctx).er_upper * 1.6, rc),
            1.0 - smoothstep((*ctx).el_upper * 0.5, (*ctx).el_upper * 1.4, p.x),
        );
        col = wp_cherry_blossom_over(
            col,
            wp_cherry_blossom_petals(ctx, css_fc, p, under, t, 22.0, 1.5, 5, 0.42, 11),
        );
        col = wp_cherry_blossom_over(
            col,
            wp_cherry_blossom_petals(
                ctx,
                css_fc + vec2f(9.0, 9.0),
                p,
                under,
                t,
                31.0,
                2.4,
                7,
                0.34,
                23,
            ),
        );
        col = wp_cherry_blossom_over(
            col,
            wp_cherry_blossom_petals(
                ctx,
                css_fc + vec2f(17.0, 17.0),
                p,
                under,
                t,
                44.0,
                3.4,
                9,
                0.24,
                37,
            ),
        );
        result = col;
        result
    }
    #[fragment]
    pub fn fragment_wallpaper(input: BlurVarying) -> Vec4f {
        let fc = wp_frag_coord(input.position.xy());
        match get!(BLUR_LOCALS).wp_mode.y as i32 {
            1 => {
                return wp_bloom(fc);
            }
            2 => {
                return wp_dusk(fc);
            }
            3 => {
                return wp_shore(fc);
            }
            4 => {
                return wp_riso_density(fc);
            }
            5 => {
                return wp_riso_print(fc);
            }
            6 => {
                return wp_intro_density(fc);
            }
            7 => {
                return wp_silk(fc);
            }
            8 => {
                return wp_riso_flow(fc);
            }
            9 => {
                return wp_lamina(fc);
            }
            10 => {
                return wp_diatom(fc);
            }
            11 => {
                return wp_diatom_overlay(fc);
            }
            12 => {
                return wp_photo_clouds(fc);
            }
            13 => {
                return wp_photo_daisies(fc);
            }
            14 => {
                return wp_stipple(fc);
            }
            15 => {
                return wp_stipple_overlay(fc);
            }
            16 => {
                return wp_grain(fc);
            }
            17 => {
                return wp_image(fc);
            }
            18 => {
                return wp_cherry_blossom(fc);
            }
            _ => {
                return vec4f(0.0, 0.0, 0.0, 0.0);
            }
        }
    }
}

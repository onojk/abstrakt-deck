struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct FrameUniforms {
    resolution_x:  f32,
    resolution_y:  f32,
    frame_color_r: f32,
    frame_color_g: f32,
    frame_color_b: f32,
    frame_color_a: f32,
    // 0 = none, 1 = circle, 2 = square, 3 = rounded, 4 = hexagon, 5 = octagon, 6 = star
    frame_shape:   f32,
    frame_size:    f32,  // fraction of half-screen the frame inscribes, typically 0.7–0.95
};

@group(0) @binding(0) var<uniform> frame:          FrameUniforms;
@group(0) @binding(1) var          kaleido_tex:    texture_2d<f32>;
@group(0) @binding(2) var          kaleido_sampler: sampler;

const TAU: f32 = 6.28318530718;
const PI:  f32 = 3.14159265359;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    var out: VertexOutput;
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

fn sdf_circle(p: vec2<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_square(p: vec2<f32>, r: f32) -> f32 {
    let d = abs(p) - vec2<f32>(r);
    return max(d.x, d.y);
}

fn sdf_rounded(p: vec2<f32>, r: f32, corner: f32) -> f32 {
    let d = abs(p) - vec2<f32>(r - corner);
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0) - corner;
}

// Regular N-sided polygon. PI/sides offset orients a flat edge at the top.
// Double-modulo pattern ensures non-negative result (WGSL % is IEEE 754 remainder).
fn sdf_polygon(p: vec2<f32>, r: f32, sides: f32) -> f32 {
    let wedge = TAU / sides;
    let offset = atan2(p.y, p.x) + PI / sides;
    let folded_angle = (offset % wedge + wedge) % wedge - wedge * 0.5;
    return length(p) * cos(folded_angle) - r;
}

// Rounded-petal flower (5 petals). Same petal-edge-projection as sdf_star but with
// a larger inner radius ratio (0.55) making the indents shallower and petal tips rounder.
fn sdf_flower(p: vec2<f32>, r: f32) -> f32 {
    let num_points = 5.0;
    let inner_r    = r * 0.55;

    let angle    = atan2(p.y, p.x);
    let radius   = length(p);
    let segment  = TAU / num_points;
    let half_seg = segment * 0.5;

    let local = ((angle % segment) + segment) % segment - half_seg;

    let tip    = vec2<f32>(r,                          0.0);
    let corner = vec2<f32>(inner_r * cos(half_seg), inner_r * sin(half_seg));
    let edge_dir    = normalize(corner - tip);
    let edge_normal = vec2<f32>(-edge_dir.y, edge_dir.x);

    let local_p = vec2<f32>(radius * cos(local), radius * sin(abs(local)));
    return -dot(local_p - tip, edge_normal);
}

// Sharp 5-pointed star. inner_r = 0.4 × r gives deep concave indents and pointed tips.
fn sdf_star(p: vec2<f32>, r: f32) -> f32 {
    let num_points = 5.0;
    let inner_r    = r * 0.4;

    let angle    = atan2(p.y, p.x);
    let radius   = length(p);
    let segment  = TAU / num_points;
    let half_seg = segment * 0.5;

    // Double-modulo: WGSL % is IEEE 754 remainder, not modulo (breaks for negative angles).
    let local = ((angle % segment) + segment) % segment - half_seg;

    let tip    = vec2<f32>(r,                          0.0);
    let corner = vec2<f32>(inner_r * cos(half_seg), inner_r * sin(half_seg));
    let edge_dir    = normalize(corner - tip);
    let edge_normal = vec2<f32>(-edge_dir.y, edge_dir.x); // 90° CCW perpendicular

    let local_p = vec2<f32>(radius * cos(local), radius * sin(abs(local)));

    // Negative inside, positive outside — matches frame.wgsl sign convention.
    return -dot(local_p - tip, edge_normal);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let kaleido_color = textureSample(kaleido_tex, kaleido_sampler, in.uv);

    // frame_shape = 0 → passthrough (no frame)
    if frame.frame_shape < 0.5 {
        return kaleido_color;
    }

    // Aspect-corrected centred coords: x in [-aspect,aspect], y in [-1,1].
    let aspect = frame.resolution_x / frame.resolution_y;
    var p = in.uv * 2.0 - 1.0;
    p.x *= aspect;

    let r = frame.frame_size;
    var dist: f32;
    if frame.frame_shape < 1.5 {
        dist = sdf_circle(p, r);
    } else if frame.frame_shape < 2.5 {
        dist = sdf_square(p, r);
    } else if frame.frame_shape < 3.5 {
        dist = sdf_rounded(p, r, r * 0.2);
    } else if frame.frame_shape < 4.5 {
        dist = sdf_polygon(p, r, 6.0);   // hexagon
    } else if frame.frame_shape < 5.5 {
        dist = sdf_polygon(p, r, 8.0);   // octagon
    } else if frame.frame_shape < 6.5 {
        dist = sdf_flower(p, r);
    } else {
        dist = sdf_star(p, r);
    }

    // 1.5-pixel anti-aliased edge: mask=0 inside (kaleido), mask=1 outside (frame).
    let aa = 1.5 / frame.resolution_y;
    let mask = smoothstep(-aa, aa, dist);

    let fc = vec4<f32>(frame.frame_color_r, frame.frame_color_g,
                       frame.frame_color_b, frame.frame_color_a);
    return vec4<f32>(mix(kaleido_color.rgb, fc.rgb, mask * fc.a), 1.0);
}

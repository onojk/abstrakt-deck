// blackhole.wgsl — Pass 5 (conditional): classic video-feedback recursion.
// Each frame the previous feedback output is sampled through a slightly-enlarged
// UV window (shrink_rate < 1 shrinks prev content toward center), producing an
// infinite-tunnel / two-mirrors effect without any beat-triggered snapshots.
// 4 bindings: uniform[0], prev_feedback[1], scene[2], sampler[3].
// Pipeline blend: REPLACE (compositing fully internal; output alpha = 1.0).

struct FeedbackUniforms {
    center_x:     f32,   // offset  0: tunnel vanishing-point U (wanders per frame)
    center_y:     f32,   // offset  4: tunnel vanishing-point V
    shrink_rate:  f32,   // offset  8: per-frame UV scale (< 1 shrinks prev into center)
    strength:     f32,   // offset 12: feedback blend weight (0 = live only)
    alpha_radius: f32,   // offset 16: spatial edge-fade start (0..1, fraction of half-diagonal)
    _pad0:        f32,   // offset 20
    _pad1:        f32,   // offset 24
    _pad2:        f32,   // offset 28
};                       // 32 bytes — 2 × vec4 rows, satisfies WGSL uniform alignment

@group(0) @binding(0) var<uniform> u: FeedbackUniforms;
@group(0) @binding(1) var prev_tex:  texture_2d<f32>;
@group(0) @binding(2) var scene_tex: texture_2d<f32>;
@group(0) @binding(3) var samp:      sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VertexOutput;
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv     = in.uv;
    let center = vec2<f32>(u.center_x, u.center_y);
    let dir    = uv - center;

    // Spatial alpha: live scene bleeds through at screen edges.
    // sqrt(0.5) = distance from (0.5,0.5) to corner in UV space.
    let max_r: f32    = 0.7071;
    let r: f32        = length(uv - vec2<f32>(0.5));
    let spatial_alpha = 1.0 - smoothstep(u.alpha_radius * max_r, max_r, r);

    // Map this pixel to where it came from in the previous frame.
    // Dividing by shrink_rate < 1 expands the lookup outward, which makes
    // the previous frame's content appear to shrink toward center each frame.
    let prev_uv = center + dir / u.shrink_rate;

    // Analytically zero out-of-bounds samples — avoids non-uniform control
    // flow restriction on textureSample while still zero-ing border pixels.
    let ib        = step(vec2<f32>(0.0), prev_uv) * step(prev_uv, vec2<f32>(1.0));
    let in_bounds = ib.x * ib.y;

    let prev  = textureSample(prev_tex,  samp, clamp(prev_uv, vec2<f32>(0.0), vec2<f32>(1.0)));
    let live  = textureSample(scene_tex, samp, uv);

    let alpha = u.strength * spatial_alpha * in_bounds;
    let rgb   = mix(live.rgb, prev.rgb, alpha);

    return vec4<f32>(rgb, 1.0);
}

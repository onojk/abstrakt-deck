// spin_trail.wgsl
// Slice 26 — rotates the kaleido FBO around its center and blends with a
// persistent history texture for motion-blur trails. Sits between the
// existing kaleido pass and the frame pass.
//
// Aspect-corrected: rotation looks circular at any window aspect ratio
// (otherwise a 16:9 viewport produces visibly elliptical spin).
//
// Uniform layout follows CLAUDE.md Rule 1: flat f32 fields only, no
// vec2<f32> in uniform structs. Total size is 16 bytes (one vec4 slot).

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct SpinTrailUniforms {
    resolution_x: f32,
    resolution_y: f32,
    spin_angle:   f32,   // radians; positive = CCW
    trail_decay:  f32,   // [0, 0.98]
};

@group(0) @binding(0) var<uniform> u:               SpinTrailUniforms;
@group(0) @binding(1) var          kaleido_tex:     texture_2d<f32>;
@group(0) @binding(2) var          kaleido_sampler: sampler;
@group(0) @binding(3) var          history_tex:     texture_2d<f32>;
@group(0) @binding(4) var          history_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    var out: VertexOutput;
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let aspect = u.resolution_x / u.resolution_y;

    // Map uv [0,1] to centred aspect-corrected coords so rotation is circular.
    // p.x in [-aspect, aspect], p.y in [-1, 1].
    var p = in.uv * 2.0 - 1.0;
    p.x *= aspect;

    // Inverse rotation: to find what pixel of the SOURCE texture lands
    // here under a +spin_angle rotation, we rotate the destination
    // coords by -spin_angle.
    let c = cos(-u.spin_angle);
    let s = sin(-u.spin_angle);
    let rotated = vec2<f32>(
        p.x * c - p.y * s,
        p.x * s + p.y * c,
    );

    // Undo aspect correction and remap to [0,1] for sampling.
    let sample_uv = vec2<f32>(
        rotated.x / aspect * 0.5 + 0.5,
        rotated.y          * 0.5 + 0.5,
    );

    // Out-of-bounds reads return black — gives a clean fade into the void
    // at the corners as the mandala rotates.
    var new_color: vec4<f32>;
    if sample_uv.x < 0.0 || sample_uv.x > 1.0 ||
       sample_uv.y < 0.0 || sample_uv.y > 1.0 {
        new_color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    } else {
        new_color = textureSample(kaleido_tex, kaleido_sampler, sample_uv);
    }

    // History is read at the SAME screen-space UV (already in the rotated
    // domain from prior frames), so we don't rotate it again.
    let history_color = textureSample(history_tex, history_sampler, in.uv);

    // Standard motion-blur feedback blend.
    let decay = clamp(u.trail_decay, 0.0, 0.98);
    let out_rgb = new_color.rgb * (1.0 - decay) + history_color.rgb * decay;

    return vec4<f32>(out_rgb, 1.0);
}

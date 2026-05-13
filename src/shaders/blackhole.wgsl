// blackhole.wgsl — Pass 5 (conditional): beat-synced snapshot-warp.
// Reads a frozen snapshot of the scene and a live scene texture.
// The snapshot warps radially toward center as it fades, creating a
// "spaghetti-ghost" heartbeat effect synced to audio beats.
// Replaces the normal blit pass when blackhole_enabled is true.

struct BlackholeUniforms {
    warp_strength:  f32,
    warp_curve:     f32,
    alpha_radius:   f32,
    cycle_progress: f32,
};

@group(0) @binding(0) var<uniform> u: BlackholeUniforms;
@group(0) @binding(1) var snapshot_tex: texture_2d<f32>;
@group(0) @binding(2) var scene_tex:    texture_2d<f32>;
@group(0) @binding(3) var samp:         sampler;

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
    let uv = in.uv;
    let dir = uv - vec2<f32>(0.5, 0.5);
    let r = length(dir);

    // sqrt(0.5) = distance from center to screen corner in UV space.
    let max_r: f32 = 0.7071;

    // Spatial alpha: solid in center, fades to 0 at screen edges.
    // Snapshot is fully opaque inside alpha_radius * max_r, transparent at max_r.
    let spatial_alpha = 1.0 - smoothstep(u.alpha_radius * max_r, max_r, r);

    // Temporal alpha: snapshot fades over its lifetime.
    let temporal_alpha = 1.0 - u.cycle_progress;

    // Radial warp: pull snapshot UV toward center. Edge pixels stretch more
    // than center pixels (power curve on r/max_r).
    let warp_amount = u.cycle_progress * u.warp_strength * pow(r / max_r, u.warp_curve);
    let warp_factor = 1.0 - warp_amount;
    let warped_uv = vec2<f32>(0.5, 0.5) + dir * warp_factor;
    let safe_uv = clamp(warped_uv, vec2<f32>(0.0), vec2<f32>(1.0));

    let snap = textureSample(snapshot_tex, samp, safe_uv);
    let live = textureSample(scene_tex, samp, uv);

    let final_alpha = spatial_alpha * temporal_alpha;
    let final_color = mix(live.rgb, snap.rgb, final_alpha);
    return vec4<f32>(final_color, 1.0);
}

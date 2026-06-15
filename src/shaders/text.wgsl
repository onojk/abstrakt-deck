// SDF lyric-text overlay shader.
//
// Uniform layout uses flat f32 fields (no vec2 in uniform block — CLAUDE.md rule 1).
// The `legibility` field is stubbed for Slice 2's shape↔word morph warp.

struct TextUniforms {
    color_r:    f32,  // offset  0
    color_g:    f32,  // offset  4
    color_b:    f32,  // offset  8
    color_a:    f32,  // offset 12
    legibility: f32,  // offset 16  — Slice 2 hook; 1.0 = full text in Slice 1
    _pad0:      f32,  // offset 20
    _pad1:      f32,  // offset 24
    _pad2:      f32,  // offset 28  — total 32 bytes, two vec4 slots
};

@group(0) @binding(0) var atlas_tex: texture_2d<f32>;
@group(0) @binding(1) var atlas_smp: sampler;
@group(0) @binding(2) var<uniform> u: TextUniforms;

struct VertOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv_x: f32,
    @location(1) uv_y: f32,
};

@vertex
fn vs_main(
    @location(0) pos_x: f32,
    @location(1) pos_y: f32,
    @location(2) uv_x:  f32,
    @location(3) uv_y:  f32,
) -> VertOut {
    var out: VertOut;
    out.clip = vec4<f32>(pos_x, pos_y, 0.0, 1.0);
    out.uv_x = uv_x;
    out.uv_y = uv_y;
    return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let dist = textureSample(atlas_tex, atlas_smp, vec2<f32>(in.uv_x, in.uv_y)).r;

    // Slice 2 warp hook: shape↔word morph driven by legibility.
    // At legibility == 1.0 (Slice 1) this branch is never taken.
    if (u.legibility < 1.0) {
        // TODO Slice 2: apply SDF warp / dissolve here.
        // The fragment can sample a noise field or morph UVs using legibility.
    }

    // Crisp SDF edge with a narrow smoothstep band.
    // Edge width of 0.04 gives ~2px of antialiasing at 64-cell resolution.
    let edge_half = 0.04;
    let alpha = smoothstep(0.5 - edge_half, 0.5 + edge_half, dist) * u.color_a;

    return vec4<f32>(u.color_r, u.color_g, u.color_b, alpha);
}

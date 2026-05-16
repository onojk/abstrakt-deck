struct ChromaUniforms {
    key_color_r:   f32,  // offset  0
    key_color_g:   f32,  // offset  4
    key_color_b:   f32,  // offset  8
    key_tolerance: f32,  // offset 12
    key_softness:  f32,  // offset 16
    key_strength:  f32,  // offset 20
    opacity:       f32,  // offset 24
    _pad:          f32,  // offset 28  — total 32 bytes
};

@group(0) @binding(0) var<uniform> chroma: ChromaUniforms;
@group(0) @binding(1) var bg_tex:    texture_2d<f32>;
@group(0) @binding(2) var ghost_tex: texture_2d<f32>;
@group(0) @binding(3) var smp:       sampler;

struct Vary { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> Vary {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: Vary;
    out.pos = vec4<f32>(x * 2.0 - 1.0, -(y * 2.0 - 1.0), 0.0, 1.0);
    out.uv  = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: Vary) -> @location(0) vec4<f32> {
    let bg    = textureSample(bg_tex,    smp, in.uv);
    let ghost = textureSample(ghost_tex, smp, in.uv);

    let key  = vec3<f32>(chroma.key_color_r, chroma.key_color_g, chroma.key_color_b);
    let diff = length(ghost.rgb - key);
    let mask = smoothstep(
        chroma.key_tolerance,
        chroma.key_tolerance + chroma.key_softness,
        diff,
    ) * chroma.key_strength;

    let blended = mix(bg.rgb, ghost.rgb, mask * chroma.opacity);
    return vec4<f32>(blended, 1.0);
}

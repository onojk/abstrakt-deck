// Explosion overlay: additively blends chunk quads over the scene.
//
// Each quad vertex carries a clip-space position, a scene UV to sample, and
// an alpha weight. The fragment samples the scene texture at that UV and
// returns (scene_color, alpha) — the pipeline's SrcAlpha/One blend does the rest.

struct VsOut {
    @builtin(position) clip:  vec4<f32>,
    @location(0)       uv:    vec2<f32>,
    @location(1)       alpha: f32,
}

@vertex
fn vs_main(
    @location(0) pos:   vec2<f32>,
    @location(1) uv:    vec2<f32>,
    @location(2) alpha: f32,
) -> VsOut {
    return VsOut(vec4<f32>(pos, 0.0, 1.0), uv, alpha);
}

@group(0) @binding(0) var scene_tex:  texture_2d<f32>;
@group(0) @binding(1) var scene_samp: sampler;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv    = clamp(in.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let color = textureSample(scene_tex, scene_samp, uv);
    return vec4<f32>(color.rgb, in.alpha);
}

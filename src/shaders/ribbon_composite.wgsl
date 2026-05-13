// ribbon_composite.wgsl
// Pass-through blit of the ribbon FBO onto the painter FBO.
// Additive blend (src=One, dst=One) is configured in the pipeline blend state,
// so this shader just samples and returns the ribbon texture.

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var ribbon_tex:  texture_2d<f32>;
@group(0) @binding(1) var ribbon_samp: sampler;

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
    return textureSample(ribbon_tex, ribbon_samp, in.uv);
}

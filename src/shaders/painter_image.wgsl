// painter_image.wgsl
// Tiles the bundled 512×512 source image 8× across the 4096-wide painter
// texture. Same texture+sampler BGL as the Skin painter.

@group(0) @binding(0) var img_tex:  texture_2d<f32>;
@group(0) @binding(1) var img_samp: sampler;

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VertexOut;
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // 8 tiles across the 4096px wide painter texture (4096 / 512 = 8).
    // V maps the 512-tall source to the 256-tall painter strip.
    let tiled_uv = vec2<f32>(fract(in.uv.x * 8.0), in.uv.y);
    return textureSample(img_tex, img_samp, tiled_uv);
}

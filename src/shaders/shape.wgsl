struct Transform {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> transform: Transform;

@group(1) @binding(0) var painter_tex:     texture_2d<f32>;
@group(1) @binding(1) var painter_sampler: sampler;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_position = transform.mvp * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(painter_tex, painter_sampler, in.uv);
}

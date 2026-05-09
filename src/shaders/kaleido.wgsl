struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// All three bindings live in group 0 for this fullscreen pass.
struct KaleidoUniforms {
    resolution_x: f32,
    resolution_y: f32,
    fold_count:   f32,
    zoom:         f32,
};

@group(0) @binding(0) var<uniform> kaleido:        KaleidoUniforms;
@group(0) @binding(1) var          shape_tex:      texture_2d<f32>;
@group(0) @binding(2) var          shape_sampler:  sampler;

const TAU: f32 = 6.28318530718;

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
    let aspect = kaleido.resolution_x / kaleido.resolution_y;

    // Centred, aspect-corrected coords: x in [-aspect, aspect], y in [-1, 1].
    var p = in.uv * 2.0 - 1.0;
    p.x *= aspect;

    // Polar
    let r     = length(p);
    let theta = atan2(p.y, p.x);    // [-π, π]

    // Fold into one wedge of size TAU/fold_count, then mirror within wedge.
    let wedge  = TAU / kaleido.fold_count;
    var folded = (theta % wedge + wedge) % wedge;   // [0, wedge)
    if folded > wedge * 0.5 {
        folded = wedge - folded;
    }

    // Reconstruct cartesian at folded angle, apply zoom (zoom < 1 zooms IN).
    let fp = vec2<f32>(cos(folded), sin(folded)) * r * kaleido.zoom;

    // Undo aspect and remap to [0,1] UV for shape FBO sampling.
    let sample_uv = vec2<f32>(
        fp.x / aspect * 0.5 + 0.5,
        fp.y          * 0.5 + 0.5,
    );

    if sample_uv.x < 0.0 || sample_uv.x > 1.0 ||
       sample_uv.y < 0.0 || sample_uv.y > 1.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    return textureSample(shape_tex, shape_sampler, sample_uv);
}

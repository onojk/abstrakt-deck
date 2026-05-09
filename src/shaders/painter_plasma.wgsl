struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct GlobalUniforms {
    time_seconds: f32,
    resolution_x: f32,
    resolution_y: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> globals: GlobalUniforms;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    var out: VertexOutput;
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let h6 = h * 6.0;
    let c = v * s;
    let x = c * (1.0 - abs((h6 % 2.0) - 1.0));
    let m = v - c;
    var rgb: vec3<f32>;
    if (h6 < 1.0) { rgb = vec3<f32>(c, x, 0.0); }
    else if (h6 < 2.0) { rgb = vec3<f32>(x, c, 0.0); }
    else if (h6 < 3.0) { rgb = vec3<f32>(0.0, c, x); }
    else if (h6 < 4.0) { rgb = vec3<f32>(0.0, x, c); }
    else if (h6 < 5.0) { rgb = vec3<f32>(x, 0.0, c); }
    else { rgb = vec3<f32>(c, 0.0, x); }
    return rgb + vec3<f32>(m);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv * 4.0 - vec2<f32>(2.0);
    let t = globals.time_seconds;

    var v: f32 = 0.0;
    v = v + sin(uv.x * 1.5 + t * 0.7);
    v = v + sin(uv.y * 1.5 + t * 0.9);
    v = v + sin((uv.x + uv.y) * 1.0 + t * 0.5);
    v = v + sin(length(uv) * 2.0 - t * 0.6);
    v = v * 0.25;

    let hue = fract(v * 0.5 + 0.5 + t * 0.03);

    let rgb = hsv2rgb(hue, 0.9, 0.95);
    return vec4<f32>(rgb, 1.0);
}

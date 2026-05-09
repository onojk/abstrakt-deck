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
    let aspect = globals.resolution_x / globals.resolution_y;
    var p = in.uv * 2.0 - vec2<f32>(1.0);
    p.x = p.x * aspect;

    let r = length(p);
    let theta = atan2(p.y, p.x);
    let t = globals.time_seconds;

    let arm_count = 5.0;
    let hue = fract(theta / 6.28318 * arm_count + r * 1.5 - t * 0.1);

    let brightness = 0.7 + 0.3 * (1.0 - r * 0.4);

    let rgb = hsv2rgb(hue, 1.0, brightness);
    return vec4<f32>(rgb, 1.0);
}

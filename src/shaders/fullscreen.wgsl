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

// HSV → RGB. h in [0,1) (not degrees), s and v in [0,1].
fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let h6 = h * 6.0;
    let c = v * s;
    let x = c * (1.0 - abs((h6 % 2.0) - 1.0));
    let m = v - c;
    var rgb: vec3<f32>;
    if h6 < 1.0      { rgb = vec3<f32>(c, x, 0.0); }
    else if h6 < 2.0 { rgb = vec3<f32>(x, c, 0.0); }
    else if h6 < 3.0 { rgb = vec3<f32>(0.0, c, x); }
    else if h6 < 4.0 { rgb = vec3<f32>(0.0, x, c); }
    else if h6 < 5.0 { rgb = vec3<f32>(x, 0.0, c); }
    else             { rgb = vec3<f32>(c, 0.0, x); }
    return rgb + vec3<f32>(m);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let t  = globals.time_seconds;

    // Hue Stripe painter — vertical color bands scrolling left over time.
    // Port of PAINTER_HUESTRIPE_FRAG from abstrakt-engine (GLSL → WGSL).
    let stripe_count  = 6.0;
    let scroll_speed  = 0.05;   // hue cycles per second
    let hue = fract(uv.x * stripe_count - t * scroll_speed * stripe_count);

    // Brightness peaks at vertical centre, dims toward top/bottom edges.
    let brightness = 1.0 - abs(uv.y - 0.5) * 0.6;

    return vec4<f32>(hsv2rgb(hue, 1.0, brightness), 1.0);
}

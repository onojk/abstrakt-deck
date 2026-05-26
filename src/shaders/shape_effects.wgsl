struct ShapeEffects {
    invert:               f32,
    colorize_enabled:     f32,
    colorize_hue:         f32,
    colorize_intensity:   f32,
    distortion_enabled:   f32,
    distortion_amplitude: f32,
    distortion_frequency: f32,
    time_seconds:         f32,
    painter_scroll_phase: f32,
    contrast:             f32,
    saturation:           f32,
    contrast_passes:      f32,
};

@group(0) @binding(0) var<uniform> effects: ShapeEffects;
@group(0) @binding(1) var shape_tex:     texture_2d<f32>;
@group(0) @binding(2) var shape_sampler: sampler;

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VertexOut;
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4<f32>(c.bg, K.wz), vec4<f32>(c.gb, K.xy), step(c.b, c.g));
    let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), step(p.x, c.r));
    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(vec3<f32>(c.x) + K.xyz) * 6.0 - vec3<f32>(K.w));
    return c.z * mix(vec3<f32>(K.x), clamp(p - vec3<f32>(K.x), vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var sample_uv = in.uv;
    if effects.distortion_enabled > 0.5 {
        let freq = effects.distortion_frequency * 6.2831853;
        let wave_u = sin(in.uv.y * freq + effects.time_seconds) * effects.distortion_amplitude;
        let wave_v = sin(in.uv.x * freq + effects.time_seconds) * effects.distortion_amplitude;
        sample_uv = vec2<f32>(in.uv.x + wave_u, in.uv.y + wave_v);
    }

    var color = textureSample(shape_tex, shape_sampler, sample_uv);

    // Contrast: N clamped passes — each pass pushes midtones toward extremes
    let contrast_passes = i32(effects.contrast_passes);
    for (var i = 0; i < contrast_passes; i = i + 1) {
        let new_rgb = (color.rgb - vec3<f32>(0.5)) * effects.contrast + vec3<f32>(0.5);
        color = vec4<f32>(clamp(new_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), color.a);
    }

    // Saturation: blend between luma and full color
    let luma = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    color = vec4<f32>(clamp(mix(vec3<f32>(luma), color.rgb, effects.saturation), vec3<f32>(0.0), vec3<f32>(1.0)), color.a);

    // Invert: flip RGB
    let inverted = vec3<f32>(1.0) - color.rgb;
    color = vec4<f32>(mix(color.rgb, inverted, effects.invert), color.a);

    // Colorize: replace hue then mix back toward original by intensity
    if effects.colorize_enabled > 0.5 {
        let hsv = rgb2hsv(color.rgb);
        let target_hue = effects.colorize_hue / 360.0;
        let colorized = hsv2rgb(vec3<f32>(target_hue, hsv.y, hsv.z));
        color = vec4<f32>(mix(color.rgb, colorized, effects.colorize_intensity), color.a);
    }

    return color;
}

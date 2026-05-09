struct Transform {
    mvp: mat4x4<f32>,
};

struct ShapeEffects {
    invert:             f32,
    colorize_enabled:   f32,
    colorize_hue:       f32,
    colorize_intensity: f32,
};

@group(0) @binding(0) var<uniform> transform: Transform;

@group(1) @binding(0) var painter_tex:     texture_2d<f32>;
@group(1) @binding(1) var painter_sampler: sampler;

@group(2) @binding(0) var<uniform> effects: ShapeEffects;

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
    var color = textureSample(painter_tex, painter_sampler, in.uv);

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

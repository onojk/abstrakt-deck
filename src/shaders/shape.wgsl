struct Transform {
    mvp: mat4x4<f32>,
};

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

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Rotation-driven scroll: shape samples a 0.25-wide window of the 4096-wide painter.
    // The window slides by 0.25 per revolution, so 4 rotations = 1 full painter cycle.
    let scroll_u = fract(in.uv.x * 0.25 + effects.painter_scroll_phase);
    return textureSample(painter_tex, painter_sampler, vec2<f32>(scroll_u, in.uv.y));
}

// ribbons.wgsl
// Ping-pong ribbon accumulator: 4 sinusoidal Gaussian strokes.
// Each frame: decay prev × 0.992, beat-driven amplitude collapse,
// draw 4 ribbons across the 4096×256 painter strip.

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct RibbonUniforms {
    time_seconds: f32,
    beat_decay:   f32,
    intensity:    f32,
    _pad0:        f32,
    // .xyz = rgb color, .w = oscillation speed (rad/s)
    colors: array<vec4<f32>, 4>,
    // .x = frequency (cycles across width), .y = phase offset,
    // .z = Gaussian stroke half-width (UV units), .w = amplitude (UV units)
    params: array<vec4<f32>, 4>,
};

@group(0) @binding(0) var<uniform> u: RibbonUniforms;
@group(0) @binding(1) var prev_tex:  texture_2d<f32>;
@group(0) @binding(2) var prev_samp: sampler;

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
    let t = u.time_seconds;

    // Decay previous frame content
    var color = textureSample(prev_tex, prev_samp, in.uv).rgb * 0.992;

    // Beat-driven collapse: beat onset compresses ribbon brightness
    color = color * (1.0 - u.beat_decay * 0.3);

    // Draw 4 sinusoidal ribbons as Gaussian strokes
    for (var i: i32 = 0; i < 4; i++) {
        let c = u.colors[i];
        let p = u.params[i];
        let freq  = p.x;
        let phase = p.y;
        let width = p.z;
        // Beat collapse: amplitude shrinks on beat onset, restores over time
        let amp   = p.w * (1.0 - u.beat_decay * 0.5);
        let speed = c.w;

        let y_ribbon = 0.5 + amp * sin(freq * in.uv.x * 6.283185 + phase + t * speed);
        let dist     = in.uv.y - y_ribbon;
        let stroke   = exp(-(dist * dist) / (2.0 * width * width));

        // (1.0 - 0.992) per-frame stroke scale; keeps the running accumulation
        // at steady-state ≈ stroke peak instead of ~125× larger. Without this,
        // ribbons saturate to white when composited onto the Rgba8Unorm painter.
        color += stroke * c.rgb * u.intensity * 0.008;
    }

    return vec4<f32>(color, 1.0);
}

// painter_print_head.wgsl
// Dot-matrix / ink-head painter.
// Beat pulse uses bass_zoom_smoothed (EMA-smoothed bass energy) with a
// threshold+boost to extract per-beat spikes. True onset-driven beat
// decay arrives in a future slice.

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct PainterAudioUniforms {
    time_seconds: f32,
    bass:         f32,
    mid:          f32,
    _pad:         f32,
};

@group(0) @binding(0) var<uniform> audio: PainterAudioUniforms;

const COLS: f32 = 24.0;
const ROWS: f32 = 8.0;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VertexOutput;
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let h6 = h * 6.0;
    let c  = v * s;
    let x  = c * (1.0 - abs((h6 % 2.0) - 1.0));
    let m  = v - c;
    var rgb: vec3<f32>;
    if      h6 < 1.0 { rgb = vec3<f32>(c, x, 0.0); }
    else if h6 < 2.0 { rgb = vec3<f32>(x, c, 0.0); }
    else if h6 < 3.0 { rgb = vec3<f32>(0.0, c, x); }
    else if h6 < 4.0 { rgb = vec3<f32>(0.0, x, c); }
    else if h6 < 5.0 { rgb = vec3<f32>(x, 0.0, c); }
    else             { rgb = vec3<f32>(c, 0.0, x); }
    return rgb + vec3<f32>(m);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = audio.time_seconds;

    // Threshold + boost to turn the continuous EMA into a snappier pulse.
    let pulse = clamp(max(0.0, audio.bass - 0.4) * 1.5, 0.0, 1.0);

    // Cell coordinates.
    let cell_uv = vec2<f32>(in.uv.x * COLS, in.uv.y * ROWS);
    let col = floor(cell_uv.x);
    let row = floor(cell_uv.y);

    // Distance from cell centre (0..1 range within cell).
    let local = fract(cell_uv) - vec2<f32>(0.5);

    // Dot radius: base + beat-driven expansion.
    let base_r = 0.30;
    let radius = base_r + pulse * 0.15;
    let dist   = length(local);
    let dot    = step(dist, radius);

    // Hue per column, slow time drift.
    let hue = fract(col / COLS + t * 0.02);

    // Brightness: base 0.25, boosted by pulse on every dot.
    let value = 0.25 + pulse * 0.75;

    let color = hsv2rgb(hue, 0.85, value) * dot;

    // Dark background where no dot.
    let bg = vec3<f32>(0.05);
    let final_rgb = mix(bg, color, dot);

    return vec4<f32>(final_rgb, 1.0);
}

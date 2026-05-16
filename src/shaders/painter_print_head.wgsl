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

struct AppliedHarmony {
    // vec4 0
    enabled:      u32,
    anchor_hue:   f32,
    saturation:   f32,
    value:        f32,
    // vec4 1
    strength:     f32,
    offset_count: u32,
    _pad0:        f32,
    _pad1:        f32,
    // vec4 2-3: up to 8 hue offsets (relative to anchor_hue)
    offsets: array<vec4<f32>, 2>,
};
@group(1) @binding(0) var<uniform> u_harmony: AppliedHarmony;

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

fn ah_rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let d  = mx - mn;
    var h: f32 = 0.0;
    if d > 1e-6 {
        if mx == c.r      { h = 60.0 * ((c.g - c.b) / d % 6.0); }
        else if mx == c.g { h = 60.0 * ((c.b - c.r) / d + 2.0); }
        else              { h = 60.0 * ((c.r - c.g) / d + 4.0); }
        if h < 0.0 { h = h + 360.0; }
    }
    let s = select(0.0, d / mx, mx > 1e-6);
    return vec3<f32>(h, s, mx);
}

fn ah_hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h = hsv.x; let s = hsv.y; let v = hsv.z;
    let c = v * s;
    let h6 = h / 60.0;
    let x = c * (1.0 - abs(h6 % 2.0 - 1.0));
    let m = v - c;
    var rgb: vec3<f32>;
    let h6i = i32(h6);
    if      h6i == 0 { rgb = vec3<f32>(c, x, 0.0); }
    else if h6i == 1 { rgb = vec3<f32>(x, c, 0.0); }
    else if h6i == 2 { rgb = vec3<f32>(0.0, c, x); }
    else if h6i == 3 { rgb = vec3<f32>(0.0, x, c); }
    else if h6i == 4 { rgb = vec3<f32>(x, 0.0, c); }
    else             { rgb = vec3<f32>(c, 0.0, x); }
    return rgb + vec3<f32>(m, m, m);
}

fn ah_offset_val(i: u32) -> f32 {
    let v = u_harmony.offsets[i / 4u];
    let lane = i % 4u;
    if      lane == 0u { return v.x; }
    else if lane == 1u { return v.y; }
    else if lane == 2u { return v.z; }
    else               { return v.w; }
}

fn ah_hue_delta(a: f32, b: f32) -> f32 {
    var d = (b - a) % 360.0;
    if d > 180.0  { d = d - 360.0; }
    if d < -180.0 { d = d + 360.0; }
    return abs(d);
}

fn ah_nearest_hue(input_hue: f32) -> f32 {
    var best = u_harmony.anchor_hue + ah_offset_val(0u);
    var best_dist = ah_hue_delta(input_hue, best);
    for (var i = 1u; i < u_harmony.offset_count; i = i + 1u) {
        let candidate = u_harmony.anchor_hue + ah_offset_val(i);
        let d = ah_hue_delta(input_hue, candidate);
        if d < best_dist { best_dist = d; best = candidate; }
    }
    var h = best % 360.0;
    if h < 0.0 { h = h + 360.0; }
    return h;
}

fn apply_harmony(c: vec3<f32>) -> vec3<f32> {
    if u_harmony.enabled == 0u { return c; }
    let hsv = ah_rgb_to_hsv(c);
    let target_h = ah_nearest_hue(hsv.x);
    let delta = (target_h - hsv.x + 540.0) % 360.0 - 180.0;
    var new_h = hsv.x + delta * u_harmony.strength;
    new_h = ((new_h % 360.0) + 360.0) % 360.0;
    return ah_hsv_to_rgb(vec3<f32>(new_h, hsv.y, hsv.z));
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
    let final_rgb = apply_harmony(mix(bg, color, dot));

    return vec4<f32>(final_rgb, 1.0);
}

// painter_audio_paint.wgsl
// 8-band audio visualizer: one horizontal row per band, hue-coded.
// Band cutoffs match Android parity (60-120, 120-250, 250-500, 500-1k,
// 1k-2k, 2k-4k, 4k-8k, 8k-16k Hz). Beat flash via beat_decay.
//
// NOTE: WGSL uniform arrays require stride ≥ 16 bytes per element.
// array<f32, 8> has stride 4 and fails validation. We pack the 8 f32
// band values into array<vec4<f32>, 2> (stride 16) which has the same
// 32-byte memory footprint as the Rust [f32; 8] at the same offset.

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct PainterAudioUniforms {
    time_seconds: f32,
    bass:         f32,
    mid:          f32,
    beat_decay:   f32,
    bands:        array<vec4<f32>, 2>,  // [0].xyzw = bands 0-3, [1].xyzw = bands 4-7
};

@group(0) @binding(0) var<uniform> audio: PainterAudioUniforms;

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

// Extract band i (0-7) from the vec4-packed bands array.
fn band(i: i32) -> f32 {
    let v = audio.bands[i / 4];
    let lane = i % 4;
    if      lane == 0 { return v.x; }
    else if lane == 1 { return v.y; }
    else if lane == 2 { return v.z; }
    else              { return v.w; }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = audio.time_seconds;

    // Row 0 (top) = band 0 (sub-bass), row 7 (bottom) = band 7 (air/treble)
    let row_f = clamp(floor(in.uv.y * 8.0), 0.0, 7.0);
    let row   = i32(row_f);

    let band_energy = band(row);

    // Evenly spaced hues across the spectrum; slow time drift + U wave
    let hue_base = row_f / 8.0;
    let hue = fract(hue_base + t * 0.005 + in.uv.x * 0.03);

    let value = clamp(band_energy * 3.0 + 0.1, 0.0, 1.0);

    // Beat flash: all rows brighten simultaneously on onset
    let flash = audio.beat_decay * 0.3;

    return vec4<f32>(hsv2rgb(hue, 1.0, clamp(value + flash, 0.0, 1.0)), 1.0);
}

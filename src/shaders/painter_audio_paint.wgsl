// painter_audio_paint.wgsl
// Currently 2-band (bass + mid). Will be upgraded to 8-band when analyzer
// is upgraded in slice 24s.

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

    // Lower half [0, 0.5) → bass, upper half (0.5, 1.0] → mid.
    // Smooth blend at UV.y == 0.5 using a 0.05-wide smoothstep band.
    let blend = smoothstep(0.45, 0.55, in.uv.y);  // 0 = bass, 1 = mid
    let energy = mix(audio.bass, audio.mid, blend);

    // Hue drifts slowly with time; U adds a gentle wave so the stripe
    // isn't a perfectly flat color bar.
    let hue_base_bass = 0.0;   // red-orange
    let hue_base_mid  = 0.55;  // cyan-blue
    let hue_base = mix(hue_base_bass, hue_base_mid, blend);
    let hue = fract(hue_base + t * 0.01 + in.uv.x * 0.04);

    let value = clamp(energy * 3.0 + 0.15, 0.0, 1.0);

    return vec4<f32>(hsv2rgb(hue, 1.0, value), 1.0);
}

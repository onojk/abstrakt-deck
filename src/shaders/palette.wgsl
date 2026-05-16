// palette.wgsl — Pass 1c: palette-mode HSV clamp applied to the painter FBO.
// Reads painter texture, writes clamped result to a scratch texture.
// The scratch is then copy_texture_to_texture'd back to the painter FBO.
// Option A (scratch+copy) chosen over a second painter FBO ping-pong:
// painter is 4096×256×4B = 4MB; the copy costs < 0.1ms and avoids touching
// the kaleido/shape bind group layouts that reference painter_view.

struct PaletteUniforms {
    mode:                u32,              // 0=Off 1=Warm 2=Cool 3=Earth 4=Neon 5=Mono 6=Harmony
    tint:                f32,              // 0.0 = original unchanged, 1.0 = fully clamped
    mono_hue:            f32,              // degrees [0, 360], only used in Mono mode
    harmony_num_offsets: u32,              // how many harmony_offsets[] entries are active
    harmony_anchor_hue:  f32,             // base hue in degrees
    harmony_saturation:  f32,             // target saturation for Harmony mode
    harmony_value:       f32,             // target value for Harmony mode
    harmony_strength:    f32,             // 0..1 blend strength for sat/val snap
    harmony_offsets:     array<vec4<f32>, 2>,  // 8 hues packed; index with harmony_offset_val()
};

@group(0) @binding(0) var<uniform> u: PaletteUniforms;
@group(0) @binding(1) var tex:  texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VertexOutput;
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

fn rgb_to_hsv(rgb: vec3<f32>) -> vec3<f32> {
    let r = rgb.x;
    let g = rgb.y;
    let b = rgb.z;
    let mx = max(max(r, g), b);
    let mn = min(min(r, g), b);
    let delta = mx - mn;
    var h: f32 = 0.0;
    var s: f32 = 0.0;
    let v: f32 = mx;
    if delta > 1e-6 {
        s = delta / mx;
        if mx == r {
            h = 60.0 * ((g - b) / delta % 6.0);
        } else if mx == g {
            h = 60.0 * ((b - r) / delta + 2.0);
        } else {
            h = 60.0 * ((r - g) / delta + 4.0);
        }
        if h < 0.0 { h = h + 360.0; }
    }
    return vec3<f32>(h, s, v);
}

fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h = hsv.x;
    let s = hsv.y;
    let v = hsv.z;
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

// Warm: [330°, 60°] crosses 0° — special wrap-aware clamp.
fn clamp_warm(h: f32) -> f32 {
    if h >= 330.0 || h <= 60.0 { return h; }
    let d60  = abs(h - 60.0);
    let d330 = abs(h - 330.0);
    if d60 < d330 { return 60.0; }
    return 330.0;
}

// Extract one hue from the packed array<vec4<f32>, 2> (holds up to 8 values).
fn harmony_offset_val(i: u32) -> f32 {
    let vec_idx = i / 4u;
    let lane    = i % 4u;
    let v = u.harmony_offsets[vec_idx];
    if      lane == 0u { return v.x; }
    else if lane == 1u { return v.y; }
    else if lane == 2u { return v.z; }
    else               { return v.w; }
}

// Shortest angular distance between two hues in [0, 360).
fn hue_delta(a: f32, b: f32) -> f32 {
    var d = (b - a) % 360.0;
    if d > 180.0  { d = d - 360.0; }
    if d < -180.0 { d = d + 360.0; }
    return abs(d);
}

// Find the harmony hue (from harmony_offsets[0..harmony_num_offsets]) nearest to input_hue.
fn nearest_harmony_hue(input_hue: f32) -> f32 {
    var best_hue  = harmony_offset_val(0u);
    var best_dist = hue_delta(input_hue, best_hue);
    for (var i = 1u; i < u.harmony_num_offsets; i = i + 1u) {
        let candidate = harmony_offset_val(i);
        let dist      = hue_delta(input_hue, candidate);
        if dist < best_dist {
            best_dist = dist;
            best_hue  = candidate;
        }
    }
    return best_hue;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sample = textureSample(tex, samp, in.uv);
    let original_rgb = sample.xyz;
    let a = sample.w;

    if u.mode == 0u { return sample; }

    let hsv = rgb_to_hsv(original_rgb);
    var ch = hsv.x;
    var cs = hsv.y;
    var cv = hsv.z;

    if u.mode == 1u {
        ch = clamp_warm(ch);
    } else if u.mode == 2u {
        ch = clamp(ch, 120.0, 270.0);
    } else if u.mode == 3u {
        ch = clamp(ch, 20.0, 100.0);
        cs = min(cs, 0.5);
    } else if u.mode == 4u {
        cs = 1.0;
        cv = 1.0;
    } else if u.mode == 6u {
        ch = nearest_harmony_hue(ch);
        cs = mix(cs, u.harmony_saturation, u.harmony_strength);
        cv = mix(cv, u.harmony_value,      u.harmony_strength);
    } else {
        ch = u.mono_hue;
    }

    let clamped_rgb = hsv_to_rgb(vec3<f32>(ch, cs, cv));
    // Lerp in RGB space — avoids hue-rotation artifacts at intermediate tint values.
    let out_rgb = mix(original_rgb, clamped_rgb, u.tint);
    return vec4<f32>(out_rgb, a);
}

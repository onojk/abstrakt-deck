// painter_image.wgsl
// Tiles the bundled 512×512 source image 8× across the 4096-wide painter
// texture. Same texture+sampler BGL as the Skin painter.

@group(0) @binding(0) var img_tex:  texture_2d<f32>;
@group(0) @binding(1) var img_samp: sampler;

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

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VertexOut;
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
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
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // 8 tiles across the 4096px wide painter texture (4096 / 512 = 8).
    // V maps the 512-tall source to the 256-tall painter strip.
    let tiled_uv = vec2<f32>(fract(in.uv.x * 8.0), in.uv.y);
    let c = textureSample(img_tex, img_samp, tiled_uv);
    return vec4<f32>(apply_harmony(c.rgb), c.a);
}

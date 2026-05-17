// bezold.wgsl — Bezold simultaneous-contrast post-process.
// Pushes each pixel's hue away from its 8-pixel neighborhood average.

struct BezoldUniforms {
    strength: f32,
    radius:   f32,
    texel_x:  f32,
    texel_y:  f32,
};

@group(0) @binding(0) var scene_tex:  texture_2d<f32>;
@group(0) @binding(1) var scene_samp: sampler;
@group(0) @binding(2) var<uniform> u: BezoldUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    var t = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 2.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(2.0, 0.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(p[vid], 0.0, 1.0);
    out.uv  = t[vid];
    return out;
}

fn rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    let cmax = max(c.r, max(c.g, c.b));
    let cmin = min(c.r, min(c.g, c.b));
    let d = cmax - cmin;
    var h: f32 = 0.0;
    if d > 1e-6 {
        if cmax == c.r {
            h = 60.0 * (((c.g - c.b) / d) - floor(((c.g - c.b) / d) / 6.0) * 6.0);
        } else if cmax == c.g {
            h = 60.0 * (((c.b - c.r) / d) + 2.0);
        } else {
            h = 60.0 * (((c.r - c.g) / d) + 4.0);
        }
    }
    if h < 0.0 { h = h + 360.0; }
    let s = select(0.0, d / cmax, cmax > 1e-6);
    return vec3<f32>(h, s, cmax);
}

fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h = hsv.x;
    let s = clamp(hsv.y, 0.0, 1.0);
    let v = clamp(hsv.z, 0.0, 1.0);
    let c = v * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - abs((h_prime - floor(h_prime / 2.0) * 2.0) - 1.0));
    let m = v - c;
    var rgb: vec3<f32>;
    let sector = i32(floor(h_prime)) % 6;
    if      sector == 0 { rgb = vec3<f32>(c, x, 0.0); }
    else if sector == 1 { rgb = vec3<f32>(x, c, 0.0); }
    else if sector == 2 { rgb = vec3<f32>(0.0, c, x); }
    else if sector == 3 { rgb = vec3<f32>(0.0, x, c); }
    else if sector == 4 { rgb = vec3<f32>(x, 0.0, c); }
    else                { rgb = vec3<f32>(c, 0.0, x); }
    return rgb + vec3<f32>(m, m, m);
}

// Shortest signed arc from avg to src; positive means src is "ahead" of avg.
fn hue_delta(src: f32, avg: f32) -> f32 {
    var d = (src - avg) - floor((src - avg) / 360.0) * 360.0;
    if d > 180.0 { d = d - 360.0; }
    return d;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let src = textureSample(scene_tex, scene_samp, in.uv);

    if u.strength < 0.001 {
        return src;
    }

    let r  = u.radius;
    let dx = u.texel_x * r;
    let dy = u.texel_y * r;

    let n0 = textureSample(scene_tex, scene_samp, in.uv + vec2<f32>(-dx, -dy)).rgb;
    let n1 = textureSample(scene_tex, scene_samp, in.uv + vec2<f32>( 0.0, -dy)).rgb;
    let n2 = textureSample(scene_tex, scene_samp, in.uv + vec2<f32>( dx, -dy)).rgb;
    let n3 = textureSample(scene_tex, scene_samp, in.uv + vec2<f32>(-dx,  0.0)).rgb;
    let n4 = textureSample(scene_tex, scene_samp, in.uv + vec2<f32>( dx,  0.0)).rgb;
    let n5 = textureSample(scene_tex, scene_samp, in.uv + vec2<f32>(-dx,  dy)).rgb;
    let n6 = textureSample(scene_tex, scene_samp, in.uv + vec2<f32>( 0.0,  dy)).rgb;
    let n7 = textureSample(scene_tex, scene_samp, in.uv + vec2<f32>( dx,  dy)).rgb;

    let avg = (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7) / 8.0;

    let src_hsv = rgb_to_hsv(src.rgb);
    let avg_hsv = rgb_to_hsv(avg);

    // Push source hue away from neighborhood average — capped at ±60°.
    let delta = hue_delta(src_hsv.x, avg_hsv.x);
    let push = sign(delta) * min(abs(delta), 60.0) * u.strength;
    let new_hue = src_hsv.x + push;
    let wrapped = new_hue - floor(new_hue / 360.0) * 360.0;

    // Slight saturation boost proportional to local hue contrast.
    let contrast_boost = 1.0 + (abs(delta) / 180.0) * u.strength * 0.3;
    let new_sat = clamp(src_hsv.y * contrast_boost, 0.0, 1.0);

    let out_rgb = hsv_to_rgb(vec3<f32>(wrapped, new_sat, src_hsv.z));
    return vec4<f32>(out_rgb, src.a);
}

// ribbons.wgsl
// Alpha-channel accumulation ribbon pass. Matches Android AbstraktRenderer ribbon pass.
// Four radial rings at V=0.30/0.50/0.72/0.90, smoothstep lines, alpha-max accumulator.
// Blend: update pass writes directly (REPLACE); composite pass uses SRC_ALPHA/ONE_MINUS.

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct RibbonUniforms {
    resolution:   vec2<f32>,           // painter texture size (4096×256)
    time_seconds: f32,
    intensity:    f32,                 // master scale on lineIntensity * 1.3
    color:        vec4<f32>,           // .rgb = ribbon color, .a unused
    collapse:     vec4<f32>,           // per-ribbon collapse 0..1
    bands:        array<vec4<f32>, 2>, // bands[0..4] in [0].xyzw, bands[4..8] in [1].xyzw
};

@group(0) @binding(0) var<uniform> u: RibbonUniforms;
@group(0) @binding(1) var prev_tex:  texture_2d<f32>;
@group(0) @binding(2) var prev_samp: sampler;

fn band(i: i32) -> f32 {
    let v = u.bands[i / 4];
    let lane = i % 4;
    if      lane == 0 { return v.x; }
    else if lane == 1 { return v.y; }
    else if lane == 2 { return v.z; }
    else              { return v.w; }
}

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
    // Polar-unwrap space: x=angle (0..1 → 0..2π), y=radius (0..1)
    let theta = in.uv.x * 6.28318530718;
    let v     = in.uv.y;
    let t     = u.time_seconds;

    let bass    = (band(0) + band(1)) * 0.5;
    let mid     = (band(2) + band(3) + band(4)) / 3.0;
    let treble  = (band(5) + band(6) + band(7)) / 3.0;
    let overall = (bass + mid + treble) / 3.0;

    // Base V positions: kaleido radii 0.16/0.26/0.38/0.47 → V 0.30/0.50/0.72/0.90
    let base0 = mix(0.30, 0.0, u.collapse.x);
    let base1 = mix(0.50, 0.0, u.collapse.y);
    let base2 = mix(0.72, 0.0, u.collapse.z);
    let base3 = mix(0.90, 0.0, u.collapse.w);

    // Harmonic wobble — kaleido fold provides angular replication, no cos(12θ) needed
    let curveV0 = base0 * (1.0 + 0.08 * cos(2.0 * theta + t * 0.2)  + bass    * 0.10);
    let curveV1 = base1 * (1.0 + 0.10 * cos(4.0 * theta - t * 0.3)  + mid     * 0.10);
    let curveV2 = base2 * (1.0 + 0.06 * cos(6.0 * theta + t * 0.4)  + treble  * 0.08);
    let curveV3 = base3 * (1.0 + 0.05 * cos(2.0 * theta - t * 0.15) + overall * 0.06);

    let maxCollapse = max(max(u.collapse.x, u.collapse.y), max(u.collapse.z, u.collapse.w));
    let coreHalfPx  = mix(1.0, 2.5, maxCollapse);
    let coreHalf    = coreHalfPx / u.resolution.y;

    let line0 = 1.0 - smoothstep(0.0, coreHalf, abs(v - curveV0));
    let line1 = 1.0 - smoothstep(0.0, coreHalf, abs(v - curveV1));
    let line2 = 1.0 - smoothstep(0.0, coreHalf, abs(v - curveV2));
    let line3 = 1.0 - smoothstep(0.0, coreHalf, abs(v - curveV3));

    let lineIntensity = clamp(line0 + line1 + line2 + line3, 0.0, 1.0);

    let prev         = textureSample(prev_tex, prev_samp, in.uv);
    let decayedAlpha = prev.a * 0.992;
    let outA         = min(max(decayedAlpha, lineIntensity * 1.3 * u.intensity), 1.0);

    return vec4<f32>(u.color.rgb, outA);
}

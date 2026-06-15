// SDF lyric-text overlay shader — Slice 2: legibility-driven UV warp.
//
// Uniform layout uses flat f32 fields (no vec2 in uniform block — CLAUDE.md rule 1).
// All eight fields pack into two vec4 slots (32 bytes total).

struct TextUniforms {
    color_r:    f32,  // offset  0
    color_g:    f32,  // offset  4
    color_b:    f32,  // offset  8
    color_a:    f32,  // offset 12
    legibility: f32,  // offset 16  — 0=abstract blob, 1=crisp letterform
    seed:       f32,  // offset 20  — per-fragment seed for warp variation
    warp_time:  f32,  // offset 24  — frame_time in seconds (animated swirl)
    _pad2:      f32,  // offset 28
};

@group(0) @binding(0) var atlas_tex: texture_2d<f32>;
@group(0) @binding(1) var atlas_smp: sampler;
@group(0) @binding(2) var<uniform> u: TextUniforms;

struct VertOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv_x: f32,
    @location(1) uv_y: f32,
};

@vertex
fn vs_main(
    @location(0) pos_x: f32,
    @location(1) pos_y: f32,
    @location(2) uv_x:  f32,
    @location(3) uv_y:  f32,
) -> VertOut {
    var out: VertOut;
    out.clip = vec4<f32>(pos_x, pos_y, 0.0, 1.0);
    out.uv_x = uv_x;
    out.uv_y = uv_y;
    return out;
}

// ── Noise helpers ─────────────────────────────────────────────────────────────

// 2D gradient hash: maps a lattice point to a random unit-ish vector in [−1,1]².
fn hash22(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)),
                      dot(p, vec2<f32>(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(q) * 43758.5453123);
}

// Smooth gradient noise over a 2D domain; returns a 2D offset in [−1, 1]².
// Uses quintic interpolation so the result has C² continuity.
fn noise2(p: vec2<f32>) -> vec2<f32> {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(
        mix(hash22(i),                        hash22(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash22(i + vec2<f32>(0.0, 1.0)), hash22(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y,
    );
}

// ── Fragment shader ───────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    var sample_uv = vec2<f32>(in.uv_x, in.uv_y);

    // When legibility < 1, displace the SDF sample UV by a smooth animated noise
    // field.  Warp strength ∝ (1 − legibility): at legibility=0 the glyph is
    // smeared into an abstract blob; at legibility=1 it is pixel-crisp.
    let warp_str = (1.0 - clamp(u.legibility, 0.0, 1.0)) * 0.28;
    if (warp_str > 0.0005) {
        // Unique phase per fragment via seed; slow drift via warp_time.
        let seed_off = u.seed * 0.00013;
        let time_off = u.warp_time * 0.35;
        let noise_uv = sample_uv * 5.5
                     + vec2<f32>(seed_off + time_off, seed_off - time_off * 0.6);
        let warp     = noise2(noise_uv);
        sample_uv    = clamp(sample_uv + warp * warp_str,
                             vec2<f32>(0.0001), vec2<f32>(0.9999));
    }

    let dist = textureSample(atlas_tex, atlas_smp, sample_uv).r;

    // Crisp SDF edge — narrow smoothstep gives ~2 px AA at 64-cell atlas resolution.
    let edge_half = 0.04;
    let alpha     = smoothstep(0.5 - edge_half, 0.5 + edge_half, dist) * u.color_a;

    return vec4<f32>(u.color_r, u.color_g, u.color_b, alpha);
}

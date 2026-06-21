// SDF lyric-text overlay shader — batched, per-vertex color + bounded smear.
//
// Per-vertex: pos (NDC), uv (atlas), col (rgba, alpha pre-folded with the
// lifetime envelope), cell (the glyph's atlas-cell rect [u0,v0,u1,v1]).
// Uniform: global warp_time + smear_strength only (flat f32 — CLAUDE.md rule 1).
//
// The fragment shader displaces the SDF sample along a slow dominant flow
// direction (directional "smear"), bounded to the glyph's OWN cell so it can
// never sample a neighbour letter.

struct TextUniforms {
    warp_time:      f32,  // offset  0
    smear_strength: f32,  // offset  4
    _pad0:          f32,  // offset  8
    _pad1:          f32,  // offset 12
};

@group(0) @binding(0) var atlas_tex: texture_2d<f32>;
@group(0) @binding(1) var atlas_smp: sampler;
@group(0) @binding(2) var<uniform> u: TextUniforms;

struct VertOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv:   vec2<f32>,
    @location(1) col:  vec4<f32>,
    @location(2) cell: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) pos:  vec2<f32>,
    @location(1) uv:   vec2<f32>,
    @location(2) col:  vec4<f32>,
    @location(3) cell: vec4<f32>,
) -> VertOut {
    var out: VertOut;
    out.clip = vec4<f32>(pos, 0.0, 1.0);
    out.uv   = uv;
    out.col  = col;
    out.cell = cell;
    return out;
}

// ── Noise helpers ─────────────────────────────────────────────────────────────

fn hash22(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)),
                      dot(p, vec2<f32>(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(q) * 43758.5453123);
}

// Smooth gradient noise → 2D offset in [−1, 1]² (quintic interpolation).
fn noise2(p: vec2<f32>) -> vec2<f32> {
    let i = floor(p);
    let f = fract(p);
    let w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(
        mix(hash22(i),                       hash22(i + vec2<f32>(1.0, 0.0)), w.x),
        mix(hash22(i + vec2<f32>(0.0, 1.0)), hash22(i + vec2<f32>(1.0, 1.0)), w.x),
        w.y,
    );
}

// ── Fragment shader ───────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    var sample_uv = in.uv;

    let strength = u.smear_strength;
    if (strength > 0.0005) {
        // Dominant flow direction: mostly horizontal, slowly wobbling over time,
        // so glyphs string/stretch along flow lines rather than jitter randomly.
        let ang  = 0.30 * sin(u.warp_time * 0.15);
        let flow = vec2<f32>(cos(ang), sin(ang));
        let perp = vec2<f32>(-flow.y, flow.x);

        // Flow-noise scrolls slowly with time; varies per glyph cell via uv.
        let nuv = in.uv * 7.0 + vec2<f32>(u.warp_time * 0.3, 0.0);
        let n   = noise2(nuv);  // [−1, 1]²

        // Bias displacement along the flow; small perpendicular wobble to liquefy.
        let disp = flow * (n.x * strength) + perp * (n.y * strength * 0.3);

        // CRITICAL: clamp the smeared sample to the glyph's OWN cell (tiny inset
        // so bilinear taps don't graze the neighbour) — smear WITHOUT cell bleed.
        let cell_min = in.cell.xy;
        let cell_max = in.cell.zw;
        let inset    = (cell_max - cell_min) * 0.04;
        sample_uv = clamp(in.uv + disp, cell_min + inset, cell_max - inset);
    }

    let dist      = textureSample(atlas_tex, atlas_smp, sample_uv).r;
    let edge_half = 0.04;  // ~2px AA at the 64-cell atlas resolution
    let alpha     = smoothstep(0.5 - edge_half, 0.5 + edge_half, dist) * in.col.a;

    // col.rgb is the rainbow/harmony color; col.a already carries the lifetime
    // envelope, so glyphs emerge from / dissolve to black via alpha → 0.
    return vec4<f32>(in.col.rgb, alpha);
}

// blackhole.wgsl — Pass 5 (conditional): multi-snapshot recursive-shrink trail.
// For each beat-captured snapshot, renders N nested copies that shrink toward a
// wandering singularity center, creating a comet-tail "ghost diving into a black hole"
// effect. Called once per active snapshot slot with ALPHA_BLENDING over an accumulator
// that was pre-loaded with the live scene.  3 bindings (no scene_tex).

struct BlackholePassUniforms {
    center_x:       f32,   // offset  0: wandering singularity center U
    center_y:       f32,   // offset  4: wandering singularity center V
    shrink_rate:    f32,   // offset  8: scale factor per nested copy (0..1)
    fade_curve:     f32,   // offset 12: temporal fade exponent
    cycle_progress: f32,   // offset 16: snapshot age 0..1
    slot_alpha:     f32,   // offset 20: pre-computed per-slot opacity (newest=1.0)
    passes:         f32,   // offset 24: number of nested copies (cast from u32)
    _pad:           f32,   // offset 28
};

@group(0) @binding(0) var<uniform> u: BlackholePassUniforms;
@group(0) @binding(1) var snapshot_tex: texture_2d<f32>;
@group(0) @binding(2) var samp:         sampler;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv  = in.uv;
    let center = vec2<f32>(u.center_x, u.center_y);
    let dir = uv - center;
    let n_passes = max(i32(u.passes), 1);

    // Temporal fade: snapshot fades to transparent as it ages.
    let temporal_alpha = pow(max(1.0 - u.cycle_progress, 0.0), u.fade_curve);
    let base_alpha = u.slot_alpha * temporal_alpha;

    // Accumulate nested copies front-to-back (i=0 = outermost/faintest first,
    // i=n-1 = innermost/brightest last = comet head on top).
    var out_rgb = vec3<f32>(0.0);
    var out_a   = 0.0;

    for (var i = 0; i < n_passes; i++) {
        // scale < 1 for i > 0: copy shrinks toward the singularity center.
        let scale      = pow(u.shrink_rate, f32(i));
        let sample_uv  = clamp(center + dir * scale, vec2<f32>(0.0), vec2<f32>(1.0));
        let s          = textureSample(snapshot_tex, samp, sample_uv);

        // Inner copies (high i, small scale) are most opaque — comet head glows brightest.
        let copy_frac  = f32(i + 1) / f32(n_passes);
        let src_a      = base_alpha * copy_frac * s.a;

        // Porter-Duff "over": this copy on top of accumulated result.
        out_rgb = s.rgb * src_a + out_rgb * (1.0 - src_a);
        out_a   = src_a + out_a * (1.0 - src_a);
    }

    return vec4<f32>(out_rgb, out_a);
}

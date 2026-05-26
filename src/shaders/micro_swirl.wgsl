// micro_swirl.wgsl — screen-space localised swirl post-process.
// Divides the screen into a density×density grid of cells; each cell runs an
// independent swirl that winds up then unwinds in a continuous loop.
// Distortion is exactly zero at every full oscillation period — no jump cuts.
// All geometry is computed in aspect-corrected square space so swirl cells are
// circular on screen, not stretched ovals.

struct MicroSwirlUniforms {
    density:   f32,   // cells per screen-width  (e.g. 10.0)
    amplitude: f32,   // peak rotation at cell centre, radians  (e.g. 0.8)
    speed:     f32,   // oscillation cycles per second  (e.g. 0.35)
    time:      f32,   // seconds since start, updated every frame
};

@group(0) @binding(0) var scene_tex:  texture_2d<f32>;
@group(0) @binding(1) var scene_samp: sampler;
@group(0) @binding(2) var<uniform> u: MicroSwirlUniforms;

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

// Deterministic per-cell hash → two independent floats in [0, 1).
fn cell_hash(id: vec2<f32>) -> vec2<f32> {
    let p = vec2<f32>(
        dot(id, vec2<f32>(127.1, 311.7)),
        dot(id, vec2<f32>(269.5, 183.3)),
    );
    return fract(sin(p) * 43758.5453123);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if u.amplitude < 0.0001 {
        return textureSample(scene_tex, scene_samp, in.uv);
    }

    // Derive aspect ratio from the texture dimensions so no uniform is needed.
    let dims   = vec2<f32>(textureDimensions(scene_tex));
    let aspect = dims.x / dims.y;

    // Work in aspect-corrected square space: x ∈ [0, aspect], y ∈ [0, 1].
    // This makes "density" cells span the screen width with square cells.
    let sq_uv = vec2<f32>(in.uv.x * aspect, in.uv.y);

    // Grid cell in square space.
    let cell_id  = floor(sq_uv * u.density);
    let cell_ctr = (cell_id + 0.5) / u.density;   // cell centre in sq space

    // Per-cell random phase [0,1) and rotation direction (±1).
    let h     = cell_hash(cell_id);
    let phase = h.x;
    let dir   = select(-1.0, 1.0, h.y > 0.5);

    // Smooth pingpong envelope: 0 → 1 → 0 → …  (raised cosine)
    // Exactly zero at every integer t/speed cycle — loop is seamless.
    let t   = u.time * u.speed + phase;
    let env = 0.5 * (1.0 - cos(t * 6.283185307));

    // Offset and distance in square space — gives circular falloff on screen.
    let offset    = sq_uv - cell_ctr;
    let dist_norm = length(offset) * u.density;   // 0 at centre, ~0.71 at corner

    // Radial falloff: full strength at centre, zero at cell boundary (≈ 0.5).
    let falloff = 1.0 - smoothstep(0.3, 0.5, dist_norm);

    // Rotate in square space.
    let theta = dir * u.amplitude * env * falloff;
    let c = cos(theta);
    let s = sin(theta);
    let rot_sq = vec2<f32>(
        offset.x * c - offset.y * s,
        offset.x * s + offset.y * c,
    );

    // Convert rotated result back to UV space and sample.
    // cell_ctr is in sq space; un-scale x by dividing by aspect.
    let sample_uv = vec2<f32>(
        (cell_ctr.x + rot_sq.x) / aspect,
         cell_ctr.y + rot_sq.y,
    );
    return textureSample(scene_tex, scene_samp, sample_uv);
}

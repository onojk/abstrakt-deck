// depth_shade.wgsl — lit-bump shading overlay driven by the micro-swirl displacement field.
// Reconstructs the micro-swirl UV displacement analytically and treats its divergence
// as a heightfield.  Central-differencing builds screen-space surface normals that are
// shaded with a configurable directional light, producing a 3-D bump appearance on
// the swirl texture.  Math is identical to micro_swirl.wgsl so shading lines up
// pixel-for-pixel with the actual distortion.

const TAU: f32 = 6.283185307;

struct DepthShadeUniforms {
    enabled:         f32,   // 1.0 = active, 0.0 = pass-through
    intensity:       f32,   // shadow darkness 0..1
    light_x:         f32,   // directional light, normalised inside shader
    light_y:         f32,
    light_z:         f32,   // depth component (forward = positive)
    softness:        f32,   // smoothstep half-width around terminator
    flatness:        f32,   // normal Z scale — higher = subtler bumps
    time:            f32,   // seconds since start, same as micro_swirl
    swirl_density:   f32,   // must match micro_swirl params exactly
    swirl_amplitude: f32,
    swirl_speed:     f32,
    aspect:          f32,   // width / height of the render target
    _pad0:           f32,
    _pad1:           f32,
    _pad2:           f32,
    _pad3:           f32,
};

@group(0) @binding(0) var<uniform> u: DepthShadeUniforms;
@group(0) @binding(1) var scene_tex:  texture_2d<f32>;
@group(0) @binding(2) var scene_samp: sampler;

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

// Deterministic per-cell hash — constants match micro_swirl.wgsl exactly.
fn cell_hash(id: vec2<f32>) -> vec2<f32> {
    let p = vec2<f32>(
        dot(id, vec2<f32>(127.1, 311.7)),
        dot(id, vec2<f32>(269.5, 183.3)),
    );
    return fract(sin(p) * 43758.5453123);
}

// Reconstructs the micro-swirl sample_uv for a given screen UV using the
// same algorithm as micro_swirl.wgsl.  Returns displacement = sample_uv - uv.
fn swirl_disp(uv: vec2<f32>) -> vec2<f32> {
    let aspect    = u.aspect;
    let density   = u.swirl_density;
    let amplitude = u.swirl_amplitude;
    let speed     = u.swirl_speed;
    let time      = u.time;

    // Aspect-corrected square space: x ∈ [0, aspect], y ∈ [0, 1].
    let sq_uv    = vec2<f32>(uv.x * aspect, uv.y);
    let cell_id  = floor(sq_uv * density);
    let cell_ctr = (cell_id + 0.5) / density;

    let h     = cell_hash(cell_id);
    let phase = h.x;
    let dir   = select(-1.0, 1.0, h.y > 0.5);

    // Smooth pingpong envelope (raised cosine) — zero at every full cycle.
    let t   = time * speed + phase;
    let env = 0.5 * (1.0 - cos(t * TAU));

    let offset    = sq_uv - cell_ctr;
    let dist_norm = length(offset) * density;
    let falloff   = 1.0 - smoothstep(0.3, 0.5, dist_norm);

    let theta = dir * amplitude * env * falloff;
    let c = cos(theta);
    let s = sin(theta);
    let rot_sq = vec2<f32>(
        offset.x * c - offset.y * s,
        offset.x * s + offset.y * c,
    );

    // Convert back to UV space (un-scale x by aspect).
    let sample_uv = vec2<f32>(
        (cell_ctr.x + rot_sq.x) / aspect,
         cell_ctr.y + rot_sq.y,
    );
    return sample_uv - uv;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let base = textureSample(scene_tex, scene_samp, in.uv);

    if u.enabled < 0.5 || u.intensity < 0.0001 {
        return base;
    }

    // Step size for central differencing: one texel in the longer dimension.
    let dims = vec2<f32>(textureDimensions(scene_tex));
    let h    = 1.0 / max(dims.x, dims.y);

    let uv = in.uv;

    // Evaluate displacement at four neighbours.
    let dx_p = swirl_disp(vec2<f32>(uv.x + h, uv.y));
    let dx_n = swirl_disp(vec2<f32>(uv.x - h, uv.y));
    let dy_p = swirl_disp(vec2<f32>(uv.x, uv.y + h));
    let dy_n = swirl_disp(vec2<f32>(uv.x, uv.y - h));

    // Finite-difference the (disp.x + disp.y) scalar field to get the surface gradient.
    // Divide by 2h to recover df/dx (the true derivative), not h*df/dx.
    let inv2h    = 1.0 / (2.0 * h);
    let height_dx = ((dx_p.x + dx_p.y) - (dx_n.x + dx_n.y)) * inv2h;
    let height_dy = ((dy_p.x + dy_p.y) - (dy_n.x + dy_n.y)) * inv2h;

    // Surface normal in screen space (Z = flatness controls bumpiness strength).
    let n = normalize(vec3<f32>(-height_dx, -height_dy, max(u.flatness, 0.01)));
    let l = normalize(vec3<f32>(u.light_x, u.light_y, u.light_z));
    let lambert = dot(n, l);

    // Shadow on the unlit side; smoothstep width = softness.
    let soft   = max(u.softness, 0.001);
    let shadow = 1.0 - smoothstep(-soft, soft, lambert);

    // Darken the scene on shadow side, leave lit side unchanged.
    let rgb = base.rgb * (1.0 - shadow * u.intensity);
    return vec4<f32>(rgb, base.a);
}

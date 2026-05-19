// splat.wgsl — billboard vertex shader and density-kernel fragment shader.
//
// VERTEX STAGE
// Each cell is drawn as 6 vertices (2 triangles = 1 quad), instanced.
// The vertex shader reads per-cell data from a storage buffer and places the
// quad corners in NDC space based on the precomputed screen center and radius.
//
// FRAGMENT STAGE
// For each pixel inside the quad, compute the 2D Mahalanobis distance from
// the cell's projected center. Evaluate the density kernel (generalized Gaussian
// blended toward hard ellipsoid by sharpness). Composite with premultiplied alpha.
//
// ALPHA BLENDING
// Pipeline uses src=ONE, dst=ONE_MINUS_SRC_ALPHA (premultiplied).
// Fragment outputs vec4(color * alpha, alpha), NOT vec4(color, alpha).
// Getting this wrong produces a white halo (the most common mistake here).

// ---- Uniforms and storage ----

struct CameraUniform {
    view_proj: mat4x4<f32>,
    eye:       vec3<f32>,
    _pad:      f32,
};

struct SplatData {
    screen_xy:   vec2<f32>,   // NDC center of the 2D footprint
    quad_radius: f32,         // half-size of bounding quad in NDC
    _pad0:       f32,
    inv_cov2d:   vec4<f32>,   // [a, b, d, 0] — upper triangle of 2×2 inverse NDC covariance
    color_inner: vec4<f32>,   // [r, g, b, 0]
    color_outer: vec4<f32>,   // [r, g, b, 0]
    params:      vec4<f32>,   // [opacity, falloff, sharpness, 0]
};

@group(0) @binding(0) var<uniform>          camera: CameraUniform;
@group(0) @binding(1) var<storage, read>    splats: array<SplatData>;

// ---- Vertex stage ----


struct VsOut {
    @builtin(position) clip:        vec4<f32>,
    @location(0)       uv:          vec2<f32>,  // offset from splat center in NDC
    @location(1)       inv_cov2d_a: f32,
    @location(2)       inv_cov2d_b: f32,
    @location(3)       inv_cov2d_d: f32,
    @location(4)       color_inner: vec3<f32>,
    @location(5)       color_outer: vec3<f32>,
    @location(6)       opacity:     f32,
    @location(7)       falloff:     f32,
    @location(8)       sharpness:   f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index)   vert_idx: u32,
    @builtin(instance_index) inst_idx: u32,
) -> VsOut {
    let s = splats[inst_idx];

    // Quad corners for two triangles (CCW winding).
    // naga doesn't allow runtime indexing into const arrays, so compute
    // the corner position directly from the vertex index.
    let cx = select(-1.0, 1.0, vert_idx == 1u || vert_idx == 4u || vert_idx == 5u);
    let cy = select(-1.0, 1.0, vert_idx == 2u || vert_idx == 3u || vert_idx == 5u);
    let corner = vec2<f32>(cx, cy);

    // NDC position of this quad vertex
    let ndc = s.screen_xy + corner * s.quad_radius;

    // The UV we pass to the fragment shader is the NDC offset from the splat center.
    // The fragment shader uses it with inv_cov2d (in 1/NDC²) to get Mahalanobis distance.
    var out: VsOut;
    out.clip        = vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
    out.uv          = corner * s.quad_radius;
    out.inv_cov2d_a = s.inv_cov2d.x;
    out.inv_cov2d_b = s.inv_cov2d.y;
    out.inv_cov2d_d = s.inv_cov2d.z;
    out.color_inner = s.color_inner.xyz;
    out.color_outer = s.color_outer.xyz;
    out.opacity     = s.params.x;
    out.falloff     = s.params.y;
    out.sharpness   = s.params.z;
    return out;
}

// ---- Fragment stage ----

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dx = in.uv.x;
    let dy = in.uv.y;

    // 2D Mahalanobis distance squared using the projected inverse covariance.
    // [[a, b], [b, d]] is symmetric, so:
    //   d² = [dx, dy] [[a,b],[b,d]] [dx,dy]^T = a*dx² + 2*b*dx*dy + d*dy²
    let d2 = in.inv_cov2d_a * dx * dx
           + 2.0 * in.inv_cov2d_b * dx * dy
           + in.inv_cov2d_d * dy * dy;

    // Mahalanobis distance. Epsilon prevents zero-gradient at d=0 for Tier 3.
    let d = sqrt(d2 + 1e-6);

    // Generalized Gaussian: falloff=1.0 → standard Gaussian exp(-d²).
    // We use 2*falloff as the exponent on d (not d²), so:
    //   falloff=0.5 → exp(-d) (soft exponential)
    //   falloff=1.0 → exp(-d²) (Gaussian)
    //   falloff=2.0 → exp(-d⁴) (super-Gaussian, flat plateau)
    let gaussian = exp(-pow(d, 2.0 * in.falloff));

    // Soft hard-edge approximation: sigmoid centered at d=1.
    // k=8 gives a near-step function while preserving nonzero gradients (Tier 3).
    // EDGE_SHARPNESS_SCALE is a named constant; reduce to ~4 if Tier 3 shows gradient vanishing.
    let crisp = 1.0 / (1.0 + exp(-8.0 * (1.0 - d)));

    // Blend: sharpness=0 → pure generalized Gaussian; sharpness=1 → near hard ellipsoid.
    let density = mix(gaussian, crisp, in.sharpness);

    // Discard fragments that contribute less than 1/255 — saves fillrate.
    // Do this before computing color to avoid unnecessary work.
    let alpha = in.opacity * density;
    if alpha < 1.0 / 255.0 {
        discard;
    }

    // Radial color gradient: interpolate inner→outer by Mahalanobis distance.
    // t=0 at center, t=1 at d=1 (the 1-sigma ellipsoid surface).
    let t     = clamp(d, 0.0, 1.0);
    let color = mix(in.color_inner, in.color_outer, t);

    // Premultiplied alpha output.
    // Pipeline blend: src=ONE, dst=ONE_MINUS_SRC_ALPHA.
    // This gives correct back-to-front compositing when cells are sorted.
    return vec4<f32>(color * alpha, alpha);
}

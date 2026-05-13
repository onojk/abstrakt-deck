struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct DistortionPlusUniforms {
    yaw:   f32,  // radians
    pitch: f32,
    roll:  f32,
    _pad:  f32,
};

@group(0) @binding(0) var<uniform> dp:      DistortionPlusUniforms;
@group(0) @binding(1) var          dp_tex:  texture_2d<f32>;
@group(0) @binding(2) var          dp_samp: sampler;

const TAU: f32 = 6.28318530718;
const PI:  f32 = 3.14159265359;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VertexOutput;
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

// Rotation around X axis.
fn rot_x(a: f32) -> mat3x3<f32> {
    let c = cos(a); let s = sin(a);
    return mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0,   c,   s),
        vec3<f32>(0.0,  -s,   c),
    );
}

// Rotation around Y axis.
fn rot_y(a: f32) -> mat3x3<f32> {
    let c = cos(a); let s = sin(a);
    return mat3x3<f32>(
        vec3<f32>( c, 0.0, -s),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>( s, 0.0,  c),
    );
}

// Rotation around Z axis.
fn rot_z(a: f32) -> mat3x3<f32> {
    let c = cos(a); let s = sin(a);
    return mat3x3<f32>(
        vec3<f32>( c,  s, 0.0),
        vec3<f32>(-s,  c, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // UV → spherical direction (equirectangular projection).
    let theta = (in.uv.x - 0.5) * TAU;   // longitude: -PI .. +PI
    let phi   = (in.uv.y - 0.5) * PI;    // latitude:  -PI/2 .. +PI/2
    var dir = vec3<f32>(
        cos(phi) * sin(theta),
        sin(phi),
        cos(phi) * cos(theta),
    );

    // ZXY Euler rotation: Ry(yaw) * Rx(pitch) * Rz(roll) — applied right-to-left.
    dir = rot_z(dp.roll)  * dir;
    dir = rot_x(dp.pitch) * dir;
    dir = rot_y(dp.yaw)   * dir;

    // Rotated direction → UV.
    let theta2 = atan2(dir.x, dir.z);
    let phi2   = asin(clamp(dir.y, -1.0, 1.0));
    let u2 = fract(theta2 / TAU + 0.5);           // yaw wraps seamlessly
    let v2 = clamp(phi2 / PI + 0.5, 0.0, 1.0);   // pitch clamps at poles

    return textureSample(dp_tex, dp_samp, vec2<f32>(u2, v2));
}

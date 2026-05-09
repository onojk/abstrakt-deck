struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Matches the Rust GlobalUniforms struct exactly (all f32, 16 bytes total).
// Using separate f32 fields rather than vec2 avoids the vec2 8-byte alignment
// offset that would mismatch the Rust-side layout.
struct GlobalUniforms {
    time_seconds: f32,
    resolution_x: f32,
    resolution_y: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> globals: GlobalUniforms;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen triangle: 3 vertices form a triangle larger than the screen,
    // clipped to the viewport — every pixel covered, zero overdraw.
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    var out: VertexOutput;
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let t = globals.time_seconds;

    // Animated UV gradient: each channel driven by a sine wave at a different
    // frequency so all three cycle independently, confirming uniforms reach the shader.
    let r = 0.5 + 0.5 * sin(uv.x * 6.28318 + t);
    let g = 0.5 + 0.5 * sin(uv.y * 6.28318 + t * 1.3);
    let b = 0.5 + 0.5 * sin((uv.x + uv.y) * 6.28318 + t * 0.7);

    return vec4<f32>(r, g, b, 1.0);
}

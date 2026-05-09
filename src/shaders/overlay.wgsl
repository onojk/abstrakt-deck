struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct OverlayUniforms {
    viewport_x: f32,
    viewport_y: f32,
    panel_x: f32,
    panel_y: f32,
    anim_progress: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> u: OverlayUniforms;
@group(0) @binding(1) var panel_tex: texture_2d<f32>;
@group(0) @binding(2) var panel_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    let margin = 20.0;
    let panel_x_pos = u.viewport_x - u.panel_x - margin;
    let off_screen_y = u.viewport_y;
    let on_screen_y  = u.viewport_y - u.panel_y - margin;
    let panel_y_pos  = mix(off_screen_y, on_screen_y, u.anim_progress);

    // Two-triangle quad (6 vertices)
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
    );

    let corner    = corners[vid];
    let pixel_pos = vec2<f32>(panel_x_pos, panel_y_pos) + corner * vec2<f32>(u.panel_x, u.panel_y);

    // Pixel space → NDC  (Y-flip: pixel y=0 is top of screen)
    let ndc = vec2<f32>(
         pixel_pos.x / u.viewport_x * 2.0 - 1.0,
        -pixel_pos.y / u.viewport_y * 2.0 + 1.0,
    );

    var out: VertexOutput;
    out.clip_position = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = corner;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(panel_tex, panel_sampler, in.uv);
    return vec4<f32>(color.rgb, color.a * u.anim_progress);
}

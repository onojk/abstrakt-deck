//! SDF text render pass — Slice 1 (static) + Slice 2 (per-fragment reactive).
//!
//! `TextPass` owns two render pipelines:
//!   • `pipeline_alpha` — standard alpha-over blending (kept for backward compat).
//!   • `pipeline_add`   — additive src-alpha blending (default for Slice 2 fragments;
//!     glows over busy visuals without darkening the background).

use bytemuck::{Pod, Zeroable};
use super::atlas::{TextAtlas, CELL};

const MAX_QUADS: usize = 512;
const MAX_VERTS: usize = MAX_QUADS * 6;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TextVertex {
    pos_x: f32,
    pos_y: f32,
    uv_x:  f32,
    uv_y:  f32,
}

/// Uniform block for the text shader.  Flat f32 fields — CLAUDE.md rule 1.
/// All 8 fields pack into two vec4 slots (32 bytes); the GPU sees them in the
/// same order as the WGSL `TextUniforms` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TextUniforms {
    pub color_r:    f32,   // offset  0
    pub color_g:    f32,   // offset  4
    pub color_b:    f32,   // offset  8
    pub color_a:    f32,   // offset 12
    pub legibility: f32,   // offset 16
    pub seed:       f32,   // offset 20 — per-fragment warp seed (cast from u32)
    pub warp_time:  f32,   // offset 24 — frame_time in seconds
    pub _pad2:      f32,   // offset 28
}

impl TextUniforms {
    pub fn opaque_white() -> Self {
        Self {
            color_r: 1.0, color_g: 1.0, color_b: 1.0, color_a: 1.0,
            legibility: 1.0, seed: 0.0, warp_time: 0.0, _pad2: 0.0,
        }
    }
}

pub struct TextPass {
    pipeline_alpha: wgpu::RenderPipeline,
    pipeline_add:   wgpu::RenderPipeline,
    bgl:            wgpu::BindGroupLayout,
    atlas_bg:       wgpu::BindGroup,
    vert_buf:       wgpu::Buffer,
    uniform_buf:    wgpu::Buffer,
    vert_count:     u32,
}

impl TextPass {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        atlas:  &TextAtlas,
    ) -> Self {
        // ── Bind group layout ──────────────────────────────────────────────
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TextPass BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TextPass uniforms"),
            size: std::mem::size_of::<TextUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let atlas_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TextPass atlas BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&atlas.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TextPass verts"),
            size: (std::mem::size_of::<TextVertex>() * MAX_VERTS) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TextPass shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/text.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TextPass pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TextVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { shader_location: 0, format: wgpu::VertexFormat::Float32, offset: 0  },
                wgpu::VertexAttribute { shader_location: 1, format: wgpu::VertexFormat::Float32, offset: 4  },
                wgpu::VertexAttribute { shader_location: 2, format: wgpu::VertexFormat::Float32, offset: 8  },
                wgpu::VertexAttribute { shader_location: 3, format: wgpu::VertexFormat::Float32, offset: 12 },
            ],
        };

        // Shared pipeline descriptor template — only the blend state differs.
        let make_pipeline = |label: &'static str, blend: wgpu::BlendState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout.clone()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

        // Alpha-over: src_alpha * src + (1 − src_alpha) * dst  (standard compositing).
        let pipeline_alpha = make_pipeline("TextPass alpha pipeline", wgpu::BlendState::ALPHA_BLENDING);

        // Additive: src_alpha * src + dst  (glow/dream look over busy visuals).
        let pipeline_add = make_pipeline("TextPass additive pipeline", wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation:  wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        });

        Self {
            pipeline_alpha, pipeline_add,
            bgl, atlas_bg,
            vert_buf, uniform_buf,
            vert_count: 0,
        }
    }

    // ── Uniform helpers ────────────────────────────────────────────────────

    pub fn write_default_uniforms(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.uniform_buf, 0,
            bytemuck::cast_slice(&[TextUniforms::opaque_white()]));
    }

    pub fn set_color(&self, queue: &wgpu::Queue, r: f32, g: f32, b: f32, a: f32) {
        queue.write_buffer(&self.uniform_buf, 0,
            bytemuck::cast_slice(&[TextUniforms {
                color_r: r, color_g: g, color_b: b, color_a: a,
                legibility: 1.0, seed: 0.0, warp_time: 0.0, _pad2: 0.0,
            }]));
    }

    // ── Vertex layout helpers ─────────────────────────────────────────────

    /// Layout `text` centred on x=0.  Used by the Slice 1 static overlay.
    pub fn set_text(
        &mut self,
        queue:          &wgpu::Queue,
        atlas:          &TextAtlas,
        text:           &str,
        font_px:        f32,
        screen_w:       u32,
        screen_h:       u32,
        baseline_y_ndc: f32,
    ) {
        self.set_text_at(queue, atlas, text, font_px, screen_w, screen_h,
                         0.0, baseline_y_ndc);
    }

    /// Layout `text` centred on `center_x_ndc`.  Used by Slice 2 fragment draws.
    pub fn set_text_at(
        &mut self,
        queue:        &wgpu::Queue,
        atlas:        &TextAtlas,
        text:         &str,
        font_px:      f32,
        screen_w:     u32,
        screen_h:     u32,
        center_x_ndc: f32,
        baseline_y_ndc: f32,
    ) {
        if text.is_empty() { self.vert_count = 0; return; }

        let s = font_px / atlas.raster_px;
        let px_to_ndcx = 2.0 / screen_w as f32;
        let px_to_ndcy = 2.0 / screen_h as f32;

        let cell_ndcw = CELL as f32 * s * px_to_ndcx;
        let above_ndc = atlas.baseline_y_cell as f32 * s * px_to_ndcy;
        let below_ndc = (CELL as f32 - atlas.baseline_y_cell as f32) * s * px_to_ndcy;

        // Measure total run width for centering.
        let mut total_ndcw = 0.0f32;
        for ch in text.chars().take(MAX_QUADS) {
            let ch = if atlas.glyphs.contains_key(&ch) { ch } else { ' ' };
            if let Some(g) = atlas.glyphs.get(&ch) {
                total_ndcw += g.advance_norm * atlas.raster_px * s * px_to_ndcx;
            }
        }

        let mut verts = Vec::with_capacity(text.len() * 6);
        let mut cursor_x = center_x_ndc - total_ndcw * 0.5;

        for ch in text.chars().take(MAX_QUADS) {
            let ch = if atlas.glyphs.contains_key(&ch) { ch } else { ' ' };
            if let Some(g) = atlas.glyphs.get(&ch) {
                let x0 = cursor_x;
                let x1 = cursor_x + cell_ndcw;
                let y0 = baseline_y_ndc - below_ndc;
                let y1 = baseline_y_ndc + above_ndc;
                let (u0, v0, u1, v1) = (g.uv[0], g.uv[1], g.uv[2], g.uv[3]);

                verts.push(TextVertex { pos_x: x0, pos_y: y0, uv_x: u0, uv_y: v1 });
                verts.push(TextVertex { pos_x: x1, pos_y: y0, uv_x: u1, uv_y: v1 });
                verts.push(TextVertex { pos_x: x1, pos_y: y1, uv_x: u1, uv_y: v0 });
                verts.push(TextVertex { pos_x: x0, pos_y: y0, uv_x: u0, uv_y: v1 });
                verts.push(TextVertex { pos_x: x1, pos_y: y1, uv_x: u1, uv_y: v0 });
                verts.push(TextVertex { pos_x: x0, pos_y: y1, uv_x: u0, uv_y: v0 });

                cursor_x += g.advance_norm * atlas.raster_px * s * px_to_ndcx;
            }
        }

        let n = verts.len().min(MAX_VERTS);
        if n > 0 {
            queue.write_buffer(&self.vert_buf, 0, bytemuck::cast_slice(&verts[..n]));
        }
        self.vert_count = n as u32;
    }

    pub fn has_text(&self) -> bool { self.vert_count > 0 }

    // ── Render calls ──────────────────────────────────────────────────────

    /// Slice 1: render the currently buffered quads with alpha-over blend.
    pub fn render(&self, enc: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        self.render_with_pipeline(enc, target, &self.pipeline_alpha);
    }

    /// Slice 2: lay out one fragment and render it with additive blend.
    ///
    /// Uploads `uniforms` (includes `legibility`, `seed`, `warp_time`) so the
    /// warp shader can smear/resolve the glyph independently per fragment.
    pub fn render_fragment(
        &mut self,
        enc:        &mut wgpu::CommandEncoder,
        target:     &wgpu::TextureView,
        queue:      &wgpu::Queue,
        atlas:      &TextAtlas,
        text:       &str,
        font_px:    f32,
        screen_w:   u32,
        screen_h:   u32,
        center_x:   f32,
        baseline_y: f32,
        uniforms:   TextUniforms,
    ) {
        self.set_text_at(queue, atlas, text, font_px, screen_w, screen_h,
                         center_x, baseline_y);
        if self.vert_count == 0 { return; }
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&[uniforms]));
        // additive blend is the default for reactive fragments (glow over visuals)
        self.render_with_pipeline(enc, target, &self.pipeline_add);
    }

    /// Render one SDF glyph with a CPU-side 2D transform (rotate + flip).
    ///
    /// The four quad corners are rotated around `(center_x, center_y)` on the
    /// CPU and passed as pre-transformed NDC positions — no matrix uniforms
    /// needed.  Alpha-over blend (pipeline_alpha) composites cleanly over other
    /// glyphs.
    pub fn render_glyph(
        &mut self,
        enc:      &mut wgpu::CommandEncoder,
        target:   &wgpu::TextureView,
        queue:    &wgpu::Queue,
        atlas:    &TextAtlas,
        ch:       char,
        font_px:  f32,
        screen_w: u32,
        screen_h: u32,
        center_x: f32,
        center_y: f32,
        rotation: f32,
        flip_x:   bool,
        flip_y:   bool,
        uniforms: TextUniforms,
    ) {
        let ch = if atlas.glyphs.contains_key(&ch) { ch } else { ' ' };
        let g = match atlas.glyphs.get(&ch) { Some(g) => *g, None => return };

        let s        = font_px / atlas.raster_px;
        let px_ndcx  = 2.0 / screen_w as f32;
        let px_ndcy  = 2.0 / screen_h as f32;
        let cell_w   = CELL as f32 * s * px_ndcx;
        let above    = atlas.baseline_y_cell as f32 * s * px_ndcy;
        let below    = (CELL as f32 - atlas.baseline_y_cell as f32) * s * px_ndcy;

        // Quad half-extents centred on (center_x, center_y).
        let hw = cell_w  * 0.5;
        let hh = (above + below) * 0.5;

        // UV coords: apply flip by swapping atlas edges.
        let (u_l, u_r) = if flip_x { (g.uv[2], g.uv[0]) } else { (g.uv[0], g.uv[2]) };
        // v_top = small v (top of atlas cell), v_bot = large v (bottom of atlas cell).
        // In NDC, y↑ = screen top → atlas top → small v.
        let (v_top, v_bot) = if flip_y { (g.uv[3], g.uv[1]) } else { (g.uv[1], g.uv[3]) };

        // Rotate a local-space point into NDC.
        let (cos_r, sin_r) = (rotation.cos(), rotation.sin());
        let rot = |lx: f32, ly: f32| -> (f32, f32) {
            (center_x + lx * cos_r - ly * sin_r,
             center_y + lx * sin_r + ly * cos_r)
        };

        // 4 corners → 6 verts (two CCW triangles).
        let (bl_x, bl_y) = rot(-hw, -hh); // bottom-left
        let (br_x, br_y) = rot( hw, -hh); // bottom-right
        let (tr_x, tr_y) = rot( hw,  hh); // top-right
        let (tl_x, tl_y) = rot(-hw,  hh); // top-left

        let verts = [
            TextVertex { pos_x: bl_x, pos_y: bl_y, uv_x: u_l, uv_y: v_bot },
            TextVertex { pos_x: br_x, pos_y: br_y, uv_x: u_r, uv_y: v_bot },
            TextVertex { pos_x: tr_x, pos_y: tr_y, uv_x: u_r, uv_y: v_top },
            TextVertex { pos_x: bl_x, pos_y: bl_y, uv_x: u_l, uv_y: v_bot },
            TextVertex { pos_x: tr_x, pos_y: tr_y, uv_x: u_r, uv_y: v_top },
            TextVertex { pos_x: tl_x, pos_y: tl_y, uv_x: u_l, uv_y: v_top },
        ];

        queue.write_buffer(&self.vert_buf, 0, bytemuck::cast_slice(&verts));
        self.vert_count = 6;
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&[uniforms]));
        self.render_with_pipeline(enc, target, &self.pipeline_alpha);
    }

    fn render_with_pipeline(
        &self,
        enc:      &mut wgpu::CommandEncoder,
        target:   &wgpu::TextureView,
        pipeline: &wgpu::RenderPipeline,
    ) {
        if self.vert_count == 0 { return; }
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("TextPass render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.atlas_bg, &[]);
        pass.set_vertex_buffer(0, self.vert_buf.slice(..));
        pass.draw(0..self.vert_count, 0..1);
    }
}

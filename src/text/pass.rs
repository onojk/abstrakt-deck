//! SDF text render pass.
//!
//! Builds glyph quads for one text string and renders them with alpha-over
//! blending into an existing frame target (LoadOp::Load).  The pass reads the
//! SDF atlas produced by `TextAtlas` and writes crisp anti-aliased glyphs at
//! any scale without re-rasterization.

use bytemuck::{Pod, Zeroable};
use super::atlas::{TextAtlas, CELL};

/// Maximum number of glyph quads (each = 6 vertices).  512 glyphs ≈ a few
/// full lyric lines.
const MAX_QUADS: usize = 512;
const MAX_VERTS: usize = MAX_QUADS * 6;

/// Per-vertex data for the text quad buffer.
/// Using four separate f32 attributes to stay compatible with the WGSL
/// vertex shader (vec2 is fine in vertex attributes — CLAUDE.md rule only
/// applies to uniform address space).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TextVertex {
    pos_x: f32,
    pos_y: f32,
    uv_x:  f32,
    uv_y:  f32,
}

/// Uniform block for the text shader.
/// Flat f32 fields — no vec2 in uniform block (CLAUDE.md rule 1).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TextUniforms {
    pub color_r:    f32,
    pub color_g:    f32,
    pub color_b:    f32,
    pub color_a:    f32,
    pub legibility: f32,
    pub _pad0:      f32,
    pub _pad1:      f32,
    pub _pad2:      f32,
}

pub struct TextPass {
    pipeline:    wgpu::RenderPipeline,
    bgl:         wgpu::BindGroupLayout,
    atlas_bg:    wgpu::BindGroup,
    vert_buf:    wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    vert_count:  u32,
}

impl TextPass {
    pub fn new(
        device:  &wgpu::Device,
        format:  wgpu::TextureFormat,
        atlas:   &TextAtlas,
    ) -> Self {
        // ── Bind group layout ──────────────────────────────────────────────
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TextPass BGL"),
            entries: &[
                // binding 0: SDF atlas texture
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
                // binding 1: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 2: TextUniforms
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

        // ── Uniform buffer ─────────────────────────────────────────────────
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TextPass uniforms"),
            size: std::mem::size_of::<TextUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Atlas bind group (atlas texture + sampler + uniforms) ──────────
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

        // ── Vertex buffer ─────────────────────────────────────────────────
        let vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TextPass verts"),
            size: (std::mem::size_of::<TextVertex>() * MAX_VERTS) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Shader + pipeline ─────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TextPass shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/text.wgsl").into(),
            ),
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
                wgpu::VertexAttribute { shader_location: 0, format: wgpu::VertexFormat::Float32, offset: 0 },
                wgpu::VertexAttribute { shader_location: 1, format: wgpu::VertexFormat::Float32, offset: 4 },
                wgpu::VertexAttribute { shader_location: 2, format: wgpu::VertexFormat::Float32, offset: 8 },
                wgpu::VertexAttribute { shader_location: 3, format: wgpu::VertexFormat::Float32, offset: 12 },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TextPass pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // Standard alpha-over composite: src_alpha * src + (1-src_alpha) * dst
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Write default uniforms (white, full alpha, legibility=1.0).
        // Caller can override with set_color() before the first render.
        let default_uniforms = TextUniforms {
            color_r: 1.0, color_g: 1.0, color_b: 1.0, color_a: 1.0,
            legibility: 1.0, _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
        };
        // We can't call queue.write_buffer here (no queue arg), so callers
        // must call set_color before first render.  Zeroed POD = transparent.
        let _ = default_uniforms; // written after queue available

        Self { pipeline, bgl, atlas_bg, vert_buf, uniform_buf, vert_count: 0 }
    }

    /// Upload default white uniforms.  Call once after construction using the queue.
    pub fn write_default_uniforms(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::cast_slice(&[TextUniforms {
                color_r: 1.0, color_g: 1.0, color_b: 1.0, color_a: 1.0,
                legibility: 1.0, _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
            }]),
        );
    }

    /// Update the fill colour and overall alpha of the text.
    pub fn set_color(&self, queue: &wgpu::Queue, r: f32, g: f32, b: f32, a: f32) {
        // Read back current legibility so we don't clobber the Slice 2 field.
        // We use a local copy; legibility = 1.0 in Slice 1.
        queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::cast_slice(&[TextUniforms {
                color_r: r, color_g: g, color_b: b, color_a: a,
                legibility: 1.0, _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
            }]),
        );
    }

    /// Lay out `text` as a centred horizontal glyph run and upload quads.
    ///
    /// * `font_px`  – desired cap-height in screen pixels
    /// * `screen_w`, `screen_h` – output resolution in pixels
    /// * `baseline_y_ndc` – vertical position of the text baseline in NDC
    ///   (-1=bottom, +1=top of screen).  Default centred near bottom: -0.55.
    pub fn set_text(
        &mut self,
        queue:         &wgpu::Queue,
        atlas:         &TextAtlas,
        text:          &str,
        font_px:       f32,
        screen_w:      u32,
        screen_h:      u32,
        baseline_y_ndc: f32,
    ) {
        if text.is_empty() {
            self.vert_count = 0;
            return;
        }

        // Scale factor from raster-space to screen pixels.
        let s = font_px / atlas.raster_px;
        // Helpers: 1 raster-pixel → NDC
        let px_to_ndcx = 2.0 / screen_w  as f32;
        let px_to_ndcy = 2.0 / screen_h as f32;

        // Cell dimensions in NDC (square cells, scaled uniformly in screen pixels).
        let cell_ndcw = CELL as f32 * s * px_to_ndcx;
        let _cell_ndch = CELL as f32 * s * px_to_ndcy;

        // Baseline offset within the cell in NDC.
        // baseline_y_cell rows from cell top → that many raster-pixels above the
        // bottom of the cell.  In NDC y-up: top of cell is at baseline_y + (CELL -
        // baseline_y_cell)*s*px_to_ndcy, bottom at baseline_y - baseline_y_cell*s*…
        let above_ndc = atlas.baseline_y_cell as f32 * s * px_to_ndcy;
        let below_ndc = (CELL as f32 - atlas.baseline_y_cell as f32) * s * px_to_ndcy;

        // Total text width so we can centre it.
        let mut total_ndcw = 0.0f32;
        for ch in text.chars().take(MAX_QUADS) {
            let ch = if atlas.glyphs.contains_key(&ch) { ch } else { ' ' };
            if let Some(g) = atlas.glyphs.get(&ch) {
                total_ndcw += g.advance_norm * s * px_to_ndcx;
            }
        }

        let mut verts: Vec<TextVertex> = Vec::with_capacity(text.len() * 6);
        let mut cursor_x = -total_ndcw * 0.5;

        for ch in text.chars().take(MAX_QUADS) {
            let ch = if atlas.glyphs.contains_key(&ch) { ch } else { ' ' };
            if let Some(g) = atlas.glyphs.get(&ch) {
                // Quad corners in NDC.
                let x0 = cursor_x;
                let x1 = cursor_x + cell_ndcw;
                let y0 = baseline_y_ndc - below_ndc; // bottom of cell
                let y1 = baseline_y_ndc + above_ndc; // top of cell

                // Atlas UVs.
                let u0 = g.uv[0]; let v0 = g.uv[1];
                let u1 = g.uv[2]; let v1 = g.uv[3];

                // Two CCW triangles (winding matches wgpu default).
                // Triangle 1: bottom-left, bottom-right, top-right
                verts.push(TextVertex { pos_x: x0, pos_y: y0, uv_x: u0, uv_y: v1 });
                verts.push(TextVertex { pos_x: x1, pos_y: y0, uv_x: u1, uv_y: v1 });
                verts.push(TextVertex { pos_x: x1, pos_y: y1, uv_x: u1, uv_y: v0 });
                // Triangle 2: bottom-left, top-right, top-left
                verts.push(TextVertex { pos_x: x0, pos_y: y0, uv_x: u0, uv_y: v1 });
                verts.push(TextVertex { pos_x: x1, pos_y: y1, uv_x: u1, uv_y: v0 });
                verts.push(TextVertex { pos_x: x0, pos_y: y1, uv_x: u0, uv_y: v0 });

                cursor_x += g.advance_norm * s * px_to_ndcx;
            }
        }

        let n = verts.len().min(MAX_VERTS);
        if n > 0 {
            queue.write_buffer(&self.vert_buf, 0, bytemuck::cast_slice(&verts[..n]));
        }
        self.vert_count = n as u32;
    }

    /// Returns `true` if `set_text` has been called with a non-empty string.
    pub fn has_text(&self) -> bool {
        self.vert_count > 0
    }

    /// Render text quads into `target` using `LoadOp::Load` (alpha-over blend).
    /// The encoder must not have an active render pass.
    pub fn render(&self, enc: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.atlas_bg, &[]);
        pass.set_vertex_buffer(0, self.vert_buf.slice(..));
        pass.draw(0..self.vert_count, 0..1);
    }
}

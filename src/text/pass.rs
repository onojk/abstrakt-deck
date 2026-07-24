//! SDF text render pass — batched, per-vertex-colored.
//!
//! All live/export glyphs for a frame are built into ONE vertex buffer and drawn
//! in ONE pass (alpha-over). Per-glyph color, alpha and the glyph's atlas-cell
//! rect travel as vertex attributes; the only uniforms are global (`warp_time`,
//! `smear_strength`). The fragment shader (`shaders/text.wgsl`) applies a bounded
//! directional smear of the SDF sample, clamped to the per-vertex cell so it can
//! never bleed into a neighbour glyph.

use std::f32::consts::{FRAC_PI_2, TAU};
use bytemuck::{Pod, Zeroable};
use super::atlas::{TextAtlas, CELL};

/// Each live glyph is drawn as GLYPH_LAYERS stacked quads (same atlas cell, varied
/// size/offset/brightness) for depth & texture — see `build_glyph_quad`.
const GLYPH_LAYERS: usize = 3;

/// Per-layer size scale (back layer biggest, front layer smallest).
const GLYPH_LAYER_SCALE: [f32; GLYPH_LAYERS] = [1.30, 1.00, 0.72];

/// Per-layer position offset in glyph-local quad-size units (fraction of the full
/// quad width/height). Scaled by the glyph's own size so big and small glyphs
/// separate proportionally, not absolutely. Small misregistration → visible
/// separation, not a blur. Reduce if it reads messy through the kaleido fold.
const GLYPH_LAYER_OFFSET: [(f32, f32); GLYPH_LAYERS] =
    [(0.06, 0.05), (0.0, 0.0), (-0.05, -0.04)];

/// Per-layer brightness multiplier on the glyph's EXISTING rgb (hue & alpha kept).
/// Back layer dim → front layer is the bright crisp core.
const GLYPH_LAYER_BRIGHTNESS: [f32; GLYPH_LAYERS] = [0.45, 0.80, 1.0];

/// Verts emitted per glyph: GLYPH_LAYERS quads × 6 verts.
const VERTS_PER_GLYPH: usize = GLYPH_LAYERS * 6;

// ── Glyph-attached accent marks ──────────────────────────────────────────────
// Every live glyph also emits ONE small accent cluster (dot-row / dots-in-dots /
// oblong tick / recursive, cycled by glyph index) riding on the glyph's position,
// appended to the SAME batch AFTER the glyph layers so it lands on top. The four
// mark types mirror the standalone Accents shape (src/influencer/accents.rs); the
// shared DEBUG_RED flag is reused from there. Marks sample the parent glyph's SDF
// cell, so they have the same guaranteed visibility as the glyph itself.

/// Cluster size as a fraction of the parent glyph's size — detail ON the glyph,
/// never bigger than it.
const ACCENT_GLYPH_SCALE: f32 = 0.45;

/// Dots in a dot-row mark.
const ACCENT_DOTROW_DOTS: usize = 5;
/// Satellites in a recursive mark (each also gets one smaller grandchild).
const ACCENT_RECURSE_SATELLITES: usize = 4;

/// Marks emitted by the WORST-CASE cluster type (recursive):
/// 1 center + ACCENT_RECURSE_SATELLITES satellites + ACCENT_RECURSE_SATELLITES
/// grandchildren = 9 marks. Every other type emits fewer.
const ACCENT_MARKS_MAX: usize = 1 + ACCENT_RECURSE_SATELLITES * 2;
/// Verts a single accent cluster can append, worst case (6 verts per mark quad).
const ACCENT_VERTS_PER_GLYPH: usize = ACCENT_MARKS_MAX * 6;

/// Max glyph quads drawn in one frame. Each live glyph now costs
/// VERTS_PER_GLYPH (18) + ACCENT_VERTS_PER_GLYPH (54) = 72 verts, so the buffer
/// must be ~4× the old layers-only size. 12288 quads = 73728 verts holds ~1024
/// fully-accented glyphs — same glyph headroom the old 3072/18432 gave for layers.
const MAX_QUADS: usize = 12288;
const MAX_VERTS: usize = MAX_QUADS * 6;

/// Default directional-smear strength in atlas-UV space. The smear is hard-clamped
/// to the glyph's own cell (≈1/16 wide), so this controls how far toward that
/// bound it pushes. Tuned down from the spec's 0.10 so letters stay legible
/// through the kaleido fold. Dial up for muddier, down for crisper.
pub const SMEAR_STRENGTH: f32 = 0.06;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TextVertex {
    pos:  [f32; 2],
    uv:   [f32; 2],
    col:  [f32; 4],   // rgba; a already folds in the lifetime envelope
    cell: [f32; 4],   // glyph's atlas cell rect [u0, v0, u1, v1] for smear clamp
}

/// Global per-frame uniform. Flat f32 fields only (CLAUDE.md rule 1) — 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TextUniforms {
    pub warp_time:      f32,  // offset  0 — seconds, drives the slow flow
    pub smear_strength: f32,  // offset  4
    pub _pad0:          f32,  // offset  8
    pub _pad1:          f32,  // offset 12
}

impl TextUniforms {
    pub fn default_uniforms() -> Self {
        Self { warp_time: 0.0, smear_strength: SMEAR_STRENGTH, _pad0: 0.0, _pad1: 0.0 }
    }
}

/// One glyph to draw this frame. Built by the caller (live or export loop).
pub struct GlyphDraw {
    pub ch:        char,
    pub center_x:  f32,
    pub center_y:  f32,
    pub font_px:   f32,
    pub rotation:  f32,
    pub flip_x:    bool,
    pub flip_y:    bool,
    /// Anisotropic horizontal stretch (y stays 1.0) — elongates letters into strands.
    pub stretch_x: f32,
    /// rgba; alpha already multiplied by the lifetime envelope.
    pub color:     [f32; 4],
}

pub struct TextPass {
    pipeline:    wgpu::RenderPipeline,
    atlas_bg:    wgpu::BindGroup,
    vert_buf:    wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
}

impl TextPass {
    pub fn new(
        device: &wgpu::Device,
        _format: wgpu::TextureFormat,  // ignored — always targets shape_post (Rgba8Unorm)
        atlas:  &TextAtlas,
    ) -> Self {
        let format = wgpu::TextureFormat::Rgba8Unorm;

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
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&atlas.view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&atlas.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: uniform_buf.as_entire_binding() },
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
                wgpu::VertexAttribute { shader_location: 0, format: wgpu::VertexFormat::Float32x2, offset: 0  },
                wgpu::VertexAttribute { shader_location: 1, format: wgpu::VertexFormat::Float32x2, offset: 8  },
                wgpu::VertexAttribute { shader_location: 2, format: wgpu::VertexFormat::Float32x4, offset: 16 },
                wgpu::VertexAttribute { shader_location: 3, format: wgpu::VertexFormat::Float32x4, offset: 32 },
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

        Self { pipeline, atlas_bg, vert_buf, uniform_buf }
    }

    pub fn write_default_uniforms(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.uniform_buf, 0,
            bytemuck::cast_slice(&[TextUniforms::default_uniforms()]));
    }

    /// Build all glyph quads and draw them in a single pass.
    #[allow(clippy::too_many_arguments)]
    pub fn render_glyphs(
        &mut self,
        enc:           &mut wgpu::CommandEncoder,
        target:        &wgpu::TextureView,
        queue:         &wgpu::Queue,
        atlas:         &TextAtlas,
        draws:         &[GlyphDraw],
        screen_w:      u32,
        screen_h:      u32,
        warp_time:     f32,
        smear_strength: f32,
    ) {
        if draws.is_empty() { return; }

        let mut verts: Vec<TextVertex> =
            Vec::with_capacity(draws.len() * (VERTS_PER_GLYPH + ACCENT_VERTS_PER_GLYPH));
        for (gi, d) in draws.iter().enumerate() {
            // Reserve room for BOTH the glyph layers AND its accent cluster, so a
            // cluster is never half-written or silently dropped past the cap.
            if verts.len() + VERTS_PER_GLYPH + ACCENT_VERTS_PER_GLYPH > MAX_VERTS { break; }
            if let Some(quad) = build_glyph_quad(atlas, d, screen_w, screen_h) {
                verts.extend_from_slice(&quad);
                // Accent cluster AFTER the 3 glyph layers → composites on top of
                // the bright glyph core instead of being painted over.
                emit_accent_cluster(&mut verts, atlas, d, screen_w, screen_h, gi);
            }
        }
        if verts.is_empty() { return; }

        queue.write_buffer(&self.vert_buf, 0, bytemuck::cast_slice(&verts));
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&[TextUniforms {
            warp_time, smear_strength, _pad0: 0.0, _pad1: 0.0,
        }]));

        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("TextPass render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.atlas_bg, &[]);
        pass.set_vertex_buffer(0, self.vert_buf.slice(..));
        pass.draw(0..verts.len() as u32, 0..1);
    }
}

/// Build one glyph as GLYPH_LAYERS stacked quads (18 verts total) — the SAME atlas
/// cell tripled with varied size/offset/brightness for depth & texture. Layers are
/// emitted back-to-front (biggest/dimmest first, small/bright core last) so they
/// composite correctly under the shared alpha blend. CPU-side anisotropic stretch,
/// rotation and flip apply to every layer. Returns None for glyphs absent from the
/// atlas.
fn build_glyph_quad(
    atlas: &TextAtlas,
    d:     &GlyphDraw,
    screen_w: u32,
    screen_h: u32,
) -> Option<[TextVertex; VERTS_PER_GLYPH]> {
    let ch = if atlas.glyphs.contains_key(&d.ch) { d.ch } else { ' ' };
    let g = *atlas.glyphs.get(&ch)?;

    let s       = d.font_px / atlas.raster_px;
    let px_ndcx = 2.0 / screen_w as f32;
    let px_ndcy = 2.0 / screen_h as f32;
    let cell_w  = CELL as f32 * s * px_ndcx;
    let above   = atlas.baseline_y_cell as f32 * s * px_ndcy;
    let below   = (CELL as f32 - atlas.baseline_y_cell as f32) * s * px_ndcy;

    // Base half-extents (layer scale 1.0): x stretched into a strand, y natural.
    let hw = cell_w * 0.5 * d.stretch_x;
    let hh = (above + below) * 0.5;

    // UVs with flip.
    let (u_l, u_r) = if d.flip_x { (g.uv[2], g.uv[0]) } else { (g.uv[0], g.uv[2]) };
    let (v_top, v_bot) = if d.flip_y { (g.uv[3], g.uv[1]) } else { (g.uv[1], g.uv[3]) };

    // Glyph's atlas cell rect (sorted), passed through for the in-cell smear clamp.
    // Identical for all layers so the smear stays clamped to the same letter.
    let cell = [g.uv[0], g.uv[1], g.uv[2], g.uv[3]];

    let (cos_r, sin_r) = (d.rotation.cos(), d.rotation.sin());
    // Map a glyph-local point through rotation into NDC about the glyph center.
    let rot = |lx: f32, ly: f32| -> [f32; 2] {
        [d.center_x + lx * cos_r - ly * sin_r,
         d.center_y + lx * sin_r + ly * cos_r]
    };

    // Full base quad dimensions — layer offsets are expressed as fractions of these
    // so misregistration scales with the glyph's own size.
    let (full_w, full_h) = (hw * 2.0, hh * 2.0);

    let mut out = [TextVertex { pos: [0.0; 2], uv: [0.0; 2], col: [0.0; 4], cell };
                   VERTS_PER_GLYPH];

    for l in 0..GLYPH_LAYERS {
        let scale = GLYPH_LAYER_SCALE[l];
        let (hw_l, hh_l) = (hw * scale, hh * scale);
        // Offset in glyph-local space (pre-rotation), scaled by the base glyph size.
        let (off_x, off_y) = GLYPH_LAYER_OFFSET[l];
        let (ox, oy) = (off_x * full_w, off_y * full_h);

        let bl = rot(ox - hw_l, oy - hh_l);
        let br = rot(ox + hw_l, oy - hh_l);
        let tr = rot(ox + hw_l, oy + hh_l);
        let tl = rot(ox - hw_l, oy + hh_l);

        // Keep hue & alpha (the lifetime envelope rides in .a); only value differs.
        let b = GLYPH_LAYER_BRIGHTNESS[l];
        let col = [d.color[0] * b, d.color[1] * b, d.color[2] * b, d.color[3]];

        let base = l * 6;
        out[base    ] = TextVertex { pos: bl, uv: [u_l, v_bot], col, cell };
        out[base + 1] = TextVertex { pos: br, uv: [u_r, v_bot], col, cell };
        out[base + 2] = TextVertex { pos: tr, uv: [u_r, v_top], col, cell };
        out[base + 3] = TextVertex { pos: bl, uv: [u_l, v_bot], col, cell };
        out[base + 4] = TextVertex { pos: tr, uv: [u_r, v_top], col, cell };
        out[base + 5] = TextVertex { pos: tl, uv: [u_l, v_top], col, cell };
    }

    Some(out)
}

// ── Accent-mark emit (glyph-attached) ────────────────────────────────────────

/// Off-white accent coloring. Core is a bright warm/neutral off-white; halo a
/// dimmer light-gray. Nudge warmer/cooler/brighter here to retune ALL glyph
/// accents (the 4 mark types and their micro dots) in one place.
const ACCENT_OFFWHITE_CORE: [f32; 3] = [0.95, 0.94, 0.90];
const ACCENT_OFFWHITE_HALO: [f32; 3] = [0.45, 0.45, 0.42];

/// Accent (core, halo) colors for a glyph, inheriting the glyph's faded alpha so
/// accents emerge/dissolve WITH the glyph. All glyph-attached accent marks render
/// light OFF-WHITE — bright core, dimmer halo — keeping the bright-core/dim-halo
/// structure. Independent of the standalone Accents shape's coloring.
fn accent_colors(parent: [f32; 4]) -> ([f32; 4], [f32; 4]) {
    let a = parent[3]; // inherit the glyph's lifetime-envelope alpha
    let c = ACCENT_OFFWHITE_CORE;
    let h = ACCENT_OFFWHITE_HALO;
    ([c[0], c[1], c[2], a], [h[0], h[1], h[2], a])
}

/// Push one accent mark: a small quad sampling the parent glyph's SDF cell (so it
/// has the same guaranteed visibility as the glyph), centered at NDC (cx, cy),
/// half-extents (half_w, half_h), rotated by `angle`.
#[allow(clippy::too_many_arguments)]
fn push_accent_mark(
    verts: &mut Vec<TextVertex>,
    cx: f32, cy: f32, half_w: f32, half_h: f32, angle: f32,
    uv: (f32, f32, f32, f32),   // (u_l, u_r, v_top, v_bot)
    col: [f32; 4],
    cell: [f32; 4],
) {
    let (c, s) = (angle.cos(), angle.sin());
    let rot = |lx: f32, ly: f32| -> [f32; 2] { [cx + lx * c - ly * s, cy + lx * s + ly * c] };
    let (u_l, u_r, v_top, v_bot) = uv;
    let bl = rot(-half_w, -half_h);
    let br = rot( half_w, -half_h);
    let tr = rot( half_w,  half_h);
    let tl = rot(-half_w,  half_h);
    verts.push(TextVertex { pos: bl, uv: [u_l, v_bot], col, cell });
    verts.push(TextVertex { pos: br, uv: [u_r, v_bot], col, cell });
    verts.push(TextVertex { pos: tr, uv: [u_r, v_top], col, cell });
    verts.push(TextVertex { pos: bl, uv: [u_l, v_bot], col, cell });
    verts.push(TextVertex { pos: tr, uv: [u_r, v_top], col, cell });
    verts.push(TextVertex { pos: tl, uv: [u_l, v_top], col, cell });
}

/// Emit ONE accent cluster anchored at the glyph `d`, appended to `verts`. Mark
/// TYPE cycles by `glyph_index` so the field varies across the frame. Never emits
/// more than ACCENT_VERTS_PER_GLYPH verts (worst case = recursive). No-op if the
/// glyph is absent from the atlas.
fn emit_accent_cluster(
    verts: &mut Vec<TextVertex>,
    atlas: &TextAtlas,
    d:     &GlyphDraw,
    screen_w: u32,
    screen_h: u32,
    glyph_index: usize,
) {
    let ch = if atlas.glyphs.contains_key(&d.ch) { d.ch } else { ' ' };
    let g = match atlas.glyphs.get(&ch) { Some(g) => *g, None => return };

    // Recompute the parent glyph's NDC half-extents (same formulas as build_glyph_quad).
    let s       = d.font_px / atlas.raster_px;
    let px_ndcx = 2.0 / screen_w as f32;
    let px_ndcy = 2.0 / screen_h as f32;
    let cell_w  = CELL as f32 * s * px_ndcx;
    let above   = atlas.baseline_y_cell as f32 * s * px_ndcy;
    let below   = (CELL as f32 - atlas.baseline_y_cell as f32) * s * px_ndcy;
    let hw = cell_w * 0.5 * d.stretch_x;
    let hh = (above + below) * 0.5;

    // Parent glyph UV rect (base orientation) + cell rect for the smear clamp.
    let uv   = (g.uv[0], g.uv[2], g.uv[1], g.uv[3]); // (u_l, u_r, v_top, v_bot)
    let cell = [g.uv[0], g.uv[1], g.uv[2], g.uv[3]];

    let (core, halo) = accent_colors(d.color);

    // One NDC length scale for cluster geometry, proportional to glyph size.
    let unit = ACCENT_GLYPH_SCALE * 0.5 * (hw + hh);
    let dh   = 0.5 * unit; // accent dot half-extent
    let (cx, cy) = (d.center_x, d.center_y);
    let ang = d.rotation; // glyph's own axis = "screen-tangent"
    let (tc, ts) = (ang.cos(), ang.sin());

    match glyph_index % 4 {
        0 => {
            // DOT-ROW: a short line of dots stepping along the glyph tangent.
            for i in 0..ACCENT_DOTROW_DOTS {
                let t = i as f32 - (ACCENT_DOTROW_DOTS as f32 - 1.0) * 0.5;
                let along = t * 0.7 * unit;
                push_accent_mark(verts, cx + along * tc, cy + along * ts,
                                 dh, dh, ang, uv, core, cell);
            }
        }
        1 => {
            // DOTS-IN-DOTS: large dim dot with a small bright dot centered inside.
            push_accent_mark(verts, cx, cy, dh * 1.7, dh * 1.7, ang, uv, halo, cell);
            push_accent_mark(verts, cx, cy, dh * 0.75, dh * 0.75, ang, uv, core, cell);
        }
        2 => {
            // OBLONG TICK: one elongated mark oriented perpendicular (radial) to
            // the tangent, so it reads as a stroke crossing the glyph.
            push_accent_mark(verts, cx, cy, unit * 1.6, dh * 0.5, ang + FRAC_PI_2,
                             uv, core, cell);
        }
        _ => {
            // RECURSIVE: a center dot with satellites, each satellite + 1 grandchild.
            push_accent_mark(verts, cx, cy, dh, dh, ang, uv, core, cell);
            for sidx in 0..ACCENT_RECURSE_SATELLITES {
                let a = ang + sidx as f32 / ACCENT_RECURSE_SATELLITES as f32 * TAU;
                let (ac, as_) = (a.cos(), a.sin());
                push_accent_mark(verts, cx + ac * 1.2 * unit, cy + as_ * 1.2 * unit,
                                 dh * 0.6, dh * 0.6, ang, uv, core, cell);
                push_accent_mark(verts, cx + ac * 1.9 * unit, cy + as_ * 1.9 * unit,
                                 dh * 0.35, dh * 0.35, ang, uv, core, cell);
            }
        }
    }
}

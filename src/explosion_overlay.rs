//! GPU overlay for the explosions burst effect.
//!
//! Maintains a vertex buffer of ChunkVert quads (6 verts per chunk, two tris).
//! Renders with SrcAlpha/One additive blending so sparks glow over whatever
//! is already in the render target.
//!
//! The overlay texture (RENDER_ATTACHMENT | TEXTURE_BINDING | COPY_SRC) is
//! only used as the final readback source during export. In the live path,
//! chunks are drawn directly to the swapchain.

use bytemuck::{Pod, Zeroable};
use crate::explosions::ChunkFrame;

/// Per-vertex data. Stride = 24 bytes (6 × f32).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ChunkVert {
    pub pos:   [f32; 2],  // clip-space NDC (-1..1 each axis)
    pub uv:    [f32; 2],  // scene texture UV (0..1 each axis)
    pub alpha: f32,
    pub _pad:  f32,       // pad to 24-byte stride
}

/// Maximum number of vertices we'll ever upload in one frame.
/// 400 simultaneous chunks × 6 verts each = 2 400; round up to 12 000 for headroom.
const MAX_VERTS: usize = 12_000;

pub struct ExplosionOverlay {
    pub texture:    wgpu::Texture,
    pub view:       wgpu::TextureView,
    pub pipeline:   wgpu::RenderPipeline,
    pub scene_bgl:  wgpu::BindGroupLayout,
    pub sampler:    wgpu::Sampler,
    pub vert_buf:   wgpu::Buffer,
    width:  u32,
    height: u32,
}

// ── vertex generation ─────────────────────────────────────────────────────────

/// Convert collected ChunkFrames into ChunkVert quads (6 verts per chunk).
///
/// `aspect` is width/height of the render target — used so each chunk appears
/// square in screen pixels rather than square in UV space.
pub fn make_chunk_quads(frames: &[ChunkFrame], aspect: f32) -> Vec<ChunkVert> {
    let mut verts = Vec::with_capacity(frames.len() * 6);
    for f in frames {
        let cx = f.uv[0];
        let cy = f.uv[1];
        // UV half-size: sy is in vertical UV units; sx corrects for aspect ratio
        let sy = f.size;
        let sx = f.size / aspect;
        let cos_t = f.tumble.cos();
        let sin_t = f.tumble.sin();

        let corner = |dx: f32, dy: f32| -> ChunkVert {
            // Rotate corner offset by tumble angle in UV space
            let rx = dx * cos_t - dy * sin_t;
            let ry = dx * sin_t + dy * cos_t;
            let su = cx + rx;
            let sv = cy + ry;
            // UV → NDC; Y is inverted in clip space
            ChunkVert {
                pos:   [su * 2.0 - 1.0, -(sv * 2.0 - 1.0)],
                uv:    [su.clamp(0.0, 1.0), sv.clamp(0.0, 1.0)],
                alpha: f.alpha,
                _pad:  0.0,
            }
        };

        let bl = corner(-sx, -sy);
        let br = corner( sx, -sy);
        let tr = corner( sx,  sy);
        let tl = corner(-sx,  sy);
        // Two clockwise triangles: bl-br-tr, bl-tr-tl
        verts.push(bl); verts.push(br); verts.push(tr);
        verts.push(bl); verts.push(tr); verts.push(tl);
    }
    verts
}

// ── ExplosionOverlay impl ─────────────────────────────────────────────────────

impl ExplosionOverlay {
    pub fn new(
        device: &wgpu::Device,
        width:  u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let (texture, view) = Self::make_texture(device, width, height, format, false);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:            Some("ExplosionOverlay sampler"),
            mag_filter:       wgpu::FilterMode::Linear,
            min_filter:       wgpu::FilterMode::Linear,
            address_mode_u:   wgpu::AddressMode::ClampToEdge,
            address_mode_v:   wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("ExplosionOverlay verts"),
            size:               (MAX_VERTS * std::mem::size_of::<ChunkVert>()) as u64,
            usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ExplosionOverlay BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count:      None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("ExplosionOverlay shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/explosion_overlay.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("ExplosionOverlay pipeline layout"),
            bind_group_layouts:   &[&scene_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("ExplosionOverlay pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:       &shader,
                entry_point:  Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 24,  // 6 × f32
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { shader_location: 0, format: wgpu::VertexFormat::Float32x2, offset:  0 },
                        wgpu::VertexAttribute { shader_location: 1, format: wgpu::VertexFormat::Float32x2, offset:  8 },
                        wgpu::VertexAttribute { shader_location: 2, format: wgpu::VertexFormat::Float32,   offset: 16 },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation:  wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation:  wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive:    wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample:  wgpu::MultisampleState::default(),
            multiview:    None,
            cache:        None,
        });

        Self { texture, view, pipeline, scene_bgl, sampler, vert_buf, width, height }
    }

    fn make_texture(
        device: &wgpu::Device,
        width:  u32,
        height: u32,
        format: wgpu::TextureFormat,
        resized: bool,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(if resized { "ExplosionOverlay texture (resized)" } else { "ExplosionOverlay texture" }),
            size:  wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) {
        if width == self.width && height == self.height { return; }
        let (tex, view) = Self::make_texture(device, width, height, format, true);
        self.texture = tex;
        self.view    = view;
        self.width   = width;
        self.height  = height;
    }

    /// Upload verts to GPU. Returns the number of verts actually written (capped at MAX_VERTS).
    pub fn upload_verts(&self, queue: &wgpu::Queue, verts: &[ChunkVert]) -> usize {
        if verts.is_empty() { return 0; }
        let n = verts.len().min(MAX_VERTS);
        queue.write_buffer(&self.vert_buf, 0, bytemuck::cast_slice(&verts[..n]));
        n
    }

    pub fn make_scene_bind_group(&self, device: &wgpu::Device, view: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("ExplosionOverlay scene BG"),
            layout:  &self.scene_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        })
    }

    /// Draw all uploaded verts. Call between begin_render_pass and end.
    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, bg: &'a wgpu::BindGroup, n_verts: u32) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.set_vertex_buffer(0, self.vert_buf.slice(..));
        pass.draw(0..n_verts, 0..1);
    }
}

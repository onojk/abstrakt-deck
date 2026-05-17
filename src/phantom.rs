const PHANTOM_WIDTH:  u32   = 1280;
const PHANTOM_HEIGHT: u32   = 720;
const RING_SIZE:      usize = 90;
const CAPTURE_STRIDE: u32   = 2;

const CAPTURE_SHADER: &str = r#"
struct Vary { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> Vary {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: Vary;
    out.pos = vec4<f32>(x * 2.0 - 1.0, -(y * 2.0 - 1.0), 0.0, 1.0);
    out.uv  = vec2<f32>(x, y);
    return out;
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;

@fragment
fn fs_main(in: Vary) -> @location(0) vec4<f32> {
    return textureSample(src, smp, in.uv);
}
"#;

pub struct PhantomAlpha {
    #[allow(dead_code)]
    ring_textures:      Vec<wgpu::Texture>,
    ring_views:         Vec<wgpu::TextureView>,
    write_head:         usize,
    frame_counter:      u32,
    sampler:            wgpu::Sampler,
    capture_bgl:        wgpu::BindGroupLayout,
    capture_pipeline:   wgpu::RenderPipeline,
    chroma_buffer:      wgpu::Buffer,
    composite_bgl:      wgpu::BindGroupLayout,
    composite_pipeline: wgpu::RenderPipeline,
}

impl PhantomAlpha {
    pub fn new(device: &wgpu::Device, swapchain_format: wgpu::TextureFormat) -> Self {
        let ring_format = wgpu::TextureFormat::Rgba8Unorm;

        // Ring textures — fixed 1280×720 RGBA8
        let mut ring_textures = Vec::with_capacity(RING_SIZE);
        let mut ring_views    = Vec::with_capacity(RING_SIZE);
        for i in 0..RING_SIZE {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Phantom ring[{}]", i)),
                size: wgpu::Extent3d { width: PHANTOM_WIDTH, height: PHANTOM_HEIGHT, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: ring_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            ring_textures.push(tex);
            ring_views.push(view);
        }

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Phantom sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // ── Capture pipeline (scene → ring slot) ────────────────────────────
        let capture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Phantom capture BGL"),
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
            ],
        });
        let capture_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Phantom capture shader"),
            source: wgpu::ShaderSource::Wgsl(CAPTURE_SHADER.into()),
        });
        let capture_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Phantom capture layout"),
            bind_group_layouts: &[&capture_bgl],
            push_constant_ranges: &[],
        });
        let capture_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Phantom capture pipeline"),
            layout: Some(&capture_layout),
            vertex: wgpu::VertexState {
                module: &capture_shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &capture_shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ring_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None, cache: None,
        });

        // ── Composite pipeline (bg + ghost → swapchain) ─────────────────────
        let chroma_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Chroma uniforms buffer"),
            size: std::mem::size_of::<crate::ChromaUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Phantom composite BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let chroma_src = include_str!("shaders/phantom_chroma.wgsl");
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Phantom chroma shader"),
            source: wgpu::ShaderSource::Wgsl(chroma_src.into()),
        });
        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Phantom composite layout"),
            bind_group_layouts: &[&composite_bgl],
            push_constant_ranges: &[],
        });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Phantom composite pipeline"),
            layout: Some(&composite_layout),
            vertex: wgpu::VertexState {
                module: &composite_shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: swapchain_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None, cache: None,
        });

        Self {
            ring_textures,
            ring_views,
            write_head: 0,
            frame_counter: 0,
            sampler,
            capture_bgl,
            capture_pipeline,
            chroma_buffer,
            composite_bgl,
            composite_pipeline,
        }
    }

    /// Write scene_view → ring[write_head]. Only fires every CAPTURE_STRIDE frames.
    pub fn capture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        scene_view: &wgpu::TextureView,
    ) {
        if !self.frame_counter.is_multiple_of(CAPTURE_STRIDE) { return; }

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Phantom capture BG"),
            layout: &self.capture_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(scene_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Phantom capture pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.ring_views[self.write_head],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None, timestamp_writes: None,
        });
        pass.set_pipeline(&self.capture_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Chroma-key composite: bg=scene_view, ghost=delayed ring slot → screen_view.
    #[allow(clippy::too_many_arguments)]
    pub fn composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene_view: &wgpu::TextureView,
        screen_view: &wgpu::TextureView,
        delay_seconds: f32,
        key_color: [f32; 3],
        key_tolerance: f32,
        key_softness: f32,
        key_strength: f32,
        opacity: f32,
    ) {
        // Convert delay in seconds to ring-buffer slots.
        const FPS: f32 = 60.0;
        let delay_slots = ((delay_seconds * FPS / CAPTURE_STRIDE as f32).round() as usize)
            .clamp(1, RING_SIZE - 1);
        let read_idx = (self.write_head + RING_SIZE - delay_slots) % RING_SIZE;

        queue.write_buffer(
            &self.chroma_buffer, 0,
            bytemuck::cast_slice(&[crate::ChromaUniforms {
                key_color_r: key_color[0],
                key_color_g: key_color[1],
                key_color_b: key_color[2],
                key_tolerance,
                key_softness,
                key_strength,
                opacity,
                _pad: 0.0,
            }]),
        );

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Phantom composite BG"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.chroma_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(scene_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.ring_views[read_idx]) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Phantom composite pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: screen_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None, timestamp_writes: None,
        });
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Advance ring-buffer state — call once per render loop, after queue.submit().
    pub fn advance_frame(&mut self) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
        if self.frame_counter.is_multiple_of(CAPTURE_STRIDE) {
            self.write_head = (self.write_head + 1) % RING_SIZE;
        }
    }
}

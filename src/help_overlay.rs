use std::time::Instant;
use wgpu::util::DeviceExt;

pub struct HelpOverlay {
    #[allow(dead_code)] pub texture: wgpu::Texture,
    #[allow(dead_code)] pub view: wgpu::TextureView,
    #[allow(dead_code)] pub sampler: wgpu::Sampler,
    pub bind_group: wgpu::BindGroup,
    pub pipeline: wgpu::RenderPipeline,
    pub uniforms_buffer: wgpu::Buffer,

    pub width: u32,
    pub height: u32,

    pub visible: bool,
    pub anim_progress: f32,
    pub target: f32,
    pub last_update: Instant,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OverlayUniforms {
    pub viewport_x: f32,
    pub viewport_y: f32,
    pub panel_x: f32,
    pub panel_y: f32,
    pub anim_progress: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

impl HelpOverlay {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        png_bytes: &[u8],
    ) -> Self {
        let img = image::load_from_memory(png_bytes)
            .expect("decode cheat sheet PNG")
            .to_rgba8();
        let (width, height) = img.dimensions();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cheat sheet texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &img.into_raw(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Cheat sheet sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Overlay uniforms"),
            contents: bytemuck::cast_slice(&[OverlayUniforms {
                viewport_x: 1280.0, viewport_y: 720.0,
                panel_x: width as f32, panel_y: height as f32,
                anim_progress: 0.0,
                _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Overlay BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Overlay BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Overlay shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/overlay.wgsl").into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Overlay pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            texture, view, sampler, bind_group, pipeline, uniforms_buffer,
            width, height,
            visible: false,
            anim_progress: 0.0,
            target: 0.0,
            last_update: Instant::now(),
        }
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
        self.target = if self.visible { 1.0 } else { 0.0 };
        log::info!("help overlay: {}", self.visible);
    }

    pub fn update_animation(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_update).as_secs_f32();
        self.last_update = now;

        let speed = 4.0;
        let delta = self.target - self.anim_progress;
        if delta.abs() > 0.001 {
            self.anim_progress =
                (self.anim_progress + delta.signum() * speed * dt).clamp(0.0, 1.0);
        } else {
            self.anim_progress = self.target;
        }
    }

    pub fn should_render(&self) -> bool {
        self.anim_progress > 0.001
    }

    pub fn write_uniforms(&self, queue: &wgpu::Queue, viewport_w: u32, viewport_h: u32) {
        let t = 1.0 - (1.0 - self.anim_progress).powi(3);
        queue.write_buffer(
            &self.uniforms_buffer,
            0,
            bytemuck::cast_slice(&[OverlayUniforms {
                viewport_x: viewport_w as f32,
                viewport_y: viewport_h as f32,
                panel_x: self.width as f32,
                panel_y: self.height as f32,
                anim_progress: t,
                _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
            }]),
        );
    }
}


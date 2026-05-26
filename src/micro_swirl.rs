//! Micro-swirl screen-space distortion pass.
//!
//! Divides the frame into a density×density grid of cells, each rotating its
//! pixels around the cell centre with an independent phase and direction.
//! The swirl winds up then unwinds continuously, returning to exactly zero
//! distortion at every full cycle — no discontinuity in the loop.
//!
//! Mirrors bezold.rs structure exactly; slots in after Bezold in both the live
//! and export render paths.

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MicroSwirlUniforms {
    pub density:   f32,   // cells per screen-width
    pub amplitude: f32,   // peak rotation at cell centre, radians
    pub speed:     f32,   // oscillation cycles per second
    pub time:      f32,   // seconds since start
}

pub struct MicroSwirl {
    pub texture:         wgpu::Texture,
    pub view:            wgpu::TextureView,
    pub pipeline:        wgpu::RenderPipeline,
    pub bgl:             wgpu::BindGroupLayout,
    pub sampler:         wgpu::Sampler,
    pub uniforms_buffer: wgpu::Buffer,
    width:  u32,
    height: u32,
}

impl MicroSwirl {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let (texture, view) = Self::make_texture(device, width, height, format, false);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("MicroSwirl sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MicroSwirl uniforms"),
            size: std::mem::size_of::<MicroSwirlUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MicroSwirl BGL"),
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MicroSwirl shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/micro_swirl.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MicroSwirl pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("MicroSwirl pipeline"),
            layout: Some(&pipeline_layout),
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
                    format,
                    blend: None,
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

        Self { texture, view, pipeline, bgl, sampler, uniforms_buffer, width, height }
    }

    fn make_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        resized: bool,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let label = if resized {
            "MicroSwirl texture (resized)"
        } else {
            "MicroSwirl texture"
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
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

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) {
        if width == self.width && height == self.height { return; }
        let (texture, view) = Self::make_texture(device, width, height, format, true);
        self.texture = texture;
        self.view    = view;
        self.width   = width;
        self.height  = height;
    }

    pub fn write_uniforms(&self, queue: &wgpu::Queue, u: MicroSwirlUniforms) {
        queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::cast_slice(&[u]));
    }

    pub fn make_bind_group(
        &self,
        device: &wgpu::Device,
        scene_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:  Some("MicroSwirl BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(scene_view),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: self.uniforms_buffer.as_entire_binding(),
                },
            ],
        })
    }

    #[allow(dead_code)]
    pub fn width(&self)  -> u32 { self.width }
    #[allow(dead_code)]
    pub fn height(&self) -> u32 { self.height }
}

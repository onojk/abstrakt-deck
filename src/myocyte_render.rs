// Myocyte splat rasterizer, ported and adapted from myocyte. Phase 5b:
// full implementation — projects cells, sorts back-to-front, uploads
// to GPU, and draws into deck's shape FBO via the existing Pass 2.

use bytemuck::cast_slice;

use crate::cell::CellGrid;
use crate::myocyte_camera::{CameraUniform, OrbitCamera};
use crate::myocyte_preprocess::{project_grid, SplatGpuData};
use crate::myocyte_sort::sort_back_to_front;

/// Maximum splats we ever upload in one frame. Sized for 32³ = 32768.
pub const MYOCYTE_MAX_SPLATS: usize = 32_768;

pub struct MyocyteRenderer {
    pub camera: OrbitCamera,

    pipeline:          wgpu::RenderPipeline,
    camera_buffer:     wgpu::Buffer,
    splat_buffer:      wgpu::Buffer,
    shared_bind_group: wgpu::BindGroup,

    // Pre-allocated scratch to avoid per-frame heap allocation.
    splats_scratch:       Vec<SplatGpuData>,
    sort_indices_scratch: Vec<u32>,
}

impl MyocyteRenderer {
    pub fn new(device: &wgpu::Device, shape_fbo_format: wgpu::TextureFormat) -> Self {
        // ---- Camera -------------------------------------------------------
        let mut camera = OrbitCamera::default();
        // Distance 18.0 gives a comfortable view of the 16³ grid (±7.5 wu).
        camera.distance = 18.0;
        camera.auto_rotation_rate_rad_per_s = std::f32::consts::TAU / 10.0;

        // ---- GPU buffers --------------------------------------------------
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("myocyte camera uniform"),
            size:               std::mem::size_of::<CameraUniform>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let splat_buffer_size = (MYOCYTE_MAX_SPLATS * std::mem::size_of::<SplatGpuData>()) as u64;
        let splat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("myocyte splat storage"),
            size:               splat_buffer_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- Bind group layout --------------------------------------------
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("myocyte shared bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let shared_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("myocyte shared bg"),
            layout:  &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: splat_buffer.as_entire_binding() },
            ],
        });

        // ---- Pipeline -----------------------------------------------------
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("myocyte splat shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/myocyte_splat.wgsl").into()
            ),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("myocyte splat pipeline layout"),
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("myocyte splat pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[],  // procedural — no vertex buffers
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: shape_fbo_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation:  wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation:  wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:           wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face:         wgpu::FrontFace::Ccw,
                cull_mode:          None,
                polygon_mode:       wgpu::PolygonMode::Fill,
                unclipped_depth:    false,
                conservative:       false,
            },
            // Must match deck's shape pass depth attachment (Depth32Float).
            // We don't write depth; every fragment passes the test (Always).
            depth_stencil: Some(wgpu::DepthStencilState {
                format:               wgpu::TextureFormat::Depth32Float,
                depth_write_enabled:  false,
                depth_compare:        wgpu::CompareFunction::Always,
                stencil:              wgpu::StencilState::default(),
                bias:                 wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview:   None,
            cache:       None,
        });

        Self {
            camera,
            pipeline,
            camera_buffer,
            splat_buffer,
            shared_bind_group,
            splats_scratch:       Vec::with_capacity(MYOCYTE_MAX_SPLATS),
            sort_indices_scratch: Vec::with_capacity(MYOCYTE_MAX_SPLATS),
        }
    }

    /// Render the cell grid into the given render pass.
    ///   1. Project all cells through self.camera → SplatWithDepth list
    ///   2. Back-to-front sort
    ///   3. Upload camera uniform + sorted splat data
    ///   4. Draw 6 vertices × N_splats instances
    pub fn render<'rpass>(
        &mut self,
        grid:   &CellGrid,
        queue:  &wgpu::Queue,
        pass:   &mut wgpu::RenderPass<'rpass>,
        _aspect: f32,
    ) {
        // Project and sort.
        let mut projected = project_grid(grid, &self.camera);
        if projected.is_empty() { return; }

        let sorted = sort_back_to_front(&mut projected);
        let n = sorted.len().min(MYOCYTE_MAX_SPLATS);
        let sorted = &sorted[..n];

        // Upload camera uniform.
        let cam = CameraUniform::from_camera(&self.camera);
        queue.write_buffer(&self.camera_buffer, 0, cast_slice(&[cam]));

        // Upload sorted splat data.
        queue.write_buffer(&self.splat_buffer, 0, cast_slice(sorted));

        // Draw: 6 vertices per splat quad, n instances.
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.shared_bind_group, &[]);
        pass.draw(0..6, 0..n as u32);

        // Keep scratch capacity for next frame (avoids re-allocation).
        self.splats_scratch.clear();
        let _ = &self.sort_indices_scratch;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.camera.set_aspect(width as f32 / height as f32);
    }
}

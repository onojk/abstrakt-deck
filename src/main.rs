mod audio;
mod midi;
mod shape;
use audio::{AudioCapture, AudioEvent};
use midi::{MidiCapture, MidiEvent};
use shape::{ShapeKind, Vertex};

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const PAINTER_TEXTURE_WIDTH: u32 = 2048;
const PAINTER_TEXTURE_HEIGHT: u32 = 1024;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobalUniforms {
    time_seconds: f32,
    resolution_x: f32,
    resolution_y: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Transform {
    mvp: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct KaleidoUniforms {
    resolution_x: f32,
    resolution_y: f32,
    fold_count: f32,
    zoom: f32,
}

// 0=none 1=circle 2=square 3=rounded 4=hexagon 5=octagon 6=star
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniforms {
    resolution_x:  f32,
    resolution_y:  f32,
    frame_color_r: f32,
    frame_color_g: f32,
    frame_color_b: f32,
    frame_color_a: f32,
    frame_shape:   f32,
    frame_size:    f32,
}

impl FrameUniforms {
    fn default_for(width: f32, height: f32) -> Self {
        Self {
            resolution_x:  width,
            resolution_y:  height,
            frame_color_r: 0.0,
            frame_color_g: 0.9,
            frame_color_b: 1.0,  // neon cyan
            frame_color_a: 1.0,
            frame_shape:   4.0,  // hexagon
            frame_size:    0.85,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum FrameShape {
    None    = 0,
    Circle  = 1,
    Square  = 2,
    Rounded = 3,
    Hexagon = 4,
    Octagon = 5,
    Star    = 6,
}

impl FrameShape {
    fn as_f32(self) -> f32 { self as i32 as f32 }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = (h % 360.0 + 360.0) % 360.0;
    let c = v * s;
    let h6 = h / 60.0;
    let x = c * (1.0 - ((h6 % 2.0) - 1.0).abs());
    let (r, g, b) = match h6 as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    (r + m, g + m, b + m)
}

struct ShapeBufferSet {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    uniforms_buffer: wgpu::Buffer,

    // Pass 1 — painter (Hue Stripe procedural → fixed 2048×1024)
    painter_uniforms_bind_group: wgpu::BindGroup,
    painter_pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    painter_texture: wgpu::Texture,
    painter_view: wgpu::TextureView,
    #[allow(dead_code)]
    painter_sampler: wgpu::Sampler,

    // Pass 2 — shape (3D mesh with painter surface → screen-res, depth-tested)
    #[allow(dead_code)]
    shape_texture: wgpu::Texture,
    shape_view: wgpu::TextureView,
    #[allow(dead_code)]
    shape_depth: wgpu::Texture,
    shape_depth_view: wgpu::TextureView,
    shape_buffers: HashMap<ShapeKind, ShapeBufferSet>,
    transform_buffer: wgpu::Buffer,
    transform_bind_group: wgpu::BindGroup,
    shape_pipeline: wgpu::RenderPipeline,
    shape_bind_group: wgpu::BindGroup,

    // Pass 3 — kaleido fold (shape FBO → kaleido FBO)
    #[allow(dead_code)]
    kaleido_texture: wgpu::Texture,
    kaleido_view: wgpu::TextureView,
    kaleido_uniforms_buffer: wgpu::Buffer,
    kaleido_bgl: wgpu::BindGroupLayout,
    kaleido_bind_group: wgpu::BindGroup,
    kaleido_pipeline: wgpu::RenderPipeline,

    // Pass 4 — frame overlay (kaleido FBO + SDF mask → sRGB swapchain)
    frame_uniforms_buffer: wgpu::Buffer,
    frame_bgl: wgpu::BindGroupLayout,
    frame_bind_group: wgpu::BindGroup,
    frame_pipeline: wgpu::RenderPipeline,
    shape_sampler: wgpu::Sampler,   // ClampToEdge sampler reused by kaleido + frame passes

    start_time: Instant,

    shake_offset: glam::Vec3,
    shake_velocity: glam::Vec3,

    // Runtime-mutable parameters (keyboard-controlled)
    fold_count: f32,
    zoom: f32,
    rotation_speed_scale: f32,
    frame_shape: FrameShape,
    frame_size: f32,
    frame_color_hue: f32,
    shake_enabled: bool,
    current_shape: ShapeKind,
}

impl GpuState {
    fn create_shape_fbo(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::Texture, wgpu::TextureView) {
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shape FBO color"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shape FBO depth"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        (color, color_view, depth, depth_view)
    }

    fn create_kaleido_fbo(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Kaleido FBO"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    /// Shared layout: group 0 = { uniform, texture, sampler }.
    /// Used by both the kaleido and frame passes.
    fn make_uts_bgl(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find suitable GPU adapter");

        log::info!("Adapter: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("abstrakt-deck device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats.iter().find(|f| f.is_srgb()).copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let w = size.width.max(1);
        let h = size.height.max(1);

        // ── Painter texture (fixed 2048×1024) ─────────────────────────────────
        let painter_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Painter texture"),
            size: wgpu::Extent3d {
                width: PAINTER_TEXTURE_WIDTH,
                height: PAINTER_TEXTURE_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let painter_view = painter_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let painter_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Painter sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // ── Screen-res FBOs ────────────────────────────────────────────────────
        let (shape_texture, shape_view, shape_depth, shape_depth_view) =
            Self::create_shape_fbo(&device, w, h);
        let (kaleido_texture, kaleido_view) = Self::create_kaleido_fbo(&device, w, h);

        let shape_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shape/kaleido/frame sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // ── Shape meshes (all built at startup, switched at runtime) ──────────
        let meshes = shape::build_all_shapes();
        let mut shape_buffers = HashMap::new();
        for (kind, mesh) in &meshes {
            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} vertex buffer", kind)),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} index buffer", kind)),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            shape_buffers.insert(*kind, ShapeBufferSet {
                vertex_buffer: vb,
                index_buffer: ib,
                index_count: mesh.indices.len() as u32,
            });
        }

        // ── Transform uniform ──────────────────────────────────────────────────
        let transform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Transform buffer"),
                contents: bytemuck::cast_slice(&[Transform {
                    mvp: glam::Mat4::IDENTITY.to_cols_array_2d(),
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let transform_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Transform BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Transform BG"),
            layout: &transform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: transform_buffer.as_entire_binding(),
            }],
        });

        // ── Shape texture BGL (painter view + sampler for shape pipeline) ──────
        let tex2_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Tex2 BGL"),
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
        let shape_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shape BG (samples painter)"),
            layout: &tex2_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&painter_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&painter_sampler),
                },
            ],
        });

        // ── Globals uniform (painter pass) ────────────────────────────────────
        let uniforms_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Globals uniform buffer"),
                contents: bytemuck::cast_slice(&[GlobalUniforms {
                    time_seconds: 0.0,
                    resolution_x: w as f32,
                    resolution_y: h as f32,
                    _pad: 0.0,
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let uniforms_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Uniforms BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let painter_uniforms_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Painter uniforms BG"),
                layout: &uniforms_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buffer.as_entire_binding(),
                }],
            });

        // ── Kaleido uniforms + BGL + BG ────────────────────────────────────────
        let kaleido_uniforms_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Kaleido uniforms"),
                contents: bytemuck::cast_slice(&[KaleidoUniforms {
                    resolution_x: w as f32,
                    resolution_y: h as f32,
                    fold_count: 12.0,
                    zoom: 0.6,
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let kaleido_bgl = Self::make_uts_bgl(&device, "Kaleido BGL");
        let kaleido_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Kaleido BG"),
            layout: &kaleido_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: kaleido_uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shape_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shape_sampler),
                },
            ],
        });

        // ── Frame uniforms + BGL + BG ─────────────────────────────────────────
        let frame_uniforms_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Frame uniforms"),
                contents: bytemuck::cast_slice(&[FrameUniforms::default_for(w as f32, h as f32)]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let frame_bgl = Self::make_uts_bgl(&device, "Frame BGL");
        let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Frame BG"),
            layout: &frame_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&kaleido_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shape_sampler),
                },
            ],
        });

        // ── Pipelines ─────────────────────────────────────────────────────────
        let painter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Painter shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fullscreen.wgsl").into()),
        });
        let painter_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Painter pipeline"),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None, bind_group_layouts: &[&uniforms_bgl], push_constant_ranges: &[],
                })),
                vertex: wgpu::VertexState {
                    module: &painter_shader, entry_point: Some("vs_main"),
                    buffers: &[], compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &painter_shader, entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::REPLACE),
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

        let shape_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shape shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shape.wgsl").into()),
        });
        let shape_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Shape pipeline"),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&transform_bgl, &tex2_bgl],
                    push_constant_ranges: &[],
                })),
                vertex: wgpu::VertexState {
                    module: &shape_shader, entry_point: Some("vs_main"),
                    buffers: &[Vertex::LAYOUT], compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shape_shader, entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None, cache: None,
            });

        let kaleido_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Kaleido shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/kaleido.wgsl").into()),
        });
        let kaleido_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Kaleido pipeline"),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None, bind_group_layouts: &[&kaleido_bgl], push_constant_ranges: &[],
                })),
                vertex: wgpu::VertexState {
                    module: &kaleido_shader, entry_point: Some("vs_main"),
                    buffers: &[], compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &kaleido_shader, entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::REPLACE),
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

        let frame_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Frame shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/frame.wgsl").into()),
        });
        let frame_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Frame pipeline"),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None, bind_group_layouts: &[&frame_bgl], push_constant_ranges: &[],
                })),
                vertex: wgpu::VertexState {
                    module: &frame_shader, entry_point: Some("vs_main"),
                    buffers: &[], compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &frame_shader, entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format, // final pass → sRGB swapchain
                        blend: Some(wgpu::BlendState::REPLACE),
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
            surface, device, queue, config, size,
            uniforms_buffer,
            painter_uniforms_bind_group, painter_pipeline,
            painter_texture, painter_view, painter_sampler,
            shape_texture, shape_view, shape_depth, shape_depth_view,
            shape_buffers,
            transform_buffer, transform_bind_group,
            shape_pipeline, shape_bind_group,
            kaleido_texture, kaleido_view,
            kaleido_uniforms_buffer, kaleido_bgl, kaleido_bind_group, kaleido_pipeline,
            frame_uniforms_buffer, frame_bgl, frame_bind_group, frame_pipeline,
            shape_sampler,
            start_time: Instant::now(),
            shake_offset: glam::Vec3::ZERO,
            shake_velocity: glam::Vec3::ZERO,
            fold_count: 12.0,
            zoom: 0.6,
            rotation_speed_scale: 1.0,
            frame_shape: FrameShape::Hexagon,
            frame_size: 0.85,
            frame_color_hue: 195.0,
            shake_enabled: true,
            current_shape: ShapeKind::Cylinder,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }
        let w = new_size.width;
        let h = new_size.height;
        self.size = new_size;
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);

        // Recreate screen-res FBOs.
        let (sc, sv, sd, sdv) = Self::create_shape_fbo(&self.device, w, h);
        self.shape_texture = sc; self.shape_view = sv;
        self.shape_depth = sd;   self.shape_depth_view = sdv;

        let (kc, kv) = Self::create_kaleido_fbo(&self.device, w, h);
        self.kaleido_texture = kc;
        self.kaleido_view = kv;

        // Kaleido BG references shape_view → recreate.
        self.kaleido_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Kaleido BG"),
                layout: &self.kaleido_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.kaleido_uniforms_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.shape_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.shape_sampler),
                    },
                ],
            });

        // Frame BG references kaleido_view → recreate.
        self.frame_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Frame BG"),
                layout: &self.frame_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.frame_uniforms_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.kaleido_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.shape_sampler),
                    },
                ],
            });

        // Update resolution in kaleido and frame uniforms.
        self.queue.write_buffer(
            &self.kaleido_uniforms_buffer, 0,
            bytemuck::cast_slice(&[KaleidoUniforms {
                resolution_x: w as f32, resolution_y: h as f32,
                fold_count: 12.0, zoom: 0.6,
            }]),
        );
        self.queue.write_buffer(
            &self.frame_uniforms_buffer, 0,
            bytemuck::cast_slice(&[FrameUniforms::default_for(w as f32, h as f32)]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let elapsed = self.start_time.elapsed().as_secs_f32();

        self.queue.write_buffer(
            &self.uniforms_buffer, 0,
            bytemuck::cast_slice(&[GlobalUniforms {
                time_seconds: elapsed,
                resolution_x: self.size.width as f32,
                resolution_y: self.size.height as f32,
                _pad: 0.0,
            }]),
        );

        self.queue.write_buffer(
            &self.kaleido_uniforms_buffer, 0,
            bytemuck::cast_slice(&[KaleidoUniforms {
                resolution_x: self.size.width as f32,
                resolution_y: self.size.height as f32,
                fold_count: self.fold_count,
                zoom: self.zoom,
            }]),
        );

        let (fr, fg, fb) = hsv_to_rgb(self.frame_color_hue, 0.85, 1.0);
        self.queue.write_buffer(
            &self.frame_uniforms_buffer, 0,
            bytemuck::cast_slice(&[FrameUniforms {
                resolution_x:  self.size.width as f32,
                resolution_y:  self.size.height as f32,
                frame_color_r: fr,
                frame_color_g: fg,
                frame_color_b: fb,
                frame_color_a: 1.0,
                frame_shape:   self.frame_shape.as_f32(),
                frame_size:    self.frame_size,
            }]),
        );

        let aspect = self.size.width as f32 / self.size.height as f32;
        let proj = glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
        let cam = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.5, 3.0), glam::Vec3::ZERO, glam::Vec3::Y,
        );
        // Spring-damping decay of shake offset toward zero
        let dt = 1.0 / 60.0;
        let stiffness = 30.0_f32;
        let damping = 8.0_f32;
        let force = -stiffness * self.shake_offset - damping * self.shake_velocity;
        self.shake_velocity += force * dt;
        self.shake_offset += self.shake_velocity * dt;

        let base_rotation_speed = std::f32::consts::TAU / 30.0;
        let model = glam::Mat4::from_translation(self.shake_offset)
            * glam::Mat4::from_rotation_y(elapsed * base_rotation_speed * self.rotation_speed_scale)
            * glam::Mat4::from_scale(glam::Vec3::splat(self.current_shape.model_scale()));
        self.queue.write_buffer(
            &self.transform_buffer, 0,
            bytemuck::cast_slice(&[Transform { mvp: (proj * cam * model).to_cols_array_2d() }]),
        );

        let output = self.surface.get_current_texture()?;
        let screen_view =
            output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame encoder"),
            });

        // Pass 1: Hue Stripe → painter FBO
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Painter pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.painter_view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None, timestamp_writes: None,
            });
            pass.set_pipeline(&self.painter_pipeline);
            pass.set_bind_group(0, &self.painter_uniforms_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 2: cylinder → shape FBO
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shape pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.shape_view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shape_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None, timestamp_writes: None,
            });
            pass.set_pipeline(&self.shape_pipeline);
            pass.set_bind_group(0, &self.transform_bind_group, &[]);
            pass.set_bind_group(1, &self.shape_bind_group, &[]);
            let buffers = &self.shape_buffers[&self.current_shape];
            pass.set_vertex_buffer(0, buffers.vertex_buffer.slice(..));
            pass.set_index_buffer(buffers.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..buffers.index_count, 0, 0..1);
        }

        // Pass 3: kaleido fold → kaleido FBO
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Kaleido pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.kaleido_view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None, timestamp_writes: None,
            });
            pass.set_pipeline(&self.kaleido_pipeline);
            pass.set_bind_group(0, &self.kaleido_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 4: frame overlay (SDF mask) → sRGB swapchain
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Frame pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &screen_view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None, timestamp_writes: None,
            });
            pass.set_pipeline(&self.frame_pipeline);
            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    pub fn kick_shake(&mut self, strength: f32) {
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let angle = (elapsed * 13.7) % std::f32::consts::TAU;
        let dir = glam::Vec3::new(angle.cos(), 0.0, angle.sin());
        self.shake_velocity += dir * strength * 0.75;
    }
}

struct FpsCounter {
    frames: u32,
    last_report: Instant,
    last_fps: f64,
}

impl FpsCounter {
    fn new() -> Self {
        Self { frames: 0, last_report: Instant::now(), last_fps: 0.0 }
    }

    fn tick(&mut self) -> Option<f64> {
        self.frames += 1;
        let elapsed = self.last_report.elapsed();
        if elapsed >= Duration::from_secs(1) {
            self.last_fps = self.frames as f64 / elapsed.as_secs_f64();
            self.frames = 0;
            self.last_report = Instant::now();
            Some(self.last_fps)
        } else {
            None
        }
    }
}

fn apply_midi_event(gpu: &mut GpuState, event: MidiEvent) {
    match event {
        MidiEvent::ControlChange(cc, value) => {
            let v = value as f32 / 127.0;
            match cc {
                1 => {
                    gpu.fold_count = (2.0 + v * 22.0).round();
                    log::debug!("MIDI CC1 → fold_count = {}", gpu.fold_count);
                }
                7 => {
                    gpu.zoom = 0.3 + v * 1.2;
                    log::debug!("MIDI CC7 → zoom = {:.2}", gpu.zoom);
                }
                10 => {
                    gpu.rotation_speed_scale = v * 4.0;
                    log::debug!("MIDI CC10 → rotation_speed_scale = {:.2}", gpu.rotation_speed_scale);
                }
                65 if value >= 64 => {
                    gpu.current_shape = gpu.current_shape.next();
                    log::debug!("MIDI CC65 → shape cycled to {}", gpu.current_shape.name());
                }
                64 if value >= 64 => {
                    gpu.frame_shape = match gpu.frame_shape {
                        FrameShape::None    => FrameShape::Circle,
                        FrameShape::Circle  => FrameShape::Square,
                        FrameShape::Square  => FrameShape::Rounded,
                        FrameShape::Rounded => FrameShape::Hexagon,
                        FrameShape::Hexagon => FrameShape::Octagon,
                        FrameShape::Octagon => FrameShape::Star,
                        FrameShape::Star    => FrameShape::None,
                    };
                    log::debug!("MIDI CC64 → frame_shape cycled to {:?}", gpu.frame_shape);
                }
                71 => {
                    gpu.frame_size = 0.4 + v * 0.6;
                    log::debug!("MIDI CC71 → frame_size = {:.2}", gpu.frame_size);
                }
                74 => {
                    gpu.frame_color_hue = v * 360.0;
                    log::debug!("MIDI CC74 → frame_color_hue = {:.0}°", gpu.frame_color_hue);
                }
                _ => {
                    log::trace!("MIDI CC {} value {} (unmapped)", cc, value);
                }
            }
        }
        MidiEvent::NoteOn(note, velocity) => {
            let strength = velocity as f32 / 127.0;
            gpu.kick_shake(strength);
            log::debug!("MIDI Note On {} vel {} → shake {:.2}", note, velocity, strength);
        }
        MidiEvent::NoteOff(_) => {}
    }
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    fps: FpsCounter,
    audio: Option<AudioCapture>,
    midi: Option<MidiCapture>,
}

impl App {
    fn new() -> Self {
        Self { window: None, gpu: None, fps: FpsCounter::new(), audio: None, midi: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        let attrs = Window::default_attributes()
            .with_title("abstrakt-deck — slice 11 — Multi-shape")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        let window = Arc::new(
            event_loop.create_window(attrs).expect("Failed to create window"),
        );
        let gpu = pollster::block_on(GpuState::new(window.clone()));
        self.window = Some(window);
        self.gpu = Some(gpu);
        log::info!("Window and GPU initialized");

        let audio = match AudioCapture::start() {
            Ok(a) => Some(a),
            Err(e) => {
                log::warn!(
                    "Audio capture failed: {} — visualizer will run without audio reactivity",
                    e
                );
                None
            }
        };
        self.audio = audio;

        let midi = match MidiCapture::start() {
            Ok(m) => Some(m),
            Err(e) => {
                log::warn!(
                    "MIDI capture failed: {} — visualizer will run without MIDI control",
                    e
                );
                None
            }
        };
        self.midi = midi;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(gpu) = self.gpu.as_mut() else { return };
        let Some(window) = self.window.as_ref() else { return };

        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested");
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(key_code),
                    ..
                },
                ..
            } => {
                match key_code {
                    KeyCode::Escape => {
                        log::info!("Escape pressed");
                        event_loop.exit();
                    }
                    KeyCode::BracketLeft => {
                        gpu.fold_count = (gpu.fold_count - 1.0).max(2.0);
                        log::info!("fold_count = {}", gpu.fold_count);
                    }
                    KeyCode::BracketRight => {
                        gpu.fold_count = (gpu.fold_count + 1.0).min(24.0);
                        log::info!("fold_count = {}", gpu.fold_count);
                    }
                    KeyCode::KeyZ => {
                        gpu.zoom = (gpu.zoom - 0.05).max(0.3);
                        log::info!("zoom = {:.2}", gpu.zoom);
                    }
                    KeyCode::KeyX => {
                        gpu.zoom = (gpu.zoom + 0.05).min(1.5);
                        log::info!("zoom = {:.2}", gpu.zoom);
                    }
                    KeyCode::Comma => {
                        gpu.rotation_speed_scale = (gpu.rotation_speed_scale - 0.25).max(0.0);
                        log::info!("rotation_speed_scale = {:.2}", gpu.rotation_speed_scale);
                    }
                    KeyCode::Period => {
                        gpu.rotation_speed_scale = (gpu.rotation_speed_scale + 0.25).min(4.0);
                        log::info!("rotation_speed_scale = {:.2}", gpu.rotation_speed_scale);
                    }
                    KeyCode::Digit1 => { gpu.frame_shape = FrameShape::None;    log::info!("frame: None"); }
                    KeyCode::Digit2 => { gpu.frame_shape = FrameShape::Circle;  log::info!("frame: Circle"); }
                    KeyCode::Digit3 => { gpu.frame_shape = FrameShape::Square;  log::info!("frame: Square"); }
                    KeyCode::Digit4 => { gpu.frame_shape = FrameShape::Rounded; log::info!("frame: Rounded"); }
                    KeyCode::Digit5 => { gpu.frame_shape = FrameShape::Hexagon; log::info!("frame: Hexagon"); }
                    KeyCode::Digit6 => { gpu.frame_shape = FrameShape::Octagon; log::info!("frame: Octagon"); }
                    KeyCode::Digit7 => { gpu.frame_shape = FrameShape::Star;    log::info!("frame: Star"); }
                    KeyCode::Minus => {
                        gpu.frame_size = (gpu.frame_size - 0.05).max(0.4);
                        log::info!("frame_size = {:.2}", gpu.frame_size);
                    }
                    KeyCode::Equal => {
                        gpu.frame_size = (gpu.frame_size + 0.05).min(1.0);
                        log::info!("frame_size = {:.2}", gpu.frame_size);
                    }
                    KeyCode::KeyR => {
                        gpu.frame_color_hue = (gpu.frame_color_hue + 30.0) % 360.0;
                        log::info!("frame_color_hue = {:.0}°", gpu.frame_color_hue);
                    }
                    KeyCode::KeyG => {
                        gpu.frame_color_hue = (gpu.frame_color_hue + 120.0) % 360.0;
                        log::info!("frame_color_hue = {:.0}°", gpu.frame_color_hue);
                    }
                    KeyCode::KeyB => {
                        gpu.frame_color_hue = (gpu.frame_color_hue + 60.0) % 360.0;
                        log::info!("frame_color_hue = {:.0}°", gpu.frame_color_hue);
                    }
                    KeyCode::Space => {
                        gpu.shake_enabled = !gpu.shake_enabled;
                        log::info!("shake_enabled = {}", gpu.shake_enabled);
                    }
                    KeyCode::Tab => {
                        gpu.current_shape = gpu.current_shape.next();
                        log::info!("shape: {}", gpu.current_shape.name());
                    }
                    _ => {}
                }
            }
            WindowEvent::Resized(physical_size) => {
                gpu.resize(physical_size);
            }
            WindowEvent::RedrawRequested => {
                if let Some(audio) = &self.audio {
                    while let Ok(event) = audio.event_rx.try_recv() {
                        match event {
                            AudioEvent::Beat(strength) => {
                                if gpu.shake_enabled {
                                    gpu.kick_shake(strength);
                                }
                            }
                        }
                    }
                }
                if let Some(midi) = &self.midi {
                    while let Ok(event) = midi.event_rx.try_recv() {
                        apply_midi_event(gpu, event);
                    }
                }
                match gpu.render() {
                    Ok(()) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        gpu.resize(gpu.size);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        log::error!("Surface out of memory");
                        event_loop.exit();
                    }
                    Err(e) => {
                        log::warn!("Surface error: {:?}", e);
                    }
                }
                if let Some(fps) = self.fps.tick() {
                    window.set_title(&format!(
                        "abstrakt-deck — slice 11 — Multi-shape — {:.1} fps",
                        fps
                    ));
                }
                window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("\nabstrakt-deck — keyboard controls:");
    println!("  [ ]    fold count   (2 to 24)");
    println!("  z x    kaleido zoom (0.30 to 1.50)");
    println!("  , .    rotation speed (0 to 4×)");
    println!("  1-7    frame shape (None/Circle/Square/Rounded/Hexagon/Octagon/Star)");
    println!("  - =    frame size");
    println!("  R G B  cycle frame color hue");
    println!("  space  toggle beat-reactive shake");
    println!("  Tab    cycle shape (Cylinder → Sphere → Cube → Tetrahedron)");
    println!("  esc    exit");
    println!("\nMIDI control (if device connected):");
    println!("  CC 1   fold count");
    println!("  CC 7   zoom");
    println!("  CC 10  rotation speed");
    println!("  CC 64  cycle frame shape");
    println!("  CC 65  cycle shape (portamento)");
    println!("  CC 71  frame size");
    println!("  CC 74  frame color hue");
    println!("  Note On  trigger shake (like a beat)\n");

    log::info!("abstrakt-deck starting");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop failed");

    log::info!("abstrakt-deck shutting down");
}

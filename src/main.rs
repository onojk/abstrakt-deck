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

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    // Uniforms
    uniforms_buffer: wgpu::Buffer,
    // Pass 1 — painter renders procedural content to offscreen texture
    painter_uniforms_bind_group: wgpu::BindGroup,
    painter_pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)] // kept for resource lifetime; resize will recreate the view
    painter_texture: wgpu::Texture,
    painter_view: wgpu::TextureView,
    #[allow(dead_code)] // kept for resource lifetime; bind group holds the GPU ref
    painter_sampler: wgpu::Sampler,
    // Pass 2 — composite blits the painter texture to the swapchain
    composite_pipeline: wgpu::RenderPipeline,
    composite_bind_group: wgpu::BindGroup,
    start_time: Instant,
}

impl GpuState {
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
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
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

        // --- Painter offscreen texture ---
        let painter_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Painter texture"),
            size: wgpu::Extent3d {
                width: PAINTER_TEXTURE_WIDTH,
                height: PAINTER_TEXTURE_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let painter_view = painter_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let painter_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Painter sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,   // wraps for future cylinder use
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // --- Uniforms (shared by painter pass) ---
        let initial_uniforms = GlobalUniforms {
            time_seconds: 0.0,
            resolution_x: size.width as f32,
            resolution_y: size.height as f32,
            _pad: 0.0,
        };

        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Globals uniform buffer"),
            contents: bytemuck::cast_slice(&[initial_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Uniforms bind group layout"),
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
                label: Some("Painter uniforms bind group"),
                layout: &uniforms_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buffer.as_entire_binding(),
                }],
            });

        // --- Painter pipeline (procedural → Rgba8Unorm texture) ---
        let painter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Painter shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/fullscreen.wgsl").into(),
            ),
        });

        let painter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Painter pipeline layout"),
                bind_group_layouts: &[&uniforms_bind_group_layout],
                push_constant_ranges: &[],
            });

        let painter_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Painter pipeline"),
                layout: Some(&painter_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &painter_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &painter_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        // Must match painter_texture format exactly.
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        // --- Composite pipeline (painter texture → swapchain) ---
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/composite.wgsl").into(),
            ),
        });

        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite bind group layout"),
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

        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Composite bind group"),
            layout: &composite_bind_group_layout,
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

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite pipeline layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[],
            });

        let composite_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Composite pipeline"),
                layout: Some(&composite_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &composite_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &composite_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format, // composite → sRGB swapchain
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            uniforms_buffer,
            painter_uniforms_bind_group,
            painter_pipeline,
            painter_texture,
            painter_view,
            painter_sampler,
            composite_pipeline,
            composite_bind_group,
            start_time: Instant::now(),
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Upload uniforms
        let uniforms = GlobalUniforms {
            time_seconds: self.start_time.elapsed().as_secs_f32(),
            resolution_x: self.size.width as f32,
            resolution_y: self.size.height as f32,
            _pad: 0.0,
        };
        self.queue.write_buffer(
            &self.uniforms_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );

        let output = self.surface.get_current_texture()?;
        let screen_view =
            output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame encoder"),
            });

        // Pass 1: painter → offscreen Rgba8Unorm texture
        {
            let mut painter_pass =
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Painter pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.painter_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
            painter_pass.set_pipeline(&self.painter_pipeline);
            painter_pass.set_bind_group(0, &self.painter_uniforms_bind_group, &[]);
            painter_pass.draw(0..3, 0..1);
        }

        // Pass 2: composite painter texture → swapchain
        {
            let mut composite_pass =
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Composite pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &screen_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
            composite_pass.set_pipeline(&self.composite_pipeline);
            composite_pass.set_bind_group(0, &self.composite_bind_group, &[]);
            composite_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
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

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    fps: FpsCounter,
}

impl App {
    fn new() -> Self {
        Self { window: None, gpu: None, fps: FpsCounter::new() }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("abstrakt-deck — slice 4")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = Arc::new(
            event_loop.create_window(attrs).expect("Failed to create window"),
        );

        let gpu = pollster::block_on(GpuState::new(window.clone()));

        self.window = Some(window);
        self.gpu = Some(gpu);

        log::info!("Window and GPU initialized");
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
                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                    ..
                },
                ..
            } => {
                log::info!("Escape pressed");
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                gpu.resize(physical_size);
            }
            WindowEvent::RedrawRequested => {
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
                        "abstrakt-deck — slice 4 — Hue Stripe — {:.1} fps",
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

    log::info!("abstrakt-deck starting");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop failed");

    log::info!("abstrakt-deck shutting down");
}

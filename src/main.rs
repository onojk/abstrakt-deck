mod audio;
mod help_overlay;
mod menu_bar;
mod midi;
mod recorder;
mod shape;
use audio::{AudioCapture, AudioEvent};
use help_overlay::HelpOverlay;
use menu_bar::MenuBar;
use midi::{MidiCapture, MidiEvent};
use recorder::Recorder;
use shape::{ShapeKind, Vertex};

static CHEAT_SHEET_PNG: &[u8] = include_bytes!("../assets/cheat_sheet.png");
use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const PAINTER_TEXTURE_WIDTH: u32 = 4096;
const PAINTER_TEXTURE_HEIGHT: u32 = 256;

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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ShapeEffects {
    invert:               f32,
    colorize_enabled:     f32,
    colorize_hue:         f32,
    colorize_intensity:   f32,
    distortion_enabled:   f32,
    distortion_amplitude: f32,
    distortion_frequency: f32,
    time_seconds:         f32,
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
pub enum FrameShape {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PainterKind {
    HueStripe,
    Spiral,
    Plasma,
}

impl PainterKind {
    pub fn next(self) -> Self {
        match self {
            PainterKind::HueStripe => PainterKind::Spiral,
            PainterKind::Spiral    => PainterKind::Plasma,
            PainterKind::Plasma    => PainterKind::HueStripe,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            PainterKind::HueStripe => "HueStripe",
            PainterKind::Spiral    => "Spiral",
            PainterKind::Plasma    => "Plasma",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisualParams {
    pub current_shape: ShapeKind,
    pub fold_count: f32,
    pub zoom: f32,
    pub rotation_speed_scale: f32,
    pub frame_shape: FrameShape,
    pub frame_size: f32,
    pub frame_color_hue: f32,
    pub invert_enabled: bool,
    pub colorize_enabled: bool,
    pub colorize_hue: f32,
    pub colorize_intensity: f32,
    pub distortion_enabled: bool,
    pub distortion_amplitude: f32,
    pub distortion_frequency: f32,
    pub shake_enabled: bool,
    pub bass_zoom_strength: f32,
    pub painter_kind: PainterKind,
}

impl Default for VisualParams {
    fn default() -> Self {
        Self {
            current_shape: ShapeKind::Cylinder,
            fold_count: 12.0,
            zoom: 0.6,
            rotation_speed_scale: 1.0,
            frame_shape: FrameShape::Hexagon,
            frame_size: 0.85,
            frame_color_hue: 195.0,
            invert_enabled: false,
            colorize_enabled: false,
            colorize_hue: 0.0,
            colorize_intensity: 0.5,
            distortion_enabled: false,
            distortion_amplitude: 0.05,
            distortion_frequency: 3.0,
            shake_enabled: true,
            bass_zoom_strength: 0.3,
            painter_kind: PainterKind::HueStripe,
        }
    }
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

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Preset {
    current_shape: String,
    fold_count: f32,
    zoom: f32,
    rotation_speed_scale: f32,
    frame_shape: String,
    frame_size: f32,
    frame_color_hue: f32,
    invert_enabled: bool,
    colorize_enabled: bool,
    colorize_hue: f32,
    colorize_intensity: f32,
    distortion_enabled: bool,
    distortion_amplitude: f32,
    distortion_frequency: f32,
    shake_enabled: bool,
    bass_zoom_strength: f32,
    painter_kind: String,
}

impl Preset {
    pub fn from_params(params: &VisualParams) -> Self {
        Self {
            current_shape: params.current_shape.name().to_string(),
            fold_count: params.fold_count,
            zoom: params.zoom,
            rotation_speed_scale: params.rotation_speed_scale,
            frame_shape: format!("{:?}", params.frame_shape),
            frame_size: params.frame_size,
            frame_color_hue: params.frame_color_hue,
            invert_enabled: params.invert_enabled,
            colorize_enabled: params.colorize_enabled,
            colorize_hue: params.colorize_hue,
            colorize_intensity: params.colorize_intensity,
            distortion_enabled: params.distortion_enabled,
            distortion_amplitude: params.distortion_amplitude,
            distortion_frequency: params.distortion_frequency,
            shake_enabled: params.shake_enabled,
            bass_zoom_strength: params.bass_zoom_strength,
            painter_kind: params.painter_kind.name().to_string(),
        }
    }

    pub fn apply_to_params(&self, params: &mut VisualParams) {
        params.current_shape = match self.current_shape.as_str() {
            "Sphere"      => ShapeKind::Sphere,
            "Cube"        => ShapeKind::Cube,
            "Tetrahedron" => ShapeKind::Tetrahedron,
            _             => ShapeKind::Cylinder,
        };
        params.fold_count = self.fold_count;
        params.zoom = self.zoom;
        params.rotation_speed_scale = self.rotation_speed_scale;
        params.frame_shape = match self.frame_shape.as_str() {
            "None"    => FrameShape::None,
            "Circle"  => FrameShape::Circle,
            "Square"  => FrameShape::Square,
            "Rounded" => FrameShape::Rounded,
            "Octagon" => FrameShape::Octagon,
            "Star"    => FrameShape::Star,
            _         => FrameShape::Hexagon,
        };
        params.frame_size = self.frame_size;
        params.frame_color_hue = self.frame_color_hue;
        params.invert_enabled = self.invert_enabled;
        params.colorize_enabled = self.colorize_enabled;
        params.colorize_hue = self.colorize_hue;
        params.colorize_intensity = self.colorize_intensity;
        params.distortion_enabled = self.distortion_enabled;
        params.distortion_amplitude = self.distortion_amplitude;
        params.distortion_frequency = self.distortion_frequency;
        params.shake_enabled = self.shake_enabled;
        params.bass_zoom_strength = self.bass_zoom_strength;
        params.painter_kind = match self.painter_kind.as_str() {
            "Spiral" => PainterKind::Spiral,
            "Plasma" => PainterKind::Plasma,
            _        => PainterKind::HueStripe,
        };
    }
}

fn preset_path() -> Option<std::path::PathBuf> {
    let mut path = dirs::config_dir()?;
    path.push("abstrakt-deck");
    path.push("preset.json");
    Some(path)
}

fn save_preset(gpu: &GpuState) -> Result<(), String> {
    let path = preset_path().ok_or_else(|| "Could not find config dir".to_string())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create config dir: {}", e))?;
    }
    let preset = Preset::from_params(&gpu.params);
    let json = serde_json::to_string_pretty(&preset)
        .map_err(|e| format!("Serialize: {}", e))?;
    std::fs::write(&path, json).map_err(|e| format!("Write: {}", e))?;
    log::info!("Preset saved to {}", path.display());
    Ok(())
}

fn load_preset(gpu: &mut GpuState) -> Result<(), String> {
    let path = preset_path().ok_or_else(|| "Could not find config dir".to_string())?;
    let json = std::fs::read_to_string(&path)
        .map_err(|e| format!("Read: {}", e))?;
    let preset: Preset = serde_json::from_str(&json)
        .map_err(|e| format!("Parse: {}", e))?;
    preset.apply_to_params(&mut gpu.params);
    log::info!("Preset loaded from {}", path.display());
    Ok(())
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

    // Pass 1 — painter (procedural → fixed 4096×256)
    painter_uniforms_bind_group: wgpu::BindGroup,
    painter_pipelines: HashMap<PainterKind, wgpu::RenderPipeline>,
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
    shape_effects_buffer: wgpu::Buffer,
    shape_effects_bind_group: wgpu::BindGroup,

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
    shape_sampler: wgpu::Sampler,   // ClampToEdge sampler reused by kaleido + frame + blit passes

    // Scene FBO: frame pass renders here; blit copies to swapchain; COPY_SRC for readback
    #[allow(dead_code)]
    scene_texture: wgpu::Texture,
    scene_view: wgpu::TextureView,
    blit_bgl: wgpu::BindGroupLayout,
    blit_bind_group: wgpu::BindGroup,
    blit_pipeline: wgpu::RenderPipeline,

    // CPU readback for recording
    readback_buffer: wgpu::Buffer,
    readback_padded_bytes_per_row: u32,
    recorder: Option<Recorder>,

    help_overlay: HelpOverlay,

    start_time: Instant,

    shake_offset: glam::Vec3,
    shake_velocity: glam::Vec3,
    bass_zoom_smoothed: f32,

    pub params: VisualParams,
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

    fn align_up(value: u32, alignment: u32) -> u32 {
        value.div_ceil(alignment) * alignment
    }

    fn create_readback_buffer(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Buffer, u32) {
        let padded_bytes_per_row =
            Self::align_up(width * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let size = (padded_bytes_per_row as u64) * (height as u64);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (buffer, padded_bytes_per_row)
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

        // ── Painter texture (fixed 4096×256, 16:1 strip to match abstrakt-engine) ──
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

        // ── Shape effects uniforms ────────────────────────────────────────────
        let shape_effects_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Shape effects uniforms"),
                contents: bytemuck::cast_slice(&[ShapeEffects {
                    invert: 0.0, colorize_enabled: 0.0, colorize_hue: 0.0, colorize_intensity: 0.5,
                    distortion_enabled: 0.0, distortion_amplitude: 0.05, distortion_frequency: 3.0,
                    time_seconds: 0.0,
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let shape_effects_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shape effects BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let shape_effects_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shape effects BG"),
            layout: &shape_effects_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: shape_effects_buffer.as_entire_binding(),
            }],
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

        // ── Scene FBO + blit resources ────────────────────────────────────────
        let scene_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene FBO"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let scene_view = scene_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blit BGL"),
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
        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit BG"),
            layout: &blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&shape_sampler),
                },
            ],
        });

        let (readback_buffer, readback_padded_bytes_per_row) =
            Self::create_readback_buffer(&device, w, h);

        let help_overlay = HelpOverlay::new(&device, &queue, config.format, CHEAT_SHEET_PNG);

        // ── Pipelines ─────────────────────────────────────────────────────────
        let painter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None, bind_group_layouts: &[&uniforms_bgl], push_constant_ranges: &[],
            });
        let painter_shaders: &[(PainterKind, &str)] = &[
            (PainterKind::HueStripe, include_str!("shaders/painter_huestripe.wgsl")),
            (PainterKind::Spiral,    include_str!("shaders/painter_spiral.wgsl")),
            (PainterKind::Plasma,    include_str!("shaders/painter_plasma.wgsl")),
        ];
        let mut painter_pipelines = HashMap::new();
        for (kind, src) in painter_shaders {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(kind.name()),
                source: wgpu::ShaderSource::Wgsl((*src).into()),
            });
            let pipeline =
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(kind.name()),
                    layout: Some(&painter_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module, entry_point: Some("vs_main"),
                        buffers: &[], compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module, entry_point: Some("fs_main"),
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
            painter_pipelines.insert(*kind, pipeline);
        }

        let shape_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shape shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shape.wgsl").into()),
        });
        let shape_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Shape pipeline"),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&transform_bgl, &tex2_bgl, &shape_effects_bgl],
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

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blit.wgsl").into()),
        });
        let blit_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Blit pipeline"),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None, bind_group_layouts: &[&blit_bgl], push_constant_ranges: &[],
                })),
                vertex: wgpu::VertexState {
                    module: &blit_shader, entry_point: Some("vs_main"),
                    buffers: &[], compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blit_shader, entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
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
            painter_uniforms_bind_group, painter_pipelines,
            painter_texture, painter_view, painter_sampler,
            shape_texture, shape_view, shape_depth, shape_depth_view,
            shape_buffers,
            transform_buffer, transform_bind_group,
            shape_pipeline, shape_bind_group,
            shape_effects_buffer, shape_effects_bind_group,
            kaleido_texture, kaleido_view,
            kaleido_uniforms_buffer, kaleido_bgl, kaleido_bind_group, kaleido_pipeline,
            frame_uniforms_buffer, frame_bgl, frame_bind_group, frame_pipeline,
            shape_sampler,
            scene_texture, scene_view,
            blit_bgl, blit_bind_group, blit_pipeline,
            readback_buffer, readback_padded_bytes_per_row,
            recorder: None,
            help_overlay,
            start_time: Instant::now(),
            shake_offset: glam::Vec3::ZERO,
            shake_velocity: glam::Vec3::ZERO,
            bass_zoom_smoothed: 0.0,
            params: VisualParams::default(),
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }
        if self.recorder.is_some() {
            log::warn!("Window resized during recording — stopping");
            if let Some(rec) = self.recorder.take() {
                let _ = rec.finalize();
            }
        }
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

        // Recreate scene FBO (different size).
        let new_scene_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene FBO"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.scene_view = new_scene_tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.scene_texture = new_scene_tex;

        self.blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit BG"),
            layout: &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.shape_sampler),
                },
            ],
        });

        let (rb, rp) = Self::create_readback_buffer(&self.device, w, h);
        self.readback_buffer = rb;
        self.readback_padded_bytes_per_row = rp;
    }

    fn render(&mut self, menu: Option<(&mut MenuBar, &winit::window::Window)>) -> Result<(), wgpu::SurfaceError> {
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
                fold_count: self.params.fold_count,
                zoom: self.params.zoom * self.params.current_shape.kaleido_zoom()
                    + self.bass_zoom_smoothed * self.params.bass_zoom_strength,
            }]),
        );

        self.queue.write_buffer(
            &self.shape_effects_buffer, 0,
            bytemuck::cast_slice(&[ShapeEffects {
                invert:               if self.params.invert_enabled { 1.0 } else { 0.0 },
                colorize_enabled:     if self.params.colorize_enabled { 1.0 } else { 0.0 },
                colorize_hue:         self.params.colorize_hue,
                colorize_intensity:   self.params.colorize_intensity,
                distortion_enabled:   if self.params.distortion_enabled { 1.0 } else { 0.0 },
                distortion_amplitude: self.params.distortion_amplitude,
                distortion_frequency: self.params.distortion_frequency,
                time_seconds:         elapsed,
            }]),
        );

        let (fr, fg, fb) = hsv_to_rgb(self.params.frame_color_hue, 0.85, 1.0);
        self.queue.write_buffer(
            &self.frame_uniforms_buffer, 0,
            bytemuck::cast_slice(&[FrameUniforms {
                resolution_x:  self.size.width as f32,
                resolution_y:  self.size.height as f32,
                frame_color_r: fr,
                frame_color_g: fg,
                frame_color_b: fb,
                frame_color_a: 1.0,
                frame_shape:   self.params.frame_shape.as_f32(),
                frame_size:    self.params.frame_size,
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

        let shape = self.params.current_shape;
        let angle = elapsed
            * (std::f32::consts::TAU / shape.rotation_period_seconds())
            * self.params.rotation_speed_scale;
        let axis = glam::Vec3::from_array(shape.rotation_axis()).normalize();
        let model = glam::Mat4::from_translation(self.shake_offset)
            * glam::Mat4::from_axis_angle(axis, angle)
            * glam::Mat4::from_scale(glam::Vec3::splat(shape.model_scale()));
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

        // Pass 1: painter → painter FBO
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
            let painter_pipeline = &self.painter_pipelines[&self.params.painter_kind];
            pass.set_pipeline(painter_pipeline);
            pass.set_bind_group(0, &self.painter_uniforms_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 2: shape → shape FBO
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
            pass.set_bind_group(2, &self.shape_effects_bind_group, &[]);
            let buffers = &self.shape_buffers[&self.params.current_shape];
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

        // Pass 4: frame overlay (SDF mask) → scene FBO
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Frame pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_view, resolve_target: None,
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

        // Pass 5: blit scene FBO → swapchain
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit pass"),
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
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, &self.blit_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 6: help overlay (animated slide-in, drawn on top of swapchain)
        self.help_overlay.update_animation();
        if self.help_overlay.should_render() {
            self.help_overlay.write_uniforms(&self.queue, self.size.width, self.size.height);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Overlay pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &screen_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.help_overlay.pipeline);
            pass.set_bind_group(0, &self.help_overlay.bind_group, &[]);
            pass.draw(0..6, 0..1);
        }

        // egui menu bar pass (draws on top of the visualizer, below recording readback)
        if let Some((menu_bar, window)) = menu {
            menu_bar.render(
                &self.device,
                &self.queue,
                &mut encoder,
                window,
                &screen_view,
                self.size.width,
                self.size.height,
            );
        }

        // If recording: copy scene texture to readback buffer in the same encoder submit.
        if self.recorder.is_some() {
            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: &self.scene_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &self.readback_buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(self.readback_padded_bytes_per_row),
                        rows_per_image: Some(self.size.height),
                    },
                },
                wgpu::Extent3d {
                    width: self.size.width,
                    height: self.size.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map readback buffer and feed frame to recorder.
        if self.recorder.is_some() {
            let frame_bytes = {
                let buffer_slice = self.readback_buffer.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
                self.device.poll(wgpu::Maintain::Wait);
                rx.recv().expect("map_async channel").expect("buffer map failed");
                let mapped = buffer_slice.get_mapped_range();
                let ubpr = (self.size.width * 4) as usize;
                let pbpr = self.readback_padded_bytes_per_row as usize;
                let mut bytes = Vec::with_capacity(ubpr * self.size.height as usize);
                for row in 0..self.size.height as usize {
                    let src = row * pbpr;
                    bytes.extend_from_slice(&mapped[src..src + ubpr]);
                }
                bytes
            };
            self.readback_buffer.unmap();

            let mut frame_bytes = frame_bytes;
            if matches!(
                self.config.format,
                wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Bgra8Unorm
            ) {
                for chunk in frame_bytes.chunks_exact_mut(4) {
                    chunk.swap(0, 2);
                }
            }

            let result = self.recorder.as_mut().unwrap().submit_frame(&frame_bytes);
            if let Err(e) = result {
                log::error!("Failed to write frame to ffmpeg: {}", e);
                if let Some(rec) = self.recorder.take() {
                    let _ = rec.finalize();
                }
            }
        }

        output.present();
        Ok(())
    }

    pub fn kick_shake(&mut self, strength: f32) {
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let angle = (elapsed * 13.7) % std::f32::consts::TAU;
        let dir = glam::Vec3::new(angle.cos(), 0.0, angle.sin());
        self.shake_velocity += dir * strength * 0.75;
    }

    pub fn update_bass_zoom(&mut self, raw_bass: f32) {
        let attack = 0.4;
        let decay  = 0.08;
        if raw_bass > self.bass_zoom_smoothed {
            self.bass_zoom_smoothed = self.bass_zoom_smoothed * (1.0 - attack) + raw_bass * attack;
        } else {
            self.bass_zoom_smoothed = self.bass_zoom_smoothed * (1.0 - decay) + raw_bass * decay;
        }
    }

    pub fn toggle_recording(&mut self) {
        if self.recorder.is_some() {
            if let Some(rec) = self.recorder.take() {
                match rec.finalize() {
                    Ok(path) => log::info!("Recording saved: {}", path.display()),
                    Err(e)   => log::error!("Recording finalize failed: {}", e),
                }
            }
        } else {
            match Recorder::start(self.size.width, self.size.height) {
                Ok(rec) => {
                    self.recorder = Some(rec);
                }
                Err(e) => log::error!("Failed to start recording: {}", e),
            }
        }
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
                    gpu.params.fold_count = (2.0 + v * 22.0).round();
                    log::debug!("MIDI CC1 → fold_count = {}", gpu.params.fold_count);
                }
                5 => {
                    gpu.params.bass_zoom_strength = v;
                    log::debug!("MIDI CC5 → bass_zoom_strength = {:.2}", gpu.params.bass_zoom_strength);
                }
                7 => {
                    gpu.params.zoom = 0.3 + v * 1.2;
                    log::debug!("MIDI CC7 → zoom = {:.2}", gpu.params.zoom);
                }
                10 => {
                    gpu.params.rotation_speed_scale = v * 4.0;
                    log::debug!("MIDI CC10 → rotation_speed_scale = {:.2}", gpu.params.rotation_speed_scale);
                }
                65 if value >= 64 => {
                    gpu.params.current_shape = gpu.params.current_shape.next();
                    log::debug!("MIDI CC65 → shape cycled to {}", gpu.params.current_shape.name());
                }
                64 if value >= 64 => {
                    gpu.params.frame_shape = match gpu.params.frame_shape {
                        FrameShape::None    => FrameShape::Circle,
                        FrameShape::Circle  => FrameShape::Square,
                        FrameShape::Square  => FrameShape::Rounded,
                        FrameShape::Rounded => FrameShape::Hexagon,
                        FrameShape::Hexagon => FrameShape::Octagon,
                        FrameShape::Octagon => FrameShape::Star,
                        FrameShape::Star    => FrameShape::None,
                    };
                    log::debug!("MIDI CC64 → frame_shape cycled to {:?}", gpu.params.frame_shape);
                }
                71 => {
                    gpu.params.frame_size = 0.4 + v * 0.6;
                    log::debug!("MIDI CC71 → frame_size = {:.2}", gpu.params.frame_size);
                }
                74 => {
                    gpu.params.frame_color_hue = v * 360.0;
                    log::debug!("MIDI CC74 → frame_color_hue = {:.0}°", gpu.params.frame_color_hue);
                }
                76 => {
                    gpu.params.colorize_hue = v * 360.0;
                    log::debug!("MIDI CC76 → colorize_hue = {:.0}°", gpu.params.colorize_hue);
                }
                91 => {
                    gpu.params.invert_enabled = value >= 64;
                    log::debug!("MIDI CC91 → invert = {}", gpu.params.invert_enabled);
                }
                92 => {
                    gpu.params.colorize_intensity = v;
                    log::debug!("MIDI CC92 → colorize_intensity = {:.2}", gpu.params.colorize_intensity);
                }
                93 => {
                    gpu.params.colorize_enabled = value >= 64;
                    log::debug!("MIDI CC93 → colorize = {}", gpu.params.colorize_enabled);
                }
                80 => {
                    gpu.params.distortion_enabled = value >= 64;
                    log::debug!("MIDI CC80 → distortion = {}", gpu.params.distortion_enabled);
                }
                81 => {
                    gpu.params.distortion_amplitude = v * 0.5;
                    log::debug!("MIDI CC81 → distortion_amplitude = {:.3}", gpu.params.distortion_amplitude);
                }
                82 => {
                    gpu.params.distortion_frequency = 0.5 + v * 7.5;
                    log::debug!("MIDI CC82 → distortion_frequency = {:.1}", gpu.params.distortion_frequency);
                }
                66 if value >= 64 => {
                    gpu.params.painter_kind = gpu.params.painter_kind.next();
                    log::debug!("MIDI CC66 → painter: {}", gpu.params.painter_kind.name());
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
    modifiers: winit::keyboard::ModifiersState,
    is_fullscreen: bool,
    menu_bar: Option<MenuBar>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            fps: FpsCounter::new(),
            audio: None,
            midi: None,
            modifiers: winit::keyboard::ModifiersState::empty(),
            is_fullscreen: false,
            menu_bar: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        let attrs = Window::default_attributes()
            .with_title("abstrakt-deck")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        let window = Arc::new(
            event_loop.create_window(attrs).expect("Failed to create window"),
        );
        let gpu = pollster::block_on(GpuState::new(window.clone()));
        let menu_bar = MenuBar::new(&gpu.device, gpu.config.format, &window);
        self.window = Some(window);
        self.gpu = Some(gpu);
        self.menu_bar = Some(menu_bar);
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
        // Forward events to egui before application handling.
        let egui_consumed = if let (Some(menu), Some(window)) =
            (self.menu_bar.as_mut(), self.window.as_ref())
        {
            menu.handle_event(window, &event).consumed
        } else {
            false
        };
        // Let Resized and RedrawRequested through regardless.
        match &event {
            WindowEvent::RedrawRequested | WindowEvent::Resized(_) => {}
            _ if egui_consumed => return,
            _ => {}
        }

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
                let ctrl = self.modifiers.control_key();
                if ctrl {
                    match key_code {
                        KeyCode::KeyS => {
                            if let Err(e) = save_preset(gpu) {
                                log::error!("Save failed: {}", e);
                            }
                            return;
                        }
                        KeyCode::KeyL => {
                            if let Err(e) = load_preset(gpu) {
                                log::warn!("Load failed: {} (no preset saved yet?)", e);
                            }
                            return;
                        }
                        _ => {}
                    }
                }
                match key_code {
                    KeyCode::Escape => {
                        log::info!("Escape pressed");
                        event_loop.exit();
                    }
                    KeyCode::BracketLeft => {
                        gpu.params.fold_count = (gpu.params.fold_count - 1.0).max(2.0);
                        log::info!("fold_count = {}", gpu.params.fold_count);
                    }
                    KeyCode::BracketRight => {
                        gpu.params.fold_count = (gpu.params.fold_count + 1.0).min(24.0);
                        log::info!("fold_count = {}", gpu.params.fold_count);
                    }
                    KeyCode::KeyZ => {
                        gpu.params.zoom = (gpu.params.zoom - 0.05).max(0.3);
                        log::info!("zoom = {:.2}", gpu.params.zoom);
                    }
                    KeyCode::KeyX => {
                        gpu.params.zoom = (gpu.params.zoom + 0.05).min(1.5);
                        log::info!("zoom = {:.2}", gpu.params.zoom);
                    }
                    KeyCode::Comma => {
                        gpu.params.rotation_speed_scale = (gpu.params.rotation_speed_scale - 0.25).max(0.0);
                        log::info!("rotation_speed_scale = {:.2}", gpu.params.rotation_speed_scale);
                    }
                    KeyCode::Period => {
                        gpu.params.rotation_speed_scale = (gpu.params.rotation_speed_scale + 0.25).min(4.0);
                        log::info!("rotation_speed_scale = {:.2}", gpu.params.rotation_speed_scale);
                    }
                    KeyCode::Digit1 => { gpu.params.frame_shape = FrameShape::None;    log::info!("frame: None"); }
                    KeyCode::Digit2 => { gpu.params.frame_shape = FrameShape::Circle;  log::info!("frame: Circle"); }
                    KeyCode::Digit3 => { gpu.params.frame_shape = FrameShape::Square;  log::info!("frame: Square"); }
                    KeyCode::Digit4 => { gpu.params.frame_shape = FrameShape::Rounded; log::info!("frame: Rounded"); }
                    KeyCode::Digit5 => { gpu.params.frame_shape = FrameShape::Hexagon; log::info!("frame: Hexagon"); }
                    KeyCode::Digit6 => { gpu.params.frame_shape = FrameShape::Octagon; log::info!("frame: Octagon"); }
                    KeyCode::Digit7 => { gpu.params.frame_shape = FrameShape::Star;    log::info!("frame: Star"); }
                    KeyCode::Minus => {
                        gpu.params.frame_size = (gpu.params.frame_size - 0.05).max(0.4);
                        log::info!("frame_size = {:.2}", gpu.params.frame_size);
                    }
                    KeyCode::Equal => {
                        gpu.params.frame_size = (gpu.params.frame_size + 0.05).min(1.0);
                        log::info!("frame_size = {:.2}", gpu.params.frame_size);
                    }
                    KeyCode::KeyR => {
                        gpu.params.frame_color_hue = (gpu.params.frame_color_hue + 30.0) % 360.0;
                        log::info!("frame_color_hue = {:.0}°", gpu.params.frame_color_hue);
                    }
                    KeyCode::KeyG => {
                        gpu.params.frame_color_hue = (gpu.params.frame_color_hue + 120.0) % 360.0;
                        log::info!("frame_color_hue = {:.0}°", gpu.params.frame_color_hue);
                    }
                    KeyCode::KeyB => {
                        gpu.params.frame_color_hue = (gpu.params.frame_color_hue + 60.0) % 360.0;
                        log::info!("frame_color_hue = {:.0}°", gpu.params.frame_color_hue);
                    }
                    KeyCode::Space => {
                        gpu.params.shake_enabled = !gpu.params.shake_enabled;
                        log::info!("shake_enabled = {}", gpu.params.shake_enabled);
                    }
                    KeyCode::Tab => {
                        gpu.params.current_shape = gpu.params.current_shape.next();
                        log::info!("shape: {}", gpu.params.current_shape.name());
                    }
                    KeyCode::KeyI => {
                        gpu.params.invert_enabled = !gpu.params.invert_enabled;
                        log::info!("invert = {}", gpu.params.invert_enabled);
                    }
                    KeyCode::KeyT => {
                        gpu.params.colorize_enabled = !gpu.params.colorize_enabled;
                        log::info!("colorize = {}", gpu.params.colorize_enabled);
                    }
                    KeyCode::Semicolon => {
                        gpu.params.colorize_hue = (gpu.params.colorize_hue + 30.0) % 360.0;
                        log::info!("colorize_hue = {:.0}°", gpu.params.colorize_hue);
                    }
                    KeyCode::Digit9 => {
                        gpu.params.colorize_intensity = (gpu.params.colorize_intensity - 0.1).max(0.0);
                        log::info!("colorize_intensity = {:.2}", gpu.params.colorize_intensity);
                    }
                    KeyCode::Digit0 => {
                        gpu.params.colorize_intensity = (gpu.params.colorize_intensity + 0.1).min(1.0);
                        log::info!("colorize_intensity = {:.2}", gpu.params.colorize_intensity);
                    }
                    KeyCode::Slash => {
                        if self.modifiers.shift_key() {
                            gpu.help_overlay.toggle();
                        } else {
                            gpu.params.bass_zoom_strength =
                                (gpu.params.bass_zoom_strength - 0.05).max(0.0);
                            log::info!("bass_zoom_strength = {:.2}", gpu.params.bass_zoom_strength);
                        }
                    }
                    KeyCode::Quote => {
                        gpu.params.bass_zoom_strength = (gpu.params.bass_zoom_strength + 0.05).min(1.0);
                        log::info!("bass_zoom_strength = {:.2}", gpu.params.bass_zoom_strength);
                    }
                    KeyCode::KeyD => {
                        gpu.params.distortion_enabled = !gpu.params.distortion_enabled;
                        log::info!("distortion = {}", gpu.params.distortion_enabled);
                    }
                    KeyCode::KeyQ => {
                        gpu.params.distortion_amplitude = (gpu.params.distortion_amplitude - 0.01).max(0.0);
                        log::info!("distortion_amplitude = {:.3}", gpu.params.distortion_amplitude);
                    }
                    KeyCode::KeyW => {
                        gpu.params.distortion_amplitude = (gpu.params.distortion_amplitude + 0.01).min(0.5);
                        log::info!("distortion_amplitude = {:.3}", gpu.params.distortion_amplitude);
                    }
                    KeyCode::KeyE => {
                        gpu.params.distortion_frequency = (gpu.params.distortion_frequency - 0.5).max(0.5);
                        log::info!("distortion_frequency = {:.1}", gpu.params.distortion_frequency);
                    }
                    KeyCode::KeyF => {
                        gpu.params.distortion_frequency = (gpu.params.distortion_frequency + 0.5).min(8.0);
                        log::info!("distortion_frequency = {:.1}", gpu.params.distortion_frequency);
                    }
                    KeyCode::KeyP => {
                        gpu.params.painter_kind = gpu.params.painter_kind.next();
                        log::info!("painter: {}", gpu.params.painter_kind.name());
                    }
                    KeyCode::F12 => {
                        gpu.toggle_recording();
                    }
                    KeyCode::F11 => {
                        self.is_fullscreen = !self.is_fullscreen;
                        window.set_fullscreen(if self.is_fullscreen {
                            Some(winit::window::Fullscreen::Borderless(None))
                        } else {
                            None
                        });
                        log::info!("Fullscreen: {}", self.is_fullscreen);
                    }
                    _ => {}
                }
            }
            WindowEvent::ModifiersChanged(new_mods) => {
                self.modifiers = new_mods.state();
            }
            WindowEvent::Resized(physical_size) => {
                gpu.resize(physical_size);
            }
            WindowEvent::RedrawRequested => {
                if let Some(audio) = &self.audio {
                    while let Ok(event) = audio.event_rx.try_recv() {
                        match event {
                            AudioEvent::Beat(strength) => {
                                if gpu.params.shake_enabled {
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
                let bass_energy = self.audio.as_ref()
                    .map(|a| a.state.lock().bass_energy)
                    .unwrap_or(0.0);
                gpu.update_bass_zoom(bass_energy);
                let menu = self.menu_bar.as_mut()
                    .map(|m| (m, window.as_ref()));
                match gpu.render(menu) {
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
                if self.menu_bar.as_ref().map_or(false, |m| m.quit_requested) {
                    log::info!("Quit via menu");
                    event_loop.exit();
                    return;
                }
                if let Some(fps) = self.fps.tick() {
                    let title = if let Some(rec) = gpu.recorder.as_ref() {
                        let secs = rec.elapsed().as_secs();
                        format!(
                            "abstrakt-deck — slice 24a — ● REC {}:{:02} — {:.1} fps",
                            secs / 60, secs % 60, fps
                        )
                    } else {
                        format!("abstrakt-deck — slice 24a — {:.1} fps", fps)
                    };
                    window.set_title(&title);
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
    println!("  / '   bass-zoom intensity (0 to 1)");
    println!("  I      toggle color invert");
    println!("  T      toggle colorize tint");
    println!("  ;      cycle colorize hue (+30°)");
    println!("  9 0    colorize intensity (0 to 1)");
    println!("  D      toggle distortion");
    println!("  Q W    distortion amplitude (0 to 0.5)");
    println!("  E F    distortion frequency (0.5 to 8)");
    println!("  P      cycle painter (HueStripe → Spiral → Plasma)");
    println!("  ?      toggle help overlay");
    println!("  F11    toggle fullscreen");
    println!("  F12    toggle video recording (saves to ~/Videos/abstrakt-deck/)");
    println!("  Ctrl+S save preset to ~/.config/abstrakt-deck/preset.json");
    println!("  Ctrl+L load preset from same file");
    println!("  esc    exit");
    println!("\nMIDI control (if device connected):");
    println!("  CC 1   fold count");
    println!("  CC 7   zoom");
    println!("  CC 10  rotation speed");
    println!("  CC 64  cycle frame shape");
    println!("  CC 5   bass-zoom intensity");
    println!("  CC 76  colorize hue");
    println!("  CC 91  invert toggle");
    println!("  CC 92  colorize intensity");
    println!("  CC 93  colorize toggle");
    println!("  CC 65  cycle shape (portamento)");
    println!("  CC 71  frame size");
    println!("  CC 74  frame color hue");
    println!("  CC 80  distortion toggle");
    println!("  CC 81  distortion amplitude");
    println!("  CC 82  distortion frequency");
    println!("  CC 66  cycle painter (HueStripe → Spiral → Plasma)");
    println!("  Note On  trigger shake (like a beat)\n");

    log::info!("abstrakt-deck starting");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop failed");

    log::info!("abstrakt-deck shutting down");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn non_default_params() -> VisualParams {
        VisualParams {
            current_shape: ShapeKind::Tetrahedron,
            fold_count: 18.0,
            zoom: 1.2,
            rotation_speed_scale: 2.5,
            frame_shape: FrameShape::Star,
            frame_size: 0.7,
            frame_color_hue: 90.0,
            invert_enabled: true,
            colorize_enabled: true,
            colorize_hue: 270.0,
            colorize_intensity: 0.75,
            distortion_enabled: true,
            distortion_amplitude: 0.3,
            distortion_frequency: 6.0,
            shake_enabled: false,
            bass_zoom_strength: 0.8,
            painter_kind: PainterKind::Plasma,
        }
    }

    #[test]
    fn preset_roundtrip_preserves_all_fields() {
        let original = non_default_params();
        let preset = Preset::from_params(&original);

        let json = serde_json::to_string(&preset).expect("serialize");
        let parsed: Preset = serde_json::from_str(&json).expect("parse");

        let mut restored = VisualParams::default();
        parsed.apply_to_params(&mut restored);

        assert_eq!(restored.current_shape, original.current_shape, "current_shape failed");
        assert_eq!(restored.fold_count, original.fold_count, "fold_count failed");
        assert_eq!(restored.zoom, original.zoom, "zoom failed");
        assert_eq!(restored.rotation_speed_scale, original.rotation_speed_scale, "rotation_speed_scale failed");
        assert_eq!(restored.frame_shape, original.frame_shape, "frame_shape failed");
        assert_eq!(restored.frame_size, original.frame_size, "frame_size failed");
        assert_eq!(restored.frame_color_hue, original.frame_color_hue, "frame_color_hue failed");
        assert_eq!(restored.invert_enabled, original.invert_enabled, "invert_enabled failed");
        assert_eq!(restored.colorize_enabled, original.colorize_enabled, "colorize_enabled failed");
        assert_eq!(restored.colorize_hue, original.colorize_hue, "colorize_hue failed");
        assert_eq!(restored.colorize_intensity, original.colorize_intensity, "colorize_intensity failed");
        assert_eq!(restored.distortion_enabled, original.distortion_enabled, "distortion_enabled failed");
        assert_eq!(restored.distortion_amplitude, original.distortion_amplitude, "distortion_amplitude failed");
        assert_eq!(restored.distortion_frequency, original.distortion_frequency, "distortion_frequency failed");
        assert_eq!(restored.shake_enabled, original.shake_enabled, "shake_enabled failed");
        assert_eq!(restored.bass_zoom_strength, original.bass_zoom_strength, "bass_zoom_strength failed");
        assert_eq!(restored.painter_kind, original.painter_kind, "painter_kind failed");
    }

    #[test]
    fn shape_kind_name_roundtrip() {
        for kind in [ShapeKind::Cylinder, ShapeKind::Sphere, ShapeKind::Cube, ShapeKind::Tetrahedron] {
            let name = kind.name();
            let parsed = match name {
                "Sphere"      => ShapeKind::Sphere,
                "Cube"        => ShapeKind::Cube,
                "Tetrahedron" => ShapeKind::Tetrahedron,
                _             => ShapeKind::Cylinder,
            };
            assert_eq!(parsed, kind, "ShapeKind {:?} did not round-trip via name()", kind);
        }
    }

    #[test]
    fn frame_shape_debug_roundtrip() {
        for shape in [
            FrameShape::None, FrameShape::Circle, FrameShape::Square,
            FrameShape::Rounded, FrameShape::Hexagon, FrameShape::Octagon, FrameShape::Star,
        ] {
            let debug_str = format!("{:?}", shape);
            let parsed = match debug_str.as_str() {
                "None"    => FrameShape::None,
                "Circle"  => FrameShape::Circle,
                "Square"  => FrameShape::Square,
                "Rounded" => FrameShape::Rounded,
                "Octagon" => FrameShape::Octagon,
                "Star"    => FrameShape::Star,
                _         => FrameShape::Hexagon,
            };
            assert_eq!(parsed, shape, "FrameShape {:?} did not round-trip via Debug format", shape);
        }
    }

    #[test]
    fn rotation_axes_are_normalized() {
        for shape in [ShapeKind::Cylinder, ShapeKind::Sphere, ShapeKind::Cube, ShapeKind::Tetrahedron] {
            let [x, y, z] = shape.rotation_axis();
            let length_sq = x * x + y * y + z * z;
            assert!(
                (length_sq - 1.0).abs() < 1e-5,
                "{:?} rotation axis is not unit length: length²={}", shape, length_sq
            );
        }
    }

    #[test]
    fn painter_kind_name_roundtrip() {
        for kind in [PainterKind::HueStripe, PainterKind::Spiral, PainterKind::Plasma] {
            let name = kind.name();
            let parsed = match name {
                "Spiral" => PainterKind::Spiral,
                "Plasma" => PainterKind::Plasma,
                _        => PainterKind::HueStripe,
            };
            assert_eq!(parsed, kind, "PainterKind {:?} did not round-trip via name()", kind);
        }
    }
}

mod audio;
mod help_overlay;
mod menu_bar;
mod midi;
mod recorder;
mod shape;
use audio::{AudioCapture, AudioEvent};
use help_overlay::HelpOverlay;
use menu_bar::{MenuAction, MenuBar, ParamChange};
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

const PAINTER_TEXTURE_WIDTH: u32  = 4096;
const PAINTER_TEXTURE_HEIGHT: u32 = 256;
const SKIN_MIP_LEVELS: u32        = 13; // ilog2(4096) + 1

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
    // Scroll phase for the 4096×256 painter window (0..1); keeps struct at 16-byte alignment.
    painter_scroll_phase: f32,
    contrast:             f32,
    saturation:           f32,
    _pad_s2:              f32,
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
    Skin,
}

impl PainterKind {
    pub fn next(self) -> Self {
        match self {
            PainterKind::HueStripe => PainterKind::Spiral,
            PainterKind::Spiral    => PainterKind::Plasma,
            PainterKind::Plasma    => PainterKind::Skin,
            PainterKind::Skin      => PainterKind::HueStripe,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            PainterKind::HueStripe => "HueStripe",
            PainterKind::Spiral    => "Spiral",
            PainterKind::Plasma    => "Plasma",
            PainterKind::Skin      => "Skin",
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
    pub contrast: f32,
    pub saturation: f32,
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
            contrast: 1.0,
            saturation: 1.0,
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
    #[serde(default = "default_one_f32")]
    contrast: f32,
    #[serde(default = "default_one_f32")]
    saturation: f32,
}

fn default_one_f32() -> f32 { 1.0 }

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
            contrast: params.contrast,
            saturation: params.saturation,
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
            "Skin"   => PainterKind::Skin,
            _        => PainterKind::HueStripe,
        };
        params.contrast   = self.contrast;
        params.saturation = self.saturation;
    }
}

fn generate_skin_mipmaps(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip_count: u32,
) {
    const BLIT_SRC: &str = r#"
struct Vary { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> Vary {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: Vary;
    out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv  = vec2<f32>(x, 1.0 - y);
    return out;
}

@group(0) @binding(0) var src:         texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

@fragment
fn fs_main(in: Vary) -> @location(0) vec4<f32> {
    return textureSample(src, src_sampler, in.uv);
}
"#;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Mip blit shader"),
        source: wgpu::ShaderSource::Wgsl(BLIT_SRC.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Mip BGL"),
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
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Mip pipeline"),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        })),
        vertex: wgpu::VertexState {
            module: &shader, entry_point: Some("vs_main"),
            buffers: &[], compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader, entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
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
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Mip source sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let views: Vec<wgpu::TextureView> = (0..mip_count).map(|i| {
        texture.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: i,
            mip_level_count: Some(1),
            ..Default::default()
        })
    }).collect();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Mip encoder"),
    });
    for i in 1..mip_count {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mip BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&views[(i - 1) as usize]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Mip pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &views[i as usize],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    queue.submit(std::iter::once(encoder.finish()));
}

fn decode_and_validate_skin(path: &std::path::Path) -> Result<image::DynamicImage, String> {
    image::open(path).map_err(|e| e.to_string())
}

fn crop_skin_image(img: &image::DynamicImage, vertical_offset: f32) -> Vec<u8> {
    use image::GenericImageView;
    let (src_w, src_h) = img.dimensions();
    let (crop_w, crop_h, crop_x, crop_y) = if src_w as f32 / src_h as f32 > 16.0 {
        // Image wider than 16:1 — crop width to center, vertical_offset unused
        let cw = (src_h as f32 * 16.0) as u32;
        (cw, src_h, (src_w - cw) / 2, 0u32)
    } else {
        // Image narrower than 16:1 — crop a thin horizontal strip at vertical_offset
        let ch = ((src_w as f32 / 16.0).round() as u32).max(1);
        let max_y = src_h.saturating_sub(ch);
        let cy = (vertical_offset.clamp(0.0, 1.0) * max_y as f32) as u32;
        (src_w, ch, 0u32, cy)
    };
    let cropped = img.crop_imm(crop_x, crop_y, crop_w, crop_h);
    let rgba = cropped.to_rgba8();
    image::imageops::resize(
        &rgba,
        PAINTER_TEXTURE_WIDTH,
        PAINTER_TEXTURE_HEIGHT,
        image::imageops::FilterType::Lanczos3,
    ).into_raw()
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

    // Skin painter resources
    skin_tex: wgpu::Texture,
    #[allow(dead_code)]
    skin_view: wgpu::TextureView,
    skin_bgl: wgpu::BindGroupLayout,
    skin_bind_group: wgpu::BindGroup,
    skin_sampler: wgpu::Sampler,
    painter_skin_pipeline: wgpu::RenderPipeline,

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
    painter_scroll_phase: f32,

    pub skin_source_image: Option<image::DynamicImage>,
    pub crop_y_offset: f32,

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
                    painter_scroll_phase: 0.0,
                    contrast: 1.0, saturation: 1.0, _pad_s2: 0.0,
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

        // ── Skin painter resources ────────────────────────────────────────────
        let skin_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Skin BGL"),
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
        let skin_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Skin texture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &skin_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &[0u8, 0, 0, 255],
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let skin_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Skin sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter:    wgpu::FilterMode::Linear,
            min_filter:    wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let skin_view = skin_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let skin_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Skin BG"),
            layout: &skin_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&skin_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&skin_sampler),
                },
            ],
        });
        let skin_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Skin painter shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/painter_skin.wgsl").into()),
        });
        let painter_skin_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Skin painter pipeline"),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&skin_bgl],
                    push_constant_ranges: &[],
                })),
                vertex: wgpu::VertexState {
                    module: &skin_shader, entry_point: Some("vs_main"),
                    buffers: &[], compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &skin_shader, entry_point: Some("fs_main"),
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
            skin_tex, skin_view, skin_bgl, skin_bind_group, skin_sampler, painter_skin_pipeline,
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
            painter_scroll_phase: 0.0,
            skin_source_image: None,
            crop_y_offset: 0.5,
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

        // Rotation-driven painter scroll: shape samples a 0.25-wide window that slides
        // across the 4096-wide painter strip as the shape rotates.
        {
            let shape = self.params.current_shape;
            let angular_velocity = std::f32::consts::TAU / shape.rotation_period_seconds();
            let rotation_radians = elapsed * angular_velocity * self.params.rotation_speed_scale;
            // 1 revolution advances the window by 0.25 (shape circumference / painter width).
            // 4 revolutions = 1 full cycle through the painter.
            self.painter_scroll_phase =
                (rotation_radians / std::f32::consts::TAU * 0.25).fract();
        }

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
                painter_scroll_phase: self.painter_scroll_phase,
                contrast:   self.params.contrast,
                saturation: self.params.saturation,
                _pad_s2: 0.0,
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
            if self.params.painter_kind == PainterKind::Skin {
                pass.set_pipeline(&self.painter_skin_pipeline);
                pass.set_bind_group(0, &self.skin_bind_group, &[]);
            } else {
                let painter_pipeline = &self.painter_pipelines[&self.params.painter_kind];
                pass.set_pipeline(painter_pipeline);
                pass.set_bind_group(0, &self.painter_uniforms_bind_group, &[]);
            }
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

        // egui menu bar + params panel pass
        if let Some((menu_bar, window)) = menu {
            menu_bar.render(
                &self.device,
                &self.queue,
                &mut encoder,
                window,
                &screen_view,
                self.size.width,
                self.size.height,
                &self.params,
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

    pub fn load_skin(&mut self, rgba: Vec<u8>) {
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Skin texture"),
            size: wgpu::Extent3d {
                width: PAINTER_TEXTURE_WIDTH,
                height: PAINTER_TEXTURE_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: SKIN_MIP_LEVELS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::COPY_DST
                 | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(PAINTER_TEXTURE_WIDTH * 4),
                rows_per_image: Some(PAINTER_TEXTURE_HEIGHT),
            },
            wgpu::Extent3d {
                width: PAINTER_TEXTURE_WIDTH,
                height: PAINTER_TEXTURE_HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        generate_skin_mipmaps(&self.device, &self.queue, &tex, SKIN_MIP_LEVELS);
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let new_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Skin BG"),
            layout: &self.skin_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.skin_sampler),
                },
            ],
        });
        self.skin_tex = tex;
        self.skin_view = view;
        self.skin_bind_group = new_bg;
    }

    pub fn upload_skin_bytes(&mut self, rgba: &[u8]) {
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.skin_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(PAINTER_TEXTURE_WIDTH * 4),
                rows_per_image: Some(PAINTER_TEXTURE_HEIGHT),
            },
            wgpu::Extent3d {
                width: PAINTER_TEXTURE_WIDTH,
                height: PAINTER_TEXTURE_HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        generate_skin_mipmaps(&self.device, &self.queue, &self.skin_tex, SKIN_MIP_LEVELS);
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
            menu.handle_event(window, &event)
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
                    KeyCode::Tab if self.modifiers.shift_key() => {
                        gpu.params.current_shape = gpu.params.current_shape.next();
                        log::info!("shape: {}", gpu.params.current_shape.name());
                    }
                    KeyCode::KeyM if !ctrl => {
                        if let Some(menu) = self.menu_bar.as_mut() {
                            menu.toggle_params_panel();
                            log::info!("params panel: {}", menu.params_panel_visible);
                        }
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
                // Drain and dispatch menu actions accumulated during this frame's render.
                let actions = self.menu_bar.as_mut()
                    .map_or_else(Vec::new, |m| m.take_actions());
                for action in actions {
                    match action {
                        MenuAction::Quit => {
                            log::info!("Quit via menu");
                            event_loop.exit();
                            return;
                        }
                        MenuAction::OpenSkin => {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Images", &["jpg", "jpeg", "png", "webp"])
                                .set_title("Select skin image")
                                .pick_file()
                            {
                                log::info!("Selected file: {}", path.display());
                                match decode_and_validate_skin(&path) {
                                    Ok(img) => {
                                        let rgba = crop_skin_image(&img, 0.5);
                                        gpu.load_skin(rgba);
                                        gpu.params.painter_kind = PainterKind::Skin;
                                        gpu.crop_y_offset = 0.5;
                                        if let Some(menu) = self.menu_bar.as_mut() {
                                            menu.set_skin_thumbnail(&img);
                                            menu.current_crop_y_offset = 0.5;
                                        }
                                        gpu.skin_source_image = Some(img);
                                        log::info!("Skin loaded — adjust crop in the Skin panel");
                                    }
                                    Err(e) => log::error!("Failed to load skin: {}", e),
                                }
                            } else {
                                log::info!("Open Skin canceled");
                            }
                        }
                        MenuAction::SavePreset => {
                            if let Err(e) = save_preset(gpu) {
                                log::error!("Save preset failed: {}", e);
                            }
                        }
                        MenuAction::LoadPreset => {
                            if let Err(e) = load_preset(gpu) {
                                log::warn!("Load preset failed: {}", e);
                            }
                        }
                        MenuAction::ToggleFullscreen => {
                            self.is_fullscreen = !self.is_fullscreen;
                            window.set_fullscreen(if self.is_fullscreen {
                                Some(winit::window::Fullscreen::Borderless(None))
                            } else {
                                None
                            });
                            log::info!("Fullscreen: {}", self.is_fullscreen);
                        }
                        MenuAction::ToggleCheatSheet => {
                            gpu.help_overlay.toggle();
                        }
                        MenuAction::ToggleRecording => {
                            gpu.toggle_recording();
                        }
                        MenuAction::TogglePanels => {
                            if let Some(menu) = self.menu_bar.as_mut() {
                                menu.toggle_params_panel();
                            }
                        }
                    }
                }

                // Apply parameter changes collected from the panel widgets.
                let changes = self.menu_bar.as_mut()
                    .map_or_else(Vec::new, |m| m.take_param_changes());
                for change in changes {
                    match change {
                        ParamChange::FoldCount(v)            => gpu.params.fold_count = v,
                        ParamChange::Zoom(v)                 => gpu.params.zoom = v,
                        ParamChange::RotationSpeedScale(v)   => gpu.params.rotation_speed_scale = v,
                        ParamChange::FrameSize(v)            => gpu.params.frame_size = v,
                        ParamChange::FrameColorHue(v)        => gpu.params.frame_color_hue = v,
                        ParamChange::InvertEnabled(v)        => gpu.params.invert_enabled = v,
                        ParamChange::ColorizeEnabled(v)      => gpu.params.colorize_enabled = v,
                        ParamChange::ColorizeHue(v)          => gpu.params.colorize_hue = v,
                        ParamChange::ColorizeIntensity(v)    => gpu.params.colorize_intensity = v,
                        ParamChange::DistortionEnabled(v)    => gpu.params.distortion_enabled = v,
                        ParamChange::DistortionAmplitude(v)  => gpu.params.distortion_amplitude = v,
                        ParamChange::DistortionFrequency(v)  => gpu.params.distortion_frequency = v,
                        ParamChange::ShakeEnabled(v)         => gpu.params.shake_enabled = v,
                        ParamChange::BassZoomStrength(v)     => gpu.params.bass_zoom_strength = v,
                        ParamChange::CurrentShape(v)         => gpu.params.current_shape = v,
                        ParamChange::FrameShape(v)           => gpu.params.frame_shape = v,
                        ParamChange::PainterKind(v)          => gpu.params.painter_kind = v,
                        ParamChange::SkinCropOffset(v) => {
                            gpu.crop_y_offset = v;
                            let rgba = gpu.skin_source_image.as_ref()
                                .map(|src| crop_skin_image(src, v));
                            if let Some(rgba) = rgba {
                                gpu.upload_skin_bytes(&rgba);
                            }
                        }
                        ParamChange::Contrast(v)   => gpu.params.contrast   = v,
                        ParamChange::Saturation(v) => gpu.params.saturation = v,
                    }
                }

                if let Some(fps) = self.fps.tick() {
                    let title = if let Some(rec) = gpu.recorder.as_ref() {
                        let secs = rec.elapsed().as_secs();
                        format!(
                            "abstrakt-deck — slice 23c.5 — ● REC {}:{:02} — {:.1} fps",
                            secs / 60, secs % 60, fps
                        )
                    } else {
                        format!("abstrakt-deck — slice 23c.5 — {:.1} fps", fps)
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
    println!("  Shift+Tab  cycle shape (Cylinder → Sphere → Cube → Tetrahedron)");
    println!("  / '   bass-zoom intensity (0 to 1)");
    println!("  I      toggle color invert");
    println!("  T      toggle colorize tint");
    println!("  ;      cycle colorize hue (+30°)");
    println!("  9 0    colorize intensity (0 to 1)");
    println!("  D      toggle distortion");
    println!("  Q W    distortion amplitude (0 to 0.5)");
    println!("  E F    distortion frequency (0.5 to 8)");
    println!("  P      cycle painter (HueStripe → Spiral → Plasma → Skin)");
    println!("  M      toggle parameters panel");
    println!("  ?      toggle help overlay");
    println!("  F11    toggle fullscreen");
    println!("  F12    toggle video recording (saves to ~/Videos/abstrakt-deck/)");
    println!("  Ctrl+S save preset to ~/.config/abstrakt-deck/preset.json");
    println!("  Ctrl+L load preset from same file");
    println!("  esc    exit");
    println!("  (Skin crop: use the Skin section in the parameters panel — M to toggle panel)");
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
    println!("  CC 66  cycle painter (HueStripe → Spiral → Plasma → Skin)");
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
            contrast: 1.5,
            saturation: 0.7,
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
        assert_eq!(restored.contrast,   original.contrast,   "contrast failed");
        assert_eq!(restored.saturation, original.saturation, "saturation failed");
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
        for kind in [PainterKind::HueStripe, PainterKind::Spiral, PainterKind::Plasma, PainterKind::Skin] {
            let name = kind.name();
            let parsed = match name {
                "Spiral" => PainterKind::Spiral,
                "Plasma" => PainterKind::Plasma,
                "Skin"   => PainterKind::Skin,
                _        => PainterKind::HueStripe,
            };
            assert_eq!(parsed, kind, "PainterKind {:?} did not round-trip via name()", kind);
        }
    }
}

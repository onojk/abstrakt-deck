mod audio;
mod color;
mod help_overlay;
mod menu_bar;
mod midi;
mod phantom;
mod recorder;
mod shape;
use audio::{AudioCapture, AudioEvent};
use help_overlay::HelpOverlay;
use menu_bar::{ExportProgress, MenuAction, MenuBar, ParamChange};
use midi::{MidiCapture, MidiEvent};
use recorder::Recorder;
use shape::{ShapeKind, Vertex};

static CHEAT_SHEET_PNG: &[u8] = include_bytes!("../assets/cheat_sheet.png");
use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
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
const PAINTER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

const RIBBON_COLOR: [f32; 4] = [0.9, 0.9, 0.9, 1.0];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FeedbackUniforms {
    center_x:     f32,
    center_y:     f32,
    shrink_rate:  f32,
    strength:     f32,
    alpha_radius: f32,
    _pad0:        f32,
    _pad1:        f32,
    _pad2:        f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChromaUniforms {
    pub key_color_r:   f32,  // offset  0
    pub key_color_g:   f32,  // offset  4
    pub key_color_b:   f32,  // offset  8
    pub key_tolerance: f32,  // offset 12
    pub key_softness:  f32,  // offset 16
    pub key_strength:  f32,  // offset 20
    pub opacity:       f32,  // offset 24
    pub _pad:          f32,  // offset 28 — total 32 bytes
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PaletteUniforms {
    mode:                u32,      // offset  0
    tint:                f32,      // offset  4
    mono_hue:            f32,      // offset  8
    harmony_num_offsets: u32,      // offset 12
    harmony_anchor_hue:  f32,      // offset 16
    harmony_saturation:  f32,      // offset 20
    harmony_value:       f32,      // offset 24
    harmony_strength:    f32,      // offset 28
    harmony_offsets:     [f32; 8], // offset 32 — total 64 bytes
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AppliedHarmonyUniforms {
    // vec4 0
    enabled:      u32,    // offset  0
    anchor_hue:   f32,    // offset  4
    saturation:   f32,    // offset  8
    value:        f32,    // offset 12
    // vec4 1
    strength:     f32,    // offset 16
    offset_count: u32,    // offset 20
    _pad0:        f32,    // offset 24
    _pad1:        f32,    // offset 28
    // vec4 2-3: hue offsets packed as [f32;8] Rust / array<vec4<f32>,2> WGSL
    offsets:      [f32; 8], // offset 32 — total 64 bytes
}

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
struct PainterAudioUniforms {
    time_seconds: f32,
    bass:         f32,       // = bands[0], convenience field for PrintHead / legacy
    mid:          f32,       // = bands[3], convenience field
    beat_decay:   f32,       // exp(-5*dt) decay, reset to 1.0 on beat onset
    bands:        [f32; 8],  // 8-band energies, Android-parity cutoffs
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
    contrast_passes:      f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DistortionPlusUniforms {
    yaw:   f32,  // radians
    pitch: f32,
    roll:  f32,
    _pad:  f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RibbonUniforms {
    resolution:   [f32; 2],      // offset  0: vec2<f32> in WGSL, [f32;2] in Rust (same layout)
    time_seconds: f32,           // offset  8
    intensity:    f32,           // offset 12
    color:        [f32; 4],      // offset 16: vec4<f32>
    collapse:     [f32; 4],      // offset 32: vec4<f32>
    bands:        [[f32; 4]; 2], // offset 48: array<vec4<f32>,2>  — total 80 bytes
}

// 0=none 1=circle 2=square 3=rounded 4=hexagon 5=octagon 6=flower 7=star
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
    Flower  = 6,
    Star    = 7,
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
    AudioPaint,
    PrintHead,
    Image,
}

impl PainterKind {
    pub fn next(self) -> Self {
        match self {
            PainterKind::HueStripe  => PainterKind::Spiral,
            PainterKind::Spiral     => PainterKind::Plasma,
            PainterKind::Plasma     => PainterKind::Skin,
            PainterKind::Skin       => PainterKind::AudioPaint,
            PainterKind::AudioPaint => PainterKind::PrintHead,
            PainterKind::PrintHead  => PainterKind::Image,
            PainterKind::Image      => PainterKind::HueStripe,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            PainterKind::HueStripe  => "HueStripe",
            PainterKind::Spiral     => "Spiral",
            PainterKind::Plasma     => "Plasma",
            PainterKind::Skin       => "Skin",
            PainterKind::AudioPaint => "AudioPaint",
            PainterKind::PrintHead  => "PrintHead",
            PainterKind::Image      => "Image",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum ResolutionPreset {
    SD480,
    #[default]
    HD720,
    FullHD,
    UHD4K,
}

impl ResolutionPreset {
    pub fn dimensions(self) -> (u32, u32) {
        match self {
            ResolutionPreset::SD480  => (854, 480),
            ResolutionPreset::HD720  => (1280, 720),
            ResolutionPreset::FullHD => (1920, 1080),
            ResolutionPreset::UHD4K  => (3840, 2160),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            ResolutionPreset::SD480  => "480p",
            ResolutionPreset::HD720  => "720p",
            ResolutionPreset::FullHD => "1080p",
            ResolutionPreset::UHD4K  => "4K",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum FramerateChoice {
    Fps30,
    #[default]
    Fps60,
}

impl FramerateChoice {
    pub fn fps(self) -> u32 {
        match self {
            FramerateChoice::Fps30 => 30,
            FramerateChoice::Fps60 => 60,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            FramerateChoice::Fps30 => "30 fps",
            FramerateChoice::Fps60 => "60 fps",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct ParamLocks {
    // Geometry
    pub painter_kind: bool,
    pub current_shape: bool,
    pub fold_count: bool,
    pub zoom: bool,
    pub rotation_speed_scale: bool,
    // Frame
    pub frame_shape: bool,
    pub frame_size: bool,
    pub frame_color_hue: bool,
    // Effects
    pub invert_enabled: bool,
    pub colorize_enabled: bool,
    pub colorize_hue: bool,
    pub colorize_intensity: bool,
    pub distortion_enabled: bool,
    pub distortion_amplitude: bool,
    pub distortion_frequency: bool,
    pub distortion_plus_enabled: bool,
    pub distortion_plus_yaw:     bool,
    pub distortion_plus_pitch:   bool,
    pub distortion_plus_roll:    bool,
    pub contrast: bool,
    pub contrast_passes: bool,
    pub saturation: bool,
    // Audio
    pub bass_zoom_strength: bool,
    pub beat_reactivity: bool,
    pub midi_shake_enabled: bool,
    pub audio_shake_enabled: bool,
    // Ribbons
    pub ribbons_enabled:   bool,
    pub ribbons_intensity: bool,
    // Palette
    pub palette_mode:     bool,
    pub palette_tint:     bool,
    pub palette_mono_hue: bool,
    // Color Theory
    pub color_harmony:           bool,
    pub color_anchor_hue:        bool,
    pub color_saturation:        bool,
    pub color_value:             bool,
    pub color_harmony_strength:  bool,
    pub applied_harmony_enabled: bool,
    // Blackhole
    pub blackhole_enabled:        bool,
    pub blackhole_warp_strength:  bool,
    pub blackhole_warp_curve:     bool,
    pub blackhole_alpha_radius:   bool,
    pub blackhole_wander_amount:  bool,
    // Phantom Alpha
    pub phantom_enabled:       bool,
    pub phantom_delay_seconds: bool,
    pub phantom_key_color:     bool,
    pub phantom_key_tolerance: bool,
    pub phantom_key_softness:  bool,
    pub phantom_key_strength:  bool,
    pub phantom_opacity:       bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaletteMode {
    #[default]
    Off,
    Warm,
    Cool,
    Earth,
    Neon,
    Monochrome,
    Harmony,
}

impl PaletteMode {
    fn to_u32(self) -> u32 {
        match self {
            PaletteMode::Off        => 0,
            PaletteMode::Warm       => 1,
            PaletteMode::Cool       => 2,
            PaletteMode::Earth      => 3,
            PaletteMode::Neon       => 4,
            PaletteMode::Monochrome => 5,
            PaletteMode::Harmony    => 6,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            PaletteMode::Off        => "Off",
            PaletteMode::Warm       => "Warm",
            PaletteMode::Cool       => "Cool",
            PaletteMode::Earth      => "Earth",
            PaletteMode::Neon       => "Neon",
            PaletteMode::Monochrome => "Monochrome",
            PaletteMode::Harmony    => "Harmony",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudioSourceMode {
    #[default]
    File,
    Mic,
    Loopback,
    Silent,
}

impl AudioSourceMode {
    fn as_str(self) -> &'static str {
        match self {
            AudioSourceMode::File     => "File",
            AudioSourceMode::Mic      => "Mic",
            AudioSourceMode::Loopback => "Loopback",
            AudioSourceMode::Silent   => "Silent",
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
    pub distortion_plus_enabled: bool,
    pub distortion_plus_yaw:     f32,
    pub distortion_plus_pitch:   f32,
    pub distortion_plus_roll:    f32,
    pub midi_shake_enabled:  bool,
    pub audio_shake_enabled: bool,
    pub ribbons_enabled:   bool,
    pub ribbons_intensity: f32,
    pub bass_zoom_strength: f32,
    /// Master multiplier for beat-driven animation magnitudes.
    /// Default 0.25 matches Android. Effective multiplier is
    /// (beat_reactivity * 4.0), so:
    ///   0.0  = no beat-driven animation
    ///   0.25 = Android-default intensity (1.0×)
    ///   1.0  = 4× intensity (kick_shake unclamped; beat_decay
    ///          and ribbon collapse saturate at 1.0 above 0.25)
    /// Excluded from random/party randomization per Android
    /// convention — user-controlled only.
    pub beat_reactivity: f32,
    pub painter_kind: PainterKind,
    pub contrast: f32,
    pub saturation: f32,
    pub contrast_passes: u32,
    pub random_mode_enabled: bool,
    pub random_mode_aggressiveness: f32,
    pub reactive_mode_enabled: bool,
    pub reactive_mode_aggressiveness: f32,
    pub party_mode_enabled: bool,
    pub party_mode_aggressiveness: f32,
    pub locks: ParamLocks,
    pub export_resolution: ResolutionPreset,
    pub export_framerate:  FramerateChoice,
    pub export_live_preview: bool,
    pub audio_source_mode: AudioSourceMode,
    pub palette_mode:     PaletteMode,
    pub palette_tint:     f32,
    pub palette_mono_hue: f32,
    pub blackhole_enabled:        bool,
    pub blackhole_warp_strength:  f32,
    pub blackhole_warp_curve:     f32,
    pub blackhole_alpha_radius:   f32,
    pub blackhole_wander_amount:  f32,
    // Color Theory harmony params
    pub color_harmony:            color::ColorHarmony,
    pub color_anchor_hue:         f32,   // 0.0 .. 360.0
    pub color_saturation:         f32,   // 0.0 .. 1.0
    pub color_value:              f32,   // 0.0 .. 1.0
    pub color_harmony_strength:   f32,   // 0.0 .. 1.0 blend into nearest harmony hue
    pub applied_harmony_enabled:  bool,  // recolor Skin/Image/PrintHead via harmony
    // Phantom Alpha (mutually exclusive with blackhole in live path)
    pub phantom_enabled:       bool,
    pub phantom_delay_seconds: f32,
    pub phantom_key_color:     [f32; 3],
    pub phantom_key_tolerance: f32,
    pub phantom_key_softness:  f32,
    pub phantom_key_strength:  f32,
    pub phantom_opacity:       f32,
}

impl Default for VisualParams {
    fn default() -> Self {
        Self {
            current_shape: ShapeKind::Cylinder,
            fold_count: 12.0,
            zoom: 1.0,
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
            distortion_plus_enabled: false,
            distortion_plus_yaw:     0.0,
            distortion_plus_pitch:   0.0,
            distortion_plus_roll:    0.0,
            midi_shake_enabled:  true,
            audio_shake_enabled: false,
            ribbons_enabled:   false,
            ribbons_intensity: 0.5,
            bass_zoom_strength: 0.3,
            beat_reactivity: 0.25,
            painter_kind: PainterKind::HueStripe,
            contrast: 1.0,
            saturation: 1.0,
            contrast_passes: 1,
            random_mode_enabled: false,
            random_mode_aggressiveness: 0.65,
            reactive_mode_enabled: false,
            reactive_mode_aggressiveness: 0.5,
            party_mode_enabled: false,
            party_mode_aggressiveness: 0.75,
            locks: ParamLocks::default(),
            export_resolution: ResolutionPreset::HD720,
            export_framerate:  FramerateChoice::Fps60,
            export_live_preview: true,
            audio_source_mode: AudioSourceMode::File,
            palette_mode:     PaletteMode::Off,
            palette_tint:     1.0,
            palette_mono_hue: 200.0,
            blackhole_enabled:       false,
            blackhole_warp_strength: 0.92,
            blackhole_warp_curve:    0.97,
            blackhole_alpha_radius:  0.5,
            blackhole_wander_amount: 0.005,
            color_harmony:           color::ColorHarmony::Analogous,
            color_anchor_hue:        210.0,
            color_saturation:        0.75,
            color_value:             0.85,
            color_harmony_strength:  0.5,
            applied_harmony_enabled: false,
            phantom_enabled:       false,
            phantom_delay_seconds: 1.0,
            phantom_key_color:     [0.0, 0.0, 1.0],
            phantom_key_tolerance: 0.15,
            phantom_key_softness:  0.05,
            phantom_key_strength:  1.0,
            phantom_opacity:       0.85,
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
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
    #[serde(default)]
    distortion_plus_enabled: bool,
    #[serde(default)]
    distortion_plus_yaw:     f32,
    #[serde(default)]
    distortion_plus_pitch:   f32,
    #[serde(default)]
    distortion_plus_roll:    f32,
    #[serde(default = "default_true")]
    midi_shake_enabled:  bool,
    #[serde(default)]
    audio_shake_enabled: bool,
    #[serde(default)]
    ribbons_enabled: bool,
    #[serde(default = "default_half_f32")]
    ribbons_intensity: f32,
    bass_zoom_strength: f32,
    #[serde(default = "default_quarter_f32")]
    beat_reactivity: f32,
    painter_kind: String,
    #[serde(default = "default_one_f32")]
    contrast: f32,
    #[serde(default = "default_one_f32")]
    saturation: f32,
    #[serde(default = "default_one_u32")]
    contrast_passes: u32,
    #[serde(default)]
    random_mode_enabled: bool,
    #[serde(default = "default_half_f32")]
    random_mode_aggressiveness: f32,
    #[serde(default)]
    reactive_mode_enabled: bool,
    #[serde(default = "default_half_f32")]
    reactive_mode_aggressiveness: f32,
    #[serde(default)]
    party_mode_enabled: bool,
    #[serde(default = "default_half_f32")]
    party_mode_aggressiveness: f32,
    #[serde(default)]
    locks: ParamLocks,
    #[serde(default)]
    export_resolution: ResolutionPreset,
    #[serde(default)]
    export_framerate: FramerateChoice,
    #[serde(default = "default_true")]
    export_live_preview: bool,
    #[serde(default = "default_audio_source_mode")]
    audio_source_mode: String,
    #[serde(default = "default_palette_mode")]
    palette_mode: String,
    #[serde(default = "default_one_f32")]
    palette_tint: f32,
    #[serde(default = "default_palette_mono_hue")]
    palette_mono_hue: f32,
    #[serde(default)]
    blackhole_enabled: bool,
    #[serde(default = "default_blackhole_warp_strength")]
    blackhole_warp_strength: f32,
    #[serde(default = "default_blackhole_warp_curve")]
    blackhole_warp_curve: f32,
    #[serde(default = "default_blackhole_alpha_radius")]
    blackhole_alpha_radius: f32,
    #[serde(default = "default_blackhole_wander_amount")]
    blackhole_wander_amount: f32,
    #[serde(default)]
    phantom_enabled: bool,
    #[serde(default = "default_phantom_delay_seconds")]
    phantom_delay_seconds: f32,
    #[serde(default = "default_phantom_key_color")]
    phantom_key_color: [f32; 3],
    #[serde(default = "default_phantom_key_tolerance")]
    phantom_key_tolerance: f32,
    #[serde(default = "default_phantom_key_softness")]
    phantom_key_softness: f32,
    #[serde(default = "default_phantom_key_strength")]
    phantom_key_strength: f32,
    #[serde(default = "default_phantom_opacity")]
    phantom_opacity: f32,
    #[serde(default = "default_color_harmony")]
    color_harmony:    color::ColorHarmony,
    #[serde(default = "default_color_anchor_hue")]
    color_anchor_hue: f32,
    #[serde(default = "default_color_saturation")]
    color_saturation: f32,
    #[serde(default = "default_color_value")]
    color_value:      f32,
    #[serde(default = "default_color_harmony_strength")]
    color_harmony_strength: f32,
    #[serde(default)]
    applied_harmony_enabled: bool,
}

fn default_one_f32()          -> f32    { 1.0 }
fn default_half_f32()         -> f32    { 0.5 }
fn default_quarter_f32()      -> f32    { 0.25 }
fn default_one_u32()          -> u32    { 1 }
fn default_true()             -> bool   { true }
fn default_audio_source_mode()  -> String { "File".to_string() }
fn default_palette_mode()          -> String { "Off".to_string() }
fn default_palette_mono_hue()      -> f32    { 200.0 }
fn default_blackhole_warp_strength() -> f32  { 0.92 }
fn default_blackhole_warp_curve()    -> f32  { 0.97 }
fn default_blackhole_alpha_radius()  -> f32  { 0.5 }
fn default_blackhole_wander_amount() -> f32  { 0.005 }
fn default_phantom_delay_seconds()   -> f32  { 1.0 }
fn default_phantom_key_color() -> [f32; 3]   { [0.0, 0.0, 1.0] }
fn default_phantom_key_tolerance()   -> f32  { 0.15 }
fn default_phantom_key_softness()    -> f32  { 0.05 }
fn default_phantom_key_strength()    -> f32  { 1.0 }
fn default_phantom_opacity()         -> f32  { 0.85 }
fn default_color_harmony()         -> color::ColorHarmony { color::ColorHarmony::Analogous }
fn default_color_anchor_hue()      -> f32                 { 210.0 }
fn default_color_saturation()      -> f32                 { 0.75 }
fn default_color_value()           -> f32                 { 0.85 }
fn default_color_harmony_strength() -> f32                { 0.5 }

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
            distortion_plus_enabled: params.distortion_plus_enabled,
            distortion_plus_yaw:     params.distortion_plus_yaw,
            distortion_plus_pitch:   params.distortion_plus_pitch,
            distortion_plus_roll:    params.distortion_plus_roll,
            midi_shake_enabled:  params.midi_shake_enabled,
            audio_shake_enabled: params.audio_shake_enabled,
            ribbons_enabled:   params.ribbons_enabled,
            ribbons_intensity: params.ribbons_intensity,
            bass_zoom_strength: params.bass_zoom_strength,
            beat_reactivity: params.beat_reactivity,
            painter_kind: params.painter_kind.name().to_string(),
            contrast: params.contrast,
            saturation: params.saturation,
            contrast_passes: params.contrast_passes,
            random_mode_enabled: params.random_mode_enabled,
            random_mode_aggressiveness: params.random_mode_aggressiveness,
            reactive_mode_enabled: params.reactive_mode_enabled,
            reactive_mode_aggressiveness: params.reactive_mode_aggressiveness,
            party_mode_enabled: params.party_mode_enabled,
            party_mode_aggressiveness: params.party_mode_aggressiveness,
            locks: params.locks,
            export_resolution: params.export_resolution,
            export_framerate:  params.export_framerate,
            export_live_preview: params.export_live_preview,
            audio_source_mode: params.audio_source_mode.as_str().to_string(),
            palette_mode:     params.palette_mode.as_str().to_string(),
            palette_tint:     params.palette_tint,
            palette_mono_hue: params.palette_mono_hue,
            blackhole_enabled:       params.blackhole_enabled,
            blackhole_warp_strength: params.blackhole_warp_strength,
            blackhole_warp_curve:    params.blackhole_warp_curve,
            blackhole_alpha_radius:  params.blackhole_alpha_radius,
            blackhole_wander_amount: params.blackhole_wander_amount,
            phantom_enabled:       params.phantom_enabled,
            phantom_delay_seconds: params.phantom_delay_seconds,
            phantom_key_color:     params.phantom_key_color,
            phantom_key_tolerance: params.phantom_key_tolerance,
            phantom_key_softness:  params.phantom_key_softness,
            phantom_key_strength:  params.phantom_key_strength,
            phantom_opacity:       params.phantom_opacity,
            color_harmony:           params.color_harmony,
            color_anchor_hue:        params.color_anchor_hue,
            color_saturation:        params.color_saturation,
            color_value:             params.color_value,
            color_harmony_strength:  params.color_harmony_strength,
            applied_harmony_enabled: params.applied_harmony_enabled,
        }
    }

    pub fn apply_to_params(&self, params: &mut VisualParams) {
        params.current_shape = match self.current_shape.as_str() {
            "Sphere"      => ShapeKind::Sphere,
            "Cube"        => ShapeKind::Cube,
            "Tetrahedron" => ShapeKind::Tetrahedron,
            "Icosahedron" => ShapeKind::Icosahedron,
            "Urchin"      => ShapeKind::Urchin,
            "Caltrop"     => ShapeKind::Caltrop,
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
            "Flower"  => FrameShape::Flower,
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
        params.distortion_plus_enabled = self.distortion_plus_enabled;
        params.distortion_plus_yaw     = self.distortion_plus_yaw;
        params.distortion_plus_pitch   = self.distortion_plus_pitch;
        params.distortion_plus_roll    = self.distortion_plus_roll;
        params.midi_shake_enabled  = self.midi_shake_enabled;
        params.audio_shake_enabled = self.audio_shake_enabled;
        params.ribbons_enabled   = self.ribbons_enabled;
        params.ribbons_intensity = self.ribbons_intensity;
        params.bass_zoom_strength = self.bass_zoom_strength;
        params.beat_reactivity    = self.beat_reactivity;
        params.painter_kind = match self.painter_kind.as_str() {
            "Spiral"     => PainterKind::Spiral,
            "Plasma"     => PainterKind::Plasma,
            "Skin"       => PainterKind::Skin,
            "AudioPaint" => PainterKind::AudioPaint,
            "PrintHead"  => PainterKind::PrintHead,
            "Image"      => PainterKind::Image,
            _            => PainterKind::HueStripe,
        };
        params.contrast        = self.contrast;
        params.saturation      = self.saturation;
        params.contrast_passes = self.contrast_passes;
        params.random_mode_enabled         = self.random_mode_enabled;
        params.random_mode_aggressiveness  = self.random_mode_aggressiveness;
        params.reactive_mode_enabled       = self.reactive_mode_enabled;
        params.reactive_mode_aggressiveness = self.reactive_mode_aggressiveness;
        params.party_mode_enabled          = self.party_mode_enabled;
        params.party_mode_aggressiveness   = self.party_mode_aggressiveness;
        params.locks              = self.locks;
        params.export_resolution  = self.export_resolution;
        params.export_framerate   = self.export_framerate;
        params.export_live_preview = self.export_live_preview;
        params.audio_source_mode = match self.audio_source_mode.as_str() {
            "Mic"      => AudioSourceMode::Mic,
            "Loopback" => AudioSourceMode::Loopback,
            "Silent"   => AudioSourceMode::Silent,
            _          => AudioSourceMode::File,
        };
        params.palette_mode = match self.palette_mode.as_str() {
            "Warm"       => PaletteMode::Warm,
            "Cool"       => PaletteMode::Cool,
            "Earth"      => PaletteMode::Earth,
            "Neon"       => PaletteMode::Neon,
            "Monochrome" => PaletteMode::Monochrome,
            "Harmony"    => PaletteMode::Harmony,
            _            => PaletteMode::Off,
        };
        params.palette_tint     = self.palette_tint;
        params.palette_mono_hue = self.palette_mono_hue;
        params.blackhole_enabled       = self.blackhole_enabled;
        params.blackhole_warp_strength = self.blackhole_warp_strength;
        params.blackhole_warp_curve    = self.blackhole_warp_curve;
        params.blackhole_alpha_radius  = self.blackhole_alpha_radius;
        params.blackhole_wander_amount = self.blackhole_wander_amount;
        params.phantom_enabled       = self.phantom_enabled;
        params.phantom_delay_seconds = self.phantom_delay_seconds;
        params.phantom_key_color     = self.phantom_key_color;
        params.phantom_key_tolerance = self.phantom_key_tolerance;
        params.phantom_key_softness  = self.phantom_key_softness;
        params.phantom_key_strength  = self.phantom_key_strength;
        params.phantom_opacity       = self.phantom_opacity;
        params.color_harmony           = self.color_harmony;
        params.color_anchor_hue        = self.color_anchor_hue;
        params.color_saturation        = self.color_saturation;
        params.color_value             = self.color_value;
        params.color_harmony_strength  = self.color_harmony_strength;
        params.applied_harmony_enabled = self.applied_harmony_enabled;
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

pub struct LoadedAudio {
    pub samples:          Vec<f32>,   // interleaved: L,R,L,R,... (or mono repeated for stereo)
    pub sample_rate:      u32,
    pub channels:         u16,
    pub duration_seconds: f32,
    pub source_path:      String,
}

impl LoadedAudio {
    pub fn duration_samples(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    pub fn sample_at_time(&self, time_seconds: f32) -> (f32, f32) {
        let frame_index = time_seconds * self.sample_rate as f32;
        let frame_floor = frame_index.floor() as usize;
        if frame_floor >= self.duration_samples() {
            return (0.0, 0.0);
        }
        let i = frame_floor * self.channels as usize;
        if self.channels == 1 {
            let s = self.samples.get(i).copied().unwrap_or(0.0);
            (s, s)
        } else {
            let l = self.samples.get(i).copied().unwrap_or(0.0);
            let r = self.samples.get(i + 1).copied().unwrap_or(0.0);
            (l, r)
        }
    }
}

fn decode_audio_file(path: &std::path::Path) -> Result<LoadedAudio, String> {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    const MAX_FILE_BYTES: u64 = 200 * 1024 * 1024;

    let metadata = std::fs::metadata(path)
        .map_err(|e| format!("can't read file metadata: {}", e))?;
    if metadata.len() > MAX_FILE_BYTES {
        return Err(format!(
            "audio file too large ({} MB, max 200 MB)",
            metadata.len() / 1024 / 1024
        ));
    }

    let file = std::fs::File::open(path)
        .map_err(|e| format!("can't open file: {}", e))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("format probe failed: {}", e))?;

    let mut format_reader = probed.format;

    let track = format_reader
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| "no audio track found".to_string())?;

    let track_id     = track.id;
    let codec_params = track.codec_params.clone();

    let sample_rate = codec_params.sample_rate
        .ok_or_else(|| "no sample rate in codec params".to_string())?;
    let channel_count = codec_params.channels
        .ok_or_else(|| "no channels in codec params".to_string())?
        .count() as u16;

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| format!("decoder creation failed: {}", e))?;

    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format_reader.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(format!("packet read error: {}", e)),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let ch = channel_count as usize;
                match decoded {
                    AudioBufferRef::F32(buf) => {
                        for f in 0..buf.frames() {
                            for c in 0..ch { samples.push(buf.chan(c)[f]); }
                        }
                    }
                    AudioBufferRef::F64(buf) => {
                        for f in 0..buf.frames() {
                            for c in 0..ch { samples.push(buf.chan(c)[f] as f32); }
                        }
                    }
                    AudioBufferRef::S32(buf) => {
                        for f in 0..buf.frames() {
                            for c in 0..ch { samples.push(buf.chan(c)[f] as f32 / i32::MAX as f32); }
                        }
                    }
                    AudioBufferRef::S24(buf) => {
                        for f in 0..buf.frames() {
                            for c in 0..ch { samples.push(buf.chan(c)[f].inner() as f32 / 8_388_607.0); }
                        }
                    }
                    AudioBufferRef::S16(buf) => {
                        for f in 0..buf.frames() {
                            for c in 0..ch { samples.push(buf.chan(c)[f] as f32 / i16::MAX as f32); }
                        }
                    }
                    AudioBufferRef::U8(buf) => {
                        for f in 0..buf.frames() {
                            for c in 0..ch { samples.push((buf.chan(c)[f] as f32 - 128.0) / 128.0); }
                        }
                    }
                    _ => return Err("unsupported sample format".to_string()),
                }
            }
            Err(SymphoniaError::DecodeError(e)) => {
                log::warn!("decode error (skipping packet): {}", e);
            }
            Err(e) => return Err(format!("fatal decode error: {}", e)),
        }
    }

    if samples.is_empty() {
        return Err("decoded zero samples".to_string());
    }

    let duration_seconds = (samples.len() / channel_count as usize) as f32 / sample_rate as f32;

    log::info!(
        "Loaded audio: {} samples, {} channels, {} Hz, {:.2}s — {}",
        samples.len(), channel_count, sample_rate, duration_seconds,
        path.display()
    );

    Ok(LoadedAudio {
        samples,
        sample_rate,
        channels: channel_count,
        duration_seconds,
        source_path: path.display().to_string(),
    })
}

// ── Audio Player ────────────────────────────────────────────────────────────

pub struct AudioPlayer {
    /// Current playback position in SOURCE FRAMES (not interleaved samples).
    position_frames: Arc<AtomicUsize>,
    is_playing: Arc<AtomicBool>,
    pub output_sample_rate: u32,
    _output_stream: cpal::Stream,
}

impl AudioPlayer {
    pub fn new(audio: Arc<LoadedAudio>) -> Result<Self, String> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host   = cpal::default_host();
        let device = host.default_output_device()
            .ok_or_else(|| "no output device found".to_string())?;
        let config = device.default_output_config()
            .map_err(|e| format!("output config: {}", e))?;

        let output_sample_rate = config.sample_rate().0;
        let output_channels    = config.channels() as usize;
        let sample_format      = config.sample_format();

        log::info!("AudioPlayer output: {} Hz, {} ch, {:?}",
            output_sample_rate, output_channels, sample_format);

        if sample_format != cpal::SampleFormat::F32 {
            return Err(format!(
                "output device uses {:?}; only F32 output is supported", sample_format
            ));
        }

        let position   = Arc::new(AtomicUsize::new(0));
        let is_playing = Arc::new(AtomicBool::new(false));

        let pos_c  = position.clone();
        let play_c = is_playing.clone();
        let aud_c  = audio.clone();

        // How many source frames advance per output frame
        let src_per_out = audio.sample_rate as f64 / output_sample_rate as f64;
        let total_src   = audio.samples.len() / audio.channels.max(1) as usize;

        let stream_config: cpal::StreamConfig = config.into();
        let stream = device.build_output_stream(
            &stream_config,
            move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let frames = output.len() / output_channels;
                if !play_c.load(Ordering::Relaxed) {
                    for s in output.iter_mut() { *s = 0.0; }
                    return;
                }
                let base   = pos_c.load(Ordering::Relaxed);
                let src_ch = aud_c.channels as usize;
                for f in 0..frames {
                    let src_frame = base + (f as f64 * src_per_out) as usize;
                    let (l, r) = if src_frame < total_src {
                        let i = src_frame * src_ch;
                        if src_ch >= 2 {
                            (aud_c.samples[i], aud_c.samples[i + 1])
                        } else {
                            let s = aud_c.samples[i];
                            (s, s)
                        }
                    } else {
                        (0.0, 0.0)
                    };
                    if output_channels == 1 {
                        output[f] = (l + r) * 0.5;
                    } else {
                        output[f * output_channels]     = l;
                        output[f * output_channels + 1] = r;
                        for c in 2..output_channels {
                            output[f * output_channels + c] = 0.0;
                        }
                    }
                }
                let new_pos = base + (frames as f64 * src_per_out) as usize;
                pos_c.store(new_pos, Ordering::Relaxed);
                if new_pos >= total_src {
                    play_c.store(false, Ordering::Relaxed);
                    pos_c.store(0, Ordering::Relaxed);
                }
            },
            |err| log::error!("audio output error: {}", err),
            None,
        ).map_err(|e| format!("stream build: {}", e))?;

        stream.play().map_err(|e| format!("stream play: {}", e))?;

        Ok(Self {
            position_frames: position,
            is_playing,
            output_sample_rate,
            _output_stream: stream,
        })
    }

    pub fn play(&self) { self.is_playing.store(true,  Ordering::Relaxed); }
    pub fn pause(&self) { self.is_playing.store(false, Ordering::Relaxed); }
    pub fn is_playing(&self) -> bool { self.is_playing.load(Ordering::Relaxed) }

    /// Current position in source frames.
    pub fn position_frames(&self) -> usize { self.position_frames.load(Ordering::Relaxed) }

    /// Seek to a specific source frame.
    pub fn seek_frames(&self, target: usize) { self.position_frames.store(target, Ordering::Relaxed); }

    /// Current position in seconds (based on source sample rate).
    pub fn position_seconds(&self, source_sample_rate: u32) -> f32 {
        self.position_frames() as f32 / source_sample_rate as f32
    }
}

/// Compute (8-band energies, rms) from the file at the player's current position.
/// Uses a 2048-frame mono window centred on the playhead (matches FFT size).
fn compute_file_energies(player: &AudioPlayer, audio: &LoadedAudio) -> ([f32; 8], f32) {
    const WINDOW: usize = 2048;
    let ch = audio.channels.max(1) as usize;
    let total_frames = audio.samples.len() / ch;
    let pos = player.position_frames();
    let start = pos.saturating_sub(WINDOW / 2).min(total_frames);
    let end   = (start + WINDOW).min(total_frames);
    if start >= end { return ([0.0; 8], 0.0); }
    let mono: Vec<f32> = (start..end)
        .map(|f| {
            let i = f * ch;
            (0..ch).map(|c| audio.samples.get(i + c).copied().unwrap_or(0.0))
                .sum::<f32>() / ch as f32
        })
        .collect();
    let rms = (mono.iter().map(|x| x * x).sum::<f32>() / mono.len() as f32).sqrt();
    let bands = audio::compute_band_energies(&mono, audio.sample_rate);
    (bands, rms)
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

fn start_audio_for_mode(mode: AudioSourceMode) -> Option<audio::AudioCapture> {
    match mode {
        AudioSourceMode::Mic => match audio::AudioCapture::start(false) {
            Ok(a)  => { log::info!("Mic capture started"); Some(a) }
            Err(e) => { log::warn!("Mic capture failed: {} — audio source will be silent", e); None }
        },
        AudioSourceMode::Loopback => match audio::AudioCapture::start(true) {
            Ok(a)  => { log::info!("Loopback capture started"); Some(a) }
            Err(e) => { log::warn!("Loopback capture failed: {} — audio source will be silent", e); None }
        },
        AudioSourceMode::File | AudioSourceMode::Silent => None,
    }
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

pub struct OfflineTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
}

impl OfflineTarget {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offline Render Target"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::COPY_SRC
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view, width, height, format }
    }
}

pub struct FrameSaveJob {
    pub frame_index: u32,
    pub width: u32,
    pub height: u32,
    pub rgba_bytes: Vec<u8>,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportPhase {
    Rendering,
    Muxing,
    Complete,
    Failed,
}

pub struct MuxResult {
    pub success: bool,
    pub output_path: PathBuf,
    pub error_message: Option<String>,
}

struct ExportRibbonState {
    #[allow(dead_code)]
    tex_a:  wgpu::Texture,
    view_a: wgpu::TextureView,
    #[allow(dead_code)]
    tex_b:  wgpu::Texture,
    view_b: wgpu::TextureView,
    ping:   bool,  // true=read A write B; false=read B write A
    bg_read_a:       wgpu::BindGroup,  // update pass: reads texture A
    bg_read_b:       wgpu::BindGroup,  // update pass: reads texture B
    composite_bg_a:  wgpu::BindGroup,  // composite pass: blits texture A
    composite_bg_b:  wgpu::BindGroup,  // composite pass: blits texture B
}

struct ExportFeedbackState {
    #[allow(dead_code)]
    tex_a: wgpu::Texture,
    view_a: wgpu::TextureView,
    #[allow(dead_code)]
    tex_b: wgpu::Texture,
    view_b: wgpu::TextureView,
    bg_a:      wgpu::BindGroup,  // feedback_bgl: prev=A, scene=offline_target; write target=B
    bg_b:      wgpu::BindGroup,  // feedback_bgl: prev=B, scene=offline_target; write target=A
    blit_bg_a: wgpu::BindGroup,  // blit_bgl: blit A → offline_target
    blit_bg_b: wgpu::BindGroup,  // blit_bgl: blit B → offline_target
    current_is_a: bool,
    wander_pos: [f32; 2],
    wander_vel: [f32; 2],
    rng: rand::rngs::StdRng,
}

pub struct ExportState {
    pub phase: ExportPhase,
    pub current_frame: u32,
    pub total_frames: u32,
    pub output_dir: PathBuf,
    pub fps: u32,
    pub start_time: Instant,
    // Rendering phase: Some; set to None when transitioning to Muxing to signal worker
    pub frame_save_sender: Option<Sender<FrameSaveJob>>,
    pub frame_save_thread: Option<std::thread::JoinHandle<()>>,
    pub mux_thread: Option<std::thread::JoinHandle<MuxResult>>,
    // Smoothed audio energies — EMA prevents FFT window-shift jitter at 60fps
    pub export_bands_smoothed: [f32; 8],
    pub export_rms_smoothed:   f32,
    pub offline_analyzer:      audio::OfflineAnalyzer,
    // f32 accumulators for deterministic mode-timer replay (replaces Instant in export path)
    pub export_random_elapsed:   f32,
    pub export_reactive_elapsed: f32,
    pub export_party_elapsed:    f32,
}

fn spawn_frame_save_worker() -> (Sender<FrameSaveJob>, std::thread::JoinHandle<()>) {
    let (tx, rx) = std::sync::mpsc::channel::<FrameSaveJob>();
    let handle = std::thread::spawn(move || {
        while let Ok(job) = rx.recv() {
            match image::RgbaImage::from_raw(job.width, job.height, job.rgba_bytes) {
                Some(img) => {
                    if let Err(e) = img.save(&job.output_path) {
                        log::error!("frame {:05}: save failed: {}", job.frame_index, e);
                    }
                }
                None => {
                    log::error!("frame {:05}: failed to construct RgbaImage", job.frame_index);
                }
            }
        }
    });
    (tx, handle)
}

pub fn run_ffmpeg_mux(
    png_dir: &std::path::Path,
    audio_path: &str,
    output_path: &std::path::Path,
    fps: u32,
) -> MuxResult {
    use std::process::Command;
    let png_pattern = png_dir.join("frame_%05d.png");
    log::info!(
        "ffmpeg mux: {} + {} → {}",
        png_pattern.display(), audio_path, output_path.display()
    );
    match Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate", &fps.to_string(),
            "-i", &png_pattern.to_string_lossy(),
            "-i", audio_path,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            &output_path.to_string_lossy(),
        ])
        .output()
    {
        Ok(out) if out.status.success() => {
            let _ = std::fs::remove_dir_all(png_dir);
            log::info!("Mux complete: {}", output_path.display());
            MuxResult { success: true, output_path: output_path.to_path_buf(), error_message: None }
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            log::error!("ffmpeg failed:\n{}", stderr);
            MuxResult { success: false, output_path: output_path.to_path_buf(), error_message: Some(stderr) }
        }
        Err(e) => {
            let msg = format!("ffmpeg subprocess error: {}", e);
            log::error!("{}", msg);
            MuxResult { success: false, output_path: output_path.to_path_buf(), error_message: Some(msg) }
        }
    }
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

    // AudioPaint + PrintHead painter resources (shared audio uniform)
    #[allow(dead_code)]
    painter_audio_bgl: wgpu::BindGroupLayout,
    painter_audio_buffer: wgpu::Buffer,
    painter_audio_bind_group: wgpu::BindGroup,
    painter_audio_paint_pipeline: wgpu::RenderPipeline,
    painter_print_head_pipeline: wgpu::RenderPipeline,

    // Image painter resources
    #[allow(dead_code)]
    painter_image_texture: wgpu::Texture,
    #[allow(dead_code)]
    painter_image_view: wgpu::TextureView,
    #[allow(dead_code)]
    painter_image_sampler: wgpu::Sampler,
    painter_image_bind_group: wgpu::BindGroup,
    painter_image_pipeline: wgpu::RenderPipeline,

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

    // Pass 2.5 — distortion plus (shape FBO → dp FBO, optional equirectangular rotation)
    #[allow(dead_code)]
    distortion_plus_texture: wgpu::Texture,
    distortion_plus_view: wgpu::TextureView,
    distortion_plus_uniforms_buffer: wgpu::Buffer,
    distortion_plus_bgl: wgpu::BindGroupLayout,
    distortion_plus_bind_group: wgpu::BindGroup,
    distortion_plus_pipeline: wgpu::RenderPipeline,

    // Pass 3 — kaleido fold (shape FBO → kaleido FBO)
    #[allow(dead_code)]
    kaleido_texture: wgpu::Texture,
    kaleido_view: wgpu::TextureView,
    kaleido_uniforms_buffer: wgpu::Buffer,
    kaleido_bgl: wgpu::BindGroupLayout,
    kaleido_bind_group: wgpu::BindGroup,
    kaleido_bind_group_distorted: wgpu::BindGroup,
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

    // Ribbon system (ping-pong RGBA16F FBOs, PAINTER_TEXTURE_WIDTH × PAINTER_TEXTURE_HEIGHT)
    ribbon_uniforms_buffer:  wgpu::Buffer,
    ribbon_bgl:              wgpu::BindGroupLayout,
    ribbon_sampler:          wgpu::Sampler,
    #[allow(dead_code)]
    ribbon_tex_a:            wgpu::Texture,
    ribbon_view_a:           wgpu::TextureView,
    #[allow(dead_code)]
    ribbon_tex_b:            wgpu::Texture,
    ribbon_view_b:           wgpu::TextureView,
    ribbon_ping:             bool,
    ribbon_bg_read_a:        wgpu::BindGroup,
    ribbon_bg_read_b:        wgpu::BindGroup,
    ribbon_composite_bg_a:   wgpu::BindGroup,
    ribbon_composite_bg_b:   wgpu::BindGroup,
    ribbon_update_pipeline:  wgpu::RenderPipeline,
    ribbon_composite_pipeline: wgpu::RenderPipeline,

    // Blackhole pass v3 (conditional pass 5): continuous video-feedback ping-pong
    feedback_uniforms_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    feedback_bgl:             wgpu::BindGroupLayout,
    #[allow(dead_code)]
    feedback_tex_a:           wgpu::Texture,
    feedback_view_a:          wgpu::TextureView,
    #[allow(dead_code)]
    feedback_tex_b:           wgpu::Texture,
    feedback_view_b:          wgpu::TextureView,
    feedback_bg_read_a:       wgpu::BindGroup,  // reads A as prev (write target = B)
    feedback_bg_read_b:       wgpu::BindGroup,  // reads B as prev (write target = A)
    feedback_blit_bg_a:       wgpu::BindGroup,  // blit_bgl for A → swapchain
    feedback_blit_bg_b:       wgpu::BindGroup,  // blit_bgl for B → swapchain
    feedback_pipeline:        wgpu::RenderPipeline,
    feedback_current_is_a:    bool,
    blackhole_wander_pos:     [f32; 2],
    blackhole_wander_vel:     [f32; 2],
    blackhole_was_enabled:    bool,

    // Palette pass (pass 1c): scratch+copy approach (Option A)
    palette_uniforms_buffer:  wgpu::Buffer,
    #[allow(dead_code)]
    palette_bgl:              wgpu::BindGroupLayout,
    palette_bind_group:       wgpu::BindGroup,
    palette_pipeline:         wgpu::RenderPipeline,
    #[allow(dead_code)]
    palette_scratch_texture:  wgpu::Texture,
    palette_scratch_view:     wgpu::TextureView,

    // Applied harmony: recolor Skin/Image/PrintHead painter output per-pixel
    applied_harmony_buffer:     wgpu::Buffer,
    #[allow(dead_code)]
    applied_harmony_bgl:        wgpu::BindGroupLayout,
    applied_harmony_bind_group: wgpu::BindGroup,

    // Export ribbon FBOs (persistent across export frames, None when not exporting)
    export_ribbon: Option<ExportRibbonState>,
    // Export feedback FBOs for blackhole pass (None when not exporting or blackhole disabled)
    export_feedback: Option<ExportFeedbackState>,

    // Phantom Alpha: chroma-keyed delayed-frame overlay
    phantom: phantom::PhantomAlpha,

    // CPU readback for recording
    readback_buffer: wgpu::Buffer,
    readback_padded_bytes_per_row: u32,
    recorder: Option<Recorder>,

    // Offline render target for export
    pub offline_target: Option<OfflineTarget>,
    pub export_state: Option<ExportState>,

    help_overlay: HelpOverlay,

    start_time: Instant,
    last_frame_time: Instant,

    shake_offset: glam::Vec3,
    shake_velocity: glam::Vec3,
    bass_zoom_smoothed: f32,
    painter_scroll_phase: f32,

    pub skin_source_image: Option<image::DynamicImage>,
    pub crop_y_offset: f32,
    pub loaded_audio: Option<Arc<LoadedAudio>>,
    pub audio_player: Option<AudioPlayer>,

    // File-beat detection state (for shake when playing from file)
    file_rms_baseline: f32,
    last_file_beat:    Instant,

    // Autonomous mode state
    last_random_change:    Instant,
    last_reactive_trigger: Instant,
    last_party_trigger:    Instant,
    bass_mid_smoothed:     f32,
    bass_mid_baseline:     f32,

    // 8-band EMA for shader uniforms; separate from bass_zoom_smoothed (used for zoom)
    bands_smoothed:      [f32; 8],
    shader_beat_decay:   f32,
    last_audio_update:   Instant,

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

    fn create_dp_fbo(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DistortionPlus FBO"),
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

    fn create_ribbon_fbo(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Ribbon FBO"),
            size: wgpu::Extent3d {
                width: PAINTER_TEXTURE_WIDTH, height: PAINTER_TEXTURE_HEIGHT, depth_or_array_layers: 1,
            },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
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

    fn make_feedback_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Feedback BGL"),
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
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
                    contrast: 1.0, saturation: 1.0, contrast_passes: 1.0,
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

        // ── DistortionPlus FBO + uniforms + BGL + BG + pipeline ──────────────
        let (distortion_plus_texture, distortion_plus_view) = Self::create_dp_fbo(&device, w, h);
        let distortion_plus_uniforms_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("DistortionPlus uniforms"),
                contents: bytemuck::cast_slice(&[DistortionPlusUniforms {
                    yaw: 0.0, pitch: 0.0, roll: 0.0, _pad: 0.0,
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let distortion_plus_bgl = Self::make_uts_bgl(&device, "DistortionPlus BGL");
        let distortion_plus_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DistortionPlus BG"),
            layout: &distortion_plus_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: distortion_plus_uniforms_buffer.as_entire_binding(),
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
        let dp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DistortionPlus shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/distortion_plus.wgsl").into()),
        });
        let distortion_plus_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("DistortionPlus pipeline"),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&distortion_plus_bgl],
                    push_constant_ranges: &[],
                })),
                vertex: wgpu::VertexState {
                    module: &dp_shader, entry_point: Some("vs_main"),
                    buffers: &[], compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &dp_shader, entry_point: Some("fs_main"),
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

        // ── Kaleido uniforms + BGL + BG ────────────────────────────────────────
        let kaleido_uniforms_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Kaleido uniforms"),
                contents: bytemuck::cast_slice(&[KaleidoUniforms {
                    resolution_x: w as f32,
                    resolution_y: h as f32,
                    fold_count: 12.0,
                    zoom: 1.0,
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
        let kaleido_bind_group_distorted = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Kaleido BG distorted"),
            layout: &kaleido_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: kaleido_uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&distortion_plus_view),
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

        // ── Applied harmony shared uniform (used by Skin/Image/PrintHead pipelines) ──
        let applied_harmony_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Applied harmony uniforms"),
            contents: bytemuck::cast_slice(&[AppliedHarmonyUniforms {
                enabled: 0, anchor_hue: 210.0, saturation: 0.75, value: 0.85,
                strength: 0.5, offset_count: 3, _pad0: 0.0, _pad1: 0.0,
                offsets: [0.0, -30.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let applied_harmony_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Applied harmony BGL"),
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
        let applied_harmony_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Applied harmony BG"),
            layout: &applied_harmony_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: applied_harmony_buffer.as_entire_binding(),
            }],
        });

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
                    bind_group_layouts: &[&skin_bgl, &applied_harmony_bgl],
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

        // ── AudioPaint + PrintHead painter resources ──────────────────────────
        let painter_audio_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PainterAudio BGL"),
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
        let painter_audio_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PainterAudio buffer"),
                contents: bytemuck::cast_slice(&[PainterAudioUniforms {
                    time_seconds: 0.0, bass: 0.0, mid: 0.0, beat_decay: 0.0,
                    bands: [0.0; 8],
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let painter_audio_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PainterAudio BG"),
            layout: &painter_audio_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: painter_audio_buffer.as_entire_binding(),
            }],
        });
        let audio_painter_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None, bind_group_layouts: &[&painter_audio_bgl], push_constant_ranges: &[],
            });
        let print_head_painter_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PrintHead pipeline layout"),
                bind_group_layouts: &[&painter_audio_bgl, &applied_harmony_bgl],
                push_constant_ranges: &[],
            });
        let ap_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("AudioPaint shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/painter_audio_paint.wgsl").into()),
        });
        let painter_audio_paint_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("AudioPaint pipeline"),
                layout: Some(&audio_painter_layout),
                vertex: wgpu::VertexState {
                    module: &ap_shader, entry_point: Some("vs_main"),
                    buffers: &[], compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &ap_shader, entry_point: Some("fs_main"),
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
        let ph_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PrintHead shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/painter_print_head.wgsl").into()),
        });
        let painter_print_head_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("PrintHead pipeline"),
                layout: Some(&print_head_painter_layout),
                vertex: wgpu::VertexState {
                    module: &ph_shader, entry_point: Some("vs_main"),
                    buffers: &[], compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &ph_shader, entry_point: Some("fs_main"),
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

        // ── Image painter resources ───────────────────────────────────────────
        // Decode the bundled PNG (512×512, 16-bit RGB) to RGBA8 for upload.
        let img_bytes = include_bytes!("../assets/default_painter_image.png");
        let img_rgba  = image::load_from_memory(img_bytes)
            .expect("default_painter_image.png failed to decode")
            .to_rgba8();
        let (img_w, img_h) = img_rgba.dimensions();
        let painter_image_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Painter image texture"),
            size: wgpu::Extent3d { width: img_w, height: img_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &painter_image_texture, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &img_rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(img_w * 4),
                rows_per_image: Some(img_h),
            },
            wgpu::Extent3d { width: img_w, height: img_h, depth_or_array_layers: 1 },
        );
        let painter_image_view =
            painter_image_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let painter_image_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Painter image sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let painter_image_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Painter image BG"),
            layout: &skin_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&painter_image_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&painter_image_sampler),
                },
            ],
        });
        let img_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Image painter shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/painter_image.wgsl").into()),
        });
        let painter_image_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Image painter pipeline"),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None, bind_group_layouts: &[&skin_bgl, &applied_harmony_bgl], push_constant_ranges: &[],
                })),
                vertex: wgpu::VertexState {
                    module: &img_shader, entry_point: Some("vs_main"),
                    buffers: &[], compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &img_shader, entry_point: Some("fs_main"),
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

        // ── Ribbon system ─────────────────────────────────────────────────────────
        let ribbon_uniforms_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Ribbon uniforms"),
                contents: bytemuck::cast_slice(&[RibbonUniforms {
                    resolution:   [PAINTER_TEXTURE_WIDTH as f32, PAINTER_TEXTURE_HEIGHT as f32],
                    time_seconds: 0.0,
                    intensity:    0.5,
                    color:        RIBBON_COLOR,
                    collapse:     [0.0; 4],
                    bands:        [[0.0; 4]; 2],
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let ribbon_bgl = Self::make_uts_bgl(&device, "Ribbon BGL");
        let ribbon_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Ribbon sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let (ribbon_tex_a, ribbon_view_a) = Self::create_ribbon_fbo(&device);
        let (ribbon_tex_b, ribbon_view_b) = Self::create_ribbon_fbo(&device);

        // Clear both ribbon FBOs to transparent black
        {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Ribbon init clear"),
            });
            for view in [&ribbon_view_a, &ribbon_view_b] {
                let _pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Ribbon init clear pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None, timestamp_writes: None,
                });
            }
            queue.submit(std::iter::once(enc.finish()));
        }

        let ribbon_bg_read_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ribbon BG read A"),
            layout: &ribbon_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ribbon_uniforms_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&ribbon_view_a) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&ribbon_sampler) },
            ],
        });
        let ribbon_bg_read_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ribbon BG read B"),
            layout: &ribbon_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ribbon_uniforms_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&ribbon_view_b) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&ribbon_sampler) },
            ],
        });
        let ribbon_composite_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ribbon composite BG A"),
            layout: &skin_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&ribbon_view_a) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&ribbon_sampler) },
            ],
        });
        let ribbon_composite_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ribbon composite BG B"),
            layout: &skin_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&ribbon_view_b) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&ribbon_sampler) },
            ],
        });
        let ribbon_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ribbon shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ribbons.wgsl").into()),
        });
        let ribbon_update_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Ribbon update pipeline"),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None, bind_group_layouts: &[&ribbon_bgl], push_constant_ranges: &[],
                })),
                vertex: wgpu::VertexState {
                    module: &ribbon_shader, entry_point: Some("vs_main"),
                    buffers: &[], compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &ribbon_shader, entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
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
        let ribbon_composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ribbon composite shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ribbon_composite.wgsl").into()),
        });
        let ribbon_composite_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Ribbon composite pipeline"),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None, bind_group_layouts: &[&skin_bgl], push_constant_ranges: &[],
                })),
                vertex: wgpu::VertexState {
                    module: &ribbon_composite_shader, entry_point: Some("vs_main"),
                    buffers: &[], compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &ribbon_composite_shader, entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: PAINTER_FORMAT,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        // ── Blackhole pass v3 resources (conditional pass 5) ────────────────
        // Two ping-pong feedback textures; single shader pass per frame.
        let feedback_tex_a = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Feedback A"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let feedback_view_a = feedback_tex_a.create_view(&wgpu::TextureViewDescriptor::default());

        let feedback_tex_b = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Feedback B"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let feedback_view_b = feedback_tex_b.create_view(&wgpu::TextureViewDescriptor::default());

        let feedback_uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Feedback uniforms"),
            contents: bytemuck::cast_slice(&[FeedbackUniforms {
                center_x: 0.5, center_y: 0.5, shrink_rate: 0.97, strength: 0.0,
                alpha_radius: 0.5, _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let feedback_bgl = Self::make_feedback_bgl(&device);

        // feedback_bg_read_a: prev = texture A (bind group used when write target is B)
        let feedback_bg_read_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback read A BG"),
            layout: &feedback_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: feedback_uniforms_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&feedback_view_a) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&scene_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&shape_sampler) },
            ],
        });

        // feedback_bg_read_b: prev = texture B (bind group used when write target is A)
        let feedback_bg_read_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback read B BG"),
            layout: &feedback_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: feedback_uniforms_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&feedback_view_b) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&scene_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&shape_sampler) },
            ],
        });

        let feedback_blit_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback blit A BG"),
            layout: &blit_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&feedback_view_a) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shape_sampler) },
            ],
        });

        let feedback_blit_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback blit B BG"),
            layout: &blit_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&feedback_view_b) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shape_sampler) },
            ],
        });

        let feedback_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Feedback shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blackhole.wgsl").into()),
        });
        let feedback_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Feedback pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None, bind_group_layouts: &[&feedback_bgl], push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module: &feedback_shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &feedback_shader, entry_point: Some("fs_main"),
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

        // ── Palette pass resources (pass 1c) ────────────────────────────────
        let palette_scratch_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Palette scratch"),
            size: wgpu::Extent3d {
                width: PAINTER_TEXTURE_WIDTH, height: PAINTER_TEXTURE_HEIGHT, depth_or_array_layers: 1,
            },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PAINTER_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let palette_scratch_view = palette_scratch_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let palette_uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Palette uniforms"),
            contents: bytemuck::cast_slice(&[PaletteUniforms {
                mode: 0, tint: 1.0, mono_hue: 200.0,
                harmony_num_offsets: 0, harmony_anchor_hue: 0.0,
                harmony_saturation: 0.75, harmony_value: 0.85, harmony_strength: 0.5,
                harmony_offsets: [0.0; 8],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let palette_bgl = Self::make_uts_bgl(&device, "Palette BGL");
        let palette_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Palette BG"),
            layout: &palette_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: palette_uniforms_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&painter_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&painter_sampler) },
            ],
        });
        let palette_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Palette shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/palette.wgsl").into()),
        });
        let palette_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Palette pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None, bind_group_layouts: &[&palette_bgl], push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module: &palette_shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &palette_shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: PAINTER_FORMAT,
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

        let phantom = phantom::PhantomAlpha::new(&device, surface_format);

        Self {
            surface, device, queue, config, size,
            uniforms_buffer,
            painter_uniforms_bind_group, painter_pipelines,
            painter_texture, painter_view, painter_sampler,
            skin_tex, skin_view, skin_bgl, skin_bind_group, skin_sampler, painter_skin_pipeline,
            painter_audio_bgl, painter_audio_buffer, painter_audio_bind_group,
            painter_audio_paint_pipeline, painter_print_head_pipeline,
            painter_image_texture, painter_image_view, painter_image_sampler,
            painter_image_bind_group, painter_image_pipeline,
            shape_texture, shape_view, shape_depth, shape_depth_view,
            shape_buffers,
            transform_buffer, transform_bind_group,
            shape_pipeline, shape_bind_group,
            shape_effects_buffer, shape_effects_bind_group,
            distortion_plus_texture, distortion_plus_view,
            distortion_plus_uniforms_buffer, distortion_plus_bgl, distortion_plus_bind_group, distortion_plus_pipeline,
            kaleido_texture, kaleido_view,
            kaleido_uniforms_buffer, kaleido_bgl, kaleido_bind_group, kaleido_bind_group_distorted, kaleido_pipeline,
            frame_uniforms_buffer, frame_bgl, frame_bind_group, frame_pipeline,
            shape_sampler,
            scene_texture, scene_view,
            blit_bgl, blit_bind_group, blit_pipeline,
            ribbon_uniforms_buffer, ribbon_bgl, ribbon_sampler,
            ribbon_tex_a, ribbon_view_a, ribbon_tex_b, ribbon_view_b,
            ribbon_ping: false,
            ribbon_bg_read_a, ribbon_bg_read_b,
            ribbon_composite_bg_a, ribbon_composite_bg_b,
            ribbon_update_pipeline, ribbon_composite_pipeline,
            feedback_uniforms_buffer,
            feedback_bgl,
            feedback_tex_a, feedback_view_a,
            feedback_tex_b, feedback_view_b,
            feedback_bg_read_a, feedback_bg_read_b,
            feedback_blit_bg_a, feedback_blit_bg_b,
            feedback_pipeline,
            feedback_current_is_a:  true,
            blackhole_wander_pos:   [0.5, 0.5],
            blackhole_wander_vel:   [0.0, 0.0],
            blackhole_was_enabled:  false,
            palette_uniforms_buffer, palette_bgl, palette_bind_group, palette_pipeline,
            palette_scratch_texture, palette_scratch_view,
            applied_harmony_buffer, applied_harmony_bgl, applied_harmony_bind_group,
            export_ribbon: None,
            export_feedback: None,
            phantom,
            readback_buffer, readback_padded_bytes_per_row,
            recorder: None,
            offline_target: None,
            export_state: None,
            help_overlay,
            start_time: Instant::now(),
            last_frame_time: Instant::now(),
            shake_offset: glam::Vec3::ZERO,
            shake_velocity: glam::Vec3::ZERO,
            bass_zoom_smoothed: 0.0,
            painter_scroll_phase: 0.0,
            skin_source_image: None,
            crop_y_offset: 0.5,
            loaded_audio: None,
            audio_player: None,
            file_rms_baseline: 0.0,
            last_file_beat: Instant::now(),
            last_random_change:    Instant::now(),
            last_reactive_trigger: Instant::now(),
            last_party_trigger:    Instant::now(),
            bass_mid_smoothed:  0.0,
            bass_mid_baseline:  0.0,
            bands_smoothed:     [0.0; 8],
            shader_beat_decay:  0.0,
            last_audio_update:  Instant::now(),
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

        // DP FBO references shape_view → recreate.
        let (dpc, dpv) = Self::create_dp_fbo(&self.device, w, h);
        self.distortion_plus_texture = dpc;
        self.distortion_plus_view = dpv;
        self.distortion_plus_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DistortionPlus BG"),
                layout: &self.distortion_plus_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.distortion_plus_uniforms_buffer.as_entire_binding(),
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

        // Kaleido BG references shape_view → recreate.
        // Kaleido BG distorted references dp_view → recreate.
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
        self.kaleido_bind_group_distorted =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Kaleido BG distorted"),
                layout: &self.kaleido_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.kaleido_uniforms_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.distortion_plus_view),
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
                fold_count: 12.0, zoom: 1.0,
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

        // Recreate blackhole v3 ping-pong feedback textures and bind groups.
        let fb_tex_a = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Feedback A"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let fb_view_a = fb_tex_a.create_view(&wgpu::TextureViewDescriptor::default());

        let fb_tex_b = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Feedback B"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let fb_view_b = fb_tex_b.create_view(&wgpu::TextureViewDescriptor::default());

        self.feedback_bg_read_a = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback read A BG"),
            layout: &self.feedback_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.feedback_uniforms_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&fb_view_a) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.scene_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.shape_sampler) },
            ],
        });
        self.feedback_bg_read_b = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback read B BG"),
            layout: &self.feedback_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.feedback_uniforms_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&fb_view_b) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.scene_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.shape_sampler) },
            ],
        });
        self.feedback_blit_bg_a = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback blit A BG"),
            layout: &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&fb_view_a) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.shape_sampler) },
            ],
        });
        self.feedback_blit_bg_b = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback blit B BG"),
            layout: &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&fb_view_b) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.shape_sampler) },
            ],
        });
        self.feedback_view_a = fb_view_a;
        self.feedback_tex_a  = fb_tex_a;
        self.feedback_view_b = fb_view_b;
        self.feedback_tex_b  = fb_tex_b;
        self.feedback_current_is_a = true;
        self.blackhole_was_enabled = false;

        let (rb, rp) = Self::create_readback_buffer(&self.device, w, h);
        self.readback_buffer = rb;
        self.readback_padded_bytes_per_row = rp;
    }

    fn render_mux_phase_preview(&mut self, menu: Option<(&mut MenuBar, &winit::window::Window)>) {
        if let Ok(swap_frame) = self.surface.get_current_texture() {
            let swap_view = swap_frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
            if let Some(target) = &self.offline_target {
                let preview_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Mux preview BG"),
                    layout: &self.blit_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0,
                            resource: wgpu::BindingResource::TextureView(&target.view) },
                        wgpu::BindGroupEntry { binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.shape_sampler) },
                    ],
                });
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Mux preview blit"),
                });
                {
                    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Mux preview pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &swap_view, resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None, timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.blit_pipeline);
                    pass.set_bind_group(0, &preview_bg, &[]);
                    pass.draw(0..3, 0..1);
                }
                if let Some((menu_bar, window)) = menu {
                    menu_bar.render(
                        &self.device, &self.queue, &mut enc, window,
                        &swap_view, self.size.width, self.size.height, &self.params,
                    );
                }
                self.queue.submit(std::iter::once(enc.finish()));
            }
            swap_frame.present();
        }
    }

    fn render(&mut self, menu: Option<(&mut MenuBar, &winit::window::Window)>) -> Result<(), wgpu::SurfaceError> {
        if self.export_state.is_some() {
            let phase = self.export_state.as_ref().unwrap().phase;
            match phase {
                ExportPhase::Rendering => {
                    self.render_export_frame(menu);
                    return Ok(());
                }
                ExportPhase::Muxing => {
                    let is_done = self.export_state.as_ref()
                        .and_then(|e| e.mux_thread.as_ref())
                        .map(|h| h.is_finished())
                        .unwrap_or(false);
                    if is_done {
                        let handle = self.export_state.as_mut().unwrap().mux_thread.take().unwrap();
                        match handle.join() {
                            Ok(result) if result.success => {
                                log::info!("✓ MP4 saved to {}", result.output_path.display());
                            }
                            Ok(result) => {
                                log::error!("✗ Mux failed: {:?}", result.error_message);
                            }
                            Err(_) => log::error!("Mux thread panicked"),
                        }
                        self.export_state = None;
                        self.export_ribbon = None;
                        self.export_feedback = None;
                        // fall through to live render below
                    } else {
                        self.render_mux_phase_preview(menu);
                        return Ok(());
                    }
                }
                ExportPhase::Complete | ExportPhase::Failed => {
                    self.export_state = None;
                    self.export_ribbon = None;
                    self.export_feedback = None;
                    // fall through to live render below
                }
            }
        }

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

        let shader_time = elapsed.rem_euclid(600.0);
        self.queue.write_buffer(
            &self.uniforms_buffer, 0,
            bytemuck::cast_slice(&[GlobalUniforms {
                time_seconds: shader_time,
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
                time_seconds:         shader_time,
                painter_scroll_phase: self.painter_scroll_phase,
                contrast:        self.params.contrast,
                saturation:      self.params.saturation,
                contrast_passes: self.params.contrast_passes as f32,
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
        // Real-time dt for spring-damper and blackhole cycle.
        // Previously hardcoded to 1/60; now frame-rate-accurate.
        let dt = self.last_frame_time.elapsed().as_secs_f32().clamp(0.001, 0.1);
        self.last_frame_time = Instant::now();
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

        let beat_r = self.params.beat_reactivity * 4.0;
        self.queue.write_buffer(&self.painter_audio_buffer, 0,
            bytemuck::cast_slice(&[PainterAudioUniforms {
                time_seconds: shader_time,
                bass:         self.bands_smoothed[0],
                mid:          self.bands_smoothed[3],
                beat_decay:   (self.shader_beat_decay * beat_r).min(1.0),
                bands:        self.bands_smoothed,
            }]));

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
                pass.set_bind_group(1, &self.applied_harmony_bind_group, &[]);
            } else if self.params.painter_kind == PainterKind::AudioPaint {
                pass.set_pipeline(&self.painter_audio_paint_pipeline);
                pass.set_bind_group(0, &self.painter_audio_bind_group, &[]);
            } else if self.params.painter_kind == PainterKind::PrintHead {
                pass.set_pipeline(&self.painter_print_head_pipeline);
                pass.set_bind_group(0, &self.painter_audio_bind_group, &[]);
                pass.set_bind_group(1, &self.applied_harmony_bind_group, &[]);
            } else if self.params.painter_kind == PainterKind::Image {
                pass.set_pipeline(&self.painter_image_pipeline);
                pass.set_bind_group(0, &self.painter_image_bind_group, &[]);
                pass.set_bind_group(1, &self.applied_harmony_bind_group, &[]);
            } else {
                let painter_pipeline = &self.painter_pipelines[&self.params.painter_kind];
                pass.set_pipeline(painter_pipeline);
                pass.set_bind_group(0, &self.painter_uniforms_bind_group, &[]);
            }
            pass.draw(0..3, 0..1);
        }

        // Pass 1a: ribbon update (ping-pong RGBA16F FBOs)
        // Pass 1b: ribbon composite → painter FBO (alpha blend, LoadOp::Load)
        if self.params.ribbons_enabled {
            let b = self.bands_smoothed;
            // All four rings driven by the same beat_decay this slice.
            // Android has per-ribbon collapse with distinct triggers
            // (see audit Section 1h "Beat-driven collapse animation");
            // a future slice can split these out for per-ring variety.
            self.queue.write_buffer(&self.ribbon_uniforms_buffer, 0,
                bytemuck::cast_slice(&[RibbonUniforms {
                    resolution:   [PAINTER_TEXTURE_WIDTH as f32, PAINTER_TEXTURE_HEIGHT as f32],
                    time_seconds: shader_time,
                    intensity:    self.params.ribbons_intensity,
                    color:        RIBBON_COLOR,
                    collapse:     [(self.shader_beat_decay * beat_r).min(1.0); 4],
                    bands:        [[b[0], b[1], b[2], b[3]], [b[4], b[5], b[6], b[7]]],
                }]));
            let ping = self.ribbon_ping;
            self.ribbon_ping = !self.ribbon_ping;
            {
                let (write_view, read_bg) = if ping {
                    (&self.ribbon_view_b, &self.ribbon_bg_read_a)
                } else {
                    (&self.ribbon_view_a, &self.ribbon_bg_read_b)
                };
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Ribbon update pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: write_view, resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
                });
                pass.set_pipeline(&self.ribbon_update_pipeline);
                pass.set_bind_group(0, read_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            {
                let composite_bg = if ping {
                    &self.ribbon_composite_bg_b   // just wrote B
                } else {
                    &self.ribbon_composite_bg_a   // just wrote A
                };
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Ribbon composite pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.painter_view, resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
                });
                pass.set_pipeline(&self.ribbon_composite_pipeline);
                pass.set_bind_group(0, composite_bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // Write applied-harmony uniforms (consumed by Skin/Image/PrintHead painters).
        {
            let offsets_slice = self.params.color_harmony.hue_offsets();
            let mut offsets = [0.0f32; 8];
            for (i, &o) in offsets_slice.iter().enumerate().take(8) {
                offsets[i] = o;
            }
            self.queue.write_buffer(&self.applied_harmony_buffer, 0,
                bytemuck::cast_slice(&[AppliedHarmonyUniforms {
                    enabled:      if self.params.applied_harmony_enabled { 1 } else { 0 },
                    anchor_hue:   self.params.color_anchor_hue,
                    saturation:   self.params.color_saturation,
                    value:        self.params.color_value,
                    strength:     self.params.color_harmony_strength,
                    offset_count: offsets_slice.len() as u32,
                    _pad0: 0.0, _pad1: 0.0,
                    offsets,
                }]));
        }

        // Pass 1c: palette clamp (optional, skipped when Off)
        if self.params.palette_mode != PaletteMode::Off {
            let h_offsets = self.params.color_harmony.hue_offsets();
            let h_anchor  = self.params.color_anchor_hue;
            let mut ho = [0.0f32; 8];
            for (i, &off) in h_offsets.iter().enumerate().take(8) {
                ho[i] = color::wrap_hue(h_anchor + off);
            }
            self.queue.write_buffer(&self.palette_uniforms_buffer, 0,
                bytemuck::cast_slice(&[PaletteUniforms {
                    mode:                self.params.palette_mode.to_u32(),
                    tint:                self.params.palette_tint,
                    mono_hue:            self.params.palette_mono_hue,
                    harmony_num_offsets: h_offsets.len().min(8) as u32,
                    harmony_anchor_hue:  h_anchor,
                    harmony_saturation:  self.params.color_saturation,
                    harmony_value:       self.params.color_value,
                    harmony_strength:    self.params.color_harmony_strength,
                    harmony_offsets:     ho,
                }]));
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Palette pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.palette_scratch_view, resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
                });
                pass.set_pipeline(&self.palette_pipeline);
                pass.set_bind_group(0, &self.palette_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.palette_scratch_texture,
                    mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.painter_texture,
                    mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: PAINTER_TEXTURE_WIDTH, height: PAINTER_TEXTURE_HEIGHT, depth_or_array_layers: 1,
                },
            );
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

        // Pass 2.5: distortion plus (optional equirectangular rotation → dp FBO)
        if self.params.distortion_plus_enabled {
            self.queue.write_buffer(
                &self.distortion_plus_uniforms_buffer, 0,
                bytemuck::cast_slice(&[DistortionPlusUniforms {
                    yaw:   self.params.distortion_plus_yaw.to_radians(),
                    pitch: self.params.distortion_plus_pitch.to_radians(),
                    roll:  self.params.distortion_plus_roll.to_radians(),
                    _pad:  0.0,
                }]),
            );
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("DistortionPlus pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.distortion_plus_view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None, timestamp_writes: None,
            });
            pass.set_pipeline(&self.distortion_plus_pipeline);
            pass.set_bind_group(0, &self.distortion_plus_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 3: kaleido fold → kaleido FBO
        {
            let kaleido_bg = if self.params.distortion_plus_enabled {
                &self.kaleido_bind_group_distorted
            } else {
                &self.kaleido_bind_group
            };
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
            pass.set_bind_group(0, kaleido_bg, &[]);
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

        // Pass 5: blit scene FBO → swapchain.
        // Priority: blackhole > phantom > plain blit. Blackhole and phantom are mutually exclusive.
        if self.params.blackhole_enabled {
            // ── Blackhole v3: continuous video-feedback ping-pong ─────────────
            let just_enabled = !self.blackhole_was_enabled;
            self.blackhole_was_enabled = true;

            // Spring-damper wander (Ornstein-Uhlenbeck): stochastic, smooth, bounded.
            {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let impulse: f32 = 0.001;
                self.blackhole_wander_vel[0] += rng.gen_range(-impulse..=impulse);
                self.blackhole_wander_vel[1] += rng.gen_range(-impulse..=impulse);
            }
            self.blackhole_wander_vel[0] *= 0.95;
            self.blackhole_wander_vel[1] *= 0.95;
            self.blackhole_wander_vel[0] -= (self.blackhole_wander_pos[0] - 0.5) * 0.02;
            self.blackhole_wander_vel[1] -= (self.blackhole_wander_pos[1] - 0.5) * 0.02;
            self.blackhole_wander_pos[0] += self.blackhole_wander_vel[0];
            self.blackhole_wander_pos[1] += self.blackhole_wander_vel[1];
            let wa = self.params.blackhole_wander_amount;
            self.blackhole_wander_pos[0] = self.blackhole_wander_pos[0].clamp(0.5 - wa, 0.5 + wa);
            self.blackhole_wander_pos[1] = self.blackhole_wander_pos[1].clamp(0.5 - wa, 0.5 + wa);

            // strength = 0 on first enable → output = live, regardless of uninitialized prev.
            let effective_strength = if just_enabled { 0.0 } else { self.params.blackhole_warp_strength };

            self.queue.write_buffer(
                &self.feedback_uniforms_buffer, 0,
                bytemuck::cast_slice(&[FeedbackUniforms {
                    center_x:     self.blackhole_wander_pos[0],
                    center_y:     self.blackhole_wander_pos[1],
                    shrink_rate:  self.params.blackhole_warp_curve,
                    strength:     effective_strength,
                    alpha_radius: self.params.blackhole_alpha_radius,
                    _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
                }]),
            );

            // Pick write target and matching prev/blit bind groups.
            let (write_view, read_bg, blit_bg) = if self.feedback_current_is_a {
                (&self.feedback_view_a, &self.feedback_bg_read_b, &self.feedback_blit_bg_a)
            } else {
                (&self.feedback_view_b, &self.feedback_bg_read_a, &self.feedback_blit_bg_b)
            };

            // Feedback pass: mix(live, prev) → write target (REPLACE blend).
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Feedback pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: write_view, resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None, timestamp_writes: None,
                });
                pass.set_pipeline(&self.feedback_pipeline);
                pass.set_bind_group(0, read_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            // Blit written feedback texture → swapchain.
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Feedback blit"),
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
                pass.set_bind_group(0, blit_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            self.feedback_current_is_a = !self.feedback_current_is_a;
        } else if self.params.phantom_enabled {
            let scene_view_for_capture = self.scene_texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.phantom.capture(&mut encoder, &self.device, &scene_view_for_capture);
            self.phantom.composite(
                &mut encoder, &self.device, &self.queue,
                &scene_view_for_capture, &screen_view,
                self.params.phantom_delay_seconds,
                self.params.phantom_key_color,
                self.params.phantom_key_tolerance,
                self.params.phantom_key_softness,
                self.params.phantom_key_strength,
                self.params.phantom_opacity,
            );
            self.blackhole_was_enabled = false;
        } else {
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
            self.blackhole_was_enabled = false;
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

        self.phantom.advance_frame();
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
        self.shake_velocity += dir * strength * (self.params.beat_reactivity * 4.0) * 0.75;
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

    pub fn update_modes(&mut self, bass: f32, mid: f32, frame_dt: Option<f32>) {
        let bass_mid = bass + mid;
        let smooth_alpha   = 0.3_f32;
        let baseline_alpha = 0.005_f32;
        self.bass_mid_smoothed = self.bass_mid_smoothed * (1.0 - smooth_alpha)  + bass_mid * smooth_alpha;
        self.bass_mid_baseline = self.bass_mid_baseline * (1.0 - baseline_alpha) + bass_mid * baseline_alpha;

        match frame_dt {
            Some(dt) => {
                // Export path: f32 accumulators instead of Instant for deterministic replay.
                if self.export_state.is_none() { return; }
                if self.params.random_mode_enabled {
                    let interval = 5.0 - 4.8 * self.params.random_mode_aggressiveness;
                    let elapsed = {
                        let exp = self.export_state.as_mut().unwrap();
                        exp.export_random_elapsed += dt;
                        exp.export_random_elapsed
                    };
                    if elapsed >= interval {
                        self.export_state.as_mut().unwrap().export_random_elapsed = 0.0;
                        self.randomize_all_params();
                        log::info!("Random Mode: full reroll (interval {:.1}s)", interval);
                    }
                }
                if self.params.reactive_mode_enabled {
                    let threshold_mult = lerp(2.5, 1.05, self.params.reactive_mode_aggressiveness);
                    let cooldown       = lerp(8.0, 0.5,  self.params.reactive_mode_aggressiveness);
                    let elapsed = {
                        let exp = self.export_state.as_mut().unwrap();
                        exp.export_reactive_elapsed += dt;
                        exp.export_reactive_elapsed
                    };
                    let trigger_level = self.bass_mid_baseline * threshold_mult;
                    if elapsed >= cooldown && self.bass_mid_smoothed > trigger_level && self.bass_mid_baseline > 0.001 {
                        self.export_state.as_mut().unwrap().export_reactive_elapsed = 0.0;
                        self.params.painter_kind = self.params.painter_kind.next();
                        log::info!("Reactive Mode: painter -> {}", self.params.painter_kind.name());
                    }
                }
                if self.params.party_mode_enabled {
                    let threshold_mult = lerp(2.0, 1.0, self.params.party_mode_aggressiveness);
                    let cooldown       = lerp(4.0, 0.3, self.params.party_mode_aggressiveness);
                    let elapsed = {
                        let exp = self.export_state.as_mut().unwrap();
                        exp.export_party_elapsed += dt;
                        exp.export_party_elapsed
                    };
                    let trigger_level = self.bass_mid_baseline * threshold_mult;
                    if elapsed >= cooldown && self.bass_mid_smoothed > trigger_level && self.bass_mid_baseline > 0.001 {
                        self.export_state.as_mut().unwrap().export_party_elapsed = 0.0;
                        self.randomize_all_params();
                        log::info!("Party Mode: full reroll on beat");
                    }
                }
            }
            None => {
                // Live path: wall-clock Instant (unchanged behavior).
                let now = Instant::now();
                if self.params.random_mode_enabled {
                    let interval = 5.0 - 4.8 * self.params.random_mode_aggressiveness;
                    if now.duration_since(self.last_random_change).as_secs_f32() >= interval {
                        self.randomize_all_params();
                        self.last_random_change = now;
                        log::info!("Random Mode: full reroll (interval {:.1}s)", interval);
                    }
                }
                if self.params.reactive_mode_enabled {
                    let threshold_mult = lerp(2.5, 1.05, self.params.reactive_mode_aggressiveness);
                    let cooldown       = lerp(8.0, 0.5,  self.params.reactive_mode_aggressiveness);
                    if now.duration_since(self.last_reactive_trigger).as_secs_f32() >= cooldown {
                        let trigger_level = self.bass_mid_baseline * threshold_mult;
                        if self.bass_mid_smoothed > trigger_level && self.bass_mid_baseline > 0.001 {
                            self.params.painter_kind = self.params.painter_kind.next();
                            self.last_reactive_trigger = now;
                            log::info!("Reactive Mode: painter -> {}", self.params.painter_kind.name());
                        }
                    }
                }
                if self.params.party_mode_enabled {
                    let threshold_mult = lerp(2.0, 1.0, self.params.party_mode_aggressiveness);
                    let cooldown       = lerp(4.0, 0.3, self.params.party_mode_aggressiveness);
                    if now.duration_since(self.last_party_trigger).as_secs_f32() >= cooldown {
                        let trigger_level = self.bass_mid_baseline * threshold_mult;
                        if self.bass_mid_smoothed > trigger_level && self.bass_mid_baseline > 0.001 {
                            self.randomize_all_params();
                            self.last_party_trigger = now;
                            log::info!("Party Mode: full reroll on beat");
                        }
                    }
                }
            }
        }
    }

    pub fn start_export(&mut self) -> Result<(), String> {
        if self.export_state.is_some() {
            return Err("export already in progress".into());
        }
        let duration = self.loaded_audio.as_ref()
            .ok_or_else(|| "no audio loaded — File → Open Audio first".to_string())?
            .duration_seconds;
        let (off_w, off_h) = self.offline_target.as_ref()
            .map(|t| (t.width, t.height))
            .ok_or_else(|| "no offline target".to_string())?;

        if let Some(player) = &self.audio_player {
            if player.is_playing() { player.pause(); }
        }

        let fps = self.params.export_framerate.fps();
        let total_frames = (duration * fps as f32).ceil() as u32;

        let timestamp = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("abstrakt-deck")
            .join("exports");
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| format!("create cache dir failed: {}", e))?;
        let output_dir = cache_dir.join(format!("export-{}", timestamp));
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| format!("create export dir failed: {}", e))?;

        let (frame_save_sender, frame_save_thread) = spawn_frame_save_worker();

        log::info!(
            "Export started: {} frames at {}fps, {}×{}, output → {}",
            total_frames, fps, off_w, off_h, output_dir.display()
        );

        self.shake_offset   = glam::Vec3::ZERO;
        self.shake_velocity = glam::Vec3::ZERO;
        self.file_rms_baseline = 0.0;

        self.export_state = Some(ExportState {
            phase: ExportPhase::Rendering,
            current_frame: 0,
            total_frames,
            output_dir,
            fps,
            start_time: Instant::now(),
            frame_save_sender: Some(frame_save_sender),
            frame_save_thread: Some(frame_save_thread),
            mux_thread: None,
            export_bands_smoothed: [0.0; 8],
            export_rms_smoothed:   0.0,
            offline_analyzer:      audio::OfflineAnalyzer::new(),
            // Zero-initialized: each mode fires relative to export start, not wall clock.
            export_random_elapsed:   0.0,
            export_reactive_elapsed: 0.0,
            export_party_elapsed:    0.0,
        });

        // Initialize export ribbon FBOs, cleared to transparent black
        {
            let (tex_a, view_a) = Self::create_ribbon_fbo(&self.device);
            let (tex_b, view_b) = Self::create_ribbon_fbo(&self.device);
            {
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Export ribbon init clear"),
                });
                for v in [&view_a, &view_b] {
                    let _pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Export ribbon clear"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: v, resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
                    });
                }
                self.queue.submit(std::iter::once(enc.finish()));
            }
            let bg_read_a = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Export ribbon BG read A"),
                layout: &self.ribbon_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.ribbon_uniforms_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view_a) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.ribbon_sampler) },
                ],
            });
            let bg_read_b = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Export ribbon BG read B"),
                layout: &self.ribbon_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.ribbon_uniforms_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view_b) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.ribbon_sampler) },
                ],
            });
            let composite_bg_a = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Export ribbon composite BG A"),
                layout: &self.skin_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view_a) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.ribbon_sampler) },
                ],
            });
            let composite_bg_b = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Export ribbon composite BG B"),
                layout: &self.skin_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view_b) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.ribbon_sampler) },
                ],
            });
            self.export_ribbon = Some(ExportRibbonState {
                tex_a, view_a, tex_b, view_b,
                ping: false,
                bg_read_a, bg_read_b,
                composite_bg_a, composite_bg_b,
            });
        }

        // Allocate export feedback ping-pong FBOs at offline_target resolution if blackhole enabled.
        if self.params.blackhole_enabled {
            use rand::SeedableRng;
            let fmt = self.offline_target.as_ref().unwrap().format;
            let make_fb_tex = |device: &wgpu::Device, label: &'static str| {
                let t = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: off_w, height: off_h, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: fmt,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let v = t.create_view(&wgpu::TextureViewDescriptor::default());
                (t, v)
            };
            let (tex_a, view_a) = make_fb_tex(&self.device, "Export feedback A");
            let (tex_b, view_b) = make_fb_tex(&self.device, "Export feedback B");
            // Clear both to opaque black so frame 0 shows live scene only.
            {
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Export feedback init clear"),
                });
                for v in [&view_a, &view_b] {
                    let _pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Export feedback clear"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: v, resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
                    });
                }
                self.queue.submit(std::iter::once(enc.finish()));
            }
            let offline_view = &self.offline_target.as_ref().unwrap().view;
            // bg_a: prev=A, scene=offline_target — bind group used when writing to B.
            let bg_a = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Export feedback BG A"),
                layout: &self.feedback_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.feedback_uniforms_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view_a) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(offline_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.shape_sampler) },
                ],
            });
            // bg_b: prev=B, scene=offline_target — bind group used when writing to A.
            let bg_b = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Export feedback BG B"),
                layout: &self.feedback_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.feedback_uniforms_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view_b) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(offline_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.shape_sampler) },
                ],
            });
            let blit_bg_a = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Export feedback blit BG A"),
                layout: &self.blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view_a) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.shape_sampler) },
                ],
            });
            let blit_bg_b = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Export feedback blit BG B"),
                layout: &self.blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view_b) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.shape_sampler) },
                ],
            });
            self.export_feedback = Some(ExportFeedbackState {
                tex_a, view_a, tex_b, view_b,
                bg_a, bg_b, blit_bg_a, blit_bg_b,
                current_is_a: true,
                wander_pos: [0.5, 0.5],
                wander_vel: [0.0, 0.0],
                // Deterministic seed: same wander path on every export of the same session.
                rng: rand::rngs::StdRng::seed_from_u64(0x600D_5EED_0000_0000),
            });
        }
        Ok(())
    }

    fn render_export_frame(&mut self, menu: Option<(&mut MenuBar, &winit::window::Window)>) {
        // ── Phase 1: snapshot state, check completion ─────────────────────────
        let (frame_index, total, fps, output_dir, sender) = {
            let e = self.export_state.as_ref().unwrap();
            (e.current_frame, e.total_frames, e.fps,
             e.output_dir.clone(), e.frame_save_sender.as_ref().unwrap().clone())
        };

        if frame_index >= total {
            let render_elapsed = self.export_state.as_ref().unwrap().start_time.elapsed().as_secs_f32();
            log::info!(
                "Render complete: {} frames in {:.1}s — starting mux",
                total, render_elapsed
            );

            // Show save dialog for output path
            let default_name = format!(
                "abstrakt-deck-{}.mp4",
                chrono::Local::now().format("%Y%m%d-%H%M%S")
            );
            let chosen_path = rfd::FileDialog::new()
                .set_title("Save MP4 export")
                .add_filter("MP4 video", &["mp4"])
                .set_file_name(&default_name)
                .save_file();

            let output_path = match chosen_path {
                Some(p) => p,
                None => {
                    log::warn!("Save canceled — PNG frames kept at {}", output_dir.display());
                    self.export_state = None;
                    self.export_feedback = None;
                    return;
                }
            };

            // Drop sender to signal PNG worker to drain and exit
            let worker_handle = {
                let exp = self.export_state.as_mut().unwrap();
                drop(exp.frame_save_sender.take());
                exp.frame_save_thread.take()
            };

            let png_dir = output_dir.clone();
            let audio_path = self.loaded_audio.as_ref().unwrap().source_path.clone();
            let mux_handle = std::thread::spawn(move || {
                // Wait for all PNG frames to be written before muxing
                if let Some(h) = worker_handle { let _ = h.join(); }
                run_ffmpeg_mux(&png_dir, &audio_path, &output_path, fps)
            });

            let exp = self.export_state.as_mut().unwrap();
            exp.phase = ExportPhase::Muxing;
            exp.mux_thread = Some(mux_handle);
            return;
        }

        let frame_time = frame_index as f32 / fps as f32;

        // ── Phase 2: audio energies at this frame (with EMA smoothing) ──────────
        // Raw FFT windows at 60fps shift by ~1400 samples with only ~648 overlap,
        // causing jitter. Alpha=0.25 gives a ~4-frame (~67ms) smoothing window.
        let (bass, mid, rms) = {
            let dt = 1.0 / fps as f32;
            let (raw_bands, raw_r) = if let Some(audio) = self.loaded_audio.clone() {
                let ch = audio.channels as usize;
                let pos = (frame_time * audio.sample_rate as f32) as usize;
                let window = 2048usize;
                let start = pos.saturating_sub(window / 2) * ch;
                let end = (start + window * ch).min(audio.samples.len());
                if start < end && ch > 0 {
                    let mut mono = Vec::with_capacity(window);
                    let mut i = start;
                    while i + ch <= end {
                        let mut sum = 0.0f32;
                        for c in 0..ch { sum += audio.samples[i + c]; }
                        mono.push(sum / ch as f32);
                        i += ch;
                    }
                    let r = (mono.iter().map(|x| x * x).sum::<f32>() / mono.len() as f32)
                        .sqrt().min(1.0);
                    let exp = self.export_state.as_mut().unwrap();
                    let (bands, _) = exp.offline_analyzer.analyze_frame(&mono, audio.sample_rate, dt);
                    (bands, r)
                } else {
                    let exp = self.export_state.as_mut().unwrap();
                    exp.offline_analyzer.analyze_frame(&[], audio.sample_rate, dt);
                    ([0.0f32; 8], 0.0)
                }
            } else { ([0.0f32; 8], 0.0) };

            const ALPHA: f32 = 0.25;
            let exp = self.export_state.as_mut().unwrap();
            for (i, &raw) in raw_bands.iter().enumerate() {
                exp.export_bands_smoothed[i] =
                    exp.export_bands_smoothed[i] * (1.0 - ALPHA) + raw * ALPHA;
            }
            exp.export_rms_smoothed = exp.export_rms_smoothed * (1.0 - ALPHA) + raw_r * ALPHA;
            (exp.export_bands_smoothed[0], exp.export_bands_smoothed[3], exp.export_rms_smoothed)
        };

        // ── Phase 3: state updates (needs &mut self) ──────────────────────────
        self.update_bass_zoom(bass);
        self.update_modes(bass, mid, Some(1.0 / fps as f32));

        // Beat-shake detection for the export timeline
        {
            let threshold = self.file_rms_baseline * 1.5 + 0.01;
            if self.params.audio_shake_enabled
                && rms > threshold
                && self.last_file_beat.elapsed().as_millis() > 120
            {
                let strength = ((rms / (threshold + 0.001)).min(3.0) / 3.0).clamp(0.0, 1.0);
                self.kick_shake(strength);
                self.last_file_beat = Instant::now();
            }
            self.file_rms_baseline = self.file_rms_baseline * 0.95 + rms * 0.05;
        }

        // Spring-damping shake physics, stepped by 1/fps
        let dt = 1.0 / fps as f32;
        let force = -30.0 * self.shake_offset - 8.0 * self.shake_velocity;
        self.shake_velocity += force * dt;
        self.shake_offset   += self.shake_velocity * dt;

        // ── Phase 4: uniforms with export resolution + frame_time ─────────────
        let (off_w, off_h) = {
            let t = self.offline_target.as_ref().unwrap();
            (t.width, t.height)
        };

        let shape = self.params.current_shape;
        let ang_vel = std::f32::consts::TAU / shape.rotation_period_seconds();
        let rot_rad = frame_time * ang_vel * self.params.rotation_speed_scale;
        self.painter_scroll_phase = (rot_rad / std::f32::consts::TAU * 0.25).fract();

        let angle = rot_rad;
        let axis  = glam::Vec3::from_array(shape.rotation_axis()).normalize();
        let model = glam::Mat4::from_translation(self.shake_offset)
            * glam::Mat4::from_axis_angle(axis, angle)
            * glam::Mat4::from_scale(glam::Vec3::splat(shape.model_scale()));
        let aspect = off_w as f32 / off_h as f32;
        let proj = glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
        let cam  = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.5, 3.0), glam::Vec3::ZERO, glam::Vec3::Y,
        );

        let shader_time = frame_time.rem_euclid(600.0);
        self.queue.write_buffer(&self.uniforms_buffer, 0,
            bytemuck::cast_slice(&[GlobalUniforms {
                time_seconds: shader_time,
                resolution_x: off_w as f32, resolution_y: off_h as f32, _pad: 0.0,
            }]));
        self.queue.write_buffer(&self.kaleido_uniforms_buffer, 0,
            bytemuck::cast_slice(&[KaleidoUniforms {
                resolution_x: off_w as f32, resolution_y: off_h as f32,
                fold_count: self.params.fold_count,
                zoom: self.params.zoom * self.params.current_shape.kaleido_zoom()
                    + self.bass_zoom_smoothed * self.params.bass_zoom_strength,
            }]));
        self.queue.write_buffer(&self.shape_effects_buffer, 0,
            bytemuck::cast_slice(&[ShapeEffects {
                invert:               if self.params.invert_enabled   { 1.0 } else { 0.0 },
                colorize_enabled:     if self.params.colorize_enabled { 1.0 } else { 0.0 },
                colorize_hue:         self.params.colorize_hue,
                colorize_intensity:   self.params.colorize_intensity,
                distortion_enabled:   if self.params.distortion_enabled { 1.0 } else { 0.0 },
                distortion_amplitude: self.params.distortion_amplitude,
                distortion_frequency: self.params.distortion_frequency,
                time_seconds:         shader_time,
                painter_scroll_phase: self.painter_scroll_phase,
                contrast:        self.params.contrast,
                saturation:      self.params.saturation,
                contrast_passes: self.params.contrast_passes as f32,
            }]));
        let (fr, fg, fb) = hsv_to_rgb(self.params.frame_color_hue, 0.85, 1.0);
        self.queue.write_buffer(&self.frame_uniforms_buffer, 0,
            bytemuck::cast_slice(&[FrameUniforms {
                resolution_x: off_w as f32, resolution_y: off_h as f32,
                frame_color_r: fr, frame_color_g: fg, frame_color_b: fb, frame_color_a: 1.0,
                frame_shape: self.params.frame_shape.as_f32(),
                frame_size:  self.params.frame_size,
            }]));
        self.queue.write_buffer(&self.transform_buffer, 0,
            bytemuck::cast_slice(&[Transform { mvp: (proj * cam * model).to_cols_array_2d() }]));

        // ── Phase 5: temporary export-resolution FBOs + bind groups ──────────
        let (_exp_shape_tex, exp_shape_view, _exp_shape_depth_tex, exp_shape_depth_view) =
            Self::create_shape_fbo(&self.device, off_w, off_h);
        let (_exp_kaleido_tex, exp_kaleido_view) =
            Self::create_kaleido_fbo(&self.device, off_w, off_h);

        // Per-frame dp FBO (must be at export resolution, not screen resolution).
        let (_exp_dp_tex, exp_dp_view) = if self.params.distortion_plus_enabled {
            let (t, v) = Self::create_dp_fbo(&self.device, off_w, off_h);
            (Some(t), Some(v))
        } else {
            (None, None)
        };

        let kaleido_input_view = exp_dp_view.as_ref().unwrap_or(&exp_shape_view);
        let exp_kaleido_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Kaleido BG"),
            layout: &self.kaleido_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0,
                    resource: self.kaleido_uniforms_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1,
                    resource: wgpu::BindingResource::TextureView(kaleido_input_view) },
                wgpu::BindGroupEntry { binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.shape_sampler) },
            ],
        });
        let exp_frame_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Frame BG"),
            layout: &self.frame_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0,
                    resource: self.frame_uniforms_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1,
                    resource: wgpu::BindingResource::TextureView(&exp_kaleido_view) },
                wgpu::BindGroupEntry { binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.shape_sampler) },
            ],
        });

        // ── Phase 6: render passes ────────────────────────────────────────────
        {
            let exp = self.export_state.as_ref().unwrap();
            let eb = exp.export_bands_smoothed;
            let bd = exp.offline_analyzer.beat_decay;
            let beat_r = self.params.beat_reactivity * 4.0;
            self.queue.write_buffer(&self.painter_audio_buffer, 0,
                bytemuck::cast_slice(&[PainterAudioUniforms {
                    time_seconds: shader_time,
                    bass:         eb[0],
                    mid:          eb[3],
                    beat_decay:   (bd * beat_r).min(1.0),
                    bands:        eb,
                }]));
        }

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Export frame encoder"),
        });

        // Pass 1: painter (unchanged — 4096×256 is resolution-independent)
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export painter pass"),
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
                pass.set_bind_group(1, &self.applied_harmony_bind_group, &[]);
            } else if self.params.painter_kind == PainterKind::AudioPaint {
                pass.set_pipeline(&self.painter_audio_paint_pipeline);
                pass.set_bind_group(0, &self.painter_audio_bind_group, &[]);
            } else if self.params.painter_kind == PainterKind::PrintHead {
                pass.set_pipeline(&self.painter_print_head_pipeline);
                pass.set_bind_group(0, &self.painter_audio_bind_group, &[]);
                pass.set_bind_group(1, &self.applied_harmony_bind_group, &[]);
            } else if self.params.painter_kind == PainterKind::Image {
                pass.set_pipeline(&self.painter_image_pipeline);
                pass.set_bind_group(0, &self.painter_image_bind_group, &[]);
                pass.set_bind_group(1, &self.applied_harmony_bind_group, &[]);
            } else {
                let p = &self.painter_pipelines[&self.params.painter_kind];
                pass.set_pipeline(p);
                pass.set_bind_group(0, &self.painter_uniforms_bind_group, &[]);
            }
            pass.draw(0..3, 0..1);
        }

        // Pass 1a/1b: ribbon update + composite (export path)
        if self.params.ribbons_enabled {
            let exp_beat = self.export_state.as_ref()
                .map(|e| e.offline_analyzer.beat_decay)
                .unwrap_or(0.0);
            let eb = self.export_state.as_ref()
                .map(|e| e.export_bands_smoothed)
                .unwrap_or([0.0; 8]);
            let beat_r = self.params.beat_reactivity * 4.0;
            // All four rings driven by the same beat_decay this slice.
            // Android has per-ribbon collapse with distinct triggers
            // (see audit Section 1h "Beat-driven collapse animation");
            // a future slice can split these out for per-ring variety.
            self.queue.write_buffer(&self.ribbon_uniforms_buffer, 0,
                bytemuck::cast_slice(&[RibbonUniforms {
                    resolution:   [PAINTER_TEXTURE_WIDTH as f32, PAINTER_TEXTURE_HEIGHT as f32],
                    time_seconds: shader_time,
                    intensity:    self.params.ribbons_intensity,
                    color:        RIBBON_COLOR,
                    collapse:     [(exp_beat * beat_r).min(1.0); 4],
                    bands:        [[eb[0], eb[1], eb[2], eb[3]], [eb[4], eb[5], eb[6], eb[7]]],
                }]));
            // Capture and flip ping state; None means no export ribbon allocated → skip
            let ping_opt = if let Some(r) = self.export_ribbon.as_mut() {
                let p = r.ping;
                r.ping = !r.ping;
                Some(p)
            } else {
                None
            };
            if let (Some(ping), Some(exp_r)) = (ping_opt, &self.export_ribbon) {
                {
                    let (write_view, read_bg) = if ping {
                        (&exp_r.view_b, &exp_r.bg_read_a)
                    } else {
                        (&exp_r.view_a, &exp_r.bg_read_b)
                    };
                    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Export ribbon update pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: write_view, resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.ribbon_update_pipeline);
                    pass.set_bind_group(0, read_bg, &[]);
                    pass.draw(0..3, 0..1);
                }
                {
                    // composite the freshly-written texture onto painter_view
                    // (painter_view is shared with live path — cross-contamination accepted for export)
                    let composite_bg = if ping { &exp_r.composite_bg_b } else { &exp_r.composite_bg_a };
                    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Export ribbon composite pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.painter_view, resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.ribbon_composite_pipeline);
                    pass.set_bind_group(0, composite_bg, &[]);
                    pass.draw(0..3, 0..1);
                }
            }
        }

        // Write applied-harmony uniforms for export path.
        {
            let offsets_slice = self.params.color_harmony.hue_offsets();
            let mut offsets = [0.0f32; 8];
            for (i, &o) in offsets_slice.iter().enumerate().take(8) {
                offsets[i] = o;
            }
            self.queue.write_buffer(&self.applied_harmony_buffer, 0,
                bytemuck::cast_slice(&[AppliedHarmonyUniforms {
                    enabled:      if self.params.applied_harmony_enabled { 1 } else { 0 },
                    anchor_hue:   self.params.color_anchor_hue,
                    saturation:   self.params.color_saturation,
                    value:        self.params.color_value,
                    strength:     self.params.color_harmony_strength,
                    offset_count: offsets_slice.len() as u32,
                    _pad0: 0.0, _pad1: 0.0,
                    offsets,
                }]));
        }

        // Pass 1c: palette clamp for export path
        if self.params.palette_mode != PaletteMode::Off {
            let h_offsets = self.params.color_harmony.hue_offsets();
            let h_anchor  = self.params.color_anchor_hue;
            let mut ho = [0.0f32; 8];
            for (i, &off) in h_offsets.iter().enumerate().take(8) {
                ho[i] = color::wrap_hue(h_anchor + off);
            }
            self.queue.write_buffer(&self.palette_uniforms_buffer, 0,
                bytemuck::cast_slice(&[PaletteUniforms {
                    mode:                self.params.palette_mode.to_u32(),
                    tint:                self.params.palette_tint,
                    mono_hue:            self.params.palette_mono_hue,
                    harmony_num_offsets: h_offsets.len().min(8) as u32,
                    harmony_anchor_hue:  h_anchor,
                    harmony_saturation:  self.params.color_saturation,
                    harmony_value:       self.params.color_value,
                    harmony_strength:    self.params.color_harmony_strength,
                    harmony_offsets:     ho,
                }]));
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Export palette pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.palette_scratch_view, resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
                });
                pass.set_pipeline(&self.palette_pipeline);
                pass.set_bind_group(0, &self.palette_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            enc.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.palette_scratch_texture,
                    mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.painter_texture,
                    mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: PAINTER_TEXTURE_WIDTH, height: PAINTER_TEXTURE_HEIGHT, depth_or_array_layers: 1,
                },
            );
        }

        // Pass 2: shape → export-res shape view
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export shape pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &exp_shape_view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &exp_shape_depth_view,
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
            let bufs = &self.shape_buffers[&self.params.current_shape];
            pass.set_vertex_buffer(0, bufs.vertex_buffer.slice(..));
            pass.set_index_buffer(bufs.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..bufs.index_count, 0, 0..1);
        }

        // Pass 2.5: distortion plus → export-res dp view (optional)
        if let Some(ref dp_view) = exp_dp_view {
            let exp_dp_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Export DP BG"),
                layout: &self.distortion_plus_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0,
                        resource: self.distortion_plus_uniforms_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1,
                        resource: wgpu::BindingResource::TextureView(&exp_shape_view) },
                    wgpu::BindGroupEntry { binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.shape_sampler) },
                ],
            });
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export DistortionPlus pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dp_view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None, timestamp_writes: None,
            });
            pass.set_pipeline(&self.distortion_plus_pipeline);
            pass.set_bind_group(0, &exp_dp_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 3: kaleido → export-res kaleido view
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export kaleido pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &exp_kaleido_view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None, timestamp_writes: None,
            });
            pass.set_pipeline(&self.kaleido_pipeline);
            pass.set_bind_group(0, &exp_kaleido_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 4: frame overlay → offline target
        {
            let offline_view = &self.offline_target.as_ref().unwrap().view;
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export frame pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: offline_view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None, timestamp_writes: None,
            });
            pass.set_pipeline(&self.frame_pipeline);
            pass.set_bind_group(0, &exp_frame_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 5 (export path): blackhole video-feedback + blit-back to offline_target.
        // Two-step: feedback(prev, scene=offline_target) → inactive_fb, then blit inactive_fb →
        // offline_target. Can't render feedback directly to offline_target while reading it.
        if self.params.blackhole_enabled {
            // Wander update (mutable borrow of export_feedback, released at end of block).
            let maybe_fb = if let Some(ef) = self.export_feedback.as_mut() {
                use rand::Rng;
                let wa = self.params.blackhole_wander_amount;
                ef.wander_vel[0] += ef.rng.gen_range(-0.001_f32..=0.001);
                ef.wander_vel[1] += ef.rng.gen_range(-0.001_f32..=0.001);
                ef.wander_vel[0] *= 0.95;
                ef.wander_vel[1] *= 0.95;
                ef.wander_vel[0] -= (ef.wander_pos[0] - 0.5) * 0.02;
                ef.wander_vel[1] -= (ef.wander_pos[1] - 0.5) * 0.02;
                ef.wander_pos[0] += ef.wander_vel[0];
                ef.wander_pos[1] += ef.wander_vel[1];
                ef.wander_pos[0] = ef.wander_pos[0].clamp(0.5 - wa, 0.5 + wa);
                ef.wander_pos[1] = ef.wander_pos[1].clamp(0.5 - wa, 0.5 + wa);
                // strength = 0 on frame 0: prev_tex is cleared black, show live scene only.
                let eff = if frame_index == 0 { 0.0 } else { self.params.blackhole_warp_strength };
                Some((ef.wander_pos, ef.current_is_a, eff))
            } else {
                None
            }; // mutable borrow of self.export_feedback released here

            if let Some((center, current_is_a, eff_str)) = maybe_fb {
                self.queue.write_buffer(&self.feedback_uniforms_buffer, 0, bytemuck::cast_slice(&[FeedbackUniforms {
                    center_x:     center[0],
                    center_y:     center[1],
                    shrink_rate:  self.params.blackhole_warp_curve,
                    strength:     eff_str,
                    alpha_radius: self.params.blackhole_alpha_radius,
                    _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
                }]));
                // Render passes (immutable borrows of separate self fields; NLL field-split OK).
                if let Some(ef) = self.export_feedback.as_ref() {
                    let offline_view = &self.offline_target.as_ref().unwrap().view;
                    // current_is_a=true → A is prev, write to B; blit B → offline_target.
                    let (write_view, read_bg, blit_bg) = if current_is_a {
                        (&ef.view_b, &ef.bg_a, &ef.blit_bg_b)
                    } else {
                        (&ef.view_a, &ef.bg_b, &ef.blit_bg_a)
                    };
                    {
                        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Export feedback pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: write_view, resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.feedback_pipeline);
                        pass.set_bind_group(0, read_bg, &[]);
                        pass.draw(0..3, 0..1);
                    }
                    {
                        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Export feedback blit-back"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: offline_view, resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.blit_pipeline);
                        pass.set_bind_group(0, blit_bg, &[]);
                        pass.draw(0..3, 0..1);
                    }
                } // immutable borrow of self.export_feedback released here
                if let Some(ef) = self.export_feedback.as_mut() {
                    ef.current_is_a = !current_is_a;
                }
            }
        }

        // Readback: copy offline_target → staging buffer
        let aligned_bpr = (off_w * 4).div_ceil(256) * 256;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Export readback"),
            size: (aligned_bpr * off_h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.offline_target.as_ref().unwrap().texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bpr),
                    rows_per_image: Some(off_h),
                },
            },
            wgpu::Extent3d { width: off_w, height: off_h, depth_or_array_layers: 1 },
        );
        self.queue.submit(std::iter::once(enc.finish()));

        // ── Phase 7: map, de-pad, BGRA→RGBA, send to worker ──────────────────
        let mut pixels = {
            let slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv().expect("map channel closed").expect("buffer map failed");
            let data = slice.get_mapped_range();
            let mut packed = Vec::with_capacity((off_w * off_h * 4) as usize);
            for row in 0..off_h {
                let s = (row * aligned_bpr) as usize;
                packed.extend_from_slice(&data[s..s + (off_w * 4) as usize]);
            }
            packed
        };
        readback.unmap();

        let fmt = self.offline_target.as_ref().unwrap().format;
        if matches!(fmt, wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Bgra8Unorm) {
            for px in pixels.chunks_exact_mut(4) { px.swap(0, 2); }
        }

        let _ = sender.send(FrameSaveJob {
            frame_index,
            width: off_w, height: off_h,
            rgba_bytes: pixels,
            output_path: output_dir.join(format!("frame_{:05}.png", frame_index)),
        });

        // ── Phase 8: preview — blit offline target to swapchain ───────────────
        if self.params.export_live_preview {
            if let Ok(swap_frame) = self.surface.get_current_texture() {
                let swap_view = swap_frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let preview_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Export preview BG"),
                    layout: &self.blit_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.offline_target.as_ref().unwrap().view,
                            ),
                        },
                        wgpu::BindGroupEntry { binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.shape_sampler),
                        },
                    ],
                });
                let mut blit_enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Export preview blit"),
                });
                {
                    let mut pass = blit_enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Export preview pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &swap_view, resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None, timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.blit_pipeline);
                    pass.set_bind_group(0, &preview_bg, &[]);
                    pass.draw(0..3, 0..1);
                }
                if let Some((menu_bar, window)) = menu {
                    menu_bar.render(
                        &self.device, &self.queue, &mut blit_enc, window,
                        &swap_view, self.size.width, self.size.height, &self.params,
                    );
                }
                self.queue.submit(std::iter::once(blit_enc.finish()));
                swap_frame.present();
            }
        }

        // ── Phase 9: advance counter + progress log ───────────────────────────
        if let Some(exp) = self.export_state.as_mut() {
            exp.current_frame += 1;
            let log_interval = (total / 10).max(30);
            if exp.current_frame % log_interval == 0 || exp.current_frame == total {
                let elapsed = exp.start_time.elapsed().as_secs_f32();
                let pct = exp.current_frame as f32 / total as f32 * 100.0;
                let render_fps = exp.current_frame as f32 / elapsed.max(0.001);
                let eta = (total - exp.current_frame) as f32 / render_fps;
                log::info!(
                    "Export progress: {}/{} ({:.1}%) — {:.1} fps render, ~{:.0}s remaining",
                    exp.current_frame, total, pct, render_fps, eta
                );
            }
        }
    }

    fn randomize_all_params(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        if !self.params.locks.painter_kind {
            self.params.painter_kind = match rng.gen_range(0u8..7) {
                0 => PainterKind::HueStripe,
                1 => PainterKind::Spiral,
                2 => PainterKind::Plasma,
                3 => PainterKind::Skin,
                4 => PainterKind::AudioPaint,
                5 => PainterKind::PrintHead,
                _ => PainterKind::Image,
            };
        }
        if !self.params.locks.current_shape {
            self.params.current_shape = match rng.gen_range(0u8..7) {
                0 => ShapeKind::Cylinder,
                1 => ShapeKind::Sphere,
                2 => ShapeKind::Cube,
                3 => ShapeKind::Tetrahedron,
                4 => ShapeKind::Icosahedron,
                5 => ShapeKind::Urchin,
                _ => ShapeKind::Caltrop,
            };
        }
        if !self.params.locks.frame_shape {
            self.params.frame_shape = match rng.gen_range(0u8..8) {
                0 => FrameShape::None,
                1 => FrameShape::Circle,
                2 => FrameShape::Square,
                3 => FrameShape::Rounded,
                4 => FrameShape::Hexagon,
                5 => FrameShape::Octagon,
                6 => FrameShape::Flower,
                _ => FrameShape::Star,
            };
        }
        if !self.params.locks.fold_count           { self.params.fold_count           = rng.gen_range(4.0_f32..=20.0).round(); }
        if !self.params.locks.zoom                 { self.params.zoom                 = rng.gen_range(0.5_f32..=1.3); }
        if !self.params.locks.rotation_speed_scale { self.params.rotation_speed_scale = rng.gen_range(0.3_f32..=2.5); }
        if !self.params.locks.frame_size           { self.params.frame_size           = rng.gen_range(0.65_f32..=1.0); }
        if !self.params.locks.frame_color_hue      { self.params.frame_color_hue      = rng.gen_range(0.0_f32..360.0); }
        if !self.params.locks.colorize_hue         { self.params.colorize_hue         = rng.gen_range(0.0_f32..360.0); }
        if !self.params.locks.colorize_intensity   { self.params.colorize_intensity   = rng.gen_range(0.2_f32..=0.9); }
        if !self.params.locks.distortion_amplitude { self.params.distortion_amplitude = rng.gen_range(0.0_f32..=0.25); }
        if !self.params.locks.distortion_frequency { self.params.distortion_frequency = rng.gen_range(1.0_f32..=6.0); }
        if !self.params.locks.contrast             { self.params.contrast             = rng.gen_range(0.7_f32..=1.8); }
        if !self.params.locks.contrast_passes      { self.params.contrast_passes      = rng.gen_range(1u32..=6); }
        if !self.params.locks.saturation           { self.params.saturation           = rng.gen_range(0.6_f32..=1.6); }
        if !self.params.locks.bass_zoom_strength   { self.params.bass_zoom_strength   = rng.gen_range(0.0_f32..=0.6); }
        if !self.params.locks.invert_enabled       { self.params.invert_enabled       = rng.gen_bool(0.5); }
        if !self.params.locks.colorize_enabled     { self.params.colorize_enabled     = rng.gen_bool(0.5); }
        if !self.params.locks.distortion_enabled   { self.params.distortion_enabled   = rng.gen_bool(0.5); }
        if !self.params.locks.midi_shake_enabled   { self.params.midi_shake_enabled   = rng.gen_bool(0.5); }
        if !self.params.locks.audio_shake_enabled  { self.params.audio_shake_enabled  = rng.gen_bool(0.5); }
        if !self.params.locks.ribbons_enabled      { self.params.ribbons_enabled      = rng.gen_bool(0.5); }
        if !self.params.locks.ribbons_intensity    { self.params.ribbons_intensity    = rng.gen_range(0.2_f32..=1.0); }
        // DP angles reroll independently of enabled (per spec).
        if !self.params.locks.distortion_plus_enabled { self.params.distortion_plus_enabled = rng.gen_bool(0.40); }
        if !self.params.locks.distortion_plus_yaw     { self.params.distortion_plus_yaw     = rng.gen_range(-180.0_f32..=180.0); }
        if !self.params.locks.distortion_plus_pitch   { self.params.distortion_plus_pitch   = rng.gen_range(-45.0_f32..=45.0); }
        if !self.params.locks.distortion_plus_roll    { self.params.distortion_plus_roll    = rng.gen_range(-180.0_f32..=180.0); }
        if !self.params.locks.palette_mode {
            self.params.palette_mode = match rng.gen_range(0u8..7) {
                0 => PaletteMode::Off,
                1 => PaletteMode::Warm,
                2 => PaletteMode::Cool,
                3 => PaletteMode::Earth,
                4 => PaletteMode::Neon,
                5 => PaletteMode::Monochrome,
                _ => PaletteMode::Harmony,
            };
        }
        if !self.params.locks.palette_tint     { self.params.palette_tint     = rng.gen_range(0.0_f32..=1.0); }
        if !self.params.locks.palette_mono_hue { self.params.palette_mono_hue = rng.gen_range(0.0_f32..=360.0); }
        if !self.params.locks.blackhole_enabled       { self.params.blackhole_enabled       = rng.gen_bool(0.10); }
        if !self.params.locks.blackhole_warp_strength { self.params.blackhole_warp_strength = rng.gen_range(0.85_f32..=0.98); }
        if !self.params.locks.blackhole_warp_curve    { self.params.blackhole_warp_curve    = rng.gen_range(0.90_f32..=0.99); }
        if !self.params.locks.blackhole_alpha_radius  { self.params.blackhole_alpha_radius  = rng.gen_range(0.3_f32..=1.0); }
        if !self.params.locks.blackhole_wander_amount { self.params.blackhole_wander_amount = rng.gen_range(0.0_f32..=0.015); }
        // Phantom: mutually exclusive with blackhole
        if !self.params.locks.phantom_enabled {
            self.params.phantom_enabled = if self.params.blackhole_enabled {
                false
            } else {
                rng.gen_bool(0.15)
            };
        }
        if !self.params.locks.phantom_delay_seconds { self.params.phantom_delay_seconds = rng.gen_range(0.5_f32..=3.0); }
        if !self.params.locks.phantom_key_tolerance { self.params.phantom_key_tolerance = rng.gen_range(0.05_f32..=0.40); }
        if !self.params.locks.phantom_key_softness  { self.params.phantom_key_softness  = rng.gen_range(0.02_f32..=0.15); }
        if !self.params.locks.phantom_key_strength  { self.params.phantom_key_strength  = rng.gen_range(0.5_f32..=1.0); }
        if !self.params.locks.phantom_opacity       { self.params.phantom_opacity       = rng.gen_range(0.5_f32..=1.0); }
        if !self.params.locks.phantom_key_color {
            let h = rng.gen_range(0.0_f32..360.0);
            let (r, g, b) = hsv_to_rgb(h, 1.0, 1.0);
            self.params.phantom_key_color = [r, g, b];
        }
        if !self.params.locks.color_harmony {
            let roll: f32 = rng.gen();
            self.params.color_harmony = if roll < 0.10 {
                color::ColorHarmony::Monochromatic
            } else if roll < 0.40 {
                color::ColorHarmony::Analogous
            } else if roll < 0.55 {
                color::ColorHarmony::Complementary
            } else if roll < 0.75 {
                color::ColorHarmony::SplitComplementary
            } else if roll < 0.90 {
                color::ColorHarmony::Triadic
            } else {
                color::ColorHarmony::Tetradic
            };
        }
        if !self.params.locks.color_anchor_hue {
            self.params.color_anchor_hue = rng.gen_range(0.0_f32..360.0);
        }
        if !self.params.locks.color_saturation {
            self.params.color_saturation = rng.gen_range(0.55_f32..=0.90);
        }
        if !self.params.locks.color_value {
            self.params.color_value = rng.gen_range(0.65_f32..=1.0);
        }
        if !self.params.locks.color_harmony_strength {
            self.params.color_harmony_strength = rng.gen_range(0.3_f32..=0.85);
        }
        if !self.params.locks.applied_harmony_enabled {
            self.params.applied_harmony_enabled = rng.gen_bool(0.25);
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
                        FrameShape::Octagon => FrameShape::Flower,
                        FrameShape::Flower  => FrameShape::Star,
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
            if gpu.params.midi_shake_enabled {
                let strength = velocity as f32 / 127.0;
                gpu.kick_shake(strength);
                log::debug!("MIDI Note On {} vel {} → shake {:.2}", note, velocity, strength);
            }
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
    prev_audio_source_mode: AudioSourceMode,
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
            prev_audio_source_mode: AudioSourceMode::File,
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
        let mut gpu = pollster::block_on(GpuState::new(window.clone()));
        let menu_bar = MenuBar::new(&gpu.device, gpu.config.format, &window);
        let (off_w, off_h) = gpu.params.export_resolution.dimensions();
        gpu.offline_target = Some(OfflineTarget::new(&gpu.device, off_w, off_h, gpu.config.format));
        self.window = Some(window);
        self.gpu = Some(gpu);
        self.menu_bar = Some(menu_bar);
        log::info!("Window and GPU initialized");

        // Audio capture is started lazily when the user selects Mic or Loopback
        // source mode. Default mode is File, so no capture at startup.
        self.audio = None;

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

        // Audio mode sync: restart (or stop) AudioCapture when source mode changes.
        // Must happen before self.gpu is mutably borrowed so we can write self.audio freely.
        // Covers both panel button clicks and preset loads.
        {
            let current_mode = self.gpu.as_ref()
                .map(|g| g.params.audio_source_mode)
                .unwrap_or_default();
            if current_mode != self.prev_audio_source_mode {
                self.audio = start_audio_for_mode(current_mode);
                self.prev_audio_source_mode = current_mode;
            }
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
                    KeyCode::Digit7 => { gpu.params.frame_shape = FrameShape::Flower;  log::info!("frame: Flower"); }
                    KeyCode::Digit8 => { gpu.params.frame_shape = FrameShape::Star;    log::info!("frame: Star"); }
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
                        if !gpu.params.blackhole_enabled {
                            gpu.params.phantom_enabled = !gpu.params.phantom_enabled;
                            log::info!("Phantom Alpha: {}", gpu.params.phantom_enabled);
                        }
                    }
                    KeyCode::KeyH => {
                        if !gpu.params.locks.color_harmony {
                            gpu.params.color_harmony = gpu.params.color_harmony.next();
                            log::info!("color harmony: {}", gpu.params.color_harmony.name());
                        } else {
                            log::info!("color harmony: LOCKED (skipping cycle)");
                        }
                    }
                    KeyCode::KeyJ => {
                        if !gpu.params.locks.applied_harmony_enabled {
                            gpu.params.applied_harmony_enabled = !gpu.params.applied_harmony_enabled;
                            log::info!("applied harmony: {}",
                                if gpu.params.applied_harmony_enabled { "ON" } else { "OFF" });
                        } else {
                            log::info!("applied harmony: LOCKED");
                        }
                    }
                    KeyCode::KeyB => {
                        gpu.params.reactive_mode_enabled = !gpu.params.reactive_mode_enabled;
                        log::info!("Reactive Mode: {}", gpu.params.reactive_mode_enabled);
                    }
                    KeyCode::KeyN if !ctrl => {
                        gpu.params.random_mode_enabled = !gpu.params.random_mode_enabled;
                        if gpu.params.random_mode_enabled {
                            gpu.last_random_change = Instant::now();
                        }
                        log::info!("Random Mode: {}", gpu.params.random_mode_enabled);
                    }
                    KeyCode::KeyY if !ctrl => {
                        gpu.params.party_mode_enabled = !gpu.params.party_mode_enabled;
                        log::info!("Party Mode: {}", gpu.params.party_mode_enabled);
                    }
                    KeyCode::Space => {
                        gpu.params.midi_shake_enabled = !gpu.params.midi_shake_enabled;
                        log::info!("midi_shake_enabled = {}", gpu.params.midi_shake_enabled);
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
                // Export mode: skip live audio/MIDI/mode processing
                if gpu.export_state.is_some() {
                    if let Some(audio) = &self.audio {
                        while audio.event_rx.try_recv().is_ok() {}
                    }
                    if let Some(midi) = &self.midi {
                        while midi.event_rx.try_recv().is_ok() {}
                    }
                    // Snapshot export progress before render (render may advance current_frame).
                    let progress_snap = gpu.export_state.as_ref().map(|e| ExportProgress {
                        current_frame: e.current_frame,
                        total_frames:  e.total_frames,
                        is_muxing:     e.phase == ExportPhase::Muxing,
                    });
                    if let Some(menu) = self.menu_bar.as_mut() {
                        menu.export_progress = progress_snap;
                    }
                    let menu = self.menu_bar.as_mut().map(|m| (m as &mut MenuBar, window.as_ref() as &winit::window::Window));
                    match gpu.render(menu) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            gpu.resize(gpu.size);
                        }
                        Err(e) => log::warn!("Surface error during export: {:?}", e),
                    }
                    if let Some(fps_val) = self.fps.tick() {
                        let title = match gpu.export_state.as_ref().map(|e| e.phase) {
                            Some(ExportPhase::Rendering) => {
                                let exp = gpu.export_state.as_ref().unwrap();
                                let headless = if !gpu.params.export_live_preview { " [headless]" } else { "" };
                                format!(
                                    "abstrakt-deck — slice 24s — EXPORTING {}/{} ({:.0}%){} — {:.1} fps",
                                    exp.current_frame, exp.total_frames,
                                    exp.current_frame as f32 / exp.total_frames as f32 * 100.0,
                                    headless, fps_val,
                                )
                            }
                            Some(ExportPhase::Muxing) => {
                                "abstrakt-deck — slice 24s — MUXING (ffmpeg)...".to_string()
                            }
                            _ => format!("abstrakt-deck — slice 24s — {:.1} fps", fps_val),
                        };
                        window.set_title(&title);
                    }
                    window.request_redraw();
                    return;
                }

                let file_playing = gpu.audio_player.as_ref().is_some_and(|p| p.is_playing());

                // Drain audio capture beat events; apply shake only in Mic/Loopback modes.
                if let Some(audio) = &self.audio {
                    let live_mode = matches!(
                        gpu.params.audio_source_mode,
                        AudioSourceMode::Mic | AudioSourceMode::Loopback
                    );
                    while let Ok(event) = audio.event_rx.try_recv() {
                        if live_mode {
                            match event {
                                AudioEvent::Beat(strength) => {
                                    if gpu.params.audio_shake_enabled {
                                        gpu.kick_shake(strength);
                                    }
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

                let mode = gpu.params.audio_source_mode;

                // Compute file energies only in File mode while the player is active.
                let file_energies: Option<([f32; 8], f32)> =
                    if matches!(mode, AudioSourceMode::File) && file_playing {
                        if let (Some(player), Some(audio)) = (&gpu.audio_player, &gpu.loaded_audio) {
                            Some(compute_file_energies(player, audio))
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                let frame_dt = gpu.last_audio_update.elapsed().as_secs_f32().clamp(0.001, 0.1);
                gpu.last_audio_update = Instant::now();

                let raw_bands: [f32; 8] = match mode {
                    AudioSourceMode::Silent => {
                        gpu.shader_beat_decay = 0.0;
                        [0.0; 8]
                    }
                    AudioSourceMode::File => {
                        match file_energies {
                            Some((bands, rms)) => {
                                if gpu.params.audio_shake_enabled {
                                    let threshold = gpu.file_rms_baseline * 1.5 + 0.01;
                                    if rms > threshold && gpu.last_file_beat.elapsed().as_millis() > 120 {
                                        let strength = ((rms / threshold).min(3.0) / 3.0).clamp(0.0, 1.0);
                                        gpu.kick_shake(strength);
                                        gpu.last_file_beat = Instant::now();
                                        gpu.shader_beat_decay = gpu.shader_beat_decay.max(strength);
                                    }
                                }
                                gpu.file_rms_baseline = gpu.file_rms_baseline * 0.95 + rms * 0.05;
                                gpu.shader_beat_decay *= (-5.0 * frame_dt).exp();
                                bands
                            }
                            None => {
                                // File mode but not playing — decay and zero out.
                                gpu.shader_beat_decay *= (-5.0 * frame_dt).exp();
                                [0.0; 8]
                            }
                        }
                    }
                    AudioSourceMode::Mic | AudioSourceMode::Loopback => {
                        let (bands, bd) = self.audio.as_ref()
                            .map(|a| { let s = a.state.lock(); (s.bands, s.beat_decay) })
                            .unwrap_or(([0.0; 8], 0.0));
                        gpu.shader_beat_decay = bd;
                        bands
                    }
                };

                const BANDS_ALPHA: f32 = 0.25;
                for (i, &raw) in raw_bands.iter().enumerate() {
                    gpu.bands_smoothed[i] =
                        gpu.bands_smoothed[i] * (1.0 - BANDS_ALPHA) + raw * BANDS_ALPHA;
                }

                gpu.update_bass_zoom(raw_bands[0]);
                gpu.update_modes(raw_bands[0], raw_bands[3], None);

                // Update player info displayed in the panel
                let player_info = if let (Some(player), Some(audio)) =
                    (&gpu.audio_player, &gpu.loaded_audio)
                {
                    let filename = std::path::Path::new(&audio.source_path)
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("?")
                        .to_string();
                    Some(menu_bar::PlayerInfo {
                        filename,
                        duration_seconds: audio.duration_seconds,
                        position_seconds: player.position_seconds(audio.sample_rate),
                        is_playing: player.is_playing(),
                    })
                } else {
                    None
                };
                if let Some(menu) = self.menu_bar.as_mut() {
                    menu.export_progress = None; // not exporting in normal render path
                    menu.player_info = player_info;
                }

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
                        MenuAction::OpenAudio => {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Audio", &["mp3", "wav", "flac", "ogg", "m4a", "aac", "opus", "aiff", "wv"])
                                .set_title("Select audio file")
                                .pick_file()
                            {
                                log::info!("Loading audio from {}", path.display());
                                match decode_audio_file(&path) {
                                    Ok(audio) => {
                                        let arc_audio = Arc::new(audio);
                                        gpu.loaded_audio = Some(arc_audio.clone());
                                        gpu.file_rms_baseline = 0.0;
                                        match AudioPlayer::new(arc_audio) {
                                            Ok(player) => {
                                                gpu.audio_player = Some(player);
                                                log::info!("Audio loaded and player ready");
                                            }
                                            Err(e) => {
                                                log::error!("Player creation failed: {}", e);
                                            }
                                        }
                                    }
                                    Err(e) => log::error!("Audio load failed: {}", e),
                                }
                            } else {
                                log::info!("Open Audio canceled");
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
                        ParamChange::DistortionEnabled(v)       => gpu.params.distortion_enabled       = v,
                        ParamChange::DistortionAmplitude(v)     => gpu.params.distortion_amplitude     = v,
                        ParamChange::DistortionFrequency(v)     => gpu.params.distortion_frequency     = v,
                        ParamChange::DistortionPlusEnabled(v)   => gpu.params.distortion_plus_enabled  = v,
                        ParamChange::DistortionPlusYaw(v)       => gpu.params.distortion_plus_yaw      = v,
                        ParamChange::DistortionPlusPitch(v)     => gpu.params.distortion_plus_pitch    = v,
                        ParamChange::DistortionPlusRoll(v)      => gpu.params.distortion_plus_roll     = v,
                        ParamChange::MidiShakeEnabled(v)     => gpu.params.midi_shake_enabled  = v,
                        ParamChange::AudioShakeEnabled(v)    => gpu.params.audio_shake_enabled = v,
                        ParamChange::RibbonsEnabled(v)       => gpu.params.ribbons_enabled     = v,
                        ParamChange::RibbonsIntensity(v)     => gpu.params.ribbons_intensity   = v,
                        ParamChange::BassZoomStrength(v)     => gpu.params.bass_zoom_strength  = v,
                        ParamChange::BeatReactivity(v)       => gpu.params.beat_reactivity     = v,
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
                        ParamChange::Contrast(v)        => gpu.params.contrast        = v,
                        ParamChange::Saturation(v)      => gpu.params.saturation      = v,
                        ParamChange::ContrastPasses(v)  => gpu.params.contrast_passes = v,
                        ParamChange::RandomModeEnabled(v) => {
                            gpu.params.random_mode_enabled = v;
                            if v { gpu.last_random_change = Instant::now(); }
                        }
                        ParamChange::RandomModeAggressiveness(v)   => gpu.params.random_mode_aggressiveness   = v,
                        ParamChange::ReactiveModeEnabled(v)        => gpu.params.reactive_mode_enabled        = v,
                        ParamChange::ReactiveModeAggressiveness(v) => gpu.params.reactive_mode_aggressiveness = v,
                        ParamChange::PartyModeEnabled(v)           => gpu.params.party_mode_enabled           = v,
                        ParamChange::PartyModeAggressiveness(v)   => gpu.params.party_mode_aggressiveness   = v,
                        ParamChange::PlayerToggle => {
                            if let Some(player) = &gpu.audio_player {
                                if player.is_playing() { player.pause(); } else { player.play(); }
                            }
                        }
                        ParamChange::PlayerStop => {
                            if let Some(player) = &gpu.audio_player {
                                player.pause();
                                player.seek_frames(0);
                            }
                        }
                        ParamChange::PlayerSeek(secs) => {
                            if let (Some(player), Some(audio)) =
                                (&gpu.audio_player, &gpu.loaded_audio)
                            {
                                player.seek_frames((secs * audio.sample_rate as f32) as usize);
                            }
                        }
                        ParamChange::ToggleLock(target) => {
                            use menu_bar::LockTarget;
                            match target {
                                LockTarget::PainterKind        => gpu.params.locks.painter_kind        = !gpu.params.locks.painter_kind,
                                LockTarget::CurrentShape       => gpu.params.locks.current_shape       = !gpu.params.locks.current_shape,
                                LockTarget::FoldCount          => gpu.params.locks.fold_count          = !gpu.params.locks.fold_count,
                                LockTarget::Zoom               => gpu.params.locks.zoom               = !gpu.params.locks.zoom,
                                LockTarget::RotationSpeedScale => gpu.params.locks.rotation_speed_scale = !gpu.params.locks.rotation_speed_scale,
                                LockTarget::FrameShape         => gpu.params.locks.frame_shape         = !gpu.params.locks.frame_shape,
                                LockTarget::FrameSize          => gpu.params.locks.frame_size          = !gpu.params.locks.frame_size,
                                LockTarget::FrameColorHue      => gpu.params.locks.frame_color_hue     = !gpu.params.locks.frame_color_hue,
                                LockTarget::InvertEnabled      => gpu.params.locks.invert_enabled      = !gpu.params.locks.invert_enabled,
                                LockTarget::ColorizeEnabled    => gpu.params.locks.colorize_enabled    = !gpu.params.locks.colorize_enabled,
                                LockTarget::ColorizeHue        => gpu.params.locks.colorize_hue        = !gpu.params.locks.colorize_hue,
                                LockTarget::ColorizeIntensity  => gpu.params.locks.colorize_intensity  = !gpu.params.locks.colorize_intensity,
                                LockTarget::DistortionEnabled    => gpu.params.locks.distortion_enabled    = !gpu.params.locks.distortion_enabled,
                                LockTarget::DistortionAmplitude  => gpu.params.locks.distortion_amplitude  = !gpu.params.locks.distortion_amplitude,
                                LockTarget::DistortionFrequency  => gpu.params.locks.distortion_frequency  = !gpu.params.locks.distortion_frequency,
                                LockTarget::DistortionPlusEnabled => gpu.params.locks.distortion_plus_enabled = !gpu.params.locks.distortion_plus_enabled,
                                LockTarget::DistortionPlusYaw     => gpu.params.locks.distortion_plus_yaw     = !gpu.params.locks.distortion_plus_yaw,
                                LockTarget::DistortionPlusPitch   => gpu.params.locks.distortion_plus_pitch   = !gpu.params.locks.distortion_plus_pitch,
                                LockTarget::DistortionPlusRoll    => gpu.params.locks.distortion_plus_roll    = !gpu.params.locks.distortion_plus_roll,
                                LockTarget::Contrast           => gpu.params.locks.contrast           = !gpu.params.locks.contrast,
                                LockTarget::ContrastPasses     => gpu.params.locks.contrast_passes     = !gpu.params.locks.contrast_passes,
                                LockTarget::Saturation         => gpu.params.locks.saturation         = !gpu.params.locks.saturation,
                                LockTarget::BassZoomStrength   => gpu.params.locks.bass_zoom_strength   = !gpu.params.locks.bass_zoom_strength,
                                LockTarget::BeatReactivity     => gpu.params.locks.beat_reactivity      = !gpu.params.locks.beat_reactivity,
                                LockTarget::MidiShakeEnabled   => gpu.params.locks.midi_shake_enabled   = !gpu.params.locks.midi_shake_enabled,
                                LockTarget::AudioShakeEnabled  => gpu.params.locks.audio_shake_enabled  = !gpu.params.locks.audio_shake_enabled,
                                LockTarget::RibbonsEnabled     => gpu.params.locks.ribbons_enabled      = !gpu.params.locks.ribbons_enabled,
                                LockTarget::RibbonsIntensity   => gpu.params.locks.ribbons_intensity    = !gpu.params.locks.ribbons_intensity,
                                LockTarget::PaletteMode        => gpu.params.locks.palette_mode         = !gpu.params.locks.palette_mode,
                                LockTarget::PaletteTint        => gpu.params.locks.palette_tint         = !gpu.params.locks.palette_tint,
                                LockTarget::PaletteMonoHue     => gpu.params.locks.palette_mono_hue     = !gpu.params.locks.palette_mono_hue,
                                LockTarget::BlackholeEnabled       => gpu.params.locks.blackhole_enabled       = !gpu.params.locks.blackhole_enabled,
                                LockTarget::BlackholeWarpStrength  => gpu.params.locks.blackhole_warp_strength = !gpu.params.locks.blackhole_warp_strength,
                                LockTarget::BlackholeWarpCurve     => gpu.params.locks.blackhole_warp_curve    = !gpu.params.locks.blackhole_warp_curve,
                                LockTarget::BlackholeAlphaRadius   => gpu.params.locks.blackhole_alpha_radius  = !gpu.params.locks.blackhole_alpha_radius,
                                LockTarget::BlackholeWanderAmount  => gpu.params.locks.blackhole_wander_amount = !gpu.params.locks.blackhole_wander_amount,
                                LockTarget::ColorHarmony          => gpu.params.locks.color_harmony           = !gpu.params.locks.color_harmony,
                                LockTarget::ColorAnchorHue        => gpu.params.locks.color_anchor_hue        = !gpu.params.locks.color_anchor_hue,
                                LockTarget::ColorSaturation       => gpu.params.locks.color_saturation        = !gpu.params.locks.color_saturation,
                                LockTarget::ColorValue            => gpu.params.locks.color_value             = !gpu.params.locks.color_value,
                                LockTarget::ColorHarmonyStrength  => gpu.params.locks.color_harmony_strength  = !gpu.params.locks.color_harmony_strength,
                                LockTarget::AppliedHarmonyEnabled => gpu.params.locks.applied_harmony_enabled = !gpu.params.locks.applied_harmony_enabled,
                                LockTarget::PhantomEnabled       => gpu.params.locks.phantom_enabled       = !gpu.params.locks.phantom_enabled,
                                LockTarget::PhantomDelaySeconds  => gpu.params.locks.phantom_delay_seconds = !gpu.params.locks.phantom_delay_seconds,
                                LockTarget::PhantomKeyColor      => gpu.params.locks.phantom_key_color     = !gpu.params.locks.phantom_key_color,
                                LockTarget::PhantomKeyTolerance  => gpu.params.locks.phantom_key_tolerance = !gpu.params.locks.phantom_key_tolerance,
                                LockTarget::PhantomKeySoftness   => gpu.params.locks.phantom_key_softness  = !gpu.params.locks.phantom_key_softness,
                                LockTarget::PhantomKeyStrength   => gpu.params.locks.phantom_key_strength  = !gpu.params.locks.phantom_key_strength,
                                LockTarget::PhantomOpacity       => gpu.params.locks.phantom_opacity       = !gpu.params.locks.phantom_opacity,
                            }
                            log::info!("Lock toggled: {:?}", target);
                        }
                        ParamChange::ExportResolution(v) => {
                            gpu.params.export_resolution = v;
                            let (w, h) = v.dimensions();
                            let fmt = gpu.config.format;
                            gpu.offline_target = Some(OfflineTarget::new(&gpu.device, w, h, fmt));
                            log::info!("Offline target resized to {}×{}", w, h);
                        }
                        ParamChange::ExportFramerate(v) => {
                            gpu.params.export_framerate = v;
                        }
                        ParamChange::SetExportLivePreview(v) => {
                            gpu.params.export_live_preview = v;
                        }
                        ParamChange::TriggerExport => {
                            match gpu.start_export() {
                                Ok(()) => log::info!("Export started via Export button"),
                                Err(e) => log::error!("Export failed to start: {}", e),
                            }
                        }
                        ParamChange::SetAudioSourceMode(v) => {
                            gpu.params.audio_source_mode = v;
                            // Actual AudioCapture restart is handled by the mode sync check
                            // at the top of window_event on the next event.
                        }
                        ParamChange::SetPaletteMode(v)  => gpu.params.palette_mode     = v,
                        ParamChange::PaletteTint(v)     => gpu.params.palette_tint     = v,
                        ParamChange::PaletteMonoHue(v)  => gpu.params.palette_mono_hue = v,
                        ParamChange::BlackholeEnabled(v)       => gpu.params.blackhole_enabled       = v,
                        ParamChange::BlackholeWarpStrength(v)  => gpu.params.blackhole_warp_strength = v,
                        ParamChange::BlackholeWarpCurve(v)     => gpu.params.blackhole_warp_curve    = v,
                        ParamChange::BlackholeAlphaRadius(v)   => gpu.params.blackhole_alpha_radius  = v,
                        ParamChange::BlackholeWanderAmount(v)  => gpu.params.blackhole_wander_amount = v,
                        ParamChange::ColorHarmony(v)          => gpu.params.color_harmony           = v,
                        ParamChange::ColorAnchorHue(v)        => gpu.params.color_anchor_hue        = v,
                        ParamChange::ColorSaturation(v)       => gpu.params.color_saturation        = v,
                        ParamChange::ColorValue(v)            => gpu.params.color_value             = v,
                        ParamChange::ColorHarmonyStrength(v)  => gpu.params.color_harmony_strength  = v,
                        ParamChange::AppliedHarmonyEnabled(v) => gpu.params.applied_harmony_enabled = v,
                        ParamChange::PhantomEnabled(v)       => gpu.params.phantom_enabled       = v,
                        ParamChange::PhantomDelaySeconds(v)  => gpu.params.phantom_delay_seconds = v,
                        ParamChange::PhantomKeyColor(v)      => gpu.params.phantom_key_color     = v,
                        ParamChange::PhantomKeyTolerance(v)  => gpu.params.phantom_key_tolerance = v,
                        ParamChange::PhantomKeySoftness(v)   => gpu.params.phantom_key_softness  = v,
                        ParamChange::PhantomKeyStrength(v)   => gpu.params.phantom_key_strength  = v,
                        ParamChange::PhantomOpacity(v)       => gpu.params.phantom_opacity       = v,
                    }
                }

                if let Some(fps) = self.fps.tick() {
                    let title = if let Some(rec) = gpu.recorder.as_ref() {
                        let secs = rec.elapsed().as_secs();
                        format!(
                            "abstrakt-deck — slice 24s — ● REC {}:{:02} — {:.1} fps",
                            secs / 60, secs % 60, fps
                        )
                    } else {
                        format!("abstrakt-deck — slice 24s — {:.1} fps", fps)
                    };
                    window.set_title(&title);
                }
                window.request_redraw();
            }
            _ => {}
        }
    }
}

fn clean_old_exports() {
    let exports_dir = match dirs::cache_dir() {
        Some(d) => d.join("abstrakt-deck").join("exports"),
        None => return,
    };
    let entries = match std::fs::read_dir(&exports_dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        if entry.file_name().to_str().is_some_and(|n| n.starts_with("export-")) {
            let path = entry.path();
            match std::fs::remove_dir_all(&path) {
                Ok(()) => log::info!("Cleaned leftover export dir: {}", path.display()),
                Err(e) => log::warn!("Could not clean {}: {}", path.display(), e),
            }
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
    println!("  R      cycle frame color hue (+30°)");
    println!("  G      toggle Phantom Alpha overlay");
    println!("  H      cycle color harmony (Mono/Analogous/Comp/Split/Triad/Tetra)");
    println!("  J      toggle applied harmony (recolor Skin/Image/PrintHead via Color Theory)");
    println!("  space  toggle MIDI shake");
    println!("  Shift+Tab  cycle shape (Cylinder → Sphere → Cube → Tetrahedron → Icosahedron → Urchin → Caltrop)");
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
    println!("  N      toggle Random Mode (timer-based painter cycle)");
    println!("  B      toggle Reactive Mode (audio-triggered painter cycle)");
    println!("  Y      toggle Party Mode (aggressive: painter + shape + hue)");
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

    clean_old_exports();
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
            distortion_plus_enabled: true,
            distortion_plus_yaw:     45.0,
            distortion_plus_pitch:   -20.0,
            distortion_plus_roll:    90.0,
            midi_shake_enabled:  false,
            audio_shake_enabled: true,
            ribbons_enabled:   true,
            ribbons_intensity: 0.8,
            bass_zoom_strength: 0.8,
            beat_reactivity: 0.75,
            painter_kind: PainterKind::PrintHead,
            contrast: 1.5,
            saturation: 0.7,
            contrast_passes: 3,
            random_mode_enabled: true,
            random_mode_aggressiveness: 0.7,
            reactive_mode_enabled: true,
            reactive_mode_aggressiveness: 0.3,
            party_mode_enabled: true,
            party_mode_aggressiveness: 0.8,
            locks: ParamLocks {
                contrast: true,
                contrast_passes: true,
                ..Default::default()
            },
            export_resolution: ResolutionPreset::FullHD,
            export_framerate:  FramerateChoice::Fps30,
            export_live_preview: false,
            audio_source_mode: AudioSourceMode::Mic,
            palette_mode:     PaletteMode::Cool,
            palette_tint:     0.7,
            palette_mono_hue: 280.0,
            blackhole_enabled:        true,
            blackhole_warp_strength:  0.95,
            blackhole_warp_curve:     0.96,
            blackhole_alpha_radius:   0.6,
            blackhole_wander_amount:  0.008,
            phantom_enabled:       true,
            phantom_delay_seconds: 1.5,
            phantom_key_color:     [1.0, 0.0, 0.0],
            phantom_key_tolerance: 0.20,
            phantom_key_softness:  0.08,
            phantom_key_strength:  0.9,
            phantom_opacity:       0.7,
            color_harmony:           color::ColorHarmony::Triadic,
            color_anchor_hue:        45.0,
            color_saturation:        0.8,
            color_value:             0.9,
            color_harmony_strength:  0.6,
            applied_harmony_enabled: true,
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
        assert_eq!(restored.distortion_plus_enabled, original.distortion_plus_enabled, "distortion_plus_enabled failed");
        assert_eq!(restored.distortion_plus_yaw,     original.distortion_plus_yaw,     "distortion_plus_yaw failed");
        assert_eq!(restored.distortion_plus_pitch,   original.distortion_plus_pitch,   "distortion_plus_pitch failed");
        assert_eq!(restored.distortion_plus_roll,    original.distortion_plus_roll,    "distortion_plus_roll failed");
        assert_eq!(restored.midi_shake_enabled,  original.midi_shake_enabled,  "midi_shake_enabled failed");
        assert_eq!(restored.audio_shake_enabled, original.audio_shake_enabled, "audio_shake_enabled failed");
        assert_eq!(restored.ribbons_enabled,   original.ribbons_enabled,   "ribbons_enabled failed");
        assert_eq!(restored.ribbons_intensity, original.ribbons_intensity, "ribbons_intensity failed");
        assert_eq!(restored.bass_zoom_strength, original.bass_zoom_strength, "bass_zoom_strength failed");
        assert_eq!(restored.beat_reactivity,    original.beat_reactivity,    "beat_reactivity failed");
        assert_eq!(restored.painter_kind, original.painter_kind, "painter_kind failed");
        assert_eq!(restored.contrast,        original.contrast,        "contrast failed");
        assert_eq!(restored.saturation,      original.saturation,      "saturation failed");
        assert_eq!(restored.contrast_passes, original.contrast_passes, "contrast_passes failed");
        assert_eq!(restored.random_mode_enabled,          original.random_mode_enabled,          "random_mode_enabled failed");
        assert_eq!(restored.random_mode_aggressiveness,   original.random_mode_aggressiveness,   "random_mode_aggressiveness failed");
        assert_eq!(restored.reactive_mode_enabled,        original.reactive_mode_enabled,        "reactive_mode_enabled failed");
        assert_eq!(restored.reactive_mode_aggressiveness, original.reactive_mode_aggressiveness, "reactive_mode_aggressiveness failed");
        assert_eq!(restored.party_mode_enabled,           original.party_mode_enabled,           "party_mode_enabled failed");
        assert_eq!(restored.party_mode_aggressiveness,    original.party_mode_aggressiveness,    "party_mode_aggressiveness failed");
        assert_eq!(restored.locks.contrast,        original.locks.contrast,        "locks.contrast failed");
        assert_eq!(restored.locks.contrast_passes, original.locks.contrast_passes, "locks.contrast_passes failed");
        assert_eq!(restored.export_resolution,   original.export_resolution,   "export_resolution failed");
        assert_eq!(restored.export_framerate,    original.export_framerate,    "export_framerate failed");
        assert_eq!(restored.export_live_preview, original.export_live_preview, "export_live_preview failed");
        assert_eq!(restored.audio_source_mode,   original.audio_source_mode,   "audio_source_mode failed");
        assert_eq!(restored.palette_mode,     original.palette_mode,     "palette_mode failed");
        assert_eq!(restored.palette_tint,     original.palette_tint,     "palette_tint failed");
        assert_eq!(restored.palette_mono_hue, original.palette_mono_hue, "palette_mono_hue failed");
        assert_eq!(restored.blackhole_enabled,        original.blackhole_enabled,        "blackhole_enabled failed");
        assert_eq!(restored.blackhole_warp_strength,  original.blackhole_warp_strength,  "blackhole_warp_strength failed");
        assert_eq!(restored.blackhole_warp_curve,     original.blackhole_warp_curve,     "blackhole_warp_curve failed");
        assert_eq!(restored.blackhole_alpha_radius,   original.blackhole_alpha_radius,   "blackhole_alpha_radius failed");
        assert_eq!(restored.blackhole_wander_amount,  original.blackhole_wander_amount,  "blackhole_wander_amount failed");
        assert_eq!(restored.phantom_enabled,       original.phantom_enabled,       "phantom_enabled failed");
        assert_eq!(restored.phantom_delay_seconds, original.phantom_delay_seconds, "phantom_delay_seconds failed");
        assert_eq!(restored.phantom_key_color,     original.phantom_key_color,     "phantom_key_color failed");
        assert_eq!(restored.phantom_key_tolerance, original.phantom_key_tolerance, "phantom_key_tolerance failed");
        assert_eq!(restored.phantom_key_softness,  original.phantom_key_softness,  "phantom_key_softness failed");
        assert_eq!(restored.phantom_key_strength,  original.phantom_key_strength,  "phantom_key_strength failed");
        assert_eq!(restored.phantom_opacity,       original.phantom_opacity,       "phantom_opacity failed");
        assert_eq!(restored.color_harmony,           original.color_harmony,           "color_harmony failed");
        assert_eq!(restored.color_anchor_hue,        original.color_anchor_hue,        "color_anchor_hue failed");
        assert_eq!(restored.color_saturation,        original.color_saturation,        "color_saturation failed");
        assert_eq!(restored.color_value,             original.color_value,             "color_value failed");
        assert_eq!(restored.color_harmony_strength,  original.color_harmony_strength,  "color_harmony_strength failed");
        assert_eq!(restored.applied_harmony_enabled, original.applied_harmony_enabled, "applied_harmony_enabled failed");
    }

    #[test]
    fn shape_kind_name_roundtrip() {
        for kind in [
            ShapeKind::Cylinder, ShapeKind::Sphere, ShapeKind::Cube, ShapeKind::Tetrahedron,
            ShapeKind::Icosahedron, ShapeKind::Urchin, ShapeKind::Caltrop,
        ] {
            let name = kind.name();
            let parsed = match name {
                "Sphere"      => ShapeKind::Sphere,
                "Cube"        => ShapeKind::Cube,
                "Tetrahedron" => ShapeKind::Tetrahedron,
                "Icosahedron" => ShapeKind::Icosahedron,
                "Urchin"      => ShapeKind::Urchin,
                "Caltrop"     => ShapeKind::Caltrop,
                _             => ShapeKind::Cylinder,
            };
            assert_eq!(parsed, kind, "ShapeKind {:?} did not round-trip via name()", kind);
        }
    }

    #[test]
    fn frame_shape_debug_roundtrip() {
        for shape in [
            FrameShape::None, FrameShape::Circle, FrameShape::Square,
            FrameShape::Rounded, FrameShape::Hexagon, FrameShape::Octagon,
            FrameShape::Flower, FrameShape::Star,
        ] {
            let debug_str = format!("{:?}", shape);
            let parsed = match debug_str.as_str() {
                "None"    => FrameShape::None,
                "Circle"  => FrameShape::Circle,
                "Square"  => FrameShape::Square,
                "Rounded" => FrameShape::Rounded,
                "Octagon" => FrameShape::Octagon,
                "Flower"  => FrameShape::Flower,
                "Star"    => FrameShape::Star,
                _         => FrameShape::Hexagon,
            };
            assert_eq!(parsed, shape, "FrameShape {:?} did not round-trip via Debug format", shape);
        }
    }

    #[test]
    fn rotation_axes_are_normalized() {
        for shape in [
            ShapeKind::Cylinder, ShapeKind::Sphere, ShapeKind::Cube, ShapeKind::Tetrahedron,
            ShapeKind::Icosahedron, ShapeKind::Urchin, ShapeKind::Caltrop,
        ] {
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

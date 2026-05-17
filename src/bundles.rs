//! Visual artist preset bundles — one-click style starting points.
//!
//! Each `BundleId` writes a curated set of visual params while respecting
//! active `ParamLocks`. Params not mentioned by a bundle are left unchanged.

use crate::{
    PaletteMode, VisualParams,
    audio::BeatRoute,
    color::{ColorHarmony, SaturationMode, ValueKey},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BundleId {
    #[default]
    Defaults,
    Midnight,
    Prism,
    Organic,
    Neon,
    Frost,
    Inferno,
    Pastel,
    Void,
    Carnival,
}

pub const ALL: &[BundleId] = &[
    BundleId::Defaults,
    BundleId::Midnight,
    BundleId::Prism,
    BundleId::Organic,
    BundleId::Neon,
    BundleId::Frost,
    BundleId::Inferno,
    BundleId::Pastel,
    BundleId::Void,
    BundleId::Carnival,
];

impl BundleId {
    pub fn name(self) -> &'static str {
        match self {
            BundleId::Defaults => "Defaults",
            BundleId::Midnight => "Midnight",
            BundleId::Prism    => "Prism",
            BundleId::Organic  => "Organic",
            BundleId::Neon     => "Neon",
            BundleId::Frost    => "Frost",
            BundleId::Inferno  => "Inferno",
            BundleId::Pastel   => "Pastel",
            BundleId::Void     => "Void",
            BundleId::Carnival => "Carnival",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            BundleId::Defaults => "Reset to factory defaults — analogous blue, no effects",
            BundleId::Midnight => "Deep indigo space — monochromatic, dark, blackhole pull",
            BundleId::Prism    => "Rainbow spectrum — complementary tension, hue cycling, Bezold pop",
            BundleId::Organic  => "Warm earth tones — analogous amber, ribbons, gentle distortion",
            BundleId::Neon     => "Electric club — tetradic purple, full saturation, neon palette",
            BundleId::Frost    => "Ice clarity — cool blue analogous, muted saturation, phantom echo",
            BundleId::Inferno  => "Heat blast — red-orange complementary, audio-driven temperature",
            BundleId::Pastel   => "Soft dreamscape — triadic pink, muted saturation, high value",
            BundleId::Void     => "Dark abyss — near-black monochromatic, distortion plus, blackhole",
            BundleId::Carnival => "Festival burst — tetradic yellow, phase-cycling hue, ribbons",
        }
    }

    /// Write bundle params into `p`, skipping fields whose lock is set.
    pub fn apply(self, p: &mut VisualParams, locks: &crate::ParamLocks) {
        match self {
            BundleId::Defaults => apply_defaults(p, locks),
            BundleId::Midnight => apply_midnight(p, locks),
            BundleId::Prism    => apply_prism(p, locks),
            BundleId::Organic  => apply_organic(p, locks),
            BundleId::Neon     => apply_neon(p, locks),
            BundleId::Frost    => apply_frost(p, locks),
            BundleId::Inferno  => apply_inferno(p, locks),
            BundleId::Pastel   => apply_pastel(p, locks),
            BundleId::Void     => apply_void(p, locks),
            BundleId::Carnival => apply_carnival(p, locks),
        }
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

macro_rules! set_if {
    ($p:expr, $locks:expr, $lock_field:ident, $param_field:ident, $val:expr) => {
        if !$locks.$lock_field { $p.$param_field = $val; }
    };
}

// ── bundle implementations ────────────────────────────────────────────────────

fn apply_defaults(p: &mut VisualParams, locks: &crate::ParamLocks) {
    let def = VisualParams::default();
    set_if!(p, locks, color_harmony,           color_harmony,           def.color_harmony);
    set_if!(p, locks, color_anchor_hue,         color_anchor_hue,         def.color_anchor_hue);
    set_if!(p, locks, color_temperature_bias,   color_temperature_bias,   def.color_temperature_bias);
    set_if!(p, locks, color_temperature_audio,  color_temperature_audio,  def.color_temperature_audio);
    set_if!(p, locks, color_saturation_mode,    color_saturation_mode,    def.color_saturation_mode);
    set_if!(p, locks, color_saturation,         color_saturation,         def.color_saturation);
    set_if!(p, locks, color_value_key,          color_value_key,          def.color_value_key);
    set_if!(p, locks, color_value,              color_value,              def.color_value);
    set_if!(p, locks, color_harmony_strength,   color_harmony_strength,   def.color_harmony_strength);
    set_if!(p, locks, color_phase_cycle_enabled, color_phase_cycle_enabled, def.color_phase_cycle_enabled);
    set_if!(p, locks, color_phase_cycle_degrees, color_phase_cycle_degrees, def.color_phase_cycle_degrees);
    set_if!(p, locks, color_phase_cycle_locked,  color_phase_cycle_locked,  def.color_phase_cycle_locked);
    set_if!(p, locks, applied_harmony_enabled,  applied_harmony_enabled,  def.applied_harmony_enabled);
    set_if!(p, locks, palette_mode,             palette_mode,             def.palette_mode);
    set_if!(p, locks, palette_tint,             palette_tint,             def.palette_tint);
    set_if!(p, locks, contrast,                 contrast,                 def.contrast);
    set_if!(p, locks, saturation,               saturation,               def.saturation);
    set_if!(p, locks, contrast_passes,          contrast_passes,          def.contrast_passes);
    set_if!(p, locks, ribbons_enabled,          ribbons_enabled,          def.ribbons_enabled);
    set_if!(p, locks, ribbons_intensity,        ribbons_intensity,        def.ribbons_intensity);
    set_if!(p, locks, distortion_enabled,       distortion_enabled,       def.distortion_enabled);
    set_if!(p, locks, distortion_amplitude,     distortion_amplitude,     def.distortion_amplitude);
    set_if!(p, locks, distortion_frequency,     distortion_frequency,     def.distortion_frequency);
    set_if!(p, locks, distortion_plus_enabled,  distortion_plus_enabled,  def.distortion_plus_enabled);
    set_if!(p, locks, distortion_plus_yaw,      distortion_plus_yaw,      def.distortion_plus_yaw);
    set_if!(p, locks, distortion_plus_pitch,    distortion_plus_pitch,    def.distortion_plus_pitch);
    set_if!(p, locks, distortion_plus_roll,     distortion_plus_roll,     def.distortion_plus_roll);
    set_if!(p, locks, blackhole_enabled,        blackhole_enabled,        def.blackhole_enabled);
    set_if!(p, locks, blackhole_warp_strength,  blackhole_warp_strength,  def.blackhole_warp_strength);
    set_if!(p, locks, blackhole_warp_curve,     blackhole_warp_curve,     def.blackhole_warp_curve);
    set_if!(p, locks, blackhole_alpha_radius,   blackhole_alpha_radius,   def.blackhole_alpha_radius);
    set_if!(p, locks, blackhole_wander_amount,  blackhole_wander_amount,  def.blackhole_wander_amount);
    set_if!(p, locks, phantom_enabled,          phantom_enabled,          def.phantom_enabled);
    set_if!(p, locks, bezold_enabled,           bezold_enabled,           def.bezold_enabled);
    set_if!(p, locks, bezold_strength,          bezold_strength,          def.bezold_strength);
    set_if!(p, locks, bezold_radius,            bezold_radius,            def.bezold_radius);
    set_if!(p, locks, audio_route_shape,        audio_route_shape,        def.audio_route_shape);
    set_if!(p, locks, audio_route_kaleido,      audio_route_kaleido,      def.audio_route_kaleido);
    set_if!(p, locks, audio_route_shake,        audio_route_shake,        def.audio_route_shake);
    set_if!(p, locks, phase_lock_enabled,       phase_lock_enabled,       def.phase_lock_enabled);
}

fn apply_midnight(p: &mut VisualParams, locks: &crate::ParamLocks) {
    set_if!(p, locks, color_harmony,           color_harmony,           ColorHarmony::Monochromatic);
    set_if!(p, locks, color_anchor_hue,         color_anchor_hue,         240.0);
    set_if!(p, locks, color_temperature_bias,   color_temperature_bias,   -0.5);
    set_if!(p, locks, color_temperature_audio,  color_temperature_audio,  0.0);
    set_if!(p, locks, color_saturation_mode,    color_saturation_mode,    SaturationMode::Free);
    set_if!(p, locks, color_saturation,         color_saturation,         0.6);
    set_if!(p, locks, color_value_key,          color_value_key,          ValueKey::Free);
    set_if!(p, locks, color_value,              color_value,              0.35);
    set_if!(p, locks, color_harmony_strength,   color_harmony_strength,   0.8);
    set_if!(p, locks, color_phase_cycle_enabled, color_phase_cycle_enabled, false);
    set_if!(p, locks, palette_mode,             palette_mode,             PaletteMode::Off);
    set_if!(p, locks, ribbons_enabled,          ribbons_enabled,          false);
    set_if!(p, locks, distortion_enabled,       distortion_enabled,       false);
    set_if!(p, locks, distortion_plus_enabled,  distortion_plus_enabled,  false);
    set_if!(p, locks, blackhole_enabled,        blackhole_enabled,        true);
    set_if!(p, locks, blackhole_warp_strength,  blackhole_warp_strength,  0.90);
    set_if!(p, locks, blackhole_warp_curve,     blackhole_warp_curve,     0.96);
    set_if!(p, locks, blackhole_alpha_radius,   blackhole_alpha_radius,   0.45);
    set_if!(p, locks, blackhole_wander_amount,  blackhole_wander_amount,  0.003);
    set_if!(p, locks, phantom_enabled,          phantom_enabled,          false);
    set_if!(p, locks, bezold_enabled,           bezold_enabled,           false);
    set_if!(p, locks, contrast,                 contrast,                 0.85);
    set_if!(p, locks, saturation,               saturation,               0.9);
}

fn apply_prism(p: &mut VisualParams, locks: &crate::ParamLocks) {
    set_if!(p, locks, color_harmony,           color_harmony,           ColorHarmony::Complementary);
    set_if!(p, locks, color_anchor_hue,         color_anchor_hue,         0.0);
    set_if!(p, locks, color_temperature_bias,   color_temperature_bias,   0.0);
    set_if!(p, locks, color_temperature_audio,  color_temperature_audio,  0.3);
    set_if!(p, locks, color_saturation_mode,    color_saturation_mode,    SaturationMode::Pure);
    set_if!(p, locks, color_saturation,         color_saturation,         0.9);
    set_if!(p, locks, color_value_key,          color_value_key,          ValueKey::Free);
    set_if!(p, locks, color_value,              color_value,              0.85);
    set_if!(p, locks, color_harmony_strength,   color_harmony_strength,   0.7);
    set_if!(p, locks, color_phase_cycle_enabled, color_phase_cycle_enabled, true);
    set_if!(p, locks, color_phase_cycle_degrees, color_phase_cycle_degrees, 360.0);
    set_if!(p, locks, color_phase_cycle_locked,  color_phase_cycle_locked,  false);
    set_if!(p, locks, palette_mode,             palette_mode,             PaletteMode::Off);
    set_if!(p, locks, contrast,                 contrast,                 1.25);
    set_if!(p, locks, saturation,               saturation,               1.1);
    set_if!(p, locks, ribbons_enabled,          ribbons_enabled,          false);
    set_if!(p, locks, distortion_enabled,       distortion_enabled,       false);
    set_if!(p, locks, distortion_plus_enabled,  distortion_plus_enabled,  false);
    set_if!(p, locks, blackhole_enabled,        blackhole_enabled,        false);
    set_if!(p, locks, phantom_enabled,          phantom_enabled,          false);
    set_if!(p, locks, bezold_enabled,           bezold_enabled,           true);
    set_if!(p, locks, bezold_strength,          bezold_strength,          0.65);
    set_if!(p, locks, bezold_radius,            bezold_radius,            3.0);
}

fn apply_organic(p: &mut VisualParams, locks: &crate::ParamLocks) {
    set_if!(p, locks, color_harmony,           color_harmony,           ColorHarmony::Analogous);
    set_if!(p, locks, color_anchor_hue,         color_anchor_hue,         30.0);
    set_if!(p, locks, color_temperature_bias,   color_temperature_bias,   0.5);
    set_if!(p, locks, color_temperature_audio,  color_temperature_audio,  0.2);
    set_if!(p, locks, color_saturation_mode,    color_saturation_mode,    SaturationMode::Free);
    set_if!(p, locks, color_saturation,         color_saturation,         0.7);
    set_if!(p, locks, color_value_key,          color_value_key,          ValueKey::Free);
    set_if!(p, locks, color_value,              color_value,              0.75);
    set_if!(p, locks, color_harmony_strength,   color_harmony_strength,   0.6);
    set_if!(p, locks, color_phase_cycle_enabled, color_phase_cycle_enabled, false);
    set_if!(p, locks, palette_mode,             palette_mode,             PaletteMode::Earth);
    set_if!(p, locks, palette_tint,             palette_tint,             0.6);
    set_if!(p, locks, ribbons_enabled,          ribbons_enabled,          true);
    set_if!(p, locks, ribbons_intensity,        ribbons_intensity,        0.55);
    set_if!(p, locks, distortion_enabled,       distortion_enabled,       true);
    set_if!(p, locks, distortion_amplitude,     distortion_amplitude,     0.04);
    set_if!(p, locks, distortion_frequency,     distortion_frequency,     2.5);
    set_if!(p, locks, distortion_plus_enabled,  distortion_plus_enabled,  false);
    set_if!(p, locks, blackhole_enabled,        blackhole_enabled,        false);
    set_if!(p, locks, phantom_enabled,          phantom_enabled,          false);
    set_if!(p, locks, bezold_enabled,           bezold_enabled,           false);
    set_if!(p, locks, contrast,                 contrast,                 1.0);
    set_if!(p, locks, saturation,               saturation,               1.0);
}

fn apply_neon(p: &mut VisualParams, locks: &crate::ParamLocks) {
    set_if!(p, locks, color_harmony,           color_harmony,           ColorHarmony::Tetradic);
    set_if!(p, locks, color_anchor_hue,         color_anchor_hue,         280.0);
    set_if!(p, locks, color_temperature_bias,   color_temperature_bias,   0.0);
    set_if!(p, locks, color_temperature_audio,  color_temperature_audio,  0.0);
    set_if!(p, locks, color_saturation_mode,    color_saturation_mode,    SaturationMode::Pure);
    set_if!(p, locks, color_saturation,         color_saturation,         1.0);
    set_if!(p, locks, color_value_key,          color_value_key,          ValueKey::Free);
    set_if!(p, locks, color_value,              color_value,              0.95);
    set_if!(p, locks, color_harmony_strength,   color_harmony_strength,   0.9);
    set_if!(p, locks, color_phase_cycle_enabled, color_phase_cycle_enabled, false);
    set_if!(p, locks, palette_mode,             palette_mode,             PaletteMode::Neon);
    set_if!(p, locks, palette_tint,             palette_tint,             0.5);
    set_if!(p, locks, contrast,                 contrast,                 1.4);
    set_if!(p, locks, saturation,               saturation,               1.2);
    set_if!(p, locks, ribbons_enabled,          ribbons_enabled,          true);
    set_if!(p, locks, ribbons_intensity,        ribbons_intensity,        0.7);
    set_if!(p, locks, distortion_enabled,       distortion_enabled,       false);
    set_if!(p, locks, distortion_plus_enabled,  distortion_plus_enabled,  false);
    set_if!(p, locks, blackhole_enabled,        blackhole_enabled,        false);
    set_if!(p, locks, phantom_enabled,          phantom_enabled,          false);
    set_if!(p, locks, bezold_enabled,           bezold_enabled,           true);
    set_if!(p, locks, bezold_strength,          bezold_strength,          0.8);
    set_if!(p, locks, bezold_radius,            bezold_radius,            2.0);
}

fn apply_frost(p: &mut VisualParams, locks: &crate::ParamLocks) {
    set_if!(p, locks, color_harmony,           color_harmony,           ColorHarmony::Analogous);
    set_if!(p, locks, color_anchor_hue,         color_anchor_hue,         200.0);
    set_if!(p, locks, color_temperature_bias,   color_temperature_bias,   -0.7);
    set_if!(p, locks, color_temperature_audio,  color_temperature_audio,  0.0);
    set_if!(p, locks, color_saturation_mode,    color_saturation_mode,    SaturationMode::Muted);
    set_if!(p, locks, color_saturation,         color_saturation,         0.5);
    set_if!(p, locks, color_value_key,          color_value_key,          ValueKey::Free);
    set_if!(p, locks, color_value,              color_value,              0.88);
    set_if!(p, locks, color_harmony_strength,   color_harmony_strength,   0.5);
    set_if!(p, locks, color_phase_cycle_enabled, color_phase_cycle_enabled, false);
    set_if!(p, locks, palette_mode,             palette_mode,             PaletteMode::Cool);
    set_if!(p, locks, palette_tint,             palette_tint,             0.4);
    set_if!(p, locks, contrast,                 contrast,                 0.9);
    set_if!(p, locks, saturation,               saturation,               0.85);
    set_if!(p, locks, ribbons_enabled,          ribbons_enabled,          false);
    set_if!(p, locks, distortion_enabled,       distortion_enabled,       false);
    set_if!(p, locks, distortion_plus_enabled,  distortion_plus_enabled,  false);
    set_if!(p, locks, blackhole_enabled,        blackhole_enabled,        false);
    set_if!(p, locks, phantom_enabled,          phantom_enabled,          true);
    set_if!(p, locks, phantom_delay_seconds,    phantom_delay_seconds,    0.5);
    set_if!(p, locks, phantom_key_tolerance,    phantom_key_tolerance,    0.12);
    set_if!(p, locks, phantom_key_softness,     phantom_key_softness,     0.06);
    set_if!(p, locks, phantom_key_strength,     phantom_key_strength,     0.9);
    set_if!(p, locks, phantom_opacity,          phantom_opacity,          0.7);
    set_if!(p, locks, bezold_enabled,           bezold_enabled,           false);
}

fn apply_inferno(p: &mut VisualParams, locks: &crate::ParamLocks) {
    set_if!(p, locks, color_harmony,           color_harmony,           ColorHarmony::Complementary);
    set_if!(p, locks, color_anchor_hue,         color_anchor_hue,         15.0);
    set_if!(p, locks, color_temperature_bias,   color_temperature_bias,   0.8);
    set_if!(p, locks, color_temperature_audio,  color_temperature_audio,  0.5);
    set_if!(p, locks, color_saturation_mode,    color_saturation_mode,    SaturationMode::Free);
    set_if!(p, locks, color_saturation,         color_saturation,         0.9);
    set_if!(p, locks, color_value_key,          color_value_key,          ValueKey::Free);
    set_if!(p, locks, color_value,              color_value,              0.8);
    set_if!(p, locks, color_harmony_strength,   color_harmony_strength,   0.75);
    set_if!(p, locks, color_phase_cycle_enabled, color_phase_cycle_enabled, false);
    set_if!(p, locks, palette_mode,             palette_mode,             PaletteMode::Warm);
    set_if!(p, locks, palette_tint,             palette_tint,             0.5);
    set_if!(p, locks, contrast,                 contrast,                 1.2);
    set_if!(p, locks, saturation,               saturation,               1.1);
    set_if!(p, locks, ribbons_enabled,          ribbons_enabled,          true);
    set_if!(p, locks, ribbons_intensity,        ribbons_intensity,        0.65);
    set_if!(p, locks, distortion_enabled,       distortion_enabled,       true);
    set_if!(p, locks, distortion_amplitude,     distortion_amplitude,     0.06);
    set_if!(p, locks, distortion_frequency,     distortion_frequency,     3.5);
    set_if!(p, locks, distortion_plus_enabled,  distortion_plus_enabled,  false);
    set_if!(p, locks, blackhole_enabled,        blackhole_enabled,        false);
    set_if!(p, locks, phantom_enabled,          phantom_enabled,          false);
    set_if!(p, locks, bezold_enabled,           bezold_enabled,           true);
    set_if!(p, locks, bezold_strength,          bezold_strength,          0.4);
    set_if!(p, locks, bezold_radius,            bezold_radius,            4.0);
    set_if!(p, locks, audio_route_shape,        audio_route_shape,        BeatRoute::Low);
    set_if!(p, locks, audio_route_kaleido,      audio_route_kaleido,      BeatRoute::Low);
}

fn apply_pastel(p: &mut VisualParams, locks: &crate::ParamLocks) {
    set_if!(p, locks, color_harmony,           color_harmony,           ColorHarmony::Triadic);
    set_if!(p, locks, color_anchor_hue,         color_anchor_hue,         300.0);
    set_if!(p, locks, color_temperature_bias,   color_temperature_bias,   0.1);
    set_if!(p, locks, color_temperature_audio,  color_temperature_audio,  0.0);
    set_if!(p, locks, color_saturation_mode,    color_saturation_mode,    SaturationMode::Muted);
    set_if!(p, locks, color_saturation,         color_saturation,         0.35);
    set_if!(p, locks, color_value_key,          color_value_key,          ValueKey::High);
    set_if!(p, locks, color_value,              color_value,              0.92);
    set_if!(p, locks, color_harmony_strength,   color_harmony_strength,   0.6);
    set_if!(p, locks, color_phase_cycle_enabled, color_phase_cycle_enabled, false);
    set_if!(p, locks, palette_mode,             palette_mode,             PaletteMode::Off);
    set_if!(p, locks, contrast,                 contrast,                 0.85);
    set_if!(p, locks, saturation,               saturation,               0.8);
    set_if!(p, locks, ribbons_enabled,          ribbons_enabled,          false);
    set_if!(p, locks, distortion_enabled,       distortion_enabled,       false);
    set_if!(p, locks, distortion_plus_enabled,  distortion_plus_enabled,  false);
    set_if!(p, locks, blackhole_enabled,        blackhole_enabled,        false);
    set_if!(p, locks, phantom_enabled,          phantom_enabled,          false);
    set_if!(p, locks, bezold_enabled,           bezold_enabled,           false);
}

fn apply_void(p: &mut VisualParams, locks: &crate::ParamLocks) {
    set_if!(p, locks, color_harmony,           color_harmony,           ColorHarmony::Monochromatic);
    set_if!(p, locks, color_anchor_hue,         color_anchor_hue,         270.0);
    set_if!(p, locks, color_temperature_bias,   color_temperature_bias,   -0.3);
    set_if!(p, locks, color_temperature_audio,  color_temperature_audio,  0.0);
    set_if!(p, locks, color_saturation_mode,    color_saturation_mode,    SaturationMode::Free);
    set_if!(p, locks, color_saturation,         color_saturation,         0.2);
    set_if!(p, locks, color_value_key,          color_value_key,          ValueKey::Free);
    set_if!(p, locks, color_value,              color_value,              0.12);
    set_if!(p, locks, color_harmony_strength,   color_harmony_strength,   1.0);
    set_if!(p, locks, color_phase_cycle_enabled, color_phase_cycle_enabled, false);
    set_if!(p, locks, palette_mode,             palette_mode,             PaletteMode::Off);
    set_if!(p, locks, contrast,                 contrast,                 0.7);
    set_if!(p, locks, saturation,               saturation,               0.6);
    set_if!(p, locks, ribbons_enabled,          ribbons_enabled,          false);
    set_if!(p, locks, distortion_enabled,       distortion_enabled,       false);
    set_if!(p, locks, distortion_plus_enabled,  distortion_plus_enabled,  true);
    set_if!(p, locks, distortion_plus_yaw,      distortion_plus_yaw,      2.5);
    set_if!(p, locks, distortion_plus_pitch,    distortion_plus_pitch,    1.0);
    set_if!(p, locks, distortion_plus_roll,     distortion_plus_roll,     0.5);
    set_if!(p, locks, blackhole_enabled,        blackhole_enabled,        true);
    set_if!(p, locks, blackhole_warp_strength,  blackhole_warp_strength,  0.95);
    set_if!(p, locks, blackhole_warp_curve,     blackhole_warp_curve,     0.98);
    set_if!(p, locks, blackhole_alpha_radius,   blackhole_alpha_radius,   0.3);
    set_if!(p, locks, blackhole_wander_amount,  blackhole_wander_amount,  0.001);
    set_if!(p, locks, phantom_enabled,          phantom_enabled,          false);
    set_if!(p, locks, bezold_enabled,           bezold_enabled,           false);
}

fn apply_carnival(p: &mut VisualParams, locks: &crate::ParamLocks) {
    set_if!(p, locks, color_harmony,           color_harmony,           ColorHarmony::Tetradic);
    set_if!(p, locks, color_anchor_hue,         color_anchor_hue,         45.0);
    set_if!(p, locks, color_temperature_bias,   color_temperature_bias,   0.2);
    set_if!(p, locks, color_temperature_audio,  color_temperature_audio,  0.3);
    set_if!(p, locks, color_saturation_mode,    color_saturation_mode,    SaturationMode::Pure);
    set_if!(p, locks, color_saturation,         color_saturation,         0.95);
    set_if!(p, locks, color_value_key,          color_value_key,          ValueKey::Free);
    set_if!(p, locks, color_value,              color_value,              0.9);
    set_if!(p, locks, color_harmony_strength,   color_harmony_strength,   0.8);
    set_if!(p, locks, color_phase_cycle_enabled, color_phase_cycle_enabled, true);
    set_if!(p, locks, color_phase_cycle_degrees, color_phase_cycle_degrees, 720.0);
    set_if!(p, locks, color_phase_cycle_locked,  color_phase_cycle_locked,  false);
    set_if!(p, locks, palette_mode,             palette_mode,             PaletteMode::Off);
    set_if!(p, locks, contrast,                 contrast,                 1.15);
    set_if!(p, locks, saturation,               saturation,               1.1);
    set_if!(p, locks, ribbons_enabled,          ribbons_enabled,          true);
    set_if!(p, locks, ribbons_intensity,        ribbons_intensity,        0.8);
    set_if!(p, locks, distortion_enabled,       distortion_enabled,       false);
    set_if!(p, locks, distortion_plus_enabled,  distortion_plus_enabled,  false);
    set_if!(p, locks, blackhole_enabled,        blackhole_enabled,        false);
    set_if!(p, locks, phantom_enabled,          phantom_enabled,          false);
    set_if!(p, locks, bezold_enabled,           bezold_enabled,           true);
    set_if!(p, locks, bezold_strength,          bezold_strength,          0.5);
    set_if!(p, locks, bezold_radius,            bezold_radius,            3.0);
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_has_ten_bundles() {
        assert_eq!(ALL.len(), 10);
    }

    #[test]
    fn all_variants_have_unique_names() {
        let names: Vec<_> = ALL.iter().map(|b| b.name()).collect();
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(names.len(), unique.len());
    }

    #[test]
    fn defaults_bundle_resets_to_factory() {
        let mut p = VisualParams { color_anchor_hue: 180.0, bezold_enabled: true, ..VisualParams::default() };
        let locks = crate::ParamLocks::default();
        BundleId::Defaults.apply(&mut p, &locks);
        assert_eq!(p.color_anchor_hue, 210.0);
        assert!(!p.bezold_enabled);
    }

    #[test]
    fn locks_prevent_apply() {
        let mut p = VisualParams::default();
        let locks = crate::ParamLocks { color_anchor_hue: true, ..crate::ParamLocks::default() };
        BundleId::Midnight.apply(&mut p, &locks);
        // anchor hue locked — should stay at default 210°, not Midnight's 240°
        assert_eq!(p.color_anchor_hue, 210.0);
        // but other unlocked params should have changed
        assert!(p.blackhole_enabled);
    }
}

// Continuous audio-reactive cell behavior driven by deck's 8-band
// analysis and beat envelopes.
#![allow(dead_code)]
//
// Design (phase 4b):
//   - Each cell has a preferred band index, fixed at init. Y-slab
//     determines the primary band; a per-cell hash perturbs it by up
//     to ±1 band so cells within a slab respond differently.
//   - Opacity continuously tracks the preferred band's smoothed level.
//   - Hue rotates around baseline color driven by beat envelopes,
//     low-pass filtered for silky glides.

use glam::Vec3;
use crate::cell::CellGrid;
use crate::influencer::{AudioSnapshot, Influencer};
use crate::myocyte_color::{hsv_to_rgb, rgb_to_hsv};

// ---- Opacity tuning -------------------------------------------------

/// Baseline visibility at zero band level. Keeps the grid dimly visible
/// when audio is silent so the user sees something before sound starts.
const BASELINE: f32 = 0.12;

/// How much each band level contributes on top of the baseline.
/// BASELINE + RESPONSE ≤ 1.0 so the original opacity is the ceiling.
const RESPONSE: f32 = 0.88;

// ---- Band assignment ------------------------------------------------

/// Per-cell band jitter range. A cell at Y=4 has primary band 2 but
/// may actually respond to bands 1, 2, or 3 with the strongest
/// weighting on 2. ±1.0 means "up to one band off in either direction
/// at full jitter."
const BAND_JITTER: f32 = 1.0;

// ---- Hue rotation ---------------------------------------------------

/// Max hue rotation in degrees at full beat energy.
/// 50° is bold but not whiplash; tweak if it feels too aggressive.
const HUE_SWING_DEG: f32 = 50.0;

/// Low-pass smoothing rate (e-folds/s) for the beat-energy hue drive.
/// 10.0 ≈ 100 ms time constant — fast enough to follow beats, slow
/// enough to avoid color jitter on micro-fluctuations.
const HUE_SMOOTH_PER_S: f32 = 10.0;

// ---- Implementation -------------------------------------------------

pub struct AudioCells {
    // Captured at init so we modulate without losing the baseline.
    base_opacity:     Vec<f32>,
    base_color_inner: Vec<Vec3>,
    base_color_outer: Vec<Vec3>,

    /// Per-cell continuous band coordinate in [0.0, 7.999]. Used to
    /// look up the band level by linear interpolation between two
    /// adjacent bands. Fractional values produce smooth response
    /// across cells within a slab.
    band_coord: Vec<f32>,

    /// Smoothed beat-energy signal driving hue rotation.
    hue_drive: f32,

    initialized: bool,
}

impl AudioCells {
    pub fn new() -> Self {
        Self {
            base_opacity:     Vec::new(),
            base_color_inner: Vec::new(),
            base_color_outer: Vec::new(),
            band_coord:       Vec::new(),
            hue_drive:        0.0,
            initialized:      false,
        }
    }

    fn initialize(&mut self, grid: &CellGrid) {
        let n = grid.cells.len();
        self.base_opacity     = grid.cells.iter().map(|c| c.opacity).collect();
        self.base_color_inner = grid.cells.iter().map(|c| c.color_inner).collect();
        self.base_color_outer = grid.cells.iter().map(|c| c.color_outer).collect();

        let dims = grid.dims;
        let ny_max = dims[1].saturating_sub(1).max(1) as f32;

        self.band_coord = Vec::with_capacity(n);

        for x in 0..dims[0] {
            for y in 0..dims[1] {
                for z in 0..dims[2] {
                    // Primary band from Y position: y∈[0,16) → band∈[0,8).
                    // Linear, so y=0 → band 0, y=15 → band 7 (close to).
                    let primary = (y as f32 / ny_max) * 7.0;

                    // Per-cell jitter from (x,y,z) hash → small offset.
                    let h = hash3(x, y, z, 0x9E3779B9);
                    let jitter = (h * 2.0 - 1.0) * BAND_JITTER;

                    let band = (primary + jitter).clamp(0.0, 7.0);
                    self.band_coord.push(band);
                }
            }
        }

        self.initialized = true;
    }

    /// Sample the band level at this cell's preferred frequency via
    /// linear interpolation between two adjacent integer bands.
    #[inline]
    fn band_level(&self, cell_idx: usize, bands: &[f32; 8]) -> f32 {
        let coord = self.band_coord[cell_idx];
        let lo = coord.floor() as usize;    // 0..=7
        let hi = (lo + 1).min(7);            // clamped
        let t  = coord - lo as f32;
        bands[lo] * (1.0 - t) + bands[hi] * t
    }
}

impl Default for AudioCells {
    fn default() -> Self { Self::new() }
}

impl Influencer for AudioCells {
    // No-audio path: identity (don't touch cells).
    fn step(&mut self, _grid: &mut CellGrid, _dt: f32) {}

    fn step_with_audio(
        &mut self,
        grid:  &mut CellGrid,
        audio: &AudioSnapshot,
        dt:    f32,
    ) {
        if !self.initialized { self.initialize(grid); }

        // Hue drive: sum of three beat envelopes (low+mid+high; broadband
        // would double-count). Low-pass filter into self.hue_drive.
        let target = (audio.beat_decay_low
                    + audio.beat_decay_mid
                    + audio.beat_decay_high) / 3.0;
        let alpha = 1.0 - (-HUE_SMOOTH_PER_S * dt).exp();
        self.hue_drive += alpha * (target - self.hue_drive);

        let hue_rot = HUE_SWING_DEG * self.hue_drive;

        // Per-cell update.
        for (i, cell) in grid.cells.iter_mut().enumerate() {
            let level = self.band_level(i, &audio.bands);

            let opaq = BASELINE + RESPONSE * level;
            cell.opacity = (self.base_opacity[i] * opaq).clamp(0.0, 1.0);

            cell.color_inner = rotate_hue(self.base_color_inner[i], hue_rot);
            cell.color_outer = rotate_hue(self.base_color_outer[i], hue_rot);
        }
    }
}

// ---- Helpers ---------------------------------------------------------

#[inline]
fn rotate_hue(rgb: Vec3, delta_deg: f32) -> Vec3 {
    let (h, s, v) = rgb_to_hsv([rgb.x, rgb.y, rgb.z]);
    let out       = hsv_to_rgb(h + delta_deg, s, v);
    Vec3::new(out[0], out[1], out[2])
}

/// 3D integer-coordinate hash → uniform f32 in [0, 1). Copied from
/// myocyte's grid.rs to avoid coupling on a helper.
#[inline]
fn hash3(x: u32, y: u32, z: u32, seed: u32) -> f32 {
    let mut v = seed;
    v ^= x.wrapping_mul(0x9e3779b9);
    v  = v.rotate_left(5).wrapping_mul(0x85ebca6b);
    v ^= y.wrapping_mul(0xc2b2ae35);
    v  = v.rotate_left(7).wrapping_mul(0xcc9e2d51);
    v ^= z.wrapping_mul(0x6b3a36f5);
    v  = v.rotate_left(11).wrapping_mul(0x1b873593);
    v ^= v >> 16;
    v  = v.wrapping_mul(0x85ebca6b);
    v ^= v >> 13;
    v  = v.wrapping_mul(0xc2b2ae35);
    v ^= v >> 16;
    (v & 0x00FF_FFFF) as f32 / 16_777_216.0
}

/// Placeholder scaffolding retained from phase 4a. Not wired to any
/// ShapeKind — kept so the module compiles without removal churn.
/// Removed in phase 5 cleanup.
#[allow(dead_code)]
pub struct AudioCellsPlaceholder {
    debug_timer: f32,
}

#[allow(dead_code)]
impl AudioCellsPlaceholder {
    pub fn new() -> Self { Self { debug_timer: 0.0 } }
}

impl Default for AudioCellsPlaceholder {
    fn default() -> Self { Self::new() }
}

impl Influencer for AudioCellsPlaceholder {
    fn step(&mut self, _grid: &mut CellGrid, _dt: f32) {}

    fn step_with_audio(
        &mut self,
        _grid: &mut CellGrid,
        audio: &AudioSnapshot,
        dt:    f32,
    ) {
        self.debug_timer += dt;
        if self.debug_timer > 1.0 {
            self.debug_timer = 0.0;
            log::info!(
                "[myocyte/audio] bands=[{:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2}] \
                 beat_low={:.2} beat_mid={:.2} beat_high={:.2} bpm={:?}",
                audio.bands[0], audio.bands[1], audio.bands[2], audio.bands[3],
                audio.bands[4], audio.bands[5], audio.bands[6], audio.bands[7],
                audio.beat_decay_low, audio.beat_decay_mid, audio.beat_decay_high,
                audio.bpm,
            );
        }
    }
}

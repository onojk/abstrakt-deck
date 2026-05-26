// Audio-reactive influencer that assigns each myocyte cell a hue derived
// from its radial angle around the vertical (Y) axis of the grid, turning
// the 16³ cube into a 3D color wheel. Band-driven opacity keeps the
// frequency response. A slow global hue drift makes the wheel breathe in
// silence and accelerates on beats.
//
// Design:
//   - base_hue[i]: angle of cell i in the XZ plane, −180..180 → 0..360°.
//     Fixed at init; cells on opposite sides carry opposite hues.
//   - band_coord[i]: Y-slab → preferred band index + per-cell jitter.
//     Same assignment as AudioCells so frequency bands still light up
//     horizontal slabs of the cube.
//   - hue_phase: global offset that drifts at DRIFT_DEG_PER_S + a
//     beat-energy boost. Each cell's final hue = base_hue[i] + hue_phase.

use glam::Vec3;
use crate::cell::CellGrid;
use crate::influencer::{AudioSnapshot, Influencer};
use crate::myocyte_color::hsv_to_rgb;

// ---- Opacity tuning -------------------------------------------------

/// Baseline visibility at zero band level. Keeps the wheel dimly lit
/// when audio is silent so the color structure is always visible.
const BASELINE: f32 = 0.12;

/// Band-level multiplier on top of baseline.
/// BASELINE + RESPONSE ≤ 1.0 so original opacity is the ceiling.
const RESPONSE: f32 = 0.88;

// ---- Band assignment ------------------------------------------------

/// Per-cell band jitter range. Matches AudioCells so horizontal slabs
/// respond to the same frequency regions as before.
const BAND_JITTER: f32 = 1.0;

// ---- Color -----------------------------------------------------

/// Inner-cell saturation and value. High saturation, full brightness —
/// the vivid core of each splat.
const INNER_SAT: f32 = 0.90;
const INNER_VAL: f32 = 1.00;

/// Outer-cell saturation and value. Darker and slightly desaturated so
/// the inner→outer gradient in myocyte_splat.wgsl still reads clearly.
const OUTER_SAT: f32 = 0.70;
const OUTER_VAL: f32 = 0.40;

// ---- Hue drift ------------------------------------------------------

/// Base hue rotation rate in degrees per second. One full revolution
/// every 60 s in silence — slow enough to feel like breathing.
const DRIFT_DEG_PER_S: f32 = 6.0;

/// Additional degrees per second added at full beat energy (hue_drive=1).
/// 30°/s means the wheel spins ~5× faster on a hard beat hit.
const BEAT_BOOST_DEG_PER_S: f32 = 30.0;

/// Low-pass smoothing rate (e-folds/s) for the beat-energy hue boost.
/// 3.0 ≈ 333 ms time constant — slower than AudioCells' hue drive so
/// the rotation accelerates and decelerates with a smooth glide rather
/// than a sharp twitch.
const BEAT_SMOOTH_PER_S: f32 = 3.0;

// ---- Implementation -------------------------------------------------

pub struct AudioWheel {
    /// Base opacity captured from the grid at init.
    base_opacity: Vec<f32>,

    /// Per-cell radial angle mapped to [0, 360°). Fixed after init.
    base_hue: Vec<f32>,

    /// Per-cell continuous band coordinate in [0.0, 7.999]. Same Y→band
    /// + per-cell jitter logic as AudioCells.
    band_coord: Vec<f32>,

    /// Accumulated global hue rotation (degrees). Wrapped each frame to
    /// stay in [0, 360) so the value stays finite over long sessions.
    hue_phase: f32,

    /// Smoothed beat-energy signal driving the rotation boost.
    hue_drive: f32,

    initialized: bool,
}

impl AudioWheel {
    pub fn new() -> Self {
        Self {
            base_opacity: Vec::new(),
            base_hue:     Vec::new(),
            band_coord:   Vec::new(),
            hue_phase:    0.0,
            hue_drive:    0.0,
            initialized:  false,
        }
    }

    fn initialize(&mut self, grid: &CellGrid) {
        let n = grid.cells.len();
        self.base_opacity = grid.cells.iter().map(|c| c.opacity).collect();

        let dims = grid.dims;
        let ny_max = dims[1].saturating_sub(1).max(1) as f32;

        self.base_hue   = Vec::with_capacity(n);
        self.band_coord = Vec::with_capacity(n);

        // Iterate in (x, y, z) order — matches grid.idx and the enumeration
        // in step_with_audio so base_hue[i] and band_coord[i] align with
        // grid.cells[i].
        for x in 0..dims[0] {
            for y in 0..dims[1] {
                for z in 0..dims[2] {
                    // Spatial hue: radial angle in the XZ plane around Y axis.
                    // Positions are already centered at the grid origin.
                    let pos = grid.cells[grid.idx(x, y, z)].position;
                    let deg = pos.z.atan2(pos.x).to_degrees();
                    self.base_hue.push(if deg < 0.0 { deg + 360.0 } else { deg });

                    // Band assignment: Y-slab primary + per-cell jitter.
                    let primary = (y as f32 / ny_max) * 7.0;
                    let h = hash3(x, y, z, 0x9E3779B9);
                    let jitter = (h * 2.0 - 1.0) * BAND_JITTER;
                    self.band_coord.push((primary + jitter).clamp(0.0, 7.0));
                }
            }
        }

        self.initialized = true;
    }

    /// Sample the band level for this cell via linear interpolation
    /// between two adjacent integer bands.
    #[inline]
    fn band_level(&self, cell_idx: usize, bands: &[f32; 8]) -> f32 {
        let coord = self.band_coord[cell_idx];
        let lo = coord.floor() as usize;
        let hi = (lo + 1).min(7);
        let t  = coord - lo as f32;
        bands[lo] * (1.0 - t) + bands[hi] * t
    }
}

impl Default for AudioWheel {
    fn default() -> Self { Self::new() }
}

impl Influencer for AudioWheel {
    // No-audio path: identity. Cells keep whatever colors they already have.
    fn step(&mut self, _grid: &mut CellGrid, _dt: f32) {}

    fn step_with_audio(
        &mut self,
        grid:  &mut CellGrid,
        audio: &AudioSnapshot,
        dt:    f32,
    ) {
        if !self.initialized { self.initialize(grid); }

        // LP-filter beat energy into a smooth drive signal.
        let beat_energy = (audio.beat_decay_low
                         + audio.beat_decay_mid
                         + audio.beat_decay_high) / 3.0;
        let alpha = 1.0 - (-BEAT_SMOOTH_PER_S * dt).exp();
        self.hue_drive += alpha * (beat_energy - self.hue_drive);

        // Advance the color wheel. Wrap to [0, 360) to stay finite.
        self.hue_phase = (self.hue_phase
            + (DRIFT_DEG_PER_S + BEAT_BOOST_DEG_PER_S * self.hue_drive) * dt)
            .rem_euclid(360.0);

        // Per-cell update.
        for (i, cell) in grid.cells.iter_mut().enumerate() {
            let level = self.band_level(i, &audio.bands);

            let opaq = BASELINE + RESPONSE * level;
            cell.opacity = (self.base_opacity[i] * opaq).clamp(0.0, 1.0);

            let hue = self.base_hue[i] + self.hue_phase;
            let [ir, ig, ib] = hsv_to_rgb(hue, INNER_SAT, INNER_VAL);
            let [or_, og, ob] = hsv_to_rgb(hue, OUTER_SAT, OUTER_VAL);
            cell.color_inner = Vec3::new(ir, ig, ib);
            cell.color_outer = Vec3::new(or_, og, ob);
        }
    }
}

// ---- Helpers ---------------------------------------------------------

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

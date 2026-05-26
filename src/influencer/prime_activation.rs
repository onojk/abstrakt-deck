// Audio-reactive influencer for the PrimeHelix shape.
//
// Behaviour:
//   A brightness wavefront sweeps upward along the Y (theta) axis, driven by
//   beat energy. Cells below the wavefront are fully lit; the wavefront band
//   itself glows brightest; cells above are dim. A per-cell band response
//   (low Y = bass, high Y = treble) makes the cylinder also act as a vertical
//   spectrum analyzer wrapped on the helix. The amber spine throbs with
//   beat_decay_max. The wavefront drifts upward in silence so the shape is
//   never frozen.

use glam::Vec3;
use crate::cell::CellGrid;
use crate::influencer::{AudioSnapshot, Influencer};

// ── Opacity tuning ────────────────────────────────────────────────────────────

/// Minimum opacity for cells far ABOVE the wavefront (nearly invisible).
const DIM_OPACITY: f32 = 0.05;

/// Target opacity for cells fully BELOW the wavefront.
const ACTIVATED_OPACITY: f32 = 0.80;

/// Extra opacity boost right AT the wavefront edge.
const WAVEFRONT_GLOW: f32 = 0.35;

/// Half-width of the Gaussian glow peak around the wavefront (world units).
const GLOW_HALF_WIDTH: f32 = 0.8;

/// Sharpness of the sigmoid transition from dim to activated.
/// Higher = sharper band. 1.5 gives a soft ~2-unit-wide falloff.
const SIGMOID_SHARP: f32 = 1.5;

// ── Band response ─────────────────────────────────────────────────────────────

/// Per-cell band level adds this much extra opacity.
const BAND_FACTOR: f32 = 0.20;

// ── Wavefront motion ──────────────────────────────────────────────────────────

/// Base upward drift rate (world units per second) in silence.
const WAVE_BASE_SPEED: f32 = 0.60;

/// Extra speed at full beat energy.
const WAVE_BEAT_SPEED: f32 = 3.0;

/// Low-pass smoothing for beat_drive (e-folds/s). Faster than AudioWheel
/// so the wavefront accelerates and decelerates crisply on beats.
const BEAT_SMOOTH_PER_S: f32 = 4.0;

// ── Spine tuning ──────────────────────────────────────────────────────────────

/// Baseline spine brightness multiplier (relative to base_opacity).
const SPINE_BASE: f32 = 0.70;

/// Maximum extra brightness added to spine at full beat_decay_max.
const SPINE_BOOST: f32 = 0.40;

// ── Spine cell detection ──────────────────────────────────────────────────────

/// A cell with |x| and |z| both less than this is a spine cell.
const SPINE_RADIUS: f32 = 0.05;

// ── Implementation ────────────────────────────────────────────────────────────

pub struct PrimeActivation {
    /// Captured base opacities at init.
    base_opacity: Vec<f32>,

    /// Per-cell Y positions (used for wavefront distance and band mapping).
    cell_y: Vec<f32>,

    /// Per-cell continuous band coordinate [0.0, 7.999] — low Y = bass.
    band_coord: Vec<f32>,

    /// Whether each cell is a spine cell (x≈0, z≈0).
    is_spine: Vec<bool>,

    /// Current wavefront height (world Y). Advances upward each frame.
    wavefront_y: f32,

    /// Smoothed beat energy driving wavefront acceleration.
    beat_drive: f32,

    /// Y bounds computed from the grid at init.
    y_min: f32,
    y_max: f32,

    initialized: bool,
}

impl PrimeActivation {
    pub fn new() -> Self {
        Self {
            base_opacity:  Vec::new(),
            cell_y:        Vec::new(),
            band_coord:    Vec::new(),
            is_spine:      Vec::new(),
            wavefront_y:   0.0,
            beat_drive:    0.0,
            y_min:         -3.0,
            y_max:          4.0,
            initialized:   false,
        }
    }

    fn initialize(&mut self, grid: &CellGrid) {
        let n = grid.cells.len();
        self.base_opacity = grid.cells.iter().map(|c| c.opacity).collect();
        self.cell_y       = grid.cells.iter().map(|c| c.position.y).collect();
        self.is_spine     = grid.cells.iter().map(|c| {
            c.position.x.abs() < SPINE_RADIUS && c.position.z.abs() < SPINE_RADIUS
        }).collect();

        // Y range from helix cells only (spine spans a slightly wider range).
        let helix_ys: Vec<f32> = grid.cells.iter()
            .filter(|c| c.position.x.abs() >= SPINE_RADIUS || c.position.z.abs() >= SPINE_RADIUS)
            .map(|c| c.position.y)
            .collect();
        if !helix_ys.is_empty() {
            self.y_min = helix_ys.iter().cloned().fold(f32::INFINITY, f32::min);
            self.y_max = helix_ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        }

        // Start wavefront below the helix so the first sweep is visible.
        self.wavefront_y = self.y_min - 1.0;

        // Band assignment: low Y → bass (band 0), high Y → treble (band 7).
        let y_span = (self.y_max - self.y_min).max(1e-3);
        self.band_coord = Vec::with_capacity(n);
        for &y in &self.cell_y {
            let normalized = ((y - self.y_min) / y_span).clamp(0.0, 1.0);
            self.band_coord.push(normalized * 7.0);
        }

        self.initialized = true;
    }

    #[inline]
    fn band_level(&self, i: usize, bands: &[f32; 8]) -> f32 {
        let coord = self.band_coord[i];
        let lo = coord.floor() as usize;
        let hi = (lo + 1).min(7);
        let t  = coord - lo as f32;
        bands[lo] * (1.0 - t) + bands[hi] * t
    }
}

impl Default for PrimeActivation {
    fn default() -> Self { Self::new() }
}

impl Influencer for PrimeActivation {
    // No-audio path: advance wavefront at base speed only, no cell updates.
    fn step(&mut self, grid: &mut CellGrid, dt: f32) {
        if !self.initialized { self.initialize(grid); }
        self.wavefront_y += WAVE_BASE_SPEED * dt;
        // Wrap when wavefront clears the top.
        let wrap_top = self.y_max + 1.0;
        if self.wavefront_y > wrap_top {
            self.wavefront_y = self.y_min - 1.0;
        }
    }

    fn step_with_audio(
        &mut self,
        grid:  &mut CellGrid,
        audio: &AudioSnapshot,
        dt:    f32,
    ) {
        if !self.initialized { self.initialize(grid); }

        // Low-pass beat energy for smooth wavefront acceleration.
        let beat_energy = (audio.beat_decay_low
                         + audio.beat_decay_mid
                         + audio.beat_decay_high) / 3.0;
        let alpha = 1.0 - (-BEAT_SMOOTH_PER_S * dt).exp();
        self.beat_drive += alpha * (beat_energy - self.beat_drive);

        // Advance wavefront and wrap.
        self.wavefront_y += (WAVE_BASE_SPEED + WAVE_BEAT_SPEED * self.beat_drive) * dt;
        let wrap_top = self.y_max + 1.0;
        if self.wavefront_y > wrap_top {
            self.wavefront_y = self.y_min - 1.0;
        }

        // Spine brightness driven by beat_decay_max.
        let spine_opaq = SPINE_BASE + SPINE_BOOST * audio.beat_decay_max;
        let spine_boost_color = 1.0 + 0.25 * audio.beat_decay_max;

        let w = self.wavefront_y;

        for (i, cell) in grid.cells.iter_mut().enumerate() {
            let y = self.cell_y[i];

            if self.is_spine[i] {
                // Spine: throb with beat, keep amber color.
                cell.opacity = (self.base_opacity[i] * spine_opaq).clamp(0.0, 1.0);
                cell.color_inner = Vec3::new(
                    (1.0_f32 * spine_boost_color).min(1.0),
                    (208.0 / 255.0 * spine_boost_color).min(1.0),
                    (128.0 / 255.0 * spine_boost_color).min(1.0),
                );
                continue;
            }

            // Wavefront activation level: sigmoid transition + Gaussian glow peak.
            let d = w - y;  // positive = cell is below wavefront (activated)
            let sigmoid = 1.0 / (1.0 + (-SIGMOID_SHARP * d).exp());
            let glow    = WAVEFRONT_GLOW * (-d * d / (2.0 * GLOW_HALF_WIDTH * GLOW_HALF_WIDTH)).exp();
            let wave_level = (sigmoid + glow).clamp(0.0, 1.5);

            // Band level adds frequency-spectrum shimmer.
            let band = self.band_level(i, &audio.bands);
            let total = (wave_level + BAND_FACTOR * band).clamp(0.0, 1.2);

            let target_opaq = DIM_OPACITY + (ACTIVATED_OPACITY - DIM_OPACITY) * total;
            cell.opacity = (self.base_opacity[i] * target_opaq / ACTIVATED_OPACITY)
                .clamp(0.0, 1.0);
        }
    }
}

// Decorative fine-marking shape — ShapeKind::Accents.
//
// A purely ornamental field built from a fresh generative base: a set of
// concentric (slightly spiralling) rings of points. Each base point spawns a
// small CLUSTER of accent splats of one of four mark types — dot-rows,
// dots-within-dots, oblong ticks, and recursive self-similar marks — all fed
// into the SHARED splat renderer (one Cell19 per splat). Marks are colored in
// CONTRAST to a per-region base hue so they POP against whatever they fold over.
//
// No new pipeline/pass: the grid is rendered by MyocyteRenderer like Myocyte and
// PrimeHelix. The influencer only applies a subtle bass-driven global size pulse.

use std::f32::consts::TAU;
use glam::{Quat, Vec3};
use crate::cell::{Cell19, CellGrid};
use crate::color::hsv_to_rgb;
use crate::influencer::{AudioSnapshot, Influencer};

// ── Generative base structure ───────────────────────────────────────────────

/// Number of concentric rings the marks sit along.
pub const BASE_RINGS: usize = 7;
/// Points (mark anchors) per ring.
pub const BASE_POINTS_PER_RING: usize = 48;
/// Innermost / outermost ring radius in world units.
const RADIUS_MIN: f32 = 0.35;
const RADIUS_MAX: f32 = 1.55;
/// Angular phase added per ring so successive rings spiral rather than aligning
/// radially — gives the field an organic, non-gridded feel.
const RING_SPIRAL_PHASE: f32 = 0.40;

/// Splat-buffer headroom for this shape. Worst case is every base point emitting
/// a recursive mark (RECURSE worst case ≈ 13 splats):
///   BASE_RINGS × BASE_POINTS_PER_RING × 13 = 7×48×13 = 4368.
/// 8192 leaves comfortable margin and stays far under MYOCYTE_MAX_SPLATS (32768).
pub const ACCENTS_MAX_SPLATS: usize = 8192;

// ── Mark type 1: dot-row ────────────────────────────────────────────────────

/// Number of dots in a dot-row (short line stepping along the ring tangent).
const DOTROW_DOTS: usize = 5;
/// Spacing between consecutive dots, world units along the tangent.
const DOTROW_SPACING: f32 = 0.045;
/// Uniform radius (σ) of each dot in the row.
const DOTROW_DOT_SCALE: f32 = 0.022;

// ── Mark type 2: dots-within-dots ───────────────────────────────────────────

/// Outer (dim, large) and inner (bright, small) concentric dot radii.
const DID_OUTER_SCALE: f32 = 0.060;
const DID_INNER_SCALE: f32 = 0.024;
/// Brightness multipliers for the outer/inner dots (inner is the bright core).
const DID_OUTER_BRIGHT: f32 = 0.45;
const DID_INNER_BRIGHT: f32 = 1.30;

// ── Mark type 3: oblong tick ────────────────────────────────────────────────

/// Long axis (σ along radial) and short axis (σ across) of the tick. The long
/// axis is oriented RADIALLY so the tick reads as a stroke crossing the ring.
const TICK_LENGTH: f32 = 0.085;
const TICK_WIDTH:  f32 = 0.016;

// ── Mark type 4: recursive mark ─────────────────────────────────────────────

/// Levels of recursion (center → satellites → satellites-of-satellites).
const RECURSE_DEPTH: usize = 2;
/// Satellites placed around each parent dot.
const RECURSE_SATELLITES: usize = 4;
/// Radius (σ) of the top-level center dot.
const RECURSE_CENTER_SCALE: f32 = 0.040;
/// Orbit distance of the first satellite ring from its parent (world units).
const RECURSE_ORBIT: f32 = 0.070;
/// Per-level shrink factor applied to both dot size and orbit distance.
const RECURSE_CHILD_RATIO: f32 = 0.50;

// ── Contrast coloring ───────────────────────────────────────────────────────

/// DEBUG: force EVERY mark (all four types) to bright red on black so mark
/// placement is clearly visible on-device through the fold. Set false to restore
/// the real contrast coloring below (which stays fully intact behind this flag).
pub const DEBUG_RED: bool = true; // TODO: set false to restore real contrast coloring

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // both modes are selectable via CONTRAST_MODE below
enum ContrastMode {
    /// Mark hue = base hue + 180° (complementary pop).
    Complementary,
    /// Mark hue = base hue, contrast carried by bright core / dark halo only.
    ValueContrast,
}

/// Active contrast mode. Default COMPLEMENTARY per the design brief.
const CONTRAST_MODE: ContrastMode = ContrastMode::Complementary;

/// Vivid, high-saturation marks (PrimeHelix-style crisp emissive treatment).
const MARK_SATURATION: f32 = 1.0;
/// Value of the bright emissive inner core, and of the dark same-hue halo.
const CORE_VALUE: f32 = 1.0;
const HALO_VALUE: f32 = 0.12;
/// Base opacity for every accent splat.
const MARK_OPACITY: f32 = 0.92;

// ── Audio reactivity ────────────────────────────────────────────────────────

/// Bass → subtle global size pulse. Envelope attack/release expressed as e-folds
/// per second (= 1/time-constant): attack 0.30 s, release 0.05 s. Gain keeps it
/// subtle — the marks are the star, not the pulse.
const PULSE_ATTACK_PER_S:  f32 = 1.0 / 0.30;
const PULSE_RELEASE_PER_S: f32 = 1.0 / 0.05;
const PULSE_GAIN: f32 = 0.15;

// ── Color helpers ───────────────────────────────────────────────────────────

/// Compute (inner, outer) RGB for a mark given its region base hue and a
/// brightness multiplier (used to make some dots dimmer/brighter within a mark).
fn mark_colors(base_hue_deg: f32, bright: f32) -> (Vec3, Vec3) {
    // DEBUG override: bright-red emissive core + dark-red halo for every mark.
    // Keeps the bright-core/dark-halo shape and the per-dot `bright` scaling so
    // dots-within-dots still read; only the hue is forced. Real coloring below is
    // untouched and resumes when DEBUG_RED is false.
    if DEBUG_RED {
        let core = Vec3::new((1.0 * bright).min(1.0), 0.0, 0.0);
        let halo = Vec3::new(0.15, 0.0, 0.0);
        return (core, halo);
    }

    let mark_hue = match CONTRAST_MODE {
        ContrastMode::Complementary => base_hue_deg + 180.0,
        ContrastMode::ValueContrast => base_hue_deg,
    };
    let core = hsv_to_rgb(mark_hue, MARK_SATURATION, (CORE_VALUE * bright).min(1.0));
    let halo = hsv_to_rgb(mark_hue, MARK_SATURATION, HALO_VALUE);
    (Vec3::from(core), Vec3::from(halo))
}

/// Push one round dot (uniform scale) into the cell list.
#[allow(clippy::too_many_arguments)]
fn push_dot(cells: &mut Vec<Cell19>, pos: Vec3, scale: f32, base_hue: f32, bright: f32) {
    if cells.len() >= ACCENTS_MAX_SPLATS { return; }
    let (inner, outer) = mark_colors(base_hue, bright);
    cells.push(Cell19 {
        position:    pos,
        rotation:    Quat::IDENTITY,
        scale:       Vec3::splat(scale),
        falloff:     1.0,
        sharpness:   0.0,
        color_inner: inner,
        color_outer: outer,
        opacity:     MARK_OPACITY,
    });
}

/// Push one anisotropic oblong (elongated Gaussian) oriented by rotation about Z.
fn push_oblong(
    cells: &mut Vec<Cell19>, pos: Vec3, long: f32, short: f32, angle: f32,
    base_hue: f32, bright: f32,
) {
    if cells.len() >= ACCENTS_MAX_SPLATS { return; }
    let (inner, outer) = mark_colors(base_hue, bright);
    cells.push(Cell19 {
        position:    pos,
        rotation:    Quat::from_rotation_z(angle), // local +X (long) → radial dir
        scale:       Vec3::new(long, short, short),
        falloff:     1.0,
        sharpness:   0.0,
        color_inner: inner,
        color_outer: outer,
        opacity:     MARK_OPACITY,
    });
}

/// Recursively add a center dot + satellites (RECURSE_DEPTH levels).
fn push_recursive(
    cells: &mut Vec<Cell19>, center: Vec3, scale: f32, orbit: f32,
    depth: usize, base_hue: f32, phase: f32,
) {
    // Center dot (brighter at the top level, dimmer deeper).
    let bright = 0.6 + 0.4 * (depth as f32 / RECURSE_DEPTH.max(1) as f32);
    push_dot(cells, center, scale, base_hue, bright);
    if depth == 0 { return; }

    for s in 0..RECURSE_SATELLITES {
        let a = phase + s as f32 / RECURSE_SATELLITES as f32 * TAU;
        let sat = center + Vec3::new(a.cos() * orbit, a.sin() * orbit, 0.0);
        push_recursive(
            cells, sat,
            scale * RECURSE_CHILD_RATIO,
            orbit * RECURSE_CHILD_RATIO,
            depth - 1, base_hue, phase + 0.7,
        );
    }
}

// ── Grid builder ────────────────────────────────────────────────────────────

/// Build the CellGrid for the Accents shape. Caller owns the grid; no GPU work.
pub fn build_accents_grid() -> CellGrid {
    let mut cells: Vec<Cell19> = Vec::with_capacity(ACCENTS_MAX_SPLATS);

    for r in 0..BASE_RINGS {
        let radius = if BASE_RINGS == 1 {
            RADIUS_MIN
        } else {
            RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * r as f32 / (BASE_RINGS - 1) as f32
        };
        // Per-ring base hue, plus a spiral phase so rings don't align radially.
        let ring_hue   = 360.0 * r as f32 / BASE_RINGS as f32;
        let ring_phase = r as f32 * RING_SPIRAL_PHASE;

        for p in 0..BASE_POINTS_PER_RING {
            let theta = ring_phase + p as f32 / BASE_POINTS_PER_RING as f32 * TAU;
            let pos = Vec3::new(theta.cos() * radius, theta.sin() * radius, 0.0);

            // Tangent (toward next point on ring) and radial (outward) directions.
            let tangent = Vec3::new(-theta.sin(), theta.cos(), 0.0);
            // base hue varies across the field: ring band + slow angular sweep.
            let base_hue = ring_hue + theta.to_degrees() * 0.10;

            // Cycle mark type by global point index for an even mix per ring.
            let global_idx = r * BASE_POINTS_PER_RING + p;
            match global_idx % 4 {
                0 => {
                    // DOT-ROW: short line of dots stepping along the tangent.
                    let start = pos - tangent * (DOTROW_DOTS as f32 - 1.0) * 0.5 * DOTROW_SPACING;
                    for d in 0..DOTROW_DOTS {
                        let q = start + tangent * d as f32 * DOTROW_SPACING;
                        push_dot(&mut cells, q, DOTROW_DOT_SCALE, base_hue, 1.0);
                    }
                }
                1 => {
                    // DOTS-WITHIN-DOTS: dim large dot with a bright small core.
                    push_dot(&mut cells, pos, DID_OUTER_SCALE, base_hue, DID_OUTER_BRIGHT);
                    push_dot(&mut cells, pos, DID_INNER_SCALE, base_hue, DID_INNER_BRIGHT);
                }
                2 => {
                    // OBLONG TICK: elongated mark oriented radially (crosses the ring).
                    push_oblong(&mut cells, pos, TICK_LENGTH, TICK_WIDTH, theta, base_hue, 1.0);
                }
                _ => {
                    // RECURSIVE MARK: self-similar cluster, RECURSE_DEPTH levels.
                    push_recursive(
                        &mut cells, pos, RECURSE_CENTER_SCALE, RECURSE_ORBIT,
                        RECURSE_DEPTH, base_hue, theta,
                    );
                }
            }
        }
    }

    let total = cells.len() as u32;
    CellGrid { cells, dims: [total, 1, 1], spacing: 1.0 }
}

// ── Influencer: subtle bass-driven global size pulse ────────────────────────

pub struct Accents {
    /// Base (unpulsed) per-cell scales captured at init.
    base_scale: Vec<Vec3>,
    /// Envelope-followed bass level driving the size pulse.
    pulse: f32,
    initialized: bool,
}

impl Accents {
    pub fn new() -> Self {
        Self { base_scale: Vec::new(), pulse: 0.0, initialized: false }
    }

    fn initialize(&mut self, grid: &CellGrid) {
        self.base_scale = grid.cells.iter().map(|c| c.scale).collect();
        self.initialized = true;
    }

    /// Apply the current pulse factor to every cell's scale.
    fn apply_pulse(&self, grid: &mut CellGrid) {
        let factor = 1.0 + PULSE_GAIN * self.pulse;
        for (cell, &base) in grid.cells.iter_mut().zip(self.base_scale.iter()) {
            cell.scale = base * factor;
        }
    }
}

impl Default for Accents {
    fn default() -> Self { Self::new() }
}

impl Influencer for Accents {
    // No-audio path: relax the pulse toward zero so the field settles.
    fn step(&mut self, grid: &mut CellGrid, dt: f32) {
        if !self.initialized { self.initialize(grid); }
        let a = 1.0 - (-PULSE_RELEASE_PER_S * dt).exp();
        self.pulse += a * (0.0 - self.pulse);
        self.apply_pulse(grid);
    }

    fn step_with_audio(&mut self, grid: &mut CellGrid, audio: &AudioSnapshot, dt: f32) {
        if !self.initialized { self.initialize(grid); }

        // Bass target: low bands + low beat envelope, kept in a modest range.
        let target = (0.5 * (audio.bands[0] + audio.bands[1]) + audio.beat_decay_low)
            .clamp(0.0, 1.5);
        // Asymmetric envelope: fast attack when rising, slow release when falling.
        let rate = if target > self.pulse { PULSE_ATTACK_PER_S } else { PULSE_RELEASE_PER_S };
        let a = 1.0 - (-rate * dt).exp();
        self.pulse += a * (target - self.pulse);

        self.apply_pulse(grid);
    }
}

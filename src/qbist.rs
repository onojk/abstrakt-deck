// Qbist abstract texture generator.
// Algorithm: Jörn Loviscach, c't 1995. Clean-room implementation from spec.
//
// LICENSE: MIT
// No deck/wgpu/engine dependencies — portable to other engines.
// Only stdlib is used; all deck-specific glue lives outside this module.

// ── Constants ────────────────────────────────────────────────────────────────

const NUM_TRANSFORMS: usize = 36;
const NUM_REGISTERS:  usize = 6;

/// Minimum SINE frequency at all detail levels (controls the largest smooth shapes).
const MIN_FREQ: f32 = 4.0;
/// Maximum SINE frequency at detail=0 (smooth sweeps, matches original behaviour).
const MAX_FREQ_LOW: f32 = 25.0;
/// Maximum SINE frequency at detail=1 (dense filigree accents).
const MAX_FREQ_HIGH: f32 = 220.0;

// ── PRNG (xorshift64*) ───────────────────────────────────────────────────────
// Public-domain algorithm by Sebastiano Vigna (xorshift64*).
// State must never be zero; seeds of 0 are promoted to 1.

#[inline(always)]
fn prng_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x.wrapping_mul(0x2545F4914F6CDD1D)
}

#[inline(always)]
fn prng_f32(state: &mut u64) -> f32 {
    // Top 24 bits → [0.0, 1.0).
    (prng_next(state) >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
}

#[inline(always)]
fn prng_bounded(state: &mut u64, n: u64) -> u64 {
    prng_next(state) % n
}

// ── Op weighting ─────────────────────────────────────────────────────────────

/// Pick one of the 9 transform ops with detail-dependent weighting.
///
/// SINE (op 6) and MULTIPLY (op 5) are the detail-producing ops; their
/// combined weight rises from 2/9 (≈ uniform) at detail=0 to ~8/15 at
/// detail=1, making them the dominant choice at high detail.
fn pick_weighted_op(rng: &mut u64, detail: f32) -> u8 {
    // Heavy weight for SINE/MULTIPLY: 1.0 at detail=0 → 4.0 at detail=1.
    let hw = 1.0_f32 + 3.0 * detail;
    // Total weight: 2 heavy ops + 7 light ops (each weight 1.0).
    let total = hw + hw + 7.0;
    let r = prng_f32(rng) * total;
    if r < hw             { return 5; } // MULTIPLY
    if r < hw + hw        { return 6; } // SINE
    // Map the remaining [2hw, total) uniformly into the 7 structural ops.
    let idx = ((r - hw - hw) as usize).min(6);
    [0u8, 1, 2, 3, 4, 7, 8][idx]
}

/// Draw a per-step SINE frequency from a detail-scaled range.
/// Uses r² bias so most steps stay moderate and only a few reach the high end.
fn pick_freq(rng: &mut u64, detail: f32) -> f32 {
    let max_freq = MAX_FREQ_LOW + (MAX_FREQ_HIGH - MAX_FREQ_LOW) * detail;
    let r = prng_f32(rng);
    MIN_FREQ + (max_freq - MIN_FREQ) * r * r
}

// ── Transforms ───────────────────────────────────────────────────────────────

#[inline(always)]
fn fract3(v: f32) -> f32 { v - v.floor() }

#[inline(always)]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Execute one transform step. `freq` is used only by SINE (op 6).
#[inline(always)]
fn apply_transform(op: u8, freq: f32, a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    match op {
        // PROJECTION: broadcast scalar dot product
        0 => { let d = dot3(a, b); [d, d, d] }
        // SHIFT: component-wise fract(a + b), wraps to [0, 1)
        1 => [fract3(a[0]+b[0]), fract3(a[1]+b[1]), fract3(a[2]+b[2])],
        // SHIFTBACK: component-wise fract(a - b)
        2 => [fract3(a[0]-b[0]), fract3(a[1]-b[1]), fract3(a[2]-b[2])],
        // ROTATE: [y, z, x]
        3 => [a[1], a[2], a[0]],
        // ROTATE2: [z, x, y]
        4 => [a[2], a[0], a[1]],
        // MULTIPLY: component-wise product
        5 => [a[0]*b[0], a[1]*b[1], a[2]*b[2]],
        // SINE: 0.5 + 0.5·sin(freq·a·b), component-wise; output ∈ [0, 1]
        6 => [
            0.5 + 0.5 * (freq * a[0] * b[0]).sin(),
            0.5 + 0.5 * (freq * a[1] * b[1]).sin(),
            0.5 + 0.5 * (freq * a[2] * b[2]).sin(),
        ],
        // CONDITIONAL: pick register with greater channel sum
        7 => if a[0]+a[1]+a[2] > b[0]+b[1]+b[2] { a } else { b },
        // COMPLEMENT: 1 - a  (op == 8; _ catches any unexpected value)
        _ => [1.0-a[0], 1.0-a[1], 1.0-a[2]],
    }
}

// ── Genome ───────────────────────────────────────────────────────────────────

/// A reproducible qbist genome. Cheap to clone; serializable later.
///
/// Fully determined by (seed, detail): same inputs → identical eval output.
#[derive(Clone)]
pub struct QbistGenome {
    op:          [u8;  NUM_TRANSFORMS],
    source:      [u8;  NUM_TRANSFORMS],
    control:     [u8;  NUM_TRANSFORMS],
    dest:        [u8;  NUM_TRANSFORMS],
    /// Per-step SINE frequencies, drawn at genome-gen time so they're
    /// part of the reproducible seed. Steps whose op ≠ SINE ignore this.
    step_freq:   [f32; NUM_TRANSFORMS],
    /// Per-genome register initialisation constants (reg 0..NUM_REGISTERS).
    seed_consts: [f32; NUM_REGISTERS * 3],
    /// Stored detail level; drives op-weighting, freq range, and supersampling.
    pub detail:  f32,
}

impl QbistGenome {
    /// Deterministically build a genome from a seed and a detail level [0, 1].
    ///
    /// `detail=0` → smooth, sweepy patterns like the original algorithm.
    /// `detail=1` → dense, intricate filigree with wide frequency range.
    pub fn from_seed(seed: u64, detail: f32) -> Self {
        let detail = detail.clamp(0.0, 1.0);
        let mut rng = if seed == 0 { 1 } else { seed };

        let mut op        = [0u8;  NUM_TRANSFORMS];
        let mut source    = [0u8;  NUM_TRANSFORMS];
        let mut control   = [0u8;  NUM_TRANSFORMS];
        let mut dest      = [0u8;  NUM_TRANSFORMS];
        let mut step_freq = [0.0f32; NUM_TRANSFORMS];

        for i in 0..NUM_TRANSFORMS {
            op[i]        = pick_weighted_op(&mut rng, detail);
            source[i]    = prng_bounded(&mut rng, NUM_REGISTERS as u64) as u8;
            control[i]   = prng_bounded(&mut rng, NUM_REGISTERS as u64) as u8;
            dest[i]      = prng_bounded(&mut rng, NUM_REGISTERS as u64) as u8;
            step_freq[i] = pick_freq(&mut rng, detail);
        }

        let mut seed_consts = [0.0f32; NUM_REGISTERS * 3];
        for c in seed_consts.iter_mut() {
            *c = prng_f32(&mut rng);
        }

        Self { op, source, control, dest, step_freq, seed_consts, detail }
    }

    /// Return a new genome with `n_changes` randomly-selected entries mutated.
    /// The mutant inherits `self.detail`. Useful for generating related variations.
    #[allow(dead_code)]
    pub fn mutate(&self, seed: u64, n_changes: usize) -> Self {
        let mut rng  = if seed == 0 { 1 } else { seed };
        let mut next = self.clone();
        for _ in 0..n_changes {
            // 5 mutable fields: op(+freq), source, control, dest, freq-only
            let field = prng_bounded(&mut rng, 5);
            let idx   = prng_bounded(&mut rng, NUM_TRANSFORMS as u64) as usize;
            match field {
                0 => {
                    next.op[idx]       = pick_weighted_op(&mut rng, self.detail);
                    next.step_freq[idx] = pick_freq(&mut rng, self.detail);
                }
                1 => next.source[idx]  = prng_bounded(&mut rng, NUM_REGISTERS as u64) as u8,
                2 => next.control[idx] = prng_bounded(&mut rng, NUM_REGISTERS as u64) as u8,
                3 => next.dest[idx]    = prng_bounded(&mut rng, NUM_REGISTERS as u64) as u8,
                _ => next.step_freq[idx] = pick_freq(&mut rng, self.detail),
            }
        }
        next
    }

    /// Evaluate one pixel at normalised (x, y) ∈ [0, 1] → [r, g, b] ∈ [0, 1].
    /// Allocation-free hot path.
    #[inline]
    pub fn eval(&self, x: f32, y: f32) -> [f32; 3] {
        // Initialise registers from per-genome constants …
        let mut reg = [[0.0f32; 3]; NUM_REGISTERS];
        for (i, r) in reg.iter_mut().enumerate() {
            *r = [
                self.seed_consts[i * 3],
                self.seed_consts[i * 3 + 1],
                self.seed_consts[i * 3 + 2],
            ];
        }
        // … then override reg[0]/reg[1] with pixel coords for spatial variation.
        reg[0] = [x, y, 0.0];
        reg[1] = [y, x, 0.0];

        // Execute the straight-line program.
        for i in 0..NUM_TRANSFORMS {
            let a = reg[self.source[i]  as usize];
            let b = reg[self.control[i] as usize];
            reg[self.dest[i] as usize] = apply_transform(self.op[i], self.step_freq[i], a, b);
        }

        // Clamp output register to [0, 1].
        [
            reg[0][0].clamp(0.0, 1.0),
            reg[0][1].clamp(0.0, 1.0),
            reg[0][2].clamp(0.0, 1.0),
        ]
    }
}

// ── Rendering ────────────────────────────────────────────────────────────────

/// Supersampling factor derived from genome detail.
/// detail < 0.4 → 1× (no supersampling); detail ≥ 0.4 → 2× (box filter).
/// Capped at 2 to keep generation within a few seconds on a background thread.
fn supersample_factor(detail: f32) -> u32 {
    if detail < 0.4 { 1 } else { 2 }
}

/// Render a full image to an RGBA8 buffer (row-major, 4 bytes per pixel, alpha=255).
///
/// Automatically applies 2× supersampling when `genome.detail ≥ 0.4` to keep
/// high-frequency patterns crisp. Output size is always `width × height × 4`.
pub fn render_rgba(genome: &QbistGenome, width: u32, height: u32) -> Vec<u8> {
    let ss = supersample_factor(genome.detail);

    if ss == 1 {
        // Fast path: one eval per output pixel.
        let mut out = Vec::with_capacity((width * height * 4) as usize);
        let inv_w = 1.0 / width  as f32;
        let inv_h = 1.0 / height as f32;
        for row in 0..height {
            let y = row as f32 * inv_h;
            for col in 0..width {
                let x = col as f32 * inv_w;
                let [r, g, b] = genome.eval(x, y);
                out.push((r * 255.0) as u8);
                out.push((g * 255.0) as u8);
                out.push((b * 255.0) as u8);
                out.push(255);
            }
        }
        out
    } else {
        // Supersampled path: process ss big-rows per output row, accumulating
        // into a per-row f32 scratch buffer. Peak extra memory: ~width×12 bytes.
        let ss_u  = ss as usize;
        let big_w = width  as usize * ss_u;
        let big_h = height as usize * ss_u;
        let inv_bw  = 1.0 / big_w  as f32;
        let inv_bh  = 1.0 / big_h  as f32;
        let inv_ss2 = 1.0 / (ss_u * ss_u) as f32;

        let mut out = Vec::with_capacity((width * height * 4) as usize);

        for oy in 0..height as usize {
            // Scratch row: one [f32;3] accumulator per output column.
            let mut row_acc: Vec<[f32; 3]> = vec![[0.0; 3]; width as usize];
            for dy in 0..ss_u {
                let big_row = oy * ss_u + dy;
                let y = big_row as f32 * inv_bh;
                for big_col in 0..big_w {
                    let x  = big_col as f32 * inv_bw;
                    let ox = big_col / ss_u;
                    let [r, g, b] = genome.eval(x, y);
                    row_acc[ox][0] += r;
                    row_acc[ox][1] += g;
                    row_acc[ox][2] += b;
                }
            }
            for acc in &row_acc {
                out.push((acc[0] * inv_ss2 * 255.0) as u8);
                out.push((acc[1] * inv_ss2 * 255.0) as u8);
                out.push((acc[2] * inv_ss2 * 255.0) as u8);
                out.push(255);
            }
        }
        out
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_seed_is_deterministic() {
        let g1 = QbistGenome::from_seed(42, 0.0);
        let g2 = QbistGenome::from_seed(42, 0.0);
        assert_eq!(g1.eval(0.25, 0.75), g2.eval(0.25, 0.75), "same seed+detail → identical eval");
    }

    #[test]
    fn deterministic_with_detail() {
        // Both seed AND detail must be identical for reproducibility.
        for &det in &[0.0f32, 0.3, 0.5, 0.7, 1.0] {
            let g1 = QbistGenome::from_seed(0xCAFE, det);
            let g2 = QbistGenome::from_seed(0xCAFE, det);
            assert_eq!(g1.eval(0.4, 0.6), g2.eval(0.4, 0.6),
                "detail={det}: same (seed,detail) must be identical");
        }
    }

    #[test]
    fn different_detail_different_output() {
        let g_low  = QbistGenome::from_seed(999, 0.0);
        let g_high = QbistGenome::from_seed(999, 1.0);
        assert_ne!(g_low.eval(0.5, 0.5), g_high.eval(0.5, 0.5),
            "different detail values should produce different output for the same seed");
    }

    #[test]
    fn eval_output_in_range() {
        for &det in &[0.0f32, 0.5, 1.0] {
            let genome = QbistGenome::from_seed(12345, det);
            for (x, y) in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.3, 0.7), (0.9, 0.1)] {
                let [r, g, b] = genome.eval(x, y);
                assert!((0.0..=1.0).contains(&r), "detail={det} r={r} out of [0,1]");
                assert!((0.0..=1.0).contains(&g), "detail={det} g={g} out of [0,1]");
                assert!((0.0..=1.0).contains(&b), "detail={det} b={b} out of [0,1]");
            }
        }
    }

    #[test]
    fn render_rgba_correct_size() {
        // Size must equal width*height*4 regardless of internal supersampling.
        for &det in &[0.0f32, 0.5, 1.0] {
            let genome = QbistGenome::from_seed(99, det);
            let buf = render_rgba(&genome, 32, 24);
            assert_eq!(buf.len(), 32 * 24 * 4,
                "detail={det}: expected {} bytes, got {}", 32*24*4, buf.len());
        }
    }

    #[test]
    fn different_seeds_different_output() {
        let g1 = QbistGenome::from_seed(1, 0.5);
        let g2 = QbistGenome::from_seed(2, 0.5);
        assert_ne!(g1.eval(0.5, 0.5), g2.eval(0.5, 0.5),
            "different seeds should produce different pixels");
    }

    #[test]
    fn mutate_changes_output() {
        // 12 mutations across 25 sample points: virtually impossible for all to be no-ops.
        let base   = QbistGenome::from_seed(777, 0.5);
        let mutant = base.mutate(123, 12);
        let differs = (0..25).any(|i| {
            let t = i as f32 / 24.0;
            base.eval(t, 1.0 - t) != mutant.eval(t, 1.0 - t)
        });
        assert!(differs, "mutated genome should produce different output");
    }
}

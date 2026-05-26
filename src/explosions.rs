//! CPU-side explosion burst simulation.
//!
//! Each Explosion spawns up to `chunk_count` Chunks. Chunks go through three phases:
//!   Phase 1 — Tremble: oscillate around home_uv with quadratic alpha ramp (0→1)
//!   Phase 2 — Flyout:  analytic drag integral, C0-continuous with tremble via release_disp
//!   Phase 3 — Fade:    alpha decreases as (1 - fly_t/lifetime)²
//!
//! C0 position continuity: at break-free, tremble displacement == release_disp, which is
//! blended out over BLEND_DUR seconds so the flyout starts smoothly from the same position.

use rand::Rng;
use std::f32::consts::TAU;

/// Seconds to blend out release_disp at break-free (C0 continuity window).
const BLEND_DUR: f32 = 0.05;

pub struct Chunk {
    pub home_uv:      [f32; 2],  // UV centre of this chunk
    pub launch_vel:   [f32; 2],  // UV/s outward velocity at break-free
    pub tremble_dir:  [f32; 2],  // unit vector for tremble oscillation direction
    pub release_disp: [f32; 2],  // tremble displacement exactly at release_at (for C0 blend-out)
    pub phase:        f32,       // individual oscillation phase offset (radians)
    pub release_at:   f32,       // seconds into the explosion when this chunk breaks free
    pub lifetime:     f32,       // seconds of flyout before fully faded
    pub size:         f32,       // half-extent in screen-height UV units
    #[allow(dead_code)]
    pub hue_seed:     f32,       // 0..1, for CPU-side colour variety tracking
    pub tumble_speed: f32,       // radians/s; rotates the rendered quad
}

pub struct Explosion {
    pub chunks:       Vec<Chunk>,
    pub elapsed:      f32,
    pub tremble_amp:  f32,  // peak oscillation amplitude (UV units)
    pub tremble_freq: f32,  // oscillation frequency (Hz)
    pub drag_k:       f32,  // drag coefficient for analytic integral
}

impl Explosion {
    pub fn new<R: Rng>(
        rng: &mut R,
        aspect: f32,
        chunk_count: usize,
        chunk_size: f32,
        tremble_duration: f32,
        flyout_duration: f32,
    ) -> Self {
        // Per-burst randomisation — wide ranges for impressive variety
        let tremble_amp  = rng.gen_range(0.012_f32..=0.048);
        let tremble_freq = rng.gen_range(4.0_f32..=15.0);
        let drag_k       = rng.gen_range(0.4_f32..=1.0);

        let chunks = (0..chunk_count).map(|_| {
            // Home UV: uniform full-frame scatter so the whole pattern shatters at once
            let hx: f32 = rng.gen();
            let hy: f32 = rng.gen();
            let home_uv = [hx, hy];

            // Launch velocity: radially outward from screen center, aspect-corrected
            // so directions are geometrically round on a non-square frame.
            let speed = rng.gen_range(0.5_f32..=2.5);
            let dx = (hx - 0.5) * aspect;
            let dy = hy - 0.5;
            let pixel_len = (dx * dx + dy * dy).sqrt();
            let (vx, vy) = if pixel_len < 0.01 {
                let a = rng.gen_range(0.0_f32..TAU);
                (a.cos() * speed / aspect, a.sin() * speed)
            } else {
                let nx = dx / pixel_len;
                let ny = dy / pixel_len;
                (nx * speed / aspect, ny * speed)
            };
            let launch_vel = [vx, vy];

            // Staggered break-free timing creates an organic cascade
            let release_at   = rng.gen_range(0.01_f32..=tremble_duration.max(0.05));
            let phase        = rng.gen_range(0.0_f32..TAU);
            let td_angle     = rng.gen_range(0.0_f32..TAU);
            let tremble_dir  = [td_angle.cos(), td_angle.sin()];

            // C0 continuity: compute exact tremble displacement at break-free moment
            let tv_at_release = tremble_amp * (release_at * tremble_freq * TAU + phase).sin();
            let release_disp  = [tremble_dir[0] * tv_at_release, tremble_dir[1] * tv_at_release];

            Chunk {
                home_uv,
                launch_vel,
                tremble_dir,
                release_disp,
                phase,
                release_at,
                lifetime: flyout_duration * rng.gen_range(0.55_f32..=1.6),
                size:     chunk_size * rng.gen_range(0.35_f32..=2.4),
                hue_seed: rng.gen::<f32>(),
                tumble_speed: rng.gen_range(-9.0_f32..=9.0),
            }
        }).collect();

        Self { chunks, elapsed: 0.0, tremble_amp, tremble_freq, drag_k }
    }

    pub fn is_done(&self) -> bool {
        let last_die = self.chunks.iter()
            .map(|c| c.release_at + c.lifetime)
            .fold(0.0_f32, f32::max);
        self.elapsed >= last_die
    }
}

/// Per-chunk render data for one frame. CPU → GPU vertex generation.
#[derive(Clone, Copy)]
pub struct ChunkFrame {
    pub uv:       [f32; 2],  // current UV position (0..1 each axis)
    pub alpha:    f32,        // 0..1 opacity
    pub size:     f32,        // half-extent in screen-height UV units
    pub tumble:   f32,        // current rotation angle (radians)
}

pub struct ExplosionSystem {
    pub explosions: Vec<Explosion>,
    pub clock:      f32,
    next_fire_at:   f32,
    interval_min:   f32,
    interval_max:   f32,
}

impl ExplosionSystem {
    pub fn new() -> Self {
        Self {
            explosions:   Vec::new(),
            clock:        0.0,
            next_fire_at: 0.0,
            interval_min: 3.0,
            interval_max: 8.0,
        }
    }

    pub fn set_interval(&mut self, min: f32, max: f32) {
        self.interval_min = min;
        self.interval_max = max;
    }

    /// Advance simulation by `dt` seconds.
    /// If `fire` is true, spawns new explosions on schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn tick<R: Rng>(
        &mut self,
        dt: f32,
        rng: &mut R,
        fire: bool,
        chunk_count: usize,
        chunk_size:  f32,
        tremble_dur: f32,
        flyout_dur:  f32,
        aspect:      f32,
    ) {
        self.clock += dt;
        self.explosions.retain(|e| !e.is_done());
        for e in &mut self.explosions { e.elapsed += dt; }

        if fire && self.clock >= self.next_fire_at {
            self.explosions.push(Explosion::new(rng, aspect, chunk_count, chunk_size, tremble_dur, flyout_dur));
            self.next_fire_at = self.clock + rng.gen_range(self.interval_min..=self.interval_max);
        }
    }

    /// Collect render data for all active chunks.
    pub fn collect_chunks(&self) -> Vec<ChunkFrame> {
        let mut out = Vec::new();
        for expl in &self.explosions {
            let t    = expl.elapsed;
            let amp  = expl.tremble_amp;
            let freq = expl.tremble_freq;
            let k    = expl.drag_k;

            for c in &expl.chunks {
                if t < c.release_at {
                    // Phase 1: Tremble — near-invisible ramp so burst reads as seamless
                    let alpha = (t / c.release_at).powi(2);
                    let tv = amp * (t * freq * TAU + c.phase).sin();
                    let uv = [
                        c.home_uv[0] + c.tremble_dir[0] * tv,
                        c.home_uv[1] + c.tremble_dir[1] * tv,
                    ];
                    out.push(ChunkFrame { uv, alpha, size: c.size, tumble: c.tumble_speed * t });
                } else {
                    // Phase 2+3: Flyout + Fade
                    let fly_t = t - c.release_at;
                    if fly_t >= c.lifetime { continue; }

                    // Analytic drag integral: s = v0 * (1 - exp(-k*fly_t)) / k
                    let s = (1.0 - (-k * fly_t).exp()) / k;

                    // C0 blend-out: add release_disp decaying to zero over BLEND_DUR
                    let blend = (1.0 - fly_t / BLEND_DUR).max(0.0);
                    let uv = [
                        c.home_uv[0] + c.launch_vel[0] * s + c.release_disp[0] * blend,
                        c.home_uv[1] + c.launch_vel[1] * s + c.release_disp[1] * blend,
                    ];
                    let alpha = (1.0 - fly_t / c.lifetime).powi(2);
                    out.push(ChunkFrame { uv, alpha, size: c.size, tumble: c.tumble_speed * t });
                }
            }
        }
        out
    }
}

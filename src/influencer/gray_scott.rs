// Ported from myocyte. 3D Gray-Scott reaction-diffusion that writes cell colors and opacity each frame.
//
// RD equations (Pearson spots/coral regime: f=0.035, k=0.065):
//   dU/dt = Du·∇²U − U·V² + f·(1−U)
//   dV/dt = Dv·∇²V + U·V² − (f+k)·V
//
// Laplacian: 6-neighbor finite difference, periodic boundary conditions.
// Integration: forward Euler, DT_RD=0.5 (conservative for 3D stability).
//
// V concentration is mapped to cell visual parameters once per step:
//   opacity     = BASE_OPACITY * (0.2 + 0.8 * V)
//   color_inner = lerp(warm peach, muted teal, V)
//   color_outer = lerp(dark rose,  dark teal,  V)

use crate::cell::CellGrid;
use crate::influencer::Influencer;
use glam::Vec3;

const F:  f32 = 0.035;
const K:  f32 = 0.065;
const DU: f32 = 0.16;
const DV: f32 = 0.08;

/// Conservative timestep. Stable for DU=0.16, DV=0.08 in 3D at cell spacing=1.
pub const DT_RD: f32 = 0.5;

const BASE_OPACITY: f32 = 0.85;

// Color anchors for V→color mapping.
const INNER_LOW:  Vec3 = Vec3::new(0.88, 0.72, 0.68);  // warm peach  (V=0)
const INNER_HIGH: Vec3 = Vec3::new(0.30, 0.62, 0.60);  // muted teal  (V=1)
const OUTER_LOW:  Vec3 = Vec3::new(0.40, 0.20, 0.18);  // dark rose   (V=0)
const OUTER_HIGH: Vec3 = Vec3::new(0.10, 0.30, 0.30);  // dark teal   (V=1)

pub struct GrayScott {
    dims:   [u32; 3],
    u:      Vec<f32>,
    v:      Vec<f32>,
    u_next: Vec<f32>,
    v_next: Vec<f32>,
}

impl GrayScott {
    pub fn new(dims: [u32; 3]) -> Self {
        let n = (dims[0] * dims[1] * dims[2]) as usize;

        let mut u = vec![1.0_f32; n];
        let mut v = vec![0.0_f32; n];

        // Seed the center 3×3×3 block to nucleate a wavefront.
        let cx = dims[0] / 2;
        let cy = dims[1] / 2;
        let cz = dims[2] / 2;
        for dx in 0..3_u32 {
            for dy in 0..3_u32 {
                for dz in 0..3_u32 {
                    let x = (cx + dx).saturating_sub(1).min(dims[0] - 1);
                    let y = (cy + dy).saturating_sub(1).min(dims[1] - 1);
                    let z = (cz + dz).saturating_sub(1).min(dims[2] - 1);
                    let i = idx3(x, y, z, dims);
                    v[i] = 0.5;
                    u[i] = 0.5;
                }
            }
        }

        // Small deterministic noise (amplitude 0.05) to break perfect symmetry.
        for xi in 0..dims[0] {
            for yi in 0..dims[1] {
                for zi in 0..dims[2] {
                    let i = idx3(xi, yi, zi, dims);
                    u[i] = (u[i] + (h(xi, yi, zi, 7) - 0.5) * 0.05).clamp(0.0, 1.0);
                    v[i] = (v[i] + (h(xi, yi, zi, 8) - 0.5) * 0.05).clamp(0.0, 1.0);
                }
            }
        }

        Self {
            dims,
            u_next: u.clone(),
            v_next: v.clone(),
            u,
            v,
        }
    }
}

impl Influencer for GrayScott {
    fn step(&mut self, grid: &mut CellGrid, _dt: f32) {
        let dims = self.dims;

        gs_step(&self.u, &self.v, &mut self.u_next, &mut self.v_next, dims);
        std::mem::swap(&mut self.u, &mut self.u_next);
        std::mem::swap(&mut self.v, &mut self.v_next);

        // Map V → cell visual parameters.
        for x in 0..dims[0] {
            for y in 0..dims[1] {
                for z in 0..dims[2] {
                    let v_val    = self.v[idx3(x, y, z, dims)].clamp(0.0, 1.0);
                    let cell_idx = grid.idx(x, y, z);
                    let cell     = &mut grid.cells[cell_idx];

                    cell.opacity     = BASE_OPACITY * (0.2 + 0.8 * v_val);
                    cell.color_inner = INNER_LOW.lerp(INNER_HIGH, v_val);
                    cell.color_outer = OUTER_LOW.lerp(OUTER_HIGH, v_val);
                }
            }
        }
    }
}

// ---- RD kernel ---------------------------------------------------------------

fn gs_step(
    u:      &[f32],
    v:      &[f32],
    u_next: &mut [f32],
    v_next: &mut [f32],
    dims:   [u32; 3],
) {
    for x in 0..dims[0] {
        for y in 0..dims[1] {
            for z in 0..dims[2] {
                let i   = idx3(x, y, z, dims);
                let ui  = u[i];
                let vi  = v[i];
                let uvv = ui * vi * vi;

                let lap_u = laplacian(u, x, y, z, dims);
                let lap_v = laplacian(v, x, y, z, dims);

                u_next[i] = (ui + DT_RD * (DU * lap_u - uvv + F * (1.0 - ui))).clamp(0.0, 1.0);
                v_next[i] = (vi + DT_RD * (DV * lap_v + uvv - (F + K) * vi)).clamp(0.0, 1.0);
            }
        }
    }
}

#[inline(always)]
fn laplacian(field: &[f32], x: u32, y: u32, z: u32, dims: [u32; 3]) -> f32 {
    let nx = dims[0];
    let ny = dims[1];
    let nz = dims[2];
    let xp = (x + 1) % nx;
    let xm = (x + nx - 1) % nx;
    let yp = (y + 1) % ny;
    let ym = (y + ny - 1) % ny;
    let zp = (z + 1) % nz;
    let zm = (z + nz - 1) % nz;

    field[idx3(xp, y,  z,  dims)]
  + field[idx3(xm, y,  z,  dims)]
  + field[idx3(x,  yp, z,  dims)]
  + field[idx3(x,  ym, z,  dims)]
  + field[idx3(x,  y,  zp, dims)]
  + field[idx3(x,  y,  zm, dims)]
  - 6.0 * field[idx3(x, y, z, dims)]
}

#[inline(always)]
fn idx3(x: u32, y: u32, z: u32, dims: [u32; 3]) -> usize {
    (x * dims[1] * dims[2] + y * dims[2] + z) as usize
}

/// Murmur3-style hash → [0, 1). Same algorithm as grid.rs for consistency.
fn h(x: u32, y: u32, z: u32, seed: u32) -> f32 {
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

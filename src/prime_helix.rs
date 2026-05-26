// PrimeHelix geometry: sieve semiprimes, map each to a point on a constant-
// radius cylinder, add an amber spine along the Y axis.
//
// Number-theory conventions match ~/primehelix/scripts/primehelix_lattice.py:
//   residue families from (p%4, q%4) with p ≤ q
//   family colors: 1x1 purple, 1x3 green, 3x3 coral, even gray, spine amber

use std::f32::consts::TAU;
use glam::{Quat, Vec3};
use crate::cell::{Cell19, CellGrid};

// ── Constants ─────────────────────────────────────────────────────────────────

/// SPF-sieve limit. All semiprimes in [4, N] are placed on the cylinder.
/// N=2000 yields ~700 semiprimes + spine cells — well under MYOCYTE_MAX_SPLATS.
pub const HELIX_N: usize = 2000;

/// Cylinder radius in world units.
const CYL_RADIUS: f32 = 5.5;

/// y = (theta − THETA_MID) * HEIGHT_SCALE maps theta ∈ (0, 0.5] to world Y.
/// Balanced semiprimes (theta→0.5) sit high; lopsided ones (theta→0) sit low.
const HEIGHT_SCALE: f32 = 12.0;
pub const THETA_MID: f32 = 0.25;

/// Number of spine cells distributed along the Y axis.
const SPINE_CELLS: usize = 48;

/// Y range for the spine (slightly outside the data's y range).
const SPINE_Y_MIN: f32 = -2.5;
const SPINE_Y_MAX: f32 = 3.5;

// ── Residue families ──────────────────────────────────────────────────────────

/// Semiprime residue family based on (p mod 4, q mod 4) with p ≤ q.
/// Matches the FAM_RGB scheme from the reference Python visualizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidueFamily { F1x1, F1x3, F3x3, Even }

impl ResidueFamily {
    /// Inner RGB matching the reference FAM_RGB colors exactly.
    pub fn color_inner(self) -> Vec3 {
        match self {
            Self::F1x1 => Vec3::new(159.0 / 255.0, 143.0 / 255.0, 248.0 / 255.0),
            Self::F1x3 => Vec3::new( 62.0 / 255.0, 207.0 / 255.0, 150.0 / 255.0),
            Self::F3x3 => Vec3::new(240.0 / 255.0, 112.0 / 255.0,  80.0 / 255.0),
            Self::Even  => Vec3::new(170.0 / 255.0, 170.0 / 255.0, 170.0 / 255.0),
        }
    }

    /// Outer RGB: darker/desaturated version of the inner color.
    pub fn color_outer(self) -> Vec3 {
        self.color_inner() * 0.30
    }
}

fn residue_family(p: u32, q: u32) -> ResidueFamily {
    // p == 2 (or q == 2) means one factor is the only even prime.
    if p == 2 || q == 2 { return ResidueFamily::Even; }
    match (p % 4, q % 4) {
        (1, 1)          => ResidueFamily::F1x1,
        (3, 3)          => ResidueFamily::F3x3,
        _               => ResidueFamily::F1x3,
    }
}

// ── Sieve ─────────────────────────────────────────────────────────────────────

/// Returns the smallest-prime-factor for every index 0..=limit.
/// spf[i] == i for primes; spf[i] < i for composites.
fn spf_sieve(limit: usize) -> Vec<u32> {
    let mut spf: Vec<u32> = (0..=(limit as u32)).collect();
    let mut i = 2usize;
    while i * i <= limit {
        if spf[i] == i as u32 {
            let mut j = i * i;
            while j <= limit {
                if spf[j] == j as u32 { spf[j] = i as u32; }
                j += i;
            }
        }
        i += 1;
    }
    spf
}

// ── Grid builder ──────────────────────────────────────────────────────────────

/// Build a CellGrid populated with PrimeHelix semiprime points + amber spine.
/// Caller owns the returned grid; no GPU work is done here.
pub fn build_prime_helix_grid() -> CellGrid {
    let spf = spf_sieve(HELIX_N);

    let mut cells: Vec<Cell19> = Vec::new();

    // ── Semiprime helix points ────────────────────────────────────────────────
    for n in 4..=HELIX_N {
        let p = spf[n];
        if p == n as u32 { continue; }  // n is prime
        let q_n = n as u32 / p;
        if spf[q_n as usize] != q_n { continue; }  // q is composite → not semiprime

        // p ≤ q is guaranteed because spf[n] is the SMALLEST prime factor.
        // theta = ln(p)/ln(n) ∈ (0, 0.5]. Balanced semiprimes approach 0.5.
        let theta = theta_for(p, n as u32);
        let angle = ((n % 30) as f32 / 30.0) * TAU;
        let x = CYL_RADIUS * angle.cos();
        let z = CYL_RADIUS * angle.sin();
        let y = (theta - THETA_MID) * HEIGHT_SCALE;

        let fam = residue_family(p, q_n);
        cells.push(Cell19 {
            position:    Vec3::new(x, y, z),
            rotation:    Quat::IDENTITY,
            scale:       Vec3::splat(0.16),
            falloff:     1.0,
            sharpness:   0.0,
            color_inner: fam.color_inner(),
            color_outer: fam.color_outer(),
            opacity:     0.70,
        });
    }

    // ── Amber θ spine ─────────────────────────────────────────────────────────
    // Dense column of cells at x=z=0, spanning the full Y range.
    // These serve as the glowing core that throbs to beats in PrimeActivation.
    let spine_inner = Vec3::new(1.0, 208.0 / 255.0, 128.0 / 255.0);
    let spine_outer = Vec3::new(0.50, 0.35, 0.08);
    for i in 0..SPINE_CELLS {
        let t = i as f32 / (SPINE_CELLS - 1) as f32;
        let y = SPINE_Y_MIN + t * (SPINE_Y_MAX - SPINE_Y_MIN);
        cells.push(Cell19 {
            position:    Vec3::new(0.0, y, 0.0),
            rotation:    Quat::IDENTITY,
            scale:       Vec3::splat(0.22),
            falloff:     1.0,
            sharpness:   0.0,
            color_inner: spine_inner,
            color_outer: spine_outer,
            opacity:     0.88,
        });
    }

    let total = cells.len() as u32;
    CellGrid { cells, dims: [total, 1, 1], spacing: 1.0 }
}

// ── Test helpers ──────────────────────────────────────────────────────────────

/// Returns the theta value for the semiprime n=p*q (p ≤ q).
/// Public so the test module can validate ranges.
pub fn theta_for(p: u32, n: u32) -> f32 {
    (p as f32).ln() / (n as f32).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prime_helix_has_semiprime_cells() {
        let grid = build_prime_helix_grid();
        // Grid must have at least the spine cells plus a healthy number of semiprimes.
        assert!(grid.cells.len() > SPINE_CELLS + 100,
            "expected >100 semiprime cells, got {}", grid.cells.len() - SPINE_CELLS);
    }

    #[test]
    fn theta_values_in_range() {
        let spf = spf_sieve(HELIX_N);
        for n in 4..=HELIX_N {
            let p = spf[n];
            if p == n as u32 { continue; }
            let q = n as u32 / p;
            if spf[q as usize] != q { continue; }
            let theta = theta_for(p, n as u32);
            assert!(theta > 0.0 && theta <= 0.5,
                "theta={} out of (0,0.5] for n={}", theta, n);
        }
    }
}

// Ported from myocyte. The 3D cell grid that backs ShapeKind::Myocyte.
// grid.rs — grid initialization helpers.
//
// Keeps CellGrid construction out of cell.rs, which only defines the data shape.

use crate::cell::{Cell19, CellGrid};
use glam::{Quat, Vec3};

/// Assign each cell's position field based on its (x,y,z) index.
/// Call this once after creating a new CellGrid before rendering.
pub fn place_cells(grid: &mut CellGrid) {
    for x in 0..grid.dims[0] {
        for y in 0..grid.dims[1] {
            for z in 0..grid.dims[2] {
                let idx = grid.idx(x, y, z);
                grid.cells[idx].position = grid.world_pos(x, y, z);
            }
        }
    }
}

/// Activate the single center cell with default visible parameters.
/// CP1 reference — not called in CP2+ but kept for debugging single-cell issues.
#[allow(dead_code)]
pub fn activate_center_cell(grid: &mut CellGrid) {
    let cx = grid.dims[0] / 2;
    let cy = grid.dims[1] / 2;
    let cz = grid.dims[2] / 2;
    let idx = grid.idx(cx, cy, cz);
    let pos = grid.cells[idx].position;
    grid.cells[idx] = Cell19 {
        position: pos,
        ..Cell19::visible_default()
    };
}

/// Activate every cell in the grid at the given scale, all other parameters
/// from Cell19::visible_default(). Used for CP2: uniform grid of identical cells.
///
/// Scale is passed explicitly because the right value depends on spacing:
///   scale=0.30 at spacing=1.0 → ~5.6% alpha at midpoint (clearly separate)
///   scale=0.45 at spacing=1.0 → ~26% alpha at midpoint (merges into fog)
#[allow(dead_code)]
pub fn activate_all_cells(grid: &mut CellGrid, scale: f32) {
    for x in 0..grid.dims[0] {
        for y in 0..grid.dims[1] {
            for z in 0..grid.dims[2] {
                let idx = grid.idx(x, y, z);
                let pos = grid.cells[idx].position;
                grid.cells[idx] = Cell19 {
                    position: pos,
                    scale: Vec3::splat(scale),
                    ..Cell19::visible_default()
                };
            }
        }
    }
}

/// Activate every cell with per-cell variation derived deterministically from (x,y,z).
/// Used for CP3+. Produces a stable baseline that does not change between runs,
/// so CP4's Gray-Scott influencer has a known reference to compare against.
///
/// Variation applied:
///   color_inner / color_outer: smooth volume gradient + local hash noise
///   scale:    ±15% around base_scale
///   rotation: uniform random quaternion (visually inert for isotropic cells;
///             baked in now so anisotropic cells at CP3+ work correctly)
pub fn activate_varied_cells(grid: &mut CellGrid, base_scale: f32) {
    for x in 0..grid.dims[0] {
        for y in 0..grid.dims[1] {
            for z in 0..grid.dims[2] {
                // Normalized position [0,1] used for smooth cross-volume color drift.
                let nx = x as f32 / grid.dims[0].saturating_sub(1).max(1) as f32;
                let ny = y as f32 / grid.dims[1].saturating_sub(1).max(1) as f32;
                let nz = z as f32 / grid.dims[2].saturating_sub(1).max(1) as f32;

                // Inner color: warm pink fading toward peach-amber across the volume.
                // Position drives a slow gradient; hash adds local grain on top.
                let ri = 0.88 + 0.10 * nx + (h(x, y, z, 1) - 0.5) * 0.06;
                let gi = 0.72 + 0.12 * ny + (h(x, y, z, 2) - 0.5) * 0.04;
                let bi = 0.68 + 0.08 * nz + (h(x, y, z, 3) - 0.5) * 0.03;
                let color_inner = Vec3::new(
                    ri.clamp(0.0, 1.0),
                    gi.clamp(0.0, 1.0),
                    bi.clamp(0.0, 1.0),
                );

                // Outer color: same drift but darker, drifting toward deeper rose.
                let ro = 0.40 + 0.12 * nx + (h(x, y, z, 4) - 0.5) * 0.06;
                let go = 0.20 + 0.10 * ny + (h(x, y, z, 5) - 0.5) * 0.04;
                let bo = 0.18 + 0.06 * nz + (h(x, y, z, 6) - 0.5) * 0.03;
                let color_outer = Vec3::new(
                    ro.clamp(0.0, 1.0),
                    go.clamp(0.0, 1.0),
                    bo.clamp(0.0, 1.0),
                );

                // Scale: ±15% around base_scale. Isotropic for now; anisotropy
                // comes later when we want cells to stretch with the RD gradient.
                let scale_mul = 1.0 + 0.15 * (h(x, y, z, 10) * 2.0 - 1.0);
                let scale = Vec3::splat(base_scale * scale_mul);

                // Rotation: uniform distribution on SO(3) via Shoemake (1992).
                // Invisible for isotropic cells, but the quaternion is stored and
                // ready for when scale becomes anisotropic at CP4+.
                let rotation = uniform_quat(x, y, z);

                let idx = grid.idx(x, y, z);
                let pos = grid.cells[idx].position;
                grid.cells[idx] = Cell19 {
                    position: pos,
                    rotation,
                    scale,
                    color_inner,
                    color_outer,
                    ..Cell19::visible_default()
                };
            }
        }
    }
}

// ---- Helpers ----------------------------------------------------------------

/// Murmur3-style hash of three grid coordinates and a seed → uniform f32 in [0, 1).
/// Good distribution down to small coordinate values (0–31 in all dimensions).
fn h(x: u32, y: u32, z: u32, seed: u32) -> f32 {
    let mut v = seed;
    v ^= x.wrapping_mul(0x9e3779b9);
    v = v.rotate_left(5).wrapping_mul(0x85ebca6b);
    v ^= y.wrapping_mul(0xc2b2ae35);
    v = v.rotate_left(7).wrapping_mul(0xcc9e2d51);
    v ^= z.wrapping_mul(0x6b3a36f5);
    v = v.rotate_left(11).wrapping_mul(0x1b873593);
    v ^= v >> 16;
    v  = v.wrapping_mul(0x85ebca6b);
    v ^= v >> 13;
    v  = v.wrapping_mul(0xc2b2ae35);
    v ^= v >> 16;
    (v & 0x00FF_FFFF) as f32 / 16_777_216.0
}

/// Uniformly distributed random rotation (Shoemake 1992) from three hash values.
fn uniform_quat(x: u32, y: u32, z: u32) -> Quat {
    let u1 = h(x, y, z, 100);
    let u2 = h(x, y, z, 101);
    let u3 = h(x, y, z, 102);
    let a  = (1.0 - u1).sqrt();
    let b  = u1.sqrt();
    Quat::from_xyzw(
        a * (std::f32::consts::TAU * u2).sin(),
        a * (std::f32::consts::TAU * u2).cos(),
        b * (std::f32::consts::TAU * u3).sin(),
        b * (std::f32::consts::TAU * u3).cos(),
    )
}

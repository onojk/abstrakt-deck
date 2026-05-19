// Ported from myocyte. The 3D cell grid that backs ShapeKind::Myocyte.
// cell.rs — the 19-float cell primitive and the grid that holds them.

use glam::{Quat, Vec3};

/// One myocyte cell: the 19-float parameterization from the design doc.
///
/// All fields are continuous reals. No enums, no branching on type.
/// This is a hard requirement for Tier 3 differentiability.
#[derive(Clone, Copy)]
pub struct Cell19 {
    pub position:    Vec3,   // world-space centroid
    pub rotation:    Quat,   // orientation (xyzw storage, identity = no rotation)
    pub scale:       Vec3,   // per-axis radii in local frame (not log-scale, not variance)
    pub falloff:     f32,    // density steepness: 0.5=soft exp, 1.0=Gaussian, 2.0=super-Gaussian
    pub sharpness:   f32,    // edge hardness: 0.0=smooth, 1.0≈hard ellipsoid
    pub color_inner: Vec3,   // linear RGB at cell center
    pub color_outer: Vec3,   // linear RGB at cell edge (Mahalanobis distance = 1)
    pub opacity:     f32,    // base transmittance coefficient at center
}

impl Cell19 {
    /// A visible cell with warm tissue colors, soft Gaussian profile.
    pub fn visible_default() -> Self {
        Self {
            position:    Vec3::ZERO,
            rotation:    Quat::IDENTITY,
            scale:       Vec3::splat(0.45),
            falloff:     1.0,
            sharpness:   0.0,
            color_inner: Vec3::new(0.92, 0.78, 0.72),
            color_outer: Vec3::new(0.45, 0.25, 0.22),
            opacity:     0.90,
        }
    }

    /// A fully transparent cell (placeholder in a grid).
    pub fn invisible() -> Self {
        Self {
            opacity: 0.0,
            ..Self::visible_default()
        }
    }
}

/// A 3D grid of cells, row-major with z varying fastest (x slowest).
///
/// Grid is centered at the world origin. Cell positions are set by
/// grid::place_cells_on_grid and stored in each cell's position field.
pub struct CellGrid {
    pub cells:    Vec<Cell19>,
    pub dims:     [u32; 3],       // [x, y, z] dimensions
    pub spacing:  f32,            // center-to-center distance between cells
}

impl CellGrid {
    pub fn new(dims: [u32; 3], spacing: f32) -> Self {
        let n = (dims[0] * dims[1] * dims[2]) as usize;
        Self {
            cells:   vec![Cell19::invisible(); n],
            dims,
            spacing,
        }
    }

    /// Flat index for grid position (x, y, z). z varies fastest.
    pub fn idx(&self, x: u32, y: u32, z: u32) -> usize {
        (x * self.dims[1] * self.dims[2] + y * self.dims[2] + z) as usize
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// World-space center for grid cell (x, y, z). Grid is centered at origin.
    pub fn world_pos(&self, x: u32, y: u32, z: u32) -> Vec3 {
        let cx = (x as f32 - (self.dims[0] - 1) as f32 * 0.5) * self.spacing;
        let cy = (y as f32 - (self.dims[1] - 1) as f32 * 0.5) * self.spacing;
        let cz = (z as f32 - (self.dims[2] - 1) as f32 * 0.5) * self.spacing;
        Vec3::new(cx, cy, cz)
    }
}

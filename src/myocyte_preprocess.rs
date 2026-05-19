// Ported from myocyte. Per-frame CPU work: project each cell to 2D, compute the
// 2D inverse covariance and bounding quad needed by the fragment shader.
//
// All math is done in NDC (normalized device coordinate) space.
//
// The projection follows the EWA splatting Jacobian approach used by 3DGS:
//   Σ_2D = J · W · Σ_3D · W^T · J^T   (2×2, in NDC²)
// where J is the Jacobian of the perspective projection at the cell center
// and W is the rotation part of the view matrix.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec2, Vec4};

use crate::myocyte_camera::OrbitCamera;
use crate::cell::CellGrid;

/// Small floor added to Σ_2D diagonal to prevent degenerate (zero-area) splats.
/// 1e-5 NDC² ≈ 2 pixels at a 960-wide viewport — sub-pixel, invisible in practice.
const NDC_REGULARIZATION: f32 = 1e-5;

/// Per-cell GPU data, produced each frame after projection and sorting.
/// Uploaded as a storage buffer; one entry per cell (in back-to-front order).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SplatGpuData {
    /// NDC-space center of the cell's 2D footprint.
    pub screen_xy:   [f32; 2],
    /// Half-size of the bounding quad in NDC units (3σ of the 2D Gaussian).
    pub quad_radius: f32,
    pub _pad0:       f32,
    /// Upper triangle of the 2×2 inverse NDC covariance: [a, b, d, 0]
    /// where the matrix is [[a, b], [b, d]].
    pub inv_cov2d:   [f32; 4],
    /// color_inner as RGB (w unused).
    pub color_inner: [f32; 4],
    /// color_outer as RGB (w unused).
    pub color_outer: [f32; 4],
    /// [opacity, falloff, sharpness, unused]
    pub params:      [f32; 4],
}

/// Depth value kept alongside SplatGpuData for CPU sorting; not uploaded to GPU.
pub struct SplatWithDepth {
    pub gpu:   SplatGpuData,
    pub depth: f32,   // view-space z (negative; more negative = farther)
}

/// Project all cells to 2D and return a list of SplatWithDepth, one per visible cell.
/// Cells behind the camera or with zero opacity are skipped.
pub fn project_grid(grid: &CellGrid, camera: &OrbitCamera) -> Vec<SplatWithDepth> {
    let view      = camera.view();
    let view_proj = camera.view_proj();
    let fov_y     = camera.fov_y;
    let aspect    = camera.aspect;

    let tan_half_fov_y = (fov_y * 0.5).tan();
    let tan_half_fov_x = tan_half_fov_y * aspect;
    // Focal lengths in NDC units (maps view-space extent to NDC extent)
    let cx = 1.0 / tan_half_fov_x;
    let cy = 1.0 / tan_half_fov_y;

    // Upper-left 3×3 of the view matrix = pure rotation (no translation).
    // Transforms world-space direction vectors to view-space.
    let w = Mat3::from_mat4(view);

    let mut out = Vec::with_capacity(grid.len());

    for cell in &grid.cells {
        // Skip invisible cells early — they'd contribute nothing to the image.
        if cell.opacity < 1.0 / 255.0 {
            continue;
        }

        // View-space center
        let vp = view.transform_point3(cell.position);
        let tz = vp.z;

        // Visible points have tz < 0 in right-handed view space (camera looks along -z).
        if tz >= -camera.z_near {
            continue;
        }

        let tx = vp.x;
        let ty = vp.y;

        // NDC center via perspective division
        let clip = view_proj * Vec4::from((cell.position, 1.0));
        if clip.w.abs() < 1e-8 {
            continue;
        }
        let ndc_center = Vec2::new(clip.x / clip.w, clip.y / clip.w);

        // Coarse clip: skip cells whose center is far outside [-2, 2] in NDC
        // (they may still contribute at the edges, but at small cell sizes this
        // is a good early-out; tighten to [-1.5, 1.5] for tighter culling).
        if ndc_center.x.abs() > 2.0 || ndc_center.y.abs() > 2.0 {
            continue;
        }

        // 3D covariance in world space: Σ_3D = R diag(s²) R^T
        let rot  = Mat3::from_quat(cell.rotation);
        let s2   = cell.scale * cell.scale;
        let s3d  = rot * Mat3::from_diagonal(s2) * rot.transpose();

        // Covariance in view space: Σ_view = W Σ_3D W^T
        let sv = w * s3d * w.transpose();

        // Jacobian of perspective projection in NDC units, evaluated at (tx, ty, tz).
        // J = [[cx/tz,  0,     -cx*tx/tz²],
        //      [ 0,    cy/tz, -cy*ty/tz²]]
        let tz2  = tz * tz;
        let j00  = cx / tz;
        let j02  = -cx * tx / tz2;
        let j11  = cy / tz;
        let j12  = -cy * ty / tz2;

        // T = J · Σ_view  (2×3), then Σ_2D = T · J^T  (2×2).
        // Expand T row by row (sv is column-major; sv.col(j)[i] = sv[i][j]):
        let t0x = j00 * sv.col(0).x + j02 * sv.col(0).z;
        let t0y = j00 * sv.col(1).x + j02 * sv.col(1).z;
        let t0z = j00 * sv.col(2).x + j02 * sv.col(2).z;

        let _t1x = j11 * sv.col(0).y + j12 * sv.col(0).z;
        let t1y = j11 * sv.col(1).y + j12 * sv.col(1).z;
        let t1z = j11 * sv.col(2).y + j12 * sv.col(2).z;

        // Σ_2D = T · J^T:
        // [0][0] = T[0] · J[0]^T = t0x*j00 + t0z*j02
        // [0][1] = T[0] · J[1]^T = t0y*j11 + t0z*j12
        // [1][1] = T[1] · J[1]^T = t1y*j11 + t1z*j12
        let mut s00 = t0x * j00 + t0z * j02;
        let     s01 = t0y * j11 + t0z * j12;  // also s10 by symmetry
        let mut s11 = t1y * j11 + t1z * j12;

        // Regularize: floor the diagonal so degenerate (point-like) cells stay renderable.
        s00 += NDC_REGULARIZATION;
        s11 += NDC_REGULARIZATION;

        // Invert the 2×2 symmetric matrix [[s00, s01], [s01, s11]].
        let det = s00 * s11 - s01 * s01;
        if det < 1e-12 {
            continue;  // numerically degenerate after regularization — skip
        }
        let inv_det = 1.0 / det;
        let inv_a = s11 * inv_det;
        let inv_b = -s01 * inv_det;
        let inv_d = s00 * inv_det;

        // Bounding quad radius: 3σ in the direction of greatest spread.
        // Max eigenvalue of [[s00, s01], [s01, s11]]:
        let mid       = (s00 + s11) * 0.5;
        let half_diff = (s00 - s11) * 0.5;
        let max_lambda = mid + (half_diff * half_diff + s01 * s01).sqrt();
        let quad_radius = 3.0 * max_lambda.sqrt();

        out.push(SplatWithDepth {
            depth: tz,
            gpu: SplatGpuData {
                screen_xy:   ndc_center.to_array(),
                quad_radius,
                _pad0:       0.0,
                inv_cov2d:   [inv_a, inv_b, inv_d, 0.0],
                color_inner: [cell.color_inner.x, cell.color_inner.y, cell.color_inner.z, 0.0],
                color_outer: [cell.color_outer.x, cell.color_outer.y, cell.color_outer.z, 0.0],
                params:      [cell.opacity, cell.falloff, cell.sharpness, 0.0],
            },
        });
    }

    out
}

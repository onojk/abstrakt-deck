// Ported from myocyte. Depth sort for back-to-front alpha compositing.
//
// Sorts splats so the one farthest from the camera is drawn first.
// In right-handed view space, depth (z) is negative for visible points;
// "more negative" means farther away, so we sort ascending by depth.

use crate::myocyte_preprocess::{SplatGpuData, SplatWithDepth};

/// Sort splats back-to-front and return the GPU data slice in draw order.
pub fn sort_back_to_front(splats: &mut [SplatWithDepth]) -> Vec<SplatGpuData> {
    // Ascending sort: most-negative depth first = farthest cell first.
    splats.sort_unstable_by(|a, b| {
        a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal)
    });
    splats.iter().map(|s| s.gpu).collect()
}

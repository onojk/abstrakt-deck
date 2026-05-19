// Myocyte splat rasterizer, ported and adapted from myocyte. Phase 5a:
// stub — owns the data structures and the wgpu objects but render()
// is a no-op. Phase 5b implements the actual drawing.

use crate::cell::CellGrid;
use crate::myocyte_camera::OrbitCamera;
use crate::myocyte_preprocess::SplatGpuData;

/// Maximum splats we ever upload in one frame. Sized for 32³ = 32768.
/// Matches myocyte's original cap.
pub const MYOCYTE_MAX_SPLATS: usize = 32_768;

pub struct MyocyteRenderer {
    pub camera: OrbitCamera,
    // Phase 5b will add: pipeline, camera_buffer, splat_buffer, bind_group, etc.

    // Pre-allocated scratch buffer for projected splats, so we don't
    // allocate every frame.
    splats_scratch: Vec<SplatGpuData>,

    // CPU sort scratch.
    sort_indices_scratch: Vec<u32>,
}

impl MyocyteRenderer {
    pub fn new(_device: &wgpu::Device, _shape_fbo_format: wgpu::TextureFormat) -> Self {
        // Phase 5a stub: no GPU resources created yet. Phase 5b fills this in.
        Self {
            camera:               OrbitCamera::default(),
            splats_scratch:       Vec::with_capacity(MYOCYTE_MAX_SPLATS),
            sort_indices_scratch: Vec::with_capacity(MYOCYTE_MAX_SPLATS),
        }
    }

    /// Render the cell grid into the given render pass. Phase 5a: no-op.
    /// Phase 5b implements:
    ///   1. project all cells through camera
    ///   2. back-to-front sort
    ///   3. upload to splat storage buffer
    ///   4. instanced draw with N_cells instances
    pub fn render<'rpass>(
        &mut self,
        _grid:   &CellGrid,
        _queue:  &wgpu::Queue,
        _pass:   &mut wgpu::RenderPass<'rpass>,
        _aspect: f32,
    ) {
        // PHASE 5a: deliberately empty. The "Shape pass" clears to black
        // and we add nothing on top, which is the verification target.
        let _ = &self.splats_scratch;
        let _ = &self.sort_indices_scratch;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.camera.set_aspect(width as f32 / height as f32);
    }
}

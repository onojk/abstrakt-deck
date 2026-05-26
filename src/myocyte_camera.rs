// Ported from myocyte. Orbit camera and GPU camera uniform for the myocyte splat renderer.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub azimuth: f32,    // around vertical axis, radians
    pub elevation: f32,  // up/down tilt, radians, clamped near poles
    pub fov_y: f32,
    pub aspect: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub auto_rotation_rate_rad_per_s: f32,
}

impl OrbitCamera {
    pub fn new(aspect: f32) -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 3.5,
            azimuth: 0.5,
            elevation: 0.3,
            fov_y: 60f32.to_radians(),
            aspect,
            z_near: 0.1,
            z_far: 100.0,
            auto_rotation_rate_rad_per_s: 0.0,
        }
    }

    /// Advance the camera's azimuth by the auto-rotation rate × dt.
    /// Called every frame so the scene rotates continuously without user input.
    pub fn tick_auto_rotation(&mut self, dt: f32) {
        self.azimuth += self.auto_rotation_rate_rad_per_s * dt;
    }

    pub fn eye(&self) -> Vec3 {
        let r = self.distance;
        let x = r * self.elevation.cos() * self.azimuth.sin();
        let y = r * self.elevation.sin();
        let z = r * self.elevation.cos() * self.azimuth.cos();
        self.target + Vec3::new(x, y, z)
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    pub fn proj(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.z_near, self.z_far)
    }

    pub fn view_proj(&self) -> Mat4 {
        self.proj() * self.view()
    }

    #[allow(dead_code)]
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.azimuth -= dx * 0.005;
        self.elevation += dy * 0.005;
        // Clamp elevation so we don't flip past the poles (looks bad with up=Y).
        let limit = std::f32::consts::FRAC_PI_2 - 0.05;
        self.elevation = self.elevation.clamp(-limit, limit);
    }

    /// Keyboard-driven orbit: d_az and d_el are already scaled by dt and rate.
    #[allow(dead_code)]
    pub fn orbit_keyboard(&mut self, d_az: f32, d_el: f32) {
        self.azimuth += d_az;
        let limit = std::f32::consts::FRAC_PI_2 - 0.05;
        self.elevation = (self.elevation + d_el).clamp(-limit, limit);
    }

    #[allow(dead_code)]
    pub fn zoom(&mut self, delta: f32) {
        // Multiplicative zoom feels right — closer = smaller steps.
        self.distance *= (1.0 - delta * 0.1).clamp(0.5, 2.0);
        self.distance = self.distance.clamp(0.8, 30.0);
    }

    /// Keyboard-driven zoom: factor > 1 moves away, factor < 1 moves closer.
    #[allow(dead_code)]
    pub fn zoom_keyboard(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(0.8, 30.0);
    }

    /// Reset to default orbit angles, preserving the current aspect ratio.
    #[allow(dead_code)]
    pub fn reset(&mut self, distance: f32) {
        self.azimuth   = 0.5;
        self.elevation = 0.3;
        self.distance  = distance;
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }
}

impl Default for OrbitCamera {
    fn default() -> Self { Self::new(1.0) }
}

/// GPU-side camera uniform. 64 bytes (one Mat4) — must match WGSL layout.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub eye: [f32; 3],
    pub _pad: f32,
}

impl CameraUniform {
    pub fn from_camera(cam: &OrbitCamera) -> Self {
        Self {
            view_proj: cam.view_proj().to_cols_array_2d(),
            eye: cam.eye().to_array(),
            _pad: 0.0,
        }
    }
}

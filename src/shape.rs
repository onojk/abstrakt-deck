use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x2,
            },
        ],
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShapeKind {
    Cylinder,
    Sphere,
    Cube,
    Tetrahedron,
    Icosahedron,
    Urchin,
    Caltrop,
}

impl ShapeKind {
    pub fn next(self) -> Self {
        match self {
            ShapeKind::Cylinder    => ShapeKind::Sphere,
            ShapeKind::Sphere      => ShapeKind::Cube,
            ShapeKind::Cube        => ShapeKind::Tetrahedron,
            ShapeKind::Tetrahedron => ShapeKind::Icosahedron,
            ShapeKind::Icosahedron => ShapeKind::Urchin,
            ShapeKind::Urchin      => ShapeKind::Caltrop,
            ShapeKind::Caltrop     => ShapeKind::Cylinder,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ShapeKind::Cylinder    => "Cylinder",
            ShapeKind::Sphere      => "Sphere",
            ShapeKind::Cube        => "Cube",
            ShapeKind::Tetrahedron => "Tetrahedron",
            ShapeKind::Icosahedron => "Icosahedron",
            ShapeKind::Urchin      => "Urchin",
            ShapeKind::Caltrop     => "Caltrop",
        }
    }

    pub fn model_scale(self) -> f32 {
        match self {
            ShapeKind::Cylinder    => 1.0,
            ShapeKind::Sphere      => 2.0,
            ShapeKind::Cube        => 1.4,
            ShapeKind::Tetrahedron => 1.5,
            ShapeKind::Icosahedron => 1.6,
            ShapeKind::Urchin      => 1.3,
            ShapeKind::Caltrop     => 1.2,
        }
    }

    pub fn rotation_axis(self) -> [f32; 3] {
        match self {
            ShapeKind::Cylinder => [0.0, 1.0, 0.0],
            ShapeKind::Sphere => {
                let tilt = 23.4_f32.to_radians();
                [tilt.sin(), tilt.cos(), 0.0]
            }
            ShapeKind::Cube => {
                let n = 1.0_f32 / 3.0_f32.sqrt();
                [n, n, n]
            }
            ShapeKind::Tetrahedron => {
                let n = 1.0_f32 / 3.0_f32.sqrt();
                [n, n, n]
            }
            ShapeKind::Icosahedron => {
                let n = 1.0_f32 / 6.0_f32.sqrt();
                [n, 2.0 * n, n]
            }
            ShapeKind::Urchin => {
                let tilt = 17.0_f32.to_radians();
                [tilt.sin(), tilt.cos(), 0.0]
            }
            ShapeKind::Caltrop => {
                let n = 1.0_f32 / 3.0_f32.sqrt();
                [n, n, n]
            }
        }
    }

    pub fn rotation_period_seconds(self) -> f32 {
        match self {
            ShapeKind::Cylinder    => 30.0,
            ShapeKind::Sphere      => 30.0,
            ShapeKind::Cube        => 25.0,
            ShapeKind::Tetrahedron => 22.0,
            ShapeKind::Icosahedron => 26.0,
            ShapeKind::Urchin      => 28.0,
            ShapeKind::Caltrop     => 20.0,
        }
    }

    pub fn kaleido_zoom(self) -> f32 {
        match self {
            ShapeKind::Cylinder    => 0.6,
            ShapeKind::Sphere      => 0.88,
            ShapeKind::Cube        => 0.7,
            ShapeKind::Tetrahedron => 0.65,
            ShapeKind::Icosahedron => 0.75,
            ShapeKind::Urchin      => 0.8,
            ShapeKind::Caltrop     => 0.55,
        }
    }
}

pub struct ShapeMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

pub fn build_cylinder(segments: u32, radius: f32, half_height: f32) -> ShapeMesh {
    let mut vertices = Vec::with_capacity(((segments + 1) * 2) as usize);
    for ring in 0..2u32 {
        let y = if ring == 0 { -half_height } else { half_height };
        let v = if ring == 0 { 1.0_f32 } else { 0.0_f32 };
        for s in 0..=segments {
            let theta = (s as f32 / segments as f32) * std::f32::consts::TAU;
            vertices.push(Vertex {
                position: [theta.cos() * radius, y, theta.sin() * radius],
                uv: [s as f32 / segments as f32, v],
            });
        }
    }
    let stride = segments + 1;
    let mut indices: Vec<u16> = Vec::with_capacity((segments * 6) as usize);
    for s in 0..segments {
        let bl = s as u16;
        let br = (s + 1) as u16;
        let tl = (stride + s) as u16;
        let tr = (stride + s + 1) as u16;
        indices.extend_from_slice(&[bl, br, tr, bl, tr, tl]);
    }
    ShapeMesh { vertices, indices }
}

pub fn build_sphere(longitudes: u32, latitudes: u32, radius: f32) -> ShapeMesh {
    let mut vertices = Vec::with_capacity(((longitudes + 1) * (latitudes + 1)) as usize);
    for lat in 0..=latitudes {
        let theta = (lat as f32 / latitudes as f32) * std::f32::consts::PI;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let v = lat as f32 / latitudes as f32;
        for lon in 0..=longitudes {
            let phi = (lon as f32 / longitudes as f32) * std::f32::consts::TAU;
            let x = sin_theta * phi.cos() * radius;
            let y = cos_theta * radius;
            let z = sin_theta * phi.sin() * radius;
            let u = lon as f32 / longitudes as f32;
            vertices.push(Vertex { position: [x, y, z], uv: [u, v] });
        }
    }
    let stride = longitudes + 1;
    let mut indices = Vec::with_capacity((longitudes * latitudes * 6) as usize);
    for lat in 0..latitudes {
        for lon in 0..longitudes {
            let bl = (lat * stride + lon) as u16;
            let br = (lat * stride + lon + 1) as u16;
            let tl = ((lat + 1) * stride + lon) as u16;
            let tr = ((lat + 1) * stride + lon + 1) as u16;
            indices.extend_from_slice(&[bl, br, tr, bl, tr, tl]);
        }
    }
    ShapeMesh { vertices, indices }
}

pub fn build_cube(half_size: f32) -> ShapeMesh {
    let s = half_size;
    let vertices = vec![
        // +X
        Vertex { position: [ s, -s,  s], uv: [0.0, 1.0] },
        Vertex { position: [ s, -s, -s], uv: [1.0, 1.0] },
        Vertex { position: [ s,  s, -s], uv: [1.0, 0.0] },
        Vertex { position: [ s,  s,  s], uv: [0.0, 0.0] },
        // -X
        Vertex { position: [-s, -s, -s], uv: [0.0, 1.0] },
        Vertex { position: [-s, -s,  s], uv: [1.0, 1.0] },
        Vertex { position: [-s,  s,  s], uv: [1.0, 0.0] },
        Vertex { position: [-s,  s, -s], uv: [0.0, 0.0] },
        // +Y
        Vertex { position: [-s,  s,  s], uv: [0.0, 1.0] },
        Vertex { position: [ s,  s,  s], uv: [1.0, 1.0] },
        Vertex { position: [ s,  s, -s], uv: [1.0, 0.0] },
        Vertex { position: [-s,  s, -s], uv: [0.0, 0.0] },
        // -Y
        Vertex { position: [-s, -s, -s], uv: [0.0, 1.0] },
        Vertex { position: [ s, -s, -s], uv: [1.0, 1.0] },
        Vertex { position: [ s, -s,  s], uv: [1.0, 0.0] },
        Vertex { position: [-s, -s,  s], uv: [0.0, 0.0] },
        // +Z
        Vertex { position: [-s, -s,  s], uv: [0.0, 1.0] },
        Vertex { position: [ s, -s,  s], uv: [1.0, 1.0] },
        Vertex { position: [ s,  s,  s], uv: [1.0, 0.0] },
        Vertex { position: [-s,  s,  s], uv: [0.0, 0.0] },
        // -Z
        Vertex { position: [ s, -s, -s], uv: [0.0, 1.0] },
        Vertex { position: [-s, -s, -s], uv: [1.0, 1.0] },
        Vertex { position: [-s,  s, -s], uv: [1.0, 0.0] },
        Vertex { position: [ s,  s, -s], uv: [0.0, 0.0] },
    ];
    let mut indices = Vec::with_capacity(36);
    for face in 0..6u16 {
        let b = face * 4;
        indices.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
    ShapeMesh { vertices, indices }
}

pub fn build_tetrahedron(scale: f32) -> ShapeMesh {
    let s = scale;
    let p0 = [ s,  s,  s];
    let p1 = [ s, -s, -s];
    let p2 = [-s,  s, -s];
    let p3 = [-s, -s,  s];
    let face_uv = [[0.0f32, 1.0], [1.0, 1.0], [0.5, 0.0]];
    let faces = [[p0, p1, p2], [p0, p3, p1], [p0, p2, p3], [p1, p3, p2]];
    let mut vertices = Vec::with_capacity(12);
    let mut indices = Vec::with_capacity(12);
    for (fi, face) in faces.iter().enumerate() {
        let base = (fi * 3) as u16;
        for (i, &pos) in face.iter().enumerate() {
            vertices.push(Vertex { position: pos, uv: face_uv[i] });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    ShapeMesh { vertices, indices }
}

/// Icosahedron: 12 vertices on the unit sphere, 20 triangular faces.
/// Per-face vertex duplication so each triangle gets its own UV.
pub fn build_icosahedron(scale: f32) -> ShapeMesh {
    let t = (1.0_f32 + 5.0_f32.sqrt()) * 0.5; // golden ratio
    let n = (1.0 + t * t).sqrt();
    let a = scale / n;
    let b = scale * t / n;

    let raw: [[f32; 3]; 12] = [
        [-a,  b,  0.0], [ a,  b,  0.0], [-a, -b,  0.0], [ a, -b,  0.0],
        [ 0.0, -a,  b], [ 0.0,  a,  b], [ 0.0, -a, -b], [ 0.0,  a, -b],
        [ b,  0.0, -a], [ b,  0.0,  a], [-b,  0.0, -a], [-b,  0.0,  a],
    ];

    let faces: [[u16; 3]; 20] = [
        [0, 11, 5],  [0, 5, 1],   [0, 1, 7],   [0, 7, 10],  [0, 10, 11],
        [1, 5, 9],   [5, 11, 4],  [11, 10, 2], [10, 7, 6],  [7, 1, 8],
        [3, 9, 4],   [3, 4, 2],   [3, 2, 6],   [3, 6, 8],   [3, 8, 9],
        [4, 9, 5],   [2, 4, 11],  [6, 2, 10],  [8, 6, 7],   [9, 8, 1],
    ];

    let mut vertices = Vec::with_capacity(faces.len() * 3);
    let mut indices  = Vec::with_capacity(faces.len() * 3);
    let face_uv = [[0.0_f32, 1.0], [1.0, 1.0], [0.5, 0.0]];
    for (fi, face) in faces.iter().enumerate() {
        let base = (fi * 3) as u16;
        for (i, &vi) in face.iter().enumerate() {
            vertices.push(Vertex {
                position: raw[vi as usize],
                uv: face_uv[i],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    ShapeMesh { vertices, indices }
}

/// Urchin: low-res sphere body with `spike_count` radial cone spikes.
/// Spike directions are distributed via Fibonacci sphere for even coverage.
pub fn build_urchin(
    spike_count: u32,
    base_radius: f32,
    tip_radius: f32,
    base_ring_segments: u32,
) -> ShapeMesh {
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices:  Vec<u16>    = Vec::new();

    // Inner sphere body (low-res; spikes carry the silhouette).
    let body = build_sphere(24, 16, base_radius * 0.95);
    let body_vert_count = body.vertices.len() as u16;
    vertices.extend_from_slice(&body.vertices);
    indices.extend_from_slice(&body.indices);

    let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    let half_base_angle = (std::f32::consts::TAU / spike_count as f32) * 0.35;

    let mut vertex_offset = body_vert_count;

    for i in 0..spike_count {
        let y = 1.0 - (i as f32 / (spike_count - 1).max(1) as f32) * 2.0;
        let r_xz = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden_angle * i as f32;
        let dir = [r_xz * theta.cos(), y, r_xz * theta.sin()];

        let up = if dir[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
        let right = normalize_cross(up, dir);
        let bitan = normalize_cross(dir, right);

        let tip = [dir[0] * tip_radius, dir[1] * tip_radius, dir[2] * tip_radius];
        let tip_index = vertex_offset;
        vertices.push(Vertex { position: tip, uv: [0.5, 0.0] });
        vertex_offset += 1;

        let ring_start = vertex_offset;
        for s in 0..base_ring_segments {
            let a = (s as f32 / base_ring_segments as f32) * std::f32::consts::TAU;
            let ca = a.cos();
            let sa = a.sin();
            let radial = [
                right[0] * ca + bitan[0] * sa,
                right[1] * ca + bitan[1] * sa,
                right[2] * ca + bitan[2] * sa,
            ];
            let half_a = half_base_angle.sin();
            let along  = half_base_angle.cos();
            let p_dir = [
                dir[0] * along + radial[0] * half_a,
                dir[1] * along + radial[1] * half_a,
                dir[2] * along + radial[2] * half_a,
            ];
            let p = [
                p_dir[0] * base_radius,
                p_dir[1] * base_radius,
                p_dir[2] * base_radius,
            ];
            vertices.push(Vertex {
                position: p,
                uv: [s as f32 / base_ring_segments as f32, 1.0],
            });
        }
        vertex_offset += base_ring_segments as u16;

        for s in 0..base_ring_segments {
            let a = ring_start + s as u16;
            let b = ring_start + ((s + 1) % base_ring_segments) as u16;
            indices.extend_from_slice(&[a, b, tip_index]);
        }
    }

    ShapeMesh { vertices, indices }
}

/// Caltrop: six cones meeting at origin, pointing along ±X, ±Y, ±Z.
pub fn build_caltrop(segments: u32, length: f32, base_radius: f32) -> ShapeMesh {
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices:  Vec<u16>    = Vec::new();

    let axes: [[f32; 3]; 6] = [
        [ 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [ 0.0, 1.0, 0.0], [ 0.0,-1.0, 0.0],
        [ 0.0, 0.0, 1.0], [ 0.0, 0.0,-1.0],
    ];

    let mut vertex_offset: u16 = 0;
    for axis in axes.iter() {
        let up = if axis[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
        let right = normalize_cross(up, *axis);
        let bitan = normalize_cross(*axis, right);

        let tip = [axis[0] * length, axis[1] * length, axis[2] * length];
        let tip_index = vertex_offset;
        vertices.push(Vertex { position: tip, uv: [0.5, 0.0] });
        vertex_offset += 1;

        let ring_start = vertex_offset;
        for s in 0..segments {
            let a = (s as f32 / segments as f32) * std::f32::consts::TAU;
            let ca = a.cos();
            let sa = a.sin();
            let p = [
                right[0] * base_radius * ca + bitan[0] * base_radius * sa,
                right[1] * base_radius * ca + bitan[1] * base_radius * sa,
                right[2] * base_radius * ca + bitan[2] * base_radius * sa,
            ];
            vertices.push(Vertex {
                position: p,
                uv: [s as f32 / segments as f32, 1.0],
            });
        }
        vertex_offset += segments as u16;

        for s in 0..segments {
            let a = ring_start + s as u16;
            let b = ring_start + ((s + 1) % segments) as u16;
            indices.extend_from_slice(&[a, b, tip_index]);
        }
    }

    ShapeMesh { vertices, indices }
}

#[inline]
fn normalize_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    let c = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let len = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().max(1e-6);
    [c[0] / len, c[1] / len, c[2] / len]
}

pub fn build_all_shapes() -> HashMap<ShapeKind, ShapeMesh> {
    let mut map = HashMap::new();
    map.insert(ShapeKind::Cylinder,    build_cylinder(64, 0.6, 1.0));
    map.insert(ShapeKind::Sphere,      build_sphere(64, 32, 0.7));
    map.insert(ShapeKind::Cube,        build_cube(0.6));
    map.insert(ShapeKind::Tetrahedron, build_tetrahedron(0.7));
    map.insert(ShapeKind::Icosahedron, build_icosahedron(0.7));
    map.insert(ShapeKind::Urchin,      build_urchin(48, 0.45, 1.0, 6));
    map.insert(ShapeKind::Caltrop,     build_caltrop(8, 1.0, 0.18));
    map
}

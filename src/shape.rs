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
}

impl ShapeKind {
    pub fn next(self) -> Self {
        match self {
            ShapeKind::Cylinder    => ShapeKind::Sphere,
            ShapeKind::Sphere      => ShapeKind::Cube,
            ShapeKind::Cube        => ShapeKind::Tetrahedron,
            ShapeKind::Tetrahedron => ShapeKind::Cylinder,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ShapeKind::Cylinder    => "Cylinder",
            ShapeKind::Sphere      => "Sphere",
            ShapeKind::Cube        => "Cube",
            ShapeKind::Tetrahedron => "Tetrahedron",
        }
    }

    pub fn model_scale(self) -> f32 {
        match self {
            ShapeKind::Cylinder    => 1.0,
            ShapeKind::Sphere      => 1.1,
            ShapeKind::Cube        => 1.3,
            ShapeKind::Tetrahedron => 1.5,
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

pub fn build_all_shapes() -> HashMap<ShapeKind, ShapeMesh> {
    let mut map = HashMap::new();
    map.insert(ShapeKind::Cylinder,    build_cylinder(64, 0.6, 1.0));
    map.insert(ShapeKind::Sphere,      build_sphere(64, 32, 0.7));
    map.insert(ShapeKind::Cube,        build_cube(0.6));
    map.insert(ShapeKind::Tetrahedron, build_tetrahedron(0.7));
    map
}

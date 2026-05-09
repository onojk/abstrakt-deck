use bytemuck::{Pod, Zeroable};

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

pub struct CylinderMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

/// Build a capped-but-open cylinder with `segments` sides, given `radius` and `half_height`.
/// The +1 seam vertex per ring avoids UV discontinuity at the wrap point.
/// V coordinate: v=0 at top ring, v=1 at bottom ring — matches painter convention.
pub fn build_cylinder(segments: u32, radius: f32, half_height: f32) -> CylinderMesh {
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
        // CCW winding viewed from outside the cylinder.
        indices.extend_from_slice(&[bl, br, tr, bl, tr, tl]);
    }

    CylinderMesh { vertices, indices }
}

//! SDF glyph atlas for the lyric-text overlay.
//!
//! Rasterizes printable ASCII using fontdue, generates a per-glyph signed
//! distance field, and packs everything into a single R8Unorm GPU texture.
//! The SDF encodes 0.5 = on boundary, 1.0 = deep inside, 0.0 = far outside.

use std::collections::HashMap;

/// Atlas cell size in pixels (square).
pub const CELL: u32 = 64;
/// SDF spread radius in atlas-pixel units.  Boundary pixels ±SPREAD from the
/// zero-crossing receive a non-trivial gradient.
pub const SPREAD: f32 = 8.0;
/// Number of columns in the atlas grid.
pub const COLS: u32 = 16;
/// Number of rows (covers all 95 printable ASCII chars 32..=126).
pub const ROWS: u32 = 6;

/// Per-glyph layout data stored alongside the atlas.
/// All `_norm` fields are normalized to `raster_px` (the internal em size),
/// so callers scale them by `(desired_font_px / raster_px)`.
#[derive(Clone, Copy, Debug)]
pub struct GlyphInfo {
    /// Atlas UV rect [u0, v0, u1, v1] for the full CELL×CELL region.
    pub uv: [f32; 4],
    /// Horizontal advance (cursor step) in em units.
    pub advance_norm: f32,
}

/// Fully-built SDF atlas uploaded to the GPU.
pub struct TextAtlas {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    /// Glyph metrics for every printable ASCII character.
    pub glyphs: HashMap<char, GlyphInfo>,
    /// The internal rasterization px size; callers divide desired_font_px by
    /// this to get the uniform scale factor.
    pub raster_px: f32,
    /// Cell baseline position (rows from cell top) at raster_px scale.
    pub baseline_y_cell: i32,
}

impl TextAtlas {
    /// Build the atlas.  `font_data` is raw TTF/OTF bytes.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, font_data: &[u8]) -> Self {
        let font = fontdue::Font::from_bytes(font_data, fontdue::FontSettings::default())
            .expect("fontdue: invalid font data");

        // Choose raster_px so the full ascent+descent fits in CELL - 2*SPREAD.
        let lm1 = font.horizontal_line_metrics(1.0)
            .expect("font has no horizontal metrics");
        let total_h = lm1.ascent - lm1.descent;      // both at scale 1.0
        let usable = CELL as f32 - 2.0 * SPREAD;
        let raster_px = (usable / total_h).floor().max(8.0);

        let lm = font.horizontal_line_metrics(raster_px)
            .expect("font has no horizontal metrics at raster_px");
        let ascent_px  = lm.ascent.ceil() as i32;
        // baseline_y_cell: cell row (from top) of the text baseline.
        let baseline_y_cell = (SPREAD as i32) + ascent_px;

        let atlas_w = COLS * CELL;
        let atlas_h = ROWS * CELL;
        let mut atlas_bytes = vec![0u8; (atlas_w * atlas_h) as usize];
        let mut glyphs: HashMap<char, GlyphInfo> = HashMap::new();

        for code in 32u8..=126 {
            let ch = code as char;
            let idx = (code - 32) as u32;
            let col = idx % COLS;
            let row = idx / COLS;

            // Rasterize glyph.
            let (metrics, bitmap) = font.rasterize(ch, raster_px);
            let bw = metrics.width as i32;
            let bh = metrics.height as i32;

            // Fill glyph bitmap into a CELL×CELL scratch buffer (inside=255, outside=0).
            let mut cell_coverage = vec![0u8; (CELL * CELL) as usize];
            for by in 0..bh {
                for bx in 0..bw {
                    // In the cell, y increases downward.
                    // Bitmap row 0 = top of bounding box = (ymin + height) above baseline.
                    let above_baseline = (metrics.ymin + bh as i32) - by;
                    let cx = (SPREAD as i32) + metrics.xmin + bx;
                    let cy = baseline_y_cell - above_baseline;
                    if cx >= 0 && cx < CELL as i32 && cy >= 0 && cy < CELL as i32 {
                        let coverage = bitmap[(by * bw + bx) as usize];
                        cell_coverage[(cy * CELL as i32 + cx) as usize] = coverage;
                    }
                }
            }

            // Compute SDF for this cell.
            let sdf_cell = compute_sdf(&cell_coverage, CELL as usize, CELL as usize, SPREAD);

            // Write SDF cell into atlas at (col, row).
            let atlas_x0 = col * CELL;
            let atlas_y0 = row * CELL;
            for cy in 0..CELL {
                for cx in 0..CELL {
                    let src = sdf_cell[(cy * CELL + cx) as usize];
                    let dst = (atlas_y0 + cy) * atlas_w + (atlas_x0 + cx);
                    atlas_bytes[dst as usize] = src;
                }
            }

            let u0 = col as f32 / COLS as f32;
            let v0 = row as f32 / ROWS as f32;
            let u1 = (col + 1) as f32 / COLS as f32;
            let v1 = (row + 1) as f32 / ROWS as f32;

            glyphs.insert(ch, GlyphInfo {
                uv: [u0, v0, u1, v1],
                advance_norm: metrics.advance_width / raster_px,
            });
        }

        // Upload atlas texture.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("TextAtlas"),
            size: wgpu::Extent3d { width: atlas_w, height: atlas_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(atlas_w),
                rows_per_image: Some(atlas_h),
            },
            wgpu::Extent3d { width: atlas_w, height: atlas_h, depth_or_array_layers: 1 },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TextAtlas sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self { texture, view, sampler, glyphs, raster_px, baseline_y_cell }
    }
}

/// Try each path in `candidates` until a readable TTF/OTF is found.
/// Returns the raw file bytes.
pub fn load_font_bytes(path_hint: Option<&str>) -> Result<Vec<u8>, String> {
    let candidates: &[&str] = &[
        // caller-supplied hint takes priority
        path_hint.unwrap_or(""),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",           // Arch Linux
        "/usr/share/fonts/fonts-go/Go-Regular.ttf",
        "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
    ];
    for &path in candidates {
        if path.is_empty() { continue; }
        if let Ok(bytes) = std::fs::read(path) {
            log::info!("TextAtlas: loaded font from {}", path);
            return Ok(bytes);
        }
    }
    Err("No system font found for text atlas. Set ABSTRAKT_FONT_PATH or install DejaVuSans.".into())
}

/// Brute-force Euclidean signed distance field.
///
/// `coverage`: W×H grayscale bitmap (0=outside, 255=inside).
/// `spread`:   maximum search radius in pixels; also determines the SDF value scale.
///
/// Returns W×H u8 buffer where:
///   0   = far outside (distance ≥ spread)
///   128 = exactly on the boundary
///   255 = deep inside (distance ≥ spread from any outside pixel)
fn compute_sdf(coverage: &[u8], w: usize, h: usize, spread: f32) -> Vec<u8> {
    let r = spread.ceil() as i32;
    let mut out = vec![0u8; w * h];
    for py in 0..h {
        for px in 0..w {
            let inside = coverage[py * w + px] > 127;
            let mut min_dist_sq = (spread * spread + 1.0) as i32;
            // Scan a (2r+1)×(2r+1) neighbourhood.
            let y0 = (py as i32 - r).max(0);
            let y1 = (py as i32 + r).min(h as i32 - 1);
            let x0 = (px as i32 - r).max(0);
            let x1 = (px as i32 + r).min(w as i32 - 1);
            'outer: for qy in y0..=y1 {
                for qx in x0..=x1 {
                    let q_inside = coverage[qy as usize * w + qx as usize] > 127;
                    if q_inside != inside {
                        let dx = px as i32 - qx;
                        let dy = py as i32 - qy;
                        let dsq = dx * dx + dy * dy;
                        if dsq < min_dist_sq {
                            min_dist_sq = dsq;
                            // Early exit: found an adjacent boundary pixel (distance=1).
                            if min_dist_sq == 1 { break 'outer; }
                        }
                    }
                }
            }
            let min_dist = (min_dist_sq as f32).sqrt().min(spread);
            // Encode: inside → > 0.5, outside → < 0.5, boundary → 0.5.
            let sdf = if inside {
                0.5 + 0.5 * (1.0 - min_dist / spread)
            } else {
                0.5 - 0.5 * (1.0 - min_dist / spread)
            };
            out[py * w + px] = (sdf.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    out
}

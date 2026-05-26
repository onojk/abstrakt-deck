// Staggered Luminance Echo: composite pass for the export-only echo effect.
//
// Three time-offset copies of the same rendered scene are composited:
//   BLACK layer (top):    double-lum-key → alpha, RGB→0  (shadow echo, leads)
//   WHITE layer (middle): double-lum-key → alpha, RGB→255 (highlight echo)
//   COLOR layer (bottom): no key, full opacity             (clean base, trails)
//
// Time stagger (output frame j, delays in frames):
//   black = source[ j + delay_color ]              (most recent → leads visually)
//   white = source[ j + delay_color − delay_white ]
//   color = source[ j ]                            (oldest → trails most)
//
// Output count = total_frames − delay_color_frames  ("trim to fit").
// No deck/wgpu dependencies; uses only std + image + log.

use std::path::{Path, PathBuf};
use image::GenericImageView;

// ── Key thresholds ────────────────────────────────────────────────────────────
// Tune these to match the manual-Kdenlive look.
// Pixels with lum ≤ LUM_LOW_FADE or lum ≥ LUM_HIGH_FADE are fully transparent.
// The opaque band [LUM_LOW_OPAQUE, LUM_HIGH_OPAQUE] gives the echo its shape.
const LUM_LOW_FADE:    f32 = 0.05;
const LUM_LOW_OPAQUE:  f32 = 0.30;
const LUM_HIGH_OPAQUE: f32 = 0.70;
const LUM_HIGH_FADE:   f32 = 0.95;

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline]
fn luma(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b  // ITU-R BT.709
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Double luminance key: midtones → 1.0, near-black and near-white → 0.0.
#[inline]
fn key_alpha(lum: f32) -> f32 {
    smoothstep(LUM_LOW_FADE, LUM_LOW_OPAQUE, lum)
        * (1.0 - smoothstep(LUM_HIGH_OPAQUE, LUM_HIGH_FADE, lum))
}

// ── Per-frame composite ───────────────────────────────────────────────────────

/// Composite one output frame from three RGBA8 slices (row-major, 4 bytes/pixel).
/// `black_src` and `white_src` carry their respective time-offset frames from the
/// same source sequence; `color_src` is the clean base (bottom layer).
pub fn composite_frame(
    black_src: &[u8],
    white_src: &[u8],
    color_src: &[u8],
    w: u32,
    h: u32,
) -> Vec<u8> {
    let n = (w * h) as usize;
    let mut out = Vec::with_capacity(n * 4);

    for i in 0..n {
        let base = i * 4;

        // COLOR layer (bottom): clean scene, opaque
        let cr = color_src[base]     as f32 / 255.0;
        let cg = color_src[base + 1] as f32 / 255.0;
        let cb = color_src[base + 2] as f32 / 255.0;

        // WHITE layer (middle): source lum-keyed, RGB pushed to 1.0
        let wr = white_src[base]     as f32 / 255.0;
        let wg = white_src[base + 1] as f32 / 255.0;
        let wb = white_src[base + 2] as f32 / 255.0;
        let wa = key_alpha(luma(wr, wg, wb));

        // BLACK layer (top): source lum-keyed, RGB pushed to 0.0
        let br = black_src[base]     as f32 / 255.0;
        let bg = black_src[base + 1] as f32 / 255.0;
        let bb = black_src[base + 2] as f32 / 255.0;
        let ba = key_alpha(luma(br, bg, bb));

        // Porter-Duff "over" compositing, bottom to top:
        //   Step 1: white over color  →  lerp(color, 1.0, wa)
        let mid_r = wa + cr * (1.0 - wa);
        let mid_g = wa + cg * (1.0 - wa);
        let mid_b = wa + cb * (1.0 - wa);
        //   Step 2: black over step-1 →  step-1 * (1.0 - ba)  (black = RGB 0)
        let scale = 1.0 - ba;
        out.push((mid_r * scale * 255.0) as u8);
        out.push((mid_g * scale * 255.0) as u8);
        out.push((mid_b * scale * 255.0) as u8);
        out.push(255u8); // output is always opaque
    }

    out
}

// ── Full composite pass ───────────────────────────────────────────────────────

/// Read source PNGs from `renders_dir`, write composited PNGs to `composite_dir`.
/// Returns the composite_dir path so the caller can hand it to ffmpeg.
///
/// Output count = `total_frames` − `delay_color_frames` (trim to fit).
/// `delay_white_frames` is clamped to ≤ `delay_color_frames`.
pub fn run_echo_composite(
    renders_dir:        &Path,
    composite_dir:      &Path,
    total_frames:       u32,
    delay_white_frames: u32,
    delay_color_frames: u32,
) -> Result<PathBuf, String> {
    let dc = delay_color_frames;
    let dw = delay_white_frames.min(dc);

    if dc >= total_frames {
        return Err(format!(
            "[echo] delay_color ({dc} frames) ≥ total frames ({total_frames}); nothing to composite"
        ));
    }

    let output_count = total_frames - dc;
    log::info!("[echo] compositing {output_count} frames  \
                (source {total_frames}, dc={dc}, dw={dw})");

    std::fs::create_dir_all(composite_dir)
        .map_err(|e| format!("[echo] create composite dir: {e}"))?;

    for j in 0..output_count {
        let black_idx = j + dc;
        let white_idx = j + dc - dw;   // ≥ 0 because dw ≤ dc
        let color_idx = j;

        let black_path = renders_dir.join(format!("frame_{black_idx:05}.png"));
        let white_path = renders_dir.join(format!("frame_{white_idx:05}.png"));
        let color_path = renders_dir.join(format!("frame_{color_idx:05}.png"));

        let bimg = image::open(&black_path)
            .map_err(|e| format!("[echo] open black frame {black_idx}: {e}"))?;
        let wimg = image::open(&white_path)
            .map_err(|e| format!("[echo] open white frame {white_idx}: {e}"))?;
        let cimg = image::open(&color_path)
            .map_err(|e| format!("[echo] open color frame {color_idx}: {e}"))?;

        let (w, h) = bimg.dimensions();
        let composited = composite_frame(
            bimg.to_rgba8().as_raw(),
            wimg.to_rgba8().as_raw(),
            cimg.to_rgba8().as_raw(),
            w, h,
        );

        let out_path = composite_dir.join(format!("frame_{j:05}.png"));
        image::save_buffer(&out_path, &composited, w, h, image::ColorType::Rgba8)
            .map_err(|e| format!("[echo] save composite frame {j}: {e}"))?;

        if j % 50 == 0 || j == output_count - 1 {
            log::info!("[echo] compositing frame {}/{output_count}", j + 1);
        }
    }

    log::info!("[echo] composite complete → {}", composite_dir.display());
    Ok(composite_dir.to_path_buf())
}

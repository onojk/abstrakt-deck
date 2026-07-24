/// Cheap luminance statistics over a downsampled RGBA8 buffer.
///
/// `frame_luma_stats` strides across the buffer at a fixed step so it never
/// allocates and never resizes — safe to call every export frame.

pub const DOWNSAMPLE: u32 = 64;   // sample grid (64×64 = 4 096 samples max)
pub const EPS: f32 = 0.04;        // distance from pure 0/1 to count as pinned
pub const BLACK_MEAN: f32 = 0.05; // mean_l below this + degenerate => BLACK
pub const WHITE_MEAN: f32 = 0.95; // mean_l above this + degenerate => WHITE
pub const DEGEN_FRAC: f32 = 0.90; // fraction pinned to an extreme to be degenerate

/// Luminance summary for one frame (or one panel sub-region).
#[derive(Debug, Clone, Copy)]
pub struct FrameStats {
    /// Average BT.709 luminance, 0.0 (black) .. 1.0 (white).
    pub mean_l: f32,
    /// Fraction of sampled pixels within EPS of pure black OR pure white.
    pub degenerate_frac: f32,
}

/// Collapse classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Degeneracy {
    None,
    Black,
    White,
}

impl FrameStats {
    /// Classify per the rule:
    ///   degenerate_frac > DEGEN_FRAC  AND  mean at an extreme.
    /// An extreme mean alone is not enough — a dark scene with bright accents
    /// has a low mean but low degenerate_frac, so it classifies as None.
    pub fn degeneracy(self) -> Degeneracy {
        if self.degenerate_frac > DEGEN_FRAC {
            if self.mean_l < BLACK_MEAN {
                return Degeneracy::Black;
            }
            if self.mean_l > WHITE_MEAN {
                return Degeneracy::White;
            }
        }
        Degeneracy::None
    }
}

/// Analyse a tightly-packed RGBA8 buffer (no row padding, RGBA byte order,
/// length exactly `w * h * 4`).
///
/// Strides by `(w / DOWNSAMPLE).max(1)` in x and y so at most DOWNSAMPLE²
/// pixels are touched regardless of resolution.
pub fn frame_luma_stats(rgba: &[u8], w: u32, h: u32) -> FrameStats {
    if w == 0 || h == 0 || rgba.len() < (w as usize * h as usize * 4) {
        return FrameStats { mean_l: 0.0, degenerate_frac: 1.0 };
    }

    let step_x = (w / DOWNSAMPLE).max(1);
    let step_y = (h / DOWNSAMPLE).max(1);

    let mut sum_l = 0.0f64;
    let mut pinned = 0u32;
    let mut count = 0u32;

    let mut y = 0u32;
    while y < h {
        let mut x = 0u32;
        while x < w {
            let idx = ((y * w + x) * 4) as usize;
            let r = rgba[idx]     as f32 / 255.0;
            let g = rgba[idx + 1] as f32 / 255.0;
            let b = rgba[idx + 2] as f32 / 255.0;
            let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            sum_l += l as f64;
            if l < EPS || l > 1.0 - EPS {
                pinned += 1;
            }
            count += 1;
            x += step_x;
        }
        y += step_y;
    }

    let count_f = count.max(1) as f32;
    FrameStats {
        mean_l: (sum_l as f32) / count_f,
        degenerate_frac: pinned as f32 / count_f,
    }
}

/// Same as `frame_luma_stats` but analyses only the sub-region
/// `[rx, rx+rw) × [ry, ry+rh)` of a buffer that represents a full frame of
/// width `full_w`.  Used by the grid-export path to check each panel.
pub fn region_luma_stats(
    rgba: &[u8],
    full_w: u32,
    rx: u32, ry: u32,
    rw: u32, rh: u32,
) -> FrameStats {
    if rw == 0 || rh == 0 || full_w == 0 {
        return FrameStats { mean_l: 0.0, degenerate_frac: 1.0 };
    }
    let required = ((ry + rh - 1) * full_w + rx + rw) as usize * 4;
    if rgba.len() < required {
        return FrameStats { mean_l: 0.0, degenerate_frac: 1.0 };
    }

    let step_x = (rw / DOWNSAMPLE).max(1);
    let step_y = (rh / DOWNSAMPLE).max(1);

    let mut sum_l = 0.0f64;
    let mut pinned = 0u32;
    let mut count = 0u32;

    let mut dy = 0u32;
    while dy < rh {
        let mut dx = 0u32;
        while dx < rw {
            let idx = (((ry + dy) * full_w + (rx + dx)) * 4) as usize;
            let r = rgba[idx]     as f32 / 255.0;
            let g = rgba[idx + 1] as f32 / 255.0;
            let b = rgba[idx + 2] as f32 / 255.0;
            let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            sum_l += l as f64;
            if l < EPS || l > 1.0 - EPS {
                pinned += 1;
            }
            count += 1;
            dx += step_x;
        }
        dy += step_y;
    }

    let count_f = count.max(1) as f32;
    FrameStats {
        mean_l: (sum_l as f32) / count_f,
        degenerate_frac: pinned as f32 / count_f,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            v.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
        }
        v
    }

    #[test]
    fn all_black_is_black() {
        let s = frame_luma_stats(&solid(128, 128, [0, 0, 0]), 128, 128);
        assert!(s.degenerate_frac > 0.99, "all-black frac={}", s.degenerate_frac);
        assert_eq!(s.degeneracy(), Degeneracy::Black);
    }

    #[test]
    fn all_white_is_white() {
        let s = frame_luma_stats(&solid(128, 128, [255, 255, 255]), 128, 128);
        assert!(s.degenerate_frac > 0.99, "all-white frac={}", s.degenerate_frac);
        assert_eq!(s.degeneracy(), Degeneracy::White);
    }

    #[test]
    fn mid_gray_is_fine() {
        let s = frame_luma_stats(&solid(128, 128, [128, 128, 128]), 128, 128);
        assert_eq!(s.degeneracy(), Degeneracy::None);
    }

    #[test]
    fn mostly_black_few_bright_still_black() {
        // 95% black pixels, 5% white scattered → degenerate_frac still > 0.9
        // (both extremes count as pinned), mean_l ≈ 0.05×1.0 = 0.05 — just on the
        // boundary; use a smaller bright fraction so mean stays comfortably < 0.05.
        let mut v = solid(100, 100, [0, 0, 0]);
        // Paint top 2 rows white (200 / 10000 = 2%)
        for i in 0..200usize {
            let idx = i * 4;
            v[idx] = 255;
            v[idx + 1] = 255;
            v[idx + 2] = 255;
        }
        let s = frame_luma_stats(&v, 100, 100);
        assert_eq!(
            s.degeneracy(),
            Degeneracy::Black,
            "mean_l={:.4} frac={:.4}",
            s.mean_l,
            s.degenerate_frac
        );
    }

    #[test]
    fn zero_dimensions_returns_safe_default() {
        let s = frame_luma_stats(&[], 0, 0);
        // degenerate_frac = 1.0 → classifies as Black but at least doesn't panic.
        assert_eq!(s.mean_l, 0.0);
        assert_eq!(s.degenerate_frac, 1.0);
    }

    #[test]
    fn region_matches_full_frame_when_equal() {
        let v = solid(64, 64, [0, 0, 0]);
        let full = frame_luma_stats(&v, 64, 64);
        let region = region_luma_stats(&v, 64, 0, 0, 64, 64);
        assert!((full.mean_l - region.mean_l).abs() < 1e-4);
        assert!((full.degenerate_frac - region.degenerate_frac).abs() < 1e-4);
    }

    #[test]
    fn region_isolates_white_quadrant() {
        // Left half black, right half white (128×64 total, two 64×64 halves)
        let mut v = solid(128, 64, [0, 0, 0]);
        for row in 0..64u32 {
            for col in 64..128u32 {
                let idx = ((row * 128 + col) * 4) as usize;
                v[idx] = 255;
                v[idx + 1] = 255;
                v[idx + 2] = 255;
            }
        }
        let left  = region_luma_stats(&v, 128, 0,  0, 64, 64);
        let right = region_luma_stats(&v, 128, 64, 0, 64, 64);
        assert_eq!(left.degeneracy(),  Degeneracy::Black);
        assert_eq!(right.degeneracy(), Degeneracy::White);
    }
}

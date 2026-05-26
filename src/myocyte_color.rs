// Minimal HSV/RGB conversion for cell hue rotation.
// Ported from myocyte/src/color.rs — only the two conversion fns,
// not the harmony palette generator.

/// Wraps any f32 hue value into [0, 360).
#[inline]
pub fn wrap_hue(hue_deg: f32) -> f32 {
    let h = hue_deg % 360.0;
    if h < 0.0 { h + 360.0 } else { h }
}

/// Standard six-sector HSV → linear RGB conversion.
/// `h` is wrapped to [0, 360) via wrap_hue. `s` and `v` are clamped to [0, 1].
/// Output components are in [0, 1] (linear RGB — no sRGB conversion).
#[inline]
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = wrap_hue(h);
    let s = s.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);

    if s == 0.0 {
        return [v, v, v];
    }

    let h6 = h / 60.0;
    let i  = h6.floor() as u32;
    let f  = h6 - i as f32;
    let p  = v * (1.0 - s);
    let q  = v * (1.0 - s * f);
    let t  = v * (1.0 - s * (1.0 - f));

    match i % 6 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// Inverse of hsv_to_rgb. Inputs clamped to [0, 1].
/// Returns `(hue_deg in [0, 360), saturation, value)`.
/// Pure grays (saturation = 0) return hue = 0 by convention.
#[allow(dead_code)]
#[inline]
pub fn rgb_to_hsv(rgb: [f32; 3]) -> (f32, f32, f32) {
    let r = rgb[0].clamp(0.0, 1.0);
    let g = rgb[1].clamp(0.0, 1.0);
    let b = rgb[2].clamp(0.0, 1.0);

    let v   = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d   = v - min;

    if v < 1e-7 {
        return (0.0, 0.0, 0.0);
    }

    let s = d / v;

    if d < 1e-7 {
        return (0.0, s, v);
    }

    let h_raw = if r >= v {
        (g - b) / d
    } else if g >= v {
        2.0 + (b - r) / d
    } else {
        4.0 + (r - g) / d
    };

    (wrap_hue(h_raw * 60.0), s, v)
}

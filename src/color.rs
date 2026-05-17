//! Color theory helpers for abstrakt-deck.
//!
//! Provides HSV ↔ RGB conversion, hue-wheel arithmetic in degrees, and a
//! `ColorHarmony` enum encoding six classical harmonic relationships drawn
//! from Hornung's *Color: A Workshop for Artists and Designers* (Part 7).
//!
//! Random color generation in HSV space avoids the muddy mid-tone bias
//! you get from picking RGB triples uniformly. Picking H/S/V independently
//! also lets us clamp to artistic ranges — e.g. "muted color" = S in 0.3..0.6,
//! "chromatic gray" = S in 0.05..0.25, "high-key" = V in 0.75..1.0, etc.
//!
//! Conventions throughout this module:
//!   * Hue is in DEGREES on [0, 360). Wraps modularly.
//!   * Saturation and Value are in [0, 1].
//!   * RGB outputs are linear [0, 1] `[f32; 3]` — same format as Params
//!     uses elsewhere, so palettes drop straight into the existing pipeline.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Six classical color-harmony relationships, defined by how derived hues
/// relate to a single anchor hue on the 360° hue wheel.
///
/// Visual analogy: imagine the anchor as 12 o'clock on a clock face; each
/// variant specifies which other clock positions the palette occupies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ColorHarmony {
    /// All colors share the anchor hue; variation comes from value/saturation only.
    Monochromatic,
    /// Anchor ± 30°. Adjacent on the wheel, "consonant" feel.
    #[default]
    Analogous,
    /// Anchor + 180°. Maximum hue tension, "dissonant resolution" feel.
    Complementary,
    /// Anchor + 150° + 210°. A softer complement — points just shy of the opposite.
    SplitComplementary,
    /// Anchor + 120° + 240°. Equilateral triangle, Hornung's "primary triad" symmetry.
    Triadic,
    /// Anchor + 90° + 180° + 270°. Rectangular harmony, four colors.
    Tetradic,
}

impl ColorHarmony {
    /// Cycle to the next harmony (used by the hotkey).
    pub fn next(self) -> Self {
        match self {
            ColorHarmony::Monochromatic      => ColorHarmony::Analogous,
            ColorHarmony::Analogous          => ColorHarmony::Complementary,
            ColorHarmony::Complementary      => ColorHarmony::SplitComplementary,
            ColorHarmony::SplitComplementary => ColorHarmony::Triadic,
            ColorHarmony::Triadic            => ColorHarmony::Tetradic,
            ColorHarmony::Tetradic           => ColorHarmony::Monochromatic,
        }
    }

    /// Human-readable name, used in menu bar dropdowns.
    pub fn name(self) -> &'static str {
        match self {
            ColorHarmony::Monochromatic      => "Monochromatic",
            ColorHarmony::Analogous          => "Analogous",
            ColorHarmony::Complementary      => "Complementary",
            ColorHarmony::SplitComplementary => "Split-Complementary",
            ColorHarmony::Triadic            => "Triadic",
            ColorHarmony::Tetradic           => "Tetradic",
        }
    }

    /// The set of hue offsets (in degrees) this harmony adds to the anchor.
    /// Always contains 0.0 (the anchor itself) as the first element.
    pub fn hue_offsets(self) -> &'static [f32] {
        match self {
            ColorHarmony::Monochromatic      => &[0.0],
            ColorHarmony::Analogous          => &[0.0, -30.0, 30.0],
            ColorHarmony::Complementary      => &[0.0, 180.0],
            ColorHarmony::SplitComplementary => &[0.0, 150.0, 210.0],
            ColorHarmony::Triadic            => &[0.0, 120.0, 240.0],
            ColorHarmony::Tetradic           => &[0.0, 90.0, 180.0, 270.0],
        }
    }
}

/// Wrap a hue value into [0, 360).
///
/// Works correctly for any finite f32, positive or negative, large or small.
#[inline]
pub fn wrap_hue(hue_deg: f32) -> f32 {
    let h = hue_deg % 360.0;
    if h < 0.0 { h + 360.0 } else { h }
}

/// Convert HSV → linear RGB.
///
/// h is in degrees [0, 360) (out-of-range inputs are wrapped); s and v are
/// clamped to [0, 1]. Returns RGB as `[f32; 3]` in [0, 1].
///
/// Algorithm: standard HSV cylinder, six-sector form. Matches the algorithm
/// described in any color theory text and produces results consistent with
/// other libraries (e.g. palette crate, Adobe Color, Photoshop).
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = wrap_hue(h);
    let s = s.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);

    let c = v * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = match h_prime as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r1 + m, g1 + m, b1 + m]
}

/// Convert linear RGB → HSV.
///
/// Inverse of `hsv_to_rgb`. RGB inputs are clamped to [0, 1]. Returns
/// `(hue_deg, saturation, value)` where hue is in [0, 360).
/// Gray colors (where r == g == b) return hue=0 by convention.
pub fn rgb_to_hsv(rgb: [f32; 3]) -> (f32, f32, f32) {
    let r = rgb[0].clamp(0.0, 1.0);
    let g = rgb[1].clamp(0.0, 1.0);
    let b = rgb[2].clamp(0.0, 1.0);

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;
    let s = if max > 1e-6 { delta / max } else { 0.0 };

    let h = if delta < 1e-6 {
        0.0  // pure gray — hue undefined; pick 0 by convention
    } else if (max - r).abs() < 1e-6 {
        60.0 * (((g - b) / delta) % 6.0)
    } else if (max - g).abs() < 1e-6 {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    (wrap_hue(h), s, v)
}

/// Build a palette of N colors from a harmony, an anchor hue, and target S/V.
///
/// The palette walks through `harmony.hue_offsets()` repeatedly, modulating
/// value and saturation slightly across each cycle so a 6-color tetradic
/// palette (4 hues, 6 slots) gets visible variety rather than two duplicates.
///
/// Returns exactly `count` colors. Useful for assigning N painter / shape /
/// kaleido segments from a single high-level harmony choice.
pub fn palette_from_harmony(
    harmony:     ColorHarmony,
    anchor_hue:  f32,
    saturation:  f32,
    value:       f32,
    count:       usize,
) -> Vec<[f32; 3]> {
    if count == 0 {
        return Vec::new();
    }

    let offsets = harmony.hue_offsets();
    let mut out = Vec::with_capacity(count);

    for i in 0..count {
        let offset = offsets[i % offsets.len()];
        let cycle = (i / offsets.len()) as f32;

        // Subtle per-cycle modulation so repeats don't duplicate exactly.
        // Each cycle nudges value down and saturation up slightly, keeping
        // the palette readable while adding variety.
        let v = (value  - cycle * 0.12).clamp(0.15, 1.0);
        let s = (saturation + cycle * 0.08).clamp(0.0, 1.0);

        out.push(hsv_to_rgb(anchor_hue + offset, s, v));
    }
    out
}

/// Pick a random color in HSV space using artistically tasteful ranges.
///
/// Compared to `rng.gen::<[f32; 3]>()` (uniform RGB), this avoids the
/// muddy-mid-tone bias and gives each random pick a real "identity" —
/// a clear hue, decent saturation, decent value. Use this everywhere
/// the visualizer currently picks a random RGB triple.
///
/// `hue_range_deg`: `[min, max]` range of allowed hues, wrapping; pass
///                  `[0.0, 360.0]` for the full wheel.
/// `sat_range`:     `[min, max]` in [0, 1]
/// `val_range`:     `[min, max]` in [0, 1]
pub fn random_hsv<R: rand::Rng>(
    rng:           &mut R,
    hue_range_deg: [f32; 2],
    sat_range:     [f32; 2],
    val_range:     [f32; 2],
) -> [f32; 3] {
    let h = rng.gen_range(hue_range_deg[0]..=hue_range_deg[1]);
    let s = rng.gen_range(sat_range[0]    ..=sat_range[1]   );
    let v = rng.gen_range(val_range[0]    ..=val_range[1]   );
    hsv_to_rgb(h, s, v)
}

/// Convenience: pick a random color uniformly across the hue wheel with
/// reasonable saturation/value defaults (S in 0.55..0.95, V in 0.65..1.0).
///
/// This is the "drop-in replacement" for `rng.gen::<[f32; 3]>()` — same
/// shape, different distribution. Use this first; reach for `random_hsv`
/// when a specific range is needed.
pub fn random_color_tasteful<R: rand::Rng>(rng: &mut R) -> [f32; 3] {
    random_hsv(rng, [0.0, 360.0], [0.55, 0.95], [0.65, 1.0])
}

/// Hornung Part 2 saturation categories. Each constrains color generation
/// to a band of saturation values appropriate to that artistic mode.
///
/// `Free` means no constraint — the saturation slider can be anywhere in
/// [0, 1] and randomization picks across the full range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SaturationMode {
    /// No constraint — saturation can be anywhere in [0, 1]
    #[default]
    Free,
    /// High saturation: vibrant, punchy, electric. Pop art / neon territory.
    Pure,
    /// Moderate saturation: sophisticated, restrained. Where most fine art lives.
    Muted,
    /// Very low saturation but still hue-identified. Atmospheric, fog-like.
    ChromaticGray,
}

impl SaturationMode {
    pub fn next(self) -> Self {
        match self {
            SaturationMode::Free          => SaturationMode::Pure,
            SaturationMode::Pure          => SaturationMode::Muted,
            SaturationMode::Muted         => SaturationMode::ChromaticGray,
            SaturationMode::ChromaticGray => SaturationMode::Free,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            SaturationMode::Free          => "Free",
            SaturationMode::Pure          => "Pure",
            SaturationMode::Muted         => "Muted",
            SaturationMode::ChromaticGray => "Chromatic Gray",
        }
    }

    pub fn range(self) -> [f32; 2] {
        match self {
            SaturationMode::Free          => [0.00, 1.00],
            SaturationMode::Pure          => [0.75, 1.00],
            SaturationMode::Muted         => [0.30, 0.65],
            SaturationMode::ChromaticGray => [0.05, 0.25],
        }
    }

    pub fn default_value(self) -> f32 {
        let r = self.range();
        (r[0] + r[1]) * 0.5
    }

    pub fn clamp(self, sat: f32) -> f32 {
        let r = self.range();
        sat.clamp(r[0], r[1])
    }
}

/// Hornung Part 4/7 value-key composition. Each key constrains color
/// generation to a brightness band that produces a distinct tonal mood.
///
/// `Free` means no constraint — value can be anywhere in [0, 1] and
/// randomization picks across the full range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ValueKey {
    /// No constraint — value can be anywhere in [0, 1]
    #[default]
    Free,
    /// Light, airy, ethereal: most colors in the upper brightness range.
    High,
    /// Balanced midtones: where most photographs and "normal" scenes live.
    Mid,
    /// Moody, cinematic, dark with selective highlights.
    Low,
}

impl ValueKey {
    pub fn next(self) -> Self {
        match self {
            ValueKey::Free => ValueKey::High,
            ValueKey::High => ValueKey::Mid,
            ValueKey::Mid  => ValueKey::Low,
            ValueKey::Low  => ValueKey::Free,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ValueKey::Free => "Free",
            ValueKey::High => "High key",
            ValueKey::Mid  => "Mid key",
            ValueKey::Low  => "Low key",
        }
    }

    pub fn range(self) -> [f32; 2] {
        match self {
            ValueKey::Free => [0.00, 1.00],
            ValueKey::High => [0.75, 1.00],
            ValueKey::Mid  => [0.40, 0.75],
            ValueKey::Low  => [0.15, 0.40],
        }
    }

    pub fn default_value(self) -> f32 {
        let r = self.range();
        (r[0] + r[1]) * 0.5
    }

    pub fn clamp(self, value: f32) -> f32 {
        let r = self.range();
        value.clamp(r[0], r[1])
    }
}

/// Pull a hue toward the warm pole (30° = orange) or cool pole (210° = cyan).
///
/// `bias` is in [-1, 1]:
///   * -1.0 → maximum cool pull (hue rotates toward 210°)
///   *  0.0 → no change
///   * +1.0 → maximum warm pull (hue rotates toward 30°)
///
/// At full bias the hue moves up to 60° around the wheel toward the target
/// pole — enough to shift temperature feel substantially without collapsing
/// all hues onto the pole.
pub fn apply_temperature_bias(hue_deg: f32, bias: f32) -> f32 {
    let bias = bias.clamp(-1.0, 1.0);
    if bias.abs() < 1e-4 {
        return wrap_hue(hue_deg);
    }

    let pole = if bias > 0.0 { 30.0_f32 } else { 210.0_f32 };

    let raw_delta = pole - wrap_hue(hue_deg);
    let delta = if raw_delta > 180.0 {
        raw_delta - 360.0
    } else if raw_delta < -180.0 {
        raw_delta + 360.0
    } else {
        raw_delta
    };

    let max_pull = 60.0_f32;
    let pull = delta.signum() * delta.abs().min(max_pull) * bias.abs();
    wrap_hue(hue_deg + pull)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn wrap_hue_handles_negatives_and_overflow() {
        assert!(approx(wrap_hue(  0.0),   0.0, 1e-4));
        assert!(approx(wrap_hue(360.0),   0.0, 1e-4));
        assert!(approx(wrap_hue(361.0),   1.0, 1e-4));
        assert!(approx(wrap_hue(-30.0), 330.0, 1e-4));
        assert!(approx(wrap_hue(-720.0),  0.0, 1e-4));
        assert!(approx(wrap_hue(720.5),   0.5, 1e-4));
    }

    #[test]
    fn hsv_primaries_round_trip() {
        let cases = [
            (  0.0, 1.0, 1.0, [1.0, 0.0, 0.0]),
            (120.0, 1.0, 1.0, [0.0, 1.0, 0.0]),
            (240.0, 1.0, 1.0, [0.0, 0.0, 1.0]),
            ( 60.0, 1.0, 1.0, [1.0, 1.0, 0.0]),
            (180.0, 1.0, 1.0, [0.0, 1.0, 1.0]),
            (300.0, 1.0, 1.0, [1.0, 0.0, 1.0]),
        ];
        for (h, s, v, expected_rgb) in cases {
            let rgb = hsv_to_rgb(h, s, v);
            for i in 0..3 {
                assert!(
                    approx(rgb[i], expected_rgb[i], 1e-4),
                    "hsv({}, {}, {})[{}] = {} expected {}",
                    h, s, v, i, rgb[i], expected_rgb[i]
                );
            }
        }
    }

    #[test]
    fn hsv_to_rgb_to_hsv_round_trip() {
        let cases = [
            ( 30.0, 0.5, 0.8),
            (150.0, 0.9, 0.4),
            (210.0, 0.3, 0.6),
            (330.0, 0.75, 0.95),
            ( 45.0, 0.1, 1.0),
        ];
        for (h, s, v) in cases {
            let rgb = hsv_to_rgb(h, s, v);
            let (h2, s2, v2) = rgb_to_hsv(rgb);
            assert!(approx(h, h2, 0.1), "hue round-trip: {} → {}", h, h2);
            assert!(approx(s, s2, 1e-3), "sat round-trip: {} → {}", s, s2);
            assert!(approx(v, v2, 1e-3), "val round-trip: {} → {}", v, v2);
        }
    }

    #[test]
    fn gray_rgb_returns_zero_hue() {
        let (h, s, _v) = rgb_to_hsv([0.5, 0.5, 0.5]);
        assert!(approx(h, 0.0, 1e-4), "gray hue should be 0, got {}", h);
        assert!(approx(s, 0.0, 1e-4), "gray sat should be 0, got {}", s);
    }

    #[test]
    fn harmony_offsets_always_contain_anchor() {
        let all = [
            ColorHarmony::Monochromatic,
            ColorHarmony::Analogous,
            ColorHarmony::Complementary,
            ColorHarmony::SplitComplementary,
            ColorHarmony::Triadic,
            ColorHarmony::Tetradic,
        ];
        for h in all {
            assert!(
                h.hue_offsets().contains(&0.0),
                "{:?} hue_offsets must include 0.0 anchor",
                h
            );
        }
    }

    #[test]
    fn harmony_next_cycles_through_all_six() {
        let mut h = ColorHarmony::Monochromatic;
        let mut seen = std::collections::HashSet::new();
        for _ in 0..6 {
            seen.insert(format!("{:?}", h));
            h = h.next();
        }
        assert_eq!(seen.len(), 6, "next() should visit all 6 variants in 6 steps");
        assert_eq!(h, ColorHarmony::Monochromatic, "should return to start after 6 calls");
    }

    #[test]
    fn palette_returns_exact_count() {
        assert_eq!(palette_from_harmony(ColorHarmony::Triadic, 0.0, 0.7, 0.8, 0).len(), 0);
        assert_eq!(palette_from_harmony(ColorHarmony::Triadic, 0.0, 0.7, 0.8, 3).len(), 3);
        assert_eq!(palette_from_harmony(ColorHarmony::Triadic, 0.0, 0.7, 0.8, 8).len(), 8);
        assert_eq!(palette_from_harmony(ColorHarmony::Monochromatic, 200.0, 0.5, 0.5, 5).len(), 5);
    }

    #[test]
    fn random_color_tasteful_is_in_range() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let rgb = random_color_tasteful(&mut rng);
            for &c in &rgb {
                assert!((0.0..=1.0).contains(&c), "channel {} out of range", c);
            }
            let (_h, s, v) = rgb_to_hsv(rgb);
            assert!(s >= 0.5,  "tasteful color should be at least 0.5 saturated, got {}", s);
            assert!(v >= 0.65, "tasteful color should be at least 0.65 value, got {}", v);
        }
    }

    #[test]
    fn saturation_mode_ranges_are_disjoint() {
        let pure  = SaturationMode::Pure.range();
        let muted = SaturationMode::Muted.range();
        let gray  = SaturationMode::ChromaticGray.range();
        assert!(gray[1]  < muted[0], "Chromatic Gray ceiling {} should be below Muted floor {}", gray[1], muted[0]);
        assert!(muted[1] < pure[0],  "Muted ceiling {} should be below Pure floor {}", muted[1], pure[0]);
    }

    #[test]
    fn saturation_mode_clamp_constrains() {
        assert_eq!(SaturationMode::Pure.clamp(0.1),  0.75);
        assert_eq!(SaturationMode::Pure.clamp(1.0),  1.0);
        assert_eq!(SaturationMode::Muted.clamp(0.9), 0.65);
        assert_eq!(SaturationMode::Muted.clamp(0.0), 0.30);
        assert_eq!(SaturationMode::ChromaticGray.clamp(0.5), 0.25);
        assert_eq!(SaturationMode::Free.clamp(0.5),  0.5);
        assert_eq!(SaturationMode::Free.clamp(-0.5), 0.0);
        assert_eq!(SaturationMode::Free.clamp(2.0),  1.0);
    }

    #[test]
    fn saturation_mode_default_is_in_range() {
        for m in [
            SaturationMode::Free,
            SaturationMode::Pure,
            SaturationMode::Muted,
            SaturationMode::ChromaticGray,
        ] {
            let v = m.default_value();
            let r = m.range();
            assert!(v >= r[0] && v <= r[1],
                "{:?} default {} should be in range [{}, {}]", m, v, r[0], r[1]);
        }
    }

    #[test]
    fn value_key_ranges_are_disjoint() {
        let low  = ValueKey::Low.range();
        let mid  = ValueKey::Mid.range();
        let high = ValueKey::High.range();
        assert!(low[1]  <= mid[0],  "Low ceiling {} should be ≤ Mid floor {}",  low[1],  mid[0]);
        assert!(mid[1]  <= high[0], "Mid ceiling {} should be ≤ High floor {}", mid[1],  high[0]);
    }

    #[test]
    fn value_key_clamp_constrains() {
        assert_eq!(ValueKey::High.clamp(0.1),  0.75);
        assert_eq!(ValueKey::High.clamp(1.0),  1.0);
        assert_eq!(ValueKey::Mid.clamp(0.9),   0.75);
        assert_eq!(ValueKey::Mid.clamp(0.0),   0.40);
        assert_eq!(ValueKey::Low.clamp(0.9),   0.40);
        assert_eq!(ValueKey::Low.clamp(0.0),   0.15);
        assert_eq!(ValueKey::Free.clamp(0.5),   0.5);
        assert_eq!(ValueKey::Free.clamp(-0.5),  0.0);
        assert_eq!(ValueKey::Free.clamp(2.0),   1.0);
    }

    #[test]
    fn value_key_default_is_in_range() {
        for k in [ValueKey::Free, ValueKey::High, ValueKey::Mid, ValueKey::Low] {
            let v = k.default_value();
            let r = k.range();
            assert!(v >= r[0] && v <= r[1],
                "{:?} default {} should be in range [{}, {}]", k, v, r[0], r[1]);
        }
    }

    #[test]
    fn value_key_next_cycles_through_all() {
        let mut k = ValueKey::Free;
        let mut seen = std::collections::HashSet::new();
        for _ in 0..4 {
            seen.insert(format!("{:?}", k));
            k = k.next();
        }
        assert_eq!(seen.len(), 4);
        assert_eq!(k, ValueKey::Free);
    }

    #[test]
    fn temperature_zero_is_identity() {
        for h in [0.0, 90.0, 180.0, 270.0, 359.9] {
            let result = apply_temperature_bias(h, 0.0);
            assert!((wrap_hue(h) - result).abs() < 1e-3,
                "hue {} with zero bias should be ~{}, got {}", h, h, result);
        }
    }

    #[test]
    fn temperature_warm_pulls_toward_orange() {
        let result = apply_temperature_bias(240.0, 1.0);
        assert!(result > 240.0 && result < 360.0,
            "blue under full warm bias should move toward warm, got {}", result);
    }

    #[test]
    fn temperature_cool_pulls_toward_cyan() {
        let result = apply_temperature_bias(0.0, -1.0);
        assert!(result > 270.0 && result < 360.0,
            "red under full cool bias should rotate through magenta toward cyan, got {}", result);
    }

    #[test]
    fn temperature_already_warm_stays_warm() {
        let result = apply_temperature_bias(30.0, 1.0);
        assert!((result - 30.0).abs() < 1.0,
            "hue at warm pole should not move under warm bias, got {}", result);
    }

    #[test]
    fn temperature_clamps_to_bounds() {
        let r1 = apply_temperature_bias(180.0,  5.0);
        let r2 = apply_temperature_bias(180.0,  1.0);
        let r3 = apply_temperature_bias(180.0, -5.0);
        let r4 = apply_temperature_bias(180.0, -1.0);
        assert!((r1 - r2).abs() < 1e-3, "bias > 1 should clamp to 1");
        assert!((r3 - r4).abs() < 1e-3, "bias < -1 should clamp to -1");
    }

    #[test]
    fn saturation_mode_next_cycles_through_all() {
        let mut m = SaturationMode::Free;
        let mut seen = std::collections::HashSet::new();
        for _ in 0..4 {
            seen.insert(format!("{:?}", m));
            m = m.next();
        }
        assert_eq!(seen.len(), 4);
        assert_eq!(m, SaturationMode::Free);
    }
}

/// Krumhansl-Schmuckler key profiles (Krumhansl & Kessler 1982).
/// Index 0 = root, 1 = minor second above root, …, 11 = major seventh.
const KS_MAJOR: [f32; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];
const KS_MINOR: [f32; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// Accumulate FFT magnitudes into 12 chroma bins.
///
/// Each bin's frequency is mapped to the nearest MIDI pitch class (C=0…B=11)
/// via `midi = 69 + 12*log2(freq/440)`. Frequencies outside the piano range
/// A0 (27.5 Hz) – C8 (4186 Hz) are ignored. Uses half-wave-rectified
/// magnitudes from the caller — pass the same `spectral_magnitudes` output
/// that the onset detector uses.
pub fn chromagram(magnitudes: &[f32], sample_rate: u32, fft_size: usize) -> [f32; 12] {
    let bin_hz = sample_rate as f32 / fft_size as f32;
    let mut chroma = [0.0f32; 12];
    for (bin, &mag) in magnitudes.iter().enumerate() {
        let freq = bin as f32 * bin_hz;
        if !(27.5..=4186.0).contains(&freq) {
            continue;
        }
        let midi = 69.0 + 12.0 * (freq / 440.0).log2();
        let pc = midi.round() as i32;
        let pc = ((pc % 12) + 12) as usize % 12;
        chroma[pc] += mag;
    }
    chroma
}

/// Normalize `v` to unit L2 norm in-place.
/// Returns `false` (and leaves `v` unchanged) if the vector is all-zero.
pub fn normalize_profile(v: &mut [f32; 12]) -> bool {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-9 {
        return false;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    true
}

/// Pearson correlation between `chroma` rotated by `rot` semitones and `profile`.
///
/// `rot=0` tests the key whose root is pitch-class 0 (C), `rot=1` tests C#, etc.
/// The rotation shifts which chroma bin aligns with which profile slot:
///   `chroma[(i + rot) % 12]` is compared to `profile[i]` for i in 0..12.
pub fn pearson_rotated(chroma: &[f32; 12], profile: &[f32; 12], rot: usize) -> f32 {
    let mean_c: f32 = chroma.iter().sum::<f32>() / 12.0;
    let mean_p: f32 = profile.iter().sum::<f32>() / 12.0;
    let mut num = 0.0f32;
    let mut ss_c = 0.0f32;
    let mut ss_p = 0.0f32;
    for i in 0..12 {
        let c = chroma[(i + rot) % 12] - mean_c;
        let p = profile[i] - mean_p;
        num += c * p;
        ss_c += c * c;
        ss_p += p * p;
    }
    let den = (ss_c * ss_p).sqrt();
    if den < 1e-9 {
        0.0
    } else {
        num / den
    }
}

/// Pitch-class name for display (C=0 … B=11).
pub fn pitch_class_name(root: u8) -> &'static str {
    match root {
        0  => "C",
        1  => "C#",
        2  => "D",
        3  => "D#",
        4  => "E",
        5  => "F",
        6  => "F#",
        7  => "G",
        8  => "G#",
        9  => "A",
        10 => "A#",
        11 => "B",
        _  => "?",
    }
}

/// Krumhansl-Schmuckler key detector.
///
/// Accumulates a sliding window of normalized chromagram frames, then runs
/// K-S analysis on the window sum every `estimate_period_chunks` chunks.
/// Pattern mirrors `TempoTracker`: public output fields are updated in-place
/// and can be read after each `process_chunk` call.
#[derive(Debug, Clone)]
pub struct KeyTracker {
    chroma_history:          Vec<[f32; 12]>,
    history_size:            usize,
    chunks_since_estimate:   u32,
    estimate_period_chunks:  u32,
    pub key_root:       u8,    // 0=C … 11=B
    pub key_is_major:   bool,
    pub key_confidence: f32,   // Pearson r of best-matching K-S profile
    pub chroma_peak:    f32,   // strongest pitch class mapped to 0..1 (for hue)
    pub chroma:         [f32; 12],
}

impl KeyTracker {
    /// How many seconds of chroma history to accumulate before K-S analysis.
    const HISTORY_SECONDS: f32 = 4.0;
    /// How often (in seconds) to re-run K-S on the accumulated window.
    const ESTIMATE_PERIOD_SECONDS: f32 = 0.5;

    pub fn new(chunk_seconds: f32) -> Self {
        let history_size =
            (Self::HISTORY_SECONDS / chunk_seconds).ceil() as usize;
        let estimate_period_chunks =
            (Self::ESTIMATE_PERIOD_SECONDS / chunk_seconds).ceil() as u32;
        Self {
            chroma_history: Vec::with_capacity(history_size),
            history_size,
            chunks_since_estimate: 0,
            estimate_period_chunks: estimate_period_chunks.max(1),
            key_root:       0,
            key_is_major:   true,
            key_confidence: 0.0,
            chroma_peak:    0.0,
            chroma:         [0.0; 12],
        }
    }

    /// Feed one raw chromagram frame. All-zero frames are silently dropped.
    pub fn process_chunk(&mut self, raw_chroma: [f32; 12]) {
        let mut c = raw_chroma;
        if !normalize_profile(&mut c) {
            return;
        }

        // Update instantaneous chroma_peak (strongest bin → 0..1 hue index).
        let peak_idx = c
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.chroma_peak = peak_idx as f32 / 12.0;
        self.chroma = c;

        self.chroma_history.push(c);
        if self.chroma_history.len() > self.history_size {
            self.chroma_history.remove(0);
        }

        self.chunks_since_estimate += 1;
        if self.chunks_since_estimate >= self.estimate_period_chunks {
            self.chunks_since_estimate = 0;
            self.estimate_key();
        }
    }

    fn estimate_key(&mut self) {
        if self.chroma_history.is_empty() {
            return;
        }

        // Accumulate the history window into a single chroma vector.
        let mut acc = [0.0f32; 12];
        for c in &self.chroma_history {
            for i in 0..12 {
                acc[i] += c[i];
            }
        }
        if !normalize_profile(&mut acc) {
            return;
        }

        let mut best_root = 0u8;
        let mut best_is_major = true;
        let mut best_corr = f32::NEG_INFINITY;

        for rot in 0..12usize {
            let corr_maj = pearson_rotated(&acc, &KS_MAJOR, rot);
            let corr_min = pearson_rotated(&acc, &KS_MINOR, rot);
            if corr_maj > best_corr {
                best_corr = corr_maj;
                best_root = rot as u8;
                best_is_major = true;
            }
            if corr_min > best_corr {
                best_corr = corr_min;
                best_root = rot as u8;
                best_is_major = false;
            }
        }

        self.key_root       = best_root;
        self.key_is_major   = best_is_major;
        self.key_confidence = best_corr;
    }

    pub fn key_root(&self)       -> u8   { self.key_root }
    pub fn key_is_major(&self)   -> bool { self.key_is_major }
    pub fn key_confidence(&self) -> f32  { self.key_confidence }
    pub fn chroma_peak(&self)    -> f32  { self.chroma_peak }
    pub fn chroma(&self) -> &[f32; 12]   { &self.chroma }
}

impl Default for KeyTracker {
    fn default() -> Self {
        Self::new(2048.0 / 48000.0)
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a chromagram whose content exactly matches key `rot` of `is_major`.
    /// When fed to `pearson_rotated(c, profile, rot)` this returns 1.0.
    fn make_chroma_for_key(rot: usize, is_major: bool) -> [f32; 12] {
        let profile = if is_major { &KS_MAJOR } else { &KS_MINOR };
        let mut c = [0.0f32; 12];
        for j in 0..12 {
            c[j] = profile[(j + 12 - rot) % 12];
        }
        c
    }

    #[test]
    fn pearson_self_correlation_is_one() {
        let r = pearson_rotated(&KS_MAJOR, &KS_MAJOR, 0);
        assert!((r - 1.0).abs() < 1e-5, "self-correlation must be 1.0, got {}", r);
    }

    #[test]
    fn pearson_profile_derived_chroma_matches_at_correct_rotation() {
        // F# major (rot=6): chroma built from KS_MAJOR rotated to F#
        let c = make_chroma_for_key(6, true);
        let r = pearson_rotated(&c, &KS_MAJOR, 6);
        assert!((r - 1.0).abs() < 1e-5, "F# major chroma vs KS_MAJOR rot=6 must be 1.0, got {}", r);
        // Must NOT match at rot=0 (C major)
        let r_c = pearson_rotated(&c, &KS_MAJOR, 0);
        assert!(r > r_c, "rot=6 must score higher than rot=0 for F# major input");
    }

    #[test]
    fn key_tracker_detects_c_major() {
        let mut kt = KeyTracker::new(0.04);
        let chroma = make_chroma_for_key(0, true);
        for _ in 0..200 {
            kt.process_chunk(chroma);
        }
        assert_eq!(kt.key_root(), 0, "expected root C (0), got {}", kt.key_root());
        assert!(kt.key_is_major(), "expected major mode");
        assert!(
            kt.key_confidence() > 0.9,
            "expected confidence > 0.9, got {}",
            kt.key_confidence()
        );
    }

    #[test]
    fn key_tracker_detects_a_minor() {
        let mut kt = KeyTracker::new(0.04);
        let chroma = make_chroma_for_key(9, false);
        for _ in 0..200 {
            kt.process_chunk(chroma);
        }
        assert_eq!(kt.key_root(), 9, "expected root A (9), got {}", kt.key_root());
        assert!(!kt.key_is_major(), "expected minor mode");
        assert!(
            kt.key_confidence() > 0.9,
            "expected confidence > 0.9, got {}",
            kt.key_confidence()
        );
    }

    #[test]
    fn key_tracker_detects_fsharp_major() {
        let mut kt = KeyTracker::new(0.04);
        let chroma = make_chroma_for_key(6, true);
        for _ in 0..200 {
            kt.process_chunk(chroma);
        }
        assert_eq!(kt.key_root(), 6, "expected root F# (6), got {}", kt.key_root());
        assert!(kt.key_is_major(), "expected major mode");
        assert!(
            kt.key_confidence() > 0.9,
            "expected confidence > 0.9, got {}",
            kt.key_confidence()
        );
    }

    #[test]
    fn key_tracker_silence_leaves_confidence_zero() {
        let mut kt = KeyTracker::new(0.04);
        for _ in 0..200 {
            kt.process_chunk([0.0; 12]);
        }
        assert_eq!(kt.key_confidence(), 0.0, "silence must leave confidence at 0.0");
    }

    #[test]
    fn c_major_scale_tones_outscores_all_other_keys() {
        // Equal-weight C major scale tones: C D E F G A B
        let c_scale: [f32; 12] = [1.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0];
        let r_c_maj = pearson_rotated(&c_scale, &KS_MAJOR, 0);
        // Must beat C minor and any rotation of F# (tritone away)
        let r_c_min = pearson_rotated(&c_scale, &KS_MINOR, 0);
        let r_fs_maj = pearson_rotated(&c_scale, &KS_MAJOR, 6);
        assert!(r_c_maj > r_c_min, "C scale should match major over minor");
        assert!(r_c_maj > r_fs_maj, "C scale should match C major over F# major");
        assert!(r_c_maj > 0.7, "C scale vs C major profile should be strongly positive, got {}", r_c_maj);
    }
}

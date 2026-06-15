//! Lyric corpus + audio-reactive fragment sampler.
//!
//! Slice 1: `LyricCorpus` — deterministic line/word accessor.
//! Slice 2: `LyricSampler` — spawns `ActiveFragment`s on beat onsets, each
//!          carrying a legibility envelope (abstract blob → crisp glyph → dissolve).

// ── Corpus ────────────────────────────────────────────────────────────────────

pub struct LyricCorpus {
    lines: Vec<String>,
    words: Vec<String>,
}

impl LyricCorpus {
    pub fn from_text(text: &str) -> Self {
        let lines: Vec<String> = text
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        let words: Vec<String> = lines
            .iter()
            .flat_map(|l| l.split_whitespace())
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();
        Self { lines, words }
    }

    /// i-th line, wrapping.
    pub fn fragment(&self, i: usize) -> &str {
        if self.lines.is_empty() { return ""; }
        &self.lines[i % self.lines.len()]
    }

    /// i-th individual word, wrapping.
    pub fn word(&self, i: usize) -> &str {
        if self.words.is_empty() { return ""; }
        &self.words[i % self.words.len()]
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

// ── PCG hash ─────────────────────────────────────────────────────────────────

fn pcg_hash(state: u32) -> u32 {
    let s = state.wrapping_mul(747796405u32).wrapping_add(2891336453u32);
    let w = ((s >> ((s >> 28).wrapping_add(4))) ^ s).wrapping_mul(277803737u32);
    (w >> 22) ^ w
}

fn pcg_f32(seed: u32) -> f32 {
    (pcg_hash(seed) >> 8) as f32 / (1u32 << 24) as f32
}

// ── Fragment ──────────────────────────────────────────────────────────────────

/// One active lyric fragment on screen.
pub struct ActiveFragment {
    pub text:       String,
    pub spawn_time: f32,
    pub lifetime:   f32,
    /// Horizontal center in NDC (−1..1).
    pub base_x:     f32,
    /// Baseline y in NDC (−1..1).
    pub base_y:     f32,
    /// Font size as fraction of screen height (e.g. 0.07 → 7 % of height in px).
    pub scale:      f32,
    pub color:      [f32; 4],
    /// Deterministic seed — fed to the warp shader for per-fragment variation.
    pub seed:       u32,
}

// ── Sampler ───────────────────────────────────────────────────────────────────

const BEAT_THRESHOLD: f32 = 0.45;
const MAX_ACTIVE:     usize = 4;

pub struct LyricSampler {
    active:          Vec<ActiveFragment>,
    next_idx:        u32,
    seed_counter:    u32,
    prev_beat_decay: f32,
}

impl LyricSampler {
    pub fn new() -> Self {
        Self {
            active:          Vec::new(),
            next_idx:        0,
            seed_counter:    0xDEAD_BEEF,
            prev_beat_decay: 0.0,
        }
    }

    /// Advance state one export frame.
    pub fn update(
        &mut self,
        frame_time:  f32,
        beat_decay:  f32,
        bands:       [f32; 8],
        _dt:         f32,
        corpus:      &LyricCorpus,
    ) {
        // Expire fragments whose lifetime has elapsed.
        self.active.retain(|f| frame_time - f.spawn_time < f.lifetime);

        // Spawn on rising edge of beat_decay crossing BEAT_THRESHOLD.
        let rising = self.prev_beat_decay < BEAT_THRESHOLD && beat_decay >= BEAT_THRESHOLD;
        self.prev_beat_decay = beat_decay;

        if rising && self.active.len() < MAX_ACTIVE && !corpus.is_empty() {
            let seed = self.seed_counter;
            self.seed_counter = seed.wrapping_add(0x9E37_79B9);

            // Every 3rd fragment is a single word; the rest are full lines.
            let text = if self.next_idx % 3 == 2 {
                let wi = pcg_hash(seed.wrapping_add(17)) as usize;
                corpus.word(wi).to_string()
            } else {
                corpus.fragment(self.next_idx as usize).to_string()
            };
            self.next_idx = self.next_idx.wrapping_add(1);

            // Deterministic scatter: position, size, lifetime.
            let cx    = pcg_f32(seed)                    * 1.4  - 0.7;   // [−0.7, 0.7]
            let by    = pcg_f32(seed.wrapping_add(1))   * 1.1  - 0.65;  // [−0.65, 0.45]
            let scale = 0.05 + pcg_f32(seed.wrapping_add(2)) * 0.07;    // [0.05, 0.12]
            let life  = 2.5  + pcg_f32(seed.wrapping_add(3)) * 3.5;     // [2.5, 6.0]

            // Tint by dominant band → soft hue shift.
            let dom = bands
                .iter()
                .cloned()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let hue = dom as f32 / 8.0;
            let (r, g, b) = hsv_to_rgb(hue, 0.40, 1.0);

            self.active.push(ActiveFragment {
                text,
                spawn_time: frame_time,
                lifetime:   life,
                base_x:     cx,
                base_y:     by,
                scale,
                color: [r, g, b, 0.88],
                seed,
            });

            log::debug!(
                "LyricSampler: spawned {:?} at ({:.2},{:.2}) scale={:.3} life={:.1}s",
                &self.active.last().unwrap().text, cx, by, scale, life
            );
        }
    }

    /// Iterate active fragments with their current legibility in [0, 1].
    pub fn fragments(
        &self,
        frame_time:  f32,
        beat_decay:  f32,
    ) -> impl Iterator<Item = (&ActiveFragment, f32)> {
        self.active.iter().map(move |f| {
            let leg = fragment_legibility(f, frame_time, beat_decay);
            (f, leg)
        })
    }

    pub fn is_empty(&self) -> bool { self.active.is_empty() }
}

// ── Legibility envelope ───────────────────────────────────────────────────────

fn fragment_legibility(frag: &ActiveFragment, frame_time: f32, beat_decay: f32) -> f32 {
    let t = ((frame_time - frag.spawn_time) / frag.lifetime).clamp(0.0, 1.0);

    // Ramp up 0→25%, hold 25→65%, ramp down 65→100% of lifetime.
    let base = if t < 0.25 {
        smoothstep01(t / 0.25)
    } else if t < 0.65 {
        1.0f32
    } else {
        1.0 - smoothstep01((t - 0.65) / 0.35)
    };

    // Beat sharpening: a strong hit momentarily pushes legibility toward 1.
    (base * (0.6 + 0.4 * beat_decay)).clamp(0.0, 1.0)
}

fn smoothstep01(x: f32) -> f32 {
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ── Colour helpers ────────────────────────────────────────────────────────────

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h6 = (h * 6.0).rem_euclid(6.0);
    let i  = h6.floor() as u32;
    let f  = h6 - i as f32;
    let p  = v * (1.0 - s);
    let q  = v * (1.0 - s * f);
    let t  = v * (1.0 - s * (1.0 - f));
    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

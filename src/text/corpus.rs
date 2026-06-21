//! Lyric corpus + audio-reactive fragment sampler.
//!
//! Slice 1: `LyricCorpus` — deterministic line/word accessor.
//! Slice 2: `LyricSampler` — spawns `ActiveFragment`s on beat onsets, each
//!          holding per-character `GlyphInstance` data with wild scale/rotation/
//!          flip variation drawn from the current harmony palette.

// ── Default corpus ──────────────────────────────────────────────────────────────

/// Built-in fallback corpus used when `ABSTRAKT_LYRIC_TEXT` is unset, at BOTH the
/// live and export load sites (they reference this same const so they can't drift).
///
/// Short, lowercase, atmospheric phrases — no punctuation needed (the font is an
/// a–z alien-script substitution). Many lines with several words each so both
/// `fragment()` (whole line) and `word()` (single word) vary widely.
pub const DEFAULT_CORPUS: &str = "\
slow tide of light
the room breathes without us
soft static under glass
we drift between the signals
amber dust on the lens
a pulse beneath the floor
nothing arrives and nothing leaves
warm noise folds inward
the horizon hums in violet
half remembered rooms
echoes thin as paper
a current pulls us under
gravity loosens its grip
quiet machines dreaming
the dark is full of motion
every shadow has a tone
we orbit a fading sun
memory leaks through the seams
glass towers melt to vapor
the sky exhales slowly
faint orbits of the heart
let the silence widen
all surfaces turn liquid
a low hymn of dust and air
the signal bends toward home
dissolving into the hum";

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

// ── Per-character variation ───────────────────────────────────────────────────

/// One glyph in an active fragment — carries its own transform and color.
pub struct GlyphInstance {
    pub glyph_char: char,
    /// Absolute NDC center position (anti-clip clamped at spawn time).
    pub abs_x:      f32,
    pub abs_y:      f32,
    /// Scale relative to the fragment's base_font_px (log-distributed [0.25, 3.0]).
    pub scale:      f32,
    /// Rotation in radians, full 2π.
    pub rotation:   f32,
    pub flip_x:     bool,
    pub flip_y:     bool,
    /// RGB from the current harmony palette.
    pub color:      [f32; 3],
    /// Per-glyph alpha [0.35, 1.0]; multiplied by the lifetime envelope at draw time.
    pub alpha:      f32,
    /// Per-glyph horizontal stretch [STRETCH_MIN, STRETCH_MAX] — letters elongate
    /// into strands under the directional smear.
    pub stretch_x:  f32,
}

// ── Fragment ──────────────────────────────────────────────────────────────────

/// One active lyric fragment — owns a set of individually varied glyphs.
pub struct ActiveFragment {
    pub spawn_time: f32,
    pub lifetime:   f32,
    /// Logical anchor (used for logging; glyph abs positions are pre-computed).
    pub base_x:     f32,
    pub base_y:     f32,
    /// Deterministic seed — fed to the warp shader for per-fragment variation.
    pub seed:       u32,
    pub glyphs:     Vec<GlyphInstance>,
}

// ── Sampler ───────────────────────────────────────────────────────────────────

/// Beat rising-edge that triggers a spawn surge. Lower = spawns on softer hits.
const BEAT_THRESHOLD: f32 = 0.30;
/// Max fragments alive at once. Higher = denser stream.
const MAX_ACTIVE:     usize = 8;
/// Maximum glyphs with scale > 2.0 across all active fragments at any moment.
const GIANT_CAP:      usize = 2;
/// Timer-spawn floor (seconds): a fragment spawns at least this often even with
/// no beats, so text keeps flowing through quiet passages — in addition to the
/// beat rising-edge surges.
const TIMER_SPAWN_INTERVAL: f32 = 0.5;

/// Per-glyph horizontal-stretch range. Each glyph elongates by a random factor
/// in [STRETCH_MIN, STRETCH_MAX] so letters read as strands under the smear.
const STRETCH_MIN: f32 = 1.3;
const STRETCH_MAX: f32 = 2.2;

pub struct LyricSampler {
    active:          Vec<ActiveFragment>,
    next_idx:        u32,
    seed_counter:    u32,
    prev_beat_decay: f32,
    last_spawn_time: f32,
}

impl LyricSampler {
    pub fn new() -> Self {
        Self {
            active:          Vec::new(),
            next_idx:        0,
            seed_counter:    0xDEAD_BEEF,
            prev_beat_decay: 0.0,
            last_spawn_time: -999.0,
        }
    }

    /// Advance state one frame.
    ///
    /// `swatches` — harmony palette from `color::palette_from_harmony`.
    /// Glyph positions scatter freely across [-1,1] with no edge clamp; the
    /// kaleido fold maps every position in the shape_post texture into petals.
    pub fn update(
        &mut self,
        frame_time: f32,
        beat_decay: f32,
        _bands:     [f32; 8],
        _dt:        f32,
        corpus:     &LyricCorpus,
        swatches:   &[[f32; 3]],
    ) {
        // Expire fragments whose lifetime has elapsed.
        self.active.retain(|f| frame_time - f.spawn_time < f.lifetime);

        // Spawn on a beat rising edge (surge on hits) OR when the timer floor
        // elapses (steady flow through quiet passages).
        let rising = self.prev_beat_decay < BEAT_THRESHOLD && beat_decay >= BEAT_THRESHOLD;
        let timer_spawn = frame_time - self.last_spawn_time >= TIMER_SPAWN_INTERVAL;
        self.prev_beat_decay = beat_decay;

        if (!rising && !timer_spawn) || self.active.len() >= MAX_ACTIVE || corpus.is_empty() {
            return;
        }

        let seed = self.seed_counter;
        self.seed_counter = seed.wrapping_add(0x9E37_79B9);

        // Every 3rd fragment is a single word; others are full lines.
        let text = if self.next_idx % 3 == 2 {
            let wi = pcg_hash(seed.wrapping_add(17)) as usize;
            corpus.word(wi).to_string()
        } else {
            corpus.fragment(self.next_idx as usize).to_string()
        };
        self.next_idx = self.next_idx.wrapping_add(1);

        let life     = 2.5 + pcg_f32(seed.wrapping_add(3)) * 3.5;   // [2.5, 6.0]
        let anchor_x = pcg_f32(seed.wrapping_add(10)) * 1.4 - 0.7;  // [-0.7, 0.7]
        let anchor_y = pcg_f32(seed.wrapping_add(11)) * 1.1 - 0.65; // [-0.65, 0.45]

        // Count giants already on screen to enforce GIANT_CAP across all fragments.
        let existing_giants: usize = self.active.iter()
            .flat_map(|f| f.glyphs.iter())
            .filter(|g| g.scale > 2.0)
            .count();
        let mut giant_budget = GIANT_CAP.saturating_sub(existing_giants);

        // Build per-character GlyphInstances.
        let chars: Vec<char> = text.chars().collect();
        let mut glyphs = Vec::with_capacity(chars.len());

        for (ci, &ch) in chars.iter().enumerate() {
            let gs = seed.wrapping_add(100 + ci as u32 * 7);

            // Log-distributed scale in [0.25, 3.0].
            let t         = pcg_f32(gs);
            let ln_min    = 0.25_f32.ln();
            let ln_max    = 3.0_f32.ln();
            let raw_scale = (ln_min + t * (ln_max - ln_min)).exp();

            let scale = if raw_scale > 2.0 {
                if giant_budget > 0 { giant_budget -= 1; raw_scale } else { 2.0 }
            } else {
                raw_scale
            };

            let rotation = pcg_f32(gs.wrapping_add(1)) * std::f32::consts::TAU;
            let flip_x   = pcg_f32(gs.wrapping_add(2)) < 0.35;
            let flip_y   = pcg_f32(gs.wrapping_add(3)) < 0.35;
            let alpha    = 0.35 + pcg_f32(gs.wrapping_add(4)) * 0.65; // [0.35, 1.0]

            // Color from harmony palette; fall back to white.
            let color = if swatches.is_empty() {
                [1.0f32, 1.0, 1.0]
            } else {
                let si = pcg_hash(gs.wrapping_add(5)) as usize % swatches.len();
                swatches[si]
            };

            // Scatter glyphs loosely around the fragment anchor.
            let scatter_x = (pcg_f32(gs.wrapping_add(6)) - 0.5) * 0.8; // [-0.4, 0.4]
            let scatter_y = (pcg_f32(gs.wrapping_add(7)) - 0.5) * 0.5; // [-0.25, 0.25]
            let abs_x = anchor_x + scatter_x;
            let abs_y = anchor_y + scatter_y;
            // No edge clamp — pre-fold, glyphs scatter freely across the full
            // shape_post texture.  The kaleido fold maps every position into petals.

            // Per-glyph horizontal stretch → strand-like elongation.
            let stretch_x = STRETCH_MIN + pcg_f32(gs.wrapping_add(8)) * (STRETCH_MAX - STRETCH_MIN);

            glyphs.push(GlyphInstance { glyph_char: ch, abs_x, abs_y, scale, rotation,
                                        flip_x, flip_y, color, alpha, stretch_x });
        }

        log::debug!(
            "LyricSampler: spawned {} glyphs at anchor ({:.2},{:.2}) life={:.1}s",
            glyphs.len(), anchor_x, anchor_y, life
        );

        self.last_spawn_time = frame_time;
        self.active.push(ActiveFragment {
            spawn_time: frame_time, lifetime: life,
            base_x: anchor_x, base_y: anchor_y,
            seed, glyphs,
        });
    }

    /// Iterate active fragments with their current legibility in [0, 1].
    pub fn fragments(
        &self,
        frame_time: f32,
        beat_decay: f32,
    ) -> impl Iterator<Item = (&ActiveFragment, f32)> {
        self.active.iter().map(move |f| {
            let leg = fragment_legibility(f, frame_time, beat_decay);
            (f, leg)
        })
    }

    pub fn is_empty(&self) -> bool { self.active.is_empty() }
}

// ── Lifetime envelope ───────────────────────────────────────────────────────

/// Fraction of lifetime spent fading IN from black (smoothstep).
const ENV_FADE_IN:  f32 = 0.20;
/// Fraction of lifetime spent fading OUT to black (smoothstep).
/// The middle (1 − IN − OUT = 0.50) holds at full brightness.
const ENV_FADE_OUT: f32 = 0.30;

/// Emerge-from-black → hold → dissolve-to-black brightness envelope in [0,1].
/// Multiplies glyph alpha only (hue/rainbow color is untouched), so glyphs
/// always materialise out of and melt back into darkness — never pop.
/// `beat_decay` is intentionally ignored: the envelope is purely time-based so
/// the emerge/dissolve always reaches true black at the ends.
fn fragment_legibility(frag: &ActiveFragment, frame_time: f32, _beat_decay: f32) -> f32 {
    let t = ((frame_time - frag.spawn_time) / frag.lifetime).clamp(0.0, 1.0);
    if t < ENV_FADE_IN {
        smoothstep01(t / ENV_FADE_IN)
    } else if t < 1.0 - ENV_FADE_OUT {
        1.0
    } else {
        1.0 - smoothstep01((t - (1.0 - ENV_FADE_OUT)) / ENV_FADE_OUT)
    }
}

fn smoothstep01(x: f32) -> f32 {
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;
use realfft::RealFftPlanner;

/// User-selectable routing target for per-band beat envelopes.
/// Superset of BeatBand: adds Combined (legacy max behavior) and Off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BeatRoute {
    Combined,
    Low,
    Mid,
    High,
    Broadband,
    Off,
}

impl Default for BeatRoute {
    fn default() -> Self { BeatRoute::Combined }
}

impl BeatRoute {
    pub fn name(self) -> &'static str {
        match self {
            BeatRoute::Combined  => "Combined",
            BeatRoute::Low       => "Low (kicks)",
            BeatRoute::Mid       => "Mid (snares)",
            BeatRoute::High      => "High (hats)",
            BeatRoute::Broadband => "Broadband",
            BeatRoute::Off       => "Off",
        }
    }

    pub fn next(self) -> Self {
        match self {
            BeatRoute::Combined  => BeatRoute::Low,
            BeatRoute::Low       => BeatRoute::Mid,
            BeatRoute::Mid       => BeatRoute::High,
            BeatRoute::High      => BeatRoute::Broadband,
            BeatRoute::Broadband => BeatRoute::Off,
            BeatRoute::Off       => BeatRoute::Combined,
        }
    }
}

/// Which part of the spectrum triggered a beat onset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeatBand {
    /// Sub-bass + bass (bands 0-1): kick drums, bass notes
    Low,
    /// Low-mid + mid + upper-mid (bands 2-4): snares, vocals
    Mid,
    /// Presence + brilliance + air (bands 5-7): hi-hats, cymbals
    High,
    /// Whole-spectrum onset — what the legacy detector used to emit
    Broadband,
}

#[derive(Debug, Clone, Copy)]
pub enum AudioEvent {
    Beat { strength: f32, band: BeatBand },
}

/// Attack/Hold/Release envelope for beat decay.
///
/// Replaces the legacy `beat_decay *= exp(-5*dt)` model. Each trigger:
///   * **Attack** (~20ms): rises to the trigger strength
///   * **Hold** (~50ms): stays at peak — gives visuals a flash to see
///   * **Release** (~180ms): exponential decay to 0
///
/// At 120 BPM the envelope reaches 0 for ~250ms between beats instead of
/// lingering near 0.08 — visuals see distinct pulses, not a constant glow.
#[derive(Debug, Clone)]
pub struct BeatEnvelope {
    value:                 f32,
    phase_seconds:         f32,
    last_trigger_strength: f32,
    pub attack_seconds:  f32,
    pub hold_seconds:    f32,
    pub release_seconds: f32,
}

impl BeatEnvelope {
    pub fn new() -> Self {
        Self {
            value:                 0.0,
            phase_seconds:         999.0,
            last_trigger_strength: 0.0,
            attack_seconds:  0.020,
            hold_seconds:    0.050,
            release_seconds: 0.180,
        }
    }

    /// Trigger a new beat. Ignores weaker triggers during attack/hold of a stronger one.
    pub fn trigger(&mut self, strength: f32) {
        let s = strength.clamp(0.0, 1.0);
        if self.phase_seconds < (self.attack_seconds + self.hold_seconds)
            && s < self.last_trigger_strength
        {
            return;
        }
        self.last_trigger_strength = s;
        self.phase_seconds = 0.0;
    }

    /// Advance by `dt` seconds and return the new value.
    pub fn update(&mut self, dt: f32) -> f32 {
        self.phase_seconds += dt;
        let p = self.phase_seconds;
        let s = self.last_trigger_strength;

        self.value = if p < self.attack_seconds {
            s * (p / self.attack_seconds.max(1e-6))
        } else if p < self.attack_seconds + self.hold_seconds {
            s
        } else {
            let rp = p - self.attack_seconds - self.hold_seconds;
            let tau = self.release_seconds / 3.0;
            s * (-rp / tau.max(1e-6)).exp()
        };

        if self.value < 0.001 { self.value = 0.0; }
        self.value
    }

    #[allow(dead_code)]
    pub fn value(&self) -> f32 { self.value }
}

impl Default for BeatEnvelope {
    fn default() -> Self { Self::new() }
}

/// Tracks recent inter-beat intervals and produces a cooldown duration
/// proportional to detected tempo. Uses median (not mean) to resist outliers.
#[derive(Debug, Clone)]
pub struct AdaptiveCooldown {
    last_beat:  Option<Instant>,
    history_ms: Vec<u64>,
    capacity:   usize,
    pub floor_ms: u64,
    pub ceil_ms:  u64,
    pub fraction: f32,
}

impl AdaptiveCooldown {
    pub fn new() -> Self {
        Self {
            last_beat:  None,
            history_ms: Vec::with_capacity(8),
            capacity:   8,
            floor_ms:   80,
            ceil_ms:    220,
            fraction:   0.40,
        }
    }

    pub fn can_fire(&self) -> bool {
        match self.last_beat {
            None    => true,
            Some(t) => t.elapsed().as_millis() as u64 >= self.current_cooldown_ms(),
        }
    }

    pub fn record_fire(&mut self) {
        let now = Instant::now();
        if let Some(prev) = self.last_beat {
            let interval = prev.elapsed().as_millis() as u64;
            if interval >= self.floor_ms && interval <= 2000 {
                if self.history_ms.len() >= self.capacity {
                    self.history_ms.remove(0);
                }
                self.history_ms.push(interval);
            }
        }
        self.last_beat = Some(now);
    }

    fn current_cooldown_ms(&self) -> u64 {
        if self.history_ms.is_empty() {
            return 120;
        }
        let mut sorted = self.history_ms.clone();
        sorted.sort_unstable();
        let median = sorted[sorted.len() / 2];
        let cooldown = (median as f32 * self.fraction) as u64;
        cooldown.clamp(self.floor_ms, self.ceil_ms)
    }
}

impl Default for AdaptiveCooldown {
    fn default() -> Self { Self::new() }
}

#[derive(Default)]
pub struct AudioState {
    pub bands:                [f32; 8],
    pub beat_decay:           f32,  // combined (max of all four) — legacy
    pub beat_decay_low:       f32,
    pub beat_decay_mid:       f32,
    pub beat_decay_high:      f32,
    pub beat_decay_broadband: f32,
    pub current_bpm:          Option<f32>,
    pub beat_phase:           f32,
    pub bpm_confidence:       f32,
}

/// Tracks tempo via autocorrelation of broadband flux history, then maintains
/// a phase-locked loop that produces a continuous beat_phase ∈ [0, 1).
#[derive(Debug, Clone)]
pub struct TempoTracker {
    flux_history:            Vec<f32>,
    history_size:            usize,
    chunk_seconds:           f32,
    locked_bpm:              Option<f32>,
    locked_confidence:       f32,
    chunks_since_estimate:   u32,
    estimate_period_chunks:  u32,
    pub phase:               f32,
    pll_gain:                f32,
    candidate_bpm:           Option<f32>,
    candidate_hits:          u32,
    candidate_hits_required: u32,
}

impl TempoTracker {
    const BPM_MIN: f32 =  60.0;
    const BPM_MAX: f32 = 180.0;
    const CONFIDENCE_THRESHOLD: f32 = 1.8;
    const HISTORY_SECONDS: f32 = 6.0;
    const ESTIMATE_PERIOD_SECONDS: f32 = 0.25;

    pub fn new(chunk_seconds: f32) -> Self {
        let history_size = (Self::HISTORY_SECONDS / chunk_seconds).ceil() as usize;
        let estimate_period_chunks =
            (Self::ESTIMATE_PERIOD_SECONDS / chunk_seconds).ceil() as u32;
        Self {
            flux_history: Vec::with_capacity(history_size),
            history_size,
            chunk_seconds,
            locked_bpm: None,
            locked_confidence: 0.0,
            chunks_since_estimate: 0,
            estimate_period_chunks: estimate_period_chunks.max(1),
            phase: 0.0,
            pll_gain: 0.10,
            candidate_bpm: None,
            candidate_hits: 0,
            candidate_hits_required: 3,
        }
    }

    pub fn process_chunk(&mut self, broadband_flux: f32) {
        self.flux_history.push(broadband_flux);
        if self.flux_history.len() > self.history_size {
            self.flux_history.remove(0);
        }
        if let Some(bpm) = self.locked_bpm {
            let beat_period_seconds = 60.0 / bpm;
            self.phase += self.chunk_seconds / beat_period_seconds;
            while self.phase >= 1.0 { self.phase -= 1.0; }
        }
        self.chunks_since_estimate += 1;
        if self.chunks_since_estimate >= self.estimate_period_chunks
            && self.flux_history.len() >= self.history_size
        {
            self.chunks_since_estimate = 0;
            self.estimate_tempo();
        }
    }

    pub fn on_broadband_onset(&mut self) {
        if self.locked_bpm.is_none() { return; }
        let error = if self.phase < 0.5 { -self.phase } else { 1.0 - self.phase };
        self.phase += error * self.pll_gain;
        while self.phase >= 1.0 { self.phase -= 1.0; }
        while self.phase <  0.0 { self.phase += 1.0; }
    }

    fn estimate_tempo(&mut self) {
        let bin_seconds = self.chunk_seconds;
        let lag_min = (60.0 / Self::BPM_MAX / bin_seconds).floor() as usize;
        let lag_max = (60.0 / Self::BPM_MIN / bin_seconds).ceil() as usize;
        if lag_max >= self.flux_history.len() || lag_min >= lag_max { return; }

        let mean: f32 = self.flux_history.iter().sum::<f32>()
            / self.flux_history.len() as f32;

        let mut best_lag: usize = lag_min;
        let mut best_score: f32 = 0.0;
        let mut score_sum: f32 = 0.0;
        let mut score_count: u32 = 0;

        for lag in lag_min..=lag_max {
            let n = self.flux_history.len() - lag;
            let mut score: f32 = 0.0;
            for i in 0..n {
                let a = self.flux_history[i] - mean;
                let b = self.flux_history[i + lag] - mean;
                score += a * b;
            }
            score /= n as f32;
            score_sum += score.max(0.0);
            score_count += 1;
            if score > best_score { best_score = score; best_lag = lag; }
        }

        if score_count == 0 || best_score <= 0.0 { return; }
        let mean_score = score_sum / score_count as f32;
        let confidence = if mean_score > 1e-9 { best_score / mean_score } else { 0.0 };

        if confidence < Self::CONFIDENCE_THRESHOLD {
            self.locked_confidence = confidence;
            return;
        }

        let bpm_candidate = 60.0 / (best_lag as f32 * bin_seconds);

        match self.locked_bpm {
            None => {
                self.locked_bpm = Some(bpm_candidate);
                self.locked_confidence = confidence;
                self.phase = 0.0;
                self.candidate_bpm = None;
                self.candidate_hits = 0;
            }
            Some(current) => {
                let drift = (bpm_candidate - current).abs() / current;
                if drift < 0.03 {
                    self.locked_confidence = confidence;
                    self.candidate_bpm = None;
                    self.candidate_hits = 0;
                } else {
                    match self.candidate_bpm {
                        Some(c) if (c - bpm_candidate).abs() / c < 0.05 => {
                            self.candidate_hits += 1;
                            if self.candidate_hits >= self.candidate_hits_required {
                                self.locked_bpm = Some(bpm_candidate);
                                self.locked_confidence = confidence;
                                self.candidate_bpm = None;
                                self.candidate_hits = 0;
                            }
                        }
                        _ => {
                            self.candidate_bpm = Some(bpm_candidate);
                            self.candidate_hits = 1;
                        }
                    }
                }
            }
        }
    }

    pub fn bpm(&self)        -> Option<f32> { self.locked_bpm }
    pub fn phase(&self)      -> f32         { self.phase }
    pub fn confidence(&self) -> f32         { self.locked_confidence }
}

impl Default for TempoTracker {
    fn default() -> Self { Self::new(2048.0 / 48000.0) }
}

/// Simple flux proxy derived from per-frame band energies.
///
/// Used by File-mode tempo detection where FFT spectral flux isn't computed.
/// Flux = positive-only change in total band energy between consecutive calls.
#[derive(Debug, Clone, Default)]
pub struct BandFlux {
    prev_total: f32,
}

impl BandFlux {
    pub fn new() -> Self { Self { prev_total: 0.0 } }

    pub fn reset(&mut self) { self.prev_total = 0.0; }

    pub fn from_bands(&mut self, bands: &[f32; 8]) -> f32 {
        let total: f32 = bands.iter().sum();
        let flux = (total - self.prev_total).max(0.0);
        self.prev_total = total;
        flux
    }
}

/// Mic noise filter: high-pass IIR at 80 Hz, RMS gate with hysteresis,
/// and per-band noise floor subtraction applied after FFT.
///
/// Designed to be cheap (one IIR coefficient, one RMS, 8 floors) and to
/// adapt over seconds, not milliseconds — so it tracks slow drift in
/// ambient noise without chasing musical content.
pub struct NoiseFilter {
    enabled: bool,

    // High-pass IIR state (one-pole, ~80 Hz at 48 kHz)
    hp_alpha: f32,
    hp_prev_in: f32,
    hp_prev_out: f32,

    // RMS gate state
    rms_smoothed: f32,
    gate_open: bool,
    gate_open_thresh: f32,
    gate_close_thresh: f32,
    gate_closed_gain: f32,

    // Per-band noise floor estimate (slow-moving minimum)
    band_floor: [f32; 8],
    floor_attack: f32,    // how fast floor rises (0..1, per FFT chunk)
    floor_release: f32,   // how fast floor falls when band is quiet
    floor_margin: f32,    // subtract floor * margin from each band
}

impl NoiseFilter {
    pub fn new(sample_rate: u32, enabled: bool) -> Self {
        // One-pole HPF: y[n] = a * (y[n-1] + x[n] - x[n-1])
        // a = exp(-2*pi*fc / fs); fc = 80 Hz
        let fc = 80.0_f32;
        let hp_alpha = (-2.0 * std::f32::consts::PI * fc / sample_rate as f32).exp();

        Self {
            enabled,
            hp_alpha,
            hp_prev_in: 0.0,
            hp_prev_out: 0.0,

            rms_smoothed: 0.0,
            gate_open: false,
            gate_open_thresh: 0.020,   // open when RMS > 0.020
            gate_close_thresh: 0.010,  // close when RMS < 0.010
            gate_closed_gain: 0.05,    // 5% bleed-through when closed

            band_floor: [0.0; 8],
            floor_attack: 0.002,       // rises slowly toward live energy
            floor_release: 0.0005,     // falls even slower
            floor_margin: 1.8,         // subtract 1.8x the estimated floor
        }
    }

    #[allow(dead_code)]
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            // Reset state so re-enabling doesn't carry stale floor estimates.
            self.hp_prev_in = 0.0;
            self.hp_prev_out = 0.0;
            self.rms_smoothed = 0.0;
            self.gate_open = false;
            self.band_floor = [0.0; 8];
        }
    }

    /// Apply HPF + RMS gate in-place on a mono sample buffer.
    /// Returns the post-gate RMS for diagnostics.
    pub fn process_samples(&mut self, samples: &mut [f32]) -> f32 {
        if !self.enabled || samples.is_empty() {
            return 0.0;
        }

        // 1) High-pass filter (one-pole IIR, removes sub-80Hz rumble/hum).
        for s in samples.iter_mut() {
            let x = *s;
            let y = self.hp_alpha * (self.hp_prev_out + x - self.hp_prev_in);
            self.hp_prev_in = x;
            self.hp_prev_out = y;
            *s = y;
        }

        // 2) Compute RMS of the post-HPF buffer.
        let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
        let rms = (sum_sq / samples.len() as f32).sqrt();
        // Fast attack, slow release — track signal level closely going up,
        // bleed off slowly so a brief silence between notes doesn't close the gate.
        let a = if rms > self.rms_smoothed { 0.5 } else { 0.05 };
        self.rms_smoothed = self.rms_smoothed * (1.0 - a) + rms * a;

        // 3) Hysteresis gate.
        if self.gate_open {
            if self.rms_smoothed < self.gate_close_thresh {
                self.gate_open = false;
            }
        } else if self.rms_smoothed > self.gate_open_thresh {
            self.gate_open = true;
        }

        // 4) Apply gate (multiplicative).
        if !self.gate_open {
            let g = self.gate_closed_gain;
            for s in samples.iter_mut() {
                *s *= g;
            }
        }

        self.rms_smoothed
    }

    /// Spectral subtraction on 8-band energies. Call AFTER your FFT/band split
    /// and BEFORE the per-band AGC, so the AGC normalises the cleaned signal.
    ///
    /// `gate_open` should be the most recent gate state — when the gate is
    /// closed, we update the floor estimate aggressively (treat as noise).
    /// When open, we update the floor only when a band is near or below it.
    pub fn process_bands(&mut self, bands: &mut [f32; 8]) {
        if !self.enabled {
            return;
        }

        for i in 0..8 {
            let live = bands[i];
            // Update floor estimate
            let target = if !self.gate_open || live < self.band_floor[i] * 1.5 {
                // Treat as noise — pull floor toward live energy.
                live
            } else {
                // Treat as signal — let floor decay slowly.
                self.band_floor[i] * 0.999
            };
            let rate = if target > self.band_floor[i] {
                self.floor_attack
            } else {
                self.floor_release
            };
            self.band_floor[i] = self.band_floor[i] * (1.0 - rate) + target * rate;

            // Spectral subtraction.
            let cleaned = (live - self.band_floor[i] * self.floor_margin).max(0.0);
            bands[i] = cleaned;
        }
    }

    #[allow(dead_code)]
    pub fn band_floor(&self) -> [f32; 8] {
        self.band_floor
    }

    #[allow(dead_code)]
    pub fn gate_open(&self) -> bool {
        self.gate_open
    }
}

pub struct AudioCapture {
    pub event_rx: Receiver<AudioEvent>,
    #[allow(dead_code)]
    pub state: Arc<Mutex<AudioState>>,
    _stream: cpal::Stream,
}

/// Android-parity band cutoffs (Hz): (lo_inclusive, hi_exclusive)
const BAND_CUTOFFS: [(f32, f32); 8] = [
    (60.0,    120.0),
    (120.0,   250.0),
    (250.0,   500.0),
    (500.0,  1000.0),
    (1000.0, 2000.0),
    (2000.0, 4000.0),
    (4000.0, 8000.0),
    (8000.0, 16000.0),
];

/// Half-wave spectral rectification: bins with im < 0 contribute zero magnitude.
/// Android uses this deliberately to select only positive-imaginary spectral components.
fn spectral_magnitudes(spectrum: &[realfft::num_complex::Complex<f32>]) -> Vec<f32> {
    spectrum
        .iter()
        .map(|c| if c.im >= 0.0 { c.norm() } else { 0.0 })
        .collect()
}

fn bands_from_magnitudes(mags: &[f32], sample_rate: u32, fft_size: usize) -> [f32; 8] {
    let bin_hz = sample_rate as f32 / fft_size as f32;
    let mut bands = [0.0f32; 8];
    for (i, &(lo, hi)) in BAND_CUTOFFS.iter().enumerate() {
        let lo_bin = (lo / bin_hz) as usize;
        let hi_bin = ((hi / bin_hz) as usize + 1).min(mags.len());
        if hi_bin > lo_bin {
            // TODO(future slice): replace fixed scaling with per-file
            // global-max normalization, per Android parity audit
            // section 2d offline-path normalization.
            bands[i] = (mags[lo_bin..hi_bin].iter().sum::<f32>()
                / (hi_bin - lo_bin) as f32
                * 0.05)
                .min(1.0);
        }
    }
    bands
}

impl AudioCapture {
    /// `prefer_monitor`: if true, prefer a "monitor" (loopback) input device; falls back to
    /// default input with a warning. If false, use default input directly (microphone).
    pub fn start(prefer_monitor: bool) -> Result<Self, String> {
        let host = cpal::default_host();

        let device = if prefer_monitor {
            let monitor = host
                .input_devices()
                .map_err(|e| format!("Failed to enumerate input devices: {}", e))?
                .find(|d| {
                    d.name()
                        .map(|n| n.to_lowercase().contains("monitor"))
                        .unwrap_or(false)
                });
            match monitor {
                Some(d) => d,
                None => {
                    log::warn!("No monitor/loopback device found — falling back to default input device");
                    host.default_input_device()
                        .ok_or_else(|| "No audio input device found".to_string())?
                }
            }
        } else {
            host.default_input_device()
                .ok_or_else(|| "No audio input device found".to_string())?
        };

        let device_name = device.name().unwrap_or_else(|_| "<unknown>".to_string());
        log::info!("Audio device: {}", device_name);

        let config = device
            .default_input_config()
            .map_err(|e| format!("Failed to get default input config: {}", e))?;

        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let sample_format = config.sample_format();

        log::info!(
            "Audio config: {} Hz, {} ch, {:?}",
            sample_rate,
            channels,
            sample_format
        );

        let (event_tx, event_rx) = bounded::<AudioEvent>(64);
        let state = Arc::new(Mutex::new(AudioState::default()));

        // Enable noise filter only when capturing from a microphone
        // (loopback / monitor sources don't need it).
        let noise_filter_enabled = !prefer_monitor;
        let mut analyzer = BeatAnalyzer::new(sample_rate, channels, noise_filter_enabled);
        log::info!(
            "Noise filter: {} (source: {})",
            if noise_filter_enabled { "ENABLED" } else { "disabled" },
            if prefer_monitor { "monitor/loopback" } else { "microphone" },
        );

        let stream = match sample_format {
            cpal::SampleFormat::F32 => {
                let analyzer_state = Arc::clone(&state);
                device.build_input_stream(
                    &config.into(),
                    move |data: &[f32], _| {
                        analyzer.process_f32(data, &event_tx, &analyzer_state);
                    },
                    move |err| log::error!("Audio stream error: {}", err),
                    None,
                )
            }
            cpal::SampleFormat::I16 => {
                let analyzer_state = Arc::clone(&state);
                device.build_input_stream(
                    &config.into(),
                    move |data: &[i16], _| {
                        let f32_buf: Vec<f32> = data
                            .iter()
                            .map(|s| *s as f32 / i16::MAX as f32)
                            .collect();
                        analyzer.process_f32(&f32_buf, &event_tx, &analyzer_state);
                    },
                    move |err| log::error!("Audio stream error: {}", err),
                    None,
                )
            }
            cpal::SampleFormat::I32 => {
                let analyzer_state = Arc::clone(&state);
                device.build_input_stream(
                    &config.into(),
                    move |data: &[i32], _| {
                        let f32_buf: Vec<f32> = data
                            .iter()
                            .map(|s| *s as f32 / i32::MAX as f32)
                            .collect();
                        analyzer.process_f32(&f32_buf, &event_tx, &analyzer_state);
                    },
                    move |err| log::error!("Audio stream error: {}", err),
                    None,
                )
            }
            other => return Err(format!("Unsupported audio sample format: {:?}", other)),
        }
        .map_err(|e| format!("Failed to build input stream: {}", e))?;

        stream
            .play()
            .map_err(|e| format!("Failed to start input stream: {}", e))?;
        log::info!("Audio capture started");

        Ok(Self {
            event_rx,
            state,
            _stream: stream,
        })
    }
}

struct BeatAnalyzer {
    sample_rate:        u32,
    channels:           usize,
    sample_buffer:      Vec<f32>,
    fft_size:           usize,
    planner:            RealFftPlanner<f32>,
    prev_spectrum:      Vec<f32>,
    flux_history_low:   VecDeque<f32>,
    flux_history_mid:   VecDeque<f32>,
    flux_history_high:  VecDeque<f32>,
    flux_history_bb:    VecDeque<f32>,
    flux_history_size:  usize,
    broadband_cooldown: AdaptiveCooldown,
    broadband_envelope: BeatEnvelope,
    low_cooldown:       AdaptiveCooldown,
    low_envelope:       BeatEnvelope,
    mid_cooldown:       AdaptiveCooldown,
    mid_envelope:       BeatEnvelope,
    high_cooldown:      AdaptiveCooldown,
    high_envelope:      BeatEnvelope,
    // Per-band AGC: tracks the running peak of each band energy.
    // Initialised to 0.0 so the first real signal immediately sets the scale.
    band_peak:          [f32; 8],
    agc_log_timer:      Instant,
    agc_logged:         bool,
    noise_filter:       NoiseFilter,
    tempo_tracker:      TempoTracker,
}

impl BeatAnalyzer {
    fn new(sample_rate: u32, channels: usize, noise_filter_enabled: bool) -> Self {
        let fft_size = 2048;
        let flux_history_size = 43;
        Self {
            sample_rate,
            channels,
            sample_buffer: Vec::with_capacity(fft_size * 2),
            fft_size,
            planner: RealFftPlanner::<f32>::new(),
            prev_spectrum: vec![0.0; fft_size / 2 + 1],
            flux_history_low:  VecDeque::with_capacity(flux_history_size + 1),
            flux_history_mid:  VecDeque::with_capacity(flux_history_size + 1),
            flux_history_high: VecDeque::with_capacity(flux_history_size + 1),
            flux_history_bb:   VecDeque::with_capacity(flux_history_size + 1),
            flux_history_size,
            broadband_cooldown: AdaptiveCooldown::new(),
            broadband_envelope: BeatEnvelope::new(),
            low_cooldown:       AdaptiveCooldown::new(),
            low_envelope:       BeatEnvelope::new(),
            mid_cooldown:       AdaptiveCooldown::new(),
            mid_envelope:       BeatEnvelope::new(),
            high_cooldown:      AdaptiveCooldown::new(),
            high_envelope:      BeatEnvelope::new(),
            band_peak: [0.0; 8],
            agc_log_timer: Instant::now(),
            agc_logged: false,
            noise_filter: NoiseFilter::new(sample_rate, noise_filter_enabled),
            tempo_tracker: TempoTracker::new(fft_size as f32 / sample_rate as f32),
        }
    }

    fn process_f32(
        &mut self,
        samples: &[f32],
        event_tx: &Sender<AudioEvent>,
        state: &Arc<Mutex<AudioState>>,
    ) {
        let chunk_count = samples.len() / self.channels;
        let start = self.sample_buffer.len();
        for i in 0..chunk_count {
            let mut mono = 0.0f32;
            for c in 0..self.channels {
                mono += samples[i * self.channels + c];
            }
            mono /= self.channels as f32;
            self.sample_buffer.push(mono);
        }

        // Apply HPF + RMS gate to the newly-pushed samples in place.
        // (No-op when noise filter is disabled.)
        let new_slice = &mut self.sample_buffer[start..];
        self.noise_filter.process_samples(new_slice);

        while self.sample_buffer.len() >= self.fft_size {
            let chunk: Vec<f32> = self.sample_buffer.drain(..self.fft_size).collect();
            self.process_fft_chunk(&chunk, event_tx, state);
        }

        if self.sample_buffer.len() > self.fft_size * 4 {
            let extra = self.sample_buffer.len() - self.fft_size;
            self.sample_buffer.drain(..extra);
        }
    }

    fn process_fft_chunk(
        &mut self,
        chunk: &[f32],
        event_tx: &Sender<AudioEvent>,
        state: &Arc<Mutex<AudioState>>,
    ) {
        let mut windowed: Vec<f32> = chunk
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let w = 0.5
                    - 0.5
                        * ((2.0 * std::f32::consts::PI * i as f32)
                            / (self.fft_size - 1) as f32)
                            .cos();
                s * w
            })
            .collect();

        let r2c = self.planner.plan_fft_forward(self.fft_size);
        let mut spectrum = r2c.make_output_vec();
        if r2c.process(&mut windowed, &mut spectrum).is_err() {
            return;
        }

        let magnitudes = spectral_magnitudes(&spectrum);

        // Per-band spectral flux: split by frequency range.
        let bin_hz  = self.sample_rate as f32 / self.fft_size as f32;
        let lo_low  = (60.0     / bin_hz) as usize;
        let hi_low  = (250.0    / bin_hz) as usize;
        let hi_mid  = (2000.0   / bin_hz) as usize;
        let hi_high = ((16000.0 / bin_hz) as usize).min(magnitudes.len());
        let n       = magnitudes.len().min(self.prev_spectrum.len());

        let flux_bb   = magnitudes[..n].iter().zip(&self.prev_spectrum[..n])
            .map(|(c, p)| (c - p).max(0.0)).sum::<f32>();
        let flux_low  = magnitudes[lo_low..hi_low.min(n)].iter()
            .zip(&self.prev_spectrum[lo_low..hi_low.min(n)])
            .map(|(c, p)| (c - p).max(0.0)).sum::<f32>();
        let flux_mid  = magnitudes[hi_low..hi_mid.min(n)].iter()
            .zip(&self.prev_spectrum[hi_low..hi_mid.min(n)])
            .map(|(c, p)| (c - p).max(0.0)).sum::<f32>();
        let flux_high = magnitudes[hi_mid..hi_high.min(n)].iter()
            .zip(&self.prev_spectrum[hi_mid..hi_high.min(n)])
            .map(|(c, p)| (c - p).max(0.0)).sum::<f32>();

        self.prev_spectrum = magnitudes.clone();

        let mut raw_bands = bands_from_magnitudes(&magnitudes, self.sample_rate, self.fft_size);

        // Per-band noise floor subtraction (no-op when filter disabled).
        // Done BEFORE AGC so AGC tracks the cleaned signal's peak.
        self.noise_filter.process_bands(&mut raw_bands);

        // Per-band AGC: track running peak, normalise output to [0, 1].
        // band_peak starts at 0.0 so the first real signal immediately sets the scale.
        let mut bands = [0.0f32; 8];
        for b in 0..8 {
            self.band_peak[b] = (self.band_peak[b] * 0.999).max(raw_bands[b]);
            bands[b] = if self.band_peak[b] > 1e-9 {
                (raw_bands[b] / self.band_peak[b]).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }

        if !self.agc_logged && self.agc_log_timer.elapsed().as_secs() >= 5 {
            log::debug!("AGC band_peak after 5s: {:?}", self.band_peak);
            self.agc_logged = true;
        }

        // Advance all envelopes by this chunk's dt; write combined decay to AudioState.
        let dt = self.fft_size as f32 / self.sample_rate as f32;
        let v_bb   = self.broadband_envelope.update(dt);
        let v_low  = self.low_envelope.update(dt);
        let v_mid  = self.mid_envelope.update(dt);
        let v_high = self.high_envelope.update(dt);
        let combined_decay = v_bb.max(v_low).max(v_mid).max(v_high);

        // Feed broadband flux to tempo estimator before onset detection.
        self.tempo_tracker.process_chunk(flux_bb);

        {
            let mut s = state.lock();
            s.bands                = bands;
            s.beat_decay           = combined_decay;
            s.beat_decay_low       = v_low;
            s.beat_decay_mid       = v_mid;
            s.beat_decay_high      = v_high;
            s.beat_decay_broadband = v_bb;
            s.current_bpm          = self.tempo_tracker.bpm();
            s.beat_phase           = self.tempo_tracker.phase();
            s.bpm_confidence       = self.tempo_tracker.confidence();
        }

        // Push flux values to per-band histories.
        let sz = self.flux_history_size;
        if self.flux_history_bb.len()   >= sz { self.flux_history_bb.pop_front(); }
        if self.flux_history_low.len()  >= sz { self.flux_history_low.pop_front(); }
        if self.flux_history_mid.len()  >= sz { self.flux_history_mid.pop_front(); }
        if self.flux_history_high.len() >= sz { self.flux_history_high.pop_front(); }
        self.flux_history_bb.push_back(flux_bb);
        self.flux_history_low.push_back(flux_low);
        self.flux_history_mid.push_back(flux_mid);
        self.flux_history_high.push_back(flux_high);

        // Per-band onset detection.
        // Broadband: capture envelope value before/after to detect fresh triggers for PLL.
        let bb_before = self.broadband_envelope.value();
        Self::detect_and_fire(flux_bb,   &self.flux_history_bb,   sz, 1.5, 0.50, &mut self.broadband_cooldown, &mut self.broadband_envelope, BeatBand::Broadband, event_tx);
        let bb_after  = self.broadband_envelope.value();
        if bb_after > bb_before + 0.2 {
            self.tempo_tracker.on_broadband_onset();
        }
        Self::detect_and_fire(flux_low,  &self.flux_history_low,  sz, 1.5, 0.30, &mut self.low_cooldown,       &mut self.low_envelope,       BeatBand::Low,       event_tx);
        Self::detect_and_fire(flux_mid,  &self.flux_history_mid,  sz, 1.5, 0.30, &mut self.mid_cooldown,       &mut self.mid_envelope,       BeatBand::Mid,       event_tx);
        Self::detect_and_fire(flux_high, &self.flux_history_high, sz, 1.5, 0.20, &mut self.high_cooldown,      &mut self.high_envelope,      BeatBand::High,      event_tx);
    }

    fn detect_and_fire(
        flux: f32,
        history: &VecDeque<f32>,
        history_size: usize,
        threshold_mult: f32,
        min_flux: f32,
        cooldown: &mut AdaptiveCooldown,
        envelope: &mut BeatEnvelope,
        band: BeatBand,
        event_tx: &Sender<AudioEvent>,
    ) {
        if history.len() < history_size { return; }
        let avg = history.iter().sum::<f32>() / history_size as f32;
        let threshold = avg * threshold_mult;
        if flux > threshold && flux > min_flux && cooldown.can_fire() {
            let strength = ((flux / threshold).min(3.0) / 3.0).clamp(0.0, 1.0);
            envelope.trigger(strength);
            cooldown.record_fire();
            let _ = event_tx.try_send(AudioEvent::Beat { strength, band });
            log::debug!(
                "BEAT {:?} strength={:.2} flux={:.3} thr={:.3}",
                band, strength, flux, threshold
            );
        }
    }
}

/// Compute 8-band energies from a mono PCM window using a 2048-pt FFT with Hann window
/// and half-wave spectral rectification. Returns normalized values in [0.0, 1.0].
/// Stateless — use OfflineAnalyzer when you need beat detection across frames.
pub fn compute_band_energies(mono: &[f32], sample_rate: u32) -> [f32; 8] {
    const FFT_SIZE: usize = 2048;
    if mono.is_empty() {
        return [0.0; 8];
    }
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(FFT_SIZE);
    let mut windowed: Vec<f32> = (0..FFT_SIZE)
        .map(|i| {
            let s = mono.get(i).copied().unwrap_or(0.0);
            let w = 0.5
                - 0.5
                    * ((2.0 * std::f32::consts::PI * i as f32) / (FFT_SIZE - 1) as f32).cos();
            s * w
        })
        .collect();
    let mut spectrum = r2c.make_output_vec();
    if r2c.process(&mut windowed, &mut spectrum).is_err() {
        return [0.0; 8];
    }
    let magnitudes = spectral_magnitudes(&spectrum);
    bands_from_magnitudes(&magnitudes, sample_rate, FFT_SIZE)
}

/// Per-export-frame analyzer: stateful spectral-flux beat detection + envelope-based beat decay.
pub struct OfflineAnalyzer {
    planner: RealFftPlanner<f32>,
    prev_spectrum: Vec<f32>,
    flux_history: VecDeque<f32>,
    cooldown_frames: u32,
    pub beat_decay: f32,
    envelope: BeatEnvelope,
}

impl OfflineAnalyzer {
    const FFT_SIZE: usize = 2048;
    const FLUX_HISTORY_SIZE: usize = 20;

    pub fn new() -> Self {
        Self {
            planner: RealFftPlanner::<f32>::new(),
            prev_spectrum: vec![0.0; Self::FFT_SIZE / 2 + 1],
            flux_history: VecDeque::with_capacity(Self::FLUX_HISTORY_SIZE + 1),
            cooldown_frames: 0,
            beat_decay: 0.0,
            envelope: BeatEnvelope::new(),
        }
    }

    /// Analyze one export frame. `dt` is the frame duration in seconds (e.g. 1.0 / fps as f32).
    /// Returns (8-band energies, beat_decay).
    pub fn analyze_frame(&mut self, mono: &[f32], sample_rate: u32, dt: f32) -> ([f32; 8], f32) {
        if mono.is_empty() {
            self.beat_decay = self.envelope.update(dt);
            return ([0.0; 8], self.beat_decay);
        }

        let r2c = self.planner.plan_fft_forward(Self::FFT_SIZE);
        let mut windowed: Vec<f32> = (0..Self::FFT_SIZE)
            .map(|i| {
                let s = mono.get(i).copied().unwrap_or(0.0);
                let w = 0.5
                    - 0.5
                        * ((2.0 * std::f32::consts::PI * i as f32)
                            / (Self::FFT_SIZE - 1) as f32)
                            .cos();
                s * w
            })
            .collect();

        let mut spectrum = r2c.make_output_vec();
        if r2c.process(&mut windowed, &mut spectrum).is_err() {
            self.beat_decay = self.envelope.update(dt);
            return ([0.0; 8], self.beat_decay);
        }

        let magnitudes = spectral_magnitudes(&spectrum);
        let bands = bands_from_magnitudes(&magnitudes, sample_rate, Self::FFT_SIZE);

        let n = self.prev_spectrum.len().min(magnitudes.len());
        let flux: f32 = magnitudes[..n]
            .iter()
            .zip(self.prev_spectrum.iter())
            .map(|(curr, prev)| (curr - prev).max(0.0))
            .sum();
        self.prev_spectrum[..n].copy_from_slice(&magnitudes[..n]);

        if self.flux_history.len() >= Self::FLUX_HISTORY_SIZE {
            self.flux_history.pop_front();
        }
        self.flux_history.push_back(flux);

        if self.flux_history.len() >= Self::FLUX_HISTORY_SIZE {
            let avg_flux =
                self.flux_history.iter().sum::<f32>() / Self::FLUX_HISTORY_SIZE as f32;
            if flux > avg_flux * 1.5 && flux > 0.01 && self.cooldown_frames == 0 {
                let strength = ((flux / (avg_flux * 1.5)).min(3.0) / 3.0).clamp(0.0, 1.0);
                self.envelope.trigger(strength);
                // 150ms cooldown — Android's 3 × 50ms windows mapped to per-frame count.
                self.cooldown_frames = (0.15 / dt).round() as u32;
            }
        }

        if self.cooldown_frames > 0 {
            self.cooldown_frames -= 1;
        }

        self.beat_decay = self.envelope.update(dt);
        (bands, self.beat_decay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_yields_zero_bands() {
        let bands = compute_band_energies(&vec![0.0f32; 2048], 44100);
        for (i, &b) in bands.iter().enumerate() {
            assert!(b < 1e-6, "band {} should be ~zero for silence, got {}", i, b);
        }
    }

    #[test]
    fn noise_gives_nonzero_bands() {
        // Xorshift32 — deterministic so the test is reproducible
        let mut state = 12345u32;
        let mono: Vec<f32> = (0..2048)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as i32) as f32 / i32::MAX as f32 * 0.5
            })
            .collect();
        let bands = compute_band_energies(&mono, 44100);
        assert!(bands[0] > 0.0, "band 0 should have energy for broadband noise");
        assert!(bands[3] > 0.0, "band 3 should have energy for broadband noise");
    }

    #[test]
    fn offline_analyzer_beat_decay_decays() {
        // With envelope-based decay: empty frames with no triggered beat keep decay at 0.0.
        let mut analyzer = OfflineAnalyzer::new();
        let dt = 1.0 / 24.0f32;
        let (_, decay) = analyzer.analyze_frame(&[], 44100, dt);
        assert!(
            decay < 1e-6,
            "beat_decay should stay near-zero with no triggered beat, got {}",
            decay
        );
    }

    #[test]
    fn beat_envelope_starts_at_zero() {
        let env = BeatEnvelope::new();
        assert!(env.value() < 1e-6, "fresh envelope should be zero, got {}", env.value());
    }

    #[test]
    fn beat_envelope_attack_rises_to_peak() {
        let mut env = BeatEnvelope::new();
        env.trigger(1.0);
        let half_attack = env.attack_seconds / 2.0;
        let v_half = env.update(half_attack);
        assert!(v_half > 0.1 && v_half < 0.9, "halfway through attack should be ~0.5, got {}", v_half);
        let v_end = env.update(half_attack);
        assert!(v_end > 0.9, "end of attack should be near 1.0, got {}", v_end);
    }

    #[test]
    fn beat_envelope_holds_then_releases() {
        let mut env = BeatEnvelope::new();
        env.trigger(1.0);
        let _ = env.update(env.attack_seconds + env.hold_seconds);
        assert!(env.value() > 0.9, "at end of hold should be near 1.0, got {}", env.value());
        let _ = env.update(env.release_seconds);
        assert!(env.value() < 0.5, "well into release should be < 0.5, got {}", env.value());
    }

    #[test]
    fn beat_envelope_retrigger_during_release_restarts() {
        let mut env = BeatEnvelope::new();
        env.trigger(1.0);
        let _ = env.update(env.attack_seconds + env.hold_seconds + 0.05);
        let v_release = env.value();
        assert!(v_release < 0.9, "should be in release, got {}", v_release);
        env.trigger(1.0);
        let v_restart = env.update(0.001);
        assert!(v_restart < 0.1, "retrigger should restart attack from ~0, got {}", v_restart);
    }

    #[test]
    fn adaptive_cooldown_starts_permissive() {
        let cooldown = AdaptiveCooldown::new();
        assert!(cooldown.can_fire(), "fresh cooldown should allow first fire");
    }

    #[test]
    fn adaptive_cooldown_blocks_immediate_refire() {
        let mut cooldown = AdaptiveCooldown::new();
        assert!(cooldown.can_fire());
        cooldown.record_fire();
        assert!(!cooldown.can_fire(), "should block immediate second fire");
    }

    #[test]
    fn adaptive_cooldown_tracks_inter_beat_interval() {
        let mut cooldown = AdaptiveCooldown::new();
        cooldown.record_fire();
        // After one fire with no previous, history stays empty — default 120ms cooldown applies.
        assert!(!cooldown.can_fire(), "should be in cooldown after first fire");
    }

    #[test]
    fn noise_filter_disabled_is_passthrough() {
        let mut nf = NoiseFilter::new(48000, false);
        let mut buf = vec![0.5_f32; 256];
        let original = buf.clone();
        nf.process_samples(&mut buf);
        assert_eq!(buf, original, "disabled filter must not modify samples");

        let mut bands = [0.3_f32; 8];
        nf.process_bands(&mut bands);
        for b in bands.iter() {
            assert!((b - 0.3).abs() < 1e-6, "disabled filter must not modify bands");
        }
    }

    #[test]
    fn noise_filter_gate_closes_on_silence() {
        let mut nf = NoiseFilter::new(48000, true);
        // Feed many chunks of near-silence.
        for _ in 0..20 {
            let mut buf = vec![0.0001_f32; 512];
            nf.process_samples(&mut buf);
        }
        assert!(!nf.gate_open(), "gate should be closed after sustained silence");
    }

    #[test]
    fn noise_filter_gate_opens_on_signal() {
        let mut nf = NoiseFilter::new(48000, true);
        // Feed chunks of a loud-ish 200 Hz tone (above the 80 Hz HPF cutoff).
        for chunk_idx in 0..20 {
            let buf: Vec<f32> = (0..512)
                .map(|i| {
                    let t = (chunk_idx * 512 + i) as f32 / 48000.0;
                    0.3 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
                })
                .collect();
            let mut buf = buf;
            nf.process_samples(&mut buf);
        }
        assert!(nf.gate_open(), "gate should be open after sustained 200 Hz tone");
    }

    #[test]
    fn noise_filter_subtracts_steady_floor() {
        let mut nf = NoiseFilter::new(48000, true);
        // Let the floor adapt to a steady "noise" level of 0.1 in band 0.
        for _ in 0..5000 {
            let mut bands = [0.0_f32; 8];
            bands[0] = 0.1;
            nf.process_bands(&mut bands);
        }
        // Now hit it with a louder signal — output should be reduced by ~floor*margin.
        let mut bands = [0.0_f32; 8];
        bands[0] = 0.5;
        nf.process_bands(&mut bands);
        assert!(
            bands[0] < 0.5,
            "band 0 should be attenuated after floor adaptation, got {}",
            bands[0]
        );
        assert!(
            bands[0] > 0.0,
            "band 0 should still be positive (signal well above floor)"
        );
    }

    #[test]
    fn tempo_tracker_reports_no_bpm_initially() {
        let t = TempoTracker::new(0.04);
        assert!(t.bpm().is_none());
        assert_eq!(t.phase(), 0.0);
    }

    #[test]
    fn tempo_tracker_locks_to_120bpm_synthetic_pulse() {
        let chunk_seconds = 0.04;
        let mut t = TempoTracker::new(chunk_seconds);
        let beats_chunks = 13_usize;
        let total_chunks = (7.5 / chunk_seconds) as usize;
        for i in 0..total_chunks {
            let flux = if i % beats_chunks == 0 { 10.0 } else { 0.1 };
            t.process_chunk(flux);
        }
        let bpm = t.bpm();
        assert!(bpm.is_some(), "tracker should have locked tempo by now");
        let detected = bpm.unwrap();
        assert!(detected > 100.0 && detected < 140.0,
            "expected BPM near 118, got {}", detected);
    }

    #[test]
    fn tempo_tracker_phase_advances_when_locked() {
        let chunk_seconds = 0.04;
        let mut t = TempoTracker::new(chunk_seconds);
        let total_chunks = (8.0 / chunk_seconds) as usize;
        for i in 0..total_chunks {
            let flux = if i % 13 == 0 { 10.0 } else { 0.1 };
            t.process_chunk(flux);
        }
        assert!(t.bpm().is_some());
        let phase_before = t.phase();
        t.process_chunk(0.1);
        let phase_after = t.phase();
        assert!(phase_after >= 0.0 && phase_after < 1.0);
        assert!(phase_after != phase_before
            || (phase_before > 0.99 && phase_after < 0.01),
            "phase should advance");
    }

    #[test]
    fn tempo_tracker_silence_keeps_no_bpm() {
        let mut t = TempoTracker::new(0.04);
        for _ in 0..200 {
            t.process_chunk(0.0);
        }
        assert!(t.bpm().is_none(), "silence should not produce a tempo lock");
    }

    #[test]
    fn tempo_tracker_onset_nudges_phase_when_locked() {
        let chunk_seconds = 0.04;
        let mut t = TempoTracker::new(chunk_seconds);
        for i in 0..200 {
            let flux = if i % 13 == 0 { 10.0 } else { 0.1 };
            t.process_chunk(flux);
        }
        if t.bpm().is_none() { return; }
        t.phase = 0.4;
        let before = t.phase;
        t.on_broadband_onset();
        let after = t.phase;
        assert!(after < before,
            "onset at phase=0.4 should pull phase down; before={}, after={}", before, after);
        assert!((before - after - 0.04).abs() < 0.01,
            "expected phase to drop by ~0.04, dropped by {}", before - after);
    }

    #[test]
    fn band_flux_starts_at_zero() {
        let mut bf = BandFlux::new();
        let bands = [0.5_f32; 8];
        let f1 = bf.from_bands(&bands);
        assert!((f1 - 4.0).abs() < 1e-4, "expected 4.0, got {}", f1);
        let f2 = bf.from_bands(&bands);
        assert!(f2 < 1e-4, "same bands should give flux≈0, got {}", f2);
    }

    #[test]
    fn band_flux_reports_positive_change_only() {
        let mut bf = BandFlux::new();
        let high = [1.0_f32; 8];
        let low  = [0.1_f32; 8];
        let _ = bf.from_bands(&high);
        let drop = bf.from_bands(&low);
        assert_eq!(drop, 0.0, "flux should clamp negative changes to 0");
        let rise = bf.from_bands(&high);
        assert!((rise - 7.2).abs() < 1e-3, "expected 7.2, got {}", rise);
    }

    #[test]
    fn band_flux_reset_clears_history() {
        let mut bf = BandFlux::new();
        let _ = bf.from_bands(&[0.5; 8]);
        bf.reset();
        let f = bf.from_bands(&[0.3; 8]);
        assert!((f - 2.4).abs() < 1e-3, "expected 2.4 after reset, got {}", f);
    }
}

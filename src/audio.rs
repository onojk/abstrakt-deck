use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;
use realfft::RealFftPlanner;

#[derive(Debug, Clone, Copy)]
pub enum AudioEvent {
    Beat(f32),
}

#[derive(Default)]
pub struct AudioState {
    pub bands: [f32; 8],
    pub beat_decay: f32,
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
    sample_rate: u32,
    channels: usize,
    sample_buffer: Vec<f32>,
    fft_size: usize,
    planner: RealFftPlanner<f32>,
    flux_history: Vec<f32>,
    flux_history_size: usize,
    prev_spectrum: Vec<f32>,
    last_beat: Option<Instant>,
    beat_cooldown_ms: u64,
    rms_smoothed: f32,
    beat_decay: f32,
    // Per-band AGC: tracks the running peak of each band energy.
    // Initialised to 0.0 so the first real signal immediately sets the scale.
    band_peak: [f32; 8],
    agc_log_timer: Instant,
    agc_logged: bool,
    noise_filter: NoiseFilter,
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
            flux_history: Vec::with_capacity(flux_history_size),
            flux_history_size,
            prev_spectrum: vec![0.0; fft_size / 2 + 1],
            last_beat: None,
            beat_cooldown_ms: 120,
            rms_smoothed: 0.0,
            beat_decay: 0.0,
            band_peak: [0.0; 8],
            agc_log_timer: Instant::now(),
            agc_logged: false,
            noise_filter: NoiseFilter::new(sample_rate, noise_filter_enabled),
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
        let sum_sq: f32 = chunk.iter().map(|s| s * s).sum();
        let rms = (sum_sq / chunk.len() as f32).sqrt();
        self.rms_smoothed = self.rms_smoothed * 0.7 + rms * 0.3;

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

        let flux: f32 = magnitudes
            .iter()
            .zip(self.prev_spectrum.iter())
            .map(|(curr, prev)| (curr - prev).max(0.0))
            .sum();
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

        // Continuous exponential decay per chunk (dt = chunk duration in seconds)
        let dt = self.fft_size as f32 / self.sample_rate as f32;
        self.beat_decay *= (-5.0 * dt).exp();

        {
            let mut s = state.lock();
            s.bands = bands;
            s.beat_decay = self.beat_decay;
        }

        self.flux_history.push(flux);
        if self.flux_history.len() > self.flux_history_size {
            self.flux_history.remove(0);
        }
        if self.flux_history.len() < self.flux_history_size {
            return;
        }

        let avg_flux: f32 =
            self.flux_history.iter().sum::<f32>() / self.flux_history_size as f32;
        let threshold = avg_flux * 1.5;

        if let Some(last) = self.last_beat {
            if last.elapsed().as_millis() < self.beat_cooldown_ms as u128 {
                return;
            }
        }

        if flux > threshold && flux > 0.5 {
            let strength = ((flux / threshold).min(3.0) / 3.0).clamp(0.0, 1.0);
            self.beat_decay = self.beat_decay.max(1.0);
            self.last_beat = Some(Instant::now());
            let _ = event_tx.try_send(AudioEvent::Beat(strength));
            log::debug!(
                "BEAT strength={:.2} flux={:.3} threshold={:.3}",
                strength,
                flux,
                threshold
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

/// Per-export-frame analyzer: stateful spectral-flux beat detection + continuous beat decay.
pub struct OfflineAnalyzer {
    planner: RealFftPlanner<f32>,
    prev_spectrum: Vec<f32>,
    flux_history: VecDeque<f32>,
    cooldown_frames: u32,
    pub beat_decay: f32,
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
        }
    }

    /// Analyze one export frame. `dt` is the frame duration in seconds (e.g. 1.0 / fps as f32).
    /// Returns (8-band energies, beat_decay).
    pub fn analyze_frame(&mut self, mono: &[f32], sample_rate: u32, dt: f32) -> ([f32; 8], f32) {
        if mono.is_empty() {
            self.beat_decay *= (-5.0 * dt).exp();
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
            self.beat_decay *= (-5.0 * dt).exp();
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
                self.beat_decay = 1.0;
                // 150ms cooldown — Android's 3 × 50ms windows mapped to per-frame count.
                self.cooldown_frames = (0.15 / dt).round() as u32;
            }
        }

        if self.cooldown_frames > 0 {
            self.cooldown_frames -= 1;
        }

        self.beat_decay *= (-5.0 * dt).exp();
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
        let mut analyzer = OfflineAnalyzer::new();
        analyzer.beat_decay = 1.0;
        let dt = 1.0 / 24.0f32;
        let (_, decay) = analyzer.analyze_frame(&[], 44100, dt);
        let expected = (-5.0_f32 * dt).exp();
        assert!(
            (decay - expected).abs() < 1e-5,
            "beat_decay should decay by exp(-5*dt), got {} expected {}",
            decay,
            expected
        );
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
}

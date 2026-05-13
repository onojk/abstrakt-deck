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
    pub fn start() -> Result<Self, String> {
        let host = cpal::default_host();

        let device = host
            .input_devices()
            .map_err(|e| format!("Failed to enumerate input devices: {}", e))?
            .find(|d| {
                d.name()
                    .map(|n| n.to_lowercase().contains("monitor"))
                    .unwrap_or(false)
            })
            .or_else(|| host.default_input_device())
            .ok_or_else(|| "No audio input device found".to_string())?;

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

        let mut analyzer = BeatAnalyzer::new(sample_rate, channels);

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
}

impl BeatAnalyzer {
    fn new(sample_rate: u32, channels: usize) -> Self {
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
        }
    }

    fn process_f32(
        &mut self,
        samples: &[f32],
        event_tx: &Sender<AudioEvent>,
        state: &Arc<Mutex<AudioState>>,
    ) {
        let chunk_count = samples.len() / self.channels;
        for i in 0..chunk_count {
            let mut mono = 0.0f32;
            for c in 0..self.channels {
                mono += samples[i * self.channels + c];
            }
            mono /= self.channels as f32;
            self.sample_buffer.push(mono);
        }

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

        let bands = bands_from_magnitudes(&magnitudes, self.sample_rate, self.fft_size);

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

        if flux > threshold && flux > 0.01 {
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
}

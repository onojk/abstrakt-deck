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
    pub rms: f32,
    pub bass_energy: f32,
    pub mid_energy: f32,
}

pub struct AudioCapture {
    pub event_rx: Receiver<AudioEvent>,
    #[allow(dead_code)]
    pub state: Arc<Mutex<AudioState>>,
    _stream: cpal::Stream,
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
}

impl BeatAnalyzer {
    fn new(sample_rate: u32, channels: usize) -> Self {
        let fft_size = 1024;
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

        let magnitudes: Vec<f32> = spectrum.iter().map(|c| c.norm()).collect();

        let flux: f32 = magnitudes
            .iter()
            .zip(self.prev_spectrum.iter())
            .map(|(curr, prev)| (curr - prev).max(0.0))
            .sum();
        self.prev_spectrum = magnitudes.clone();

        let bin_hz = self.sample_rate as f32 / self.fft_size as f32;

        let bass_lo = (60.0 / bin_hz) as usize;
        let bass_hi = ((200.0 / bin_hz) as usize).min(magnitudes.len());
        let bass_energy: f32 = if bass_hi > bass_lo {
            magnitudes[bass_lo..bass_hi].iter().sum::<f32>() / (bass_hi - bass_lo) as f32
        } else {
            0.0
        };

        let mid_lo = bass_hi;
        let mid_hi = ((2000.0 / bin_hz) as usize).min(magnitudes.len());
        let mid_energy: f32 = if mid_hi > mid_lo {
            magnitudes[mid_lo..mid_hi].iter().sum::<f32>() / (mid_hi - mid_lo) as f32
        } else {
            0.0
        };

        {
            let mut s = state.lock();
            s.rms = self.rms_smoothed.min(1.0);
            s.bass_energy = (bass_energy * 0.05).min(1.0);
            s.mid_energy  = (mid_energy  * 0.05).min(1.0);
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

/// Compute bass (60–200 Hz) and mid (200–2000 Hz) energies from a mono PCM window.
/// Returns the same normalized values the mic analyzer produces so both sources
/// feed identical ranges into update_bass_zoom / update_modes.
pub fn compute_band_energies(mono: &[f32], sample_rate: u32) -> (f32, f32) {
    const FFT_SIZE: usize = 1024;
    if mono.is_empty() {
        return (0.0, 0.0);
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
        return (0.0, 0.0);
    }
    let magnitudes: Vec<f32> = spectrum.iter().map(|c| c.norm()).collect();
    let bin_hz = sample_rate as f32 / FFT_SIZE as f32;
    let bass_lo = (60.0 / bin_hz) as usize;
    let bass_hi = ((200.0 / bin_hz) as usize).min(magnitudes.len());
    let bass = if bass_hi > bass_lo {
        magnitudes[bass_lo..bass_hi].iter().sum::<f32>() / (bass_hi - bass_lo) as f32
    } else {
        0.0
    };
    let mid_lo = bass_hi;
    let mid_hi = ((2000.0 / bin_hz) as usize).min(magnitudes.len());
    let mid = if mid_hi > mid_lo {
        magnitudes[mid_lo..mid_hi].iter().sum::<f32>() / (mid_hi - mid_lo) as f32
    } else {
        0.0
    };
    ((bass * 0.05).min(1.0), (mid * 0.05).min(1.0))
}

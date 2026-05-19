// Audio-reactive cell influencer. Reads AudioSnapshot via the new
// step_with_audio trait method. Phase 4a: scaffolding only; logs once
// per second to prove audio data is flowing. Phase 4b implements the
// actual cell-state modulation.

use crate::cell::CellGrid;
use crate::influencer::{AudioSnapshot, Influencer};

pub struct AudioCellsPlaceholder {
    debug_timer: f32,
}

impl AudioCellsPlaceholder {
    pub fn new() -> Self {
        Self { debug_timer: 0.0 }
    }
}

impl Default for AudioCellsPlaceholder {
    fn default() -> Self { Self::new() }
}

impl Influencer for AudioCellsPlaceholder {
    // No-op for the audio-less path; nothing meaningful to do without
    // audio data.
    fn step(&mut self, _grid: &mut CellGrid, _dt: f32) {}

    fn step_with_audio(
        &mut self,
        _grid: &mut CellGrid,
        audio: &AudioSnapshot,
        dt:    f32,
    ) {
        self.debug_timer += dt;
        if self.debug_timer > 1.0 {
            self.debug_timer = 0.0;
            log::info!(
                "[myocyte/audio] bands=[{:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2}] \
                 beat_low={:.2} beat_mid={:.2} beat_high={:.2} bpm={:?}",
                audio.bands[0], audio.bands[1], audio.bands[2], audio.bands[3],
                audio.bands[4], audio.bands[5], audio.bands[6], audio.bands[7],
                audio.beat_decay_low, audio.beat_decay_mid, audio.beat_decay_high,
                audio.bpm,
            );
        }
    }
}

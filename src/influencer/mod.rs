// Ported from myocyte. Trait for procedural drivers of cell parameters.
// influencer.rs — trait for procedural drivers of cell parameters.
//
// An Influencer runs a simulation step and writes results back into the
// CellGrid. Deck ships with NoOpInfluencer as a placeholder; phase 3+
// will wire in GrayScott and Authored influencers.

pub mod audio_cells;
pub mod audio_wheel;
pub mod authored;
pub mod gray_scott;
pub mod prime_activation;

use crate::cell::CellGrid;

/// Per-frame audio analysis values, sourced from deck's GpuState fields
/// (which are themselves snapshotted from AudioCapture once per frame).
/// Passed by reference into Influencer::step_with_audio so influencers
/// can react to audio without holding shared state.
///
/// Field choices match what GpuState already exposes at render time —
/// no new analysis, just a structured view.
#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)]
pub struct AudioSnapshot {
    /// 8 exponentially-smoothed band energies.
    /// Index layout (Hz ranges per Task 0 findings):
    ///   0: 60-120, 1: 120-250, 2: 250-500, 3: 500-1000,
    ///   4: 1-2k,   5-7: 2-16k
    pub bands: [f32; 8],

    /// Decaying beat envelopes, one per coarse range.
    /// Spike on onset, decay over ~0.3s. Treat > ~0.3 as "beat active".
    pub beat_decay_low:       f32,
    pub beat_decay_mid:       f32,
    pub beat_decay_high:      f32,
    pub beat_decay_broadband: f32,

    /// Max of the four beat_decay_* values; useful as a single "any beat" signal.
    pub beat_decay_max: f32,

    /// Phase within the current detected beat cycle, [0, 1).
    pub beat_phase: f32,

    /// Detected BPM if confident, None otherwise. Confidence in [0, 1].
    pub bpm:            Option<f32>,
    pub bpm_confidence: f32,
}

pub trait Influencer {
    /// Update cells from internal state only (no audio).
    /// Used by RD systems, authored shapes, etc.
    fn step(&mut self, grid: &mut CellGrid, dt: f32);

    /// Update cells with access to current audio analysis.
    /// Default impl delegates to step() — non-audio influencers ignore audio.
    /// Audio-reactive influencers override this and may ignore step().
    fn step_with_audio(
        &mut self,
        grid:  &mut CellGrid,
        audio: &AudioSnapshot,
        dt:    f32,
    ) {
        let _ = audio;
        self.step(grid, dt);
    }
}

/// Does nothing — used as the default influencer until a real one is selected.
pub struct NoOpInfluencer;

impl Influencer for NoOpInfluencer {
    fn step(&mut self, _grid: &mut CellGrid, _dt: f32) {}
}

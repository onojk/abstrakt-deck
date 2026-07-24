/// Debounced collapse detector.
///
/// Feed one `FrameStats` per frame via `observe()`.  The detector accumulates
/// a run of degenerate frames; when that run length reaches `frames_threshold`
/// it fires a `Trigger` **exactly once** per continuous collapse (subsequent
/// calls during the same collapse return `None`).  A single non-degenerate
/// frame resets the run so the detector can fire again on the next collapse.
///
/// Instantiate one `DegenerateDetector` per panel (including the single-panel
/// live path).  It holds no wgpu state and no heap allocation — clone freely.

use crate::frame_metric::{Degeneracy, FrameStats};

/// Caller-visible firing record returned by `observe()`.
#[derive(Debug, Clone, Copy)]
pub struct Trigger {
    /// Dominant degeneracy direction over the triggering run.
    pub reason: Degeneracy,
    /// Luminance stats from the frame that crossed the threshold.
    pub mean_l: f32,
    pub degenerate_frac: f32,
}

/// Debounce window: frames that must be continuously degenerate before a
/// Trigger fires.  Derived from FPS by the caller:
///   `(fps as f32 * THRESHOLD_SECS).round().max(1) as u32`
pub const THRESHOLD_SECS: f32 = 0.75;

/// Per-panel detector instance.  Cheap to clone; no allocations.
#[derive(Debug, Clone)]
pub struct DegenerateDetector {
    /// How many consecutive degenerate frames must accumulate before firing.
    frames_threshold: u32,
    run_len: u32,
    black_run: u32,
    white_run: u32,
    /// Prevents re-firing during the same unbroken collapse.
    fired_this_run: bool,
}

impl DegenerateDetector {
    /// `frames_threshold`: caller computes `(fps as f32 * THRESHOLD_SECS).round().max(1) as u32`.
    pub fn new(frames_threshold: u32) -> Self {
        Self {
            frames_threshold: frames_threshold.max(1),
            run_len: 0,
            black_run: 0,
            white_run: 0,
            fired_this_run: false,
        }
    }

    /// Convenience constructor: derive threshold from framerate.
    pub fn for_fps(fps: u32) -> Self {
        let threshold = (fps as f32 * THRESHOLD_SECS).round().max(1.0) as u32;
        Self::new(threshold)
    }

    /// Feed one frame's stats.
    ///
    /// Returns `Some(Trigger)` the first time a collapse run reaches
    /// `frames_threshold`.  Returns `None` otherwise (including during the
    /// remainder of the same unbroken collapse).
    pub fn observe(&mut self, stats: FrameStats) -> Option<Trigger> {
        match stats.degeneracy() {
            Degeneracy::None => {
                self.reset();
                None
            }
            d => {
                self.run_len += 1;
                match d {
                    Degeneracy::Black => self.black_run += 1,
                    Degeneracy::White => self.white_run += 1,
                    Degeneracy::None  => {}
                }
                if self.run_len >= self.frames_threshold && !self.fired_this_run {
                    self.fired_this_run = true;
                    let reason = if self.black_run >= self.white_run {
                        Degeneracy::Black
                    } else {
                        Degeneracy::White
                    };
                    Some(Trigger {
                        reason,
                        mean_l: stats.mean_l,
                        degenerate_frac: stats.degenerate_frac,
                    })
                } else {
                    None
                }
            }
        }
    }

    /// Reset the run state (called automatically on a non-degenerate frame;
    /// callers may also call this explicitly, e.g. after a param swap).
    pub fn reset(&mut self) {
        self.run_len = 0;
        self.black_run = 0;
        self.white_run = 0;
        self.fired_this_run = false;
    }

    /// Current unbroken run length (useful for logging / progress indicators).
    pub fn run_len(&self) -> u32 {
        self.run_len
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_metric::FrameStats;

    fn black() -> FrameStats {
        FrameStats { mean_l: 0.0, degenerate_frac: 1.0 }
    }
    fn white() -> FrameStats {
        FrameStats { mean_l: 1.0, degenerate_frac: 1.0 }
    }
    fn ok() -> FrameStats {
        FrameStats { mean_l: 0.5, degenerate_frac: 0.0 }
    }

    #[test]
    fn fires_after_threshold() {
        let mut d = DegenerateDetector::new(3);
        assert!(d.observe(black()).is_none(), "frame 1 — too early");
        assert!(d.observe(black()).is_none(), "frame 2 — too early");
        let t = d.observe(black());
        assert!(t.is_some(), "frame 3 — should fire");
        assert_eq!(t.unwrap().reason, Degeneracy::Black);
    }

    #[test]
    fn does_not_refire_mid_run() {
        let mut d = DegenerateDetector::new(3);
        d.observe(black());
        d.observe(black());
        d.observe(black()); // fires
        assert!(d.observe(black()).is_none(), "must not re-fire during same run");
        assert!(d.observe(black()).is_none(), "must not re-fire during same run");
    }

    #[test]
    fn good_frame_resets_and_allows_refire() {
        let mut d = DegenerateDetector::new(3);
        d.observe(black());
        d.observe(black());
        d.observe(black()); // fires
        d.observe(ok());    // reset
        d.observe(black());
        d.observe(black());
        let t = d.observe(black());
        assert!(t.is_some(), "should fire again after reset");
    }

    #[test]
    fn partial_run_then_reset_does_not_fire() {
        let mut d = DegenerateDetector::new(5);
        d.observe(black());
        d.observe(black());
        d.observe(ok()); // only 2 frames; reset
        assert_eq!(d.run_len(), 0);
    }

    #[test]
    fn dominant_reason_is_majority() {
        let mut d = DegenerateDetector::new(4);
        d.observe(white());
        d.observe(white());
        d.observe(black());
        let t = d.observe(white()); // 3 white vs 1 black → WHITE
        assert_eq!(t.unwrap().reason, Degeneracy::White);
    }

    #[test]
    fn for_fps_threshold_makes_sense() {
        // At 60fps, threshold should be ~45 frames.
        let d = DegenerateDetector::for_fps(60);
        assert!(d.frames_threshold >= 40 && d.frames_threshold <= 50,
            "expected ~45, got {}", d.frames_threshold);
    }

    #[test]
    fn threshold_one_fires_immediately() {
        let mut d = DegenerateDetector::new(1);
        assert!(d.observe(black()).is_some());
    }
}

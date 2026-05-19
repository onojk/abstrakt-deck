// Ported from myocyte. Trait for procedural drivers of cell parameters.
// influencer.rs — trait for procedural drivers of cell parameters.
//
// An Influencer runs a simulation step and writes results back into the
// CellGrid. Deck ships with NoOpInfluencer as a placeholder; phase 3+
// will wire in GrayScott and Authored influencers.

use crate::cell::CellGrid;

pub trait Influencer {
    /// Advance the influencer's internal state by `dt` seconds (real time),
    /// then write derived values into `grid`.
    fn step(&mut self, grid: &mut CellGrid, dt: f32);
}

/// Does nothing — used as the default influencer until a real one is selected.
pub struct NoOpInfluencer;

impl Influencer for NoOpInfluencer {
    fn step(&mut self, _grid: &mut CellGrid, _dt: f32) {}
}

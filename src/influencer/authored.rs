// Ported from myocyte. Authored shapes (Sphere, Shell, LetterA) that write cell state each frame to form fixed geometric shapes.
#![allow(dead_code)]
//
// Unlike Gray-Scott, these shapes don't evolve — they're computed once on the
// first step() call and then left alone. Serves as a proof of concept that the
// cell grid can represent intentional content, not just emergent simulation.
//
// Opacity mapping:
//   "on"  cells → ON_OPACITY (retains CP3 color variation set by activate_varied_cells)
//   "off" cells → 0.0        (truly invisible; filtered at projection step)

use crate::cell::CellGrid;
use crate::influencer::Influencer;

pub enum AuthoredShape {
    Sphere,
    Shell,
    LetterA,
}

const ON_OPACITY:    f32 = 0.90;

// Sphere outer radius as a fraction of the grid's half-extent.
const OUTER_R_FRAC:  f32 = 0.40;
// Shell inner radius as a fraction of outer R (shell thickness = (1-0.70)*outer_R).
const INNER_R_FRAC:  f32 = 0.70;

pub struct Authored {
    shape:       AuthoredShape,
    initialized: bool,
}

impl Authored {
    pub fn new(shape: AuthoredShape) -> Self {
        Self { shape, initialized: false }
    }
}

impl Influencer for Authored {
    fn step(&mut self, grid: &mut CellGrid, _dt: f32) {
        if self.initialized { return; }
        self.initialized = true;

        let dims    = grid.dims;
        let spacing = grid.spacing;

        // Half-extent: distance from origin to the last cell center along one axis.
        let half_ext = (dims[0] - 1) as f32 * 0.5 * spacing;
        let outer_r  = OUTER_R_FRAC * half_ext;
        let inner_r  = INNER_R_FRAC * outer_r;
        // Compare squared distances to avoid sqrt in the inner loop.
        let outer_r2 = outer_r * outer_r;
        let inner_r2 = inner_r * inner_r;

        for x in 0..dims[0] {
            for y in 0..dims[1] {
                for z in 0..dims[2] {
                    let cx = (x as f32 - (dims[0] - 1) as f32 * 0.5) * spacing;
                    let cy = (y as f32 - (dims[1] - 1) as f32 * 0.5) * spacing;
                    let cz = (z as f32 - (dims[2] - 1) as f32 * 0.5) * spacing;
                    let d2 = cx * cx + cy * cy + cz * cz;

                    let active = match self.shape {
                        AuthoredShape::Sphere  => d2 < outer_r2,
                        AuthoredShape::Shell   => d2 > inner_r2 && d2 < outer_r2,
                        // Only activate the center z-slice so the face-on view shows a
                        // clean 2D silhouette. With all 16 z-layers active the perspective
                        // projection spreads front/back layers 2× apart in NDC, smearing
                        // each letter stroke into a radial ray that obscures the shape.
                        AuthoredShape::LetterA => z == dims[2] / 2 && letter_a_active(x, y),
                    };

                    let i = (x * dims[1] * dims[2] + y * dims[2] + z) as usize;
                    grid.cells[i].opacity = if active { ON_OPACITY } else { 0.0 };
                }
            }
        }
    }
}

// ---- Letter A bitmap ---------------------------------------------------------

// Each u16 is a row: bit x = column x. x=0 is left, x=15 is right.
// LETTER_A[y] = active columns for grid row y (y=0 is bottom, y=15 is top).
// Designed for 16×16 — matches the default GRID dimension exactly.
//
// Visual layout (y ascending = up on screen, looking down -z):
//   y=14  .......##.......   peak
//   y=13  ......####......
//   y=12  .....######.....
//   y=11  ....##....##....
//   y=10  ....##....##....
//   y=9   ...##......##...   upper legs (above crossbar)
//   y=8   ...###########..   crossbar   (cols 3-13)
//   y=7   ...##......##...
//   y=6   ...##......##...
//   y=5   ...##......##...   lower legs (below crossbar)
//   y=4   ..##........##..
//   y=3   ..##........##..
//   y=2   .##..........##.
//   y=1   .##..........##.   feet
//   y=0,15 empty
const LETTER_A: [u16; 16] = [
    0x0000, // y= 0  empty
    0x6006, // y= 1  cols 1,2,13,14
    0x6006, // y= 2
    0x300C, // y= 3  cols 2,3,12,13
    0x300C, // y= 4
    0x1818, // y= 5  cols 3,4,11,12
    0x1818, // y= 6
    0x1818, // y= 7
    0x3FF8, // y= 8  cols 3-13 (crossbar)
    0x1818, // y= 9
    0x1818, // y=10
    0x0C30, // y=11  cols 4,5,10,11
    0x07E0, // y=12  cols 5-10
    0x03C0, // y=13  cols 6-9
    0x0180, // y=14  cols 7,8  (peak)
    0x0000, // y=15  empty
];

fn letter_a_active(x: u32, y: u32) -> bool {
    if x >= 16 || y >= 16 { return false; }
    (LETTER_A[y as usize] >> x) & 1 == 1
}

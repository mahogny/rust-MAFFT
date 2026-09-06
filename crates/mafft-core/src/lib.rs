//! Core MAFFT alignment engine.
//!
//! Orchestrates the full alignment pipeline:
//! 1. Progressive alignment following a guide tree.
//! 2. Iterative refinement with score-based acceptance.
//! 3. Adding sequences to existing alignments.
//!
//! Ports the C `TreeDependentIteration()` from `tditeration.c`,
//! progressive alignment from `disttbfast.c`/`tbfast.c`, and
//! `addonetip()` from `addfunctions.c`.

mod add;
pub mod adjust_direction;
mod engine;
pub mod external;
pub mod progressive;
mod refinement;
mod varidist;

pub use add::{add_sequences, add_sequences_keeplength, add_sequences_keeplength_with_map};
pub use engine::{AlignmentMode, MafftEngine, dndpre_offset_shift, pair_penalty_scales};
pub use progressive::{
    MultipleAlignment, StepTrace, progressive_align, progressive_align_partial,
    progressive_align_unweighted, progressive_align_with_weights_override,
};
pub use refinement::{RefinementParams, iterative_refine};

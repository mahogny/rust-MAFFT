//! Pure Rust equivalents of MAFFT's core C data structures.
//!
//! These types use idiomatic Rust (Vec, Option, enums) instead of raw pointers
//! and linked lists.

mod complex;
mod local_hom;
mod scoring;
mod segment;
mod seq;
mod tree; // intentionally empty — tree types live in mafft-tree

pub use complex::*;
pub use local_hom::*;
pub use scoring::*;
pub use segment::*;
pub use seq::*;

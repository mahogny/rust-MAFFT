//! MAFFT multiple sequence alignment — high-level Rust API.
//!
//! This is the ergonomic entry point. Add it to your project with:
//!
//! ```text
//! cargo add mafft
//! ```
//!
//! and write:
//!
//! ```no_run
//! use mafft::{MafftEngine, AlignmentMode, SequenceSet, read_fasta};
//!
//! let input: SequenceSet = read_fasta("input.fasta").unwrap();
//! let engine = MafftEngine::new(AlignmentMode::FftNs2);
//! let msa = engine.align(&input);
//! for (name, seq) in msa.names.iter().zip(msa.sequences.iter()) {
//!     println!(">{name}");
//!     println!("{}", std::str::from_utf8(seq).unwrap());
//! }
//! ```
//!
//! # What this crate re-exports
//!
//! * [`mafft_core`] — alignment engine, modes, MSA result types
//! * [`mafft_types`] — `Sequence`, `SequenceSet`, scoring models, segment types
//! * [`mafft_io`] — FASTA / hat2 / localhom readers and writers
//!
//! Items from these crates are flattened into the root namespace below,
//! so most callers never need to write `mafft::core::` / `mafft::io::`
//! paths — `use mafft::*` is enough.
//!
//! # When to depend on the sub-crates directly
//!
//! Reach for `mafft-core` / `mafft-align` / `mafft-tree` / `mafft-scoring`
//! / `mafft-fft` directly only if you need to:
//!
//! * cut compile time by avoiding the I/O layer,
//! * pin a sub-crate to a specific version independently of the rest, or
//! * extend internals (e.g. custom guide trees, custom scoring matrices).
//!
//! For everything else, depend on `mafft`.
//!
//! # Related crates
//!
//! * [`mafft-rs`](https://crates.io/crates/mafft-rs) — standalone CLI:
//!   `cargo install mafft-rs`
//! * [`pymafft`](https://pypi.org/project/pymafft/) — Python bindings
//!   (`pip install pymafft`)

pub use mafft_core::*;
pub use mafft_io::*;
pub use mafft_types::*;

/// Sub-crate re-exports under explicit names, for callers who prefer
/// disambiguation over the flattened root namespace.
pub mod core {
    pub use mafft_core::*;
}

/// Sequence / scoring / segment types.
pub mod types {
    pub use mafft_types::*;
}

/// FASTA / hat2 / localhom I/O.
pub mod io {
    pub use mafft_io::*;
}

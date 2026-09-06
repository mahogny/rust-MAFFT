//! FFT-based homology detection for MAFFT.
//!
//! Uses `rustfft` with `num_complex::Complex64` (replacing the C hand-rolled
//! Cooley-Tukey FFT on `Fukusosuu` arrays).

mod block_align;
mod candidates;
mod correlation;
pub mod fft_c_compat;
mod segments;
mod vectorize;

pub use block_align::{block_align, block_align3};
pub use candidates::get_top_candidates;
pub use correlation::{cross_correlate, inner_product};
pub use segments::{AlignableSegment, SegmentParams, alignable_segments};
pub use vectorize::{
    DNA_CHANNELS, PROTEIN_CHANNELS, multichannel_correlate, sequences_to_channels,
};

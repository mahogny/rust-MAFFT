//! # Floating-point contraction: the reference C build emits no FMA
//!
//! Several routines here accumulate `acc = a * b + acc`. Whether that is one
//! rounding (fused multiply-add) or two (separate multiply then add) changes
//! the last ulp, and in tie-break-sensitive DP that changes the alignment.
//!
//! **Measured, not assumed:** disassembling C MAFFT 7.526 — both the conda
//! binary this project defines parity against and a clean build from the
//! pinned source with the project's own `-O3` flags — gives
//! `vfmadd`/`vfmsub` counts of **0** across `disttbfast`, `dvtditr` and
//! `tbfast`, against ~1250 `mulsd` and ~1550 `addsd`. Baseline x86-64 has no
//! FMA, so gcc cannot contract; it rounds twice.
//!
//! So this crate uses plain `a * b + c` and **must not** use `f64::mul_add`
//! in any path that mirrors C arithmetic. Earlier code did, on the belief
//! that `gcc -O3` contracts; several comments still describing that belief
//! predate the measurement. Removing it closed long-standing parity residues
//! (BAliBASE DNA default 139/141 -> 141/141, protein default 379/386 ->
//! 386/386) and removed the software-FMA emulation, cutting a 120x1.4kb
//! FFT-NS-i run from 22.9 s to 15.1 s.
//!
//! A build of C with `-march=native` on an FMA-capable host, or a clang
//! build with `FP_CONTRACT=on`, *would* contract — so "which C build" is
//! part of the parity definition, not an implementation detail.
//!
//! Pairwise and multi-sequence alignment algorithms for MAFFT.
//!
//! Phase 4a: Single-sequence pairwise alignment (Galign11, Lalign11, genalign11).
//! Phase 5: Profile alignment (MSalignmm), FFT-accelerated alignment (Falign),
//!          local homology constraint building (pairlocalalign),
//!          and constrained alignment (Falign_localhom, partSalignmm).

mod dp;
mod global;
mod local;
mod genaffine;
mod profile;
mod multimtx;
mod fft_align;
mod constraints;
mod constrained_align;
mod msalign;

pub use dp::{Alignment, AlignOp, GapModel, matrix_i32_to_f64};
pub use global::global_align;
pub use local::{local_align, LocalAlignment};
pub use genaffine::{genaffine_local_align, GenAffineGapModel};
pub use profile::{
    Profile, profile_align, profile_align_imp, profile_align_imp_with_tiebreak,
    profile_align_imp_with_boundary, profile_align_imp_multimtx, BoundaryFreqs,
    pairwise_align11, pairwise_align11_ex, align_with_anchors,
    reset_dp_pools, reset_cpmx_memo,
};
pub use multimtx::MultiMtx;
pub use fft_align::{fft_profile_align, find_fft_anchors, FftAlignParams, Anchor};
pub use constraints::{
    build_local_homology_table, build_homology_table,
    build_homology_table_with_unalign, build_imp_matrix,
    recompute_importance, PairAligner, FASTATHRESHOLD_DEFAULT,
    extract_putlocalhom2_regions, build_seed_homology_table,
    merge_homology_tables, parse_hat3_seed, SeedGroup,
};
pub use constrained_align::{
    constrained_profile_align, partial_profile_align, ConstrainedAlignParams,
};
pub use msalign::msalignmm;

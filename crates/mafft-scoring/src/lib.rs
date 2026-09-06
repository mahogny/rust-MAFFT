//! Scoring matrices and initialization for MAFFT alignment.
//!
//! Ports the C `constants.c`, `blosum.c`, `JTT.c`, `DNA.h`, and `miyata.h`
//! into pure Rust. Provides owned `ScoringContext` structs that replace the
//! C global variables.

mod alphabet;
mod ambiguity;
mod blosum;
mod dna;
pub mod jtt;
mod normalize;
mod penalties;
mod properties;

pub use alphabet::{Alphabet, DNA_ALPHABET, DnaAlphabet, PROTEIN_ALPHABET, ProteinAlphabet};
pub use ambiguity::{fill_dna_ambiguity_scores, fill_dna_n_scores};
pub use blosum::blosum_matrix;
pub use dna::{DNA_RIBOSUM4, DNA_RIBOSUM16, build_ribosumdis, default_dna_matrix};
pub use jtt::{jtt_frequencies, jtt_matrix, tm_frequencies};
pub use normalize::{build_scoring_matrix, normalize_matrix};
pub use penalties::{GapParams, default_dna_gap_params, default_protein_gap_params};
pub use properties::{POLARITY, VOLUME, normalized_polarity, normalized_volume};

use mafft_types::{GapPenalties, ScoringContext, ScoringModel, SeqType};

/// Build a complete `ScoringContext` for the given model and sequence type.
///
/// This is the main entry point, replacing the C `constants()` function.
/// Unlike the C version, this returns an owned struct instead of setting
/// globals.
pub fn build_context(model: ScoringModel, seq_type: SeqType) -> ScoringContext {
    build_context_with_kimura(model, seq_type, 2)
}

/// Mutate a DNA scoring context to enable C MAFFT's `--nwildcard`
/// behaviour. Ports `constants.c::nscore` exactly:
///
/// ```text
/// for i in 0..26:
///     n_dis[i][amino_n['n']] = round(0.25 * n_dis[i][i])
///     n_dis[amino_n['n']][i] = n_dis[i][amino_n['n']]
/// n_dis[amino_n['n']][amino_n['n']] = round(0.25 * (n_dis[0][0] + n_dis[1][1] + n_dis[2][2] + n_dis[3][3]))
/// ```
///
/// `amino_n['n']` in the DNA alphabet (`agctuAGCTUnNbdhkmnrsvwyx-O`)
/// is `17` — the SECOND occurrence of `'n'` wins because
/// `build_amino_map` overwrites on duplicates. The uppercase `'N'`
/// at index 11 is NOT touched (matches C exactly: `nscore` only
/// fills `amino_n['n']`, so `'N'` inputs do not benefit from
/// `--nwildcard`).
///
/// Updates `substitution_matrix`, `consweight_matrix`, and
/// `fft_matrix` so all DP paths see the new scores. The
/// `ln_matrix` and `ribosumdis` are out of the scored region
/// (10x10 / explicit fill) and not touched — matching C.
///
/// `--nzero` is the default behaviour (N-row stays zero), so no
/// separate function is needed for it.
pub fn apply_nwildcard(scoring: &mut ScoringContext) {
    if !scoring.seq_type.is_nucleotide() {
        return; // C `nscore` is DNA-only.
    }
    let n_idx = scoring.amino_map[b'n' as usize] as usize;
    if n_idx >= scoring.substitution_matrix.len() {
        return;
    }

    for i in 0..26 {
        let self_score = scoring.substitution_matrix[i][i] as f64;
        let v = (0.25 * self_score).round() as i32;
        scoring.substitution_matrix[i][n_idx] = v;
        scoring.substitution_matrix[n_idx][i] = v;
        scoring.consweight_matrix[i][n_idx] = v as f64;
        scoring.consweight_matrix[n_idx][i] = v as f64;
        // C `constants.c:525` runs `nscore` before `n_disFFT` is
        // rebuilt (`constants.c:553`-ish). Keep `fft_matrix` in
        // sync — for the scored region it's just `substitution_matrix
        // + offset` (offsetFFT = 0 in practice).
        if i < scoring.fft_matrix.len() && n_idx < scoring.fft_matrix[i].len() {
            let off = scoring.gap.offset;
            scoring.fft_matrix[i][n_idx] = v + off;
            scoring.fft_matrix[n_idx][i] = v + off;
        }
    }
    // C 2017/Jan/2 form: 0.25 * sum of A,G,C,T diagonals (NOT
    // 0.25 * 0.25 * sum, the older formula at constants.c:35).
    let sum_diag = (scoring.substitution_matrix[0][0]
        + scoring.substitution_matrix[1][1]
        + scoring.substitution_matrix[2][2]
        + scoring.substitution_matrix[3][3]) as f64;
    let v = (0.25 * sum_diag).round() as i32;
    scoring.substitution_matrix[n_idx][n_idx] = v;
    scoring.consweight_matrix[n_idx][n_idx] = v as f64;
    if n_idx < scoring.fft_matrix.len() {
        scoring.fft_matrix[n_idx][n_idx] = v + scoring.gap.offset;
    }
}

/// Build scoring context with a custom Kimura R parameter for DNA.
///
/// `kimura_r` controls the transition/transversion ratio in the Kimura
/// 2-parameter model used for DNA PAM matrix generation. Default is 2.
pub fn build_context_with_kimura(
    model: ScoringModel,
    seq_type: SeqType,
    kimura_r: i32,
) -> ScoringContext {
    match seq_type {
        SeqType::Dna | SeqType::Rna => build_dna_context_with_kimura(kimura_r),
        SeqType::Protein => build_protein_context(model),
        _ => build_protein_context(model),
    }
}

/// Build the FFT-specific matrix: n_disFFT[i][j] = n_dis[i][j] + offset - offsetFFT.
/// Since offsetFFT = 0 in practice, this is just n_dis[i][j] + offset.
/// Only applies to the scored region (20x20 for protein, 10x10 for DNA),
/// matching C which loops `for i<20; for j<20` leaving extended entries as zero.
/// Build the LN (log-normal) matrix: n_disLN[i][j] = n_dis[i][j] + offset - offsetLN.
fn build_ln_matrix(
    matrix: &[Vec<i32>],
    offset: i32,
    offset_ln: i32,
    nscored: usize,
) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let adj = (offset - offset_ln) as f64;
    let mut ln = vec![vec![0.0f64; n]; n];
    for i in 0..nscored.min(n) {
        for j in 0..nscored.min(n) {
            ln[i][j] = matrix[i][j] as f64 + adj;
        }
    }
    ln
}

fn build_fft_matrix(matrix: &[Vec<i32>], offset: i32, nscored: usize) -> Vec<Vec<i32>> {
    let n = matrix.len();
    let mut fft = vec![vec![0i32; n]; n];
    for i in 0..nscored.min(n) {
        for j in 0..nscored.min(n) {
            fft[i][j] = matrix[i][j] + offset; // offsetFFT = 0
        }
    }
    fft
}

fn build_dna_context_with_kimura(kimura_r: i32) -> ScoringContext {
    let gap = default_dna_gap_params();
    let n = DNA_ALPHABET.len();

    // Generate 4x4 PAM matrix via Kimura model (default: R=2, pamN=200),
    // matching C's generatenuc1pam() + exponentiation + normalization.
    let pam4x4 = dna::generate_dna_pam(kimura_r, 200, gap.offset);

    // Expand 4x4 → 10x10 (a,g,c,t,u,A,G,C,T,U) then → 26x26
    let mut matrix = vec![vec![0i32; n]; n];
    for i in 0..4 {
        for j in 0..4 {
            matrix[i][j] = pam4x4[i][j];
        }
    }

    // Copy u -> t mappings (index 4 copies from index 3)
    for i in 0..5 {
        matrix[4][i] = matrix[3][i];
        matrix[i][4] = matrix[i][3];
    }
    // Upper-case copies (indices 5-9 mirror 0-4)
    for i in 5..10 {
        for j in 5..10 {
            matrix[i][j] = matrix[i - 5][j - 5];
        }
    }

    // Fill DNA ambiguity codes (IUPAC: R,Y,K,M,S,W,B,D,H,V).
    // N wildcard scores are only filled when nwildcard flag is set (not default).
    fill_dna_ambiguity_scores(&mut matrix);

    let fft_matrix = build_fft_matrix(&matrix, gap.offset, 10);
    let ln_matrix = build_ln_matrix(&matrix, gap.offset, gap.offset_ln, 10);

    let consweight = matrix
        .iter()
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect();

    ScoringContext {
        substitution_matrix: matrix,
        consweight_matrix: consweight,
        polarity: [0.0; 256],
        volume: [0.0; 256],
        amino_map: DNA_ALPHABET.build_amino_map(),
        model: ScoringModel::Dna,
        seq_type: SeqType::Dna,
        gap: GapPenalties {
            open: gap.penalty,
            extend: gap.penalty_ex,
            offset: gap.offset,
        },
        nalphabets: 26,
        nscoredalphabets: 10,
        fft_matrix,
        ln_matrix,
        ribosumdis: Some(build_ribosumdis(gap.offset)),
    }
}

fn build_protein_context(model: ScoringModel) -> ScoringContext {
    let gap_params = default_protein_gap_params();

    let (raw_20x20, freq) = match model {
        ScoringModel::Blosum(n) => {
            let mat = blosum_matrix(n);
            let freq = blosum::blosum_frequencies();
            (mat, freq)
        }
        ScoringModel::Jtt(pam) => {
            let mat = jtt::build_jtt_pam_matrix(false, pam.max(1) as usize);
            let freq = jtt_frequencies();
            (mat, freq)
        }
        ScoringModel::Tm(pam) => {
            let mat = jtt::build_jtt_pam_matrix(true, pam.max(1) as usize);
            let freq = tm_frequencies();
            (mat, freq)
        }
        _ => {
            let mat = jtt::build_jtt_pam_matrix(false, 200);
            let freq = jtt_frequencies();
            (mat, freq)
        }
    };

    // Both BLOSUM and JTT/TM use the full normalization pipeline:
    // subtract weighted average, scale by 600/diag_avg, subtract offset.
    //
    // IMPORTANT: C's shell script (mafft.tmpl) sets defaultaof="0.000",
    // which means the scoring matrix offset (aof/poffset) is 0, not -123.
    // The -123 default in constants.c is overridden by the shell script.
    // So the scoring matrix does NOT include the offset subtraction.
    // The gap_params.offset (-73) is only used for FFT/LN matrices.
    let normalized = build_scoring_matrix(&raw_20x20, &freq, 0, true);

    // Expand 20x20 -> 26x26.
    // Matching C: initialize to zero, fill only the 20x20 core.
    // Entries 20-25 (B, Z, X, '.', '-', J) remain zero, matching C's behavior.
    let n = PROTEIN_ALPHABET.len();
    let mut matrix = vec![vec![0i32; n]; n];
    for i in 0..20 {
        for j in 0..20 {
            matrix[i][j] = normalized[i][j];
        }
    }

    let fft_matrix = build_fft_matrix(&matrix, gap_params.offset, 20);
    let ln_matrix = build_ln_matrix(&matrix, gap_params.offset, gap_params.offset_ln, 20);

    let consweight = matrix
        .iter()
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect();

    let pol = normalized_polarity();
    let vol = normalized_volume();
    let mut polarity_arr = [0.0f64; 256];
    let mut volume_arr = [0.0f64; 256];
    let alpha = &PROTEIN_ALPHABET;
    for (i, &ch) in alpha.chars.iter().enumerate() {
        if i < 20 {
            polarity_arr[ch as usize] = pol[i];
            volume_arr[ch as usize] = vol[i];
        }
    }

    ScoringContext {
        substitution_matrix: matrix,
        consweight_matrix: consweight,
        polarity: polarity_arr,
        volume: volume_arr,
        amino_map: PROTEIN_ALPHABET.build_amino_map(),
        model,
        seq_type: SeqType::Protein,
        gap: GapPenalties {
            open: gap_params.penalty,
            extend: gap_params.penalty_ex,
            offset: gap_params.offset,
        },
        nalphabets: 26,
        nscoredalphabets: 20,
        fft_matrix,
        ln_matrix,
        ribosumdis: None,
    }
}

/// Round to nearest integer, rounding 0.5 away from zero (C's shishagonyuu).
pub fn round_half_away(x: f64) -> i32 {
    if x > 0.0 {
        (x + 0.5) as i32
    } else if x < 0.0 {
        (x - 0.5) as i32
    } else {
        0
    }
}

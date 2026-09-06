/// Matrix normalization: the three-stage transform from raw scores to final
/// integer scoring matrix.
///
/// Ports the normalization logic from C `constants()`:
/// 1. Subtract weighted average (background correction)
/// 2. Scale by 600 / diagonal average
/// 3. Subtract offset
/// 4. Round to integer (shishagonyuu)
use crate::round_half_away;

/// Normalize a 20x20 raw scoring matrix into final integer scores.
///
/// - `raw`: the input matrix (BLOSUM integers or JTT log-transformed PAM)
/// - `freq`: amino acid frequencies (20 elements)
/// - `offset`: gap-related offset to subtract
/// - `rescale`: whether to apply average subtraction + 600 scaling
///
/// Returns a 20x20 integer matrix ready for alignment scoring.
pub fn build_scoring_matrix(
    raw: &[[f64; 20]; 20],
    freq: &[f64; 20],
    offset: i32,
    rescale: bool,
) -> [[i32; 20]; 20] {
    let mut mtx = [[0.0f64; 20]; 20];
    for i in 0..20 {
        for j in 0..20 {
            mtx[i][j] = raw[i][j];
        }
    }

    if rescale {
        // Step 1: subtract weighted average
        let avg2: f64 = (0..20)
            .flat_map(|i| (0..20).map(move |j| mtx[i][j] * freq[i] * freq[j]))
            .sum();
        for i in 0..20 {
            for j in 0..20 {
                mtx[i][j] -= avg2;
            }
        }

        // Step 2: scale by 600 / weighted diagonal average
        let diag_avg: f64 = (0..20).map(|i| mtx[i][i] * freq[i]).sum();
        if diag_avg.abs() > 1e-10 {
            let scale = 600.0 / diag_avg;
            for i in 0..20 {
                for j in 0..20 {
                    mtx[i][j] *= scale;
                }
            }
        }
    }

    // Step 3: subtract offset
    let offset_f = offset as f64;
    for i in 0..20 {
        for j in 0..20 {
            mtx[i][j] -= offset_f;
        }
    }

    // Step 4: round to integer
    let mut result = [[0i32; 20]; 20];
    for i in 0..20 {
        for j in 0..20 {
            result[i][j] = round_half_away(mtx[i][j]);
        }
    }
    result
}

/// Normalize a raw matrix without the full pipeline (used for already-
/// normalized matrices like JTT PAM output).
pub fn normalize_matrix(raw: &[[f64; 20]; 20], freq: &[f64; 20], offset: i32) -> [[i32; 20]; 20] {
    build_scoring_matrix(raw, freq, offset, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blosum::{blosum_frequencies, blosum_matrix};
    use crate::penalties::default_protein_gap_params;

    #[test]
    fn blosum62_diagonal_positive() {
        let raw = blosum_matrix(62);
        let freq = blosum_frequencies();
        let gap = default_protein_gap_params();
        let mtx = build_scoring_matrix(&raw, &freq, gap.offset, true);

        // All diagonal entries (self-scores) should be positive
        for i in 0..20 {
            assert!(
                mtx[i][i] > 0,
                "diagonal [{i}][{i}] = {} should be positive",
                mtx[i][i]
            );
        }
    }

    #[test]
    fn blosum62_symmetric() {
        let raw = blosum_matrix(62);
        let freq = blosum_frequencies();
        let gap = default_protein_gap_params();
        let mtx = build_scoring_matrix(&raw, &freq, gap.offset, true);

        for i in 0..20 {
            for j in 0..20 {
                assert_eq!(mtx[i][j], mtx[j][i], "asymmetric at [{i}][{j}]");
            }
        }
    }

    #[test]
    fn jtt_diagonal_positive() {
        let raw = crate::jtt::build_jtt_pam_matrix(false, 200);
        let freq = crate::jtt::jtt_frequencies();
        let gap = default_protein_gap_params();
        let mtx = build_scoring_matrix(&raw, &freq, gap.offset, true);

        for i in 0..20 {
            assert!(
                mtx[i][i] > 0,
                "JTT diagonal [{i}][{i}] = {} should be positive",
                mtx[i][i]
            );
        }
    }
}

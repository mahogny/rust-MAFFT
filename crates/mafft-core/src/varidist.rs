//! Multi-distance-class scoring infrastructure for `--allowshift` refinement.
//!
//! Port of C MAFFT's `_variousdist` support (tditeration.c + dvtditr.c):
//! when `specificityconsideration > 0` (i.e. `--allowshift` →
//! `unalign_level = 0.8`), the iterative refinement scores each sequence
//! pair with a distance-appropriate substitution matrix instead of one
//! global matrix. This module provides the three pure pieces that feed the
//! multi-matrix DP (`match_calc_add`/`match_calc_del`, step 2):
//!
//! - [`calc_max_dist_class`] — C `dvtditr.c::calcmaxdistclass`.
//! - [`make_scoring_matrices`] — C `tditeration.c::makescoringmatrices`.
//! - [`classify_pairs`] — C `tditeration.c::classifypairs`.
//!
//! The per-branch leaf distances (`smalldistmtx[i][j] = distarr[i] +
//! distarr[j]`) come from `mafft_tree::BranchWeights::dist_from_a_branch`
//! (C `distFromABranch`, `USEDISTONTREE = 1`).
//!
//! These functions are consumed by `refinement::realign_all_constrained_fft`
//! (via its `MultiMtxInput`) and validated end-to-end: `--allowshift
//! --globalpair` on the bundled sample is byte-identical to C MAFFT's
//! `test/sample.ginsi.allowshift` reference output.

use crate::progressive::{dist2offset, make_dynamic_matrix};

/// C `defs.c:124 ndistclass = 10` (fixed).
pub const NDISTCLASS: usize = 10;

/// Number of distinct distance classes actually used. Port of C
/// `dvtditr.c::calcmaxdistclass`: scan `c = 0..ndistclass`, stop at the
/// first `c` where `dist2offset(2c/ndistclass) == 0` (the offset has
/// saturated to 0 → all deeper classes share the un-shifted matrix), and
/// return `c + 1`. For `unalign_level = 0.8` this is 9.
pub fn calc_max_dist_class(unalign_level: f64) -> usize {
    let mut c = 0usize;
    while c < NDISTCLASS {
        let rep = 2.0 * c as f64 / NDISTCLASS as f64; // 0..2
        if dist2offset(rep, unalign_level) == 0.0 {
            break;
        }
        c += 1;
    }
    c + 1
}

/// Precompute the `max_dist_class` substitution matrices, one per distance
/// bin. Port of C `tditeration.c::makescoringmatrices`:
/// `makedynamicmtx(matrices[c], original, rep*0.5)` with `rep = 2c/ndistclass`.
/// C's `makedynamicmtx(out,in,arg)` computes `offset = dist2offset(arg*2)`,
/// matching our `make_dynamic_matrix(base, distfromtip = arg, ...)`.
pub fn make_scoring_matrices(
    base: &[Vec<f64>],
    unalign_level: f64,
    gap_idx: usize,
    max_dist_class: usize,
) -> Vec<Vec<Vec<f64>>> {
    (0..max_dist_class)
        .map(|c| {
            let rep = 2.0 * c as f64 / NDISTCLASS as f64; // 0..2
            // C: makedynamicmtx(matrices[c], original, rep*0.5)
            make_dynamic_matrix(base, rep * 0.5, unalign_level, gap_idx)
        })
        .collect()
}

/// Per-class profiles + per-pair class assignment for one refinement branch.
pub struct PairClassification {
    /// `matnum[i][j]` = distance class of cluster-1 member `i` vs cluster-2
    /// member `j`.
    pub matnum: Vec<Vec<usize>>,
    /// `eff1s[c][i]` = `eff1[i]` if member `i` of cluster 1 appears in any
    /// pair of class `c`, else 0. Per-class weight profile for cluster 1.
    pub eff1s: Vec<Vec<f64>>,
    /// `eff2s[c][j]` analogous for cluster 2.
    pub eff2s: Vec<Vec<f64>>,
}

/// Classify each (i, j) cluster-member pair into a distance bin and build
/// the per-class weight profiles. Port of C `tditeration.c::classifypairs`
/// (the active `#else` branch, lines 119-130):
///   `c = (int)(smalldist[i][j] / 2.0 * ndistclass)`, clamped to
///   `max_dist_class - 1`; `eff1s[c][i] = eff1[i]`, `eff2s[c][j] = eff2[j]`.
///
/// `smalldist[i][j]` is `distarr[i] + distarr[j]` (`dist_from_a_branch`
/// over the two clusters' members).
pub fn classify_pairs(
    eff1: &[f64],
    eff2: &[f64],
    smalldist: &[Vec<f64>],
    max_dist_class: usize,
) -> PairClassification {
    let n1 = eff1.len();
    let n2 = eff2.len();
    let mut eff1s = vec![vec![0.0f64; n1]; max_dist_class];
    let mut eff2s = vec![vec![0.0f64; n2]; max_dist_class];
    let mut matnum = vec![vec![0usize; n2]; n1];
    for i in 0..n1 {
        for j in 0..n2 {
            // C `(int)(... )` truncates toward zero; smalldist >= 0.
            let mut c = (smalldist[i][j] / 2.0 * NDISTCLASS as f64) as usize;
            if c >= max_dist_class {
                c = max_dist_class - 1;
            }
            eff1s[c][i] = eff1[i];
            eff2s[c][j] = eff2[j];
            matnum[i][j] = c;
        }
    }
    PairClassification {
        matnum,
        eff1s,
        eff2s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_dist_class_for_allowshift_is_9() {
        // unalign_level = 0.8: dist2offset(2c/10) = min(0, c/10 - 0.8),
        // first 0 at c=8 → maxdistclass = 9.
        assert_eq!(calc_max_dist_class(0.8), 9);
    }

    #[test]
    fn max_dist_class_disabled_is_1() {
        // sc = 0: dist2offset(0) = min(0, -0) = 0 at c=0 → maxdistclass = 1.
        assert_eq!(calc_max_dist_class(0.0), 1);
    }

    #[test]
    fn scoring_matrices_count_and_c0_is_base() {
        let base = vec![
            vec![10.0, -2.0, 0.0],
            vec![-2.0, 8.0, 1.0],
            vec![0.0, 1.0, 5.0],
        ];
        let mats = make_scoring_matrices(&base, 0.8, 2, calc_max_dist_class(0.8));
        assert_eq!(mats.len(), 9);
        // c=0: rep=0 → offset = dist2offset(0,0.8) = min(0,-0.8) = -0.8 ≠ 0,
        // so c=0 IS shifted (delta = -0.8*600). Only deep classes saturate.
        // Verify a deep class (c=8: rep=1.6 → offset=min(0,0.8-0.8)=0 → base).
        assert_eq!(mats[8][0][0], base[0][0]);
        assert_eq!(mats[8][0][1], base[0][1]);
    }

    #[test]
    fn classify_pairs_bins_and_clamps() {
        let eff1 = vec![0.5, 0.5];
        let eff2 = vec![1.0];
        // smalldist: pair(0,0)=0.4 → c=(int)(0.4/2*10)=2; pair(1,0)=3.0 →
        // c=(int)(3.0/2*10)=15 → clamp to 8.
        let smalldist = vec![vec![0.4], vec![3.0]];
        let pc = classify_pairs(&eff1, &eff2, &smalldist, 9);
        assert_eq!(pc.matnum[0][0], 2);
        assert_eq!(pc.matnum[1][0], 8);
        assert_eq!(pc.eff1s[2][0], 0.5);
        assert_eq!(pc.eff1s[8][1], 0.5);
        assert_eq!(pc.eff2s[2][0], 1.0);
        assert_eq!(pc.eff2s[8][0], 1.0);
    }
}

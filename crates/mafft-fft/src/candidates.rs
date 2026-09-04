/// Candidate lag selection from cross-correlation profiles.
///
/// Ports the C `getKouho()` function from fftFunctions.c.

/// A candidate lag position with its correlation score.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// The lag (shift) between the two sequences.
    /// Positive = seq2 shifted right relative to seq1.
    pub lag: i32,
    /// The correlation score at this lag.
    pub score: f64,
}

/// Find the top `n` candidate lags from a correlation profile.
///
/// The correlation array has length `nlen2`. The lag is computed relative
/// to the midpoint (`nlen2 / 2`), matching the C convention where
/// `kouho[j] = ikouho - nlen4`.
///
/// This is a greedy selection: after picking the best, its score is
/// suppressed and the next-best is found, etc.
///
/// Ports the C `getKouho()` function.
pub fn get_top_candidates(correlation: &[f64], n: usize) -> Vec<Candidate> {
    let nlen2 = correlation.len();
    let nlen4 = nlen2 / 2;

    let mut scores = correlation.to_vec();
    let mut candidates = Vec::with_capacity(n);

    for _ in 0..n {
        // C's `getKouho` (fftFunctions.c:104-114) uses a strict `>` comparison
        // when scanning the correlation array — so among tied peaks, the
        // FIRST (smallest-index) wins. `Iterator::max_by` returns the LAST
        // among equals, which inverts the tie-break and shifts FFT anchors
        // by one column on inputs whose normalized matrix lands two
        // correlation peaks within FP rounding distance (e.g. `--bl 50` step
        // 33, `--jtt 100` FFT, `--tm * FFT`).
        let mut best_idx = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (i, &s) in scores.iter().enumerate() {
            if s > best_score {
                best_idx = i;
                best_score = s;
            }
        }

        // Suppress this peak
        scores[best_idx] = f64::NEG_INFINITY;

        candidates.push(Candidate {
            lag: best_idx as i32 - nlen4 as i32,
            score: best_score,
        });
    }

    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_strongest_peak() {
        let mut corr = vec![0.0; 64];
        corr[20] = 5.0; // lag = 20 - 32 = -12
        corr[40] = 10.0; // lag = 40 - 32 = 8
        corr[50] = 3.0; // lag = 50 - 32 = 18

        let cands = get_top_candidates(&corr, 3);
        assert_eq!(cands.len(), 3);
        assert_eq!(cands[0].lag, 8); // strongest peak
        assert!((cands[0].score - 10.0).abs() < 1e-10);
        assert_eq!(cands[1].lag, -12); // second
        assert_eq!(cands[2].lag, 18); // third
    }

    #[test]
    fn lag_is_relative_to_midpoint() {
        let mut corr = vec![0.0; 128];
        corr[64] = 1.0; // exactly at midpoint → lag 0
        let cands = get_top_candidates(&corr, 1);
        assert_eq!(cands[0].lag, 0);
    }
}

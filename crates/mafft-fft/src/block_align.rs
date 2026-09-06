/// Optimal anchor pair selection via DP.
///
/// Ports the C `blockAlign2()` from fftFunctions.c.
///
/// Given cross-scores between segment pairs from two sequence groups,
/// selects the optimal non-overlapping subset of anchor pairs that
/// maximizes the total alignment score.

/// Select optimal anchor pairs from a cross-score matrix.
///
/// - `cross_scores[i][j]`: score of pairing segment `i` from group1 with
///   segment `j` from group2.
/// - `gap_penalty`: cost for skipping segments (non-diagonal moves).
///
/// Returns the selected pairs as `(cut1_indices, cut2_indices)`.
///
/// **Bug-for-bug port of C's `blockAlign2`** (`fftFunctions.c:455-509`).
/// C has a typo where the FIRST inner loop reads `maxj` (which is reset
/// only by the SECOND inner loop on the previous iteration) instead of
/// `maxi`. The `maxi`/`maxj` are also static TLS so they persist across
/// (i, j) iterations. We reproduce this state propagation faithfully —
/// changing the comparison to the "correct" `> maxi` shifts anchor
/// selection on inputs whose top-N candidates have similar correlation
/// scores (e.g. BL62 default step 15).
pub fn block_align(cross_scores: &[Vec<f64>], gap_penalty: f64) -> (Vec<usize>, Vec<usize>) {
    let ncut = cross_scores.len();
    if ncut == 0 {
        return (Vec::new(), Vec::new());
    }
    if ncut == 1 {
        return (vec![0], vec![0]);
    }

    // DP: score[i][j] = best cumulative score up to pairing segment i with j
    let mut dp = vec![vec![0.0f64; ncut]; ncut];
    // Track: 0 = diagonal, positive = skip in j, negative = skip in i
    let mut track = vec![vec![0i32; ncut]; ncut];

    // Initialize first row and column
    for i in 0..ncut {
        for j in 0..ncut {
            dp[i][j] = cross_scores[i][j];
        }
    }

    // Cross-iteration state mirroring C's `static TLS double maxj`.
    // `maxj` is reset to 0.0 inside the SECOND inner loop, but read by
    // the FIRST inner loop — so its value at the start of each (i, j)
    // iteration is whatever the previous iteration's second loop left it.
    let mut maxj_state: f64 = 0.0;

    for i in 1..ncut {
        for j in 1..ncut {
            // First loop: skip in j (column skip). C's typo compares
            // crossscore[i-1][k] against `maxj` (the stale outer state),
            // not `maxi` — see fftFunctions.c:470.
            let mut pointi: usize = 0;
            let mut maxi: f64 = 0.0;
            let klim_j = (j as i32 - 2).max(0) as usize;
            for k in 0..klim_j {
                // permit() == 0 in production C → skip iff (k != 0 AND
                // k < ncut-1 AND j < ncut-1).
                if k != 0 && k < ncut - 1 && j < ncut - 1 {
                    continue;
                }
                if dp[i - 1][k] > maxj_state {
                    pointi = k;
                    maxi = dp[i - 1][k];
                }
            }

            // Second loop: skip in i (row skip). Resets maxj to 0.0
            // BEFORE scanning, so it sees fresh-state — but the freshly-
            // computed maxj will be observed as the stale state by the
            // NEXT (i, j) iteration's first loop.
            let mut pointj: usize = 0;
            let mut maxj: f64 = 0.0;
            let klim_i = (i as i32 - 2).max(0) as usize;
            for k in 0..klim_i {
                if k != 0 && k < ncut - 1 && i < ncut - 1 {
                    continue;
                }
                if dp[k][j - 1] > maxj {
                    pointj = k;
                    maxj = dp[k][j - 1];
                }
            }

            let maxi_pen = maxi + gap_penalty;
            let maxj_pen = maxj + gap_penalty;

            let mut maximum = dp[i - 1][j - 1];
            track[i][j] = 0;

            if maximum < maxi_pen {
                maximum = maxi_pen;
                track[i][j] = (j - pointi) as i32;
            }
            if maximum < maxj_pen {
                maximum = maxj_pen;
                track[i][j] = -((i as i32) - pointj as i32);
            }

            dp[i][j] = cross_scores[i][j] + maximum;

            // Propagate the local maxj into outer state for the NEXT
            // iteration's first loop (mirrors C's static TLS persistence).
            maxj_state = maxj;
        }
    }

    // Traceback from (ncut-1, ncut-1). Mirrors `fftFunctions.c:520-545`:
    // walk back via `track` until either coordinate hits 0.
    let mut path_i: Vec<usize> = Vec::new();
    let mut path_j: Vec<usize> = Vec::new();

    let mut i = ncut - 1;
    let mut j = ncut - 1;
    path_i.push(i);
    path_j.push(j);

    loop {
        if i == 0 || j == 0 {
            break;
        }
        let shift = track[i][j];
        if shift == 0 {
            i -= 1;
            j -= 1;
        } else if shift > 0 {
            // gap in group2: came from (i-1, j-shift)
            let new_j = j.checked_sub(shift as usize);
            i -= 1;
            j = match new_j {
                Some(v) => v,
                None => break,
            };
        } else {
            // gap in group1: came from (i+shift, j-1)
            let new_i = i.checked_sub((-shift) as usize);
            j -= 1;
            i = match new_i {
                Some(v) => v,
                None => break,
            };
        }
        path_i.push(i);
        path_j.push(j);
    }

    path_i.reverse();
    path_j.reverse();

    // Filter + dedupe: mirrors `fftFunctions.c:547-560`. Drop cells whose
    // `cross_scores` is zero (corner sentinels stripped, unfilled cells
    // skipped), and when a kept cell shares a row OR column with the
    // previous kept cell, keep only the higher-scoring one.
    let mut result_i: Vec<usize> = Vec::new();
    let mut result_j: Vec<usize> = Vec::new();
    for (&ci, &cj) in path_i.iter().zip(path_j.iter()) {
        if cross_scores[ci][cj] == 0.0 {
            continue;
        }
        if let (Some(&prev_i), Some(&prev_j)) = (result_i.last(), result_j.last()) {
            if (ci == prev_i || cj == prev_j) && cross_scores[ci][cj] > cross_scores[prev_i][prev_j]
            {
                result_i.pop();
                result_j.pop();
            } else if ci == prev_i || cj == prev_j {
                // tie or worse — keep the previous, drop current
                continue;
            }
        }
        result_i.push(ci);
        result_j.push(cj);
    }

    (result_i, result_j)
}

/// Variant of block_align using running-max tracking (O(n²) time).
///
/// Ports C's `blockAlign3()`. Uses `jumpscore`/`jumppos` arrays to track
/// the best previous score in each row/column, avoiding the inner loop
/// scan of `blockAlign2`.
pub fn block_align3(cross_scores: &[Vec<f64>], gap_penalty: f64) -> (Vec<usize>, Vec<usize>) {
    let ncut = cross_scores.len();
    if ncut == 0 {
        return (Vec::new(), Vec::new());
    }
    if ncut == 1 {
        return (vec![0], vec![0]);
    }

    let mut dp = vec![vec![0.0f64; ncut]; ncut];
    let mut track = vec![vec![0i32; ncut]; ncut];

    for i in 0..ncut {
        for j in 0..ncut {
            dp[i][j] = cross_scores[i][j];
        }
    }

    // Running max trackers
    let mut jumpscore_col = vec![f64::NEG_INFINITY; ncut]; // best score in column j
    let mut jumppos_col = vec![0usize; ncut];

    for i in 1..ncut {
        let mut jumpscore_row = f64::NEG_INFINITY;
        let mut jumppos_row = 0usize;

        for j in 1..ncut {
            let mut best = dp[i - 1][j - 1];
            track[i][j] = 0;

            // Jump from best previous in row (skip columns)
            let row_score = jumpscore_row + gap_penalty;
            if row_score > best {
                best = row_score;
                track[i][j] = (j - jumppos_row) as i32;
            }

            // Jump from best previous in column (skip rows)
            let col_score = jumpscore_col[j] + gap_penalty;
            if col_score > best {
                best = col_score;
                track[i][j] = -((i - jumppos_col[j]) as i32);
            }

            dp[i][j] = cross_scores[i][j] + best;

            // Update running max for row
            if dp[i - 1][j] > jumpscore_row {
                jumpscore_row = dp[i - 1][j];
                jumppos_row = j;
            }

            // Update running max for column
            if dp[i][j - 1] > jumpscore_col[j] {
                jumpscore_col[j] = dp[i][j - 1];
                jumppos_col[j] = i;
            }
        }
    }

    // Traceback (same as blockAlign2)
    let mut result_i = Vec::new();
    let mut result_j = Vec::new();
    let mut i = ncut - 1;
    let mut j = ncut - 1;

    loop {
        if cross_scores[i][j] > 0.0 {
            result_i.push(i);
            result_j.push(j);
        }
        let shift = track[i][j];
        if shift == 0 {
            if i == 0 || j == 0 {
                break;
            }
            i -= 1;
            j -= 1;
        } else if shift > 0 {
            if i == 0 {
                break;
            }
            j -= shift as usize;
            i -= 1;
            if j == 0 {
                break;
            }
        } else {
            if j == 0 {
                break;
            }
            i -= (-shift) as usize;
            j -= 1;
            if i == 0 {
                break;
            }
        }
    }

    result_i.reverse();
    result_j.reverse();
    (result_i, result_j)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_scores_selected() {
        // Strong diagonal, weak off-diagonal
        let scores = vec![
            vec![10.0, 0.0, 0.0],
            vec![0.0, 10.0, 0.0],
            vec![0.0, 0.0, 10.0],
        ];
        let (ci, cj) = block_align(&scores, -5.0);
        assert_eq!(ci, vec![0, 1, 2]);
        assert_eq!(cj, vec![0, 1, 2]);
    }

    #[test]
    fn block_align3_diagonal() {
        let scores = vec![
            vec![10.0, 0.0, 0.0],
            vec![0.0, 10.0, 0.0],
            vec![0.0, 0.0, 10.0],
        ];
        let (ci, cj) = block_align3(&scores, -5.0);
        assert_eq!(ci, vec![0, 1, 2]);
        assert_eq!(cj, vec![0, 1, 2]);
    }

    #[test]
    fn skips_zero_score_pairs() {
        let scores = vec![vec![10.0, 0.0], vec![0.0, 0.0]];
        let (ci, cj) = block_align(&scores, -5.0);
        // Should only include the non-zero pair
        assert!(ci.contains(&0));
        assert!(
            !ci.iter()
                .zip(cj.iter())
                .any(|(&i, &j)| scores[i][j] == 0.0 && i > 0)
        );
    }

    #[test]
    fn empty_input() {
        let (ci, cj) = block_align(&[], -5.0);
        assert!(ci.is_empty());
        assert!(cj.is_empty());
    }
}

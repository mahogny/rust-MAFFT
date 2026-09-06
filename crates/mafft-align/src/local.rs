/// Local alignment (Smith-Waterman) with affine gap penalties.
///
/// Ports the C `L__align11()` from Lalign11.c.
use std::cell::RefCell;

use crate::dp::{AlignOp, Alignment, GapModel};

#[derive(Default)]
struct LocalScratch {
    seq2_idx: Vec<u8>,
    initverticalw: Vec<f64>,
    currentw: Vec<f64>,
    previousw: Vec<f64>,
    m_arr: Vec<f64>,
    mp_arr: Vec<i32>,
    ijp: Vec<i32>,
}

impl LocalScratch {
    const fn new() -> Self {
        Self {
            seq2_idx: Vec::new(),
            initverticalw: Vec::new(),
            currentw: Vec::new(),
            previousw: Vec::new(),
            m_arr: Vec::new(),
            mp_arr: Vec::new(),
            ijp: Vec::new(),
        }
    }
}

thread_local! {
    static LOCAL_SCRATCH: RefCell<LocalScratch> = const { RefCell::new(LocalScratch::new()) };
}

/// Result of a local alignment, including the offsets into original sequences.
#[derive(Debug, Clone)]
pub struct LocalAlignment {
    /// The alignment itself.
    pub alignment: Alignment,
    /// Start position in sequence 1 (0-based).
    pub offset1: usize,
    /// Start position in sequence 2 (0-based).
    pub offset2: usize,
}

/// Perform local alignment of two sequences.
///
/// Returns the highest-scoring local alignment. Unlike global alignment,
/// the alignment can start and end at any position in either sequence.
///
/// - `score_offset`: additional offset subtracted from all scores (shifts
///   the effective zero threshold for local alignment termination).
///   Corresponds to the C `scoreoffset` parameter.
pub fn local_align(
    seq1: &[u8],
    seq2: &[u8],
    matrix: &[Vec<f64>],
    amino_map: &[u8; 256],
    gap: &GapModel,
    score_offset: f64,
) -> LocalAlignment {
    let n = seq1.len();
    let m = seq2.len();

    if n == 0 || m == 0 {
        return LocalAlignment {
            alignment: Alignment {
                seq1: Vec::new(),
                seq2: Vec::new(),
                score: 0.0,
                operations: Vec::new(),
            },
            offset1: 0,
            offset2: 0,
        };
    }

    // Byte-for-byte port of C's `L__align11` from `Lalign11.c:208`.
    //
    // C's DP cell formula:
    //   currentw[j] = match(seq1[i], seq2[j]) + max(
    //       previousw[j-1],   // diagonal
    //       mi + open,        // horizontal gap (gap in seq1)
    //       m[j] + open       // vertical gap (gap in seq2)
    //   )
    // where `mi`/`m[j]` are running-max trackers updated AFTER use, so
    // gap of length k from row i-1 has cost `(k-1)*ext + open`. Indexing
    // is 1-based; cell (i, j) consumes seq1[i] and seq2[j] (0-indexed).
    //
    // C's loops run i=1..=lgth1, j=1..=lgth2 (one beyond seq end). At the
    // boundary i=lgth1 / j=lgth2 the seq access reads the null terminator,
    // contributing match score `mtx[0][?]` (typically 0 or near-0). We
    // emulate by treating those rows/columns as having 0 match score.
    // The boundary iteration is required so `maxwm` captures the H value
    // at the optimal cell — `wm` at (i, j) = previousw[j-1] reads the
    // real H(i-1, j-1).
    //
    // `ijp[i][j]` traceback codes (Lalign11.c:91-205):
    //   0          = diagonal: came from (i-1, j-1)
    //   -k (k>0)   = horizontal gap of length k: came from (i-1, j-k)
    //                emit (seq1[i-1], seq2[j-1]) then k-1 (gap, seq2[j-l])
    //   +k (k>0)   = vertical gap of length k: came from (i-k, j-1)
    //                emit (seq1[i-1], seq2[j-1]) then k-1 (seq1[i-l], gap)
    //   localstop  = local reset
    let f_open = gap.open;
    let f_ext = gap.extend;
    let localthr = -score_offset * 600.0;

    let n_alpha = matrix.len();
    let LocalScratch {
        mut seq2_idx,
        mut initverticalw,
        mut currentw,
        mut previousw,
        mut m_arr,
        mut mp_arr,
        mut ijp,
    } = LOCAL_SCRATCH.with_borrow_mut(std::mem::take);

    seq2_idx.clear();
    seq2_idx.reserve(m);
    seq2_idx.extend(seq2.iter().map(|&c| amino_map[c as usize]));
    let score_from_row = |row: &[f64], j: usize| -> f64 {
        let idx = unsafe { *seq2_idx.get_unchecked(j) } as usize;
        if idx < row.len() {
            unsafe { *row.get_unchecked(idx) }
        } else {
            0.0
        }
    };

    // initverticalw[k] = match(seq1[k], seq2[0]) for k in 0..n; 0 at k=n
    initverticalw.clear();
    initverticalw.resize(n + 1, 0.0);
    let seq2_0 = unsafe { *seq2_idx.get_unchecked(0) } as usize;
    for k in 0..n {
        let i1 = amino_map[seq1[k] as usize] as usize;
        initverticalw[k] = if i1 < n_alpha && seq2_0 < matrix[i1].len() {
            matrix[i1][seq2_0]
        } else {
            0.0
        };
    }
    // currentw[k] = match(seq1[0], seq2[k]) initially — row 0
    currentw.clear();
    currentw.resize(m + 1, 0.0);
    previousw.clear();
    previousw.resize(m + 1, 0.0);
    let seq1_0 = amino_map[seq1[0] as usize] as usize;
    let seq1_0_row = if seq1_0 < n_alpha {
        Some(&matrix[seq1_0])
    } else {
        None
    };
    for k in 0..m {
        currentw[k] = if let Some(row) = seq1_0_row {
            score_from_row(row, k)
        } else {
            0.0
        };
    }

    // m[j] / mp[j]: best H(k, j-1) + (i-k-1)*ext (k < i)
    m_arr.clear();
    m_arr.resize(m + 1, 0.0);
    mp_arr.clear();
    mp_arr.resize(m + 1, 0);
    for j in 1..=m {
        if j - 1 < m {
            m_arr[j] = currentw[j - 1];
        }
        mp_arr[j] = 0;
    }

    const LOCALSTOP: i32 = i32::MIN;
    let ijp_stride = m + 1;
    ijp.clear();
    ijp.resize((n + 1) * ijp_stride, 0);
    for i in 0..=n {
        ijp[i * ijp_stride] = LOCALSTOP;
    }
    for j in 0..=m {
        ijp[j] = LOCALSTOP;
    }

    let mut maxwm = f64::NEG_INFINITY;
    let mut endali = 0i32;
    let mut endalj = 0i32;

    for i in 1..=n {
        std::mem::swap(&mut previousw, &mut currentw);
        // Match scores for row i are fused into the DP loop below. This
        // preserves C's `currentw[j] = match; currentw[j] += wm` arithmetic
        // order while avoiding a separate full-row fill pass.
        let row_i = if i < n {
            let idx = amino_map[seq1[i] as usize] as usize;
            if idx < n_alpha {
                Some(&matrix[idx])
            } else {
                None
            }
        } else {
            None
        };

        // currentw[0] = initverticalw[i] (row column-0 match score)
        currentw[0] = if i < n { initverticalw[i] } else { 0.0 };

        let mut mi = previousw[0];
        let mut mpi: i32 = 0;
        // Hot DP loop: all slices below have length m + 1, and j runs 1..=m.
        // Bind rows once and use unchecked indexing to avoid repeating bounds
        // checks and the outer ijp row lookup in every cell.
        let row_start = i * ijp_stride;
        let ijp_row = &mut ijp[row_start..row_start + ijp_stride];
        let prev = &previousw;
        let cur = &mut currentw;
        let m_state = &mut m_arr;
        let mp_state = &mut mp_arr;

        if let Some(score_row) = row_i {
            for j in 1..m {
                let prev_diag = unsafe { *prev.get_unchecked(j - 1) };
                let mut wm = prev_diag;
                unsafe {
                    *ijp_row.get_unchecked_mut(j) = 0;
                }

                let g = mi + f_open;
                if g > wm {
                    wm = g;
                    unsafe {
                        *ijp_row.get_unchecked_mut(j) = -((j as i32) - mpi);
                    }
                }
                if prev_diag > mi {
                    mi = prev_diag;
                    mpi = (j as i32) - 1;
                }
                mi += f_ext;

                let m_j = unsafe { *m_state.get_unchecked(j) };
                let g = m_j + f_open;
                if g > wm {
                    wm = g;
                    unsafe {
                        *ijp_row.get_unchecked_mut(j) = (i as i32) - *mp_state.get_unchecked(j);
                    }
                }
                if prev_diag > m_j {
                    unsafe {
                        *m_state.get_unchecked_mut(j) = prev_diag;
                        *mp_state.get_unchecked_mut(j) = (i as i32) - 1;
                    }
                }
                unsafe {
                    *m_state.get_unchecked_mut(j) += f_ext;
                }

                if maxwm < wm {
                    maxwm = wm;
                    endali = i as i32;
                    endalj = j as i32;
                }
                if wm < localthr {
                    unsafe {
                        *ijp_row.get_unchecked_mut(j) = LOCALSTOP;
                    }
                    wm = localthr;
                }
                unsafe {
                    let match_score = score_from_row(score_row, j);
                    *cur.get_unchecked_mut(j) = match_score + wm;
                }
            }

            let j = m;
            let prev_diag = unsafe { *prev.get_unchecked(j - 1) };
            let mut wm = prev_diag;
            unsafe {
                *ijp_row.get_unchecked_mut(j) = 0;
            }

            let g = mi + f_open;
            if g > wm {
                wm = g;
                unsafe {
                    *ijp_row.get_unchecked_mut(j) = -((j as i32) - mpi);
                }
            }
            let m_j = unsafe { *m_state.get_unchecked(j) };
            let g = m_j + f_open;
            if g > wm {
                wm = g;
                unsafe {
                    *ijp_row.get_unchecked_mut(j) = (i as i32) - *mp_state.get_unchecked(j);
                }
            }
            if prev_diag > m_j {
                unsafe {
                    *m_state.get_unchecked_mut(j) = prev_diag;
                    *mp_state.get_unchecked_mut(j) = (i as i32) - 1;
                }
            }
            unsafe {
                *m_state.get_unchecked_mut(j) += f_ext;
            }

            if maxwm < wm {
                maxwm = wm;
                endali = i as i32;
                endalj = j as i32;
            }
            if wm < localthr {
                unsafe {
                    *ijp_row.get_unchecked_mut(j) = LOCALSTOP;
                }
                wm = localthr;
            }
            unsafe {
                *cur.get_unchecked_mut(j) = wm;
            }
        } else {
            for j in 1..=m {
                let prev_diag = unsafe { *prev.get_unchecked(j - 1) };
                let mut wm = prev_diag;
                unsafe {
                    *ijp_row.get_unchecked_mut(j) = 0;
                }

                let g = mi + f_open;
                if g > wm {
                    wm = g;
                    unsafe {
                        *ijp_row.get_unchecked_mut(j) = -((j as i32) - mpi);
                    }
                }
                if prev_diag > mi {
                    mi = prev_diag;
                    mpi = (j as i32) - 1;
                }
                mi += f_ext;

                let m_j = unsafe { *m_state.get_unchecked(j) };
                let g = m_j + f_open;
                if g > wm {
                    wm = g;
                    unsafe {
                        *ijp_row.get_unchecked_mut(j) = (i as i32) - *mp_state.get_unchecked(j);
                    }
                }
                if prev_diag > m_j {
                    unsafe {
                        *m_state.get_unchecked_mut(j) = prev_diag;
                        *mp_state.get_unchecked_mut(j) = (i as i32) - 1;
                    }
                }
                unsafe {
                    *m_state.get_unchecked_mut(j) += f_ext;
                }

                if maxwm < wm {
                    maxwm = wm;
                    endali = i as i32;
                    endalj = j as i32;
                }
                if wm < localthr {
                    unsafe {
                        *ijp_row.get_unchecked_mut(j) = LOCALSTOP;
                    }
                    wm = localthr;
                }
                unsafe {
                    *cur.get_unchecked_mut(j) = wm;
                }
            }
        }
    }

    // Empty alignment: no positive cell found
    if endali == 0 || endalj == 0 || maxwm <= 0.0 {
        LOCAL_SCRATCH.with_borrow_mut(|s| {
            *s = LocalScratch {
                seq2_idx,
                initverticalw,
                currentw,
                previousw,
                m_arr,
                mp_arr,
                ijp,
            };
        });
        return LocalAlignment {
            alignment: Alignment {
                seq1: Vec::new(),
                seq2: Vec::new(),
                score: 0.0,
                operations: Vec::new(),
            },
            offset1: 0,
            offset2: 0,
        };
    }

    // Traceback: walk ijp from (endali, endalj) back to a localstop or
    // to (0, *) / (*, 0). Mirrors `Ltracking` (Lalign11.c:91-205).
    //
    // Note: `endali`/`endalj` may equal `n`/`m` (the boundary row/column).
    // The match score added at that cell is 0 (C reads null terminator),
    // so we don't emit a residue pair there — instead start the traceback
    // by stepping back one cell to (endali-1, endalj-1) effectively. The
    // `*curpt += wm` at (endali, endalj) added wm to a 0 match, so the
    // returned `score = maxwm` already equals H(endali-1, endalj-1).
    let traceback_cap = endali.max(0) as usize + endalj.max(0) as usize;
    let mut ops: Vec<AlignOp> = Vec::with_capacity(traceback_cap);
    let mut a1: Vec<u8> = Vec::with_capacity(traceback_cap);
    let mut a2: Vec<u8> = Vec::with_capacity(traceback_cap);
    let mut iin = endali;
    let mut jin = endalj;
    // If we ended at the boundary (i=n or j=m), there's no residue at
    // (iin-1, jin-1) to emit for that cell — just step back.
    if iin == n as i32 && jin == m as i32 {
        // No-op: process the move from (iin, jin) which has ijp set
    }

    let mut last_ifi = iin;
    let mut last_jfi = jin;

    loop {
        if iin <= 0 || jin <= 0 {
            break;
        }
        if (iin as usize) > n || (jin as usize) > m {
            break;
        }
        let v = ijp[iin as usize * ijp_stride + jin as usize];

        if v == LOCALSTOP {
            // No residue emitted at the localstop cell; chain ends.
            last_ifi = iin;
            last_jfi = jin;
            break;
        }

        let (ifi, jfi);
        if v < 0 {
            // Horizontal gap of length k = -v
            ifi = iin - 1;
            jfi = jin + v; // jin - k
            // Emit gap-residue cells for columns jfi+1..jin-1 of seq2
            let mut l = jin - jfi;
            loop {
                l -= 1;
                if l <= 0 {
                    break;
                }
                a1.push(b'-');
                a2.push(seq2[(jfi + l) as usize]);
                ops.push(AlignOp::Insert);
            }
        } else if v > 0 {
            // Vertical gap of length k = v
            ifi = iin - v;
            jfi = jin - 1;
            let mut l = iin - ifi;
            loop {
                l -= 1;
                if l <= 0 {
                    break;
                }
                a1.push(seq1[(ifi + l) as usize]);
                a2.push(b'-');
                ops.push(AlignOp::Delete);
            }
        } else {
            // Diagonal
            ifi = iin - 1;
            jfi = jin - 1;
        }

        // At the boundary (iin == n or jin == m), the "match" at (iin, jin)
        // is bogus (read past sequence end). Skip residue emission for
        // that cell. For real cells, emit residue pair at (ifi, jfi) =
        // the source — note seq1[ifi]/seq2[jfi] only valid when ifi<n and
        // jfi<m. Match cells beyond the seq are just structural.
        if (ifi as usize) < n && (jfi as usize) < m {
            a1.push(seq1[ifi as usize]);
            a2.push(seq2[jfi as usize]);
            ops.push(AlignOp::Match);
        }
        last_ifi = ifi;
        last_jfi = jfi;

        if ifi <= 0 || jfi <= 0 || ijp[ifi as usize * ijp_stride + jfi as usize] == LOCALSTOP {
            break;
        }
        iin = ifi;
        jin = jfi;
    }

    a1.reverse();
    a2.reverse();
    ops.reverse();

    let offset1 = last_ifi.max(0) as usize;
    let offset2 = last_jfi.max(0) as usize;
    LOCAL_SCRATCH.with_borrow_mut(|s| {
        *s = LocalScratch {
            seq2_idx,
            initverticalw,
            currentw,
            previousw,
            m_arr,
            mp_arr,
            ijp,
        };
    });

    LocalAlignment {
        alignment: Alignment {
            seq1: a1,
            seq2: a2,
            score: maxwm,
            operations: ops,
        },
        offset1,
        offset2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_matrix() -> (Vec<Vec<f64>>, [u8; 256]) {
        let mut mtx = vec![vec![-100.0f64; 5]; 5];
        for i in 0..4 {
            mtx[i][i] = 100.0;
        }
        let mut map = [0xFFu8; 256];
        map[b'A' as usize] = 0;
        map[b'C' as usize] = 1;
        map[b'G' as usize] = 2;
        map[b'T' as usize] = 3;
        map[b'-' as usize] = 4;
        (mtx, map)
    }

    #[test]
    fn finds_local_match() {
        let (mtx, map) = simple_matrix();
        let gap = GapModel::new(-200.0, -10.0);
        // Embed a matching region in mismatching flanks
        let s1 = b"TTTTACGTTTTT";
        let s2 = b"GGGGACGTGGGG";
        let result = local_align(s1, s2, &mtx, &map, &gap, 0.0);

        // Should find the ACGT match
        let ungapped1: Vec<u8> = result
            .alignment
            .seq1
            .iter()
            .filter(|&&c| c != b'-')
            .cloned()
            .collect();
        let ungapped2: Vec<u8> = result
            .alignment
            .seq2
            .iter()
            .filter(|&&c| c != b'-')
            .cloned()
            .collect();
        assert_eq!(ungapped1, b"ACGT");
        assert_eq!(ungapped2, b"ACGT");
        assert_eq!(result.offset1, 4);
        assert_eq!(result.offset2, 4);
    }

    #[test]
    fn local_score_non_negative() {
        let (mtx, map) = simple_matrix();
        let gap = GapModel::new(-200.0, -10.0);
        let result = local_align(b"AAAA", b"CCCC", &mtx, &map, &gap, 0.0);
        assert!(result.alignment.score >= 0.0);
    }

    #[test]
    fn identical_sequences_full_match() {
        let (mtx, map) = simple_matrix();
        let gap = GapModel::new(-200.0, -10.0);
        let result = local_align(b"ACGT", b"ACGT", &mtx, &map, &gap, 0.0);
        assert!((result.alignment.score - 400.0).abs() < 1e-6);
        assert_eq!(result.offset1, 0);
        assert_eq!(result.offset2, 0);
    }
}

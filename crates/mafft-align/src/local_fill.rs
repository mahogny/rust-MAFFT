//! The DP fill of [`crate::local::local_align`], four rows at a time. Not in C MAFFT.
//!
//! Each cell of `L__align11` carries a running horizontal-gap maximum `mi` from the cell to
//! its left, so a row is one serial chain of compare, select and add. With AVX2, rows
//! `i..i+4` run together in the lanes of one vector, row `i+k` at column `j-k`
//! ([`crate::local_fill_avx2`]): every cell still reads exactly the values it read row by
//! row (the row above has always passed the column first) and does the same IEEE operations,
//! so every value is the same, but four chains advance per step.
//!
//! The one order-dependent result is the running maximum: `L__align11` keeps the first
//! maximal cell in row-major order. Each row keeps its own first maximum, and rows are merged
//! in order with the same strict comparison, which picks the same cell.

use std::ops::RangeInclusive;

/// Traceback code for a local restart.
pub(crate) const LOCALSTOP: i32 = i32::MIN;

pub(crate) struct Fill<'a> {
    pub n: usize,
    pub m: usize,
    pub matrix: &'a [Vec<f64>],
    /// `amino_map` of each residue
    pub seq1_idx: &'a [u8],
    pub seq2_idx: &'a [u8],
    /// `H(r, 0)` for `r >= 1`
    pub initverticalw: &'a [f64],
    pub f_open: f64,
    pub f_ext: f64,
    pub localthr: f64,
}

/// `h` holds the latest row computed at each column, `m_arr`/`mp_arr` the vertical-gap
/// state; all three have `m + 1` entries, `ijp` `(n + 1) * (m + 1)`.
pub(crate) struct State<'s> {
    pub h: &'s mut [f64],
    pub m_arr: &'s mut [f64],
    pub mp_arr: &'s mut [i32],
    pub ijp: &'s mut [i32],
}

/// One row's running state while it crosses the matrix.
#[derive(Clone, Copy)]
pub(crate) struct Lane {
    pub i: usize,
    /// `H(i - 1, j - 1)` for the lane's next column `j`
    pub diag: f64,
    pub mi: f64,
    pub mpi: i32,
    pub best: f64,
    pub best_j: i32,
}

impl Fill<'_> {
    /// Fills `ijp` rows `1..=n` and returns `(maxwm, endali, endalj)`. On entry `st.h` holds
    /// row 0 and `m_arr`/`mp_arr` the vertical-gap state after it.
    pub fn run(&self, st: &mut State) -> (f64, i32, i32) {
        let (n, m) = (self.n, self.m);
        assert!(st.h.len() == m + 1 && st.m_arr.len() == m + 1 && st.mp_arr.len() == m + 1);
        assert!(st.ijp.len() == (n + 1) * (m + 1) && self.initverticalw.len() > n.saturating_sub(1));
        assert!(self.seq1_idx.len() == n && self.seq2_idx.len() == m);
        let profile = (m >= 4 && n >= 4).then(|| crate::local_fill_avx2::Profile::new(self)).flatten();
        let width = if profile.is_some() { 4 } else { 1 };
        let (mut maxwm, mut endali, mut endalj) = (f64::NEG_INFINITY, 0i32, 0i32);
        let mut i0 = 1;
        while i0 <= n {
            let rows = width.min(n + 1 - i0);
            let mut lanes = [Lane { i: 0, diag: 0.0, mi: 0.0, mpi: 0, best: f64::NEG_INFINITY, best_j: 0 }; 4];
            for (k, lane) in lanes.iter_mut().enumerate().take(rows) {
                let i = i0 + k;
                let left = if i == 1 { st.h[0] } else { self.initverticalw[i - 1] };
                *lane = Lane { i, diag: left, mi: left, ..*lane };
            }
            match &profile {
                Some(p) if rows == 4 => crate::local_fill_avx2::strip(self, p, &mut lanes, st),
                _ => self.steps(&mut lanes[..rows], 1..=m + rows - 1, st),
            }
            for lane in &lanes[..rows] {
                if maxwm < lane.best {
                    maxwm = lane.best;
                    endali = lane.i as i32;
                    endalj = lane.best_j;
                }
            }
            i0 += rows;
        }
        (maxwm, endali, endalj)
    }

    /// Steps `s` of a strip, one cell per lane in range: lane `k` at column `s - k`.
    pub(crate) fn steps(&self, lanes: &mut [Lane], s: RangeInclusive<usize>, st: &mut State) {
        for s in s {
            for (k, lane) in lanes.iter_mut().enumerate() {
                if s > k && s - k <= self.m {
                    self.cell(lane, s - k, st);
                }
            }
        }
    }

    pub(crate) fn score_row(&self, i: usize) -> Option<&[f64]> {
        self.seq1_idx.get(i).and_then(|&c| self.matrix.get(c as usize)).map(Vec::as_slice)
    }

    /// One `L__align11` cell, as selects; see `local_align` for the codes.
    #[inline(always)]
    fn cell(&self, lane: &mut Lane, j: usize, st: &mut State) {
        let (i, ii, ji) = (lane.i, lane.i as i32, j as i32);
        let prev_diag = lane.diag;
        lane.diag = st.h[j];

        let g = lane.mi + self.f_open;
        let horizontal = g > prev_diag;
        let mut wm = if horizontal { g } else { prev_diag };
        let mut code = if horizontal { lane.mpi - ji } else { 0 };
        let opens = prev_diag > lane.mi;
        lane.mi = (if opens { prev_diag } else { lane.mi }) + self.f_ext;
        lane.mpi = if opens { ji - 1 } else { lane.mpi };

        let (m_j, mp_j) = (st.m_arr[j], st.mp_arr[j]);
        let g = m_j + self.f_open;
        let vertical = g > wm;
        wm = if vertical { g } else { wm };
        code = if vertical { ii - mp_j } else { code };
        let opens = prev_diag > m_j;
        st.m_arr[j] = (if opens { prev_diag } else { m_j }) + self.f_ext;
        st.mp_arr[j] = if opens { ii - 1 } else { mp_j };

        if lane.best < wm {
            lane.best = wm;
            lane.best_j = ji;
        }
        let stop = wm < self.localthr;
        st.ijp[i * (self.m + 1) + j] = if stop { LOCALSTOP } else { code };
        wm = if stop { self.localthr } else { wm };
        // C adds the match score only inside both sequences; past either end it stores `wm`
        st.h[j] = match self.score_row(i) {
            Some(row) if j < self.m => {
                let idx = self.seq2_idx[j] as usize;
                (if idx < row.len() { row[idx] } else { 0.0 }) + wm
            }
            _ => wm,
        };
    }
}

#[cfg(test)]
#[path = "local_fill_tests.rs"]
mod tests;

//! AVX2 form of [`crate::local_fill`]'s four-row strip. Not in C MAFFT.
//!
//! Lane `e` is row `i0 + e` at column `s - e` on step `s`. What a lane needs from the row
//! above at its column (`H` and the vertical-gap state) is what lane `e - 1` produced on the
//! previous step, so it arrives by a one-lane register shift; lane 0 reads it from memory
//! (the previous strip's last row), and lane 3's results go back to memory for the next
//! strip. The first and last three steps, where some lanes are outside the matrix, run as
//! scalar cells. Integer state (`mpi`, codes) rides in 64-bit lanes so one mask selects both.

use crate::local_fill::{Fill, Lane, State};

/// Match score of each residue class of seq1 against every column of seq2, `m + 1` per
/// class, laid out for one gather per step. Past seq2's end, and for rows with no score row,
/// it holds -0.0: `wm + -0.0` is bit-for-bit `wm`, which is what the scalar cell stores there.
pub(crate) struct Profile {
    q: Vec<f64>,
    /// start of each class's row in `q`; classes with no score row share the -0.0 row
    offset: [usize; 256],
}

impl Profile {
    /// `None` without AVX2.
    pub fn new(f: &Fill) -> Option<Self> {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            let stride = f.m + 1;
            let mut q = vec![-0.0; stride];
            let mut offset = [0usize; 256];
            let mut seen = [false; 256];
            for &c in f.seq1_idx {
                let Some(row) = f.matrix.get(c as usize) else { continue };
                if std::mem::replace(&mut seen[c as usize], true) {
                    continue;
                }
                offset[c as usize] = q.len();
                q.extend(f.seq2_idx.iter().map(|&b| row.get(b as usize).copied().unwrap_or(0.0)));
                q.push(-0.0);
            }
            return Some(Profile { q, offset });
        }
        None
    }

    fn row_offset(&self, f: &Fill, i: usize) -> usize {
        f.seq1_idx.get(i).map_or(0, |&c| self.offset[c as usize])
    }
}

/// Rows `lanes[0].i..+4`, all of `1..=m` (`m >= 4`).
pub(crate) fn strip(f: &Fill, p: &Profile, lanes: &mut [Lane; 4], st: &mut State) {
    f.steps(lanes, 1..=3, st);
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a `Profile` exists only when AVX2 was detected; `run` checked the lengths.
    unsafe {
        kernel(f, p, lanes, st)
    };
    f.steps(lanes, f.m + 1..=f.m + 3, st);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn kernel(f: &Fill, p: &Profile, lanes: &mut [Lane; 4], st: &mut State) { unsafe {
    use std::arch::x86_64::*;
    use std::mem::transmute;

    // macros, not closures: a closure would not inherit `target_feature`
    macro_rules! sel {
        ($a:expr, $b:expr, $mask:expr) => {
            _mm256_blendv_pd($a, $b, $mask)
        };
    }
    macro_rules! sel_i {
        ($a:expr, $b:expr, $mask:expr) => {
            _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd($a), _mm256_castsi256_pd($b), $mask))
        };
    }
    macro_rules! pd {
        (|$l:ident| $e:expr) => {{
            let g = |$l: &Lane| -> f64 { $e };
            _mm256_setr_pd(g(&lanes[0]), g(&lanes[1]), g(&lanes[2]), g(&lanes[3]))
        }};
    }
    macro_rules! ep {
        (|$l:ident| $e:expr) => {{
            let g = |$l: &Lane| -> i64 { $e };
            _mm256_setr_epi64x(g(&lanes[0]), g(&lanes[1]), g(&lanes[2]), g(&lanes[3]))
        }};
    }
    // lane e lags e columns: shift each lane's value up one, lane 0 takes `first`
    macro_rules! shift {
        ($v:expr, $first:expr) => {
            _mm256_blend_pd::<1>(_mm256_permute4x64_pd::<0x93>($v), _mm256_set1_pd($first))
        };
    }
    macro_rules! shift_i {
        ($v:expr, $first:expr) => {
            _mm256_blend_epi32::<3>(_mm256_permute4x64_epi64::<0x93>($v), _mm256_set1_epi64x($first as i64))
        };
    }

    let (m, stride, i0) = (f.m, f.m + 1, lanes[0].i);
    let (open, ext, thr) = (_mm256_set1_pd(f.f_open), _mm256_set1_pd(f.f_ext), _mm256_set1_pd(f.localthr));
    let stopcode = _mm256_set1_epi64x(crate::local_fill::LOCALSTOP as i64);
    let lag = _mm256_setr_epi64x(0, -1, -2, -3);
    let iv = ep!(|l| l.i as i64);
    let iv1 = _mm256_sub_epi64(iv, _mm256_set1_epi64x(1));
    let qbase = _mm256_add_epi64(ep!(|l| p.row_offset(f, l.i) as i64), lag);
    let ijp_at: [usize; 4] = std::array::from_fn(|e| (i0 + e) * stride - e);

    let mut diag = pd!(|l| l.diag);
    let mut mi = pd!(|l| l.mi);
    let mut mpi = ep!(|l| l.mpi as i64);
    let mut best = pd!(|l| l.best);
    let mut best_j = ep!(|l| l.best_j as i64);
    // step 3's results; lane 3 was still at column 0
    let (h, ma, mp) = (&mut *st.h, &mut *st.m_arr, &mut *st.mp_arr);
    let mut out = _mm256_setr_pd(h[3], h[2], h[1], 0.0);
    let mut mout = _mm256_setr_pd(ma[3], ma[2], ma[1], 0.0);
    let mut mpout = _mm256_setr_epi64x(mp[3] as i64, mp[2] as i64, mp[1] as i64, 0);

    for s in 4..=m {
        if s > 4 {
            let (o, mo, mpo): ([f64; 4], [f64; 4], [i64; 4]) = (transmute(out), transmute(mout), transmute(mpout));
            *h.get_unchecked_mut(s - 4) = o[3];
            *ma.get_unchecked_mut(s - 4) = mo[3];
            *mp.get_unchecked_mut(s - 4) = mpo[3] as i32;
        }
        let jv = _mm256_add_epi64(_mm256_set1_epi64x(s as i64), lag);
        let above = shift!(out, *h.get_unchecked(s));
        let m_above = shift!(mout, *ma.get_unchecked(s));
        let mp_above = shift_i!(mpout, *mp.get_unchecked(s));
        let prev_diag = diag;
        diag = above;

        let g = _mm256_add_pd(mi, open);
        let horizontal = _mm256_cmp_pd::<_CMP_GT_OQ>(g, prev_diag);
        let mut wm = sel!(prev_diag, g, horizontal);
        let mut code = sel_i!(_mm256_setzero_si256(), _mm256_sub_epi64(mpi, jv), horizontal);
        let opens = _mm256_cmp_pd::<_CMP_GT_OQ>(prev_diag, mi);
        mi = _mm256_add_pd(sel!(mi, prev_diag, opens), ext);
        mpi = sel_i!(mpi, _mm256_sub_epi64(jv, _mm256_set1_epi64x(1)), opens);

        let g = _mm256_add_pd(m_above, open);
        let vertical = _mm256_cmp_pd::<_CMP_GT_OQ>(g, wm);
        wm = sel!(wm, g, vertical);
        code = sel_i!(code, _mm256_sub_epi64(iv, mp_above), vertical);
        let opens = _mm256_cmp_pd::<_CMP_GT_OQ>(prev_diag, m_above);
        mout = _mm256_add_pd(sel!(m_above, prev_diag, opens), ext);
        mpout = sel_i!(mp_above, iv1, opens);

        let better = _mm256_cmp_pd::<_CMP_LT_OQ>(best, wm);
        best = sel!(best, wm, better);
        best_j = sel_i!(best_j, jv, better);
        let stop = _mm256_cmp_pd::<_CMP_LT_OQ>(wm, thr);
        code = sel_i!(code, stopcode, stop);
        wm = sel!(wm, thr, stop);
        let q = _mm256_i64gather_pd::<8>(p.q.as_ptr(), _mm256_add_epi64(qbase, _mm256_set1_epi64x(s as i64)));
        out = _mm256_add_pd(q, wm);

        let c: [i64; 4] = transmute(code);
        for e in 0..4 {
            *st.ijp.get_unchecked_mut(ijp_at[e] + s) = c[e] as i32;
        }
    }

    // hand every lane's last results to the scalar steps, as the scalar cell would have left them
    let (o, mo, mpo): ([f64; 4], [f64; 4], [i64; 4]) = (transmute(out), transmute(mout), transmute(mpout));
    let (d, mi, mpi): ([f64; 4], [f64; 4], [i64; 4]) = (transmute(diag), transmute(mi), transmute(mpi));
    let (b, bj): ([f64; 4], [i64; 4]) = (transmute(best), transmute(best_j));
    for e in 0..4 {
        h[m - e] = o[e];
        ma[m - e] = mo[e];
        mp[m - e] = mpo[e] as i32;
        lanes[e] = Lane { diag: d[e], mi: mi[e], mpi: mpi[e] as i32, best: b[e], best_j: bj[e] as i32, ..lanes[e] };
    }
}}

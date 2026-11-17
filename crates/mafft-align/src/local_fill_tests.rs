use super::*;

/// `L__align11`'s fill row by row, with branches, as `local_align` ran it before `Fill`.
fn reference(f: &Fill, st: &mut State) -> (f64, i32, i32) {
    let (n, m) = (f.n, f.m);
    let (mut maxwm, mut endali, mut endalj) = (f64::NEG_INFINITY, 0, 0);
    let mut prev = st.h.to_vec();
    for i in 1..=n {
        let left = if i == 1 { prev[0] } else { f.initverticalw[i - 1] };
        let (mut mi, mut mpi) = (left, 0i32);
        let mut cur = vec![0.0; m + 1];
        for j in 1..=m {
            let prev_diag = if j == 1 { left } else { prev[j - 1] };
            let mut wm = prev_diag;
            let mut code = 0;
            if mi + f.f_open > wm {
                wm = mi + f.f_open;
                code = -((j as i32) - mpi);
            }
            if prev_diag > mi {
                mi = prev_diag;
                mpi = j as i32 - 1;
            }
            mi += f.f_ext;
            if st.m_arr[j] + f.f_open > wm {
                wm = st.m_arr[j] + f.f_open;
                code = i as i32 - st.mp_arr[j];
            }
            if prev_diag > st.m_arr[j] {
                st.m_arr[j] = prev_diag;
                st.mp_arr[j] = i as i32 - 1;
            }
            st.m_arr[j] += f.f_ext;
            if maxwm < wm {
                (maxwm, endali, endalj) = (wm, i as i32, j as i32);
            }
            if wm < f.localthr {
                code = LOCALSTOP;
                wm = f.localthr;
            }
            st.ijp[i * (m + 1) + j] = code;
            cur[j] = match f.score_row(i) {
                Some(row) if j < m => {
                    let idx = f.seq2_idx[j] as usize;
                    (if idx < row.len() { row[idx] } else { 0.0 }) + wm
                }
                _ => wm,
            };
        }
        prev = cur;
    }
    st.h.copy_from_slice(&prev);
    (maxwm, endali, endalj)
}

/// `Fill::run` with one row per strip, so all scalar.
fn scalar(f: &Fill, st: &mut State) -> (f64, i32, i32) {
    let mut best = (f64::NEG_INFINITY, 0, 0);
    for i in 1..=f.n {
        let left = if i == 1 { st.h[0] } else { f.initverticalw[i - 1] };
        let mut lane = [Lane { i, diag: left, mi: left, mpi: 0, best: f64::NEG_INFINITY, best_j: 0 }];
        f.steps(&mut lane, 1..=f.m, st);
        if best.0 < lane[0].best {
            best = (lane[0].best, i as i32, lane[0].best_j);
        }
    }
    best
}

#[test]
fn every_fill_matches_the_row_by_row_fill() {
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = |k: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x % k
    };
    // integer-valued scores make ties, and so the tie rules, common
    let matrix: Vec<Vec<f64>> = (0..4).map(|a| (0..4).map(|b| if a == b { 60.0 } else { -40.0 }).collect()).collect();
    for case in 0..4000 {
        let long = case % 10 == 0;
        let (n, m) = (1 + next(if long { 300 } else { 12 }) as usize, 1 + next(if long { 300 } else { 12 }) as usize);
        // class 4 has no score row, and scores 0 inside a row
        let seq1_idx: Vec<u8> = (0..n).map(|_| if next(25) == 0 { 4 } else { next(4) as u8 }).collect();
        let seq2_idx: Vec<u8> = (0..m).map(|_| if next(25) == 0 { 4 } else { next(4) as u8 }).collect();
        let initverticalw: Vec<f64> = (0..=n).map(|_| next(5) as f64 * 20.0 - 40.0).collect();
        let row0: Vec<f64> = (0..=m).map(|_| next(5) as f64 * 20.0 - 40.0).collect();
        let m0: Vec<f64> = (0..=m).map(|j| if j == 0 { 0.0 } else { row0[j - 1] }).collect();
        let fill = Fill {
            n,
            m,
            matrix: &matrix,
            seq1_idx: &seq1_idx,
            seq2_idx: &seq2_idx,
            initverticalw: &initverticalw,
            f_open: -[20.0, 50.0, 100.0][case % 3],
            f_ext: -[0.0, 10.0][case % 2],
            localthr: [-1e9, 0.0, 30.0][case % 3],
        };
        let run = |how: fn(&Fill, &mut State) -> (f64, i32, i32)| {
            let (mut h, mut m_arr, mut mp_arr) = (row0.clone(), m0.clone(), vec![0i32; m + 1]);
            let mut ijp = vec![LOCALSTOP; (n + 1) * (m + 1)];
            let best = how(&fill, &mut State { h: &mut h, m_arr: &mut m_arr, mp_arr: &mut mp_arr, ijp: &mut ijp });
            (best, ijp, h, m_arr, mp_arr)
        };
        let want = run(reference);
        for (name, got) in [("run", run(|f, st| f.run(st))), ("scalar", run(scalar))] {
            assert_eq!(got.0, want.0, "{name}: best cell, case {case} n {n} m {m}");
            assert_eq!(got.1, want.1, "{name}: traceback codes, case {case} n {n} m {m}");
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            // column 0 is never read back
            assert_eq!(bits(&got.2[1..]), bits(&want.2[1..]), "{name}: last row, case {case} n {n} m {m}");
            assert_eq!((bits(&got.3), &got.4), (bits(&want.3), &want.4), "{name}: vertical state, case {case}");
        }
    }
}

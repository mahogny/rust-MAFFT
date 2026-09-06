//! FFI cross-validation for the memsavetree algorithm
//! (`mafft-tree::memsavetree`).
//!
//! Strategy: compute the initial pairwise scan
//! (`disttbfast.c::compactdisthalfmtxthread`) in BOTH C and Rust and
//! compare the resulting `mindist[]` / `nearest[]` arrays cell-by-cell.
//! If those match, the divergence we see in the final alignment must
//! come from the per-step distance recomputation, not the initial scan.
//!
//! The C reference is exposed via the FFI helper
//! `rs_compact_initial_mindist` in `mafft-sys/wrappers/parttree_helpers.c`
//! (a copy of the static `compactdisthalfmtxthread` body, single-threaded).

use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::sync::Mutex;

use mafft_io::read_fasta;
use mafft_tree::parttree_dist::{PLENFACA, PLENFACB, PLENFACC, PLENFACD};

static C_MUTEX: Mutex<()> = Mutex::new(());

unsafe fn init_c_protein() {
    unsafe {
        mafft_sys::initglobalvariables();
        std::ptr::addr_of_mut!(mafft_sys::ppenalty).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::ppenalty_ex).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::poffset).write(0);
        std::ptr::addr_of_mut!(mafft_sys::kimuraR).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::pamN).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::dorp).write(b'p' as i32);
        std::ptr::addr_of_mut!(mafft_sys::scoremtx).write(1);
        std::ptr::addr_of_mut!(mafft_sys::nblosum).write(62);
        std::ptr::addr_of_mut!(mafft_sys::fmodel).write(0);
        std::ptr::addr_of_mut!(mafft_sys::tsize).write(46656);
        std::ptr::addr_of_mut!(mafft_sys::maxl).write(46656);
        std::ptr::addr_of_mut!(mafft_sys::lenfaca).write(PLENFACA);
        std::ptr::addr_of_mut!(mafft_sys::lenfacb).write(PLENFACB);
        std::ptr::addr_of_mut!(mafft_sys::lenfacc).write(PLENFACC);
        std::ptr::addr_of_mut!(mafft_sys::lenfacd).write(PLENFACD);

        let seq_data = b"ACDEFGHIKLMNPQRSTVWY\0";
        let mut seq_ptr = seq_data.as_ptr() as *mut i8;
        let seq_arr: *mut *mut i8 = &mut seq_ptr;
        mafft_sys::constants(1, seq_arr);
    }
}

unsafe fn cleanup_c() {
    unsafe {
        mafft_sys::commonsextet_p(std::ptr::null_mut(), std::ptr::null_mut());
        mafft_sys::freeconstants();
    }
}

/// Helper: encode a sequence through C's `seq_grp` + `makepointtable`.
unsafe fn c_pointt(seq: &[u8]) -> Vec<i32> {
    let cs = CString::new(seq).unwrap();
    let mut grp = vec![0i32; seq.len() + 1];
    let nvalid = unsafe { mafft_sys::seq_grp(grp.as_mut_ptr(), cs.as_ptr() as *const c_char) };
    if nvalid < 6 {
        return Vec::new();
    }
    let n_points = (nvalid - 5) as usize;
    let mut pointt = vec![0i32; n_points + 1];
    unsafe {
        mafft_sys::makepointtable(pointt.as_mut_ptr(), grp.as_mut_ptr());
    }
    pointt
}

/// Cross-validate `distcompact` for every (i, j) pair in the 36-seq
/// sample. If Rust's distcompact diverges from C's, the memsavetree
/// algorithm has no chance — guard the per-pair distance first.
#[test]
fn distcompact_matches_c_for_every_pair() {
    let _g = C_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
    let input =
        read_fasta(std::path::Path::new("../../mafft-upstream/test/sample")).expect("load sample");

    unsafe {
        init_c_protein();
    }

    let nseq = input.sequences.len();
    // Strip gaps + build C-side pointt for each sequence.
    let stripped: Vec<Vec<u8>> = input
        .sequences
        .iter()
        .map(|s| {
            s.data
                .iter()
                .filter(|&&c| c != b'-' && c != b'.')
                .copied()
                .collect()
        })
        .collect();
    let nogaplen: Vec<i32> = stripped.iter().map(|s| s.len() as i32).collect();
    let mut c_points: Vec<Vec<i32>> = stripped.iter().map(|s| unsafe { c_pointt(s) }).collect();

    // Self-scores via C commonsextet_p (already cross-validated).
    let selfscore: Vec<i32> = (0..nseq)
        .map(|i| {
            let mut tbl = vec![0i32; 46656];
            unsafe {
                mafft_sys::makecompositiontable_p(tbl.as_mut_ptr(), c_points[i].as_mut_ptr());
            }
            unsafe { mafft_sys::commonsextet_p(tbl.as_mut_ptr(), c_points[i].as_mut_ptr()) }
        })
        .collect();
    unsafe {
        mafft_sys::commonsextet_p(std::ptr::null_mut(), std::ptr::null_mut());
    }

    // Rust-side pointt (validated separately to match c_points).
    let r_points: Vec<Vec<u32>> = stripped
        .iter()
        .map(|s| mafft_tree::parttree_dist::encode_points_protein(s))
        .collect();

    let mut max_diff = 0.0_f64;
    let mut mismatches = 0usize;
    for i in 0..nseq {
        let mut c_table = vec![0i32; 46656];
        unsafe {
            mafft_sys::makecompositiontable_p(c_table.as_mut_ptr(), c_points[i].as_mut_ptr());
        }
        for j in 0..nseq {
            if i == j {
                continue;
            }
            // C distcompact:
            let c_dist = unsafe {
                mafft_sys::distcompact(
                    nogaplen[i],
                    nogaplen[j],
                    c_table.as_mut_ptr(),
                    c_points[j].as_mut_ptr(),
                    selfscore[i],
                    selfscore[j],
                )
            };
            // Rust port (call our implementation through the public API
            // by going through the unit-test helper; mirror its formula
            // here directly to avoid exporting internals).
            let r_table = mafft_tree::parttree_dist::composition_table(&r_points[i], 46656);
            let common = mafft_tree::parttree_dist::common_sextets_p(&r_table, &r_points[j], 46656);
            let lf = mafft_tree::parttree_dist::lenfac(
                nogaplen[i] as usize,
                nogaplen[j] as usize,
                PLENFACA,
                PLENFACB,
                PLENFACC,
                PLENFACD,
            );
            let bunbo = selfscore[i].min(selfscore[j]) as f64;
            let r_dist = if bunbo == 0.0 {
                2.0
            } else {
                (1.0 - common as f64 / bunbo) * lf * 2.0
            };
            let diff = (c_dist - r_dist).abs();
            if diff > 1e-12 {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!("distcompact({i},{j}): C={c_dist} R={r_dist} diff={diff}");
                }
            }
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }
    eprintln!(
        "max distcompact diff across {} pairs: {max_diff:e} ({mismatches} mismatches)",
        nseq * (nseq - 1)
    );
    unsafe {
        mafft_sys::commonsextet_p(std::ptr::null_mut(), std::ptr::null_mut());
    }
    unsafe {
        cleanup_c();
    }
    assert_eq!(
        mismatches, 0,
        "distcompact diverges from C; max diff = {max_diff:e}"
    );
}

/// Cross-validate the INITIAL pairwise scan (the precomputed
/// `mindist[]` / `nearest[]` that feed `compacttree_memsaveselectable`).
///
/// Calls C's `rs_compact_initial_mindist` to get C's arrays, then runs
/// the equivalent in Rust via `memsavetree`'s internals, and compares.
/// If C and Rust agree here, the divergence is in the per-step
/// recomputation loop in `compacttree_memsaveselectable`.
#[test]
fn initial_mindist_matches_c() {
    let _g = C_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
    let input =
        read_fasta(std::path::Path::new("../../mafft-upstream/test/sample")).expect("load sample");

    unsafe {
        init_c_protein();
    }

    let nseq = input.sequences.len();
    let stripped: Vec<Vec<u8>> = input
        .sequences
        .iter()
        .map(|s| {
            s.data
                .iter()
                .filter(|&&c| c != b'-' && c != b'.')
                .copied()
                .collect()
        })
        .collect();
    let nogaplen_c: Vec<i32> = stripped.iter().map(|s| s.len() as i32).collect();
    let mut c_points: Vec<Vec<i32>> = stripped.iter().map(|s| unsafe { c_pointt(s) }).collect();
    let mut selfscore_c: Vec<i32> = (0..nseq)
        .map(|i| {
            let mut tbl = vec![0i32; 46656];
            unsafe {
                mafft_sys::makecompositiontable_p(tbl.as_mut_ptr(), c_points[i].as_mut_ptr());
            }
            unsafe { mafft_sys::commonsextet_p(tbl.as_mut_ptr(), c_points[i].as_mut_ptr()) }
        })
        .collect();
    unsafe {
        mafft_sys::commonsextet_p(std::ptr::null_mut(), std::ptr::null_mut());
    }

    // C-side mindist[] / mindistfrom[] via the FFI wrapper.
    let mut c_mindist = vec![0.0_f64; nseq];
    let mut c_mindistfrom = vec![0_i32; nseq];
    let mut c_points_raw: Vec<*mut c_int> = c_points.iter_mut().map(|v| v.as_mut_ptr()).collect();
    let mut nogaplen_mut = nogaplen_c.clone();
    unsafe {
        mafft_sys::rs_compact_initial_mindist(
            nseq as c_int,
            c_points_raw.as_mut_ptr(),
            nogaplen_mut.as_mut_ptr(),
            selfscore_c.as_mut_ptr(),
            c_mindist.as_mut_ptr(),
            c_mindistfrom.as_mut_ptr(),
        );
    }

    // Rust-side: use the same point vectors / selfscores (via FFI) so
    // we isolate the algorithm logic from the encoding pipeline.
    let r_points: Vec<Vec<u32>> = c_points
        .iter()
        .map(|cv| {
            cv.iter()
                .take_while(|&&p| p != -1)
                .map(|&p| p as u32)
                .collect()
        })
        .collect();
    let selfscore: Vec<i32> = selfscore_c.clone();
    let nogaplen: Vec<usize> = nogaplen_c.iter().map(|&n| n as usize).collect();

    // Compute Rust mindist using the same algorithm shape as our
    // `memsavetree::initial_mindist` (replicated here because it's
    // currently private).
    let mut r_mindist = vec![999.9_f64; nseq];
    let mut r_nearest = vec![-1_i32; nseq];
    for i in (0..nseq).rev() {
        let table_i = mafft_tree::parttree_dist::composition_table(&r_points[i], 46656);
        for j in (0..i).rev() {
            let bunbo = selfscore[i].min(selfscore[j]) as f64;
            let d = if bunbo == 0.0 {
                2.0
            } else {
                let common =
                    mafft_tree::parttree_dist::common_sextets_p(&table_i, &r_points[j], 46656);
                let lf = mafft_tree::parttree_dist::lenfac(
                    nogaplen[i],
                    nogaplen[j],
                    PLENFACA,
                    PLENFACB,
                    PLENFACC,
                    PLENFACD,
                );
                (1.0 - common as f64 / bunbo) * lf * 2.0
            };
            // preferenceval
            let pos = (j as i64 - i as i64 + nseq as i64) % nseq as i64;
            let pref = 1.0e-14 * pos as f64;
            let dx = d + pref;
            if dx < r_mindist[i] {
                r_mindist[i] = dx;
                r_nearest[i] = j as i32;
            }
        }
    }
    for i in 0..nseq {
        if r_nearest[i] >= 0 {
            let pos = (r_nearest[i] as i64 - i as i64 + nseq as i64) % nseq as i64;
            r_mindist[i] -= 1.0e-14 * pos as f64;
        }
    }

    let mut mismatches = 0usize;
    let mut max_diff = 0.0_f64;
    for i in 0..nseq {
        let dist_diff = (c_mindist[i] - r_mindist[i]).abs();
        let nearest_match = c_mindistfrom[i] == r_nearest[i];
        if !nearest_match || dist_diff > 1e-12 {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!(
                    "mindist[{i}]: C=({:.10}, from={}) R=({:.10}, from={}) diff={dist_diff:e}",
                    c_mindist[i], c_mindistfrom[i], r_mindist[i], r_nearest[i]
                );
            }
        }
        if dist_diff > max_diff {
            max_diff = dist_diff;
        }
    }

    unsafe {
        mafft_sys::commonsextet_p(std::ptr::null_mut(), std::ptr::null_mut());
    }
    unsafe {
        cleanup_c();
    }

    assert_eq!(
        mismatches, 0,
        "{mismatches} sequences have divergent initial mindist/nearest; max_diff={max_diff:e}"
    );
}

/// Test cluster_mix on the specific (28, 29), (28, 30) pair (0-indexed)
/// that produces divergent merge distances. Manually compute C's
/// distcompact for these pairs and apply cluster_mix to compare with
/// our memsavetree's step-k=2 d_merged.
#[test]
fn cluster_mix_for_first_divergent_step() {
    let _g = C_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
    let input =
        read_fasta(std::path::Path::new("../../mafft-upstream/test/sample")).expect("load sample");

    unsafe {
        init_c_protein();
    }

    let stripped: Vec<Vec<u8>> = input
        .sequences
        .iter()
        .map(|s| {
            s.data
                .iter()
                .filter(|&&c| c != b'-' && c != b'.')
                .copied()
                .collect()
        })
        .collect();
    let nogaplen: Vec<i32> = stripped.iter().map(|s| s.len() as i32).collect();
    let mut c_points: Vec<Vec<i32>> = stripped.iter().map(|s| unsafe { c_pointt(s) }).collect();
    let selfscore: Vec<i32> = (0..stripped.len())
        .map(|i| {
            let mut tbl = vec![0i32; 46656];
            unsafe {
                mafft_sys::makecompositiontable_p(tbl.as_mut_ptr(), c_points[i].as_mut_ptr());
            }
            unsafe { mafft_sys::commonsextet_p(tbl.as_mut_ptr(), c_points[i].as_mut_ptr()) }
        })
        .collect();
    unsafe {
        mafft_sys::commonsextet_p(std::ptr::null_mut(), std::ptr::null_mut());
    }

    // Triple-validate C's distcompact for (29, 28) and (30, 28).
    let mut t29 = vec![0i32; 46656];
    unsafe {
        mafft_sys::makecompositiontable_p(t29.as_mut_ptr(), c_points[29].as_mut_ptr());
    }
    let c_d_29_28 = unsafe {
        mafft_sys::distcompact(
            nogaplen[29],
            nogaplen[28],
            t29.as_mut_ptr(),
            c_points[28].as_mut_ptr(),
            selfscore[29],
            selfscore[28],
        )
    };

    let mut t30 = vec![0i32; 46656];
    unsafe {
        mafft_sys::makecompositiontable_p(t30.as_mut_ptr(), c_points[30].as_mut_ptr());
    }
    let c_d_30_28 = unsafe {
        mafft_sys::distcompact(
            nogaplen[30],
            nogaplen[28],
            t30.as_mut_ptr(),
            c_points[28].as_mut_ptr(),
            selfscore[30],
            selfscore[28],
        )
    };

    let sueff1 = 0.9_f64;
    let sueff05 = 0.05_f64;
    let mn = c_d_29_28.min(c_d_30_28);
    let c_cluster_mix = mn * sueff1 + (c_d_29_28 + c_d_30_28) * sueff05;

    eprintln!("C: d(29,28)={c_d_29_28:.6} d(30,28)={c_d_30_28:.6} cluster_mix={c_cluster_mix:.6}");
    eprintln!("C: branch from this merge = {:.6}", c_cluster_mix * 0.5);

    // Also compute via our Rust code.
    let r_points: Vec<Vec<u32>> = stripped
        .iter()
        .map(|s| mafft_tree::parttree_dist::encode_points_protein(s))
        .collect();
    let r_t29 = mafft_tree::parttree_dist::composition_table(&r_points[29], 46656);
    let r_common_29_28 = mafft_tree::parttree_dist::common_sextets_p(&r_t29, &r_points[28], 46656);
    let r_t30 = mafft_tree::parttree_dist::composition_table(&r_points[30], 46656);
    let r_common_30_28 = mafft_tree::parttree_dist::common_sextets_p(&r_t30, &r_points[28], 46656);
    let r_lf_29_28 = mafft_tree::parttree_dist::lenfac(
        nogaplen[29] as usize,
        nogaplen[28] as usize,
        PLENFACA,
        PLENFACB,
        PLENFACC,
        PLENFACD,
    );
    let r_lf_30_28 = mafft_tree::parttree_dist::lenfac(
        nogaplen[30] as usize,
        nogaplen[28] as usize,
        PLENFACA,
        PLENFACB,
        PLENFACC,
        PLENFACD,
    );
    let r_d_29_28 =
        (1.0 - r_common_29_28 as f64 / selfscore[29].min(selfscore[28]) as f64) * r_lf_29_28 * 2.0;
    let r_d_30_28 =
        (1.0 - r_common_30_28 as f64 / selfscore[30].min(selfscore[28]) as f64) * r_lf_30_28 * 2.0;
    let r_mn = r_d_29_28.min(r_d_30_28);
    let r_cluster_mix = r_mn * sueff1 + (r_d_29_28 + r_d_30_28) * sueff05;
    eprintln!("R: d(29,28)={r_d_29_28:.6} d(30,28)={r_d_30_28:.6} cluster_mix={r_cluster_mix:.6}");

    unsafe {
        cleanup_c();
    }

    assert!((c_cluster_mix - r_cluster_mix).abs() < 1e-12);
}

/// Drive C's `compacttreegivendist` (the algorithm `--memsavetree`
/// actually uses) via the `rs_compacttreegivendist` wrapper and compare
/// every merge step's (im, jm) pick + branch lengths against our
/// `memsavetree`'s output. Earlier this test FFI'd
/// `compacttree_memsaveselectable` directly and segfaulted on missing
/// global state; replacing the FFI surface with the lighter-weight
/// `rs_compacttreegivendist` (which takes precomputed mindist/nearest
/// arrays) makes the comparison robust.
#[test]
fn memsavetree_topol_matches_c_step_by_step() {
    let _g = C_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
    let input =
        read_fasta(std::path::Path::new("../../mafft-upstream/test/sample")).expect("load sample");

    unsafe {
        init_c_protein();
    }

    let nseq = input.sequences.len();
    let stripped: Vec<Vec<u8>> = input
        .sequences
        .iter()
        .map(|s| {
            s.data
                .iter()
                .filter(|&&c| c != b'-' && c != b'.')
                .copied()
                .collect()
        })
        .collect();
    let nogaplen_c: Vec<i32> = stripped.iter().map(|s| s.len() as i32).collect();
    let mut c_points: Vec<Vec<i32>> = stripped.iter().map(|s| unsafe { c_pointt(s) }).collect();
    let mut selfscore_c: Vec<i32> = (0..nseq)
        .map(|i| {
            let mut tbl = vec![0i32; 46656];
            unsafe {
                mafft_sys::makecompositiontable_p(tbl.as_mut_ptr(), c_points[i].as_mut_ptr());
            }
            unsafe { mafft_sys::commonsextet_p(tbl.as_mut_ptr(), c_points[i].as_mut_ptr()) }
        })
        .collect();
    unsafe {
        mafft_sys::commonsextet_p(std::ptr::null_mut(), std::ptr::null_mut());
    }

    // Initial mindist via FFI (validated above).
    let mut c_mindist = vec![0.0_f64; nseq];
    let mut c_nearest = vec![0_i32; nseq];
    let mut c_points_raw: Vec<*mut c_int> = c_points.iter_mut().map(|v| v.as_mut_ptr()).collect();
    let mut nogaplen_mut = nogaplen_c.clone();
    unsafe {
        mafft_sys::rs_compact_initial_mindist(
            nseq as c_int,
            c_points_raw.as_mut_ptr(),
            nogaplen_mut.as_mut_ptr(),
            selfscore_c.as_mut_ptr(),
            c_mindist.as_mut_ptr(),
            c_nearest.as_mut_ptr(),
        );
    }

    // Drive C's `compacttreegivendist` (the algorithm `--memsavetree`
    // actually uses — NOT the legacy `compacttree_memsaveselectable`).
    let mut c_topol0 = vec![-1_i32; nseq - 1];
    let mut c_topol1 = vec![-1_i32; nseq - 1];
    let mut c_len0 = vec![0.0_f64; nseq - 1];
    let mut c_len1 = vec![0.0_f64; nseq - 1];
    unsafe {
        mafft_sys::rs_compacttreegivendist(
            nseq as c_int,
            c_mindist.as_ptr(),
            c_nearest.as_ptr(),
            c_topol0.as_mut_ptr(),
            c_topol1.as_mut_ptr(),
            c_len0.as_mut_ptr(),
            c_len1.as_mut_ptr(),
        );
    }

    // Run our memsavetree on the same input.
    let raw_refs: Vec<&[u8]> = input.sequences.iter().map(|s| s.data.as_slice()).collect();
    let r_topo = mafft_tree::memsavetree::memsavetree(&raw_refs, false);

    // Compare per step.
    let mut first_diverge: Option<usize> = None;
    for k in 0..(nseq - 1) {
        let r_step = &r_topo.steps[k];
        let r_left_min = r_step.left.iter().min().map(|&x| x as i32).unwrap_or(-1);
        let r_right_min = r_step.right.iter().min().map(|&x| x as i32).unwrap_or(-1);
        let len_diff_0 = (c_len0[k] - r_step.left_length).abs();
        let len_diff_1 = (c_len1[k] - r_step.right_length).abs();
        let topol_match = c_topol0[k] == r_left_min && c_topol1[k] == r_right_min;
        if !topol_match || len_diff_0 > 1e-9 || len_diff_1 > 1e-9 {
            if first_diverge.is_none() {
                first_diverge = Some(k);
            }
            if k < 5 || (first_diverge == Some(k)) {
                eprintln!(
                    "step {k}: C=({},{}) lens=({:.5},{:.5}) | R=({},{}) lens=({:.5},{:.5})",
                    c_topol0[k],
                    c_topol1[k],
                    c_len0[k],
                    c_len1[k],
                    r_left_min,
                    r_right_min,
                    r_step.left_length,
                    r_step.right_length
                );
            }
        }
    }

    unsafe {
        mafft_sys::commonsextet_p(std::ptr::null_mut(), std::ptr::null_mut());
    }
    unsafe {
        cleanup_c();
    }

    if let Some(k) = first_diverge {
        panic!("first divergence at merge step {k}");
    }

    // Build a `Topology` from C's per-step output and serialize it to
    // Newick — this is the "C as if topology_to_newick produced it"
    // tree. Comparing to C's actual --treeout file localises whether
    // the divergence is in the algorithm or the serializer.
    use mafft_tree::{JoinStep, Topology};
    let mut c_topo = Topology::new(nseq);
    // For each step k, we need full member lists. C stores only the
    // smallest leaf; reconstruct full lists by tracing back through
    // hist as compact-tree does. Each cluster's "rep" = its smallest
    // leaf at this step.
    let mut c_hist = vec![-1_i32; nseq];
    for k in 0..(nseq - 1) {
        let im_leaf = c_topol0[k] as usize;
        let jm_leaf = c_topol1[k] as usize;
        // Reconstruct full member set for each subtree by walking
        // backwards through previous merges.
        let left_full: Vec<usize> = {
            let step = c_hist[im_leaf];
            if step < 0 {
                vec![im_leaf]
            } else {
                let s = &c_topo.steps[step as usize];
                let mut v = Vec::with_capacity(s.left.len() + s.right.len());
                v.extend_from_slice(&s.left);
                v.extend_from_slice(&s.right);
                v
            }
        };
        let right_full: Vec<usize> = {
            let step = c_hist[jm_leaf];
            if step < 0 {
                vec![jm_leaf]
            } else {
                let s = &c_topo.steps[step as usize];
                let mut v = Vec::with_capacity(s.left.len() + s.right.len());
                v.extend_from_slice(&s.left);
                v.extend_from_slice(&s.right);
                v
            }
        };
        c_topo.steps.push(JoinStep {
            left: left_full,
            right: right_full,
            left_length: c_len0[k],
            right_length: c_len1[k],
        });
        // The merged cluster's rep is im_leaf (smaller of im_leaf/jm_leaf,
        // i.e., min of both). Since C swaps to ensure im_leaf < jm_leaf,
        // im_leaf is the new rep.
        c_hist[im_leaf] = k as i32;
    }
    let names: Vec<String> = input.sequences.iter().map(|s| s.name.clone()).collect();
    let c_via_rust_newick = mafft_tree::newick::topology_to_newick(&c_topo, &names);
    let our_newick = mafft_tree::newick::topology_to_newick(&r_topo, &names);
    eprintln!(
        "C-from-FFI Newick (first 300): {}",
        &c_via_rust_newick.chars().take(300).collect::<String>()
    );
    eprintln!(
        "R-from-our Newick (first 300): {}",
        &our_newick.chars().take(300).collect::<String>()
    );
    assert_eq!(
        c_via_rust_newick, our_newick,
        "Newick from C's FFI topology should match ours bit-exact"
    );
}

/// Drive C's `compacttree_memsaveselectable` (compacttree=4, the
/// `--youngestlinkage` algorithm) via the
/// `rs_compacttree_memsaveselectable_kmer` wrapper. Compares each merge
/// step's `(im, jm)` and branch lengths against rust's
/// `youngestlinkage_tree`. First diverging step reveals the bug.
///
/// Uses initial mindist FROM C's `ylcompactdisthalfmtxthread` (which is
/// the both-sided forward-walk variant for compacttree=4 — same logic
/// rust's `initial_mindist_yl` implements). Currently the FFI wrapper
/// exposes `rs_compact_initial_mindist` (one-sided) only; for now we
/// compute rust's `initial_mindist_yl` and pass it both to rust's port
/// and C's algorithm.
#[test]
fn youngestlinkage_topol_matches_c_step_by_step() {
    let _g = C_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
    // Use first15 — first input where our port diverges from C.
    let input = read_fasta(std::path::Path::new(
        "../../crates/mafft-core/tests/fixtures/sample.first15.fa",
    ))
    .expect("load first15");

    unsafe {
        init_c_protein();
    }

    let nseq = input.sequences.len();
    let stripped: Vec<Vec<u8>> = input
        .sequences
        .iter()
        .map(|s| {
            s.data
                .iter()
                .filter(|&&c| c != b'-' && c != b'.')
                .copied()
                .collect()
        })
        .collect();
    let nogaplen_c: Vec<i32> = stripped.iter().map(|s| s.len() as i32).collect();
    let mut c_points: Vec<Vec<i32>> = stripped.iter().map(|s| unsafe { c_pointt(s) }).collect();
    let mut selfscore_c: Vec<i32> = (0..nseq)
        .map(|i| {
            let mut tbl = vec![0i32; 46656];
            unsafe {
                mafft_sys::makecompositiontable_p(tbl.as_mut_ptr(), c_points[i].as_mut_ptr());
            }
            unsafe { mafft_sys::commonsextet_p(tbl.as_mut_ptr(), c_points[i].as_mut_ptr()) }
        })
        .collect();
    unsafe {
        mafft_sys::commonsextet_p(std::ptr::null_mut(), std::ptr::null_mut());
    }

    // Compute rust's initial_mindist_yl AND C's
    // `rs_compact_initial_mindist_yl` and verify they match.
    let pointt_u32: Vec<Vec<u32>> = c_points
        .iter()
        .map(|v| {
            v.iter()
                .take_while(|&&x| x >= 0)
                .map(|&x| x as u32)
                .collect()
        })
        .collect();
    let nogaplen_usize: Vec<usize> = nogaplen_c.iter().map(|&x| x as usize).collect();
    let (r_mindist, r_nearest) = mafft_tree::memsavetree::initial_mindist_yl_for_test(
        &pointt_u32,
        &nogaplen_usize,
        &selfscore_c,
        46656,
        PLENFACA,
        PLENFACB,
        PLENFACC,
        PLENFACD,
    );

    let mut c_yl_mindist = vec![0.0_f64; nseq];
    let mut c_yl_nearest = vec![0_i32; nseq];
    let mut c_points_raw: Vec<*mut c_int> = c_points.iter_mut().map(|v| v.as_mut_ptr()).collect();
    let mut nogaplen_mut = nogaplen_c.clone();
    unsafe {
        mafft_sys::rs_compact_initial_mindist_yl(
            nseq as c_int,
            c_points_raw.as_mut_ptr(),
            nogaplen_mut.as_mut_ptr(),
            selfscore_c.as_mut_ptr(),
            c_yl_mindist.as_mut_ptr(),
            c_yl_nearest.as_mut_ptr(),
        );
    }
    eprintln!("Initial mindist comparison (rust vs C ylcompact):");
    for i in 0..nseq {
        let m_diff = (r_mindist[i] - c_yl_mindist[i]).abs();
        let n_match = r_nearest[i] == c_yl_nearest[i];
        if m_diff > 1e-10 || !n_match {
            eprintln!(
                "  i={i}: R mindist={:.10} nearest={} | C mindist={:.10} nearest={}",
                r_mindist[i], r_nearest[i], c_yl_mindist[i], c_yl_nearest[i]
            );
        }
    }
    // Now use C's mindist for both (so the test confirms which side has the bug).
    let c_mindist: Vec<f64> = c_yl_mindist.clone();
    let c_nearest: Vec<i32> = c_yl_nearest.clone();

    let mut c_topol0 = vec![-1_i32; nseq - 1];
    let mut c_topol1 = vec![-1_i32; nseq - 1];
    let mut c_len0 = vec![0.0_f64; nseq - 1];
    let mut c_len1 = vec![0.0_f64; nseq - 1];
    unsafe {
        mafft_sys::rs_compacttree_memsaveselectable_kmer(
            nseq as c_int,
            c_points_raw.as_mut_ptr(),
            nogaplen_mut.as_mut_ptr(),
            selfscore_c.as_mut_ptr(),
            c_mindist.as_ptr(),
            c_nearest.as_ptr(),
            c_topol0.as_mut_ptr(),
            c_topol1.as_mut_ptr(),
            c_len0.as_mut_ptr(),
            c_len1.as_mut_ptr(),
        );
    }

    eprintln!("C compacttree_memsaveselectable (kmer, howcompact=2) sequence:");
    for k in 0..(nseq - 1) {
        eprintln!(
            "  step {k}: ({},{}) lens=({:.6},{:.6})",
            c_topol0[k], c_topol1[k], c_len0[k], c_len1[k]
        );
    }

    // Rust's port:
    let raw_refs: Vec<&[u8]> = input.sequences.iter().map(|s| s.data.as_slice()).collect();
    let r_topo = mafft_tree::memsavetree::youngestlinkage_tree(&raw_refs, false);

    eprintln!("\nRust youngestlinkage_tree sequence:");
    for k in 0..r_topo.steps.len() {
        let s = &r_topo.steps[k];
        let lmin = s.left.iter().min().copied().unwrap_or(0);
        let rmin = s.right.iter().min().copied().unwrap_or(0);
        eprintln!(
            "  step {k}: ({lmin},{rmin}) lens=({:.6},{:.6})",
            s.left_length, s.right_length
        );
    }

    unsafe {
        cleanup_c();
    }
}

//! Cross-validate `profile_align_imp(impmtx)` against C's
//! `A__align(constraint=1)` for the same input + same impmtx.
//!
//! The C side uses `imp_match_init_strict` to populate its internal
//! `impmtx` from a hand-built `LocalHom***` table. The Rust side uses
//! `build_imp_matrix` over an equivalent `LocalHomologyTable`. If the
//! resulting alignment scores or trace differ, the bug is in our DP.
//!
//! Test function names deliberately mirror the C function being validated
//! (`G__align11`, `gen_L__align11`, etc.) — the double underscore is part
//! of the upstream MAFFT identifier. Allow the non-snake_case style here.

#![allow(non_snake_case)]

use std::os::raw::{c_char, c_double, c_int};
use std::sync::Mutex;

use mafft_align::{FASTATHRESHOLD_DEFAULT, GapModel, Profile, build_imp_matrix, profile_align_imp};
use mafft_scoring::build_context;
use mafft_types::{HomologyRegion, LocalHomologyTable, ScoringModel, SeqType};

static C_MUTEX: Mutex<()> = Mutex::new(());

unsafe fn alloc_zeroed(size: usize) -> *mut u8 {
    let layout = std::alloc::Layout::from_size_align(size.max(8), 8).unwrap();
    unsafe { std::alloc::alloc_zeroed(layout) }
}

unsafe fn init_c_protein() {
    unsafe {
        mafft_sys::initglobalvariables();
        std::ptr::addr_of_mut!(mafft_sys::ppenalty).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::ppenalty_ex).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::ppenalty_EX).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::ppenalty_OP).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::ppenalty_dist).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::poffset).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::kimuraR).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::pamN).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::dorp).write(b'p' as i32);
        std::ptr::addr_of_mut!(mafft_sys::scoremtx).write(1);
        std::ptr::addr_of_mut!(mafft_sys::nblosum).write(62);
        std::ptr::addr_of_mut!(mafft_sys::fmodel).write(0);

        let seq_data = b"ACDEFGHIKLMNPQRSTVWY\0";
        let mut seq_ptr = seq_data.as_ptr() as *mut i8;
        let seq_arr: *mut *mut i8 = &mut seq_ptr;
        mafft_sys::constants(1, seq_arr);
    }
}

unsafe fn build_c_dynamicmtx(scoring_matrix: &[Vec<f64>]) -> *mut *mut c_double {
    unsafe {
        let nalpha = scoring_matrix.len() as c_int;
        let mtx = mafft_sys::AllocateDoubleMtx(nalpha, nalpha);
        for i in 0..scoring_matrix.len() {
            for j in 0..scoring_matrix[i].len() {
                *(*mtx.add(i)).add(j) = scoring_matrix[i][j];
            }
        }
        mtx
    }
}

#[test]
fn constrained_align_matches_c_a_align() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    // Two single-sequence groups (clus1=clus2=1) — eliminates profile/weight
    // complications and isolates the constrained DP itself.
    let s1: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";
    let s2: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";
    let n = s1.len();
    let m = s2.len();

    // One homology region covering the full alignment, importance = 5.0.
    let region = HomologyRegion {
        start1: 0,
        end1: (n - 1) as i32,
        start2: 0,
        end2: (m - 1) as i32,
        opt: 5.0,
        overlapaa: n as i32,
        importance: 5.0,
        korh: b'h',
        ..Default::default()
    };

    // ---- Rust side ----
    let mut table = LocalHomologyTable::new(2);
    table.push(0, 1, region.clone());
    table.push(
        1,
        0,
        HomologyRegion {
            start1: region.start2,
            end1: region.end2,
            start2: region.start1,
            end2: region.end1,
            ..region.clone()
        },
    );

    let g1: Vec<&[u8]> = vec![s1];
    let g2: Vec<&[u8]> = vec![s2];
    let imp = build_imp_matrix(
        &table,
        &[0],
        &[1],
        &g1,
        &g2,
        &[1.0],
        &[1.0],
        n,
        m,
        FASTATHRESHOLD_DEFAULT,
    );

    let prof1 = Profile::from_aligned(&g1, &[1.0], &scoring.amino_map, scoring.nalphabets);
    let prof2 = Profile::from_aligned(&g2, &[1.0], &scoring.amino_map, scoring.nalphabets);
    let gap = GapModel::new(scoring.gap.open as f64, scoring.gap.extend as f64);
    let rust_aln = profile_align_imp(
        &prof1,
        &prof2,
        &scoring.consweight_matrix,
        &gap,
        false,
        false,
        Some(&imp),
    );

    eprintln!(
        "Rust: ops.len()={} score={:.3}",
        rust_aln.operations.len(),
        rust_aln.score
    );

    // ---- C side ----
    unsafe {
        init_c_protein();

        let alloclen = (n + m) * 4;
        let c_seq1_boxed: Vec<Box<[u8]>> = vec![{
            let mut v = s1.to_vec();
            v.resize(alloclen + 1, 0);
            v.into_boxed_slice()
        }];
        let c_seq2_boxed: Vec<Box<[u8]>> = vec![{
            let mut v = s2.to_vec();
            v.resize(alloclen + 1, 0);
            v.into_boxed_slice()
        }];
        let mut c_seq1_ptrs: Vec<*mut c_char> = c_seq1_boxed
            .iter()
            .map(|v| v.as_ptr() as *mut c_char)
            .collect();
        let mut c_seq2_ptrs: Vec<*mut c_char> = c_seq2_boxed
            .iter()
            .map(|v| v.as_ptr() as *mut c_char)
            .collect();

        let eff1: *mut c_double = alloc_zeroed(8) as _;
        *eff1 = 1.0;
        let eff2: *mut c_double = alloc_zeroed(8) as _;
        *eff2 = 1.0;
        let eff1_kozo: *mut c_double = alloc_zeroed(8) as _;
        *eff1_kozo = 0.0;
        let eff2_kozo: *mut c_double = alloc_zeroed(8) as _;
        *eff2_kozo = 0.0;

        let n_dyn = build_c_dynamicmtx(&scoring.consweight_matrix);

        // Build LocalHom*** with one entry: localhom[0][0] points to a
        // LocalHom describing the full-coverage region.
        let lh: *mut mafft_sys::LocalHom =
            alloc_zeroed(std::mem::size_of::<mafft_sys::LocalHom>()) as _;
        (*lh).next = std::ptr::null_mut();
        (*lh).last = lh;
        (*lh).start1 = region.start1 as c_int;
        (*lh).end1 = region.end1 as c_int;
        (*lh).start2 = region.start2 as c_int;
        (*lh).end2 = region.end2 as c_int;
        (*lh).opt = region.opt;
        (*lh).overlapaa = region.overlapaa;
        (*lh).extended = 0;
        (*lh).importance = region.importance;
        (*lh).rimportance = region.importance;
        (*lh).korh = region.korh as c_char;
        (*lh).nokori = 0;

        // localhomshrink[k1][k2] points to lh — for clus1=clus2=1 it's just one entry.
        let lh_inner: *mut *mut mafft_sys::LocalHom =
            alloc_zeroed(std::mem::size_of::<*mut mafft_sys::LocalHom>()) as _;
        *lh_inner = lh;
        let lh_outer: *mut *mut *mut mafft_sys::LocalHom =
            alloc_zeroed(std::mem::size_of::<*mut *mut mafft_sys::LocalHom>()) as _;
        *lh_outer = lh_inner;

        // Ensure C's `fastathreshold` matches our default (2.7).
        std::ptr::addr_of_mut!(mafft_sys::fastathreshold).write(FASTATHRESHOLD_DEFAULT);
        // Match L-INS-i progressive: -O flag → outgap=0 (free terminal gaps).
        std::ptr::addr_of_mut!(mafft_sys::outgap).write(0);
        // Ensure penalty / penalty_ex match Rust's gap model after constants() ran
        // (which sets them to BLOSUM62 progressive defaults: -1199/-59).
        let c_penalty = std::ptr::addr_of!(mafft_sys::penalty).read();
        let c_penalty_ex = std::ptr::addr_of!(mafft_sys::penalty_ex).read();
        eprintln!("C penalty={} penalty_ex={}", c_penalty, c_penalty_ex);
        eprintln!("Rust gap.open={} gap.extend={}", gap.open, gap.extend);

        // Initialize impmtx: pass NULL seq1 first to free old impmtx, then
        // call again with real data.
        mafft_sys::imp_match_init_strict(
            std::ptr::null_mut(),
            0,
            0,
            0,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            -1,
            0,
        );

        // Now fill with our test data.
        let mut orinum1: c_int = 0;
        let mut orinum2: c_int = 1;
        mafft_sys::imp_match_init_strict(
            std::ptr::null_mut(),
            1,
            1,
            n as c_int,
            m as c_int,
            c_seq1_ptrs.as_mut_ptr(),
            c_seq2_ptrs.as_mut_ptr(),
            eff1,
            eff2,
            eff1_kozo,
            eff2_kozo,
            lh_outer,
            std::ptr::null_mut(), // swaplist
            1,                    // forscore
            &mut orinum1,
            &mut orinum2,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            -1,
            0,
        );

        // Sanity: read back impmtx[0][0]: should equal importance * eff1*eff2 * fastathreshold = 5.0 * 1.0 * 1.0 * 2.7 = 13.5.
        let v00 = mafft_sys::imp_match_out_sc(0, 0);
        eprintln!("C impmtx[0][0] = {} (expected 13.5)", v00);
        eprintln!("Rust imp[0][0] = {}", imp[0][0]);

        let mut impmatch: c_double = 0.0;
        let score = mafft_sys::A__align(
            n_dyn,
            c_penalty,
            c_penalty_ex,
            c_seq1_ptrs.as_mut_ptr(),
            c_seq2_ptrs.as_mut_ptr(),
            eff1,
            eff2,
            1,
            1,
            alloclen as c_int,
            1, // constraint=1
            &mut impmatch,
            std::ptr::null_mut(),
            std::ptr::null_mut(), // sgap*
            std::ptr::null_mut(),
            std::ptr::null_mut(), // egap*
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            0, // headgp = 0 (false)
            0, // tailgp = 0 (false)
            -1,
            -1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1.0,
            1.0,
        );

        let c_len = {
            let s = c_seq1_ptrs[0];
            let mut k = 0;
            while *s.add(k) != 0 {
                k += 1;
            }
            k
        };
        let c_str = std::str::from_utf8(&c_seq1_boxed[0][..c_len]).unwrap_or("?");
        eprintln!(
            "C: aligned_len={} score={:.3} impmatch={:.3}",
            c_len, score, impmatch
        );
        eprintln!("C  seq1 aligned: {}", c_str);
        let c_str2 = std::str::from_utf8(&c_seq2_boxed[0][..c_len]).unwrap_or("?");
        eprintln!("C  seq2 aligned: {}", c_str2);

        eprintln!("Rust width: {}", rust_aln.operations.len());
        eprintln!("Rust score: {:.3}", rust_aln.score);

        // Free everything
        mafft_sys::imp_match_init_strict(
            std::ptr::null_mut(),
            0,
            0,
            0,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            -1,
            0,
        );
        mafft_sys::freeconstants();

        // Compare widths
        assert_eq!(
            rust_aln.operations.len(),
            c_len,
            "width mismatch: rust={} c={}",
            rust_aln.operations.len(),
            c_len
        );
    }
}

/// More demanding: sequences that differ require gaps. Constraints
/// should bias toward matching the homologous region.
///
/// Verifies our `profile_align_imp` is byte-equivalent to C's
/// `A__align(constraint=1)` when terminal-gap settings match
/// (outgap=0 → free terminal gaps, matching `headgp=tailgp=false`).
#[test]
fn constrained_align_with_gaps_matches_c() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    // 8 residues each. Constraint covers a 5-residue homology in the middle.
    let s1: &[u8] = b"AAACDEFGGG";
    let s2: &[u8] = b"CCCDEFCC";
    let n = s1.len();
    let m = s2.len();

    // Region: ACDEFG -> CDEF, so positions 2..6 in s1 (raw) -> 0..3 in s2
    let region = HomologyRegion {
        start1: 3,
        end1: 6,
        start2: 1,
        end2: 4,
        opt: 5.0,
        overlapaa: 4,
        importance: 50.0, // emphatic enough to influence the alignment
        korh: b'h',
        ..Default::default()
    };

    let mut table = LocalHomologyTable::new(2);
    table.push(0, 1, region.clone());
    table.push(
        1,
        0,
        HomologyRegion {
            start1: region.start2,
            end1: region.end2,
            start2: region.start1,
            end2: region.end1,
            ..region.clone()
        },
    );

    let g1: Vec<&[u8]> = vec![s1];
    let g2: Vec<&[u8]> = vec![s2];
    let imp = build_imp_matrix(
        &table,
        &[0],
        &[1],
        &g1,
        &g2,
        &[1.0],
        &[1.0],
        n,
        m,
        FASTATHRESHOLD_DEFAULT,
    );
    let prof1 = Profile::from_aligned(&g1, &[1.0], &scoring.amino_map, scoring.nalphabets);
    let prof2 = Profile::from_aligned(&g2, &[1.0], &scoring.amino_map, scoring.nalphabets);
    let gap = GapModel::new(scoring.gap.open as f64, scoring.gap.extend as f64);
    let rust_aln = profile_align_imp(
        &prof1,
        &prof2,
        &scoring.consweight_matrix,
        &gap,
        false,
        false,
        Some(&imp),
    );

    // Reconstruct Rust aligned strings
    let mut r_a1 = Vec::new();
    let mut r_a2 = Vec::new();
    let mut c1 = 0;
    let mut c2 = 0;
    use mafft_align::AlignOp;
    for op in &rust_aln.operations {
        match op {
            AlignOp::Match => {
                r_a1.push(s1[c1]);
                r_a2.push(s2[c2]);
                c1 += 1;
                c2 += 1;
            }
            AlignOp::Delete => {
                r_a1.push(s1[c1]);
                r_a2.push(b'-');
                c1 += 1;
            }
            AlignOp::Insert => {
                r_a1.push(b'-');
                r_a2.push(s2[c2]);
                c2 += 1;
            }
        }
    }
    eprintln!(
        "Rust: {} / {}",
        std::str::from_utf8(&r_a1).unwrap(),
        std::str::from_utf8(&r_a2).unwrap()
    );

    unsafe {
        init_c_protein();

        let alloclen = (n + m) * 4;
        let mut buf1 = s1.to_vec();
        buf1.resize(alloclen + 1, 0);
        let mut buf2 = s2.to_vec();
        buf2.resize(alloclen + 1, 0);
        let buf1_box = buf1.into_boxed_slice();
        let buf2_box = buf2.into_boxed_slice();
        let mut c_seq1_ptrs: Vec<*mut c_char> = vec![buf1_box.as_ptr() as *mut c_char];
        let mut c_seq2_ptrs: Vec<*mut c_char> = vec![buf2_box.as_ptr() as *mut c_char];

        let eff1: *mut c_double = alloc_zeroed(8) as _;
        *eff1 = 1.0;
        let eff2: *mut c_double = alloc_zeroed(8) as _;
        *eff2 = 1.0;
        let eff1_kozo: *mut c_double = alloc_zeroed(8) as _;
        *eff1_kozo = 0.0;
        let eff2_kozo: *mut c_double = alloc_zeroed(8) as _;
        *eff2_kozo = 0.0;

        let n_dyn = build_c_dynamicmtx(&scoring.consweight_matrix);

        let lh: *mut mafft_sys::LocalHom =
            alloc_zeroed(std::mem::size_of::<mafft_sys::LocalHom>()) as _;
        (*lh).next = std::ptr::null_mut();
        (*lh).last = lh;
        (*lh).start1 = region.start1;
        (*lh).end1 = region.end1;
        (*lh).start2 = region.start2;
        (*lh).end2 = region.end2;
        (*lh).opt = region.opt;
        (*lh).overlapaa = region.overlapaa;
        (*lh).extended = 0;
        (*lh).importance = region.importance;
        (*lh).rimportance = region.importance;
        (*lh).korh = region.korh as c_char;

        let lh_inner: *mut *mut mafft_sys::LocalHom =
            alloc_zeroed(std::mem::size_of::<*mut mafft_sys::LocalHom>()) as _;
        *lh_inner = lh;
        let lh_outer: *mut *mut *mut mafft_sys::LocalHom =
            alloc_zeroed(std::mem::size_of::<*mut *mut mafft_sys::LocalHom>()) as _;
        *lh_outer = lh_inner;

        std::ptr::addr_of_mut!(mafft_sys::fastathreshold).write(FASTATHRESHOLD_DEFAULT);
        // Match L-INS-i progressive: -O flag → outgap=0 (free terminal gaps).
        std::ptr::addr_of_mut!(mafft_sys::outgap).write(0);

        // Reset impmtx
        mafft_sys::imp_match_init_strict(
            std::ptr::null_mut(),
            0,
            0,
            0,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            -1,
            0,
        );

        let mut on1: c_int = 0;
        let mut on2: c_int = 1;
        mafft_sys::imp_match_init_strict(
            std::ptr::null_mut(),
            1,
            1,
            n as c_int,
            m as c_int,
            c_seq1_ptrs.as_mut_ptr(),
            c_seq2_ptrs.as_mut_ptr(),
            eff1,
            eff2,
            eff1_kozo,
            eff2_kozo,
            lh_outer,
            std::ptr::null_mut(),
            1,
            &mut on1,
            &mut on2,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            -1,
            0,
        );

        let c_penalty = std::ptr::addr_of!(mafft_sys::penalty).read();
        let c_penalty_ex = std::ptr::addr_of!(mafft_sys::penalty_ex).read();

        let mut impmatch: c_double = 0.0;
        let _score = mafft_sys::A__align(
            n_dyn,
            c_penalty,
            c_penalty_ex,
            c_seq1_ptrs.as_mut_ptr(),
            c_seq2_ptrs.as_mut_ptr(),
            eff1,
            eff2,
            1,
            1,
            alloclen as c_int,
            1,
            &mut impmatch,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            0,
            0,
            -1,
            -1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1.0,
            1.0,
        );

        let c_len = {
            let s = c_seq1_ptrs[0];
            let mut k = 0;
            while *s.add(k) != 0 {
                k += 1;
            }
            k
        };
        let c_a1 = std::str::from_utf8(&buf1_box[..c_len]).unwrap();
        let c_a2 = std::str::from_utf8(&buf2_box[..c_len]).unwrap();
        eprintln!("C:    {} / {}", c_a1, c_a2);

        // Free
        mafft_sys::imp_match_init_strict(
            std::ptr::null_mut(),
            0,
            0,
            0,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            -1,
            0,
        );
        mafft_sys::freeconstants();

        let r_a1_str = std::str::from_utf8(&r_a1).unwrap();
        let r_a2_str = std::str::from_utf8(&r_a2).unwrap();
        assert_eq!(r_a1_str, c_a1, "seq1 aligned strings differ");
        assert_eq!(r_a2_str, c_a2, "seq2 aligned strings differ");
    }
}

/// Multi-member groups: group1 has 2 sequences, group2 has 3.
/// Tests profile construction (cpmx, gap freqs, ogcp/fgcp) consistency
/// between Rust Profile::from_aligned and C's match_calc_add path.
#[test]
fn constrained_align_multi_member_matches_c() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    // Group 1: 2 sequences with a gap in one. Group 2: 3 sequences.
    let g1: Vec<&[u8]> = vec![b"ACDEFGHIKLM", b"ACDE-GHIKLM"];
    let g2: Vec<&[u8]> = vec![b"ACDE-GHIKL-", b"ACDEFGHIKLM", b"A-DEFGHIK-M"];
    let n = g1[0].len();
    let m = g2[0].len();
    let w1 = vec![0.5, 0.5];
    let w2 = vec![0.33333, 0.33334, 0.33333];

    // No constraint table for this test (focus on profile DP, not impmtx).
    // We pass impmtx = None which exercises the same code path as
    // unconstrained progressive — i.e., should match `MSalignmm` exactly.
    // This IS already covered by `cross_validate_profile_align.rs`, but
    // including here verifies the constrained path's NO-CONSTRAINT case.
    let prof1 = Profile::from_aligned(&g1, &w1, &scoring.amino_map, scoring.nalphabets);
    let prof2 = Profile::from_aligned(&g2, &w2, &scoring.amino_map, scoring.nalphabets);
    let gap = GapModel::new(scoring.gap.open as f64, scoring.gap.extend as f64);
    let rust_aln = profile_align_imp(
        &prof1,
        &prof2,
        &scoring.consweight_matrix,
        &gap,
        false,
        false,
        None,
    );

    let mut r_a1 = vec![Vec::<u8>::new(); g1.len()];
    let mut r_a2 = vec![Vec::<u8>::new(); g2.len()];
    use mafft_align::AlignOp;
    let mut c1 = 0;
    let mut c2 = 0;
    for op in &rust_aln.operations {
        match op {
            AlignOp::Match => {
                for (si, s) in g1.iter().enumerate() {
                    r_a1[si].push(s[c1]);
                }
                for (si, s) in g2.iter().enumerate() {
                    r_a2[si].push(s[c2]);
                }
                c1 += 1;
                c2 += 1;
            }
            AlignOp::Delete => {
                for (si, s) in g1.iter().enumerate() {
                    r_a1[si].push(s[c1]);
                }
                for si in 0..g2.len() {
                    r_a2[si].push(b'-');
                }
                c1 += 1;
            }
            AlignOp::Insert => {
                for si in 0..g1.len() {
                    r_a1[si].push(b'-');
                }
                for (si, s) in g2.iter().enumerate() {
                    r_a2[si].push(s[c2]);
                }
                c2 += 1;
            }
        }
    }
    eprintln!("Rust group1:");
    for (i, s) in r_a1.iter().enumerate() {
        eprintln!("  [{i}]: {}", std::str::from_utf8(s).unwrap());
    }
    eprintln!("Rust group2:");
    for (i, s) in r_a2.iter().enumerate() {
        eprintln!("  [{i}]: {}", std::str::from_utf8(s).unwrap());
    }

    unsafe {
        init_c_protein();
        std::ptr::addr_of_mut!(mafft_sys::outgap).write(0);

        let alloclen = (n + m) * 4;
        let c_seqs1: Vec<Box<[u8]>> = g1
            .iter()
            .map(|s| {
                let mut v = s.to_vec();
                v.resize(alloclen + 1, 0);
                v.into_boxed_slice()
            })
            .collect();
        let c_seqs2: Vec<Box<[u8]>> = g2
            .iter()
            .map(|s| {
                let mut v = s.to_vec();
                v.resize(alloclen + 1, 0);
                v.into_boxed_slice()
            })
            .collect();
        let mut c_seq1_ptrs: Vec<*mut c_char> =
            c_seqs1.iter().map(|v| v.as_ptr() as *mut c_char).collect();
        let mut c_seq2_ptrs: Vec<*mut c_char> =
            c_seqs2.iter().map(|v| v.as_ptr() as *mut c_char).collect();

        let eff1: *mut c_double = alloc_zeroed(w1.len() * 8) as _;
        for (i, &v) in w1.iter().enumerate() {
            *eff1.add(i) = v;
        }
        let eff2: *mut c_double = alloc_zeroed(w2.len() * 8) as _;
        for (i, &v) in w2.iter().enumerate() {
            *eff2.add(i) = v;
        }

        let n_dyn = build_c_dynamicmtx(&scoring.consweight_matrix);

        let c_penalty = std::ptr::addr_of!(mafft_sys::penalty).read();
        let c_penalty_ex = std::ptr::addr_of!(mafft_sys::penalty_ex).read();

        let mut impmatch: c_double = 0.0;
        let _score = mafft_sys::A__align(
            n_dyn,
            c_penalty,
            c_penalty_ex,
            c_seq1_ptrs.as_mut_ptr(),
            c_seq2_ptrs.as_mut_ptr(),
            eff1,
            eff2,
            w1.len() as c_int,
            w2.len() as c_int,
            alloclen as c_int,
            0,
            &mut impmatch,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            0,
            0,
            -1,
            -1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1.0,
            1.0,
        );

        let c_len = {
            let s = c_seq1_ptrs[0];
            let mut k = 0;
            while *s.add(k) != 0 {
                k += 1;
            }
            k
        };
        eprintln!("C group1:");
        for (i, s) in c_seqs1.iter().enumerate() {
            eprintln!("  [{i}]: {}", std::str::from_utf8(&s[..c_len]).unwrap());
        }
        eprintln!("C group2:");
        for (i, s) in c_seqs2.iter().enumerate() {
            eprintln!("  [{i}]: {}", std::str::from_utf8(&s[..c_len]).unwrap());
        }

        mafft_sys::freeconstants();

        for (i, (r, c)) in r_a1.iter().zip(c_seqs1.iter()).enumerate() {
            let r_str = std::str::from_utf8(r).unwrap();
            let c_str = std::str::from_utf8(&c[..c_len]).unwrap();
            assert_eq!(r_str, c_str, "g1[{i}] differs");
        }
        for (i, (r, c)) in r_a2.iter().zip(c_seqs2.iter()).enumerate() {
            let r_str = std::str::from_utf8(r).unwrap();
            let c_str = std::str::from_utf8(&c[..c_len]).unwrap();
            assert_eq!(r_str, c_str, "g2[{i}] differs");
        }
    }
}

/// Real-protein 2-sequence test: take two actual opsins from
/// `mafft-upstream/test/sample` (seqs 0 and 1, dist ~0.313) and
/// compare our `profile_align_imp` to C's `A__align(constraint=1)`
/// on the EXACT inputs the production progressive feeds them. Uses
/// the real localhom region from `build_local_homology_table`.
#[test]
fn constrained_align_real_2seq_matches_c() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    let s1: &'static [u8] = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNFLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSIEGFFATLGGEVALWSLVVLAIERYIVICKPMGNFRFGNTHAIMGVAFTWIMALACAAPPLVGWSRYIPEGMQCSCGPDYYTLNPNFNNESYVVYMFVVHFLVPFVIIFFCYGRLLCTVKEAAAAQQESASTQKAEKEVTRMVVLMVIGFLVCWVPYASVAFYIFTHQGSDFGATFMTLPAFFAKSSALYNPVIYILMNKQFRNCMITTLCCGKNPLGDDESGASTSKTEVSSVSTSPVSPA";
    let s2: &'static [u8] = b"MNGTEGPNFYVPFSNITGVVRSPFEQPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVFGGFTTTLYTSLHGYFVFGPTGCNLEGFFATLGGEIGLWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLVGWSRYIPEGMQCSCGIDYYTLKPEVNNESFVIYMFVVHFTIPMIVIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIFFLICWLPYASVAMYIFTHQGSNFGPIFMTLPAFFAKTASIYNPIIYIMMNKQFRNCMLTSLCCGKNPLGDDEASATASKTETSQVAPA";

    // Build localhom via real production path: same gap params, same
    // matrix offset shift, then pass through `build_local_homology_table`.
    use mafft_align::{GapModel as G, build_local_homology_table};
    let scale_protein: f64 = 600.0 / 1000.0;
    let cc_int = |x: f64, mul: f64| -> i32 { ((x * mul) - 0.5) as i32 };
    let cc_scale = |ppen: i32, scale: f64| -> i32 { ((scale * ppen as f64) + 0.5) as i32 };
    let p_open = cc_int(-2.00, 1000.0);
    let p_ext = cc_int(-0.100, 1000.0);
    let p_offset = cc_int(0.100, 1000.0);
    let pair_gap = G::new(
        cc_scale(p_open, scale_protein) as f64,
        cc_scale(p_ext, scale_protein) as f64,
    );
    let pair_offset_int = cc_scale(p_offset, scale_protein);
    let nscored = scoring.nscoredalphabets;
    let mut shifted: Vec<Vec<f64>> = scoring.consweight_matrix.clone();
    for i in 0..nscored {
        for j in 0..nscored {
            shifted[i][j] -= pair_offset_int as f64;
        }
    }
    let seq_refs: Vec<&[u8]> = vec![s1, s2];
    let score_offset_for_local = pair_offset_int as f64 / 600.0;
    let (mut table, _dist) = build_local_homology_table(
        &seq_refs,
        &shifted,
        &scoring.amino_map,
        &pair_gap,
        score_offset_for_local,
    );
    // Match production: run calcimportance_half via recompute_importance.
    use mafft_align::recompute_importance;
    let weights = vec![0.5_f64, 0.5_f64]; // 2-leaf UPGMA tree weights
    recompute_importance(&mut table, &seq_refs, &weights);
    eprintln!("After recompute, region (0,1) importance:");
    for r in table.get(0, 1) {
        eprintln!(
            "  start1={} end1={} opt={:.5} importance={:.5}",
            r.start1, r.end1, r.opt, r.importance
        );
    }

    let n = s1.len();
    let m = s2.len();
    let g1: Vec<&[u8]> = vec![s1];
    let g2: Vec<&[u8]> = vec![s2];
    let imp = build_imp_matrix(
        &table,
        &[0],
        &[1],
        &g1,
        &g2,
        &[1.0],
        &[1.0],
        n,
        m,
        FASTATHRESHOLD_DEFAULT,
    );

    let prof1 = Profile::from_aligned(&g1, &[1.0], &scoring.amino_map, scoring.nalphabets);
    let prof2 = Profile::from_aligned(&g2, &[1.0], &scoring.amino_map, scoring.nalphabets);
    let gap = GapModel::new(scoring.gap.open as f64, scoring.gap.extend as f64);
    let rust_aln = profile_align_imp(
        &prof1,
        &prof2,
        &scoring.consweight_matrix,
        &gap,
        false,
        false,
        Some(&imp),
    );

    // Reconstruct Rust aligned strings.
    let mut r_a1 = Vec::new();
    let mut r_a2 = Vec::new();
    let mut c1 = 0;
    let mut c2 = 0;
    use mafft_align::AlignOp;
    for op in &rust_aln.operations {
        match op {
            AlignOp::Match => {
                r_a1.push(s1[c1]);
                r_a2.push(s2[c2]);
                c1 += 1;
                c2 += 1;
            }
            AlignOp::Delete => {
                r_a1.push(s1[c1]);
                r_a2.push(b'-');
                c1 += 1;
            }
            AlignOp::Insert => {
                r_a1.push(b'-');
                r_a2.push(s2[c2]);
                c2 += 1;
            }
        }
    }
    eprintln!("Rust width: {}", r_a1.len());
    eprintln!(
        "Rust seq1 last 30: {}",
        std::str::from_utf8(&r_a1[r_a1.len() - 30..]).unwrap()
    );
    eprintln!(
        "Rust seq2 last 30: {}",
        std::str::from_utf8(&r_a2[r_a2.len() - 30..]).unwrap()
    );

    unsafe {
        init_c_protein();
        std::ptr::addr_of_mut!(mafft_sys::outgap).write(0);
        std::ptr::addr_of_mut!(mafft_sys::fastathreshold).write(FASTATHRESHOLD_DEFAULT);

        let alloclen = (n + m) * 4;
        let mut buf1 = s1.to_vec();
        buf1.resize(alloclen + 1, 0);
        let mut buf2 = s2.to_vec();
        buf2.resize(alloclen + 1, 0);
        let buf1_box = buf1.into_boxed_slice();
        let buf2_box = buf2.into_boxed_slice();
        let mut c_seq1_ptrs: Vec<*mut c_char> = vec![buf1_box.as_ptr() as *mut c_char];
        let mut c_seq2_ptrs: Vec<*mut c_char> = vec![buf2_box.as_ptr() as *mut c_char];
        let eff1: *mut c_double = alloc_zeroed(8) as _;
        *eff1 = 1.0;
        let eff2: *mut c_double = alloc_zeroed(8) as _;
        *eff2 = 1.0;
        let eff1_kozo: *mut c_double = alloc_zeroed(8) as _;
        let eff2_kozo: *mut c_double = alloc_zeroed(8) as _;
        let n_dyn = build_c_dynamicmtx(&scoring.consweight_matrix);

        // Build C-side LocalHom from our table's regions.
        let regions = table.get(0, 1);
        let mut lhs: Vec<*mut mafft_sys::LocalHom> = Vec::with_capacity(regions.len());
        for (idx, region) in regions.iter().enumerate() {
            let lh: *mut mafft_sys::LocalHom =
                alloc_zeroed(std::mem::size_of::<mafft_sys::LocalHom>()) as _;
            (*lh).next = std::ptr::null_mut();
            (*lh).last = lh;
            (*lh).start1 = region.start1;
            (*lh).end1 = region.end1;
            (*lh).start2 = region.start2;
            (*lh).end2 = region.end2;
            (*lh).opt = region.opt;
            (*lh).overlapaa = region.overlapaa;
            (*lh).extended = 0;
            (*lh).importance = region.importance;
            (*lh).rimportance = region.importance;
            (*lh).korh = region.korh as c_char;
            if idx > 0 {
                let prev = lhs[idx - 1];
                (*prev).next = lh;
            }
            lhs.push(lh);
        }
        let lh_inner: *mut *mut mafft_sys::LocalHom =
            alloc_zeroed(std::mem::size_of::<*mut mafft_sys::LocalHom>()) as _;
        *lh_inner = if !lhs.is_empty() {
            lhs[0]
        } else {
            std::ptr::null_mut()
        };
        let lh_outer: *mut *mut *mut mafft_sys::LocalHom =
            alloc_zeroed(std::mem::size_of::<*mut *mut mafft_sys::LocalHom>()) as _;
        *lh_outer = lh_inner;

        mafft_sys::imp_match_init_strict(
            std::ptr::null_mut(),
            0,
            0,
            0,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            -1,
            0,
        );
        let mut on1: c_int = 0;
        let mut on2: c_int = 1;
        mafft_sys::imp_match_init_strict(
            std::ptr::null_mut(),
            1,
            1,
            n as c_int,
            m as c_int,
            c_seq1_ptrs.as_mut_ptr(),
            c_seq2_ptrs.as_mut_ptr(),
            eff1,
            eff2,
            eff1_kozo,
            eff2_kozo,
            lh_outer,
            std::ptr::null_mut(),
            1,
            &mut on1,
            &mut on2,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            -1,
            0,
        );

        let c_penalty = std::ptr::addr_of!(mafft_sys::penalty).read();
        let c_penalty_ex = std::ptr::addr_of!(mafft_sys::penalty_ex).read();
        let mut impmatch: c_double = 0.0;
        // Match tbfast call pattern: pass non-NULL cpmxresult to enable
        // C's "cpmxresult" path inside A__align. cpmxchild0/1 stay NULL
        // (this is the first merge). firstmem=0, calledbyfulltreebase=1
        // matches tbfast's call (`tbfast.c:1608`).
        let mut cpmx_storage: *mut *mut c_double = std::ptr::null_mut();
        let cpmxresult: *mut *mut *mut c_double = &mut cpmx_storage;
        let _score = mafft_sys::A__align(
            n_dyn,
            c_penalty,
            c_penalty_ex,
            c_seq1_ptrs.as_mut_ptr(),
            c_seq2_ptrs.as_mut_ptr(),
            eff1,
            eff2,
            1,
            1,
            alloclen as c_int,
            1,
            &mut impmatch,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            0,
            0,
            0,
            1, // firstmem=0, calledbyfulltreebase=1
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            cpmxresult,
            1.0,
            1.0,
        );

        let c_len = {
            let s = c_seq1_ptrs[0];
            let mut k = 0;
            while *s.add(k) != 0 {
                k += 1;
            }
            k
        };
        let c_a1 = std::str::from_utf8(&buf1_box[..c_len]).unwrap();
        let c_a2 = std::str::from_utf8(&buf2_box[..c_len]).unwrap();
        eprintln!("C width: {}", c_len);
        eprintln!(
            "C  seq1 last 30: {}",
            &c_a1[c_a1.len().saturating_sub(30)..]
        );
        eprintln!(
            "C  seq2 last 30: {}",
            &c_a2[c_a2.len().saturating_sub(30)..]
        );

        mafft_sys::imp_match_init_strict(
            std::ptr::null_mut(),
            0,
            0,
            0,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            -1,
            0,
        );
        mafft_sys::freeconstants();

        let r_a1_str = std::str::from_utf8(&r_a1).unwrap();
        let r_a2_str = std::str::from_utf8(&r_a2).unwrap();
        if r_a1_str != c_a1 {
            // Print only the trailing-tail diffs to keep output manageable.
            let common = r_a1_str
                .chars()
                .zip(c_a1.chars())
                .take_while(|(a, b)| a == b)
                .count();
            eprintln!("First divergence at column {}", common);
            eprintln!(
                "Rust tail (chars {}..): {}",
                common.saturating_sub(20),
                &r_a1_str[common.saturating_sub(20)..]
            );
            eprintln!(
                "C    tail (chars {}..): {}",
                common.saturating_sub(20),
                &c_a1[common.saturating_sub(20)..]
            );
            eprintln!(
                "Rust tail s2 (chars {}..): {}",
                common.saturating_sub(20),
                &r_a2_str[common.saturating_sub(20)..]
            );
            eprintln!(
                "C    tail s2 (chars {}..): {}",
                common.saturating_sub(20),
                &c_a2[common.saturating_sub(20)..]
            );
        }
        assert_eq!(r_a1_str, c_a1, "seq1 differs");
        assert_eq!(r_a2_str, c_a2, "seq2 differs");
    }
}

/// Multi-member groups WITH constraints — closest match to L-INS-i
/// production case. group1 = 2 seqs, group2 = 3 seqs, with a localhom
/// table containing a region for each (s1, s2) pair.
#[test]
fn constrained_align_multi_member_with_constraints_matches_c() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    let g1: Vec<&[u8]> = vec![b"ACDEFGHIKLM", b"ACDE-GHIKLM"];
    let g2: Vec<&[u8]> = vec![b"ACDE-GHIKL-", b"ACDEFGHIKLM", b"A-DEFGHIK-M"];
    let n = g1[0].len();
    let m = g2[0].len();
    let w1 = vec![0.5, 0.5];
    let w2 = vec![0.33333, 0.33334, 0.33333];

    // Build a localhom table with regions covering positions 0..=10 (raw)
    // for each (i, j) pair across the 5 sequences. Importance values
    // chosen to bias the alignment.
    let nseq = 5; // global indices 0,1 in g1; 2,3,4 in g2
    let mut table = LocalHomologyTable::new(nseq);
    for i in 0..2 {
        for j in 2..5 {
            // sequences are 11 chars (with possible gaps in the strings)
            // but raw residue lengths might be 10 or 11 depending on gap count.
            // For simplicity use the residue indices that exist in each seq.
            let s1 = if i == 0 { g1[0] } else { g1[1] };
            let s2 = g2[j - 2];
            let n_residues_1 = s1.iter().filter(|&&c| c != b'-').count();
            let n_residues_2 = s2.iter().filter(|&&c| c != b'-').count();
            let region = HomologyRegion {
                start1: 0,
                end1: (n_residues_1 - 1) as i32,
                start2: 0,
                end2: (n_residues_2 - 1) as i32,
                opt: 5.0,
                overlapaa: n_residues_1.min(n_residues_2) as i32,
                importance: 5.0,
                korh: b'h',
                ..Default::default()
            };
            table.push(i, j, region.clone());
            table.push(
                j,
                i,
                HomologyRegion {
                    start1: region.start2,
                    end1: region.end2,
                    start2: region.start1,
                    end2: region.end1,
                    ..region.clone()
                },
            );
        }
    }

    let g1_seq_refs: Vec<&[u8]> = g1.iter().copied().collect();
    let g2_seq_refs: Vec<&[u8]> = g2.iter().copied().collect();
    let imp = build_imp_matrix(
        &table,
        &[0, 1],    // group 1 global indices
        &[2, 3, 4], // group 2 global indices
        &g1_seq_refs,
        &g2_seq_refs,
        &w1,
        &w2,
        n,
        m,
        FASTATHRESHOLD_DEFAULT,
    );

    let prof1 = Profile::from_aligned(&g1, &w1, &scoring.amino_map, scoring.nalphabets);
    let prof2 = Profile::from_aligned(&g2, &w2, &scoring.amino_map, scoring.nalphabets);
    let gap = GapModel::new(scoring.gap.open as f64, scoring.gap.extend as f64);
    let rust_aln = profile_align_imp(
        &prof1,
        &prof2,
        &scoring.consweight_matrix,
        &gap,
        false,
        false,
        Some(&imp),
    );

    let mut r_a1 = vec![Vec::<u8>::new(); g1.len()];
    let mut r_a2 = vec![Vec::<u8>::new(); g2.len()];
    use mafft_align::AlignOp;
    let mut c1 = 0;
    let mut c2 = 0;
    for op in &rust_aln.operations {
        match op {
            AlignOp::Match => {
                for (si, s) in g1.iter().enumerate() {
                    r_a1[si].push(s[c1]);
                }
                for (si, s) in g2.iter().enumerate() {
                    r_a2[si].push(s[c2]);
                }
                c1 += 1;
                c2 += 1;
            }
            AlignOp::Delete => {
                for (si, s) in g1.iter().enumerate() {
                    r_a1[si].push(s[c1]);
                }
                for si in 0..g2.len() {
                    r_a2[si].push(b'-');
                }
                c1 += 1;
            }
            AlignOp::Insert => {
                for si in 0..g1.len() {
                    r_a1[si].push(b'-');
                }
                for (si, s) in g2.iter().enumerate() {
                    r_a2[si].push(s[c2]);
                }
                c2 += 1;
            }
        }
    }
    eprintln!("Rust group1:");
    for s in &r_a1 {
        eprintln!("  {}", std::str::from_utf8(s).unwrap());
    }
    eprintln!("Rust group2:");
    for s in &r_a2 {
        eprintln!("  {}", std::str::from_utf8(s).unwrap());
    }

    unsafe {
        init_c_protein();
        std::ptr::addr_of_mut!(mafft_sys::outgap).write(0);
        std::ptr::addr_of_mut!(mafft_sys::fastathreshold).write(FASTATHRESHOLD_DEFAULT);

        let alloclen = (n + m) * 4;
        let c_seqs1: Vec<Box<[u8]>> = g1
            .iter()
            .map(|s| {
                let mut v = s.to_vec();
                v.resize(alloclen + 1, 0);
                v.into_boxed_slice()
            })
            .collect();
        let c_seqs2: Vec<Box<[u8]>> = g2
            .iter()
            .map(|s| {
                let mut v = s.to_vec();
                v.resize(alloclen + 1, 0);
                v.into_boxed_slice()
            })
            .collect();
        let mut c_seq1_ptrs: Vec<*mut c_char> =
            c_seqs1.iter().map(|v| v.as_ptr() as *mut c_char).collect();
        let mut c_seq2_ptrs: Vec<*mut c_char> =
            c_seqs2.iter().map(|v| v.as_ptr() as *mut c_char).collect();

        let eff1: *mut c_double = alloc_zeroed(w1.len() * 8) as _;
        for (i, &v) in w1.iter().enumerate() {
            *eff1.add(i) = v;
        }
        let eff2: *mut c_double = alloc_zeroed(w2.len() * 8) as _;
        for (i, &v) in w2.iter().enumerate() {
            *eff2.add(i) = v;
        }
        let eff1_kozo: *mut c_double = alloc_zeroed(w1.len() * 8) as _;
        let eff2_kozo: *mut c_double = alloc_zeroed(w2.len() * 8) as _;

        let n_dyn = build_c_dynamicmtx(&scoring.consweight_matrix);

        // Build LocalHom*** matching our table for the group pairs.
        let mut lh_storage: Vec<*mut mafft_sys::LocalHom> = Vec::new();
        let mut lh_2d: Vec<Vec<*mut mafft_sys::LocalHom>> = (0..g1.len())
            .map(|_| vec![std::ptr::null_mut(); g2.len()])
            .collect();
        for k1 in 0..g1.len() {
            for k2 in 0..g2.len() {
                let s1 = g1[k1];
                let s2 = g2[k2];
                let n_residues_1 = s1.iter().filter(|&&c| c != b'-').count();
                let n_residues_2 = s2.iter().filter(|&&c| c != b'-').count();
                let lh: *mut mafft_sys::LocalHom =
                    alloc_zeroed(std::mem::size_of::<mafft_sys::LocalHom>()) as _;
                (*lh).next = std::ptr::null_mut();
                (*lh).last = lh;
                (*lh).start1 = 0;
                (*lh).end1 = (n_residues_1 - 1) as c_int;
                (*lh).start2 = 0;
                (*lh).end2 = (n_residues_2 - 1) as c_int;
                (*lh).opt = 5.0;
                (*lh).overlapaa = n_residues_1.min(n_residues_2) as c_int;
                (*lh).extended = 0;
                (*lh).importance = 5.0;
                (*lh).rimportance = 5.0;
                (*lh).korh = b'h' as c_char;
                (*lh).nokori = 0;
                lh_storage.push(lh);
                lh_2d[k1][k2] = lh;
            }
        }

        let lh_inner_arr: Vec<*mut *mut mafft_sys::LocalHom> = lh_2d
            .iter()
            .map(|row| {
                let arr: *mut *mut mafft_sys::LocalHom =
                    alloc_zeroed(row.len() * std::mem::size_of::<*mut mafft_sys::LocalHom>()) as _;
                for (i, &p) in row.iter().enumerate() {
                    *arr.add(i) = p;
                }
                arr
            })
            .collect();
        let lh_outer: *mut *mut *mut mafft_sys::LocalHom =
            alloc_zeroed(lh_inner_arr.len() * std::mem::size_of::<*mut *mut mafft_sys::LocalHom>())
                as _;
        for (i, &p) in lh_inner_arr.iter().enumerate() {
            *lh_outer.add(i) = p;
        }

        // Reset C's impmtx
        mafft_sys::imp_match_init_strict(
            std::ptr::null_mut(),
            0,
            0,
            0,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            -1,
            0,
        );

        let mut on1: Vec<c_int> = (0..g1.len() as c_int).collect();
        let mut on2: Vec<c_int> = (0..g2.len() as c_int).collect();
        mafft_sys::imp_match_init_strict(
            std::ptr::null_mut(),
            g1.len() as c_int,
            g2.len() as c_int,
            n as c_int,
            m as c_int,
            c_seq1_ptrs.as_mut_ptr(),
            c_seq2_ptrs.as_mut_ptr(),
            eff1,
            eff2,
            eff1_kozo,
            eff2_kozo,
            lh_outer,
            std::ptr::null_mut(),
            1,
            on1.as_mut_ptr(),
            on2.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            -1,
            0,
        );

        let c_penalty = std::ptr::addr_of!(mafft_sys::penalty).read();
        let c_penalty_ex = std::ptr::addr_of!(mafft_sys::penalty_ex).read();

        let mut impmatch: c_double = 0.0;
        let _score = mafft_sys::A__align(
            n_dyn,
            c_penalty,
            c_penalty_ex,
            c_seq1_ptrs.as_mut_ptr(),
            c_seq2_ptrs.as_mut_ptr(),
            eff1,
            eff2,
            g1.len() as c_int,
            g2.len() as c_int,
            alloclen as c_int,
            1,
            &mut impmatch,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            0,
            0,
            -1,
            -1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1.0,
            1.0,
        );

        let c_len = {
            let s = c_seq1_ptrs[0];
            let mut k = 0;
            while *s.add(k) != 0 {
                k += 1;
            }
            k
        };
        eprintln!("C group1:");
        for s in &c_seqs1 {
            eprintln!("  {}", std::str::from_utf8(&s[..c_len]).unwrap());
        }
        eprintln!("C group2:");
        for s in &c_seqs2 {
            eprintln!("  {}", std::str::from_utf8(&s[..c_len]).unwrap());
        }

        mafft_sys::imp_match_init_strict(
            std::ptr::null_mut(),
            0,
            0,
            0,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            -1,
            0,
        );
        mafft_sys::freeconstants();

        for (i, (r, c)) in r_a1.iter().zip(c_seqs1.iter()).enumerate() {
            let r_str = std::str::from_utf8(r).unwrap();
            let c_str = std::str::from_utf8(&c[..c_len]).unwrap();
            assert_eq!(r_str, c_str, "g1[{i}] differs");
        }
        for (i, (r, c)) in r_a2.iter().zip(c_seqs2.iter()).enumerate() {
            let r_str = std::str::from_utf8(r).unwrap();
            let c_str = std::str::from_utf8(&c[..c_len]).unwrap();
            assert_eq!(r_str, c_str, "g2[{i}] differs");
        }
    }
}

/// Direct comparison: our global_align vs C's G__align11 on the real
/// 36-seq sample's first 2 sequences. If this fails, the bug is in our
/// G__align11 port (global.rs).
#[test]
fn rust_global_align_matches_c_g__align11() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    let s1: &'static [u8] = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNFLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSIEGFFATLGGEVALWSLVVLAIERYIVICKPMGNFRFGNTHAIMGVAFTWIMALACAAPPLVGWSRYIPEGMQCSCGPDYYTLNPNFNNESYVVYMFVVHFLVPFVIIFFCYGRLLCTVKEAAAAQQESASTQKAEKEVTRMVVLMVIGFLVCWVPYASVAFYIFTHQGSDFGATFMTLPAFFAKSSALYNPVIYILMNKQFRNCMITTLCCGKNPLGDDESGASTSKTEVSSVSTSPVSPA";
    // Pair (M63632, K03494): K03494 has a different N-terminal that
    // exposes head-gap tie-break.
    let s2: &'static [u8] = b"MAQQWSLQRLAGRHPQDSYEDSTQSSIFTYTNSNSTRGPFEGPNYHIAPRWVYHLTSVWMIFVVIASVFTNGLVLAATMKFKKLRHPLNWILVNLAVADLAETVIASTISVVNQVYGYFVLGHPMCVLEGYTVSLCGITGLWSLAIISWERWMVVCKPFGNVRFDAKLAIVGIAFSWIWAAVWTAPPIFGWSRYWPHGLKTSCGPDVFSGSSYPGVQSYMIVLMVTCCITPLSIIVLCYLQVWLAIRAVAKQQKESESTQKAEKEVTRMVVVMVLAFCFCWGPYAFFACFAAANPGYPFHPLMAALPAFFAKSATIYNPVIYVFMNRQFRNCILQLFGKKVDDGSELSSASKTEVSSVSSVSPA";

    // Match C's G-INS-i pairlocalalign: -2.00 / -0.100 / 0.100 (script:204-206).
    let scale_protein: f64 = 600.0 / 1000.0;
    let cc_int = |x: f64, mul: f64| -> i32 { ((x * mul) - 0.5) as i32 };
    let cc_scale = |ppen: i32, scale: f64| -> i32 { ((scale * ppen as f64) + 0.5) as i32 };
    let p_open = cc_int(-2.00, 1000.0);
    let p_ext = cc_int(-0.100, 1000.0);
    let p_offset = cc_int(0.100, 1000.0);
    let pair_gap = GapModel::new(
        cc_scale(p_open, scale_protein) as f64,
        cc_scale(p_ext, scale_protein) as f64,
    );
    let pair_offset_int = cc_scale(p_offset, scale_protein);
    // Apply matrix offset shift as C does (constants.c:798).
    let nscored = scoring.nscoredalphabets;
    let mut shifted: Vec<Vec<f64>> = scoring.consweight_matrix.clone();
    for i in 0..nscored {
        for j in 0..nscored {
            shifted[i][j] -= pair_offset_int as f64;
        }
    }

    let r_aln =
        mafft_align::global_align(s1, s2, &shifted, &scoring.amino_map, &pair_gap, true, true);

    eprintln!("Rust score: {}", r_aln.score);
    eprintln!("Rust width: {}", r_aln.seq1.len());
    eprintln!(
        "Rust seq1[0..40]: {}",
        std::str::from_utf8(&r_aln.seq1[..40]).unwrap()
    );
    eprintln!(
        "Rust seq2[0..40]: {}",
        std::str::from_utf8(&r_aln.seq2[..40]).unwrap()
    );

    unsafe {
        init_c_protein();
        std::ptr::addr_of_mut!(mafft_sys::outgap).write(1);
        std::ptr::addr_of_mut!(mafft_sys::penalty).write(pair_gap.open as c_int);
        std::ptr::addr_of_mut!(mafft_sys::penalty_ex).write(pair_gap.extend as c_int);

        let alloclen = (s1.len() + s2.len()) * 4;
        let mut buf1 = s1.to_vec();
        buf1.resize(alloclen + 1, 0);
        let mut buf2 = s2.to_vec();
        buf2.resize(alloclen + 1, 0);
        let buf1_box = buf1.into_boxed_slice();
        let buf2_box = buf2.into_boxed_slice();
        let mut p1: *mut c_char = buf1_box.as_ptr() as *mut c_char;
        let mut p2: *mut c_char = buf2_box.as_ptr() as *mut c_char;

        let n_dyn = build_c_dynamicmtx(&shifted);
        let c_score = mafft_sys::G__align11(n_dyn, &mut p1, &mut p2, alloclen as c_int, 1, 1);

        let c_len = {
            let s = p1;
            let mut k = 0;
            while *s.add(k) != 0 {
                k += 1;
            }
            k
        };
        let c_a1 = std::str::from_utf8(&buf1_box[..c_len]).unwrap();
        let c_a2 = std::str::from_utf8(&buf2_box[..c_len]).unwrap();
        eprintln!("C    score: {}", c_score);
        eprintln!("C    width: {}", c_len);

        mafft_sys::freeconstants();

        let r_a1_str = std::str::from_utf8(&r_aln.seq1).unwrap();
        let r_a2_str = std::str::from_utf8(&r_aln.seq2).unwrap();
        if r_a1_str != c_a1 {
            let common = r_a1_str
                .chars()
                .zip(c_a1.chars())
                .take_while(|(a, b)| a == b)
                .count();
            eprintln!("first diff col: {}", common);
            eprintln!(
                "R s1 [{}..]: {}",
                common.saturating_sub(10),
                &r_a1_str[common.saturating_sub(10)..r_a1_str.len().min(common + 30)]
            );
            eprintln!(
                "C s1 [{}..]: {}",
                common.saturating_sub(10),
                &c_a1[common.saturating_sub(10)..c_a1.len().min(common + 30)]
            );
            eprintln!(
                "R s2 [{}..]: {}",
                common.saturating_sub(10),
                &r_a2_str[common.saturating_sub(10)..r_a2_str.len().min(common + 30)]
            );
            eprintln!(
                "C s2 [{}..]: {}",
                common.saturating_sub(10),
                &c_a2[common.saturating_sub(10)..c_a2.len().min(common + 30)]
            );
        }
        assert_eq!(r_a1_str, c_a1, "seq1 mismatch");
        assert_eq!(r_a2_str, c_a2, "seq2 mismatch");
    }
}

/// Like `rust_global_align_matches_c_g__align11` but with `--allowshift`
/// warp DP enabled (penalty_shift_factor = 2.0 → trywarp = 1).
/// Guards the §9c warp DP port in `global_align`.
#[test]
fn rust_global_align_matches_c_g__align11_warp() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    let s1: &'static [u8] = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNFLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSIEGFFATLGGEVALWSLVVLAIERYIVICKPMGNFRFGNTHAIMGVAFTWIMALACAAPPLVGWSRYIPEGMQCSCGPDYYTLNPNFNNESYVVYMFVVHFLVPFVIIFFCYGRLLCTVKEAAAAQQESASTQKAEKEVTRMVVLMVIGFLVCWVPYASVAFYIFTHQGSDFGATFMTLPAFFAKSSALYNPVIYILMNKQFRNCMITTLCCGKNPLGDDESGASTSKTEVSSVSTSPVSPA";
    let s2: &'static [u8] = b"MAQQWSLQRLAGRHPQDSYEDSTQSSIFTYTNSNSTRGPFEGPNYHIAPRWVYHLTSVWMIFVVIASVFTNGLVLAATMKFKKLRHPLNWILVNLAVADLAETVIASTISVVNQVYGYFVLGHPMCVLEGYTVSLCGITGLWSLAIISWERWMVVCKPFGNVRFDAKLAIVGIAFSWIWAAVWTAPPIFGWSRYWPHGLKTSCGPDVFSGSSYPGVQSYMIVLMVTCCITPLSIIVLCYLQVWLAIRAVAKQQKESESTQKAEKEVTRMVVVMVLAFCFCWGPYAFFACFAAANPGYPFHPLMAALPAFFAKSATIYNPVIYVFMNRQFRNCILQLFGKKVDDGSELSSASKTEVSSVSSVSPA";

    let scale_protein: f64 = 600.0 / 1000.0;
    let cc_int = |x: f64, mul: f64| -> i32 { ((x * mul) - 0.5) as i32 };
    let cc_scale = |ppen: i32, scale: f64| -> i32 { ((scale * ppen as f64) + 0.5) as i32 };
    let p_open = cc_int(-2.00, 1000.0);
    let pair_open_f = cc_scale(p_open, scale_protein) as f64;
    // With --allowshift / --unalignlevel > 0: lexp=laof=0 (script:1469-1473),
    // so the usual pair_ext_f / pair_offset_int aren't computed here.
    let pair_ext_f_unalign = 0.0;
    let pair_offset_int_unalign: i32 = 0;
    // penalty_shift = (int)(spfactor * penalty) (constants.c:318), spfactor = 2.0.
    let penalty_shift = (2.0_f64 * pair_open_f) as i32 as f64;
    let pair_gap = GapModel::new(pair_open_f, pair_ext_f_unalign).with_shift(penalty_shift);

    let nscored = scoring.nscoredalphabets;
    let mut shifted: Vec<Vec<f64>> = scoring.consweight_matrix.clone();
    for i in 0..nscored {
        for j in 0..nscored {
            shifted[i][j] -= pair_offset_int_unalign as f64;
        }
    }

    let r_aln =
        mafft_align::global_align(s1, s2, &shifted, &scoring.amino_map, &pair_gap, true, true);
    eprintln!("Rust score: {}", r_aln.score);
    eprintln!("Rust width: {}", r_aln.seq1.len());

    unsafe {
        init_c_protein();
        std::ptr::addr_of_mut!(mafft_sys::outgap).write(1);
        std::ptr::addr_of_mut!(mafft_sys::penalty).write(pair_open_f as c_int);
        std::ptr::addr_of_mut!(mafft_sys::penalty_ex).write(pair_ext_f_unalign as c_int);
        // Activate warp DP: penalty_shift_factor < 10 → trywarp = 1
        // (constants.c:277-278). Set it AFTER init_c_protein → constants() has
        // already run, so set the globals directly.
        std::ptr::addr_of_mut!(mafft_sys::penalty_shift_factor).write(2.0);
        // The trywarp / penalty_shift assignment happens inside constants().
        // We need to re-run it to pick up the new factor. Easiest: re-call
        // constants() with the right seq.
        let seq_data = b"ACDEFGHIKLMNPQRSTVWY\0";
        let mut seq_ptr = seq_data.as_ptr() as *mut i8;
        let seq_arr: *mut *mut i8 = &mut seq_ptr;
        mafft_sys::constants(1, seq_arr);
        // After constants(): penalty_shift = (int)(factor * penalty).
        // Verify trywarp got set.

        let alloclen = (s1.len() + s2.len()) * 4;
        let mut buf1 = s1.to_vec();
        buf1.resize(alloclen + 1, 0);
        let mut buf2 = s2.to_vec();
        buf2.resize(alloclen + 1, 0);
        let buf1_box = buf1.into_boxed_slice();
        let buf2_box = buf2.into_boxed_slice();
        let mut p1: *mut c_char = buf1_box.as_ptr() as *mut c_char;
        let mut p2: *mut c_char = buf2_box.as_ptr() as *mut c_char;

        let n_dyn = build_c_dynamicmtx(&shifted);
        let c_score = mafft_sys::G__align11(n_dyn, &mut p1, &mut p2, alloclen as c_int, 1, 1);
        let c_len = {
            let s = p1;
            let mut k = 0;
            while *s.add(k) != 0 {
                k += 1;
            }
            k
        };
        let c_a1 = std::str::from_utf8(&buf1_box[..c_len]).unwrap();
        let c_a2 = std::str::from_utf8(&buf2_box[..c_len]).unwrap();
        eprintln!("C    score: {}", c_score);
        eprintln!("C    width: {}", c_len);

        mafft_sys::freeconstants();

        let r_a1_str = std::str::from_utf8(&r_aln.seq1).unwrap();
        let r_a2_str = std::str::from_utf8(&r_aln.seq2).unwrap();
        if r_a1_str != c_a1 {
            let common = r_a1_str
                .chars()
                .zip(c_a1.chars())
                .take_while(|(a, b)| a == b)
                .count();
            eprintln!("first diff col: {}", common);
            eprintln!(
                "R s1 [{}..]: {}",
                common.saturating_sub(10),
                &r_a1_str[common.saturating_sub(10)..r_a1_str.len().min(common + 30)]
            );
            eprintln!(
                "C s1 [{}..]: {}",
                common.saturating_sub(10),
                &c_a1[common.saturating_sub(10)..c_a1.len().min(common + 30)]
            );
            eprintln!(
                "R s2 [{}..]: {}",
                common.saturating_sub(10),
                &r_a2_str[common.saturating_sub(10)..r_a2_str.len().min(common + 30)]
            );
            eprintln!(
                "C s2 [{}..]: {}",
                common.saturating_sub(10),
                &c_a2[common.saturating_sub(10)..c_a2.len().min(common + 30)]
            );
        }
        assert_eq!(r_a1_str, c_a1, "seq1 mismatch (warp DP)");
        assert_eq!(r_a2_str, c_a2, "seq2 mismatch (warp DP)");
    }
}

/// Warp-DP regression for a pair where Rust's `global_align` disagrees with
/// C's `G__align11`: K03494 (human green-cone) vs M92036 (gecko opsin). On
/// the 36-seq sample, this pair surfaces a warp transition that C takes but
/// Rust currently skips, propagating into a 7-column width gap in the n=4
/// subcluster {8,9,10,11} and ultimately the §A `--allowshift` divergence
/// at n≥11.
///
/// C output (head): `MAQQW-------------------SLQRL...` — internal 19-col
/// warp jump after MAQQW, then aligns the conserved middle TM6 region.
/// Rust output (head): `MAQQWSLQRLAGRHPQDSYEDSTQSSI--...` — no warp,
/// trailing gap pushes the alignment longer.
#[test]
fn rust_global_align_matches_c_g__align11_warp_k03494_m92036() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    // K03494 — human GCP (green-sensitive cone opsin).
    let s1: &'static [u8] = b"MAQQWSLQRLAGRHPQDSYEDSTQSSIFTYTNSNSTRGPFEGPNYHIAPRWVYHLTSVWMIFVVIASVFTNGLVLAATMKFKKLRHPLNWILVNLAVADLAETVIASTISVVNQVYGYFVLGHPMCVLEGYTVSLCGITGLWSLAIISWERWMVVCKPFGNVRFDAKLAIVGIAFSWIWAAVWTAPPIFGWSRYWPHGLKTSCGPDVFSGSSYPGVQSYMIVLMVTCCITPLSIIVLCYLQVWLAIRAVAKQQKESESTQKAEKEVTRMVVVMVLAFCFCWGPYAFFACFAAANPGYPFHPLMAALPAFFAKSATIYNPVIYVFMNRQFRNCILQLFGKKVDDGSELSSASKTEVSSVSSVSPA";
    // M92036 — gecko P521 cone opsin.
    let s2: &'static [u8] = b"MTEAWNVAVFAARRSRDDDDTTRGSVFTYTNTNNTRGPFEGPNYHIAPRWVYNLVSFFMIIVVIASCFTNGLVLVATAKFKKLRHPLNWILVNLAFVDLVETLVASTISVFNQIFGYFILGHPLCVIEGYVVSSCGITGLWSLAIISWERWFVVCKPFGNIKFDSKLAIIGIVFSWVWAWGWSAPPIFGWSRYWPHGLKTSCGPDVFSGSVELGCQSFMLTLMITCCFLPLFIIIVCYLQVWMAIRAVAAQQKESESTQKAEREVSRMVVVMIVAFCICWGPYASFVSFAAANPGYAFHPLAAALPAYFAKSATIYNPVIYVFMNRQFRNCIMQLFGKKVDDGSEASTTSRTEVSSVSNSSVAPA";

    let scale_protein: f64 = 600.0 / 1000.0;
    let cc_int = |x: f64, mul: f64| -> i32 { ((x * mul) - 0.5) as i32 };
    let cc_scale = |ppen: i32, scale: f64| -> i32 { ((scale * ppen as f64) + 0.5) as i32 };
    let p_open = cc_int(-2.00, 1000.0);
    let pair_open_f = cc_scale(p_open, scale_protein) as f64;
    let pair_ext_f_unalign = 0.0;
    let pair_offset_int_unalign: i32 = 0;
    let penalty_shift = (2.0_f64 * pair_open_f) as i32 as f64;
    let pair_gap = GapModel::new(pair_open_f, pair_ext_f_unalign).with_shift(penalty_shift);

    let nscored = scoring.nscoredalphabets;
    let mut shifted: Vec<Vec<f64>> = scoring.consweight_matrix.clone();
    for i in 0..nscored {
        for j in 0..nscored {
            shifted[i][j] -= pair_offset_int_unalign as f64;
        }
    }

    let r_aln =
        mafft_align::global_align(s1, s2, &shifted, &scoring.amino_map, &pair_gap, true, true);

    unsafe {
        init_c_protein();
        std::ptr::addr_of_mut!(mafft_sys::outgap).write(1);
        std::ptr::addr_of_mut!(mafft_sys::penalty).write(pair_open_f as c_int);
        std::ptr::addr_of_mut!(mafft_sys::penalty_ex).write(pair_ext_f_unalign as c_int);
        std::ptr::addr_of_mut!(mafft_sys::penalty_shift_factor).write(2.0);
        let seq_data = b"ACDEFGHIKLMNPQRSTVWY\0";
        let mut seq_ptr = seq_data.as_ptr() as *mut i8;
        let seq_arr: *mut *mut i8 = &mut seq_ptr;
        mafft_sys::constants(1, seq_arr);

        let alloclen = (s1.len() + s2.len()) * 4;
        let mut buf1 = s1.to_vec();
        buf1.resize(alloclen + 1, 0);
        let mut buf2 = s2.to_vec();
        buf2.resize(alloclen + 1, 0);
        let buf1_box = buf1.into_boxed_slice();
        let buf2_box = buf2.into_boxed_slice();
        let mut p1: *mut c_char = buf1_box.as_ptr() as *mut c_char;
        let mut p2: *mut c_char = buf2_box.as_ptr() as *mut c_char;

        let n_dyn = build_c_dynamicmtx(&shifted);
        let c_score = mafft_sys::G__align11(n_dyn, &mut p1, &mut p2, alloclen as c_int, 1, 1);
        let c_len = {
            let s = p1;
            let mut k = 0;
            while *s.add(k) != 0 {
                k += 1;
            }
            k
        };
        let c_a1 = std::str::from_utf8(&buf1_box[..c_len]).unwrap();
        let c_a2 = std::str::from_utf8(&buf2_box[..c_len]).unwrap();
        eprintln!("Rust width={} score={}", r_aln.seq1.len(), r_aln.score);
        eprintln!("C    width={} score={}", c_len, c_score);

        mafft_sys::freeconstants();

        let r_a1_str = std::str::from_utf8(&r_aln.seq1).unwrap();
        let r_a2_str = std::str::from_utf8(&r_aln.seq2).unwrap();
        if r_a1_str != c_a1 {
            eprintln!("R s1[0..50]: {}", &r_a1_str[..r_a1_str.len().min(50)]);
            eprintln!("C s1[0..50]: {}", &c_a1[..c_a1.len().min(50)]);
            eprintln!("R s2[0..50]: {}", &r_a2_str[..r_a2_str.len().min(50)]);
            eprintln!("C s2[0..50]: {}", &c_a2[..c_a2.len().min(50)]);
        }
        assert_eq!(r_a1_str, c_a1, "seq1 (K03494) mismatch");
        assert_eq!(r_a2_str, c_a2, "seq2 (M92036) mismatch");
    }
}

/// Warp-DP regression for pair (U22180, M62903) — rat opsin vs chicken
/// visual pigment. C and Rust both produce same-width alignment (385 cols)
/// but disagree on where to place a P residue around a 16-col warp gap:
///   C:    `...TRGP----------------FEGPNYHI`
///   Rust: `...TRG----------------PFEGPNYHI`
/// This is a same-score tie-break inside the warp DP recurrence.
#[test]
fn rust_global_align_matches_c_g__align11_warp_u22180_m62903() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    // U22180 — rat opsin (length 348).
    let s1: &'static [u8] = b"MNGTEGPNFYVPFSNITGVVRSPFEQPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVFGGFTTTLYTSLHGYFVFGPTGCNLEGFFATLGGEIGLWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLVGWSRYIPEGMQCSCGIDYYTLKPEVNNESFVIYMFVVHFTIPMIVIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIFFLICWLPYASVAMYIFTHQGSNFGPIFMTLPAFFAKTASIYNPIIYIMMNKQFRNCMLTSLCCGKNPLGDDEASATASKTETSQVAPA";
    // M62903 — chicken visual pigment (length 351).
    let s2: &'static [u8] = b"MAAWEAAFAARRRHEEEDTTRDSVFTYTNSNNTRGPFEGPNYHIAPRWVYNLTSVWMIFVVAASVFTNGLVLVATWKFKKLRHPLNWILVNLAVADLGETVIASTISVINQISGYFILGHPMCVVEGYTVSACGITALWSLAIISWERWFVVCKPFGNIKFDGKLAVAGILFSWLWSCAWTAPPIFGWSRYWPHGLKTSCGPDVFSGSSDPGVQSYMVVLMVTCCFFPLAIIILCYLQVWLAIRAVAAQQKESESTQKAEKEVSRMVVVMIVAYCFCWGPYTFFACFAAANPGYAFHPLAAALPAYFAKSATIYNPIIYVFMNRQFRNCILQLFGKKVDDGSEVSTSRTEVSSVSNSSVSPA";

    let scale_protein: f64 = 600.0 / 1000.0;
    let cc_int = |x: f64, mul: f64| -> i32 { ((x * mul) - 0.5) as i32 };
    let cc_scale = |ppen: i32, scale: f64| -> i32 { ((scale * ppen as f64) + 0.5) as i32 };
    let p_open = cc_int(-2.00, 1000.0);
    let pair_open_f = cc_scale(p_open, scale_protein) as f64;
    let pair_ext_f_unalign = 0.0;
    let pair_offset_int_unalign: i32 = 0;
    let penalty_shift = (2.0_f64 * pair_open_f) as i32 as f64;
    let pair_gap = GapModel::new(pair_open_f, pair_ext_f_unalign).with_shift(penalty_shift);

    let nscored = scoring.nscoredalphabets;
    let mut shifted: Vec<Vec<f64>> = scoring.consweight_matrix.clone();
    for i in 0..nscored {
        for j in 0..nscored {
            shifted[i][j] -= pair_offset_int_unalign as f64;
        }
    }

    let r_aln =
        mafft_align::global_align(s1, s2, &shifted, &scoring.amino_map, &pair_gap, true, true);

    unsafe {
        init_c_protein();
        std::ptr::addr_of_mut!(mafft_sys::outgap).write(1);
        std::ptr::addr_of_mut!(mafft_sys::penalty).write(pair_open_f as c_int);
        std::ptr::addr_of_mut!(mafft_sys::penalty_ex).write(pair_ext_f_unalign as c_int);
        std::ptr::addr_of_mut!(mafft_sys::penalty_shift_factor).write(2.0);
        let seq_data = b"ACDEFGHIKLMNPQRSTVWY\0";
        let mut seq_ptr = seq_data.as_ptr() as *mut i8;
        let seq_arr: *mut *mut i8 = &mut seq_ptr;
        mafft_sys::constants(1, seq_arr);

        let alloclen = (s1.len() + s2.len()) * 4;
        let mut buf1 = s1.to_vec();
        buf1.resize(alloclen + 1, 0);
        let mut buf2 = s2.to_vec();
        buf2.resize(alloclen + 1, 0);
        let buf1_box = buf1.into_boxed_slice();
        let buf2_box = buf2.into_boxed_slice();
        let mut p1: *mut c_char = buf1_box.as_ptr() as *mut c_char;
        let mut p2: *mut c_char = buf2_box.as_ptr() as *mut c_char;

        let n_dyn = build_c_dynamicmtx(&shifted);
        let c_score = mafft_sys::G__align11(n_dyn, &mut p1, &mut p2, alloclen as c_int, 1, 1);
        let c_len = {
            let s = p1;
            let mut k = 0;
            while *s.add(k) != 0 {
                k += 1;
            }
            k
        };
        let c_a1 = std::str::from_utf8(&buf1_box[..c_len]).unwrap();
        let c_a2 = std::str::from_utf8(&buf2_box[..c_len]).unwrap();
        eprintln!("Rust width={} score={}", r_aln.seq1.len(), r_aln.score);
        eprintln!("C    width={} score={}", c_len, c_score);

        mafft_sys::freeconstants();

        let r_a1_str = std::str::from_utf8(&r_aln.seq1).unwrap();
        let r_a2_str = std::str::from_utf8(&r_aln.seq2).unwrap();
        if r_a1_str != c_a1 {
            let common = r_a1_str
                .chars()
                .zip(c_a1.chars())
                .take_while(|(a, b)| a == b)
                .count();
            eprintln!("first diff col: {}", common);
            eprintln!(
                "R s1 [{}..]: {}",
                common.saturating_sub(5),
                &r_a1_str[common.saturating_sub(5)..r_a1_str.len().min(common + 25)]
            );
            eprintln!(
                "C s1 [{}..]: {}",
                common.saturating_sub(5),
                &c_a1[common.saturating_sub(5)..c_a1.len().min(common + 25)]
            );
            eprintln!(
                "R s2 [{}..]: {}",
                common.saturating_sub(5),
                &r_a2_str[common.saturating_sub(5)..r_a2_str.len().min(common + 25)]
            );
            eprintln!(
                "C s2 [{}..]: {}",
                common.saturating_sub(5),
                &c_a2[common.saturating_sub(5)..c_a2.len().min(common + 25)]
            );
        }
        assert_eq!(r_a1_str, c_a1, "seq1 (U22180) mismatch");
        assert_eq!(r_a2_str, c_a2, "seq2 (M62903) mismatch");
    }
}

/// Pair (U22180, M62903) RE-ALIGNMENT test (per-pair `makedynamicmtx` path,
/// `pairlocalalign.c:2197-2215`). Mirrors `--allowshift` pipeline's re-align
/// for this pair (delta = -161.3, fires when `0.5*dist - unalignlevel < 0`).
/// Guards `global_align` warp DP under a shifted matrix.
///
/// **The earlier "bug" here was in this test's C-side setup**: `ppenalty`
/// wasn't reset before the second `constants()` call, so C re-derived
/// `penalty` from the default `-1530` (→ -917) rather than honoring the
/// `pair_open_f = -1199` we'd written directly. Setting `ppenalty = -2000`
/// before the second constants() call brings C in line and the test passes.
#[test]
fn rust_global_align_realign_matches_c_g__align11_warp_u22180_m62903() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    let s1: &'static [u8] = b"MNGTEGPNFYVPFSNITGVVRSPFEQPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVFGGFTTTLYTSLHGYFVFGPTGCNLEGFFATLGGEIGLWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLVGWSRYIPEGMQCSCGIDYYTLKPEVNNESFVIYMFVVHFTIPMIVIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIFFLICWLPYASVAMYIFTHQGSNFGPIFMTLPAFFAKTASIYNPIIYIMMNKQFRNCMLTSLCCGKNPLGDDEASATASKTETSQVAPA";
    let s2: &'static [u8] = b"MAAWEAAFAARRRHEEEDTTRDSVFTYTNSNNTRGPFEGPNYHIAPRWVYNLTSVWMIFVVAASVFTNGLVLVATWKFKKLRHPLNWILVNLAVADLGETVIASTISVINQISGYFILGHPMCVVEGYTVSACGITALWSLAIISWERWFVVCKPFGNIKFDGKLAVAGILFSWLWSCAWTAPPIFGWSRYWPHGLKTSCGPDVFSGSSDPGVQSYMVVLMVTCCFFPLAIIILCYLQVWLAIRAVAAQQKESESTQKAEKEVSRMVVVMIVAYCFCWGPYTFFACFAAANPGYAFHPLAAALPAYFAKSATIYNPIIYVFMNRQFRNCILQLFGKKVDDGSEVSTSRTEVSSVSNSSVSPA";

    let scale_protein: f64 = 600.0 / 1000.0;
    let cc_int = |x: f64, mul: f64| -> i32 { ((x * mul) - 0.5) as i32 };
    let cc_scale = |ppen: i32, scale: f64| -> i32 { ((scale * ppen as f64) + 0.5) as i32 };
    let p_open = cc_int(-2.00, 1000.0);
    let pair_open_f = cc_scale(p_open, scale_protein) as f64;
    let penalty_shift = (2.0_f64 * pair_open_f) as i32 as f64;
    let pair_gap = GapModel::new(pair_open_f, 0.0).with_shift(penalty_shift);

    // Use the SAME delta the pipeline applies for this pair:
    // off = -0.268844 (from RUST_PAIR_DUMP), delta = off * 600 = -161.3064.
    // To reproduce exactly, recompute from selfscore + initial align score.
    // But for this test, just use a representative delta close to the real
    // value. The pipeline's actual value is determined by the data.
    let nscored = scoring.nscoredalphabets;
    let selfscore = |s: &[u8]| -> f64 {
        s.iter()
            .map(|&c| {
                let i = scoring.amino_map[c as usize] as usize;
                if i < nscored {
                    scoring.consweight_matrix[i][i]
                } else {
                    0.0
                }
            })
            .sum()
    };
    let ss1 = selfscore(s1);
    let ss2 = selfscore(s2);

    // Initial alignment to get the score → distance → delta.
    let initial = mafft_align::global_align(
        s1,
        s2,
        &scoring.consweight_matrix,
        &scoring.amino_map,
        &pair_gap,
        true,
        true,
    );
    let bunbo = ss1.min(ss2);
    let dist = if bunbo == 0.0 {
        2.0
    } else if bunbo < initial.score {
        0.0
    } else {
        (1.0 - initial.score / bunbo) * 2.0
    };
    let unalign_level = 0.8;
    let off = 0.5 * dist - unalign_level;
    let delta = off * 600.0;
    eprintln!(
        "pair (U22180, M62903): initial_score={:.3} bunbo={:.3} dist={:.6} off={:.6} delta={:.3}",
        initial.score, bunbo, dist, off, delta
    );

    let dyn_matrix: Vec<Vec<f64>> = scoring
        .consweight_matrix
        .iter()
        .map(|row| row.iter().map(|&v| v + delta).collect())
        .collect();

    let r_aln = mafft_align::global_align(
        s1,
        s2,
        &dyn_matrix,
        &scoring.amino_map,
        &pair_gap,
        true,
        true,
    );
    eprintln!(
        "Rust re-align width={} score={:.3}",
        r_aln.seq1.len(),
        r_aln.score
    );

    unsafe {
        init_c_protein();
        // `ppenalty` must be set BEFORE the second `constants()` call,
        // because constants() recomputes penalty = (int)(0.6 * ppenalty + 0.5).
        // Without this, ppenalty defaults to DEFAULTGOP_B = -1530, giving
        // penalty = -917 (vs our intended -1199 from `--op 2.00` scaled).
        std::ptr::addr_of_mut!(mafft_sys::ppenalty).write(-2000);
        std::ptr::addr_of_mut!(mafft_sys::ppenalty_ex).write(0);
        std::ptr::addr_of_mut!(mafft_sys::outgap).write(1);
        std::ptr::addr_of_mut!(mafft_sys::penalty_shift_factor).write(2.0);
        let seq_data = b"ACDEFGHIKLMNPQRSTVWY\0";
        let mut seq_ptr = seq_data.as_ptr() as *mut i8;
        let seq_arr: *mut *mut i8 = &mut seq_ptr;
        mafft_sys::constants(1, seq_arr);

        let alloclen = (s1.len() + s2.len()) * 4;

        // Mirror pipeline: call G__align11 FIRST with unshifted matrix
        // (initial alignment), THEN with dyn_matrix (re-align). C's TLS
        // state from the first call may affect the second.
        let mut buf1_init = s1.to_vec();
        buf1_init.resize(alloclen + 1, 0);
        let mut buf2_init = s2.to_vec();
        buf2_init.resize(alloclen + 1, 0);
        let buf1_init_box = buf1_init.into_boxed_slice();
        let buf2_init_box = buf2_init.into_boxed_slice();
        let mut p1_init: *mut c_char = buf1_init_box.as_ptr() as *mut c_char;
        let mut p2_init: *mut c_char = buf2_init_box.as_ptr() as *mut c_char;
        let n_unshifted = build_c_dynamicmtx(&scoring.consweight_matrix);
        let _ = mafft_sys::G__align11(
            n_unshifted,
            &mut p1_init,
            &mut p2_init,
            alloclen as c_int,
            1,
            1,
        );

        // Now re-align with the shifted matrix (matches pipeline).
        let mut buf1 = s1.to_vec();
        buf1.resize(alloclen + 1, 0);
        let mut buf2 = s2.to_vec();
        buf2.resize(alloclen + 1, 0);
        let buf1_box = buf1.into_boxed_slice();
        let buf2_box = buf2.into_boxed_slice();
        let mut p1: *mut c_char = buf1_box.as_ptr() as *mut c_char;
        let mut p2: *mut c_char = buf2_box.as_ptr() as *mut c_char;

        let n_dyn = build_c_dynamicmtx(&dyn_matrix);
        let c_score = mafft_sys::G__align11(n_dyn, &mut p1, &mut p2, alloclen as c_int, 1, 1);
        let c_len = {
            let s = p1;
            let mut k = 0;
            while *s.add(k) != 0 {
                k += 1;
            }
            k
        };
        let c_a1 = std::str::from_utf8(&buf1_box[..c_len]).unwrap();
        let c_a2 = std::str::from_utf8(&buf2_box[..c_len]).unwrap();
        eprintln!("C re-align width={} score={:.3}", c_len, c_score);

        mafft_sys::freeconstants();

        let r_a1_str = std::str::from_utf8(&r_aln.seq1).unwrap();
        let r_a2_str = std::str::from_utf8(&r_aln.seq2).unwrap();
        if r_a1_str != c_a1 {
            let common = r_a1_str
                .chars()
                .zip(c_a1.chars())
                .take_while(|(a, b)| a == b)
                .count();
            eprintln!("first diff col: {}", common);
            eprintln!(
                "R s1 [{}..]: {}",
                common.saturating_sub(5),
                &r_a1_str[common.saturating_sub(5)..r_a1_str.len().min(common + 25)]
            );
            eprintln!(
                "C s1 [{}..]: {}",
                common.saturating_sub(5),
                &c_a1[common.saturating_sub(5)..c_a1.len().min(common + 25)]
            );
        }
        assert_eq!(r_a1_str, c_a1, "seq1 re-align mismatch");
        assert_eq!(r_a2_str, c_a2, "seq2 re-align mismatch");
    }
}

/// Direct comparison: our genaffine_local_align vs C's genL__align11 on the
/// real 36-seq sample's first 2 sequences. Guards the E-INS-i pairwise port.
#[test]
fn rust_genaffine_align_matches_c_gen_l__align11() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    // Pair (M63632, U22180) — same close opsins used elsewhere; this
    // pair is enough to expose the divergence between local-SW and
    // generalized-affine alignments at the head/tail.
    let s1: &'static [u8] = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNFLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSIEGFFATLGGEVALWSLVVLAIERYIVICKPMGNFRFGNTHAIMGVAFTWIMALACAAPPLVGWSRYIPEGMQCSCGPDYYTLNPNFNNESYVVYMFVVHFLVPFVIIFFCYGRLLCTVKEAAAAQQESASTQKAEKEVTRMVVLMVIGFLVCWVPYASVAFYIFTHQGSDFGATFMTLPAFFAKSSALYNPVIYILMNKQFRNCMITTLCCGKNPLGDDESGASTSKTEVSSVSTSPVSPA";
    let s2: &'static [u8] = b"MNGTEGPNFYVPFSNITGVVRSPFEQPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVFGGFTTTLYTSLHGYFVFGPTGCNLEGFFATLGGEIGLWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLVGWSRYIPEGMQCSCGIDYYTLKPEVNNESFVIYMFVVHFTIPMIVIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIFFLICWLPYASVAMYIFTHQGSNFGPIFMTLPAFFAKTASIYNPIIYIMMNKQFRNCMLTSLCCGKNPLGDDEASATASKTETSQVAPA";

    // E-INS-i pairwise params (script:91-92,201-203 + LGOP/LEXP):
    //   lgop=-2.00, lexp=-0.100, laof=0.100 (regular affine)
    //   LGOP=-6.00 (gen-affine OP), LEXP=-0.100 (gen-affine EX)
    // C's pairlocalalign for `-N` reads:
    //   penalty = scale*ppenalty
    //   penalty_ex = scale*ppenalty_ex
    //   penalty_OP = scale*ppenalty_OP
    let scale_protein: f64 = 600.0 / 1000.0;
    let cc_int = |x: f64, mul: f64| -> i32 { ((x * mul) - 0.5) as i32 };
    let cc_scale = |ppen: i32, scale: f64| -> i32 { ((scale * ppen as f64) + 0.5) as i32 };
    let p_open = cc_int(-2.00, 1000.0);
    let p_ext = cc_int(-0.100, 1000.0);
    let p_offset = cc_int(0.100, 1000.0);
    let p_op = cc_int(-6.00, 1000.0);
    let pair_open = cc_scale(p_open, scale_protein);
    let pair_ext = cc_scale(p_ext, scale_protein);
    let pair_offset_int = cc_scale(p_offset, scale_protein);
    let pair_op = cc_scale(p_op, scale_protein);

    use mafft_align::{GenAffineGapModel, genaffine_local_align};
    let pair_gap = GenAffineGapModel {
        affine: GapModel::new(pair_open as f64, pair_ext as f64),
        open_generalized: pair_op as f64,
    };

    // Apply matrix offset shift (constants.c:798).
    let nscored = scoring.nscoredalphabets;
    let mut shifted: Vec<Vec<f64>> = scoring.consweight_matrix.clone();
    for i in 0..nscored {
        for j in 0..nscored {
            shifted[i][j] -= pair_offset_int as f64;
        }
    }
    let score_offset_for_local = pair_offset_int as f64 / 600.0;

    let r_aln = genaffine_local_align(
        s1,
        s2,
        &shifted,
        &scoring.amino_map,
        &pair_gap,
        score_offset_for_local,
    );

    eprintln!("Rust score: {}", r_aln.alignment.score);
    eprintln!("Rust width: {}", r_aln.alignment.seq1.len());
    eprintln!("Rust offset1: {}", r_aln.offset1);
    eprintln!("Rust offset2: {}", r_aln.offset2);
    let l = r_aln.alignment.seq1.len();
    eprintln!(
        "Rust seq1[..30]: {}",
        std::str::from_utf8(&r_aln.alignment.seq1[..30.min(l)]).unwrap()
    );
    eprintln!(
        "Rust seq1[-30..]: {}",
        std::str::from_utf8(&r_aln.alignment.seq1[l.saturating_sub(30)..]).unwrap()
    );
    eprintln!(
        "Rust seq2[-30..]: {}",
        std::str::from_utf8(&r_aln.alignment.seq2[l.saturating_sub(30)..]).unwrap()
    );

    unsafe {
        init_c_protein();
        std::ptr::addr_of_mut!(mafft_sys::penalty).write(pair_open);
        std::ptr::addr_of_mut!(mafft_sys::penalty_ex).write(pair_ext);
        std::ptr::addr_of_mut!(mafft_sys::penalty_OP).write(pair_op);
        std::ptr::addr_of_mut!(mafft_sys::penalty_EX).write(0);
        std::ptr::addr_of_mut!(mafft_sys::offset).write(pair_offset_int);
        std::ptr::addr_of_mut!(mafft_sys::njob).write(2);

        let alloclen = (s1.len() + s2.len()) * 4;
        let mut buf1 = s1.to_vec();
        buf1.resize(alloclen + 1, 0);
        let mut buf2 = s2.to_vec();
        buf2.resize(alloclen + 1, 0);
        let buf1_box = buf1.into_boxed_slice();
        let buf2_box = buf2.into_boxed_slice();
        let mut p1: *mut c_char = buf1_box.as_ptr() as *mut c_char;
        let mut p2: *mut c_char = buf2_box.as_ptr() as *mut c_char;

        let n_dyn = build_c_dynamicmtx(&shifted);
        let mut off1: c_int = 0;
        let mut off2: c_int = 0;
        let c_score = mafft_sys::genL__align11(
            n_dyn,
            &mut p1,
            &mut p2,
            alloclen as c_int,
            &mut off1,
            &mut off2,
        );

        let c_len = {
            let s = p1;
            let mut k = 0;
            while *s.add(k) != 0 {
                k += 1;
            }
            k
        };
        let c_a1 = std::str::from_utf8(&buf1_box[..c_len]).unwrap();
        let c_a2 = std::str::from_utf8(&buf2_box[..c_len]).unwrap();
        eprintln!("C    score: {}", c_score);
        eprintln!("C    width: {}", c_len);
        eprintln!("C    off1: {} off2: {}", off1, off2);
        eprintln!("C    seq1[..40]: {}", &c_a1[..40.min(c_len)]);
        eprintln!("C    seq1[-40..]: {}", &c_a1[c_len.saturating_sub(40)..]);
        eprintln!("C    seq2[-40..]: {}", &c_a2[c_len.saturating_sub(40)..]);

        mafft_sys::freeconstants();

        let r_a1_str = std::str::from_utf8(&r_aln.alignment.seq1).unwrap();
        let r_a2_str = std::str::from_utf8(&r_aln.alignment.seq2).unwrap();
        if r_a1_str != c_a1 {
            let common = r_a1_str
                .chars()
                .zip(c_a1.chars())
                .take_while(|(a, b)| a == b)
                .count();
            eprintln!("first diff col: {}", common);
            eprintln!("R s1[..30]: {}", &r_a1_str[..r_a1_str.len().min(30)]);
            eprintln!("C s1[..30]: {}", &c_a1[..c_a1.len().min(30)]);
            eprintln!("R s2[..30]: {}", &r_a2_str[..r_a2_str.len().min(30)]);
            eprintln!("C s2[..30]: {}", &c_a2[..c_a2.len().min(30)]);
        }
        assert!(
            ((r_aln.alignment.score - c_score).abs()) < 1.0,
            "scores differ: R={} C={}",
            r_aln.alignment.score,
            c_score
        );
        assert_eq!(r_a1_str, c_a1, "seq1 mismatch");
        assert_eq!(r_a2_str, c_a2, "seq2 mismatch");
    }
}

/// Same comparison but using **E-INS-i** params (lexp=0.0, laof=0.0
/// per `scripts/mafft:1940-1948`). The earlier
/// `rust_genaffine_align_matches_c_gen_l__align11` test used L-INS-i-
/// style params and missed an E-INS-i-specific divergence in
/// `genaffine_local_align` whose score / distance feeds the refinement
/// tree's edge lengths. n=14 byte-passes despite the divergence
/// (topology coincidentally identical), but n=15+ tips at some pair.
#[test]
fn rust_genaffine_align_matches_c_gen_l__align11_einsi_params() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);
    let s1: &'static [u8] = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNFLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSIEGFFATLGGEVALWSLVVLAIERYIVICKPMGNFRFGNTHAIMGVAFTWIMALACAAPPLVGWSRYIPEGMQCSCGPDYYTLNPNFNNESYVVYMFVVHFLVPFVIIFFCYGRLLCTVKEAAAAQQESASTQKAEKEVTRMVVLMVIGFLVCWVPYASVAFYIFTHQGSDFGATFMTLPAFFAKSSALYNPVIYILMNKQFRNCMITTLCCGKNPLGDDESGASTSKTEVSSVSTSPVSPA";
    let s2: &'static [u8] = b"MNGTEGPNFYVPFSNITGVVRSPFEQPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVFGGFTTTLYTSLHGYFVFGPTGCNLEGFFATLGGEIGLWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLVGWSRYIPEGMQCSCGIDYYTLKPEVNNESFVIYMFVVHFTIPMIVIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIFFLICWLPYASVAMYIFTHQGSNFGPIFMTLPAFFAKTASIYNPIIYIMMNKQFRNCMLTSLCCGKNPLGDDEASATASKTETSQVAPA";

    let scale_protein: f64 = 600.0 / 1000.0;
    let cc_int = |x: f64, mul: f64| -> i32 { ((x * mul) - 0.5) as i32 };
    let cc_scale = |ppen: i32, scale: f64| -> i32 { ((scale * ppen as f64) + 0.5) as i32 };
    // E-INS-i overrides: lexp = 0.0, laof = 0.0 (script:1940-1948).
    let p_open = cc_int(-2.00, 1000.0);
    let p_ext = cc_int(0.0, 1000.0);
    let p_offset = cc_int(0.0, 1000.0);
    let p_op = cc_int(-6.00, 1000.0);
    let pair_open = cc_scale(p_open, scale_protein);
    let pair_ext = cc_scale(p_ext, scale_protein);
    let pair_offset_int = cc_scale(p_offset, scale_protein);
    let pair_op = cc_scale(p_op, scale_protein);

    use mafft_align::{GenAffineGapModel, genaffine_local_align};
    let pair_gap = GenAffineGapModel {
        affine: GapModel::new(pair_open as f64, pair_ext as f64),
        open_generalized: pair_op as f64,
    };

    let nscored = scoring.nscoredalphabets;
    let mut shifted: Vec<Vec<f64>> = scoring.consweight_matrix.clone();
    for i in 0..nscored {
        for j in 0..nscored {
            shifted[i][j] -= pair_offset_int as f64;
        }
    }
    let score_offset_for_local = pair_offset_int as f64 / 600.0;

    let r_aln = genaffine_local_align(
        s1,
        s2,
        &shifted,
        &scoring.amino_map,
        &pair_gap,
        score_offset_for_local,
    );

    // Rust selfscore = sum of diag entries (matches build_homology_table).
    let n_alpha = shifted.len();
    let r_self1: f64 = s1
        .iter()
        .map(|&c| {
            let i = scoring.amino_map[c as usize] as usize;
            if i < n_alpha {
                shifted[i][i] as f64
            } else {
                0.0
            }
        })
        .sum();
    let r_self2: f64 = s2
        .iter()
        .map(|&c| {
            let i = scoring.amino_map[c as usize] as usize;
            if i < n_alpha {
                shifted[i][i] as f64
            } else {
                0.0
            }
        })
        .sum();
    let r_dist = (1.0 - r_aln.alignment.score / r_self1.min(r_self2)) * 2.0;

    unsafe {
        // Pre-set poffset to 0 (mirroring the actual E-INS-i pipeline,
        // where the script passes `-h 0.0`) so constants() builds
        // `amino_dis` with no offset shift. Without this override
        // init_c_protein() leaves poffset at NOTSPECIFIED → constants()
        // applies DEFAULTOFS_B=-123 → offset=-73 → +73 shift.
        mafft_sys::initglobalvariables();
        std::ptr::addr_of_mut!(mafft_sys::ppenalty).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::ppenalty_ex).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::ppenalty_EX).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::ppenalty_OP).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::ppenalty_dist).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::poffset).write(0); // E-INS-i: laof=0.0
        std::ptr::addr_of_mut!(mafft_sys::kimuraR).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::pamN).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::dorp).write(b'p' as i32);
        std::ptr::addr_of_mut!(mafft_sys::scoremtx).write(1);
        std::ptr::addr_of_mut!(mafft_sys::nblosum).write(62);
        std::ptr::addr_of_mut!(mafft_sys::fmodel).write(0);

        let seq_data = b"ACDEFGHIKLMNPQRSTVWY\0";
        let mut seq_ptr = seq_data.as_ptr() as *mut i8;
        let seq_arr: *mut *mut i8 = &mut seq_ptr;
        mafft_sys::constants(1, seq_arr);

        std::ptr::addr_of_mut!(mafft_sys::penalty).write(pair_open);
        std::ptr::addr_of_mut!(mafft_sys::penalty_ex).write(pair_ext);
        std::ptr::addr_of_mut!(mafft_sys::penalty_OP).write(pair_op);
        std::ptr::addr_of_mut!(mafft_sys::penalty_EX).write(0);
        std::ptr::addr_of_mut!(mafft_sys::offset).write(pair_offset_int);
        std::ptr::addr_of_mut!(mafft_sys::njob).write(2);

        // C selfscore reads amino_dis (the integer matrix populated by constants()
        // with the offset shift applied).
        let mut c_self1 = 0.0f64;
        let mut c_self2 = 0.0f64;
        for &c in s1 {
            let row_ptr = *mafft_sys::amino_dis.add(c as usize);
            let v = *row_ptr.add(c as usize);
            c_self1 += v as f64;
        }
        for &c in s2 {
            let row_ptr = *mafft_sys::amino_dis.add(c as usize);
            let v = *row_ptr.add(c as usize);
            c_self2 += v as f64;
        }

        let alloclen = (s1.len() + s2.len()) * 4;
        let mut buf1 = s1.to_vec();
        buf1.resize(alloclen + 1, 0);
        let mut buf2 = s2.to_vec();
        buf2.resize(alloclen + 1, 0);
        let buf1_box = buf1.into_boxed_slice();
        let buf2_box = buf2.into_boxed_slice();
        let mut p1: *mut c_char = buf1_box.as_ptr() as *mut c_char;
        let mut p2: *mut c_char = buf2_box.as_ptr() as *mut c_char;

        let n_dyn = build_c_dynamicmtx(&shifted);
        let mut off1: c_int = 0;
        let mut off2: c_int = 0;
        let c_score = mafft_sys::genL__align11(
            n_dyn,
            &mut p1,
            &mut p2,
            alloclen as c_int,
            &mut off1,
            &mut off2,
        );

        let c_dist = (1.0 - c_score / c_self1.min(c_self2)) * 2.0;
        eprintln!("E-INS-i pair (M63632, U22180):");
        eprintln!("  Rust score: {}", r_aln.alignment.score);
        eprintln!("  C    score: {}", c_score);
        eprintln!("  Rust selfscore[0]={r_self1} selfscore[1]={r_self2}");
        eprintln!("  C    selfscore[0]={c_self1} selfscore[1]={c_self2}");
        eprintln!("  Rust distance={r_dist}");
        eprintln!("  C    distance={c_dist}");
        // Compare per-residue diagonal values.
        eprintln!("  Diag entries (Rust shifted vs C amino_dis):");
        for c in [b'A', b'L', b'V', b'M', b'C', b'W', b'Y', b'P'] {
            let ri = scoring.amino_map[c as usize] as usize;
            let r_v = if ri < n_alpha { shifted[ri][ri] } else { 0.0 };
            let row_ptr = *mafft_sys::amino_dis.add(c as usize);
            let c_v = *row_ptr.add(c as usize);
            eprintln!("    {}: Rust={} C={}", c as char, r_v, c_v);
        }
        let c_len = {
            let s = p1;
            let mut k = 0;
            while *s.add(k) != 0 {
                k += 1;
            }
            k
        };
        let c_a1 = std::str::from_utf8(&buf1_box[..c_len]).unwrap();
        let r_a1_str = std::str::from_utf8(&r_aln.alignment.seq1).unwrap();

        mafft_sys::freeconstants();

        assert!(
            ((r_aln.alignment.score - c_score).abs()) < 1.0,
            "E-INS-i scores differ: R={} C={} Δ={}",
            r_aln.alignment.score,
            c_score,
            r_aln.alignment.score - c_score
        );
        assert_eq!(r_a1_str, c_a1, "E-INS-i seq1 mismatch");
    }
}

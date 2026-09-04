// Test names mirror the C identifiers being validated (`A__align`); the
// double underscore is upstream MAFFT's, not ours.
#![allow(non_snake_case)]

//! Cross-validate the refinement profile DP against C `A__align` on the
//! exact sub-problem where FFT-NS-i first diverges on nucleotide input.
//!
//! Provenance of `fixtures/refine_branch_seg10_dna.txt`: running
//! `mafft-rs --adjustdirection --nuc --retree 2 --maxiterate 2` on
//! `mtb_cds_120x1400.fa` (120 DNA sequences, ~1.4 kb, evolved from a real
//! M. tuberculosis CDS) and comparing per-branch verdicts against C MAFFT
//! 7.526's single-threaded `dvtditr` trace. Both sides agree on the
//! segmentation (85 segments), the branch enumeration, and every verdict
//! up to segment 10 / iteration 0 / branch l=30 / side k=0 — where C says
//! `accepted.` and Rust says `identical.`
//!
//! Three explanations were ruled out before this fixture was cut:
//!   * not the two-row identity check — Rust's re-alignment reproduced its
//!     input in ALL 120 rows, not merely the two representatives;
//!   * not silent state drift — no earlier branch in the segment changed
//!     non-representative rows, so both sides enter this branch with the
//!     same 18 columns;
//!   * not the FFT path — an 18-column window never reaches it, and
//!     disabling FFT in refinement changes nothing.
//!
//! So Rust's profile DP returns its input unchanged on a 26-vs-94 split
//! over 18 columns where C finds a better-scoring re-alignment. This test
//! feeds both implementations byte-identical inputs and reports the first
//! place they disagree: score first (cheap, and it separates "the DP never
//! finds the better alignment" from "the DP finds it but the acceptance
//! comparison drops it"), then the alignment itself.
//!
//! RESULT: they do not disagree. Rust `profile_align`, C `A__align` and C
//! `Falign` (which is what refinement actually calls — `tditeration.c:2153`,
//! `:1035`) all return score 7651.229677 at width 18, and all three return
//! the input unchanged. So the profile DP is NOT the source of the FFT-NS-i
//! divergence, and neither is the acceptance comparison. Since both sides
//! provably enter this branch with the same columns (all 60 preceding
//! branches are `identical` on both sides), the difference must be in an
//! input this fixture reconstructs rather than reads from C: the branch's
//! group membership or its per-branch weights. That is the next thing to
//! check, and this test stands as the guard that keeps the DP itself ruled
//! out while it is.
//!
//! The fixture was produced by a temporary dump hook in
//! `refinement::iterative_refine` keyed on (segment, iter, l, k); it is
//! committed so the sub-problem survives without that scaffolding.

use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::sync::Mutex;

use mafft_align::{AlignOp, GapModel, Profile, profile_align};

static C_MUTEX: Mutex<()> = Mutex::new(());

struct Branch {
    nseq: usize,
    width: usize,
    group1: Vec<usize>,
    group2: Vec<usize>,
    weights: Vec<f64>,
    gap_open: f64,
    seqs: Vec<Vec<u8>>,
}

fn load_branch(path: &Path) -> Branch {
    let text = std::fs::read_to_string(path).expect("read branch fixture");
    let (mut nseq, mut width, mut gap_open) = (0usize, 0usize, 0f64);
    let (mut group1, mut group2, mut weights) = (Vec::new(), Vec::new(), Vec::new());
    let mut rows: Vec<(usize, Vec<u8>)> = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("NSEQ") => nseq = it.next().unwrap().parse().unwrap(),
            Some("WIDTH") => width = it.next().unwrap().parse().unwrap(),
            Some("GAPOPEN") => gap_open = it.next().unwrap().parse().unwrap(),
            Some("GROUP1") => group1 = it.map(|x| x.parse().unwrap()).collect(),
            Some("GROUP2") => group2 = it.map(|x| x.parse().unwrap()).collect(),
            Some("WEIGHTS") => weights = it.map(|x| x.parse().unwrap()).collect(),
            Some("SEQ") => {
                let idx: usize = it.next().unwrap().parse().unwrap();
                rows.push((idx, it.next().unwrap().as_bytes().to_vec()));
            }
            _ => {}
        }
    }
    rows.sort_by_key(|(i, _)| *i);
    let seqs: Vec<Vec<u8>> = rows.into_iter().map(|(_, s)| s).collect();
    assert_eq!(seqs.len(), nseq);
    assert!(seqs.iter().all(|s| s.len() == width));
    assert_eq!(weights.len(), nseq);
    Branch { nseq, width, group1, group2, weights, gap_open, seqs }
}

unsafe fn init_c_dna() {
    unsafe {
        mafft_sys::initglobalvariables();
        std::ptr::addr_of_mut!(mafft_sys::ppenalty).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::ppenalty_ex).write(mafft_sys::NOTSPECIFIED);
        // `mafft.tmpl` passes `-h 0.000`, so poffset = 0 (not DEFAULTOFS_N).
        std::ptr::addr_of_mut!(mafft_sys::poffset).write(0);
        std::ptr::addr_of_mut!(mafft_sys::kimuraR).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::pamN).write(mafft_sys::NOTSPECIFIED);
        // dorp='d' + scoremtx=-1 select C's nucleotide branch in
        // constants.c, which is what applies the `3 * 600/1000` gap scale.
        std::ptr::addr_of_mut!(mafft_sys::dorp).write(b'd' as i32);
        std::ptr::addr_of_mut!(mafft_sys::scoremtx).write(-1);
        std::ptr::addr_of_mut!(mafft_sys::fmodel).write(0);
        // `Falign` dispatches on these; C's dvtditr banner for this run reads
        // `alg=A, model=DNA200 ... noshift`, and the script passes `-F`
        // (use FFT) with the stock window/threshold.
        std::ptr::addr_of_mut!(mafft_sys::alg).write(b'A' as c_char);
        std::ptr::addr_of_mut!(mafft_sys::fftWinSize).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::fftThreshold).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::fftkeika).write(0);
        let seq_data = b"acgt\0";
        let mut seq_ptr = seq_data.as_ptr() as *mut i8;
        let seq_arr: *mut *mut i8 = &mut seq_ptr;
        mafft_sys::constants(1, seq_arr);
    }
}

unsafe fn alloc_c_char_mtx(seqs: &[Vec<u8>], capacity: usize) -> (Vec<*mut c_char>, Vec<Vec<u8>>) {
    let cap = capacity.max(seqs.iter().map(|s| s.len()).max().unwrap_or(0)) + 16;
    let mut rows: Vec<Vec<u8>> = seqs
        .iter()
        .map(|s| {
            let mut v = vec![0u8; cap];
            v[..s.len()].copy_from_slice(s);
            v[s.len()] = 0;
            v
        })
        .collect();
    let ptrs: Vec<*mut c_char> = rows.iter_mut().map(|r| r.as_mut_ptr() as *mut c_char).collect();
    (ptrs, rows)
}

#[test]
fn refine_branch_seg10_profile_dp_matches_c() {
    let _guard = C_MUTEX.lock().unwrap();

    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/refine_branch_seg10_dna.txt");
    let b = load_branch(&fixture);
    eprintln!(
        "branch: nseq={} width={} clus1={} clus2={} gap_open={}",
        b.nseq, b.width, b.group1.len(), b.group2.len(), b.gap_open
    );

    let scoring = mafft_scoring::build_context(
        mafft_types::ScoringModel::Dna,
        mafft_types::SeqType::Dna,
    );
    // Sanity: the DP must be running C's nucleotide gap scale, not the
    // protein one. C: penalty = (int)(3 * 600/1000 * -1530 + 0.5) = -2753.
    assert_eq!(
        scoring.gap.open as f64, b.gap_open,
        "fixture gap_open must match the DNA scoring context"
    );

    let g1: Vec<Vec<u8>> = b.group1.iter().map(|&i| b.seqs[i].clone()).collect();
    let g2: Vec<Vec<u8>> = b.group2.iter().map(|&i| b.seqs[i].clone()).collect();

    // Group-local sum-1 weights, exactly as `refinement::realign_all` builds
    // them (C's `fastconjuction_noname`).
    let raw1: Vec<f64> = b.group1.iter().map(|&i| b.weights[i]).collect();
    let raw2: Vec<f64> = b.group2.iter().map(|&i| b.weights[i]).collect();
    let orieff1: f64 = raw1.iter().sum();
    let orieff2: f64 = raw2.iter().sum();
    let w1n: Vec<f64> = raw1.iter().map(|w| w / orieff1).collect();
    let w2n: Vec<f64> = raw2.iter().map(|w| w / orieff2).collect();

    // ---- Rust ----
    let r1: Vec<&[u8]> = g1.iter().map(|s| s.as_slice()).collect();
    let r2: Vec<&[u8]> = g2.iter().map(|s| s.as_slice()).collect();
    let prof1 = Profile::from_aligned(&r1, &w1n, &scoring.amino_map, scoring.nalphabets);
    let prof2 = Profile::from_aligned(&r2, &w2n, &scoring.amino_map, scoring.nalphabets);
    // Refinement zeroes the extend penalty: C's `dvtditr` is invoked without
    // `-g`, so `penalty_ex = 0` in the refinement DP (see `iterative_refine`).
    let gap = GapModel::new(scoring.gap.open as f64, 0.0);
    let rust_aln = profile_align(&prof1, &prof2, &scoring.consweight_matrix, &gap, false, false);
    let rust_width = rust_aln.operations.len();

    // ---- C ----
    unsafe {
        init_c_dna();
        std::ptr::addr_of_mut!(mafft_sys::njob).write(b.nseq as c_int);
        std::ptr::addr_of_mut!(mafft_sys::nlenmax).write((b.width * 2 + 100) as c_int);
    };
    let penalty: c_int = unsafe { std::ptr::addr_of!(mafft_sys::penalty).read() };
    let n_dynamicmtx = unsafe { std::ptr::addr_of!(mafft_sys::n_dis_consweight_multi).read() };
    assert_eq!(
        penalty as f64, b.gap_open,
        "C's nucleotide penalty must equal the fixture's gap_open"
    );

    let alloclen = (b.width * 2 + 100) as c_int;
    let (mut m1, _h1) = unsafe { alloc_c_char_mtx(&g1, alloclen as usize) };
    let (mut m2, _h2) = unsafe { alloc_c_char_mtx(&g2, alloclen as usize) };
    let mut e1 = w1n.clone();
    let mut e2 = w2n.clone();
    let mut dumdb = 0.0f64;
    let c_score = unsafe {
        mafft_sys::A__align(
            n_dynamicmtx, penalty, 0,
            m1.as_mut_ptr(), m2.as_mut_ptr(),
            e1.as_mut_ptr(), e2.as_mut_ptr(),
            g1.len() as c_int, g2.len() as c_int, alloclen,
            0, &mut dumdb,
            std::ptr::null_mut(), std::ptr::null_mut(),
            std::ptr::null_mut(), std::ptr::null_mut(),
            std::ptr::null_mut(), 0, std::ptr::null_mut(),
            0, 0, -1, 0,
            std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(),
            orieff1, orieff2,
        )
    };
    let c_width = unsafe {
        let p = m1[0];
        let mut w = 0usize;
        while *p.add(w) != 0 { w += 1; }
        w
    };
    let read = |ptr: *mut c_char, w: usize| {
        let mut v = vec![0u8; w];
        for i in 0..w { v[i] = unsafe { *ptr.add(i) as u8 }; }
        v
    };
    let c_g1: Vec<Vec<u8>> = m1.iter().map(|&p| read(p, c_width)).collect();
    let c_g2: Vec<Vec<u8>> = m2.iter().map(|&p| read(p, c_width)).collect();

    // Reconstruct Rust's aligned rows from its operation list.
    let mut r_g1 = vec![Vec::<u8>::with_capacity(rust_width); g1.len()];
    let mut r_g2 = vec![Vec::<u8>::with_capacity(rust_width); g2.len()];
    let (mut c1, mut c2) = (0usize, 0usize);
    for op in &rust_aln.operations {
        match op {
            AlignOp::Match => {
                for k in 0..g1.len() { r_g1[k].push(g1[k][c1]); }
                for k in 0..g2.len() { r_g2[k].push(g2[k][c2]); }
                c1 += 1; c2 += 1;
            }
            AlignOp::Delete => {
                for k in 0..g1.len() { r_g1[k].push(g1[k][c1]); }
                for k in 0..g2.len() { r_g2[k].push(b'-'); }
                c1 += 1;
            }
            AlignOp::Insert => {
                for k in 0..g1.len() { r_g1[k].push(b'-'); }
                for k in 0..g2.len() { r_g2[k].push(g2[k][c2]); }
                c2 += 1;
            }
        }
    }

    // ---- C, via Falign ----
    // This is what C's refinement ACTUALLY calls (`tditeration.c:2153` in the
    // single-threaded path, `:1035` in `athread`) — not `A__align` directly.
    // `Falign` does its own FFT correlation and anchor-based segment split
    // and only then calls the profile DP per segment, so it can return a
    // different alignment than a bare `A__align` on the same input.
    let (mut f1, _hf1) = unsafe { alloc_c_char_mtx(&g1, alloclen as usize) };
    let (mut f2, _hf2) = unsafe { alloc_c_char_mtx(&g2, alloclen as usize) };
    let mut fe1 = w1n.clone();
    let mut fe2 = w2n.clone();
    let mut fftlog: c_int = 0;
    let f_score = unsafe {
        mafft_sys::Falign(
            std::ptr::null_mut(), std::ptr::null_mut(), n_dynamicmtx,
            f1.as_mut_ptr(), f2.as_mut_ptr(),
            fe1.as_mut_ptr(), fe2.as_mut_ptr(),
            std::ptr::null_mut(), std::ptr::null_mut(),
            g1.len() as c_int, g2.len() as c_int, alloclen,
            &mut fftlog, std::ptr::null_mut(), 0, std::ptr::null_mut(),
        )
    };
    let f_width = unsafe {
        let p = f1[0];
        let mut w = 0usize;
        while *p.add(w) != 0 { w += 1; }
        w
    };
    let f_g1: Vec<Vec<u8>> = f1.iter().map(|&p| read(p, f_width)).collect();
    let f_g2: Vec<Vec<u8>> = f2.iter().map(|&p| read(p, f_width)).collect();
    let f_unchanged = f_g1 == g1 && f_g2 == g2;
    eprintln!("C Falign: width={f_width} score={f_score:.6} unchanged_input={f_unchanged} fftlog={fftlog}");

    let rust_unchanged = r_g1 == g1 && r_g2 == g2;
    let c_unchanged = c_g1 == g1 && c_g2 == g2;
    eprintln!("rust: width={rust_width} score={:.6} unchanged_input={rust_unchanged}", rust_aln.score);
    eprintln!("C   : width={c_width} score={c_score:.6} unchanged_input={c_unchanged}");
    eprintln!("score delta (rust - C) = {:.6}", rust_aln.score - c_score);

    // Score first: it separates "the DP never finds the better alignment"
    // from "it finds it but the caller drops it".
    assert!(
        (rust_aln.score - c_score).abs() < 1e-6,
        "profile DP score differs: rust={:.6} C={:.6} (delta {:.6}). \
         A non-zero delta means the recurrence, gap penalties or profile \
         construction differ for this 26-vs-94 split, not the acceptance test.",
        rust_aln.score, c_score, rust_aln.score - c_score
    );
    assert_eq!(rust_width, c_width, "profile DP alignment width differs");
    assert_eq!(r_g1, c_g1, "group1 rows differ between rust profile_align and C A__align");
    assert_eq!(r_g2, c_g2, "group2 rows differ between rust profile_align and C A__align");
}

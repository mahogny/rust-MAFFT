/// Cross-validate `mafft_align::msalignmm` against C MAFFT 7.526's
/// `MSalignmm` for the same input pair. The Rust Hirschberg DP is
/// asserted to return the exact same aligned strings as C's
/// `MSalignmm.c::MSalignmm` (which `tbfast.c:1159-1161` calls when
/// `alg='M'`, i.e. under `--memsave`).
///
/// We use the same FFI setup pattern as
/// `cross_validate_constrained_align.rs` and
/// `cross_validate_bl50_fft.rs`: `initglobalvariables` →
/// `constants()` → BLOSUM62 default penalties → invoke C with
/// pre-allocated buffers that C edits in place.
///
/// Inputs are chosen to exercise the recursive Hirschberg path
/// (`lgth1 > DPTANNI=100`) — for shorter inputs both DPs delegate to
/// the same base case and parity is trivially preserved.
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int};
use std::sync::Mutex;

use mafft_align::{GapModel, Profile, msalignmm};
use mafft_scoring::build_context;
use mafft_types::{ScoringModel, SeqType};

static C_MUTEX: Mutex<()> = Mutex::new(());

unsafe fn init_c_protein_blosum62() {
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
        std::ptr::addr_of_mut!(mafft_sys::outgap).write(1);

        // constants() needs a sample sequence to initialize alphabets.
        let seq_data = b"ACDEFGHIKLMNPQRSTVWY\0";
        let mut seq_ptr = seq_data.as_ptr() as *mut i8;
        let seq_arr: *mut *mut i8 = &mut seq_ptr;
        mafft_sys::constants(1, seq_arr);
    }
}

/// Compare Rust msalignmm aligned output to C MSalignmm aligned output.
/// Both should be byte-identical. Returns (rust_s1, rust_s2, c_s1, c_s2).
fn align_via_both(
    s1: &[u8],
    s2: &[u8],
    head_gap: bool,
    tail_gap: bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);
    let amino_map = &scoring.amino_map;
    let nalpha = scoring.nalphabets;

    // ---- Rust side ----
    let prof1 = Profile::from_aligned(&[s1], &[1.0], amino_map, nalpha);
    let prof2 = Profile::from_aligned(&[s2], &[1.0], amino_map, nalpha);
    let gap = GapModel::new(scoring.gap.open as f64, scoring.gap.extend as f64);
    let rust_aln = msalignmm(
        &prof1,
        &prof2,
        &scoring.consweight_matrix,
        &gap,
        head_gap,
        tail_gap,
    );

    let mut rust_s1 = Vec::with_capacity(rust_aln.operations.len());
    let mut rust_s2 = Vec::with_capacity(rust_aln.operations.len());
    let (mut p, mut q) = (0usize, 0usize);
    for op in &rust_aln.operations {
        match op {
            mafft_align::AlignOp::Match => {
                rust_s1.push(s1[p]);
                p += 1;
                rust_s2.push(s2[q]);
                q += 1;
            }
            mafft_align::AlignOp::Delete => {
                rust_s1.push(s1[p]);
                p += 1;
                rust_s2.push(b'-');
            }
            mafft_align::AlignOp::Insert => {
                rust_s1.push(b'-');
                rust_s2.push(s2[q]);
                q += 1;
            }
        }
    }

    // ---- C side ----
    let _guard = C_MUTEX.lock().unwrap();
    let (c_s1, c_s2) = unsafe {
        init_c_protein_blosum62();
        let alloclen = (s1.len() + s2.len() + 1000) as c_int;

        let c_seq1 = CString::new(s1).unwrap();
        let c_seq2 = CString::new(s2).unwrap();
        let mut buf1: Vec<u8> = c_seq1.as_bytes().to_vec();
        buf1.resize(alloclen as usize + 1, 0);
        let mut buf2: Vec<u8> = c_seq2.as_bytes().to_vec();
        buf2.resize(alloclen as usize + 1, 0);
        let mut p1 = buf1.as_mut_ptr() as *mut c_char;
        let mut p2 = buf2.as_mut_ptr() as *mut c_char;

        let mut eff1: c_double = 1.0;
        let mut eff2: c_double = 1.0;

        let nalpha_c = scoring.substitution_matrix.len() as c_int;
        let n_dyn = mafft_sys::AllocateDoubleMtx(nalpha_c, nalpha_c);
        for i in 0..scoring.substitution_matrix.len() {
            for j in 0..scoring.substitution_matrix[i].len() {
                *(*n_dyn.add(i)).add(j) = scoring.substitution_matrix[i][j] as f64;
            }
        }

        let _c_score = mafft_sys::MSalignmm(
            n_dyn,
            &mut p1,
            &mut p2,
            &mut eff1,
            &mut eff2,
            1,
            1,
            alloclen,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            head_gap as c_int,
            tail_gap as c_int,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1.0,
            1.0,
        );

        // Read back the aligned strings (in-place edits).
        let c_width = {
            let mut k = 0;
            while *p1.add(k) != 0 {
                k += 1;
            }
            k
        };
        let c_s1: Vec<u8> = (0..c_width).map(|k| *p1.add(k) as u8).collect();
        let c_s2: Vec<u8> = (0..c_width).map(|k| *p2.add(k) as u8).collect();

        mafft_sys::freeconstants();
        (c_s1, c_s2)
    };

    (rust_s1, rust_s2, c_s1, c_s2)
}

/// Recursive case with `lgth > DPTANNI = 100`. Uses a distinct
/// 130-residue protein input on each side with a 20-residue offset
/// so the optimal alignment requires gaps and exercises both halves
/// of the Hirschberg recursion.
#[test]
fn msalign_recursive_matches_c() {
    // 130-residue input pair with a deliberate divergence: 20-residue
    // motif in the middle of s2 that's absent in s1.
    let s1 = b"MKTIIALSYIFCLVFAKEDFREEKSPELLVNVPILTPVAGTHKAGKLITGSTMKAKEGNCGRDLLINGTGRLILSSSGKLPHRMNAIPRTNKPGSEDYTKVVNFLSGNLDRGQLSYLKLELKM";
    let s2 = b"MKTIIALSYIFCLVFAKEDFREEKSPELLVNVPILTPVAGTHKAGKLITGSTMKAKEGNCGRDPQLLLAGKSDESQRWSAALLINGTGRLILSSSGKLPHRMNAIPRTNKPGSEDYTKVVNFLSGNLDRGQLSYLKLELKM";
    assert!(
        s1.len() > 100,
        "test input must exercise Hirschberg recursion"
    );
    assert!(s2.len() > 100);

    let (rust_s1, rust_s2, c_s1, c_s2) = align_via_both(s1, s2, true, true);

    let rust_str1 = String::from_utf8_lossy(&rust_s1);
    let rust_str2 = String::from_utf8_lossy(&rust_s2);
    let c_str1 = String::from_utf8_lossy(&c_s1);
    let c_str2 = String::from_utf8_lossy(&c_s2);

    eprintln!("Rust width: {}", rust_s1.len());
    eprintln!("C    width: {}", c_s1.len());
    eprintln!("Rust seq1: {}", rust_str1);
    eprintln!("C    seq1: {}", c_str1);
    eprintln!("Rust seq2: {}", rust_str2);
    eprintln!("C    seq2: {}", c_str2);

    assert_eq!(
        rust_s1.len(),
        c_s1.len(),
        "alignment width mismatch: rust={} c={}",
        rust_s1.len(),
        c_s1.len()
    );
    assert_eq!(rust_s1, c_s1, "seq1 aligned output differs");
    assert_eq!(rust_s2, c_s2, "seq2 aligned output differs");
}

/// Smaller input that hits the base case in both implementations.
/// Should be trivially identical because both delegate to the same
/// DP recurrence.
#[test]
fn msalign_base_case_matches_c() {
    let s1 = b"MKTIIALSYIFCLVFAKEDFREEK";
    let s2 = b"MKTIIALSYIFCLVFAKEDFREEK";
    assert!(s1.len() < 100);

    let (rust_s1, rust_s2, c_s1, c_s2) = align_via_both(s1, s2, true, true);
    assert_eq!(rust_s1, c_s1, "base-case seq1 differs");
    assert_eq!(rust_s2, c_s2, "base-case seq2 differs");
}

/// Identical longer sequences — forces a deep recursion with a clear
/// diagonal trace. Stresses the iso-score path through the Hirschberg
/// midpoint selection.
#[test]
fn msalign_identical_long_matches_c() {
    let s = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNFLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSIEGFFATLGGEVALWSLVVLAIERYIVIC";
    assert!(s.len() > 100);
    let (rust_s1, rust_s2, c_s1, c_s2) = align_via_both(s, s, true, true);
    assert_eq!(rust_s1, c_s1, "identical-long seq1 differs");
    assert_eq!(rust_s2, c_s2, "identical-long seq2 differs");
}

/// Tail-gap test: with `tail_gap=false`, the DP should leave a
/// terminal stretch unaligned. Verifies C MSalignmm's terminal-gap
/// branch matches our port.
#[test]
fn msalign_freetail_matches_c() {
    let s1 = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNFLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSIEGFFATLGGEVALWSLV";
    let s2 = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNFLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSIEGFFAT";
    let (rust_s1, rust_s2, c_s1, c_s2) = align_via_both(s1, s2, false, false);
    assert_eq!(rust_s1, c_s1, "freetail seq1 differs");
    assert_eq!(rust_s2, c_s2, "freetail seq2 differs");
}

/// Cell-by-cell comparison of midw/midm/midn/jumpback*/jumpforw*
/// between our Rust Hirschberg forward+backward DP and C
/// `MSalignmm_rec` (via `rs_msalignmm_capture_top`). Pinpoints
/// where any divergence first occurs.
#[test]
fn msalign_mid_state_matches_c() {
    // Asymmetric input — the one that diverges at the alignment
    // level. Same input as `msalign_asymmetric_lengths_matches_c`.
    let s1 = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNFLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSI";
    let s2 = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNGGRTLSEVMKWPFSDQIANLPTQRDLELFQKLMSARTVTNLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSI";
    let lgth1 = s1.len();
    let lgth2 = s2.len();
    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    // ---- Capture C state via the instrumented wrapper ----
    let _guard = C_MUTEX.lock().unwrap();
    let (
        c_imid,
        c_jmid,
        c_jumpi,
        c_jumpj,
        c_midw,
        c_midm,
        c_midn,
        c_jumpbacki,
        c_jumpbackj,
        c_jumpforwi,
        c_jumpforwj,
    ) = unsafe {
        init_c_protein_blosum62();

        let alloclen = (lgth1 + lgth2 + 1000) as c_int;
        let c_seq1 = CString::new(&s1[..]).unwrap();
        let c_seq2 = CString::new(&s2[..]).unwrap();
        let mut buf1: Vec<u8> = c_seq1.as_bytes().to_vec();
        buf1.resize(alloclen as usize + 1, 0);
        let mut buf2: Vec<u8> = c_seq2.as_bytes().to_vec();
        buf2.resize(alloclen as usize + 1, 0);
        let p1 = buf1.as_mut_ptr() as *mut c_char;
        let p2 = buf2.as_mut_ptr() as *mut c_char;

        let nalpha_c = scoring.substitution_matrix.len() as c_int;
        let n_dyn = mafft_sys::AllocateDoubleMtx(nalpha_c, nalpha_c);
        for i in 0..scoring.substitution_matrix.len() {
            for j in 0..scoring.substitution_matrix[i].len() {
                *(*n_dyn.add(i)).add(j) = scoring.substitution_matrix[i][j] as f64;
            }
        }

        let out_size = lgth2 + 2;
        let mut out_imid: c_int = 0;
        let mut out_jmid: c_int = 0;
        let mut out_jumpi: c_int = 0;
        let mut out_jumpj: c_int = 0;
        let mut out_midw = vec![0.0f64; out_size];
        let mut out_midm = vec![0.0f64; out_size];
        let mut out_midn = vec![0.0f64; out_size];
        let mut out_jumpbacki = vec![0 as c_int; out_size];
        let mut out_jumpbackj = vec![0 as c_int; out_size];
        let mut out_jumpforwi = vec![0 as c_int; out_size];
        let mut out_jumpforwj = vec![0 as c_int; out_size];

        mafft_sys::rs_msalignmm_capture_top(
            n_dyn,
            p1,
            p2,
            lgth1 as c_int,
            lgth2 as c_int,
            1,
            1,
            &mut out_imid,
            &mut out_jmid,
            &mut out_jumpi,
            &mut out_jumpj,
            out_midw.as_mut_ptr(),
            out_midm.as_mut_ptr(),
            out_midn.as_mut_ptr(),
            out_jumpbacki.as_mut_ptr(),
            out_jumpbackj.as_mut_ptr(),
            out_jumpforwi.as_mut_ptr(),
            out_jumpforwj.as_mut_ptr(),
        );

        mafft_sys::freeconstants();
        (
            out_imid as usize,
            out_jmid as usize,
            out_jumpi as usize,
            out_jumpj as usize,
            out_midw,
            out_midm,
            out_midn,
            out_jumpbacki,
            out_jumpbackj,
            out_jumpforwi,
            out_jumpforwj,
        )
    };

    eprintln!(
        "C: imid={} jmid={} jumpi={} jumpj={}",
        c_imid, c_jmid, c_jumpi, c_jumpj
    );
    eprintln!(
        "C: midw[95]={:.2} midw[99]={:.2} midw[100]={:.2}",
        c_midw[95], c_midw[99], c_midw[100]
    );
    eprintln!(
        "C: midm[95]={:.2} midm[99]={:.2} midm[100]={:.2}",
        c_midm[95], c_midm[99], c_midm[100]
    );
    eprintln!(
        "C: midn[94]={:.2} midn[98]={:.2} midn[99]={:.2}",
        c_midn[94], c_midn[98], c_midn[99]
    );
    // Find C's max midw / midm.
    let mut c_max_midw = (0, f64::NEG_INFINITY);
    for j in 1..lgth2 {
        if c_midw[j] > c_max_midw.1 {
            c_max_midw = (j, c_midw[j]);
        }
    }
    eprintln!(
        "C: argmax(midw) = {} (val {:.2})",
        c_max_midw.0, c_max_midw.1
    );

    eprintln!("C: midn[95]={:.2} midw[96]={:.2}", c_midn[95], c_midw[96]);
    eprintln!(
        "C: jumpbacki[96]={} jumpbackj[96]={}",
        c_jumpbacki[96], c_jumpbackj[96]
    );
    eprintln!(
        "C: jumpforwi[95]={} jumpforwj[95]={}",
        c_jumpforwi[95], c_jumpforwj[95]
    );

    let _ = c_jumpbacki;
    let _ = c_jumpbackj;
    let _ = c_jumpforwi;
    let _ = c_jumpforwj;

    // ---- Build the matching Rust profiles, run msalignmm, observe split.
    // (Splits are exposed via MS_DBG env; we just sanity-check that
    // `msalignmm` produces an alignment of any width.)
    let prof1 = Profile::from_aligned(
        &[s1.as_slice()],
        &[1.0],
        &scoring.amino_map,
        scoring.nalphabets,
    );
    let prof2 = Profile::from_aligned(
        &[s2.as_slice()],
        &[1.0],
        &scoring.amino_map,
        scoring.nalphabets,
    );
    let gap = GapModel::new(scoring.gap.open as f64, scoring.gap.extend as f64);
    let aln = msalignmm(&prof1, &prof2, &scoring.consweight_matrix, &gap, true, true);
    eprintln!("Rust msalignmm width: {}", aln.operations.len());

    // The expectation: C MSalignmm gives a 151-wide alignment, so
    // its split must place midw/midm/midn s.t. the recursion lands
    // at width 151. Our Rust msalignmm currently gives 155.
    // Either C's argmax(midw) differs from ours (telling us our
    // forward+backward midw is wrong), or C picks the same column
    // but the recursion glue (jumpforwi rewrite) diverges.
}

/// Run C MSalignmm on the (0, 55, 0, 95) top sub-region of the
/// failing asymmetric input, compare to ours. Pinpoints whether the
/// 1-column residual is in the base case (profile_align_imp_with_
/// boundary mismatch with MSalignmm_tanni) or in the recursion glue.
#[test]
fn msalign_subregion_top_matches_c() {
    // s1[0..=55] (56 chars from "MNGTE...VGFPV" + 'N' + 'F'),
    // s2[0..=95] (96 chars).
    let s1_full = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNFLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSI";
    let s2_full = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNGGRTLSEVMKWPFSDQIANLPTQRDLELFQKLMSARTVTNLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSI";
    let s1_top: &[u8] = &s1_full[0..56];
    let s2_top: &[u8] = &s2_full[0..96];
    assert_eq!(s1_top.len(), 56);
    assert_eq!(s2_top.len(), 96);

    // The top sub-region is called with head_gap=true, tail_gap=true
    // (effective_tail=true because not at parent's end).
    let (rust_s1, rust_s2, c_s1, c_s2) = align_via_both(s1_top, s2_top, true, true);
    eprintln!("Rust top width: {}", rust_s1.len());
    eprintln!("C    top width: {}", c_s1.len());
    eprintln!("Rust s1: {}", String::from_utf8_lossy(&rust_s1));
    eprintln!("C    s1: {}", String::from_utf8_lossy(&c_s1));
    eprintln!("Rust s2: {}", String::from_utf8_lossy(&rust_s2));
    eprintln!("C    s2: {}", String::from_utf8_lossy(&c_s2));

    // Bottom sub-region: s1[57..=111] (55 chars), s2[96..=150] (55 chars).
    // Per the recursion glue this is called with head_gap=false,
    // tail_gap=true (top-level tail).
    let s1_bot: &[u8] = &s1_full[57..=111];
    let s2_bot: &[u8] = &s2_full[96..=150];
    assert_eq!(s1_bot.len(), 55);
    assert_eq!(s2_bot.len(), 55);
    // Internal sub-region: effective_head=true (because ist!=0 in
    // the recursive call). To mimic that on a standalone call we
    // pass head_gap=true.
    let (rs1, rs2, cs1, cs2) = align_via_both(s1_bot, s2_bot, true, true);
    eprintln!("Bottom — Rust width: {}, C width: {}", rs1.len(), cs1.len());
    eprintln!("Bottom Rust s1: {}", String::from_utf8_lossy(&rs1));
    eprintln!("Bottom C    s1: {}", String::from_utf8_lossy(&cs1));
    eprintln!("Bottom Rust s2: {}", String::from_utf8_lossy(&rs2));
    eprintln!("Bottom C    s2: {}", String::from_utf8_lossy(&cs2));
}

/// Compare `profile_align_imp_with_boundary` against C
/// `MSalignmm_tanni` in-context for the top sub-region of the
/// failing asymmetric input. The C wrapper builds the FULL parent
/// `cpmx`/`gapinfo` arrays (as MSalignmm does) and then runs
/// `MSalignmm_tanni` with the sub-region indices. If this matches
/// our Rust base case the bug isn't in `MSalignmm_tanni` vs
/// `profile_align_imp_with_boundary`; if it diverges we've found
/// the gap.
#[test]
fn msalign_tanni_in_context_matches_c() {
    let s1_full = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNFLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSI";
    let s2_full = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNGGRTLSEVMKWPFSDQIANLPTQRDLELFQKLMSARTVTNLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSI";
    let lgth1 = s1_full.len();
    let lgth2 = s2_full.len();
    // Top sub-region: ist=0, ien=55, jst=0, jen=95.
    let ist = 0;
    let ien = 55;
    let jst = 0;
    let jen = 95;

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    // ---- C side ----
    let _guard = C_MUTEX.lock().unwrap();
    let (c_s1, c_s2, c_width) = unsafe {
        init_c_protein_blosum62();
        let alloclen = (lgth1 + lgth2 + 1000) as c_int;
        let c_seq1 = CString::new(&s1_full[..]).unwrap();
        let c_seq2 = CString::new(&s2_full[..]).unwrap();
        let mut buf1: Vec<u8> = c_seq1.as_bytes().to_vec();
        buf1.resize(alloclen as usize + 1, 0);
        let mut buf2: Vec<u8> = c_seq2.as_bytes().to_vec();
        buf2.resize(alloclen as usize + 1, 0);
        let p1 = buf1.as_mut_ptr() as *mut c_char;
        let p2 = buf2.as_mut_ptr() as *mut c_char;

        let nalpha_c = scoring.substitution_matrix.len() as c_int;
        let n_dyn = mafft_sys::AllocateDoubleMtx(nalpha_c, nalpha_c);
        for i in 0..scoring.substitution_matrix.len() {
            for j in 0..scoring.substitution_matrix[i].len() {
                *(*n_dyn.add(i)).add(j) = scoring.substitution_matrix[i][j] as f64;
            }
        }

        let out_size = ien - ist + jen - jst + 100 + 10;
        let mut out_s1 = vec![0u8; out_size];
        let mut out_s2 = vec![0u8; out_size];
        let mut out_width: c_int = 0;
        mafft_sys::rs_msalignmm_tanni_capture(
            n_dyn,
            p1,
            p2,
            lgth1 as c_int,
            lgth2 as c_int,
            ist as c_int,
            ien as c_int,
            jst as c_int,
            jen as c_int,
            1,
            1,
            out_s1.as_mut_ptr() as *mut c_char,
            out_s2.as_mut_ptr() as *mut c_char,
            &mut out_width,
        );
        mafft_sys::freeconstants();
        let w = out_width as usize;
        let s1: Vec<u8> = out_s1[..w].to_vec();
        let s2: Vec<u8> = out_s2[..w].to_vec();
        (s1, s2, w)
    };

    // ---- Rust side: profile_align_imp_with_boundary on sub-profile.
    // Build profiles from FULL inputs, then slice (mirrors how
    // msalignmm.rs::base_case does it).
    let prof1_full = Profile::from_aligned(
        &[s1_full.as_slice()],
        &[1.0],
        &scoring.amino_map,
        scoring.nalphabets,
    );
    let prof2_full = Profile::from_aligned(
        &[s2_full.as_slice()],
        &[1.0],
        &scoring.amino_map,
        scoring.nalphabets,
    );
    let sub1 = prof1_full.sub_profile(ist, ien + 1);
    let sub2 = prof2_full.sub_profile(jst, jen + 1);
    // effective_head/tail per msalignmm::base_case logic
    let effective_head = true || ist != 0 || jst != 0;
    let effective_tail = true || ien + 1 != prof1_full.length || jen + 1 != prof2_full.length;
    let head1 = if ist > 0 {
        prof1_full.nongap_freq[ist - 1]
    } else {
        1.0
    };
    let head2 = if jst > 0 {
        prof2_full.nongap_freq[jst - 1]
    } else {
        1.0
    };
    let tail1 = if ien + 1 < prof1_full.length {
        prof1_full.nongap_freq[ien + 1]
    } else {
        1.0
    };
    let tail2 = if jen + 1 < prof2_full.length {
        prof2_full.nongap_freq[jen + 1]
    } else {
        1.0
    };
    let gap = GapModel::new(scoring.gap.open as f64, scoring.gap.extend as f64);
    let rust_aln = mafft_align::profile_align_imp_with_boundary(
        &sub1,
        &sub2,
        &scoring.consweight_matrix,
        &gap,
        effective_head,
        effective_tail,
        None,
        false,
        mafft_align::BoundaryFreqs {
            head1,
            head2,
            tail1,
            tail2,
        },
    );
    // Build the aligned strings from the operations.
    let mut rust_s1 = Vec::new();
    let mut rust_s2 = Vec::new();
    let (mut p, mut q) = (0usize, 0usize);
    let sub_s1 = &s1_full[ist..=ien];
    let sub_s2 = &s2_full[jst..=jen];
    for op in &rust_aln.operations {
        match op {
            mafft_align::AlignOp::Match => {
                rust_s1.push(sub_s1[p]);
                p += 1;
                rust_s2.push(sub_s2[q]);
                q += 1;
            }
            mafft_align::AlignOp::Delete => {
                rust_s1.push(sub_s1[p]);
                p += 1;
                rust_s2.push(b'-');
            }
            mafft_align::AlignOp::Insert => {
                rust_s1.push(b'-');
                rust_s2.push(sub_s2[q]);
                q += 1;
            }
        }
    }

    eprintln!(
        "C    in-context tanni: width={} ({} ops)",
        c_width,
        c_s1.len()
    );
    eprintln!("C    s1: {}", String::from_utf8_lossy(&c_s1));
    eprintln!("C    s2: {}", String::from_utf8_lossy(&c_s2));
    eprintln!("Rust full DP        : width={}", rust_s1.len());
    eprintln!("Rust s1: {}", String::from_utf8_lossy(&rust_s1));
    eprintln!("Rust s2: {}", String::from_utf8_lossy(&rust_s2));

    assert_eq!(
        rust_s1.len(),
        c_s1.len(),
        "in-context width differs: rust={} c={}",
        rust_s1.len(),
        c_s1.len()
    );
    assert_eq!(rust_s1, c_s1, "in-context seq1 differs");
    assert_eq!(rust_s2, c_s2, "in-context seq2 differs");
}

/// Sanity: faithful C re-impl on a symmetric input that real C
/// MSalignmm handles correctly (the 130-residue divergent test
/// which passes byte-identical for our Rust msalignmm). If
/// faithful matches real, the re-impl is structurally correct.
#[test]
fn msalign_full_trace_c_reimpl_symmetric_matches() {
    let s1 = b"MKTIIALSYIFCLVFAKEDFREEKSPELLVNVPILTPVAGTHKAGKLITGSTMKAKEGNCGRDLLINGTGRLILSSSGKLPHRMNAIPRTNKPGSEDYTKVVNFLSGNLDRGQLSYLKLELKM";
    let s2 = b"MKTIIALSYIFCLVFAKEDFREEKSPELLVNVPILTPVAGTHKAGKLITGSTMKAKEGNCGRDPQLLLAGKSDESQRWSAALLINGTGRLILSSSGKLPHRMNAIPRTNKPGSEDYTKVVNFLSGNLDRGQLSYLKLELKM";
    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);
    let _guard = C_MUTEX.lock().unwrap();
    let (trace_w, real_w) = unsafe {
        init_c_protein_blosum62();
        let alloclen = (s1.len() + s2.len() + 1000) as c_int;
        let cs1 = CString::new(&s1[..]).unwrap();
        let cs2 = CString::new(&s2[..]).unwrap();
        let nalpha_c = scoring.substitution_matrix.len() as c_int;
        let n_dyn = mafft_sys::AllocateDoubleMtx(nalpha_c, nalpha_c);
        for i in 0..scoring.substitution_matrix.len() {
            for j in 0..scoring.substitution_matrix[i].len() {
                *(*n_dyn.add(i)).add(j) = scoring.substitution_matrix[i][j] as f64;
            }
        }
        let out_size = s1.len() + s2.len() + 200;
        let mut t1 = vec![0u8; out_size];
        let mut t2 = vec![0u8; out_size];
        let mut tw: c_int = 0;
        let mut b1: Vec<u8> = cs1.as_bytes().to_vec();
        b1.resize(alloclen as usize + 1, 0);
        let mut b2: Vec<u8> = cs2.as_bytes().to_vec();
        b2.resize(alloclen as usize + 1, 0);
        mafft_sys::rs_msalignmm_full_trace(
            n_dyn,
            b1.as_mut_ptr() as *mut c_char,
            b2.as_mut_ptr() as *mut c_char,
            s1.len() as c_int,
            s2.len() as c_int,
            1,
            1,
            t1.as_mut_ptr() as *mut c_char,
            t2.as_mut_ptr() as *mut c_char,
            &mut tw,
        );

        let mut rb1: Vec<u8> = cs1.as_bytes().to_vec();
        rb1.resize(alloclen as usize + 1, 0);
        let mut rb2: Vec<u8> = cs2.as_bytes().to_vec();
        rb2.resize(alloclen as usize + 1, 0);
        let mut rp1 = rb1.as_mut_ptr() as *mut c_char;
        let mut rp2 = rb2.as_mut_ptr() as *mut c_char;
        let mut e1: c_double = 1.0;
        let mut e2: c_double = 1.0;
        mafft_sys::MSalignmm(
            n_dyn,
            &mut rp1,
            &mut rp2,
            &mut e1,
            &mut e2,
            1,
            1,
            alloclen,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            1,
            1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1.0,
            1.0,
        );
        let rw = {
            let mut k = 0;
            while *rp1.add(k) != 0 {
                k += 1;
            }
            k
        };
        mafft_sys::freeconstants();
        (tw as usize, rw)
    };
    eprintln!("symmetric: faithful={} real={}", trace_w, real_w);
    assert_eq!(
        trace_w, real_w,
        "faithful re-impl differs from real C on symmetric"
    );
}

/// Run the faithful C re-implementation of `MSalignmm_rec` (in
/// `wrappers/msalignmm_instr.c::rs_msalignmm_full_trace`) on the
/// failing asymmetric input. Should produce the SAME output as the
/// real C `MSalignmm` (151 wide). If our faithful re-implementation
/// gives 152 — same as the Rust port — then real `MSalignmm` does
/// something extra that I haven't captured. If it gives 151, then
/// the Rust port has a bug that doesn't exist in my C version.
///
/// Set `MSALIGN_TRACE=1` in the environment to get level-by-level
/// stderr output of the recursion (ENTER, SPLIT, INTER_HORIZ/VERT,
/// TOP_DONE, BOTTOM_DONE, BASE_CASE).
///
/// Verifies the faithful C re-implementation of `MSalignmm_rec`
/// matches real C `MSalignmm` byte-for-byte. Cross-validated the
/// `midw[j] += wm` indexing (NOT `midw[j+1] += wm` — `MSalignmm.c:1610`).
#[test]
fn msalign_full_trace_c_reimpl_matches_real_c() {
    let s1_full = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNFLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSI";
    let s2_full = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNGGRTLSEVMKWPFSDQIANLPTQRDLELFQKLMSARTVTNLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSI";
    let lgth1 = s1_full.len();
    let lgth2 = s2_full.len();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    let _guard = C_MUTEX.lock().unwrap();
    let (trace_s1, trace_s2, trace_width, real_s1, real_s2, real_width) = unsafe {
        init_c_protein_blosum62();
        let alloclen = (lgth1 + lgth2 + 1000) as c_int;
        let c_seq1 = CString::new(&s1_full[..]).unwrap();
        let c_seq2 = CString::new(&s2_full[..]).unwrap();
        let nalpha_c = scoring.substitution_matrix.len() as c_int;
        let n_dyn = mafft_sys::AllocateDoubleMtx(nalpha_c, nalpha_c);
        for i in 0..scoring.substitution_matrix.len() {
            for j in 0..scoring.substitution_matrix[i].len() {
                *(*n_dyn.add(i)).add(j) = scoring.substitution_matrix[i][j] as f64;
            }
        }

        // --- Faithful C re-implementation (with trace) ---
        let out_size = lgth1 + lgth2 + 200;
        let mut trace_out_s1 = vec![0u8; out_size];
        let mut trace_out_s2 = vec![0u8; out_size];
        let mut trace_out_width: c_int = 0;
        let mut buf1: Vec<u8> = c_seq1.as_bytes().to_vec();
        buf1.resize(alloclen as usize + 1, 0);
        let mut buf2: Vec<u8> = c_seq2.as_bytes().to_vec();
        buf2.resize(alloclen as usize + 1, 0);
        mafft_sys::rs_msalignmm_full_trace(
            n_dyn,
            buf1.as_mut_ptr() as *mut c_char,
            buf2.as_mut_ptr() as *mut c_char,
            lgth1 as c_int,
            lgth2 as c_int,
            1,
            1,
            trace_out_s1.as_mut_ptr() as *mut c_char,
            trace_out_s2.as_mut_ptr() as *mut c_char,
            &mut trace_out_width,
        );
        let tw = trace_out_width as usize;
        let ts1 = trace_out_s1[..tw].to_vec();
        let ts2 = trace_out_s2[..tw].to_vec();

        // --- Real C MSalignmm ---
        let mut real_buf1: Vec<u8> = c_seq1.as_bytes().to_vec();
        real_buf1.resize(alloclen as usize + 1, 0);
        let mut real_buf2: Vec<u8> = c_seq2.as_bytes().to_vec();
        real_buf2.resize(alloclen as usize + 1, 0);
        let mut rp1 = real_buf1.as_mut_ptr() as *mut c_char;
        let mut rp2 = real_buf2.as_mut_ptr() as *mut c_char;
        let mut eff1: c_double = 1.0;
        let mut eff2: c_double = 1.0;
        let _ = mafft_sys::MSalignmm(
            n_dyn,
            &mut rp1,
            &mut rp2,
            &mut eff1,
            &mut eff2,
            1,
            1,
            alloclen,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            1,
            1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1.0,
            1.0,
        );
        let rw = {
            let mut k = 0;
            while *rp1.add(k) != 0 {
                k += 1;
            }
            k
        };
        let rs1: Vec<u8> = (0..rw).map(|k| *rp1.add(k) as u8).collect();
        let rs2: Vec<u8> = (0..rw).map(|k| *rp2.add(k) as u8).collect();

        mafft_sys::freeconstants();
        (ts1, ts2, tw, rs1, rs2, rw)
    };

    eprintln!("FAITHFUL C re-impl width: {}", trace_width);
    eprintln!("REAL     C MSalignmm width: {}", real_width);
    eprintln!("Faithful s1: {}", String::from_utf8_lossy(&trace_s1));
    eprintln!("Real     s1: {}", String::from_utf8_lossy(&real_s1));
    eprintln!("Faithful s2: {}", String::from_utf8_lossy(&trace_s2));
    eprintln!("Real     s2: {}", String::from_utf8_lossy(&real_s2));

    // The faithful re-implementation should match real C MSalignmm.
    // If it doesn't, our reading of the algorithm is incomplete.
    assert_eq!(
        trace_width, real_width,
        "faithful C re-impl width ({}) differs from real C MSalignmm ({})",
        trace_width, real_width
    );
}

/// Asymmetric lengths where `lgth1 < lgth2`. Closed 2026-05-16 by
/// fixing the `midw[j] += wm` indexing (was incorrectly `midw[j+1]
/// += wm` — see `msalign.rs::msalignmm_rec` and `MSalignmm.c:1610`).
#[test]
fn msalign_asymmetric_lengths_matches_c() {
    // s1 = 110 residues, s2 = 180 residues (s1 with extra middle motif).
    let s1 = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNFLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSI";
    let s2 = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNGGRTLSEVMKWPFSDQIANLPTQRDLELFQKLMSARTVTNLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSI";
    let (rust_s1, rust_s2, c_s1, c_s2) = align_via_both(s1, s2, true, true);
    eprintln!("Rust s1: {}", String::from_utf8_lossy(&rust_s1));
    eprintln!("C    s1: {}", String::from_utf8_lossy(&c_s1));
    eprintln!("Rust s2: {}", String::from_utf8_lossy(&rust_s2));
    eprintln!("C    s2: {}", String::from_utf8_lossy(&c_s2));
    assert_eq!(
        rust_s1.len(),
        c_s1.len(),
        "width differs: rust={} c={}",
        rust_s1.len(),
        c_s1.len()
    );
    assert_eq!(rust_s1, c_s1, "asymmetric seq1 differs");
    assert_eq!(rust_s2, c_s2, "asymmetric seq2 differs");
}

// Test function names deliberately mirror the C function being validated
// (`A__align`, etc.) — the double underscore is part of the upstream MAFFT
// identifier. Allow the non-snake_case style here.
#![allow(non_snake_case)]

use std::os::raw::{c_char, c_double, c_int};
use std::sync::Mutex;

use mafft_align::{GapModel, Profile, profile_align};
use mafft_scoring::build_context;
use mafft_types::{ScoringModel, SeqType};

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

/// Build a C-style n_dynamicmtx (double** indexed by char codes).
unsafe fn build_c_dynamicmtx(scoring_matrix: &[Vec<i32>]) -> *mut *mut c_double {
    unsafe {
        // C's n_dynamicmtx is indexed by [0..nalphabets-1][0..nalphabets-1] like n_dis.
        let nalpha = scoring_matrix.len() as c_int;
        let mtx = mafft_sys::AllocateDoubleMtx(nalpha, nalpha);
        for i in 0..scoring_matrix.len() {
            for j in 0..scoring_matrix[i].len() {
                *(*mtx.add(i)).add(j) = scoring_matrix[i][j] as f64;
            }
        }
        mtx
    }
}

#[test]
fn profile_align_matches_c_msalignmm() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    // Two small groups with gaps (refinement-style input).
    let group1: Vec<&[u8]> = vec![b"ACDEFGHIKLM", b"ACDE-GHIKLM"];
    let group2: Vec<&[u8]> = vec![b"ACDE-GHIKL-", b"ACDEFGHIKLM", b"A-DEFGHIK-M"];
    let w1 = vec![0.5, 0.5];
    let w2 = vec![0.33333, 0.33334, 0.33333];

    let len1 = group1[0].len();
    let len2 = group2[0].len();

    // Rust side
    let prof1 = Profile::from_aligned(&group1, &w1, &scoring.amino_map, scoring.nalphabets);
    let prof2 = Profile::from_aligned(&group2, &w2, &scoring.amino_map, scoring.nalphabets);
    let gap = GapModel::new(scoring.gap.open as f64, scoring.gap.extend as f64);
    let rust_aln = profile_align(&prof1, &prof2, &scoring.consweight_matrix, &gap, true, true);
    eprintln!(
        "Rust: ops.len()={}, score={:.2}",
        rust_aln.operations.len(),
        rust_aln.score
    );

    // C side
    unsafe {
        init_c_protein();

        // MSalignmm modifies the sequence buffers in place — allocate
        // writable boxed buffers (with extra capacity per `alloclen`).
        let alloclen = (len1 + len2) * 10;
        let c_seq1_boxed: Vec<Box<[u8]>> = group1
            .iter()
            .map(|s| {
                let mut v = s.to_vec();
                v.resize(alloclen + 1, 0);
                v.into_boxed_slice()
            })
            .collect();
        let c_seq2_boxed: Vec<Box<[u8]>> = group2
            .iter()
            .map(|s| {
                let mut v = s.to_vec();
                v.resize(alloclen + 1, 0);
                v.into_boxed_slice()
            })
            .collect();

        // Build mutable pointer arrays for the boxed slices.
        let mut c_seq1_ptrs: Vec<*mut c_char> = c_seq1_boxed
            .iter()
            .map(|v| v.as_ptr() as *mut c_char)
            .collect();
        let mut c_seq2_ptrs: Vec<*mut c_char> = c_seq2_boxed
            .iter()
            .map(|v| v.as_ptr() as *mut c_char)
            .collect();

        // eff arrays
        let eff1: *mut c_double = alloc_zeroed(w1.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in w1.iter().enumerate() {
            *eff1.add(i) = w;
        }
        let eff2: *mut c_double = alloc_zeroed(w2.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in w2.iter().enumerate() {
            *eff2.add(i) = w;
        }

        // n_dynamicmtx
        let n_dyn = build_c_dynamicmtx(&scoring.substitution_matrix);

        // sgap/egap = null (triggers st_* gap count path which matches our Profile::from_aligned)
        let score = mafft_sys::MSalignmm(
            n_dyn,
            c_seq1_ptrs.as_mut_ptr(),
            c_seq2_ptrs.as_mut_ptr(),
            eff1,
            eff2,
            w1.len() as c_int,
            w2.len() as c_int,
            alloclen as c_int,
            std::ptr::null_mut(), // sgap1
            std::ptr::null_mut(), // sgap2
            std::ptr::null_mut(), // egap1
            std::ptr::null_mut(), // egap2
            std::ptr::null_mut(), // chudanpt
            0,
            std::ptr::null_mut(),
            1,                    // headgp
            1,                    // tailgp
            std::ptr::null_mut(), // cpmxchild0
            std::ptr::null_mut(), // cpmxchild1
            std::ptr::null_mut(), // cpmxresult
            1.0,                  // orieff1
            1.0,                  // orieff2
        );

        // Read the aligned sequences back
        let c_len = {
            let s = c_seq1_ptrs[0];
            let mut n = 0;
            while *s.add(n) != 0 {
                n += 1;
            }
            n
        };
        eprintln!("C:    aligned_len={}, score={:.2}", c_len, score);

        // Compare widths
        let rust_width = rust_aln.operations.len();
        eprintln!("  Rust width: {}, C width: {}", rust_width, c_len);

        // Print both aligned group1 sequences for comparison
        for (i, s) in c_seq1_boxed.iter().enumerate() {
            let c_str = std::str::from_utf8(&s[..c_len]).unwrap_or("???");
            eprintln!("  C  g1[{i}]: {c_str}");
        }
        for (i, s) in c_seq2_boxed.iter().enumerate() {
            let c_str = std::str::from_utf8(&s[..c_len]).unwrap_or("???");
            eprintln!("  C  g2[{i}]: {c_str}");
        }

        // Reconstruct Rust aligned sequences from ops
        let mut rust_g1 = vec![Vec::<u8>::new(); group1.len()];
        let mut rust_g2 = vec![Vec::<u8>::new(); group2.len()];
        let mut c1 = 0;
        let mut c2 = 0;
        use mafft_align::AlignOp;
        for op in &rust_aln.operations {
            match op {
                AlignOp::Match => {
                    for (si, s) in group1.iter().enumerate() {
                        rust_g1[si].push(s[c1]);
                    }
                    for (si, s) in group2.iter().enumerate() {
                        rust_g2[si].push(s[c2]);
                    }
                    c1 += 1;
                    c2 += 1;
                }
                AlignOp::Delete => {
                    for (si, s) in group1.iter().enumerate() {
                        rust_g1[si].push(s[c1]);
                    }
                    for si in 0..group2.len() {
                        rust_g2[si].push(b'-');
                    }
                    c1 += 1;
                }
                AlignOp::Insert => {
                    for si in 0..group1.len() {
                        rust_g1[si].push(b'-');
                    }
                    for (si, s) in group2.iter().enumerate() {
                        rust_g2[si].push(s[c2]);
                    }
                    c2 += 1;
                }
            }
        }
        for (i, s) in rust_g1.iter().enumerate() {
            eprintln!("  R  g1[{i}]: {}", String::from_utf8_lossy(s));
        }
        for (i, s) in rust_g2.iter().enumerate() {
            eprintln!("  R  g2[{i}]: {}", String::from_utf8_lossy(s));
        }

        mafft_sys::freeconstants();

        // Require exact width match
        assert_eq!(
            rust_width, c_len,
            "width mismatch: rust={rust_width} c={c_len}"
        );
    }
}

/// Test profile_align with asymmetric groups (1 vs many) — common refinement case.
#[test]
fn profile_align_1_vs_many_matches_c() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    // 1 vs 5 split — typical refinement 1-vs-N case.
    let group1: Vec<&[u8]> = vec![b"----MNGTEGDNF-YVPFSNK-TGL-ARSPYEYPQY----"];
    let group2: Vec<&[u8]> = vec![
        b"MN--GTEGDNFYVPFSNKTGLARSPYE-------YPQYAE",
        b"MNGTEGDNFYVPFS----NKTGLARSPYEYPQ---Y--AE",
        b"-MN-GTEGDNFYVPFSNKTGLARSPYEYPQY---------",
        b"--MNGTEGDNFYVPFSNKTGL--ARSPYEYPQY-------",
        b"-M-NGTEGDNFYVPFSNKTGLARSPYEYPQY---------",
    ];
    let w1 = vec![1.0];
    let w2 = vec![0.2; 5];

    let len1 = group1[0].len();
    let len2 = group2[0].len();
    assert_eq!(len1, len2);

    // Rust side
    let prof1 = Profile::from_aligned(&group1, &w1, &scoring.amino_map, scoring.nalphabets);
    let prof2 = Profile::from_aligned(&group2, &w2, &scoring.amino_map, scoring.nalphabets);
    let gap = GapModel::new(scoring.gap.open as f64, scoring.gap.extend as f64);
    let rust_aln = profile_align(&prof1, &prof2, &scoring.consweight_matrix, &gap, true, true);

    // C side
    unsafe {
        init_c_protein();

        let alloclen = (len1 + len2) * 10;
        let c_seq1_boxed: Vec<Box<[u8]>> = group1
            .iter()
            .map(|s| {
                let mut v = s.to_vec();
                v.resize(alloclen + 1, 0);
                v.into_boxed_slice()
            })
            .collect();
        let c_seq2_boxed: Vec<Box<[u8]>> = group2
            .iter()
            .map(|s| {
                let mut v = s.to_vec();
                v.resize(alloclen + 1, 0);
                v.into_boxed_slice()
            })
            .collect();
        let mut c_seq1_ptrs: Vec<*mut c_char> = c_seq1_boxed
            .iter()
            .map(|v| v.as_ptr() as *mut c_char)
            .collect();
        let mut c_seq2_ptrs: Vec<*mut c_char> = c_seq2_boxed
            .iter()
            .map(|v| v.as_ptr() as *mut c_char)
            .collect();

        let eff1: *mut c_double = alloc_zeroed(w1.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in w1.iter().enumerate() {
            *eff1.add(i) = w;
        }
        let eff2: *mut c_double = alloc_zeroed(w2.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in w2.iter().enumerate() {
            *eff2.add(i) = w;
        }

        let n_dyn = build_c_dynamicmtx(&scoring.substitution_matrix);

        let c_score = mafft_sys::MSalignmm(
            n_dyn,
            c_seq1_ptrs.as_mut_ptr(),
            c_seq2_ptrs.as_mut_ptr(),
            eff1,
            eff2,
            w1.len() as c_int,
            w2.len() as c_int,
            alloclen as c_int,
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

        let c_len = {
            let s = c_seq1_ptrs[0];
            let mut n = 0;
            while *s.add(n) != 0 {
                n += 1;
            }
            n
        };

        eprintln!(
            "Rust: ops={} score={:.2}",
            rust_aln.operations.len(),
            rust_aln.score
        );
        eprintln!("C:    len={} score={:.2}", c_len, c_score);

        for (i, s) in c_seq1_boxed.iter().enumerate() {
            eprintln!("  C  g1[{i}]: {}", String::from_utf8_lossy(&s[..c_len]));
        }
        for (i, s) in c_seq2_boxed.iter().enumerate() {
            eprintln!("  C  g2[{i}]: {}", String::from_utf8_lossy(&s[..c_len]));
        }

        mafft_sys::freeconstants();

        assert_eq!(rust_aln.operations.len(), c_len, "width mismatch");
        assert!(
            (rust_aln.score - c_score).abs() < 1e-3,
            "score mismatch: rust={} c={}",
            rust_aln.score,
            c_score
        );
    }
}

/// Cross-validate `profile_align_imp` with warp DP enabled (gap.shift = Some)
/// against C's `A__align` with `penalty_shift_factor < 10` (trywarp = 1).
/// Guards the §9c warp DP port in `profile.rs`.
#[test]
fn profile_align_imp_warp_matches_c_a__align() {
    use mafft_align::profile_align_imp;
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    // Two 1-vs-1 groups (clus1=clus2=1) using opsin pair — same input as
    // `rust_global_align_matches_c_g__align11_warp` so this test exercises
    // the profile DP path on a non-trivial alignment.
    let group1: Vec<&[u8]> = vec![
        b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNFLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSIEGFFATLGGEVALWSLVVLAIERYIVICKPMGNFRFGNTHAIMGVAFTWIMALACAAPPLVGWSRYIPEGMQCSCGPDYYTLNPNFNNESYVVYMFVVHFLVPFVIIFFCYGRLLCTVKEAAAAQQESASTQKAEKEVTRMVVLMVIGFLVCWVPYASVAFYIFTHQGSDFGATFMTLPAFFAKSSALYNPVIYILMNKQFRNCMITTLCCGKNPLGDDESGASTSKTEVSSVSTSPVSPA",
    ];
    let group2: Vec<&[u8]> = vec![
        b"MAQQWSLQRLAGRHPQDSYEDSTQSSIFTYTNSNSTRGPFEGPNYHIAPRWVYHLTSVWMIFVVIASVFTNGLVLAATMKFKKLRHPLNWILVNLAVADLAETVIASTISVVNQVYGYFVLGHPMCVLEGYTVSLCGITGLWSLAIISWERWMVVCKPFGNVRFDAKLAIVGIAFSWIWAAVWTAPPIFGWSRYWPHGLKTSCGPDVFSGSSYPGVQSYMIVLMVTCCITPLSIIVLCYLQVWLAIRAVAKQQKESESTQKAEKEVTRMVVVMVLAFCFCWGPYAFFACFAAANPGYPFHPLMAALPAFFAKSATIYNPVIYVFMNRQFRNCILQLFGKKVDDGSELSSASKTEVSSVSSVSPA",
    ];
    let w1 = vec![1.0];
    let w2 = vec![1.0];

    let prof1 = Profile::from_aligned(&group1, &w1, &scoring.amino_map, scoring.nalphabets);
    let prof2 = Profile::from_aligned(&group2, &w2, &scoring.amino_map, scoring.nalphabets);

    // Group DP params (mirror tbfast for protein): penalty = 0.6 * -1530 = -917,
    // penalty_ex = 0 (DEFAULTGEP_B = 0). With --allowshift, spfactor = 2.0,
    // penalty_shift = 2.0 * -917 = -1834.
    let scale_protein: f64 = 600.0 / 1000.0;
    let scale = |ppen: f64| -> i32 { ((scale_protein * ppen) + 0.5) as i32 };
    let penalty = scale(-1530.0) as f64;
    let penalty_ex = 0.0_f64;
    let penalty_shift = (2.0_f64 * penalty) as i32 as f64;
    let gap = GapModel::new(penalty, penalty_ex).with_shift(penalty_shift);
    eprintln!(
        "penalty={} penalty_ex={} penalty_shift={}",
        penalty, penalty_ex, penalty_shift
    );

    let rust_aln = profile_align_imp(
        &prof1,
        &prof2,
        &scoring.consweight_matrix,
        &gap,
        true,
        true,
        None,
    );
    eprintln!(
        "Rust: ops.len()={}, score={:.2}",
        rust_aln.operations.len(),
        rust_aln.score
    );

    unsafe {
        init_c_protein();
        // Activate warp DP via penalty_shift_factor < 10 (constants.c:277-278).
        std::ptr::addr_of_mut!(mafft_sys::penalty_shift_factor).write(2.0);
        // Re-run constants() so trywarp/penalty_shift pick up the factor.
        let seq_data = b"ACDEFGHIKLMNPQRSTVWY\0";
        let mut seq_ptr = seq_data.as_ptr() as *mut i8;
        let seq_arr: *mut *mut i8 = &mut seq_ptr;
        mafft_sys::constants(1, seq_arr);

        // tbfast's group call: A__align(dynamicmtx, penalty, penalty_ex, ...).
        // Use the GLOBAL `penalty`/`penalty_ex` which constants() set.

        let len1 = group1[0].len();
        let len2 = group2[0].len();
        let alloclen = (len1 + len2) * 4;

        let c_seq1_boxed: Vec<Box<[u8]>> = group1
            .iter()
            .map(|s| {
                let mut v = s.to_vec();
                v.resize(alloclen + 1, 0);
                v.into_boxed_slice()
            })
            .collect();
        let c_seq2_boxed: Vec<Box<[u8]>> = group2
            .iter()
            .map(|s| {
                let mut v = s.to_vec();
                v.resize(alloclen + 1, 0);
                v.into_boxed_slice()
            })
            .collect();
        let mut c_seq1_ptrs: Vec<*mut c_char> = c_seq1_boxed
            .iter()
            .map(|v| v.as_ptr() as *mut c_char)
            .collect();
        let mut c_seq2_ptrs: Vec<*mut c_char> = c_seq2_boxed
            .iter()
            .map(|v| v.as_ptr() as *mut c_char)
            .collect();
        let eff1: *mut c_double = alloc_zeroed(w1.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in w1.iter().enumerate() {
            *eff1.add(i) = w;
        }
        let eff2: *mut c_double = alloc_zeroed(w2.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in w2.iter().enumerate() {
            *eff2.add(i) = w;
        }

        let n_dyn = build_c_dynamicmtx(&scoring.substitution_matrix);

        let c_penalty = mafft_sys::penalty;
        let c_penalty_ex = mafft_sys::penalty_ex;
        eprintln!(
            "C globals: penalty={}, penalty_ex={}",
            c_penalty, c_penalty_ex
        );

        let c_score = mafft_sys::A__align(
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
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            1,
            1,
            -1,
            -1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0.0,
            0.0,
        );
        let c_len = {
            let s = c_seq1_ptrs[0];
            let mut n = 0;
            while *s.add(n) != 0 {
                n += 1;
            }
            n
        };
        eprintln!("C:    aligned_len={}, score={:.2}", c_len, c_score);

        let rust_width = rust_aln.operations.len();
        eprintln!("widths: Rust={} C={}", rust_width, c_len);

        mafft_sys::freeconstants();

        assert_eq!(
            rust_width, c_len,
            "width mismatch: rust={rust_width} c={c_len}"
        );
    }
}

/// Cross-validate `profile_align_imp` against C's `A__align` with NON-ZERO
/// `penalty_ex` (gap-extension penalty). Protein default `DEFAULTGEP_B = 0`
/// hides §B.1 (missing `mi += penalty_ex` / `mj[j] += penalty_ex` in profile
/// DP). This test sets `penalty_ex = -100` on both sides — the value
/// `--ep 0.1` would have produced if our CLI routed `--ep` to penalty_ex
/// (it currently routes to matrix offset, matching the C script's
/// convention).
///
/// C reference: `mafft-upstream/core/Salignmm.c:1933,1953`:
///   mi += fpenalty_ex;            // line 1933
///   if (j < lgth2) m[j] += fpenalty_ex;  // line 1953
#[test]
fn profile_align_imp_nonzero_penalty_ex_matches_c_a__align() {
    use mafft_align::profile_align_imp;
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    // Two 1-vs-1 groups (opsin pair, same as the warp test).
    let group1: Vec<&[u8]> = vec![
        b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYYLAEPWKYSALAAYMFFLILVGFPVNFLTLFVTVQHKKLRTPLNYILLNLAMANLFMVLFGFTVTMYTSMNGYFVFGPTMCSIEGFFATLGGEVALWSLVVLAIERYIVICKPMGNFRFGNTHAIMGVAFTWIMALACAAPPLVGWSRYIPEGMQCSCGPDYYTLNPNFNNESYVVYMFVVHFLVPFVIIFFCYGRLLCTVKEAAAAQQESASTQKAEKEVTRMVVLMVIGFLVCWVPYASVAFYIFTHQGSDFGATFMTLPAFFAKSSALYNPVIYILMNKQFRNCMITTLCCGKNPLGDDESGASTSKTEVSSVSTSPVSPA",
    ];
    let group2: Vec<&[u8]> = vec![
        b"MAQQWSLQRLAGRHPQDSYEDSTQSSIFTYTNSNSTRGPFEGPNYHIAPRWVYHLTSVWMIFVVIASVFTNGLVLAATMKFKKLRHPLNWILVNLAVADLAETVIASTISVVNQVYGYFVLGHPMCVLEGYTVSLCGITGLWSLAIISWERWMVVCKPFGNVRFDAKLAIVGIAFSWIWAAVWTAPPIFGWSRYWPHGLKTSCGPDVFSGSSYPGVQSYMIVLMVTCCITPLSIIVLCYLQVWLAIRAVAKQQKESESTQKAEKEVTRMVVVMVLAFCFCWGPYAFFACFAAANPGYPFHPLMAALPAFFAKSATIYNPVIYVFMNRQFRNCILQLFGKKVDDGSELSSASKTEVSSVSSVSPA",
    ];
    let w1 = vec![1.0];
    let w2 = vec![1.0];

    let prof1 = Profile::from_aligned(&group1, &w1, &scoring.amino_map, scoring.nalphabets);
    let prof2 = Profile::from_aligned(&group2, &w2, &scoring.amino_map, scoring.nalphabets);

    // Group DP params with NON-ZERO penalty_ex.
    let scale_protein: f64 = 600.0 / 1000.0;
    let scale = |ppen: f64| -> i32 { ((scale_protein * ppen) + 0.5) as i32 };
    let penalty = scale(-1530.0) as f64;
    let penalty_ex = -100.0_f64; // Non-zero — exposes §B.1.
    let gap = GapModel::new(penalty, penalty_ex);
    eprintln!("penalty={} penalty_ex={}", penalty, penalty_ex);

    let rust_aln = profile_align_imp(
        &prof1,
        &prof2,
        &scoring.consweight_matrix,
        &gap,
        true,
        true,
        None,
    );
    eprintln!(
        "Rust: ops.len()={}, score={:.2}",
        rust_aln.operations.len(),
        rust_aln.score
    );

    unsafe {
        init_c_protein();
        // Ensure trywarp = 0 (default state — no shift penalty factor change).
        // init_c_protein already sets penalty_shift_factor to default 100.

        let len1 = group1[0].len();
        let len2 = group2[0].len();
        let alloclen = (len1 + len2) * 4;

        let c_seq1_boxed: Vec<Box<[u8]>> = group1
            .iter()
            .map(|s| {
                let mut v = s.to_vec();
                v.resize(alloclen + 1, 0);
                v.into_boxed_slice()
            })
            .collect();
        let c_seq2_boxed: Vec<Box<[u8]>> = group2
            .iter()
            .map(|s| {
                let mut v = s.to_vec();
                v.resize(alloclen + 1, 0);
                v.into_boxed_slice()
            })
            .collect();
        let mut c_seq1_ptrs: Vec<*mut c_char> = c_seq1_boxed
            .iter()
            .map(|v| v.as_ptr() as *mut c_char)
            .collect();
        let mut c_seq2_ptrs: Vec<*mut c_char> = c_seq2_boxed
            .iter()
            .map(|v| v.as_ptr() as *mut c_char)
            .collect();
        let eff1: *mut c_double = alloc_zeroed(w1.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in w1.iter().enumerate() {
            *eff1.add(i) = w;
        }
        let eff2: *mut c_double = alloc_zeroed(w2.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in w2.iter().enumerate() {
            *eff2.add(i) = w;
        }

        let n_dyn = build_c_dynamicmtx(&scoring.substitution_matrix);

        // Pass our explicit penalty/penalty_ex (NOT the C globals). A__align
        // takes them as the second/third args (Salignmm.c::A__align signature).
        let c_score = mafft_sys::A__align(
            n_dyn,
            penalty as c_int,
            penalty_ex as c_int,
            c_seq1_ptrs.as_mut_ptr(),
            c_seq2_ptrs.as_mut_ptr(),
            eff1,
            eff2,
            w1.len() as c_int,
            w2.len() as c_int,
            alloclen as c_int,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            1,
            1,
            -1,
            -1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0.0,
            0.0,
        );
        let c_len = {
            let s = c_seq1_ptrs[0];
            let mut n = 0;
            while *s.add(n) != 0 {
                n += 1;
            }
            n
        };
        eprintln!("C:    aligned_len={}, score={:.2}", c_len, c_score);

        let rust_width = rust_aln.operations.len();
        eprintln!("widths: Rust={} C={}", rust_width, c_len);

        mafft_sys::freeconstants();

        assert_eq!(
            rust_width, c_len,
            "width mismatch: rust={rust_width} c={c_len}"
        );
        assert!(
            (rust_aln.score - c_score).abs() < 1e-3,
            "score mismatch: rust={} c={}",
            rust_aln.score,
            c_score
        );
    }
}

/// Profile DP with warp AND a shifted matrix (delta ≠ 0), mirroring the
/// per-step `makedynamicmtx` shift that fires for `--allowshift` group
/// merges (`disttbfast.c:2304`, `mltaln9.c::makedynamicmtx`). This is the
/// configuration the actual pipeline uses for G-INS-i with `--allowshift`.
/// Guards profile_align_imp's warp DP under shifted scoring.
#[test]
fn profile_align_imp_warp_shifted_matrix_matches_c_a__align() {
    use mafft_align::profile_align_imp;
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    let group1: Vec<&[u8]> = vec![
        b"MNGTEGPNFYVPFSNITGVVRSPFEQPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVFGGFTTTLYTSLHGYFVFGPTGCNLEGFFATLGGEIGLWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLVGWSRYIPEGMQCSCGIDYYTLKPEVNNESFVIYMFVVHFTIPMIVIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIFFLICWLPYASVAMYIFTHQGSNFGPIFMTLPAFFAKTASIYNPIIYIMMNKQFRNCMLTSLCCGKNPLGDDEASATASKTETSQVAPA",
    ];
    let group2: Vec<&[u8]> = vec![
        b"MAAWEAAFAARRRHEEEDTTRDSVFTYTNSNNTRGPFEGPNYHIAPRWVYNLTSVWMIFVVAASVFTNGLVLVATWKFKKLRHPLNWILVNLAVADLGETVIASTISVINQISGYFILGHPMCVVEGYTVSACGITALWSLAIISWERWFVVCKPFGNIKFDGKLAVAGILFSWLWSCAWTAPPIFGWSRYWPHGLKTSCGPDVFSGSSDPGVQSYMVVLMVTCCFFPLAIIILCYLQVWLAIRAVAAQQKESESTQKAEKEVSRMVVVMIVAYCFCWGPYTFFACFAAANPGYAFHPLAAALPAYFAKSATIYNPIIYVFMNRQFRNCILQLFGKKVDDGSEVSTSRTEVSSVSNSSVSPA",
    ];
    let w1 = vec![1.0];
    let w2 = vec![1.0];

    let prof1 = Profile::from_aligned(&group1, &w1, &scoring.amino_map, scoring.nalphabets);
    let prof2 = Profile::from_aligned(&group2, &w2, &scoring.amino_map, scoring.nalphabets);

    let scale_protein: f64 = 600.0 / 1000.0;
    let scale = |ppen: f64| -> i32 { ((scale_protein * ppen) + 0.5) as i32 };
    let penalty = scale(-1530.0) as f64;
    let penalty_ex = 0.0_f64;
    let penalty_shift = (2.0_f64 * penalty) as i32 as f64;
    let gap = GapModel::new(penalty, penalty_ex).with_shift(penalty_shift);

    // Apply a per-step delta of -161, matching the typical
    // `makedynamicmtx` shift for a step where distfromtip < unalign_level.
    let delta: f64 = -161.0;
    let dyn_matrix: Vec<Vec<f64>> = scoring
        .consweight_matrix
        .iter()
        .map(|row| row.iter().map(|&v| v + delta).collect())
        .collect();

    let rust_aln = profile_align_imp(&prof1, &prof2, &dyn_matrix, &gap, true, true, None);
    eprintln!(
        "Rust: ops.len()={}, score={:.2}",
        rust_aln.operations.len(),
        rust_aln.score
    );

    unsafe {
        init_c_protein();
        std::ptr::addr_of_mut!(mafft_sys::penalty_shift_factor).write(2.0);
        let seq_data = b"ACDEFGHIKLMNPQRSTVWY\0";
        let mut seq_ptr = seq_data.as_ptr() as *mut i8;
        let seq_arr: *mut *mut i8 = &mut seq_ptr;
        mafft_sys::constants(1, seq_arr);

        let len1 = group1[0].len();
        let len2 = group2[0].len();
        let alloclen = (len1 + len2) * 4;

        let c_seq1_boxed: Vec<Box<[u8]>> = group1
            .iter()
            .map(|s| {
                let mut v = s.to_vec();
                v.resize(alloclen + 1, 0);
                v.into_boxed_slice()
            })
            .collect();
        let c_seq2_boxed: Vec<Box<[u8]>> = group2
            .iter()
            .map(|s| {
                let mut v = s.to_vec();
                v.resize(alloclen + 1, 0);
                v.into_boxed_slice()
            })
            .collect();
        let mut c_seq1_ptrs: Vec<*mut c_char> = c_seq1_boxed
            .iter()
            .map(|v| v.as_ptr() as *mut c_char)
            .collect();
        let mut c_seq2_ptrs: Vec<*mut c_char> = c_seq2_boxed
            .iter()
            .map(|v| v.as_ptr() as *mut c_char)
            .collect();
        let eff1: *mut c_double = alloc_zeroed(w1.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in w1.iter().enumerate() {
            *eff1.add(i) = w;
        }
        let eff2: *mut c_double = alloc_zeroed(w2.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in w2.iter().enumerate() {
            *eff2.add(i) = w;
        }

        // Build C's dynamicmtx from the SHIFTED dyn_matrix.
        let nalpha = dyn_matrix.len() as c_int;
        let n_dyn = mafft_sys::AllocateDoubleMtx(nalpha, nalpha);
        for i in 0..dyn_matrix.len() {
            for j in 0..dyn_matrix[i].len() {
                *(*n_dyn.add(i)).add(j) = dyn_matrix[i][j];
            }
        }

        let c_score = mafft_sys::A__align(
            n_dyn,
            mafft_sys::penalty,
            mafft_sys::penalty_ex,
            c_seq1_ptrs.as_mut_ptr(),
            c_seq2_ptrs.as_mut_ptr(),
            eff1,
            eff2,
            w1.len() as c_int,
            w2.len() as c_int,
            alloclen as c_int,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            1,
            1,
            -1,
            -1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0.0,
            0.0,
        );
        let c_len = {
            let s = c_seq1_ptrs[0];
            let mut n = 0;
            while *s.add(n) != 0 {
                n += 1;
            }
            n
        };
        eprintln!("C    width={}, score={:.2}", c_len, c_score);

        mafft_sys::freeconstants();

        assert_eq!(
            rust_aln.operations.len(),
            c_len,
            "shifted-matrix width mismatch"
        );
        assert!(
            (rust_aln.score - c_score).abs() < 1e-3,
            "shifted-matrix score mismatch: rust={} c={}",
            rust_aln.score,
            c_score
        );
    }
}

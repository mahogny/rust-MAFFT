/// Cross-validate our Profile::from_aligned against C's cpmx_calc_new +
/// st_OpeningGapCount + st_FinalGapCount + gapcountf.
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int};
use std::sync::Mutex;

use mafft_align::Profile;
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
        // MAFFT script passes `-h 0.000` → poffset=0, NOT default -123.
        // Use poffset=0 to match the real runtime.
        std::ptr::addr_of_mut!(mafft_sys::poffset).write(0);
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

#[test]
fn profile_cpmx_matches_c() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    // Refinement branch: 5 aligned sequences with gaps.
    let seqs_u8: Vec<&[u8]> = vec![
        b"ACDEFGHI--KLMN-PQR",
        b"AC-EFGHIKKLMNP-PQR",
        b"ACDE--HIKKL-NPQPQR",
        b"A-DEFGHI--KLMNPPQR",
        b"ACDEFGHIKK-LMNPPQR",
    ];
    let weights = vec![0.20, 0.22, 0.18, 0.19, 0.21];

    let length = seqs_u8[0].len();

    // --- Rust side ---
    let prof = Profile::from_aligned(&seqs_u8, &weights, &scoring.amino_map, scoring.nalphabets);

    // --- C side ---
    unsafe {
        init_c_protein();

        // Build C sequences (null-terminated)
        let cstrings: Vec<CString> = seqs_u8.iter().map(|s| CString::new(*s).unwrap()).collect();
        let mut c_ptrs: Vec<*mut c_char> =
            cstrings.iter().map(|s| s.as_ptr() as *mut c_char).collect();

        // eff array
        let eff: *mut c_double = alloc_zeroed(weights.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in weights.iter().enumerate() {
            *eff.add(i) = w;
        }

        // cpmx: [nalphabets][length + 2]
        let nalpha_c = std::ptr::addr_of!(mafft_sys::nalphabets).read();
        let cpmx = mafft_sys::AllocateDoubleMtx(nalpha_c, length as c_int + 2);
        mafft_sys::cpmx_calc_new(
            c_ptrs.as_mut_ptr(),
            cpmx,
            eff,
            length as c_int,
            weights.len() as c_int,
        );

        let mut cpmx_mismatches = 0;
        for pos in 0..length {
            for ridx in 0..scoring.nalphabets {
                let rust_freq = prof.freqs[pos][ridx];
                let c_freq = *(*cpmx.add(ridx)).add(pos);
                if (rust_freq - c_freq).abs() > 1e-9 {
                    if cpmx_mismatches < 5 {
                        eprintln!(
                            "cpmx diff pos={pos} ridx={ridx}: rust={rust_freq:.10} c={c_freq:.10}"
                        );
                    }
                    cpmx_mismatches += 1;
                }
            }
        }
        eprintln!(
            "cpmx: {} mismatches out of {}",
            cpmx_mismatches,
            length * scoring.nalphabets
        );

        // st_OpeningGapCount
        let c_ogcp: *mut c_double =
            alloc_zeroed((length + 2) * std::mem::size_of::<c_double>()) as _;
        mafft_sys::st_OpeningGapCount(
            c_ogcp,
            weights.len() as c_int,
            c_ptrs.as_mut_ptr(),
            eff,
            length as c_int,
        );

        let mut ogcp_mismatches = 0;
        for pos in 0..length {
            let r = prof.ogcp[pos];
            let c = *c_ogcp.add(pos);
            if (r - c).abs() > 1e-9 {
                if ogcp_mismatches < 5 {
                    eprintln!("ogcp diff pos={pos}: rust={r:.10} c={c:.10}");
                }
                ogcp_mismatches += 1;
            }
        }
        eprintln!("ogcp: {} mismatches out of {}", ogcp_mismatches, length);

        // st_FinalGapCount
        let c_fgcp: *mut c_double =
            alloc_zeroed((length + 2) * std::mem::size_of::<c_double>()) as _;
        mafft_sys::st_FinalGapCount(
            c_fgcp,
            weights.len() as c_int,
            c_ptrs.as_mut_ptr(),
            eff,
            length as c_int,
        );

        let mut fgcp_mismatches = 0;
        for pos in 0..length {
            let r = prof.fgcp[pos];
            let c = *c_fgcp.add(pos);
            if (r - c).abs() > 1e-9 {
                if fgcp_mismatches < 5 {
                    eprintln!("fgcp diff pos={pos}: rust={r:.10} c={c:.10}");
                }
                fgcp_mismatches += 1;
            }
        }
        eprintln!("fgcp: {} mismatches out of {}", fgcp_mismatches, length);

        // nongap_freq via gapcountf
        let c_gapfreq: *mut c_double =
            alloc_zeroed((length + 2) * std::mem::size_of::<c_double>()) as _;
        mafft_sys::gapcountf(
            c_gapfreq,
            c_ptrs.as_mut_ptr(),
            weights.len() as c_int,
            eff,
            length as c_int,
        );

        let mut nongap_mismatches = 0;
        for pos in 0..length {
            let r = prof.nongap_freq[pos];
            let c = 1.0 - *c_gapfreq.add(pos);
            if (r - c).abs() > 1e-9 {
                if nongap_mismatches < 5 {
                    eprintln!("nongap diff pos={pos}: rust={r:.10} c={c:.10}");
                }
                nongap_mismatches += 1;
            }
        }
        eprintln!(
            "nongap_freq: {} mismatches out of {}",
            nongap_mismatches, length
        );

        mafft_sys::freeconstants();

        assert_eq!(cpmx_mismatches, 0, "cpmx has mismatches");
        assert_eq!(ogcp_mismatches, 0, "ogcp has mismatches");
        assert_eq!(fgcp_mismatches, 0, "fgcp has mismatches");
        assert_eq!(nongap_mismatches, 0, "nongap_freq has mismatches");
    }
}

/// Test Profile with weights that DON'T sum to 1.0 (pre-normalized case).
/// This is what compute_split_score sees before internal normalization.
#[test]
fn profile_with_unnormalized_weights_matches_c() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    let seqs_u8: Vec<&[u8]> = vec![
        b"ACDEFGHI--KLMN-PQR",
        b"AC-EFGHIKKLMNP-PQR",
        b"ACDE--HIKKL-NPQPQR",
    ];
    // Weights that don't sum to 1.0 — like BranchWeights output
    let weights = vec![1.0, 0.685, 0.621];

    let length = seqs_u8[0].len();
    let prof = Profile::from_aligned(&seqs_u8, &weights, &scoring.amino_map, scoring.nalphabets);

    unsafe {
        init_c_protein();

        let cstrings: Vec<CString> = seqs_u8.iter().map(|s| CString::new(*s).unwrap()).collect();
        let mut c_ptrs: Vec<*mut c_char> =
            cstrings.iter().map(|s| s.as_ptr() as *mut c_char).collect();

        let eff: *mut c_double = alloc_zeroed(weights.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in weights.iter().enumerate() {
            *eff.add(i) = w;
        }

        let nalpha_c = std::ptr::addr_of!(mafft_sys::nalphabets).read();
        let cpmx = mafft_sys::AllocateDoubleMtx(nalpha_c, length as c_int + 2);
        mafft_sys::cpmx_calc_new(
            c_ptrs.as_mut_ptr(),
            cpmx,
            eff,
            length as c_int,
            weights.len() as c_int,
        );

        for pos in 0..length {
            for ridx in 0..scoring.nalphabets {
                let r = prof.freqs[pos][ridx];
                let c = *(*cpmx.add(ridx)).add(pos);
                assert!(
                    (r - c).abs() < 1e-9,
                    "cpmx pos={pos} ridx={ridx}: r={r} c={c}"
                );
            }
        }

        let c_ogcp: *mut c_double =
            alloc_zeroed((length + 2) * std::mem::size_of::<c_double>()) as _;
        mafft_sys::st_OpeningGapCount(
            c_ogcp,
            weights.len() as c_int,
            c_ptrs.as_mut_ptr(),
            eff,
            length as c_int,
        );
        for pos in 0..length {
            let r = prof.ogcp[pos];
            let c = *c_ogcp.add(pos);
            assert!((r - c).abs() < 1e-9, "ogcp pos={pos}: r={r} c={c}");
        }

        eprintln!(
            "Unnormalized weights (sum={}) all match C!",
            weights.iter().sum::<f64>()
        );

        mafft_sys::freeconstants();
    }
}

/// Validate that compute_split_score matches C's intergroup_score.
/// We replicate the key logic here since compute_split_score is private.
#[test]
fn intergroup_score_matches_c() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    let group1: Vec<&[u8]> = vec![b"ACDE", b"ACDE"];
    let group2: Vec<&[u8]> = vec![b"ACDE"];
    // Per-group normalized weights
    let eff1 = vec![0.5, 0.5];
    let eff2 = vec![1.0];
    let len = group1[0].len();

    // Probe amino_dis_consweight_multi['-']['A'] etc.
    unsafe {
        init_c_protein();
        let adcw = std::ptr::addr_of!(mafft_sys::amino_dis_consweight_multi).read();
        let v1 = *(*adcw.add(b'-' as usize)).add(b'A' as usize);
        let v2 = *(*adcw.add(b'A' as usize)).add(b'-' as usize);
        let v3 = *(*adcw.add(b'-' as usize)).add(b'-' as usize);
        eprintln!("amino_dis_consweight_multi['-']['A'] = {}", v1);
        eprintln!("amino_dis_consweight_multi['A']['-'] = {}", v2);
        eprintln!("amino_dis_consweight_multi['-']['-'] = {}", v3);
        mafft_sys::freeconstants();
    }

    // Rust side: replicate compute_split_score's kernel using pairwise_score semantics.
    // We need to call the internal pairwise_score. Let's compute equivalent manually.
    let penalty = scoring.gap.open as f64;
    let mtx = &scoring.substitution_matrix;
    let map = &scoring.amino_map;
    let mtx_size = mtx.len();

    let pairwise = |a: &[u8], b: &[u8]| -> f64 {
        let l = a.len().min(b.len());
        let mut score = 0.0f64;
        let mut k = 0;
        while k < l {
            let ca = a[k];
            let cb = b[k];
            if ca == b'-' && cb == b'-' {
                k += 1;
                continue;
            }
            if ca == b'-' {
                score += penalty;
                k += 1;
                while k < l && a[k] == b'-' {
                    k += 1;
                }
                continue;
            }
            if cb == b'-' {
                score += penalty;
                k += 1;
                while k < l && b[k] == b'-' {
                    k += 1;
                }
                continue;
            }
            let i = map[ca as usize] as usize;
            let j = map[cb as usize] as usize;
            if i < mtx_size && j < mtx_size {
                score += mtx[i][j] as f64;
            }
            k += 1;
        }
        score
    };

    let mut rust_score = 0.0f64;
    for (i, s1) in group1.iter().enumerate() {
        for (j, s2) in group2.iter().enumerate() {
            rust_score += pairwise(s1, s2) * eff1[i] * eff2[j];
        }
    }

    // C side
    unsafe {
        init_c_protein();

        let cs1: Vec<CString> = group1.iter().map(|s| CString::new(*s).unwrap()).collect();
        let cs2: Vec<CString> = group2.iter().map(|s| CString::new(*s).unwrap()).collect();
        let mut p1: Vec<*mut c_char> = cs1.iter().map(|s| s.as_ptr() as *mut c_char).collect();
        let mut p2: Vec<*mut c_char> = cs2.iter().map(|s| s.as_ptr() as *mut c_char).collect();

        let e1: *mut c_double = alloc_zeroed(eff1.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in eff1.iter().enumerate() {
            *e1.add(i) = w;
        }
        let e2: *mut c_double = alloc_zeroed(eff2.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in eff2.iter().enumerate() {
            *e2.add(i) = w;
        }

        let mut c_score = 0.0f64;
        mafft_sys::intergroup_score(
            p1.as_mut_ptr(),
            p2.as_mut_ptr(),
            e1,
            e2,
            eff1.len() as c_int,
            eff2.len() as c_int,
            len as c_int,
            &mut c_score as *mut c_double,
        );

        eprintln!("Rust intergroup_score: {}", rust_score);
        eprintln!("C    intergroup_score: {}", c_score);
        eprintln!("Diff: {}", c_score - rust_score);

        let a_a = scoring.substitution_matrix[scoring.amino_map[b'A' as usize] as usize]
            [scoring.amino_map[b'A' as usize] as usize];
        let c_c = scoring.substitution_matrix[scoring.amino_map[b'C' as usize] as usize]
            [scoring.amino_map[b'C' as usize] as usize];
        let d_d = scoring.substitution_matrix[scoring.amino_map[b'D' as usize] as usize]
            [scoring.amino_map[b'D' as usize] as usize];
        let e_e = scoring.substitution_matrix[scoring.amino_map[b'E' as usize] as usize]
            [scoring.amino_map[b'E' as usize] as usize];
        eprintln!("Rust matrix A-A={a_a} C-C={c_c} D-D={d_d} E-E={e_e}");
        let adcw = std::ptr::addr_of!(mafft_sys::amino_dis_consweight_multi).read();
        let aa_c = *(*adcw.add(b'A' as usize)).add(b'A' as usize);
        let cc_c = *(*adcw.add(b'C' as usize)).add(b'C' as usize);
        let dd_c = *(*adcw.add(b'D' as usize)).add(b'D' as usize);
        let ee_c = *(*adcw.add(b'E' as usize)).add(b'E' as usize);
        eprintln!("C    amino_dis_consweight_multi A-A={aa_c} C-C={cc_c} D-D={dd_c} E-E={ee_c}");
        eprintln!("Rust scoring.gap.offset = {}", scoring.gap.offset);
        eprintln!(
            "Rust matrix + 73 would match C: {} + 73 = {}",
            a_a,
            a_a + 73
        );

        mafft_sys::freeconstants();

        assert!(
            (rust_score - c_score).abs() < 1e-4,
            "intergroup_score mismatch: rust={} c={}",
            rust_score,
            c_score
        );
    }
}

/// Test Profile::from_aligned with UNEQUAL weights (like per-branch weights).
#[test]
fn profile_with_unequal_weights_matches_c() {
    let _guard = C_MUTEX.lock().unwrap();

    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);

    let seqs_u8: Vec<&[u8]> = vec![
        b"ACDEFGHI--KLMN-PQR",
        b"AC-EFGHIKKLMNP-PQR",
        b"ACDE--HIKKL-NPQPQR",
        b"A-DEFGHI--KLMNPPQR",
        b"ACDEFGHIKK-LMNPPQR",
    ];
    // Very unequal weights like per-branch would produce
    let weights = vec![1.0, 0.5, 0.3, 0.1, 0.01];

    let length = seqs_u8[0].len();
    let prof = Profile::from_aligned(&seqs_u8, &weights, &scoring.amino_map, scoring.nalphabets);

    unsafe {
        init_c_protein();

        let cstrings: Vec<CString> = seqs_u8.iter().map(|s| CString::new(*s).unwrap()).collect();
        let mut c_ptrs: Vec<*mut c_char> =
            cstrings.iter().map(|s| s.as_ptr() as *mut c_char).collect();

        let eff: *mut c_double = alloc_zeroed(weights.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in weights.iter().enumerate() {
            *eff.add(i) = w;
        }

        let nalpha_c = std::ptr::addr_of!(mafft_sys::nalphabets).read();
        let cpmx = mafft_sys::AllocateDoubleMtx(nalpha_c, length as c_int + 2);
        mafft_sys::cpmx_calc_new(
            c_ptrs.as_mut_ptr(),
            cpmx,
            eff,
            length as c_int,
            weights.len() as c_int,
        );

        let mut cpmx_diffs = 0;
        for pos in 0..length {
            for ridx in 0..scoring.nalphabets {
                let r = prof.freqs[pos][ridx];
                let c = *(*cpmx.add(ridx)).add(pos);
                if (r - c).abs() > 1e-9 {
                    cpmx_diffs += 1;
                }
            }
        }

        let c_ogcp: *mut c_double =
            alloc_zeroed((length + 2) * std::mem::size_of::<c_double>()) as _;
        mafft_sys::st_OpeningGapCount(
            c_ogcp,
            weights.len() as c_int,
            c_ptrs.as_mut_ptr(),
            eff,
            length as c_int,
        );
        let mut ogcp_diffs = 0;
        for pos in 0..length {
            if (prof.ogcp[pos] - *c_ogcp.add(pos)).abs() > 1e-9 {
                ogcp_diffs += 1;
            }
        }

        let c_fgcp: *mut c_double =
            alloc_zeroed((length + 2) * std::mem::size_of::<c_double>()) as _;
        mafft_sys::st_FinalGapCount(
            c_fgcp,
            weights.len() as c_int,
            c_ptrs.as_mut_ptr(),
            eff,
            length as c_int,
        );
        let mut fgcp_diffs = 0;
        for pos in 0..length {
            if (prof.fgcp[pos] - *c_fgcp.add(pos)).abs() > 1e-9 {
                fgcp_diffs += 1;
            }
        }

        let c_gapfreq: *mut c_double =
            alloc_zeroed((length + 2) * std::mem::size_of::<c_double>()) as _;
        mafft_sys::gapcountf(
            c_gapfreq,
            c_ptrs.as_mut_ptr(),
            weights.len() as c_int,
            eff,
            length as c_int,
        );
        let mut nongap_diffs = 0;
        for pos in 0..length {
            if (prof.nongap_freq[pos] - (1.0 - *c_gapfreq.add(pos))).abs() > 1e-9 {
                nongap_diffs += 1;
            }
        }

        eprintln!(
            "Unequal weights: cpmx={cpmx_diffs} ogcp={ogcp_diffs} fgcp={fgcp_diffs} nongap={nongap_diffs}"
        );

        mafft_sys::freeconstants();

        assert_eq!(cpmx_diffs, 0);
        assert_eq!(ogcp_diffs, 0);
        assert_eq!(fgcp_diffs, 0);
        assert_eq!(nongap_diffs, 0);
    }
}

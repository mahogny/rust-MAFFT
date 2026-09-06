//! Cell-by-cell comparison of Rust profile DP vs C `A__align` on the
//! BB20027 pass-1 step-13 input (the first divergent merge: 8+2,
//! same width 593, Rust score 100653.4 vs C 100625.6).
//!
//! Prior tests confirmed `Profile::from_aligned` and `blend_profiles_exact`
//! are bit-identical to C. So the divergence is in the DP itself.
//! This test feeds bit-identical inputs to both `A__align` (via FFI)
//! and Rust's `profile_align_imp_with_boundary`, then compares output
//! alignments cell-by-cell to pinpoint where the DP differs.

use std::os::raw::{c_char, c_double, c_int};

use mafft_core::progressive::progressive_align_partial;

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
        let seq_data = b"ACDEFGHIKLMNPQRSTVWY\0";
        let mut seq_ptr = seq_data.as_ptr() as *mut i8;
        let seq_arr: *mut *mut i8 = &mut seq_ptr;
        mafft_sys::constants(1, seq_arr);
    }
}

/// Allocate a `**c_char` of `nseq` rows × `len+10` cols, each row a
/// null-terminated copy of `seqs[i]`. Returns (top_ptr, row_buf_handles
/// to keep alive).
unsafe fn alloc_c_char_mtx(seqs: &[Vec<u8>], capacity: usize) -> (Vec<*mut c_char>, Vec<Vec<u8>>) {
    let cap = capacity.max(seqs.iter().map(|s| s.len()).max().unwrap_or(0)) + 16;
    let mut rows: Vec<Vec<u8>> = seqs
        .iter()
        .map(|s| {
            let mut v = vec![0u8; cap];
            v[..s.len()].copy_from_slice(s);
            v[s.len()] = 0; // null terminator
            v
        })
        .collect();
    let ptrs: Vec<*mut c_char> = rows
        .iter_mut()
        .map(|r| r.as_mut_ptr() as *mut c_char)
        .collect();
    (ptrs, rows)
}

/// Step 12 is the FIRST diverging merge in BB20027 pass 1: same widths
/// + same scores through step 11, but step 12's POST-MERGE alignment
/// differs by a single-residue shift in cluster1's 3 sequences.
/// Step 11's output is byte-identical between Rust and C, so the
/// inputs to step 12 are bit-identical. This test feeds those exact
/// inputs to Rust profile_align and C A__align separately and checks
/// whether they agree on which tied alignment to return.
#[test]
#[ignore]
fn bb20027_step12_rust_dp_vs_c_aalign_cell_by_cell() {
    let path = std::path::Path::new("/tmp/balibase/bench1.0/bali3/in/BB20027");
    if !path.exists() {
        eprintln!("BB20027 fixture missing; skipping");
        return;
    }
    let input = mafft_io::read_fasta(path).expect("read BB20027");
    let engine = mafft_core::MafftEngine::new(mafft_core::AlignmentMode::FftNs2);
    let msa = engine.align(&input);
    let rebuilt_tree = msa.guide_tree.clone().expect("guide_tree missing");
    let step12 = &rebuilt_tree.steps[12];
    let group1 = &step12.left;
    let group2 = &step12.right;
    eprintln!("Step 12 left ({}-way): {:?}", group1.len(), group1);
    eprintln!("Step 12 right ({}-way): {:?}", group2.len(), group2);

    let scoring = mafft_scoring::build_context(
        mafft_types::ScoringModel::Blosum(62),
        mafft_types::SeqType::Protein,
    );

    // Replay pass 1 up to step 11 (run 12 steps, 0..=11).
    // IMPORTANT: use the same use_fft as the engine for this run. The
    // engine in default mode uses FFT, in --nofft uses no FFT. We
    // want our test inputs to MATCH what the engine sees at step 12.
    let use_fft = std::env::var("REPLAY_NOFFT").is_err();
    eprintln!("Replay use_fft={}", use_fft);
    let raw_seqs: Vec<Vec<u8>> = input.sequences.iter().map(|s| s.data.clone()).collect();
    let pre_step12 =
        progressive_align_partial(&raw_seqs, &rebuilt_tree, &scoring, use_fft, None, 12);

    let g1_seqs: Vec<Vec<u8>> = group1.iter().map(|&i| pre_step12[i].clone()).collect();
    let g2_seqs: Vec<Vec<u8>> = group2.iter().map(|&i| pre_step12[i].clone()).collect();
    let lgth1 = g1_seqs[0].len();
    let lgth2 = g2_seqs[0].len();
    assert!(g1_seqs.iter().all(|s| s.len() == lgth1));
    assert!(g2_seqs.iter().all(|s| s.len() == lgth2));
    eprintln!("Pre-step-12: lgth1={}, lgth2={}", lgth1, lgth2);

    let leaf_weights = mafft_tree::sequence_weights(&rebuilt_tree);
    let raw_w1: Vec<f64> = group1.iter().map(|&i| leaf_weights[i]).collect();
    let raw_w2: Vec<f64> = group2.iter().map(|&i| leaf_weights[i]).collect();
    let orieff1: f64 = raw_w1.iter().sum();
    let orieff2: f64 = raw_w2.iter().sum();
    let w1n: Vec<f64> = raw_w1.iter().map(|w| w / orieff1).collect();
    let w2n: Vec<f64> = raw_w2.iter().map(|w| w / orieff2).collect();

    use mafft_align::{AlignOp, GapModel, Profile, profile_align};
    let r1: Vec<&[u8]> = g1_seqs.iter().map(|s| s.as_slice()).collect();
    let r2: Vec<&[u8]> = g2_seqs.iter().map(|s| s.as_slice()).collect();
    let prof1 = Profile::from_aligned(&r1, &w1n, &scoring.amino_map, scoring.nalphabets);
    let prof2 = Profile::from_aligned(&r2, &w2n, &scoring.amino_map, scoring.nalphabets);
    let gap = GapModel::new(scoring.gap.open as f64, scoring.gap.extend as f64);
    let rust_aln = profile_align(
        &prof1,
        &prof2,
        &scoring.consweight_matrix,
        &gap,
        false,
        false,
    );
    let rust_width = rust_aln.operations.len();
    eprintln!(
        "\nRust profile_align: width={} score={:.4}",
        rust_width, rust_aln.score
    );

    unsafe {
        init_c_protein();
    }
    let penalty: c_int = unsafe { std::ptr::addr_of!(mafft_sys::penalty).read() };
    let penalty_ex: c_int = unsafe { std::ptr::addr_of!(mafft_sys::penalty_ex).read() };
    let n_dynamicmtx = unsafe { std::ptr::addr_of!(mafft_sys::n_dis_consweight_multi).read() };
    let alloclen = (lgth1 + lgth2 + 100) as c_int;
    let n1 = g1_seqs.len();
    let n2 = g2_seqs.len();
    let (mut mseq1_ptrs, _h1) = unsafe { alloc_c_char_mtx(&g1_seqs, alloclen as usize) };
    let (mut mseq2_ptrs, _h2) = unsafe { alloc_c_char_mtx(&g2_seqs, alloclen as usize) };
    let mut e1: Vec<f64> = w1n.clone();
    let mut e2: Vec<f64> = w2n.clone();
    let mut dumdb = 0.0f64;
    let c_score = unsafe {
        mafft_sys::A__align(
            n_dynamicmtx,
            penalty,
            penalty_ex,
            mseq1_ptrs.as_mut_ptr(),
            mseq2_ptrs.as_mut_ptr(),
            e1.as_mut_ptr(),
            e2.as_mut_ptr(),
            n1 as c_int,
            n2 as c_int,
            alloclen,
            0,
            &mut dumdb,
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
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            orieff1,
            orieff2,
        )
    };
    let c_width = unsafe {
        let p = mseq1_ptrs[0];
        let mut w = 0usize;
        while *p.add(w) != 0 {
            w += 1;
        }
        w
    };
    eprintln!("\nC A__align: width={} score={:.4}", c_width, c_score);
    eprintln!("Score delta: {:.6}", rust_aln.score - c_score);

    // Read C alignments.
    let read = |ptr: *mut c_char, w: usize| {
        let mut v = vec![0u8; w];
        for i in 0..w {
            v[i] = unsafe { *ptr.add(i) as u8 };
        }
        v
    };
    let c_aln1: Vec<Vec<u8>> = mseq1_ptrs.iter().map(|&p| read(p, c_width)).collect();
    let c_aln2: Vec<Vec<u8>> = mseq2_ptrs.iter().map(|&p| read(p, c_width)).collect();

    // Reconstruct Rust alignment from operations.
    let mut r_aln1 = vec![Vec::<u8>::with_capacity(rust_width); n1];
    let mut r_aln2 = vec![Vec::<u8>::with_capacity(rust_width); n2];
    let mut c1 = 0usize;
    let mut c2 = 0usize;
    for op in &rust_aln.operations {
        match op {
            AlignOp::Match => {
                for k in 0..n1 {
                    r_aln1[k].push(g1_seqs[k][c1]);
                }
                for k in 0..n2 {
                    r_aln2[k].push(g2_seqs[k][c2]);
                }
                c1 += 1;
                c2 += 1;
            }
            AlignOp::Delete => {
                for k in 0..n1 {
                    r_aln1[k].push(g1_seqs[k][c1]);
                }
                for k in 0..n2 {
                    r_aln2[k].push(b'-');
                }
                c1 += 1;
            }
            AlignOp::Insert => {
                for k in 0..n1 {
                    r_aln1[k].push(b'-');
                }
                for k in 0..n2 {
                    r_aln2[k].push(g2_seqs[k][c2]);
                }
                c2 += 1;
            }
        }
    }

    // Dump test outputs for comparison vs engine dumps.
    {
        use std::io::Write;
        let mut f = std::fs::File::create("/tmp/test_step12_rust.fa").unwrap();
        writeln!(
            f,
            ">TEST_RUST width={} score={:.4}",
            rust_width, rust_aln.score
        )
        .unwrap();
        for (k, s) in r_aln1.iter().enumerate() {
            writeln!(f, ">g1_{}", k).unwrap();
            f.write_all(s).unwrap();
            writeln!(f).unwrap();
        }
        for (k, s) in r_aln2.iter().enumerate() {
            writeln!(f, ">g2_{}", k).unwrap();
            f.write_all(s).unwrap();
            writeln!(f).unwrap();
        }
        let mut f = std::fs::File::create("/tmp/test_step12_c.fa").unwrap();
        writeln!(f, ">TEST_C width={} score={:.4}", c_width, c_score).unwrap();
        for (k, s) in c_aln1.iter().enumerate() {
            writeln!(f, ">g1_{}", k).unwrap();
            f.write_all(s).unwrap();
            writeln!(f).unwrap();
        }
        for (k, s) in c_aln2.iter().enumerate() {
            writeln!(f, ">g2_{}", k).unwrap();
            f.write_all(s).unwrap();
            writeln!(f).unwrap();
        }
    }

    if rust_width == c_width {
        let mut first_diff: Option<usize> = None;
        for col in 0..rust_width {
            let mut differs = false;
            for k in 0..n1 {
                if r_aln1[k][col] != c_aln1[k][col] {
                    differs = true;
                    break;
                }
            }
            if !differs {
                for k in 0..n2 {
                    if r_aln2[k][col] != c_aln2[k][col] {
                        differs = true;
                        break;
                    }
                }
            }
            if differs {
                first_diff = Some(col);
                break;
            }
        }
        if let Some(col) = first_diff {
            eprintln!("\nFirst diverging column: {} (of {})", col, rust_width);
            let lo = col.saturating_sub(3);
            let hi = (col + 8).min(rust_width);
            eprintln!("\nRust columns [{}..{}]:", lo, hi);
            for k in 0..n1 {
                let s: String = r_aln1[k][lo..hi].iter().map(|&c| c as char).collect();
                eprintln!("  g1[{}]: {}", k, s);
            }
            for k in 0..n2 {
                let s: String = r_aln2[k][lo..hi].iter().map(|&c| c as char).collect();
                eprintln!("  g2[{}]: {}", k, s);
            }
            eprintln!("\nC columns [{}..{}]:", lo, hi);
            for k in 0..n1 {
                let s: String = c_aln1[k][lo..hi].iter().map(|&c| c as char).collect();
                eprintln!("  g1[{}]: {}", k, s);
            }
            for k in 0..n2 {
                let s: String = c_aln2[k][lo..hi].iter().map(|&c| c as char).collect();
                eprintln!("  g2[{}]: {}", k, s);
            }
        } else {
            eprintln!("\nAll columns match — DP outputs are bit-identical.");
        }
    } else {
        eprintln!("\nWidths differ: rust={} c={}", rust_width, c_width);
    }
}

#[test]
#[ignore]
fn bb20027_step13_rust_dp_vs_c_aalign_cell_by_cell() {
    let path = std::path::Path::new("/tmp/balibase/bench1.0/bali3/in/BB20027");
    if !path.exists() {
        eprintln!("BB20027 fixture missing; skipping");
        return;
    }
    let input = mafft_io::read_fasta(path).expect("read BB20027");

    // Run engine to get the rebuilt pass-1 tree (msa.guide_tree).
    let engine = mafft_core::MafftEngine::new(mafft_core::AlignmentMode::FftNs2);
    let msa = engine.align(&input);
    let rebuilt_tree = msa.guide_tree.clone().expect("guide_tree missing");
    assert_eq!(rebuilt_tree.steps.len(), 28, "expected 28 merge steps");

    let step13 = &rebuilt_tree.steps[13];
    let group1 = &step13.left; // 8 leaves
    let group2 = &step13.right; // 2 leaves
    assert_eq!(group1.len(), 8, "step 13 left should be 8-way");
    assert_eq!(group2.len(), 2, "step 13 right should be 2-way");
    eprintln!("Step 13 group1 (8-way) leaves: {:?}", group1);
    eprintln!("Step 13 group2 (2-way) leaves: {:?}", group2);

    // Build the protein scoring context.
    let scoring = mafft_scoring::build_context(
        mafft_types::ScoringModel::Blosum(62),
        mafft_types::SeqType::Protein,
    );

    // Replay pass 1 progressive up to step 12 (n_steps = 13 runs steps
    // 0..=12 and stops before step 13 — exactly what we want).
    let raw_seqs: Vec<Vec<u8>> = input.sequences.iter().map(|s| s.data.clone()).collect();
    let pre_step13_state = progressive_align_partial(
        &raw_seqs,
        &rebuilt_tree,
        &scoring,
        true, // use_fft (default mode)
        None, // no shift
        13,   // run 13 steps (0..=12)
    );

    // Extract the 8-way and 2-way profile sequences as they stood
    // entering step 13.
    let g1_seqs: Vec<Vec<u8>> = group1
        .iter()
        .map(|&i| pre_step13_state[i].clone())
        .collect();
    let g2_seqs: Vec<Vec<u8>> = group2
        .iter()
        .map(|&i| pre_step13_state[i].clone())
        .collect();
    let lgth1 = g1_seqs[0].len();
    let lgth2 = g2_seqs[0].len();
    assert!(
        g1_seqs.iter().all(|s| s.len() == lgth1),
        "g1 seqs not all same width — replay broken?"
    );
    assert!(
        g2_seqs.iter().all(|s| s.len() == lgth2),
        "g2 seqs not all same width — replay broken?"
    );
    eprintln!("Pre-step-13: lgth1={}, lgth2={}", lgth1, lgth2);

    // Compute per-leaf weights (normalized within each cluster).
    let leaf_weights = mafft_tree::sequence_weights(&rebuilt_tree);
    let raw_w1: Vec<f64> = group1.iter().map(|&i| leaf_weights[i]).collect();
    let raw_w2: Vec<f64> = group2.iter().map(|&i| leaf_weights[i]).collect();
    let orieff1: f64 = raw_w1.iter().sum();
    let orieff2: f64 = raw_w2.iter().sum();
    let w1n: Vec<f64> = raw_w1.iter().map(|w| w / orieff1).collect();
    let w2n: Vec<f64> = raw_w2.iter().map(|w| w / orieff2).collect();
    eprintln!("orieff1={:.10}, orieff2={:.10}", orieff1, orieff2);

    // ===== RUST: profile_align via the same path the engine uses. =====
    // The engine's merge_step_cached calls fft_profile_align (when
    // use_fft) or profile_align. For BB20027 default mode we use FFT.
    // But the divergence reproduces in --nofft too, and the non-FFT
    // path is simpler to compare against A__align directly. Use
    // profile_align here.
    use mafft_align::{GapModel, Profile, profile_align};
    let r1: Vec<&[u8]> = g1_seqs.iter().map(|s| s.as_slice()).collect();
    let r2: Vec<&[u8]> = g2_seqs.iter().map(|s| s.as_slice()).collect();
    let prof1 = Profile::from_aligned(&r1, &w1n, &scoring.amino_map, scoring.nalphabets);
    let prof2 = Profile::from_aligned(&r2, &w2n, &scoring.amino_map, scoring.nalphabets);
    let gap_open = scoring.gap.open as f64;
    let gap_ext = scoring.gap.extend as f64;
    let gap = GapModel::new(gap_open, gap_ext);
    // For FFT-NS-2 default mode, outgap=0 → head/tail gap not penalized.
    let rust_aln = profile_align(
        &prof1,
        &prof2,
        &scoring.consweight_matrix,
        &gap,
        false,
        false,
    );
    let rust_width = rust_aln.operations.len();
    eprintln!(
        "\nRust profile_align: width={} score={:.4}",
        rust_width, rust_aln.score
    );

    // ===== C: A__align via FFI. =====
    unsafe {
        init_c_protein();
    }

    // C's penalty / penalty_ex are i32 (in units of 1/1000ths).
    let penalty: c_int = unsafe { std::ptr::addr_of!(mafft_sys::penalty).read() };
    let penalty_ex: c_int = unsafe { std::ptr::addr_of!(mafft_sys::penalty_ex).read() };
    eprintln!("C penalty={} penalty_ex={}", penalty, penalty_ex);

    // C n_dis_consweight_multi is set by constants() from BLOSUM62.
    let n_dynamicmtx: *mut *mut c_double =
        unsafe { std::ptr::addr_of!(mafft_sys::n_dis_consweight_multi).read() };

    // Allocate mseq1, mseq2 buffers large enough for the merged result.
    let alloclen = (lgth1 + lgth2 + 100) as c_int;
    let n1 = g1_seqs.len();
    let n2 = g2_seqs.len();
    let (mut mseq1_ptrs, _hold1) = unsafe { alloc_c_char_mtx(&g1_seqs, alloclen as usize) };
    let (mut mseq2_ptrs, _hold2) = unsafe { alloc_c_char_mtx(&g2_seqs, alloclen as usize) };

    let mut eff1_arr: Vec<f64> = w1n.clone();
    let mut eff2_arr: Vec<f64> = w2n.clone();
    let mut dumdb = 0.0f64;

    let c_score = unsafe {
        mafft_sys::A__align(
            n_dynamicmtx,
            penalty,
            penalty_ex,
            mseq1_ptrs.as_mut_ptr(),
            mseq2_ptrs.as_mut_ptr(),
            eff1_arr.as_mut_ptr(),
            eff2_arr.as_mut_ptr(),
            n1 as c_int,
            n2 as c_int,
            alloclen,
            0, // constraint = 0
            &mut dumdb as *mut c_double,
            std::ptr::null_mut(),
            std::ptr::null_mut(), // sgap1, sgap2
            std::ptr::null_mut(),
            std::ptr::null_mut(), // egap1, egap2
            std::ptr::null_mut(), // chudanpt
            0,                    // chudanref
            std::ptr::null_mut(), // chudanres
            0,
            0,                    // headgp=0, tailgp=0 (matches default --fft mode)
            -1,                   // firstmem=-1 (disable memo)
            0,                    // calledbyfulltreebase=0
            std::ptr::null_mut(), // cpmxchild0 (force from-scratch)
            std::ptr::null_mut(), // cpmxchild1
            std::ptr::null_mut(), // cpmxresult (don't save)
            orieff1,
            orieff2,
        )
    };

    // Read back the aligned sequences from C.
    let c_aln_width = unsafe {
        let p = mseq1_ptrs[0];
        let mut w = 0usize;
        while *p.add(w) != 0 {
            w += 1;
        }
        w
    };
    eprintln!("\nC A__align: width={} score={:.4}", c_aln_width, c_score);

    // Read aligned mseq1 / mseq2.
    let read_aln = |ptr: *mut c_char, w: usize| -> Vec<u8> {
        let mut v = vec![0u8; w];
        for i in 0..w {
            v[i] = unsafe { *ptr.add(i) as u8 };
        }
        v
    };
    let c_aln1: Vec<Vec<u8>> = mseq1_ptrs
        .iter()
        .map(|&p| read_aln(p, c_aln_width))
        .collect();
    let c_aln2: Vec<Vec<u8>> = mseq2_ptrs
        .iter()
        .map(|&p| read_aln(p, c_aln_width))
        .collect();

    // Reconstruct Rust-aligned sequences from rust_aln.operations.
    use mafft_align::AlignOp;
    let mut r_aln1 = vec![Vec::<u8>::with_capacity(rust_width); n1];
    let mut r_aln2 = vec![Vec::<u8>::with_capacity(rust_width); n2];
    let mut c1 = 0usize;
    let mut c2 = 0usize;
    for op in &rust_aln.operations {
        match op {
            AlignOp::Match => {
                for k in 0..n1 {
                    r_aln1[k].push(g1_seqs[k][c1]);
                }
                for k in 0..n2 {
                    r_aln2[k].push(g2_seqs[k][c2]);
                }
                c1 += 1;
                c2 += 1;
            }
            AlignOp::Delete => {
                for k in 0..n1 {
                    r_aln1[k].push(g1_seqs[k][c1]);
                }
                for k in 0..n2 {
                    r_aln2[k].push(b'-');
                }
                c1 += 1;
            }
            AlignOp::Insert => {
                for k in 0..n1 {
                    r_aln1[k].push(b'-');
                }
                for k in 0..n2 {
                    r_aln2[k].push(g2_seqs[k][c2]);
                }
                c2 += 1;
            }
        }
    }

    eprintln!("\nFirst-difference position scan:");
    if rust_width == c_aln_width {
        // Same width. Find first differing column.
        for col in 0..rust_width {
            let mut differs = false;
            for k in 0..n1 {
                if r_aln1[k][col] != c_aln1[k][col] {
                    differs = true;
                    break;
                }
            }
            for k in 0..n2 {
                if r_aln2[k][col] != c_aln2[k][col] {
                    differs = true;
                    break;
                }
            }
            if differs {
                eprintln!(
                    "  First diverging COLUMN: {} (of {} total)",
                    col, rust_width
                );
                let col_lo = col.saturating_sub(2);
                let col_hi = (col + 5).min(rust_width);
                eprintln!("\n  Rust columns [{}..{}]:", col_lo, col_hi);
                for k in 0..n1 {
                    let s: String = r_aln1[k][col_lo..col_hi]
                        .iter()
                        .map(|&c| c as char)
                        .collect();
                    eprintln!("    g1[{}]: {}", k, s);
                }
                for k in 0..n2 {
                    let s: String = r_aln2[k][col_lo..col_hi]
                        .iter()
                        .map(|&c| c as char)
                        .collect();
                    eprintln!("    g2[{}]: {}", k, s);
                }
                eprintln!("\n  C columns [{}..{}]:", col_lo, col_hi);
                for k in 0..n1 {
                    let s: String = c_aln1[k][col_lo..col_hi]
                        .iter()
                        .map(|&c| c as char)
                        .collect();
                    eprintln!("    g1[{}]: {}", k, s);
                }
                for k in 0..n2 {
                    let s: String = c_aln2[k][col_lo..col_hi]
                        .iter()
                        .map(|&c| c as char)
                        .collect();
                    eprintln!("    g2[{}]: {}", k, s);
                }
                break;
            }
        }
    } else {
        eprintln!("  Widths differ: rust={} c={}", rust_width, c_aln_width);
    }

    eprintln!(
        "\nScores: rust={:.4} c={:.4} delta={:.4}",
        rust_aln.score,
        c_score,
        rust_aln.score - c_score
    );
}

/// Cross-validate Rust's FFT segment-detection against C's
/// `alignableReagion` on a BL50 input. Confirms our `match_score` /
/// `alignable_segments` produce byte-identical scores to C — eliminates
/// FFT scoring as the source of the BL50 §4 divergence.
///
/// Also includes a step-33 profile-DP comparison test that drives the
/// progressive merge to step 32, then compares Rust's `profile_align`
/// fallback to C's `A__align` on the divergent input.
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int};
use std::sync::Mutex;

use mafft_align::{GapModel, Profile, profile_align};
use mafft_io::read_fasta;
use mafft_scoring::build_context;
use mafft_tree::{ClusterMethod, DistanceMatrix, ktuple_distance, musclesupg};
use mafft_types::{ScoringModel, SeqType};

static C_MUTEX: Mutex<()> = Mutex::new(());

unsafe fn alloc_zeroed(size: usize) -> *mut u8 {
    let layout = std::alloc::Layout::from_size_align(size.max(8), 8).unwrap();
    unsafe { std::alloc::alloc_zeroed(layout) }
}

unsafe fn init_c_protein_bl50() {
    unsafe {
        mafft_sys::initglobalvariables();
        std::ptr::addr_of_mut!(mafft_sys::ppenalty).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::ppenalty_ex).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::ppenalty_EX).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::ppenalty_OP).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::ppenalty_dist).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::poffset).write(0);
        std::ptr::addr_of_mut!(mafft_sys::kimuraR).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::pamN).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::dorp).write(b'p' as i32);
        std::ptr::addr_of_mut!(mafft_sys::scoremtx).write(1);
        std::ptr::addr_of_mut!(mafft_sys::nblosum).write(50);
        std::ptr::addr_of_mut!(mafft_sys::fmodel).write(0);
        std::ptr::addr_of_mut!(mafft_sys::fftWinSize).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::fftThreshold).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::consweight_multi).write(1.0);

        let seq_data = b"ACDEFGHIKLMNPQRSTVWY\0";
        let mut seq_ptr = seq_data.as_ptr() as *mut i8;
        let seq_arr: *mut *mut i8 = &mut seq_ptr;
        mafft_sys::constants(1, seq_arr);
    }
}

#[test]
fn bl50_alignable_reagion_matches_c() {
    let _guard = C_MUTEX.lock().unwrap();

    let input = read_fasta(std::path::Path::new("../../mafft-upstream/test/sample"))
        .expect("load mafft-upstream/test/sample");
    let scoring = build_context(ScoringModel::Blosum(50), SeqType::Protein);
    let sequences: Vec<Vec<u8>> = input.sequences.iter().map(|s| s.data.clone()).collect();

    // Pick the first two sequences. These are 353 and 348 residues long,
    // exercising the same per-position scoring as the post-step-32 13×8
    // cluster pair (which contains them) without needing to drive the
    // full progressive merge.
    let s1 = &sequences[0];
    let s2 = &sequences[1];

    // Rust: build single-sequence profiles and compute per-position
    // scores at lag=0.
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

    let len = prof1.length.min(prof2.length);
    let mut rust_scores = vec![0.0f64; len];
    for i in 0..len {
        rust_scores[i] = prof1.match_score(i, &prof2, i, &scoring.consweight_matrix);
    }

    // Run Rust segment detection (matches the BL50 default protein params).
    let rust_segs =
        mafft_fft::alignable_segments(&rust_scores, &mafft_fft::SegmentParams::protein());

    // C: call alignableReagion with the same input.
    let mut c_seg_data: Vec<(i32, i32, i32, f64)> = Vec::new();
    unsafe {
        init_c_protein_bl50();

        let cs1 = CString::new(s1.clone()).unwrap();
        let cs2 = CString::new(s2.clone()).unwrap();
        let mut cs1_ptr: *mut c_char = cs1.as_ptr() as *mut c_char;
        let mut cs2_ptr: *mut c_char = cs2.as_ptr() as *mut c_char;

        let mut eff1 = vec![1.0f64];
        let mut eff2 = vec![1.0f64];

        const MAX_SEG: usize = 1000;
        let seg_buf: *mut mafft_sys::Segment =
            alloc_zeroed(MAX_SEG * std::mem::size_of::<mafft_sys::Segment>()) as _;

        let s1_arr: *mut *mut c_char = &mut cs1_ptr;
        let s2_arr: *mut *mut c_char = &mut cs2_ptr;

        let count = mafft_sys::alignableReagion(
            1,
            1,
            s1_arr,
            s2_arr,
            eff1.as_mut_ptr(),
            eff2.as_mut_ptr(),
            seg_buf,
        ) as usize;

        for i in 0..count {
            let s = &*seg_buf.add(i);
            c_seg_data.push((s.start, s.end, s.center, s.score));
        }

        // Cleanup
        mafft_sys::alignableReagion(
            0,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        mafft_sys::freeconstants();
    }

    eprintln!(
        "Rust: {} segments, C: {} segments",
        rust_segs.len(),
        c_seg_data.len()
    );
    assert_eq!(
        rust_segs.len(),
        c_seg_data.len(),
        "BL50 segment count differs: Rust={} C={}",
        rust_segs.len(),
        c_seg_data.len()
    );

    for (i, (rs, (cs, ce, cc, cscore))) in rust_segs.iter().zip(c_seg_data.iter()).enumerate() {
        assert_eq!(
            rs.start, *cs as usize,
            "seg[{i}] start: rust={} c={}",
            rs.start, cs
        );
        assert_eq!(
            rs.end, *ce as usize,
            "seg[{i}] end: rust={} c={}",
            rs.end, ce
        );
        assert_eq!(
            rs.center, *cc as usize,
            "seg[{i}] center: rust={} c={}",
            rs.center, cc
        );
        assert!(
            (rs.score - cscore).abs() < 1e-6,
            "seg[{i}] score: rust={} c={}",
            rs.score,
            cscore
        );
    }
}

/// Drive Rust progressive merge through step 23 (so step 24 is next),
/// then directly compare Rust's `profile_align` and C's `A__align`
/// outputs on the BL50 step-24 (clus1=6, clus2=1) input. Step 24 was
/// the first divergent step before the FMA fix landed (TODO §4) — it
/// stays a sharper-than-end-to-end regression because it directly
/// compares the inner DP against C and would re-fire if any future
/// change reintroduces a 1-ULP FMA-vs-non-FMA accumulation
/// difference.
#[test]
fn bl50_step24_profile_dp_matches_c_a_align() {
    let _guard = C_MUTEX.lock().unwrap();

    // Build the same topology the engine uses for FFT-NS-2 (ktuple
    // distance + UPGMA).
    let input = read_fasta(std::path::Path::new("../../mafft-upstream/test/sample"))
        .expect("load mafft-upstream/test/sample");
    let scoring = build_context(ScoringModel::Blosum(50), SeqType::Protein);
    let nseq = input.sequences.len();
    let raw_seqs: Vec<Vec<u8>> = input.sequences.iter().map(|s| s.data.clone()).collect();

    let mut dm = DistanceMatrix::new(nseq);
    for i in 0..nseq {
        for j in (i + 1)..nseq {
            dm.set(i, j, ktuple_distance(&raw_seqs[i], &raw_seqs[j], 6));
        }
    }
    let topo = musclesupg(&dm, ClusterMethod::default());

    const TARGET_STEP: usize = 24;

    // Run progressive_align for the first TARGET_STEP merges, capturing
    // the intermediate aligned[] state. Each sequence's length matches
    // its current cluster width.
    let aligned = mafft_core::progressive_align_partial(
        &raw_seqs,
        &topo,
        &scoring,
        /* use_fft = */ true,
        /* shift_penalty = */ None,
        TARGET_STEP,
    );

    let step = &topo.steps[TARGET_STEP];
    let group1: Vec<usize> = step.left.clone();
    let group2: Vec<usize> = step.right.clone();

    // Sequence-weights computed once for the whole topology, normalized
    // per group (mirrors merge_step_cached's wn).
    let weights = mafft_tree::sequence_weights(&topo);
    let w1: Vec<f64> = group1.iter().map(|&i| weights[i]).collect();
    let w2: Vec<f64> = group2.iter().map(|&i| weights[i]).collect();
    let s1w: f64 = w1.iter().sum();
    let s2w: f64 = w2.iter().sum();
    let w1n: Vec<f64> = w1.iter().map(|w| w / s1w).collect();
    let w2n: Vec<f64> = w2.iter().map(|w| w / s2w).collect();

    let s1_refs: Vec<&[u8]> = group1.iter().map(|&i| aligned[i].as_slice()).collect();
    let s2_refs: Vec<&[u8]> = group2.iter().map(|&i| aligned[i].as_slice()).collect();
    let prof1 = Profile::from_aligned(&s1_refs, &w1n, &scoring.amino_map, scoring.nalphabets);
    let prof2 = Profile::from_aligned(&s2_refs, &w2n, &scoring.amino_map, scoring.nalphabets);

    let gap = GapModel::new(scoring.gap.open as f64, scoring.gap.extend as f64);

    // Rust DP: mirror the FFT fallback path (head_gap=false, tail_gap=false
    // — engine's FftAlignParams sets both to false).
    let rust_aln = profile_align(
        &prof1,
        &prof2,
        &scoring.consweight_matrix,
        &gap,
        false,
        false,
    );

    // Materialize Rust's first-of-cluster1 aligned sequence for byte
    // comparison against C.
    let mut rust_s1: Vec<u8> = Vec::with_capacity(rust_aln.operations.len());
    let mut p = 0usize;
    for op in &rust_aln.operations {
        match op {
            mafft_align::AlignOp::Match | mafft_align::AlignOp::Delete => {
                rust_s1.push(aligned[group1[0]][p]);
                p += 1;
            }
            mafft_align::AlignOp::Insert => rust_s1.push(b'-'),
        }
    }

    // Now run C's A__align via FFI on the same prof1/prof2 input.
    unsafe {
        init_c_protein_bl50();
        std::ptr::addr_of_mut!(mafft_sys::njob).write(nseq as c_int);
        let c_penalty = std::ptr::addr_of!(mafft_sys::penalty).read();
        let c_penalty_ex = std::ptr::addr_of!(mafft_sys::penalty_ex).read();

        let alloclen = (aligned[group1[0]].len() + aligned[group2[0]].len() + 1000) as c_int;

        // C strings (CString to ensure trailing null), then resize each
        // buffer to alloclen+1 for in-place A__align edits.
        let c_seqs1: Vec<CString> = group1
            .iter()
            .map(|&i| CString::new(aligned[i].clone()).unwrap())
            .collect();
        let c_seqs2: Vec<CString> = group2
            .iter()
            .map(|&i| CString::new(aligned[i].clone()).unwrap())
            .collect();
        let mut buf1: Vec<Vec<u8>> = c_seqs1
            .iter()
            .map(|s| {
                let mut v = s.as_bytes().to_vec();
                v.resize(alloclen as usize + 1, 0);
                v
            })
            .collect();
        let mut buf2: Vec<Vec<u8>> = c_seqs2
            .iter()
            .map(|s| {
                let mut v = s.as_bytes().to_vec();
                v.resize(alloclen as usize + 1, 0);
                v
            })
            .collect();
        let mut p1: Vec<*mut c_char> = buf1
            .iter_mut()
            .map(|v| v.as_mut_ptr() as *mut c_char)
            .collect();
        let mut p2: Vec<*mut c_char> = buf2
            .iter_mut()
            .map(|v| v.as_mut_ptr() as *mut c_char)
            .collect();
        let mut e1: Vec<c_double> = w1n.clone();
        let mut e2: Vec<c_double> = w2n.clone();

        let nalpha = scoring.substitution_matrix.len() as c_int;
        let n_dyn = mafft_sys::AllocateDoubleMtx(nalpha, nalpha);
        for i in 0..scoring.substitution_matrix.len() {
            for j in 0..scoring.substitution_matrix[i].len() {
                *(*n_dyn.add(i)).add(j) = scoring.substitution_matrix[i][j] as f64;
            }
        }

        let mut impmatch = 0.0f64;
        let c_score = mafft_sys::A__align(
            n_dyn,
            c_penalty,
            c_penalty_ex,
            p1.as_mut_ptr(),
            p2.as_mut_ptr(),
            e1.as_mut_ptr(),
            e2.as_mut_ptr(),
            group1.len() as c_int,
            group2.len() as c_int,
            alloclen,
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
            0.0,
            0.0,
        );

        let c_width = {
            let s = p1[0];
            let mut k = 0;
            while *s.add(k) != 0 {
                k += 1;
            }
            k
        };
        let c_s1: Vec<u8> = (0..c_width).map(|k| *p1[0].add(k) as u8).collect();

        // Cleanup before we panic on assert mismatch.
        mafft_sys::A__align(
            std::ptr::null_mut(),
            0,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
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
            0,
            std::ptr::null_mut(),
            0,
            0,
            -1,
            -1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0.0,
            0.0,
        );
        mafft_sys::freeconstants();

        // Both DPs should produce byte-identical output (post-FMA fix).
        // Pre-fix this differed at the F-residue placement around column
        // 28-32 (Rust `DNFYVPF----SNK` vs C `DNFYVP----FSNK`).
        assert_eq!(
            rust_s1.len(),
            c_width,
            "BL50 step-24 width: Rust={} C={} Rust_score={:.2} C_score={:.2}",
            rust_s1.len(),
            c_width,
            rust_aln.score,
            c_score
        );
        assert!(
            (rust_aln.score - c_score).abs() < 0.01,
            "BL50 step-24 score: Rust={:.6} C={:.6}",
            rust_aln.score,
            c_score
        );
        assert_eq!(
            rust_s1,
            c_s1,
            "BL50 step-24 cluster1[0] alignment differs at first byte position {:?}",
            rust_s1.iter().zip(c_s1.iter()).position(|(a, b)| a != b)
        );
    }
}

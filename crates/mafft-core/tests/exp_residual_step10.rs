//! Focused dive into the `--exp 5.0` first21 step-10 divergence.
//! Step 10 is a 2-vs-2 merge that rust outputs at width 354 and C outputs
//! at width 353 — the first FFT-segmented merge where they disagree.
//! Replays both rust's `fft_profile_align` and C's `Falign` on the exact
//! step-10 inputs and compares.

#![allow(non_snake_case)]

use std::os::raw::{c_char, c_double, c_int};
use std::sync::Mutex;

use mafft_align::{FftAlignParams, GapModel, Profile, fft_profile_align};
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
#[cfg_attr(target_os = "linux", ignore = "potential glibc teardown")]
fn exp_residual_step10_rust_vs_c_falign() {
    let _g = C_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
    let dump_path = "/tmp/rs_dp.txt";
    let content = match std::fs::read_to_string(dump_path) {
        Ok(c) => c,
        Err(_) => {
            eprintln!("Dump not found at {dump_path}; run dump-generating engine first");
            return;
        }
    };
    let line = match content.lines().nth(10) {
        Some(l) => l,
        None => {
            eprintln!("dump has < 11 steps");
            return;
        }
    };
    let parts: std::collections::HashMap<&str, &str> = line
        .split('\t')
        .filter_map(|kv| kv.split_once('='))
        .collect();
    let g1: Vec<Vec<u8>> = parts["g1"]
        .split(';')
        .map(|s| s.as_bytes().to_vec())
        .collect();
    let g2: Vec<Vec<u8>> = parts["g2"]
        .split(';')
        .map(|s| s.as_bytes().to_vec())
        .collect();
    let w1: Vec<f64> = parts["w1"]
        .split(',')
        .filter_map(|x| x.parse().ok())
        .collect();
    let w2: Vec<f64> = parts["w2"]
        .split(',')
        .filter_map(|x| x.parse().ok())
        .collect();
    let penalty: i32 = parts["pen"].parse().unwrap();
    let penalty_ex: i32 = parts["pen_ex"].parse().unwrap();
    let headgp: c_int = parts["hgp"].parse::<i32>().unwrap();
    let tailgp: c_int = parts["tgp"].parse::<i32>().unwrap();
    let r_out1: Vec<Vec<u8>> = parts["out1"]
        .split(';')
        .map(|s| s.as_bytes().to_vec())
        .collect();
    let r_out2: Vec<Vec<u8>> = parts["out2"]
        .split(';')
        .map(|s| s.as_bytes().to_vec())
        .collect();

    eprintln!("=== Step 10 inputs ===");
    eprintln!("  g1: {} seqs of width {}", g1.len(), g1[0].len());
    eprintln!("  g2: {} seqs of width {}", g2.len(), g2[0].len());
    eprintln!("  penalty={}, penalty_ex={}", penalty, penalty_ex);
    eprintln!("  w1={:?}, w2={:?}", w1, w2);
    eprintln!(
        "Rust output widths: g1={}, g2={}",
        r_out1[0].len(),
        r_out2[0].len()
    );

    // Re-run rust's fft_profile_align on the SAME inputs to verify the
    // dump faithfully captures what merge_step_cached produced.
    let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);
    // Normalize weights as merge_step_cached does (per-group sum-to-1).
    let s1: f64 = w1.iter().sum();
    let s2: f64 = w2.iter().sum();
    let w1n: Vec<f64> = if s1 > 0.0 {
        w1.iter().map(|x| x / s1).collect()
    } else {
        vec![1.0; w1.len()]
    };
    let w2n: Vec<f64> = if s2 > 0.0 {
        w2.iter().map(|x| x / s2).collect()
    } else {
        vec![1.0; w2.len()]
    };
    let g1_refs: Vec<&[u8]> = g1.iter().map(|v| v.as_slice()).collect();
    let g2_refs: Vec<&[u8]> = g2.iter().map(|v| v.as_slice()).collect();
    let prof1 = Profile::from_aligned(&g1_refs, &w1n, &scoring.amino_map, scoring.nalphabets);
    let prof2 = Profile::from_aligned(&g2_refs, &w2n, &scoring.amino_map, scoring.nalphabets);

    // Build polarity/volume property channels for FFT (matches engine setup).
    let nalpha = scoring.nscoredalphabets;
    let polarity: Vec<f64> = (0..nalpha)
        .map(|a| scoring.polarity[(a + b'A' as usize).min(255)])
        .collect();
    let volume: Vec<f64> = (0..nalpha)
        .map(|a| scoring.volume[(a + b'A' as usize).min(255)])
        .collect();
    let property_channels = if !polarity.is_empty() && !volume.is_empty() {
        Some((polarity.clone(), volume.clone()))
    } else {
        None
    };

    let gap = GapModel::new(penalty as f64, penalty_ex as f64);
    let mut fft_params = FftAlignParams::protein();
    fft_params.gap = gap.clone();
    fft_params.head_gap = headgp != 0;
    fft_params.tail_gap = tailgp != 0;
    fft_params.num_channels = nalpha;
    fft_params.property_channels = property_channels;
    let rust_aln = fft_profile_align(&prof1, &prof2, &scoring.consweight_matrix, &fft_params);
    let rust_width = rust_aln.operations.len();
    eprintln!(
        "rust replay fft_profile_align: width = {} (recorded out: {})",
        rust_width,
        r_out1[0].len()
    );

    // Now call C's Falign on the same inputs.
    unsafe {
        init_c_protein();

        let nalpha_c = scoring.substitution_matrix.len() as c_int;
        let n_dyn = mafft_sys::AllocateDoubleMtx(nalpha_c, nalpha_c);
        for i in 0..scoring.substitution_matrix.len() {
            for j in 0..scoring.substitution_matrix[i].len() {
                *(*n_dyn.add(i)).add(j) = scoring.substitution_matrix[i][j] as f64;
            }
        }
        std::ptr::addr_of_mut!(mafft_sys::alg).write(b'A' as i8);
        std::ptr::addr_of_mut!(mafft_sys::fftkeika).write(1);
        std::ptr::addr_of_mut!(mafft_sys::kobetsubunkatsu).write(0);
        std::ptr::addr_of_mut!(mafft_sys::use_fft).write(1);
        std::ptr::addr_of_mut!(mafft_sys::outgap).write(tailgp);
        std::ptr::addr_of_mut!(mafft_sys::fftThreshold).write(50);
        std::ptr::addr_of_mut!(mafft_sys::penalty).write(penalty);
        std::ptr::addr_of_mut!(mafft_sys::penalty_ex).write(penalty_ex);

        let alloclen = (g1[0].len() + g2[0].len()) * 4 + 100;
        let c_s1_boxed: Vec<Box<[u8]>> = g1
            .iter()
            .map(|s| {
                let mut v = s.clone();
                v.resize(alloclen + 1, 0);
                v.into_boxed_slice()
            })
            .collect();
        let c_s2_boxed: Vec<Box<[u8]>> = g2
            .iter()
            .map(|s| {
                let mut v = s.clone();
                v.resize(alloclen + 1, 0);
                v.into_boxed_slice()
            })
            .collect();
        let mut c_s1_ptrs: Vec<*mut c_char> = c_s1_boxed
            .iter()
            .map(|v| v.as_ptr() as *mut c_char)
            .collect();
        let mut c_s2_ptrs: Vec<*mut c_char> = c_s2_boxed
            .iter()
            .map(|v| v.as_ptr() as *mut c_char)
            .collect();
        let e1: *mut c_double = alloc_zeroed(g1.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in w1n.iter().enumerate() {
            *e1.add(i) = w;
        }
        let e2: *mut c_double = alloc_zeroed(g2.len() * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in w2n.iter().enumerate() {
            *e2.add(i) = w;
        }
        let mut fftlog: c_int = 0;
        let c_score = mafft_sys::Falign(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            n_dyn,
            c_s1_ptrs.as_mut_ptr(),
            c_s2_ptrs.as_mut_ptr(),
            e1,
            e2,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            g1.len() as c_int,
            g2.len() as c_int,
            alloclen as c_int,
            &mut fftlog as *mut c_int,
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
        );

        let c_w1 = {
            let s = c_s1_ptrs[0];
            let mut n = 0;
            while *s.add(n) != 0 {
                n += 1;
            }
            n
        };
        let c_w2 = {
            let s = c_s2_ptrs[0];
            let mut n = 0;
            while *s.add(n) != 0 {
                n += 1;
            }
            n
        };
        eprintln!(
            "C Falign: width g1={}, width g2={}, score={:.2}",
            c_w1, c_w2, c_score
        );

        let c_aln1: Vec<u8> = c_s1_boxed[0][..c_w1].to_vec();
        let c_aln2: Vec<u8> = c_s2_boxed[0][..c_w2].to_vec();
        eprintln!("Rust aligned[0]: width {}", r_out1[0].len());
        let rust_w_g1 = r_out1[0].len();
        // First diverging column between rust and C
        for i in 0..rust_w_g1.min(c_w1) {
            if r_out1[0][i] != c_aln1[i] {
                eprintln!(
                    "First diff in g1 row 0 at col {i}: rust={} c={}",
                    r_out1[0][i] as char, c_aln1[i] as char
                );
                let lo = i.saturating_sub(15);
                let hi = (i + 15).min(rust_w_g1).min(c_w1);
                eprintln!("  rust: {}", String::from_utf8_lossy(&r_out1[0][lo..hi]));
                eprintln!("  C:    {}", String::from_utf8_lossy(&c_aln1[lo..hi]));
                break;
            }
        }
        for i in 0..rust_w_g1.min(c_w2) {
            if r_out2[0][i] != c_aln2[i] {
                eprintln!(
                    "First diff in g2 row 0 at col {i}: rust={} c={}",
                    r_out2[0][i] as char, c_aln2[i] as char
                );
                break;
            }
        }

        mafft_sys::FreeDoubleMtx(n_dyn);
    }
}

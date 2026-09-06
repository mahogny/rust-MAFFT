//! Cell-level investigation of the `--exp ≥ 4.30` FFT-NS-2 tied-trace
//! residual. Reads a per-merge-step dump produced by setting
//! `RS_DP_DUMP=<path>` on a rust run, then for each merge step calls C's
//! `A__align` via mafft-sys FFI on the SAME inputs and compares the
//! aligned outputs against rust's recorded outputs.
//!
//! The dump is whitespace-tolerant; lines look like:
//!   g1=A;B;C\tg2=X;Y\tw1=0.1,0.2,0.3\tw2=0.4,0.5\tpen=-917\tpen_ex=-2999\t
//!   hgp=0\ttgp=0\tfft=1\tcon=0\tout1=...\tout2=...
//!
//! Usage:
//!   1. cargo build --release --bin mafft-rs
//!   2. RS_DP_DUMP=/tmp/rs_dp.txt target/release/mafft-rs --exp 5.0 \
//!        /path/to/first21.fa > /tmp/r.fa
//!   3. cargo test --release -p mafft-core --test exp_residual_dump_compare \
//!        -- --nocapture
//!
//! The test searches for `/tmp/rs_dp.txt`; if absent, prints a hint and
//! returns success (CI-safe).

#![allow(non_snake_case)]

use std::os::raw::{c_char, c_double, c_int};

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

unsafe fn alloc_zeroed(size: usize) -> *mut u8 {
    let layout = std::alloc::Layout::from_size_align(size.max(8), 8).unwrap();
    unsafe { std::alloc::alloc_zeroed(layout) }
}

unsafe fn build_c_dynamicmtx() -> *mut *mut c_double {
    unsafe {
        // Build a fresh double** by copying from the rust scoring context
        // (which mirrors C's BLOSUM62 matrix after constants() runs).
        use mafft_scoring::build_context;
        use mafft_types::{ScoringModel, SeqType};
        let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);
        let nalpha = scoring.substitution_matrix.len() as c_int;
        let mtx = mafft_sys::AllocateDoubleMtx(nalpha, nalpha);
        for i in 0..scoring.substitution_matrix.len() {
            for j in 0..scoring.substitution_matrix[i].len() {
                *(*mtx.add(i)).add(j) = scoring.substitution_matrix[i][j] as f64;
            }
        }
        mtx
    }
}

struct DumpEntry {
    g1: Vec<Vec<u8>>,
    g2: Vec<Vec<u8>>,
    w1: Vec<f64>,
    w2: Vec<f64>,
    penalty: i32,
    penalty_ex: i32,
    headgp: i32,
    tailgp: i32,
    use_fft: bool,
    has_constraint: bool,
    out1: Vec<Vec<u8>>,
    out2: Vec<Vec<u8>>,
}

fn parse_dump(path: &str) -> Vec<DumpEntry> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for line in content.lines() {
        let parts: std::collections::HashMap<&str, &str> = line
            .split('\t')
            .filter_map(|kv| kv.split_once('='))
            .collect();
        let parse_seqs = |k: &str| -> Vec<Vec<u8>> {
            parts
                .get(k)
                .map(|s| s.split(';').map(|x| x.as_bytes().to_vec()).collect())
                .unwrap_or_default()
        };
        let parse_weights = |k: &str| -> Vec<f64> {
            parts
                .get(k)
                .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
                .unwrap_or_default()
        };
        let parse_i = |k: &str| -> i32 { parts.get(k).and_then(|s| s.parse().ok()).unwrap_or(0) };
        out.push(DumpEntry {
            g1: parse_seqs("g1"),
            g2: parse_seqs("g2"),
            w1: parse_weights("w1"),
            w2: parse_weights("w2"),
            penalty: parse_i("pen"),
            penalty_ex: parse_i("pen_ex"),
            headgp: parse_i("hgp"),
            tailgp: parse_i("tgp"),
            use_fft: parse_i("fft") != 0,
            has_constraint: parse_i("con") != 0,
            out1: parse_seqs("out1"),
            out2: parse_seqs("out2"),
        });
    }
    out
}

#[test]
#[cfg_attr(
    target_os = "linux",
    ignore = "potential glibc teardown issue mirroring other FFI tests"
)]
fn exp_residual_compare_per_step_with_a__align() {
    let dump_path = "/tmp/rs_dp.txt";
    if !std::path::Path::new(dump_path).exists() {
        eprintln!("Dump not found at {dump_path}; run with RS_DP_DUMP=/tmp/rs_dp.txt first");
        return;
    }
    let entries = parse_dump(dump_path);
    if entries.is_empty() {
        eprintln!("Dump empty");
        return;
    }
    eprintln!("Loaded {} merge steps from {}", entries.len(), dump_path);

    unsafe {
        init_c_protein();
        let n_dyn = build_c_dynamicmtx();

        // Set globals Falign reads. Mirrors trace_refinement.rs setup.
        std::ptr::addr_of_mut!(mafft_sys::alg).write(b'A' as i8);
        std::ptr::addr_of_mut!(mafft_sys::fftkeika).write(1);
        std::ptr::addr_of_mut!(mafft_sys::kobetsubunkatsu).write(0);
        std::ptr::addr_of_mut!(mafft_sys::use_fft).write(1);
        std::ptr::addr_of_mut!(mafft_sys::fftThreshold).write(50);

        let mut first_divergence: Option<usize> = None;
        for (idx, entry) in entries.iter().enumerate() {
            if entry.has_constraint {
                continue;
            } // constraint-aware path uses a different DP
            // Set up A__align inputs.
            let len1_max = entry.g1.iter().map(|s| s.len()).max().unwrap_or(0);
            let len2_max = entry.g2.iter().map(|s| s.len()).max().unwrap_or(0);
            let alloclen = (len1_max + len2_max) * 4 + 100;

            let c_seq1_boxed: Vec<Box<[u8]>> = entry
                .g1
                .iter()
                .map(|s| {
                    let mut v = s.clone();
                    v.resize(alloclen + 1, 0);
                    v.into_boxed_slice()
                })
                .collect();
            let c_seq2_boxed: Vec<Box<[u8]>> = entry
                .g2
                .iter()
                .map(|s| {
                    let mut v = s.clone();
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

            // Normalize weights per-group (matches merge_step_cached's
            // C `fastconjuction_noname` per-group sum-to-1 normalization).
            let s1: f64 = entry.w1.iter().sum();
            let s2: f64 = entry.w2.iter().sum();
            let w1n: Vec<f64> = if s1 > 0.0 {
                entry.w1.iter().map(|x| x / s1).collect()
            } else {
                vec![1.0; entry.w1.len()]
            };
            let w2n: Vec<f64> = if s2 > 0.0 {
                entry.w2.iter().map(|x| x / s2).collect()
            } else {
                vec![1.0; entry.w2.len()]
            };
            let eff1: *mut c_double =
                alloc_zeroed(entry.w1.len() * std::mem::size_of::<c_double>()) as _;
            for (i, &w) in w1n.iter().enumerate() {
                *eff1.add(i) = w;
            }
            let eff2: *mut c_double =
                alloc_zeroed(entry.w2.len() * std::mem::size_of::<c_double>()) as _;
            for (i, &w) in w2n.iter().enumerate() {
                *eff2.add(i) = w;
            }

            // For FFT-mode steps, call Falign (the FFT-segmented DP).
            // For non-FFT steps, call A__align (the direct profile DP).
            if entry.use_fft {
                // NO Falign reset — let state accumulate as in C MAFFT's
                // actual progressive flow.
                std::ptr::addr_of_mut!(mafft_sys::penalty).write(entry.penalty);
                std::ptr::addr_of_mut!(mafft_sys::penalty_ex).write(entry.penalty_ex);
                std::ptr::addr_of_mut!(mafft_sys::outgap).write(entry.tailgp);
                let mut fftlog: c_int = 0;
                let _c_score = mafft_sys::Falign(
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    n_dyn,
                    c_seq1_ptrs.as_mut_ptr(),
                    c_seq2_ptrs.as_mut_ptr(),
                    eff1,
                    eff2,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    entry.g1.len() as c_int,
                    entry.g2.len() as c_int,
                    alloclen as c_int,
                    &mut fftlog as *mut c_int,
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null_mut(),
                );
            } else {
                let _c_score = mafft_sys::A__align(
                    n_dyn,
                    entry.penalty as c_int,
                    entry.penalty_ex as c_int,
                    c_seq1_ptrs.as_mut_ptr(),
                    c_seq2_ptrs.as_mut_ptr(),
                    eff1,
                    eff2,
                    entry.g1.len() as c_int,
                    entry.g2.len() as c_int,
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
                    entry.headgp,
                    entry.tailgp,
                    -1,
                    -1,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0.0,
                    0.0,
                );
            }

            // Read back C's output for each row.
            let read_cstr = |ptr: *mut c_char| -> Vec<u8> {
                let mut v = Vec::new();
                let mut k = 0;
                loop {
                    let c = *ptr.add(k);
                    if c == 0 {
                        break;
                    }
                    v.push(c as u8);
                    k += 1;
                }
                v
            };
            let c_out1: Vec<Vec<u8>> = (0..entry.g1.len())
                .map(|i| read_cstr(c_seq1_ptrs[i]))
                .collect();
            let c_out2: Vec<Vec<u8>> = (0..entry.g2.len())
                .map(|i| read_cstr(c_seq2_ptrs[i]))
                .collect();

            // Compare to rust's stored output.
            let widths_c = (
                c_out1.first().map(|v| v.len()).unwrap_or(0),
                c_out2.first().map(|v| v.len()).unwrap_or(0),
            );
            let widths_r = (
                entry.out1.first().map(|v| v.len()).unwrap_or(0),
                entry.out2.first().map(|v| v.len()).unwrap_or(0),
            );

            let mut divergent_rows: Vec<usize> = Vec::new();
            for i in 0..entry.g1.len() {
                if c_out1.get(i) != entry.out1.get(i) {
                    divergent_rows.push(i);
                }
            }
            for i in 0..entry.g2.len() {
                if c_out2.get(i) != entry.out2.get(i) {
                    divergent_rows.push(entry.g1.len() + i);
                }
            }

            if widths_c != widths_r {
                eprintln!(
                    "Step {idx}: WIDTH DIVERGE — rust=({},{}) C=({},{})  (g1.len={}, g2.len={}, use_fft={}, penalty_ex={})",
                    widths_r.0,
                    widths_r.1,
                    widths_c.0,
                    widths_c.1,
                    entry.g1.len(),
                    entry.g2.len(),
                    entry.use_fft,
                    entry.penalty_ex
                );
                if first_divergence.is_none() {
                    first_divergence = Some(idx);
                }
            } else if !divergent_rows.is_empty() {
                eprintln!(
                    "Step {idx}: CONTENT DIVERGE in {} row(s) ({}+{} rows, widths {},{}, use_fft={}, penalty_ex={})",
                    divergent_rows.len(),
                    entry.g1.len(),
                    entry.g2.len(),
                    widths_r.0,
                    widths_r.1,
                    entry.use_fft,
                    entry.penalty_ex
                );
                if first_divergence.is_none() {
                    first_divergence = Some(idx);
                }
            }
        }
        mafft_sys::FreeDoubleMtx(n_dyn);

        if let Some(k) = first_divergence {
            eprintln!("FIRST DIVERGENCE at merge step {k}");
        } else {
            eprintln!("All steps match between rust output and C A__align on same inputs.");
        }
    }
}

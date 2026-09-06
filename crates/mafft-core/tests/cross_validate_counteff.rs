//! FFI cross-check `sequence_weights` (Rust port of C's
//! `counteff_simple_double_nostatic_memsave`) against C's
//! `counteff_simple_double` on real-world BB20027 pass-1 topology.
//!
//! Goal: pin down whether the pass-1 progressive divergence (BB20027,
//! +28 col delta) is caused by per-leaf weight drift. Pass-1 Newick
//! is byte-identical between C and Rust, so if weights also match, the
//! bug is in profile blending / DP — not weights.

use std::os::raw::{c_double, c_int};
use std::path::PathBuf;

fn bb20027_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bali3.BB20027.fa")
}

use mafft_tree::{Topology, sequence_weights};

unsafe fn alloc_zero(size: usize) -> *mut std::ffi::c_void {
    let layout = std::alloc::Layout::from_size_align(size.max(8), 8).unwrap();
    unsafe { std::alloc::alloc_zeroed(layout) as *mut _ }
}

unsafe fn c_counteff(topo: &Topology) -> Vec<f64> {
    let nseq = topo.nseq as c_int;
    let nsteps = topo.steps.len();

    let topol: *mut *mut *mut c_int =
        unsafe { alloc_zero(nsteps * std::mem::size_of::<*mut *mut c_int>()) } as _;
    for k in 0..nsteps {
        let row: *mut *mut c_int =
            unsafe { alloc_zero(2 * std::mem::size_of::<*mut c_int>()) } as _;
        let left = &topo.steps[k].left;
        let left_arr: *mut c_int =
            unsafe { alloc_zero((left.len() + 1) * std::mem::size_of::<c_int>()) } as _;
        for (i, &s) in left.iter().enumerate() {
            unsafe {
                *left_arr.add(i) = s as c_int;
            }
        }
        unsafe {
            *left_arr.add(left.len()) = -1;
        }
        unsafe {
            *row.add(0) = left_arr;
        }
        let right = &topo.steps[k].right;
        let right_arr: *mut c_int =
            unsafe { alloc_zero((right.len() + 1) * std::mem::size_of::<c_int>()) } as _;
        for (i, &s) in right.iter().enumerate() {
            unsafe {
                *right_arr.add(i) = s as c_int;
            }
        }
        unsafe {
            *right_arr.add(right.len()) = -1;
        }
        unsafe {
            *row.add(1) = right_arr;
        }
        unsafe {
            *topol.add(k) = row;
        }
    }
    let len: *mut *mut c_double =
        unsafe { alloc_zero(nsteps * std::mem::size_of::<*mut c_double>()) } as _;
    for k in 0..nsteps {
        let row: *mut c_double = unsafe { alloc_zero(2 * std::mem::size_of::<c_double>()) } as _;
        unsafe {
            *row.add(0) = topo.steps[k].left_length;
            *row.add(1) = topo.steps[k].right_length;
            *len.add(k) = row;
        }
    }

    let mut node = vec![0.0f64; topo.nseq];
    unsafe {
        mafft_sys::counteff_simple_double(nseq, topol, len, node.as_mut_ptr());
    }

    // Leak allocations on purpose — mixing Rust alloc / C free is UB.
    // This is a one-shot test; the OS reclaims everything on exit.

    node
}

#[test]
fn bb20027_pass1_widths_step_by_step() {
    // Run the engine on BB20027 and capture msa.step_trace from pass 0 + pass 1.
    let path = bb20027_fixture();
    let input = mafft_io::read_fasta(&path).expect("read BB20027");
    let engine = mafft_core::MafftEngine::new(mafft_core::AlignmentMode::FftNs2);
    let msa = engine.align(&input);
    let nsteps_per_pass = msa.sequences.len() - 1; // 28 for BB20027
    eprintln!(
        "step_trace.len() = {}, nsteps_per_pass = {}",
        msa.step_trace.len(),
        nsteps_per_pass
    );
    eprintln!(
        "\n=== pass 1 step widths (last {} steps) ===",
        nsteps_per_pass
    );
    for (i, st) in msa
        .step_trace
        .iter()
        .skip(msa.step_trace.len().saturating_sub(nsteps_per_pass))
        .enumerate()
    {
        eprintln!(
            "  step {}: clus1={} clus2={} width={} score={:.1}",
            i, st.clus1, st.clus2, st.width, st.score
        );
    }
}

#[test]
fn bb20027_pass1_weights_match_c() {
    let path = bb20027_fixture();
    let input = mafft_io::read_fasta(&path).expect("read BB20027");
    let engine = mafft_core::MafftEngine::new(mafft_core::AlignmentMode::FftNs2);
    let msa = engine.align(&input);
    let topo = msa.guide_tree.expect("guide_tree missing");
    eprintln!(
        "BB20027 pass-1 topology: {} seqs, {} steps",
        topo.nseq,
        topo.steps.len()
    );

    let rust_w = sequence_weights(&topo);

    unsafe {
        mafft_sys::initglobalvariables();
        std::ptr::addr_of_mut!(mafft_sys::sueff_global).write(0.1);
        std::ptr::addr_of_mut!(mafft_sys::treemethod).write(b'X' as c_int);
        std::ptr::addr_of_mut!(mafft_sys::njob).write(topo.nseq as c_int);
    }
    let c_w = unsafe { c_counteff(&topo) };

    println!("\n{:<5} {:<22} {:<22} {:<14}", "idx", "rust", "c", "diff");
    let mut max_diff: f64 = 0.0;
    for i in 0..topo.nseq {
        let d = (rust_w[i] - c_w[i]).abs();
        max_diff = max_diff.max(d);
        let flag = if d > 1e-12 { " <-- DIFF" } else { "" };
        println!(
            "{:<5} {:<22.16} {:<22.16} {:<14.3e}{}",
            i, rust_w[i], c_w[i], d, flag
        );
    }
    println!("max |diff| = {:.3e}", max_diff);
    assert!(
        max_diff < 1e-10,
        "BB20027 pass-1 weights diverge by up to {:.3e}",
        max_diff
    );
}

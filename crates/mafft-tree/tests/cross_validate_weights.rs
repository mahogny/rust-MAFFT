/// Cross-validate Rust BranchWeights against C's weightFromABranch.
use std::os::raw::{c_double, c_int};
use std::ptr;

use mafft_tree::{BranchWeights, ClusterMethod, DistanceMatrix, JoinStep, Topology, musclesupg};

/// Build C's topology arrays (int*** with -1 sentinels) and branch length
/// arrays (double**) from a Rust Topology, call treeCnv + calcBranchWeight
/// + weightFromABranch, and return the per-branch weight vectors.
///
/// This is the ground truth from C.
unsafe fn c_branch_weights(topo: &Topology) -> Vec<Vec<Vec<f64>>> {
    unsafe {
        let nseq = topo.nseq as c_int;
        let nsteps = topo.steps.len();

        // Set C globals.
        mafft_sys::sueff_global = 0.1;
        mafft_sys::treemethod = b'X' as c_int;

        // Allocate topol: int*** — topol[step][0/1] = int* (member list, -1 terminated)
        let topol: *mut *mut *mut c_int =
            libc_alloc_zeroed(nsteps * std::mem::size_of::<*mut *mut c_int>()) as _;
        for k in 0..nsteps {
            let row: *mut *mut c_int =
                libc_alloc_zeroed(2 * std::mem::size_of::<*mut c_int>()) as _;
            // Left members
            let left = &topo.steps[k].left;
            let left_arr: *mut c_int =
                libc_alloc_zeroed((left.len() + 1) * std::mem::size_of::<c_int>()) as _;
            for (i, &s) in left.iter().enumerate() {
                *left_arr.add(i) = s as c_int;
            }
            *left_arr.add(left.len()) = -1;
            *row.add(0) = left_arr;

            // Right members
            let right = &topo.steps[k].right;
            let right_arr: *mut c_int =
                libc_alloc_zeroed((right.len() + 1) * std::mem::size_of::<c_int>()) as _;
            for (i, &s) in right.iter().enumerate() {
                *right_arr.add(i) = s as c_int;
            }
            *right_arr.add(right.len()) = -1;
            *row.add(1) = right_arr;

            *topol.add(k) = row;
        }

        // Allocate len: double** — len[step][0/1]
        let len: *mut *mut c_double =
            libc_alloc_zeroed(nsteps * std::mem::size_of::<*mut c_double>()) as _;
        for k in 0..nsteps {
            let row: *mut c_double = libc_alloc_zeroed(2 * std::mem::size_of::<c_double>()) as _;
            *row.add(0) = topo.steps[k].left_length;
            *row.add(1) = topo.steps[k].right_length;
            *len.add(k) = row;
        }

        // Allocate bw: double** — bw[step][0/1]
        let bw: *mut *mut c_double =
            libc_alloc_zeroed(nsteps * std::mem::size_of::<*mut c_double>()) as _;
        for k in 0..nsteps {
            let row: *mut c_double = libc_alloc_zeroed(2 * std::mem::size_of::<c_double>()) as _;
            *row.add(0) = 1.0;
            *row.add(1) = 1.0;
            *bw.add(k) = row;
        }

        // Allocate stopol: Node* — 2*nseq nodes
        let total_nodes = 2 * nseq as usize;
        let stopol: *mut mafft_sys::Node =
            libc_alloc_zeroed(total_nodes * std::mem::size_of::<mafft_sys::Node>()) as _;
        // Initialize children to NULL, tmpChildren to -1
        for i in 0..total_nodes {
            let node = &mut *stopol.add(i);
            node.children = [ptr::null_mut(); 3];
            node.tmpChildren = [-1; 3];
            node.length = [0.0; 3];
            node.weightptr = [ptr::null_mut(); 3];
            node.top = [-1; 3];
            node.members = [ptr::null_mut(); 3];
        }

        // Call C's treeCnv
        mafft_sys::treeCnv(stopol, nseq, topol, len, bw);

        // Call C's calcBranchWeight
        mafft_sys::calcBranchWeight(bw, nseq, stopol, topol, len);

        // Call C's weightFromABranch for each step/side
        let mut all_weights = Vec::new();
        for k in 0..nsteps {
            let mut step_weights = Vec::new();
            for side in 0..2u32 {
                let mut result = vec![0.0f64; nseq as usize];
                mafft_sys::weightFromABranch(
                    nseq,
                    result.as_mut_ptr(),
                    stopol,
                    topol,
                    k as c_int,
                    side as c_int,
                );
                step_weights.push(result);
            }
            all_weights.push(step_weights);
        }

        // Cleanup (leak for now — test only)
        all_weights
    }
}

unsafe fn libc_alloc_zeroed(size: usize) -> *mut u8 {
    unsafe {
        let layout = std::alloc::Layout::from_size_align(size.max(8), 8).unwrap();
        std::alloc::alloc_zeroed(layout)
    }
}

#[test]
fn branch_weights_match_c_6seq() {
    let nseq = 6;
    let mut dm = DistanceMatrix::new(nseq);
    for i in 0..nseq {
        for j in (i + 1)..nseq {
            dm.set(i, j, (j - i) as f64 * 0.1);
        }
    }
    let topo = musclesupg(&dm, ClusterMethod::default());

    // Get C's weights
    let c_weights = unsafe { c_branch_weights(&topo) };

    // Get Rust's weights
    let bw = BranchWeights::new(&topo);

    eprintln!("Topology:");
    for (k, step) in topo.steps.iter().enumerate() {
        eprintln!(
            "  step {k}: left={:?} right={:?} ll={:.6} rl={:.6}",
            step.left, step.right, step.left_length, step.right_length
        );
    }

    let mut max_diff = 0.0f64;
    let mut first_mismatch = None;

    for k in 0..topo.steps.len() {
        for side in 0..2 {
            let rust_w = bw.weights_for_branch(&topo, k, side);
            let c_w = &c_weights[k][side];

            for seq in 0..nseq {
                let diff = (rust_w[seq] - c_w[seq]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > 1e-6 && first_mismatch.is_none() {
                    first_mismatch = Some((k, side, seq, rust_w[seq], c_w[seq]));
                }
            }

            let rust_str: Vec<String> = rust_w.iter().map(|v| format!("{:.6}", v)).collect();
            let c_str: Vec<String> = c_w.iter().map(|v| format!("{:.6}", v)).collect();
            let match_str = if rust_w
                .iter()
                .zip(c_w.iter())
                .all(|(r, c)| (r - c).abs() < 1e-6)
            {
                "OK"
            } else {
                "DIFF"
            };
            eprintln!("  step={k} side={side} {match_str}");
            eprintln!("    Rust: [{}]", rust_str.join(", "));
            eprintln!("    C:    [{}]", c_str.join(", "));
        }
    }

    eprintln!("Max diff: {:.10}", max_diff);
    if let Some((k, side, seq, rw, cw)) = first_mismatch {
        eprintln!("First mismatch: step={k} side={side} seq={seq} rust={rw:.8} c={cw:.8}");
    }

    // Assert all weights match within tolerance
    for k in 0..topo.steps.len() {
        for side in 0..2 {
            let rust_w = bw.weights_for_branch(&topo, k, side);
            let c_w = &c_weights[k][side];
            for seq in 0..nseq {
                assert!(
                    (rust_w[seq] - c_w[seq]).abs() < 1e-4,
                    "step={k} side={side} seq={seq}: rust={:.8} c={:.8}",
                    rust_w[seq],
                    c_w[seq]
                );
            }
        }
    }
}

/// Test that our per-group normalization matches C's fastconjuction_noname.
#[test]
fn per_group_normalization_matches_c() {
    use std::os::raw::c_char;

    let nseq = 6;
    let mut dm = DistanceMatrix::new(nseq);
    for i in 0..nseq {
        for j in (i + 1)..nseq {
            dm.set(i, j, (j - i) as f64 * 0.1);
        }
    }
    let topo = musclesupg(&dm, ClusterMethod::default());

    // Build a sample branch split: group1 = {0}, group2 = {1,2,3,4,5}
    let group1: Vec<i32> = vec![0, -1];
    let group2: Vec<i32> = vec![1, 2, 3, 4, 5, -1];

    // Get C's branch weights for step=0 side=0
    unsafe {
        let c_weights = c_branch_weights(&topo);
        let effarr = &c_weights[0][0];

        // Create fake char** seq and aseq arrays (null pointers ok, we just use peff)
        let fake_seqs: Vec<*mut c_char> = (0..nseq).map(|_| ptr::null_mut()).collect();
        let mut aseq1: Vec<*mut c_char> = vec![ptr::null_mut(); nseq];
        let mut aseq2: Vec<*mut c_char> = vec![ptr::null_mut(); nseq];

        let mut peff1 = vec![0.0f64; nseq];
        let mut peff2 = vec![0.0f64; nseq];

        let eff_mut: *mut c_double = libc_alloc_zeroed(nseq * std::mem::size_of::<c_double>()) as _;
        for (i, &w) in effarr.iter().enumerate() {
            *eff_mut.add(i) = w;
        }

        let d1_buf: *mut c_char = libc_alloc_zeroed(200) as _;
        let d2_buf: *mut c_char = libc_alloc_zeroed(200) as _;

        let mut memlist1 = group1.clone();
        let mut memlist2 = group2.clone();

        let c_clus1 = mafft_sys::fastconjuction_noname(
            memlist1.as_mut_ptr(),
            fake_seqs.clone().as_mut_ptr(),
            aseq1.as_mut_ptr(),
            peff1.as_mut_ptr(),
            eff_mut,
            d1_buf,
            0.00001,
            ptr::null_mut(),
        );
        let c_clus2 = mafft_sys::fastconjuction_noname(
            memlist2.as_mut_ptr(),
            fake_seqs.clone().as_mut_ptr(),
            aseq2.as_mut_ptr(),
            peff2.as_mut_ptr(),
            eff_mut,
            d2_buf,
            0.00001,
            ptr::null_mut(),
        );

        let c_w1: Vec<f64> = (0..c_clus1 as usize).map(|i| peff1[i]).collect();
        let c_w2: Vec<f64> = (0..c_clus2 as usize).map(|i| peff2[i]).collect();

        // Now compute Rust's per-group normalization
        const MINIMUM_WEIGHT: f64 = 0.00001;
        let r_raw1: Vec<f64> = vec![0]
            .iter()
            .map(|&i: &usize| effarr[i].max(MINIMUM_WEIGHT))
            .collect();
        let r_raw2: Vec<f64> = (1..nseq).map(|i| effarr[i].max(MINIMUM_WEIGHT)).collect();
        let s1: f64 = r_raw1.iter().sum();
        let s2: f64 = r_raw2.iter().sum();
        let r_w1: Vec<f64> = r_raw1.iter().map(|w| w / s1).collect();
        let r_w2: Vec<f64> = r_raw2.iter().map(|w| w / s2).collect();

        eprintln!("C group1 weights: {:?}", c_w1);
        eprintln!("R group1 weights: {:?}", r_w1);
        eprintln!("C group2 weights: {:?}", c_w2);
        eprintln!("R group2 weights: {:?}", r_w2);

        for (i, (&c, &r)) in c_w1.iter().zip(r_w1.iter()).enumerate() {
            assert!((c - r).abs() < 1e-10, "group1[{i}]: c={c} r={r}");
        }
        for (i, (&c, &r)) in c_w2.iter().zip(r_w2.iter()).enumerate() {
            assert!((c - r).abs() < 1e-10, "group2[{i}]: c={c} r={r}");
        }
    }
}

#[test]
fn symmetric_4seq_weights_match_expected() {
    let mut topo = Topology::new(4);
    topo.steps.push(JoinStep {
        left: vec![0],
        right: vec![1],
        left_length: 0.1,
        right_length: 0.1,
    });
    topo.steps.push(JoinStep {
        left: vec![2],
        right: vec![3],
        left_length: 0.1,
        right_length: 0.1,
    });
    topo.steps.push(JoinStep {
        left: vec![0, 1],
        right: vec![2, 3],
        left_length: 0.2,
        right_length: 0.2,
    });

    let bw = BranchWeights::new(&topo);

    // Root split: all equal weights
    let w = bw.weights_for_branch(&topo, 2, 0);
    let first = w[0];
    for (i, &v) in w.iter().enumerate() {
        assert!(
            (v - first).abs() < 1e-10,
            "seq {i}: weight {v} != {first} in symmetric tree"
        );
    }

    // Leaf split: proper ordering
    let w = bw.weights_for_branch(&topo, 0, 0);
    assert!((w[0] - 1.0).abs() < 1e-10);
    assert!((w[2] - w[3]).abs() < 1e-10);
    assert!(w[1] > w[2]);
}

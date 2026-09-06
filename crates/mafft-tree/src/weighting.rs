/// Sequence weighting from tree topology.
///
/// Ports the C `counteff_simple_double()` from mltaln9.c (global weights)
/// and `weightFromABranch()` from treeOperation.c (per-branch weights).
use crate::topology::Topology;

/// Small constant added to all weights to prevent zero weights.
const GETA: f64 = 0.001;

/// Compute sequence weights from a guide tree topology.
///
/// Sequences that are more isolated in the tree (longer branches) get
/// higher weights, while closely related sequences get lower weights.
/// This improves alignment quality for datasets with uneven sampling.
///
/// Returns a weight vector of length `nseq`, normalized to sum to 1.0.
pub fn sequence_weights(topo: &Topology) -> Vec<f64> {
    let n = topo.nseq;
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }

    let mut rootnode = vec![0.0f64; n];
    let mut eff = vec![1.0f64; n];

    // Match C `mltaln9.c:9893,9902`: rootnode[s] += len * eff[s].
    // Apple clang -O3 on arm64 does NOT fuse this `+= a*b` to FMA by default;
    // earlier code here used `mul_add` (=explicit FMA) which produced 1-ULP
    // drift vs C for non-trivial trees (BB20027 — surfaced by cpmx diff at
    // call=23). Use plain mul+add to match C on this platform.
    for step in &topo.steps {
        for &s in &step.left {
            rootnode[s] += step.left_length * eff[s];
            eff[s] *= 0.5;
        }
        for &s in &step.right {
            rootnode[s] += step.right_length * eff[s];
            eff[s] *= 0.5;
        }
    }

    for w in &mut rootnode {
        *w += GETA;
    }
    let total: f64 = rootnode.iter().sum();
    if total > 0.0 {
        for w in &mut rootnode {
            *w /= total;
        }
    }
    if let Ok(f) = std::env::var("RS_WEIGHTS") {
        use std::io::Write;
        if let Ok(mut fp) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&f)
        {
            for (i, w) in rootnode.iter().enumerate() {
                let _ = writeln!(fp, "weight[{}]={:.18e}", i, w);
            }
            let _ = writeln!(fp, "---");
        }
    }
    rootnode
}

// ---------------------------------------------------------------------------
// Per-branch weighting (C's weightFromABranch / calcBranchWeight)
// ---------------------------------------------------------------------------

/// Unrooted tree node for per-branch weight computation.
struct WNode {
    children: [i32; 3],
    length: [f64; 3],
    branch_weight: [f64; 3],
    members: [Vec<i32>; 3], // -1 terminated
}

/// Pre-computed per-branch weight structure.
pub struct BranchWeights {
    nodes: Vec<WNode>,
    nseq: usize,
}

impl BranchWeights {
    /// Debug: get (children_indices, branch_weights) for a node.
    pub fn debug_node(&self, idx: usize) -> ([i32; 3], [f64; 3]) {
        (self.nodes[idx].children, self.nodes[idx].branch_weight)
    }

    /// Debug: get members[d] for a node.
    pub fn debug_members(&self, idx: usize, d: usize) -> Vec<i32> {
        self.nodes[idx].members[d].clone()
    }

    pub fn nseq(&self) -> usize {
        self.nseq
    }
}

/// Search for which topology step (in range start..end) contains `seq_idx`
/// in either its left or right member list. Returns (step, lor).
/// Ports C's `searchParent`.
fn search_parent(topo: &Topology, seq_idx: usize, start: usize, end: usize) -> (usize, usize) {
    for step_idx in start..end {
        if topo.steps[step_idx].left.contains(&seq_idx) {
            return (step_idx, 0);
        }
        if topo.steps[step_idx].right.contains(&seq_idx) {
            return (step_idx, 1);
        }
    }
    panic!("searchParent failed for seq {seq_idx} in range {start}..{end}");
}

/// Get the member list for a topology step's side, with -1 sentinel.
fn topo_members(topo: &Topology, step: usize, lor: usize) -> Vec<i32> {
    let members = if lor == 0 {
        &topo.steps[step].left
    } else {
        &topo.steps[step].right
    };
    let mut v: Vec<i32> = members.iter().map(|&s| s as i32).collect();
    v.push(-1);
    v
}

/// Compute the complement of a member list (sequences NOT in it).
fn negative_members(members: &[i32], nseq: usize) -> Vec<i32> {
    let mut present = vec![false; nseq];
    for &m in members {
        if m >= 0 && (m as usize) < nseq {
            present[m as usize] = true;
        }
    }
    let mut result: Vec<i32> = (0..nseq)
        .filter(|&i| !present[i])
        .map(|i| i as i32)
        .collect();
    result.push(-1);
    result
}

impl BranchWeights {
    /// Build the per-branch weight structure from a topology.
    /// Ports C's `treeCnv` + `calcBranchWeight` from treeOperation.c.
    pub fn new(topo: &Topology) -> Self {
        let nseq = topo.nseq;
        if nseq <= 2 {
            return Self {
                nodes: Vec::new(),
                nseq,
            };
        }

        let total = 2 * nseq;
        let mut nodes: Vec<WNode> = (0..total)
            .map(|_| WNode {
                children: [-1; 3],
                length: [0.0; 3],
                branch_weight: [1.0; 3],
                members: [Vec::new(), Vec::new(), Vec::new()],
            })
            .collect();
        let mut count = vec![0usize; total];

        // C's checkMinusLength (treeOperation.c:16): clamp lengths < MINLEN=0.001.
        // This prevents degenerate cases where identical sequences produce 0-length
        // branches, which would otherwise make calc_w return MAXBW (=1.0) and
        // propagate wrong weights through the whole tree.
        const MINLEN: f64 = 0.001;
        let clamped_lengths: Vec<(f64, f64)> = topo
            .steps
            .iter()
            .map(|s| (s.left_length.max(MINLEN), s.right_length.max(MINLEN)))
            .collect();

        // Phase 1: Connect leaf nodes (C lines 160-183).
        // For each sequence (leaf), find the first topology step that mentions it.
        for seq_idx in 0..nseq {
            let leaf = nseq + seq_idx;
            let (pstep, plor) = search_parent(topo, seq_idx, 0, nseq - 1);
            let branch_len = if plor == 0 {
                clamped_lengths[pstep].0
            } else {
                clamped_lengths[pstep].1
            };
            let members = topo_members(topo, pstep, plor);

            // Parent → leaf
            let cc = count[pstep];
            nodes[pstep].children[cc] = leaf as i32;
            nodes[pstep].length[cc] = branch_len;
            nodes[pstep].members[cc] = members.clone();
            count[pstep] += 1;

            // Leaf → parent
            let cc = count[leaf];
            nodes[leaf].children[cc] = pstep as i32;
            nodes[leaf].length[cc] = branch_len;
            nodes[leaf].members[cc] = members;
            count[leaf] += 1;
        }

        // Phase 2: Connect internal nodes (C lines 184-215).
        for i in 0..nseq.saturating_sub(2) {
            let rep = topo.steps[i].left[0].min(topo.steps[i].right[0]);
            let (pstep, plor) = search_parent(topo, rep, i + 1, nseq - 1);
            let branch_len = if plor == 0 {
                clamped_lengths[pstep].0
            } else {
                clamped_lengths[pstep].1
            };
            let members = topo_members(topo, pstep, plor);

            // Parent → internal node i
            let cc = count[pstep];
            nodes[pstep].children[cc] = i as i32;
            nodes[pstep].length[cc] = branch_len;
            nodes[pstep].members[cc] = members.clone();
            count[pstep] += 1;

            // Internal node i → parent (with complement members)
            let cc = count[i];
            nodes[i].children[cc] = pstep as i32;
            nodes[i].length[cc] = branch_len;
            nodes[i].members[cc] = negative_members(&members, nseq);
            count[i] += 1;
        }

        // Phase 3: Root restructuring (C lines 235-255).
        // Make unrooted by connecting nseq-3 to nseq-2's other child.
        if nseq >= 3 {
            let root = nseq - 2;
            let subroot = nseq - 3;

            // Find which child of root is NOT subroot.
            let sibling_dir = if nodes[root].children[0] == subroot as i32 {
                1
            } else if nodes[root].children[1] == subroot as i32 {
                0
            } else {
                // This can happen if the tree structure doesn't have root
                // directly connected to subroot. Skip restructuring.
                usize::MAX
            };

            if sibling_dir != usize::MAX {
                let sibling = nodes[root].children[sibling_dir] as usize;
                let combined = clamped_lengths[root].0 + clamped_lengths[root].1;

                // subroot slot 2 → sibling
                nodes[subroot].children[2] = sibling as i32;
                nodes[subroot].length[2] = combined;
                nodes[subroot].members[2] = nodes[root].members[sibling_dir].clone();

                // sibling slot 2 → subroot.
                // C `treeOperation.c:246-249` *sets* `stopol[tmpint].children[2]`
                // without clearing the original sibling→root link in slots [0,1].
                // The sibling thus has the root in its original slot AND subroot
                // in slot 2 simultaneously. `calcW`/`syntheticLength` skip the
                // "opposite" slot via `op != ob->children[i]`, so the redundant
                // root link is naturally ignored — but only when the caller
                // passes `op = subroot`, which it does post-restructure. An
                // earlier Rust port REPLACED the link in [0,1] instead of
                // adding at [2]; the slot index drove synthetic_len recursion
                // into the wrong subtree and produced ~0.1% drift in branch
                // weights (BB11002 refinement iter 2).
                nodes[sibling].children[2] = subroot as i32;
                nodes[sibling].length[2] = combined;
            }
        }

        // Phase 4: Pre-compute branch weights (calcBranchWeight).
        // C computes one weight per EDGE and stores it in bw[parent.step][parent.LorR].
        // Both endpoints reference the same weight via weightptr. We mirror this
        // by computing the weight once per edge and storing it on both endpoints.
        //
        // C skips edges where parent.step == nseq-2 (root), and sets root's
        // bw[nseq-2][0] and bw[nseq-2][1] separately at the end.
        let root = nseq - 2;

        // For leaves: compute weight for the edge from leaf to parent.
        for seq_idx in 0..nseq {
            let leaf = nseq + seq_idx;
            let (pstep, _) = search_parent(topo, seq_idx, 0, nseq - 1);
            if pstep == root {
                continue;
            } // skip root edges
            let w = calc_branch_weight(&nodes, pstep, leaf, nseq);
            // Store on both endpoints
            for d in 0..3 {
                if nodes[pstep].children[d] == leaf as i32 {
                    nodes[pstep].branch_weight[d] = w;
                }
                if nodes[leaf].children[d] == pstep as i32 {
                    nodes[leaf].branch_weight[d] = w;
                }
            }
        }

        // For internal nodes: compute weight for edge from internal to parent.
        for i in 0..nseq.saturating_sub(3) {
            let rep = topo.steps[i].left[0].min(topo.steps[i].right[0]);
            let (pstep, _plor) = search_parent(topo, rep, i + 1, nseq - 1);
            if pstep == root {
                continue;
            }
            let w = calc_branch_weight(&nodes, pstep, i, nseq);
            for d in 0..3 {
                if nodes[pstep].children[d] == i as i32 {
                    nodes[pstep].branch_weight[d] = w;
                }
                if nodes[i].children[d] == pstep as i32 {
                    nodes[i].branch_weight[d] = w;
                }
            }
        }

        // Root edge: bw[root][0] = calcW(sibling of subroot, subroot),
        // bw[root][1] = 1.0. After restructuring, the root edge connects
        // node subroot (slot 2) to its sibling. This is the weight for that edge.
        if nseq >= 3 {
            let subroot = nseq - 3;
            let sibling = nodes[subroot].children[2];
            if sibling >= 0 {
                let w = calc_branch_weight(&nodes, sibling as usize, subroot, nseq);
                // Set weight on subroot's slot 2
                nodes[subroot].branch_weight[2] = w;
                // Set weight on sibling's slot pointing to subroot
                for d in 0..3 {
                    if nodes[sibling as usize].children[d] == subroot as i32 {
                        nodes[sibling as usize].branch_weight[d] = w;
                    }
                }
            }
        }

        Self { nodes, nseq }
    }

    /// Compute per-sequence weights for a specific branch split.
    /// Ports C's `weightFromABranch`.
    pub fn weights_for_branch(&self, topo: &Topology, step: usize, side: usize) -> Vec<f64> {
        let nseq = self.nseq;
        if nseq <= 2 || self.nodes.is_empty() {
            return vec![1.0; nseq];
        }

        let mut result = vec![1.0f64; nseq];

        // C special case: step == nseq-2 (root step).
        // topNode = stopol[nseq-2].children[0], btmNode = stopol[nseq-3]
        if step == nseq - 2 {
            let top = self.nodes[nseq - 2].children[0];
            let btm = (nseq - 3) as i32;
            if top >= 0 && btm >= 0 {
                self.weight_rec(&mut result, btm as usize, top as usize);
                self.weight_rec(&mut result, top as usize, btm as usize);
            }
            return result;
        }

        // Find which child direction matches the requested side's members.
        let target = if side == 0 {
            &topo.steps[step].left
        } else {
            &topo.steps[step].right
        };
        let first_member = target[0] as i32;

        let mut btm_dir = 0;
        for d in 0..3 {
            if self.nodes[step].members[d].first().copied() == Some(first_member) {
                btm_dir = d;
                break;
            }
        }

        let btm = self.nodes[step].children[btm_dir];
        if btm < 0 {
            return result;
        }

        self.weight_rec(&mut result, btm as usize, step);
        self.weight_rec(&mut result, step, btm as usize);

        result
    }

    fn weight_rec(&self, result: &mut [f64], node: usize, from: usize) {
        if node >= self.nseq {
            return;
        } // leaf
        for d in 0..3 {
            let child = self.nodes[node].children[d];
            if child >= 0 && child as usize != from {
                let w = self.nodes[node].branch_weight[d];
                for &s in &self.nodes[node].members[d] {
                    if s >= 0 && (s as usize) < self.nseq {
                        result[s as usize] *= w;
                    }
                }
                self.weight_rec(result, child as usize, node);
            }
        }
    }

    /// Distance from each leaf to the branch `(step, side)` on the tree.
    /// Port of C `treeOperation.c::distFromABranch` (`USEDISTONTREE=1` path):
    /// sum of edge lengths from each leaf down to the given branch. Used by
    /// `--allowshift` refinement to classify sequence pairs into distance
    /// bins (`smalldistmtx[i][j] = distarr[i] + distarr[j]`,
    /// `tddis.c::OneClusterAndTheOther_fast`).
    pub fn dist_from_a_branch(&self, topo: &Topology, step: usize, side: usize) -> Vec<f64> {
        let nseq = self.nseq;
        if nseq == 2 {
            // C: result[0] = len[0][0], result[1] = len[0][1].
            let s = &topo.steps[0];
            return vec![s.left_length, s.right_length];
        }
        let mut result = vec![0.0f64; nseq];
        if nseq <= 2 || self.nodes.is_empty() {
            return result;
        }
        if step == nseq - 2 {
            let top = self.nodes[nseq - 2].children[0];
            let btm = (nseq - 3) as i32;
            if top >= 0 && btm >= 0 {
                self.dist_rec(&mut result, btm as usize, top as usize);
                self.dist_rec(&mut result, top as usize, btm as usize);
            }
            return result;
        }
        let target = if side == 0 {
            &topo.steps[step].left
        } else {
            &topo.steps[step].right
        };
        let first_member = target[0] as i32;
        let mut btm_dir = 0;
        for d in 0..3 {
            if self.nodes[step].members[d].first().copied() == Some(first_member) {
                btm_dir = d;
                break;
            }
        }
        let btm = self.nodes[step].children[btm_dir];
        if btm < 0 {
            return result;
        }
        self.dist_rec(&mut result, btm as usize, step);
        self.dist_rec(&mut result, step, btm as usize);
        result
    }

    fn dist_rec(&self, result: &mut [f64], node: usize, from: usize) {
        if node >= self.nseq {
            return;
        } // leaf
        for d in 0..3 {
            let child = self.nodes[node].children[d];
            if child >= 0 && child as usize != from {
                let len = self.nodes[node].length[d];
                for &s in &self.nodes[node].members[d] {
                    if s >= 0 && (s as usize) < self.nseq {
                        result[s as usize] += len;
                    }
                }
                self.dist_rec(result, child as usize, node);
            }
        }
    }
}

/// Compute branch weight = calcW(top) * calcW(btm).
fn calc_branch_weight(nodes: &[WNode], node: usize, child: usize, nseq: usize) -> f64 {
    calc_w(nodes, node, child, nseq) * calc_w(nodes, child, node, nseq)
}

/// C's calcW: compute weight for one side of a branch.
fn calc_w(nodes: &[WNode], ob: usize, op: usize, nseq: usize) -> f64 {
    if ob >= nseq {
        return 1.0;
    } // leaf

    let mut dir_ch = Vec::new();
    let mut dir_pa = 0;
    for d in 0..3 {
        if nodes[ob].children[d] == op as i32 {
            dir_pa = d;
        } else if nodes[ob].children[d] >= 0 {
            dir_ch.push(d);
        }
    }
    if dir_ch.len() < 2 {
        return 1.0;
    }

    let a = synthetic_len(nodes, nodes[ob].children[dir_ch[0]] as usize, ob, nseq);
    let b = synthetic_len(nodes, nodes[ob].children[dir_ch[1]] as usize, ob, nseq);
    let c = synthetic_len(nodes, nodes[ob].children[dir_pa] as usize, ob, nseq);

    if c == 0.0 {
        return 1.0;
    }
    if a == 0.0 || b == 0.0 {
        return 0.01;
    }

    // C `treeOperation.c:400` `s = b*c + c*a + a*b` — Apple clang at -O3 with
    // FP_CONTRACT=on lowers this to AArch64 instructions:
    //   fmul  d3, a, c                  ;  a*c (plain)
    //   fmadd d2, b, c, d3              ;  b*c + a*c
    //   fmadd d0, a, b, d2              ;  a*b + (b*c + a*c)
    // i.e. fma(a, b, fma(b, c, a*c)). Plain Rust `+` doesn't auto-FMA,
    // producing a 1-2 ULP drift in `s` that cascades into calcW's value and
    // (since calcW feeds branch_weight which multiplies through every leaf
    // path) into the per-cluster eff used for cpmx — surfaces as the
    // BB30018/BB40043/BB40010 1-column residue shifts.
    let s = a * b + (b * c + (a * c));
    if s == 0.0 {
        return 1.0;
    }

    let value = (a * b * (c + a) * (c + b) / (c * (a + b) * s)).sqrt();

    if let Ok(f) = std::env::var("RS_CALCW") {
        use std::io::Write;
        if let Ok(mut fp) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&f)
        {
            let _ = writeln!(
                fp,
                "R_CALCW ob={} op={} a={:.17e} b={:.17e} c={:.17e} s={:.17e} value={:.17e}",
                ob, op, a, b, c, s, value
            );
        }
    }

    value
}

/// C's syntheticLength: harmonic mean of children's lengths + branch to parent.
fn synthetic_len(nodes: &[WNode], ob: usize, op: usize, nseq: usize) -> f64 {
    // C `treeOperation.c:308-313` returns `ob->length[0]` for leaves —
    // the *original* parent edge length, NOT the edge to `op`. For true
    // leaves these coincide (one neighbour). But the root-restructured
    // sibling (`ob >= nseq`, with children=[old_root, -1, subroot]) has
    // length[0]=old-to-root and length[2]=combined-to-subroot; C still
    // returns length[0] because `isLeaf` only checks children[1]. An
    // earlier port returned the (correct) `len_to_op` for the
    // post-restructure case and produced 0.1% branch-weight drift on
    // refinement (BB11002 --maxiterate 100, iter 2). Mirror C exactly.
    if ob >= nseq {
        return nodes[ob].length[0];
    }
    let len_to_op = (0..3)
        .find(|&d| nodes[ob].children[d] == op as i32)
        .map(|d| nodes[ob].length[d])
        .unwrap_or(0.0);

    let mut child_lens = Vec::new();
    for d in 0..3 {
        let child = nodes[ob].children[d];
        if child >= 0 && child as usize != op {
            child_lens.push(synthetic_len(nodes, child as usize, ob, nseq));
        }
    }
    if child_lens.len() < 2 {
        return len_to_op;
    }

    let (a, b) = (child_lens[0], child_lens[1]);
    let hm = if a == 0.0 || b == 0.0 {
        0.0
    } else {
        1.0 / (1.0 / a + 1.0 / b)
    };
    hm + len_to_op
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::JoinStep;

    #[test]
    fn weights_sum_to_one() {
        let mut topo = Topology::new(3);
        topo.steps.push(JoinStep {
            left: vec![0],
            right: vec![1],
            left_length: 0.1,
            right_length: 0.2,
        });
        topo.steps.push(JoinStep {
            left: vec![0, 1],
            right: vec![2],
            left_length: 0.3,
            right_length: 0.5,
        });
        let w = sequence_weights(&topo);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "weights sum to {sum}");
    }

    #[test]
    fn isolated_sequence_gets_higher_weight() {
        let mut topo = Topology::new(3);
        topo.steps.push(JoinStep {
            left: vec![0],
            right: vec![1],
            left_length: 0.05,
            right_length: 0.05,
        });
        topo.steps.push(JoinStep {
            left: vec![0, 1],
            right: vec![2],
            left_length: 0.1,
            right_length: 0.9,
        });
        let w = sequence_weights(&topo);
        assert!(w[2] > w[0]);
        assert!(w[2] > w[1]);
    }

    #[test]
    fn single_sequence() {
        let w = sequence_weights(&Topology::new(1));
        assert_eq!(w, vec![1.0]);
    }

    #[test]
    fn branch_weights_tree_structure() {
        // 4 sequences: ((0,1),(2,3))
        // Step 0: {0} + {1}, len 0.1, 0.1
        // Step 1: {2} + {3}, len 0.1, 0.1
        // Step 2: {0,1} + {2,3}, len 0.2, 0.2
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

        // After treeCnv, the unrooted tree should be:
        //   leaf4(seq0) -- node0 -- leaf5(seq1)
        //                   |
        //                   | (combined length 0.2+0.2=0.4)
        //                   |
        //   leaf6(seq2) -- node1 -- leaf7(seq3)
        //
        // Node 0: children = [leaf4, leaf5, node1], lengths = [0.1, 0.1, 0.4]
        // Node 1: children = [leaf6, leaf7, node0], lengths = [0.1, 0.1, 0.4]
        // (node 2 = root, bypassed by restructuring)

        // Verify node 0 structure
        let n0 = &bw.nodes[0];
        assert_eq!(n0.children[0], 4, "node0.child0 = leaf4 (seq0)");
        assert_eq!(n0.children[1], 5, "node0.child1 = leaf5 (seq1)");
        assert_eq!(
            n0.children[2], 1,
            "node0.child2 = node1 (after root restructure)"
        );
        assert!((n0.length[0] - 0.1).abs() < 1e-10);
        assert!((n0.length[1] - 0.1).abs() < 1e-10);
        assert!(
            (n0.length[2] - 0.4).abs() < 1e-10,
            "combined len: {}",
            n0.length[2]
        );

        // Verify node 1 structure
        let n1 = &bw.nodes[1];
        assert_eq!(n1.children[0], 6, "node1.child0 = leaf6 (seq2)");
        assert_eq!(n1.children[1], 7, "node1.child1 = leaf7 (seq3)");
        assert_eq!(
            n1.children[2], 0,
            "node1.child2 = node0 (after root restructure)"
        );
        assert!((n1.length[2] - 0.4).abs() < 1e-10);

        // For the symmetric ((0,1),(2,3)) tree with equal lengths,
        // all per-branch weights should give equal weight to the
        // sequences on each side of the split.
        // Step 0, side 0: split {0} vs {1,2,3}
        let w = bw.weights_for_branch(&topo, 0, 0);
        eprintln!("step=0 side=0 w={:?}", w);
        assert!(w.iter().all(|&v| v > 0.0 && v.is_finite()), "{:?}", w);
        // In a symmetric tree, seqs 2 and 3 should have the same weight
        assert!(
            (w[2] - w[3]).abs() < 1e-10,
            "seqs 2,3 should be symmetric: {} {}",
            w[2],
            w[3]
        );

        // Step 2, side 0: split {0,1} vs {2,3} — the root split
        let w = bw.weights_for_branch(&topo, 2, 0);
        eprintln!("step=2 side=0 w={:?}", w);
        assert!(w.iter().all(|&v| v > 0.0 && v.is_finite()), "{:?}", w);
        // Symmetric: w[0]==w[1], w[2]==w[3]
        assert!((w[0] - w[1]).abs() < 1e-10, "seqs 0,1 should be symmetric");
        assert!((w[2] - w[3]).abs() < 1e-10, "seqs 2,3 should be symmetric");
    }
}

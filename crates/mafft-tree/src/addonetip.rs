/// Add a new sequence to an existing guide tree.
///
/// Ports the C `addonetip()` from `mltaln9.c`.
///
/// Given an existing topology for `norg` sequences and a distance vector
/// from a new sequence to each original sequence, produces a new topology
/// for `norg + 1` sequences with the new sequence inserted at the
/// phylogenetically appropriate position.
use crate::topology::{JoinStep, Topology};

/// Result of placing a new sequence in the tree.
pub struct AddResult {
    /// The new topology with the added sequence.
    pub topology: Topology,
    /// Index of the nearest original sequence.
    pub nearest: usize,
    /// Distance to nearest sequence.
    pub distance: f64,
}

/// Add a single sequence to an existing tree topology.
///
/// `topology` is the guide tree for the original `norg` sequences.
/// `distances_to_new` has length `norg`: `distances_to_new[i]` is the
/// distance from original sequence `i` to the new sequence.
/// The new sequence gets index `norg` in the output topology.
///
/// `sueff` is the clustering weight (default 0.1, matching C's SUEFF).
pub fn addonetip(topology: &Topology, distances_to_new: &[f64], sueff: f64) -> AddResult {
    let norg = topology.nseq;
    let nstep = if norg >= 2 { norg - 1 } else { 0 };
    let new_idx = norg;

    assert_eq!(distances_to_new.len(), norg);

    // Working copy of distances from each original seq to the new seq.
    // Uses same indexing as C: iscorec[i][norg-i] → dist_to_new[i].
    let mut dist_to_new: Vec<f64> = distances_to_new.to_vec();

    // Active cluster tracking (Bchain equivalent)
    let mut active: Vec<bool> = vec![true; norg];

    // Find initial nearest sequence
    let mut minscore = f64::MAX;
    let mut nearest = 0usize;
    for i in 0..norg {
        if dist_to_new[i] < minscore {
            minscore = dist_to_new[i];
            nearest = i;
        }
    }
    let _nearesto = nearest;
    let minscoreo = minscore;

    // Compute distfromtip for each step in the original topology.
    // distfromtip[i] = node height at step i (distance from leaves to this node).
    let distfromtip = compute_distfromtip(topology);

    // leaf2node[i] = step index where leaf i was last incorporated (-1 if never)
    let mut leaf2node: Vec<Option<usize>> = vec![None; norg];

    let sueff1 = 1.0 - sueff;
    let sueff05 = sueff * 0.5;

    // Build new topology with norg steps (for norg+1 sequences)
    let mut new_steps: Vec<JoinStep> = Vec::with_capacity(norg);
    let mut repnorg: Option<usize> = None; // None until insertion
    let mut addedlen: f64 = 0.0;

    for i in 0..nstep {
        let step = &topology.steps[i];
        let mem0 = step.left[0];
        let mem1 = step.right[0];

        // Check insertion condition: dep[i].distfromtip * 2 > minscore
        // AND the new seq hasn't been inserted yet
        if repnorg.is_none() && distfromtip[i] * 2.0 > minscore {
            let nearestnode = leaf2node[nearest];

            // Build the insertion step
            let left_group = if let Some(node_idx) = nearestnode {
                // nearest is part of an internal node — use that node's full clade
                let node_step = &topology.steps[node_idx];
                let mut group = node_step.left.clone();
                group.extend_from_slice(&node_step.right);
                group
            } else {
                // nearest is still a leaf
                vec![nearest]
            };

            let left_len = if nearestnode.is_some() {
                // addedlen = dep[i].distfromtip - minscore / 2
                distfromtip[i] - minscore / 2.0
            } else {
                minscore / 2.0
            };

            addedlen = left_len;

            new_steps.push(JoinStep {
                left: left_group,
                right: vec![new_idx],
                left_length: left_len,
                right_length: minscore / 2.0,
            });

            repnorg = Some(nearest);
        }

        // Update distances: merge mem0 and mem1
        let eff0 = dist_to_new[mem0];
        let eff1 = dist_to_new[mem1];
        dist_to_new[mem0] = eff0.min(eff1) * sueff1 + (eff0 + eff1) * sueff05;
        dist_to_new[mem1] = 9999.9;

        // Remove mem1 from active set
        active[mem1] = false;

        // If nearest was merged, recompute
        if nearest == mem1 || nearest == mem0 {
            minscore = f64::MAX;
            for j in 0..norg {
                if active[j] && dist_to_new[j] < minscore {
                    minscore = dist_to_new[j];
                    nearest = j;
                }
            }
        }

        // Copy original step to new topology, possibly appending new_idx
        let (left, left_len) = if step.left[0] == repnorg.unwrap_or(usize::MAX) {
            let mut group = step.left.clone();
            group.push(new_idx);
            let len = step.left_length - addedlen;
            addedlen = 0.0;
            (group, len)
        } else {
            (step.left.clone(), step.left_length)
        };

        let (right, right_len) = if step.right[0] == repnorg.unwrap_or(usize::MAX) {
            let mut group = step.right.clone();
            group.push(new_idx);
            let len = step.right_length - addedlen;
            addedlen = 0.0;
            // Update repnorg to track through the tree (C: repnorg = topolc[posinnew][0][0])
            repnorg = Some(left[0]);
            (group, len)
        } else {
            (step.right.clone(), step.right_length)
        };

        new_steps.push(JoinStep {
            left,
            right,
            left_length: left_len,
            right_length: right_len,
        });

        // Update leaf2node for all members in this step
        for &m in &step.left {
            leaf2node[m] = Some(i);
        }
        for &m in &step.right {
            leaf2node[m] = Some(i);
        }
    }

    // If never inserted (new seq is the most distant), insert at root
    if repnorg.is_none() {
        let left_group = if nstep > 0 {
            // Combine last step's left and right
            let last = &topology.steps[nstep - 1];
            let mut group = last.left.clone();
            group.extend_from_slice(&last.right);
            group
        } else if norg == 1 {
            vec![0]
        } else {
            (0..norg).collect()
        };

        let left_len = if nstep > 0 {
            minscore / 2.0 - distfromtip[nstep - 1]
        } else {
            minscore / 2.0
        };

        new_steps.push(JoinStep {
            left: left_group,
            right: vec![new_idx],
            left_length: left_len,
            right_length: minscore / 2.0,
        });
    }

    let mut result_topo = Topology::new(norg + 1);
    result_topo.steps = new_steps;

    AddResult {
        topology: result_topo,
        nearest: _nearesto,
        distance: minscoreo,
    }
}

/// Compute the node height (distfromtip) for each step in a topology.
///
/// For UPGMA trees, the height at step i is determined by:
///   height = parent_height_of_left_child + left_branch_length
///
/// For leaves, height = 0. Each step's height = max(left_child_height, right_child_height) + branch_length.
/// Port of C `mltaln9.c::generatesubalignmentstable` (lines 15330-15407).
///
/// Walks the guide tree and identifies "sub-alignment" clusters whose
/// internal merges are ALL at heights ≤ `threshold` (the
/// `--skipiterate F` value). For each cluster, records the
/// representative-side member list that crossed the threshold.
///
/// Returns `(sub_alignments, all_below_threshold)` where:
/// - `sub_alignments` is the per-cluster list of sequence indices.
///   Each cluster covers one subtree whose own height has just
///   exceeded `threshold` (the merge step where `distfromtip[rep]`
///   crosses F is recorded; the cluster's members are
///   `step.left` or `step.right` accordingly).
/// - `all_below_threshold = true` mirrors C's "return 1" branch
///   (`distfromtip[0] <= threshold`) — the whole tree is below the
///   threshold, so the caller should skip refinement entirely
///   (matches C `dvtditr.c:817-833`'s WARNING + early return).
///   Caller can short-circuit on this.
///
/// Mirrors C exactly: `distfromtip[]` is tracked per representative
/// (`topol[step][side][0]`); at each step both sides advance their
/// representative's height by `len[step][side]`; a cluster is
/// recorded iff `topol[step][side]` has more than one member
/// (`topol[i][side][1] != -1`) AND `distfromtip_before <= threshold
/// < distfromtip_after`.
pub fn generate_subalignments_table(
    topology: &Topology,
    threshold: f64,
) -> (Vec<Vec<usize>>, bool) {
    let nseq = topology.nseq;
    let mut distfromtip: Vec<f64> = vec![0.0; nseq];
    let mut sub_alignments: Vec<Vec<usize>> = Vec::new();

    for step in &topology.steps {
        let rep0 = step.left[0];
        let dist0_before = distfromtip[rep0];
        distfromtip[rep0] = dist0_before + step.left_length;
        let dist0_after = distfromtip[rep0];

        let rep1 = step.right[0];
        let dist1_before = distfromtip[rep1];
        distfromtip[rep1] = dist1_before + step.right_length;
        let dist1_after = distfromtip[rep1];

        // C `topol[i][0][1] != -1`: side has more than one member.
        // In rust JoinStep, left/right are full accumulated lists,
        // so this is `step.left.len() > 1`.
        if step.left.len() > 1 && dist0_before <= threshold && threshold < dist0_after {
            sub_alignments.push(step.left.clone());
        }
        if step.right.len() > 1 && dist1_before <= threshold && threshold < dist1_after {
            sub_alignments.push(step.right.clone());
        }
    }

    // C `mltaln9.c:15399`: `if (distfromtip[0] <= threshold) return 1;`
    // i.e., even the root sequence didn't cross the threshold → the
    // whole tree is below F → skip refinement entirely.
    let all_below = distfromtip.first().copied().unwrap_or(0.0) <= threshold;

    (sub_alignments, all_below)
}

pub fn compute_distfromtip(topology: &Topology) -> Vec<f64> {
    let nseq = topology.nseq;
    let mut heights: Vec<f64> = vec![0.0; nseq]; // height of each sequence/cluster representative
    let mut step_heights: Vec<f64> = Vec::with_capacity(topology.steps.len());

    for step in &topology.steps {
        let left_rep = step.left[0];
        let right_rep = step.right[0];
        let node_height = heights[left_rep] + step.left_length;
        // For ultrametric trees, both sides should give the same height,
        // but use left as canonical (matching C behavior).
        step_heights.push(node_height);

        // Update representatives' heights
        heights[left_rep] = node_height;
        heights[right_rep] = node_height;
    }

    step_heights
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::DistanceMatrix;
    use crate::musclesupg::{ClusterMethod, musclesupg};

    #[test]
    fn add_to_two_seq_tree() {
        let mut dm = DistanceMatrix::new(2);
        dm.set(0, 1, 0.2);
        let topo = musclesupg(&dm, ClusterMethod::default());

        // New seq equally distant from both
        let result = addonetip(&topo, &[0.5, 0.5], 0.1);
        assert_eq!(result.topology.nseq, 3);
        assert_eq!(result.topology.steps.len(), 2);

        // All sequences should appear
        let last = result.topology.steps.last().unwrap();
        let mut all: Vec<usize> = last.left.iter().chain(last.right.iter()).copied().collect();
        all.sort();
        assert_eq!(all, vec![0, 1, 2]);
    }

    #[test]
    fn add_close_sequence() {
        let mut dm = DistanceMatrix::new(3);
        dm.set(0, 1, 0.1);
        dm.set(0, 2, 0.5);
        dm.set(1, 2, 0.5);
        let topo = musclesupg(&dm, ClusterMethod::default());

        // New seq very close to seq 0
        let result = addonetip(&topo, &[0.02, 0.12, 0.52], 0.1);
        assert_eq!(result.topology.nseq, 4);
        assert_eq!(result.topology.steps.len(), 3);
        assert_eq!(result.nearest, 0);

        // Verify all seqs present in the last step
        let last = result.topology.steps.last().unwrap();
        let mut all: Vec<usize> = last.left.iter().chain(last.right.iter()).copied().collect();
        all.sort();
        assert_eq!(all, vec![0, 1, 2, 3]);
    }

    #[test]
    fn add_distant_sequence() {
        let mut dm = DistanceMatrix::new(3);
        dm.set(0, 1, 0.1);
        dm.set(0, 2, 0.2);
        dm.set(1, 2, 0.15);
        let topo = musclesupg(&dm, ClusterMethod::default());

        // New seq is very distant → should be added at root
        let result = addonetip(&topo, &[5.0, 5.0, 5.0], 0.1);
        assert_eq!(result.topology.nseq, 4);
        // New seq (idx 3) should be in the last step's right branch
        let last = result.topology.steps.last().unwrap();
        assert!(last.right.contains(&3) || last.left.contains(&3));
    }

    #[test]
    fn add_to_single_sequence() {
        let topo = Topology::new(1);
        let result = addonetip(&topo, &[0.3], 0.1);
        assert_eq!(result.topology.nseq, 2);
        assert_eq!(result.topology.steps.len(), 1);
        let step = &result.topology.steps[0];
        let mut all: Vec<usize> = step.left.iter().chain(step.right.iter()).copied().collect();
        all.sort();
        assert_eq!(all, vec![0, 1]);
    }

    #[test]
    fn distfromtip_simple() {
        let mut topo = Topology::new(3);
        topo.steps.push(JoinStep {
            left: vec![0],
            right: vec![1],
            left_length: 0.1,
            right_length: 0.1,
        });
        topo.steps.push(JoinStep {
            left: vec![0, 1],
            right: vec![2],
            left_length: 0.15,
            right_length: 0.25,
        });
        let heights = compute_distfromtip(&topo);
        assert!((heights[0] - 0.1).abs() < 1e-10); // step 0: 0 + 0.1
        assert!((heights[1] - 0.25).abs() < 1e-10); // step 1: 0.1 + 0.15
    }
}

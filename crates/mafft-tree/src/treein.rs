//! Load a user-supplied guide tree (`--treein`), mirroring C's
//! `mltaln9.c::loadtree` and `mltaln9.c::loadtreeoneline`.
//!
//! The expected file format matches C MAFFT exactly: one merge step per line,
//! four whitespace-separated fields — `im jm len0 len1` where `im` and `jm`
//! are 1-indexed sequence numbers (or internal-node IDs encoded as the
//! `min(member_indices)` of the cluster), and `len0`/`len1` are branch
//! lengths. There are `nseq - 1` lines. This is the format produced by
//! `mafft-upstream/core/newick2mafft.rb`. C requires `im < jm` per line.
//!
//! Internal-node encoding: when two clusters {A,B,...} and {C,D,...} are
//! merged at step `k`, the resulting cluster is given the ID
//! `min(A,B,...,C,D,...) + 1` (1-indexed) which is then used as the `im`
//! or `jm` of a later step. This is C's convention from `loadtree`.

use crate::topology::{JoinStep, Topology};
use std::path::Path;

/// Parse a MAFFT-format guide tree file into a `Topology`.
///
/// The file has `nseq - 1` lines, each with `im jm len0 len1` (1-indexed
/// sequence numbers and branch lengths). Internal nodes are encoded as the
/// minimum-index member of the cluster (1-indexed).
///
/// Mirrors C `mltaln9.c::loadtree` (lines 2492-) which calls `loadtreeoneline`
/// per merge step and builds the `topol` array by tracking cluster
/// memberships in a `Bchain` linked list.
pub fn parse_mafft_tree(path: impl AsRef<Path>, nseq: usize) -> Result<Topology, String> {
    let content = std::fs::read_to_string(path.as_ref())
        .map_err(|e| format!("cannot open guide tree file: {e}"))?;
    parse_mafft_tree_str(&content, nseq)
}

/// Like `parse_mafft_tree` but reads from a string (for testing).
pub fn parse_mafft_tree_str(content: &str, nseq: usize) -> Result<Topology, String> {
    let mut topo = Topology::new(nseq);

    // C tracks cluster membership in `Bchain` linked list indexed by the
    // 0-based "current representative" of each cluster. When clusters i and j
    // merge (with i < j), the merged cluster keeps representative i and j is
    // removed from the chain. The merged cluster's member list is built by
    // appending j's members to i's. (`mltaln9.c::2622-2738`.)
    let mut members: Vec<Vec<usize>> = (0..nseq).map(|i| vec![i]).collect();
    let mut alive: Vec<bool> = vec![true; nseq];

    let mut step = 0usize;
    for (line_no, raw) in content.lines().enumerate() {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        if step >= nseq - 1 {
            return Err(format!(
                "guide tree has too many merge steps (expected {})",
                nseq - 1
            ));
        }

        let mut parts = trimmed.split_ascii_whitespace();
        let im_s = parts
            .next()
            .ok_or_else(|| format!("line {}: missing im", line_no + 1))?;
        let jm_s = parts
            .next()
            .ok_or_else(|| format!("line {}: missing jm", line_no + 1))?;
        let l0_s = parts
            .next()
            .ok_or_else(|| format!("line {}: missing len0", line_no + 1))?;
        let l1_s = parts
            .next()
            .ok_or_else(|| format!("line {}: missing len1", line_no + 1))?;

        let im_1: usize = im_s
            .parse()
            .map_err(|_| format!("line {}: invalid im '{im_s}'", line_no + 1))?;
        let jm_1: usize = jm_s
            .parse()
            .map_err(|_| format!("line {}: invalid jm '{jm_s}'", line_no + 1))?;
        let len0: f64 = l0_s
            .parse()
            .map_err(|_| format!("line {}: invalid len0 '{l0_s}'", line_no + 1))?;
        let len1: f64 = l1_s
            .parse()
            .map_err(|_| format!("line {}: invalid len1 '{l1_s}'", line_no + 1))?;

        if im_1 == 0 || jm_1 == 0 || im_1 > nseq || jm_1 > nseq {
            return Err(format!(
                "line {}: im={} or jm={} out of range [1, {nseq}]",
                line_no + 1,
                im_1,
                jm_1
            ));
        }
        if im_1 >= jm_1 {
            // C's `loadtreeoneline` enforces `ar[0] < ar[1]` and exits
            // otherwise (`mltaln9.c:1508`).
            return Err(format!(
                "line {}: im ({im_1}) must be strictly less than jm ({jm_1}) — use newick2mafft.rb to convert",
                line_no + 1
            ));
        }

        let i0 = im_1 - 1;
        let j0 = jm_1 - 1;

        if !alive[i0] || !alive[j0] {
            return Err(format!(
                "line {}: cluster {} or {} already merged (tree is not bifurcated/rooted?)",
                line_no + 1,
                im_1,
                jm_1
            ));
        }

        let left = members[i0].clone();
        let right_taken = std::mem::take(&mut members[j0]);

        topo.steps.push(JoinStep {
            left,
            right: right_taken.clone(),
            left_length: len0,
            right_length: len1,
        });

        // Merge: append j's members to i. C's `loadtree` chooses the lower
        // index as the representative (line 2622-onwards constructs the
        // `topol[k][0]` / `topol[k][1]` arrays so that `topol[k][0]` starts
        // with the smaller-min cluster).
        members[i0].extend(right_taken);
        alive[j0] = false;

        step += 1;
    }

    if step != nseq - 1 {
        return Err(format!(
            "guide tree has {step} merge steps, expected {}",
            nseq - 1
        ));
    }

    Ok(topo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_seqs_simple() {
        // Tree: ((1, 2):0.1:0.1, 3):0.3:0.3
        // Step 1: merge {1} {2} with len0=0.1 len1=0.1 → cluster {1,2} with rep 1
        // Step 2: merge {1,2} {3} with len0=0.3 len1=0.3
        let content = "    1     2    0.10000    0.10000\n    1     3    0.30000    0.30000\n";
        let topo = parse_mafft_tree_str(content, 3).unwrap();
        assert_eq!(topo.num_steps(), 2);
        assert_eq!(topo.steps[0].left, vec![0]);
        assert_eq!(topo.steps[0].right, vec![1]);
        assert!((topo.steps[0].left_length - 0.1).abs() < 1e-9);
        assert_eq!(topo.steps[1].left, vec![0, 1]);
        assert_eq!(topo.steps[1].right, vec![2]);
        assert!(topo.is_complete());
    }

    #[test]
    fn five_seqs_balanced() {
        // ((1,2),(3,4),5) → in MAFFT bifurcated form:
        // (((1,2),(3,4)),5)
        // Steps: (1,2), (3,4), (1,3) [cluster 1+2 merging with cluster 3+4
        // is represented as im=1, jm=3 because reps are min indices],
        // (1,5).
        let content = "\
1 2 0.1 0.1
3 4 0.2 0.2
1 3 0.3 0.3
1 5 0.4 0.4
";
        let topo = parse_mafft_tree_str(content, 5).unwrap();
        assert_eq!(topo.num_steps(), 4);
        // Final step's union should be the DFS order.
        assert_eq!(topo.dfs_order().len(), 5);
        let mut sorted = topo.dfs_order();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn rejects_swapped_indices() {
        let content = "2 1 0.1 0.1\n1 3 0.3 0.3\n";
        let err = parse_mafft_tree_str(content, 3).unwrap_err();
        assert!(err.contains("must be strictly less"));
    }

    #[test]
    fn rejects_already_merged() {
        // Step 1: merge 1+2 (cluster keeps rep 1).
        // Step 2: try to merge 2 again — error.
        let content = "1 2 0.1 0.1\n2 3 0.3 0.3\n";
        let err = parse_mafft_tree_str(content, 3).unwrap_err();
        assert!(err.contains("already merged"));
    }

    #[test]
    fn rejects_too_few_steps() {
        let content = "1 2 0.1 0.1\n";
        let err = parse_mafft_tree_str(content, 3).unwrap_err();
        assert!(err.contains("merge steps"));
    }

    #[test]
    fn skips_blank_lines() {
        let content = "\n\n1 2 0.1 0.1\n\n1 3 0.3 0.3\n";
        let topo = parse_mafft_tree_str(content, 3).unwrap();
        assert_eq!(topo.num_steps(), 2);
    }
}

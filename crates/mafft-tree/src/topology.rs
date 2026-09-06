/// Tree topology representation.
///
/// Replaces the C `int ***topol` + `double **len` + `Treedep` structures
/// with a flat list of join steps.

/// A single join step in the tree construction.
///
/// Each step merges two clusters into one. The clusters are represented
/// as lists of sequence indices.
#[derive(Debug, Clone)]
pub struct JoinStep {
    /// Sequence indices in the first (left) cluster.
    pub left: Vec<usize>,
    /// Sequence indices in the second (right) cluster.
    pub right: Vec<usize>,
    /// Branch length to the left cluster.
    pub left_length: f64,
    /// Branch length to the right cluster.
    pub right_length: f64,
}

/// A guide tree represented as a sequence of join steps.
///
/// For `n` sequences, there are `n-1` join steps. Step `i` can reference
/// sequences from any earlier step's clusters.
///
/// This replaces the C `topol[step][0/1][]` + `len[step][0/1]` arrays.
#[derive(Debug, Clone)]
pub struct Topology {
    pub steps: Vec<JoinStep>,
    pub nseq: usize,
}

impl Topology {
    pub fn new(nseq: usize) -> Self {
        Self {
            steps: Vec::with_capacity(nseq.saturating_sub(1)),
            nseq,
        }
    }

    /// Build a comb-tree topology — sequence 0 joins 1, then that
    /// pair joins 2, then 3, etc. Mirrors C MAFFT's `--pileup`
    /// (`mltaln9.c::createchain` with `shuffle=0`) which writes
    /// "pileup" to the `_guidetree` file and `disttbfast` consumes
    /// it via `createchain(..., shuffle=0, ...)`.
    ///
    /// Branch lengths follow C exactly: every "chain side" branch
    /// length is `l = 2.0 / nseq`; the new-singleton branch grows
    /// linearly with the step index (`ll = l, 2l, 3l, ...`).
    /// Empty for `nseq < 2`.
    pub fn pileup_chain(nseq: usize) -> Self {
        let mut topo = Self::new(nseq);
        if nseq < 2 {
            return topo;
        }
        let l = 2.0 / nseq as f64;
        // Step 0: join sequences 0 and 1 with equal lengths.
        topo.steps.push(JoinStep {
            left: vec![0],
            right: vec![1],
            left_length: l,
            right_length: l,
        });
        let mut ll = 2.0 * l;
        // Each subsequent step adds the next input sequence to the
        // accumulated chain. `mm = 0` always in the no-shuffle case
        // since the accumulated cluster's representative stays at
        // index 0 (mm = min(mm, jm) and 0 <= every new jm), so the
        // accumulated side is always `left` and the new singleton
        // is always `right`. `len[i][0] = l` (chain branch),
        // `len[i][1] = ll` (new-singleton full depth).
        for i in 1..(nseq - 1) {
            let accumulated: Vec<usize> = (0..=i).collect();
            topo.steps.push(JoinStep {
                left: accumulated,
                right: vec![i + 1],
                left_length: l,
                right_length: ll,
            });
            ll += l;
        }
        topo
    }

    /// Number of join steps (= nseq - 1 for a complete tree).
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Check if the topology is complete (all sequences joined).
    pub fn is_complete(&self) -> bool {
        self.steps.len() + 1 == self.nseq
    }

    /// Sequence indices in tree-DFS order, mirroring C's `topolorderz`
    /// (`mltaln9.c:1928`). Each merge step's accumulated `left` / `right`
    /// vectors are already in subtree DFS order, so the root step's
    /// `left ++ right` is the full leaf order. For `nseq <= 1`, returns
    /// `[0]` / `[]` accordingly.
    pub fn dfs_order(&self) -> Vec<usize> {
        if self.nseq <= 1 {
            return (0..self.nseq).collect();
        }
        let last = self.steps.last().expect("complete topology");
        let mut order = Vec::with_capacity(self.nseq);
        order.extend_from_slice(&last.left);
        order.extend_from_slice(&last.right);
        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_topology() {
        let t = Topology::new(5);
        assert_eq!(t.num_steps(), 0);
        assert!(!t.is_complete());
    }

    #[test]
    fn complete_topology() {
        let mut t = Topology::new(3);
        t.steps.push(JoinStep {
            left: vec![0],
            right: vec![1],
            left_length: 0.1,
            right_length: 0.2,
        });
        t.steps.push(JoinStep {
            left: vec![0, 1],
            right: vec![2],
            left_length: 0.3,
            right_length: 0.4,
        });
        assert!(t.is_complete());
    }

    #[test]
    fn dfs_order_simple() {
        let mut t = Topology::new(4);
        // Tree: ((0,2), (1,3)) — left ++ right gives DFS order.
        t.steps.push(JoinStep {
            left: vec![0],
            right: vec![2],
            left_length: 0.0,
            right_length: 0.0,
        });
        t.steps.push(JoinStep {
            left: vec![1],
            right: vec![3],
            left_length: 0.0,
            right_length: 0.0,
        });
        t.steps.push(JoinStep {
            left: vec![0, 2],
            right: vec![1, 3],
            left_length: 0.0,
            right_length: 0.0,
        });
        assert_eq!(t.dfs_order(), vec![0, 2, 1, 3]);
    }
}

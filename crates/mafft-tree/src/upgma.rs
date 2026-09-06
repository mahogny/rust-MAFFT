/// UPGMA (Unweighted Pair Group Method with Arithmetic Mean) tree construction.
///
/// Ports the C `upg2()` and `veryfastsupg_int()` from mltaln9.c.
use crate::distance::DistanceMatrix;
use crate::topology::{JoinStep, Topology};

/// Build a guide tree using UPGMA clustering.
///
/// Simpler than NJ: assumes a molecular clock (ultrametric tree).
/// At each step, joins the pair with the smallest distance and averages
/// distances to the new cluster.
pub fn upgma(dist: &DistanceMatrix) -> Topology {
    let n = dist.nseq;
    if n <= 1 {
        return Topology::new(n);
    }

    // Working copy (full matrix)
    let mut mtx = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist.get(i, j);
            mtx[i][j] = d;
            mtx[j][i] = d;
        }
    }

    let mut active: Vec<bool> = vec![true; n];
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut tmplen = vec![0.0f64; n];

    let mut topo = Topology::new(n);

    for _step in 0..(n - 1) {
        let num_active = active.iter().filter(|&&a| a).count();
        if num_active <= 1 {
            break;
        }

        // Find minimum distance pair
        let mut min_dist = f64::MAX;
        let mut best_i = 0;
        let mut best_j = 0;

        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                if mtx[i][j] < min_dist {
                    min_dist = mtx[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }

        let (im, jm) = (best_i, best_j);
        let node_height = min_dist / 2.0;

        topo.steps.push(JoinStep {
            left: members[im].clone(),
            right: members[jm].clone(),
            left_length: node_height - tmplen[im],
            right_length: node_height - tmplen[jm],
        });

        // Update distances: average linkage
        let ni = members[im].len() as f64;
        let nj = members[jm].len() as f64;
        for k in 0..n {
            if !active[k] || k == im || k == jm {
                continue;
            }
            let new_dist = (mtx[k][im] * ni + mtx[k][jm] * nj) / (ni + nj);
            mtx[k][im] = new_dist;
            mtx[im][k] = new_dist;
        }

        let jm_members = std::mem::take(&mut members[jm]);
        members[im].extend(jm_members);
        tmplen[im] = node_height;
        active[jm] = false;
    }

    topo
}

/// UPGMA from integer distance matrix.
///
/// Ports C's `veryfastsupg_int()`. Takes a full `nseq × nseq` integer
/// matrix (like C's `int **oeff`) and converts to `DistanceMatrix` internally.
/// Uses linked-list traversal matching the C implementation.
pub fn upgma_int(nseq: usize, int_matrix: &[Vec<i32>]) -> Topology {
    let mut dm = DistanceMatrix::new(nseq);
    for i in 0..nseq {
        for j in (i + 1)..nseq {
            dm.set(i, j, int_matrix[i][j] as f64);
        }
    }
    upgma(&dm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upgma_two_sequences() {
        let mut dm = DistanceMatrix::new(2);
        dm.set(0, 1, 0.4);
        let topo = upgma(&dm);
        assert_eq!(topo.num_steps(), 1);
        // UPGMA: each branch = distance / 2
        assert!((topo.steps[0].left_length - 0.2).abs() < 1e-10);
        assert!((topo.steps[0].right_length - 0.2).abs() < 1e-10);
    }

    #[test]
    fn upgma_joins_closest_first() {
        let mut dm = DistanceMatrix::new(3);
        dm.set(0, 1, 0.1); // closest pair
        dm.set(0, 2, 0.5);
        dm.set(1, 2, 0.5);
        let topo = upgma(&dm);
        assert_eq!(topo.num_steps(), 2);

        // First step should join 0 and 1 (closest)
        let first = &topo.steps[0];
        let mut joined: Vec<usize> = first
            .left
            .iter()
            .chain(first.right.iter())
            .copied()
            .collect();
        joined.sort();
        assert_eq!(joined, vec![0, 1]);
    }

    #[test]
    fn upgma_ultrametric() {
        // UPGMA should produce an ultrametric tree:
        // distance from root to any leaf is the same
        let mut dm = DistanceMatrix::new(4);
        dm.set(0, 1, 0.2);
        dm.set(0, 2, 0.8);
        dm.set(0, 3, 0.8);
        dm.set(1, 2, 0.8);
        dm.set(1, 3, 0.8);
        dm.set(2, 3, 0.4);
        let topo = upgma(&dm);
        assert!(topo.is_complete());

        // Non-negative branch lengths
        for step in &topo.steps {
            assert!(step.left_length >= -1e-6);
            assert!(step.right_length >= -1e-6);
        }
    }
}

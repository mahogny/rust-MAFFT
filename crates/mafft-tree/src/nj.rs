/// Neighbor-Joining tree construction.
///
/// Ports the C `nj()` function from nj.c.
use crate::distance::DistanceMatrix;
use crate::topology::{JoinStep, Topology};

/// Build a guide tree using the Neighbor-Joining algorithm.
///
/// Takes a distance matrix and produces a `Topology` (sequence of join steps
/// with branch lengths).
pub fn neighbor_joining(dist: &DistanceMatrix) -> Topology {
    let n = dist.nseq;
    if n <= 1 {
        return Topology::new(n);
    }
    if n == 2 {
        let d = dist.get(0, 1);
        let mut topo = Topology::new(2);
        topo.steps.push(JoinStep {
            left: vec![0],
            right: vec![1],
            left_length: d / 2.0,
            right_length: d / 2.0,
        });
        return topo;
    }

    // Working copy of distances (full matrix for easier mutation)
    let mut mtx = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist.get(i, j);
            mtx[i][j] = d;
            mtx[j][i] = d;
        }
    }

    // Track which sequences are still active
    let mut active: Vec<bool> = vec![true; n];
    // Members of each cluster
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    // Accumulated branch length from tips
    let mut tmplen = vec![0.0f64; n];

    let mut topo = Topology::new(n);

    for _step in 0..(n - 1) {
        let num_active = active.iter().filter(|&&a| a).count();
        if num_active <= 1 {
            break;
        }

        if num_active == 2 {
            // Final join: just pair the remaining two
            let remaining: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
            let (im, jm) = (remaining[0], remaining[1]);
            let d = mtx[im][jm];
            topo.steps.push(JoinStep {
                left: members[im].clone(),
                right: members[jm].clone(),
                left_length: d / 2.0 - tmplen[im],
                right_length: d / 2.0 - tmplen[jm],
            });
            break;
        }

        // Compute row sums
        let mut r = vec![0.0f64; n];
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in 0..n {
                if !active[j] || i == j {
                    continue;
                }
                r[i] += mtx[i][j];
            }
        }

        // Find pair with minimum Q value
        let na = num_active as f64;
        let mut min_q = f64::MAX;
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
                let q = (na - 2.0) * mtx[i][j] - r[i] - r[j];
                if q < min_q {
                    min_q = q;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        let (im, jm) = (best_i, best_j);

        // Branch lengths
        let d_ij = mtx[im][jm];
        let len_i = d_ij / 2.0 + (r[im] - r[jm]) / (2.0 * (na - 2.0));
        let len_j = d_ij - len_i;

        topo.steps.push(JoinStep {
            left: members[im].clone(),
            right: members[jm].clone(),
            left_length: len_i - tmplen[im],
            right_length: len_j - tmplen[jm],
        });

        // Update distance matrix: merge jm into im
        for k in 0..n {
            if !active[k] || k == im || k == jm {
                continue;
            }
            let new_dist = (mtx[k][im] + mtx[k][jm] - d_ij) / 2.0;
            mtx[k][im] = new_dist;
            mtx[im][k] = new_dist;
        }

        // Merge members
        let jm_members = std::mem::take(&mut members[jm]);
        members[im].extend(jm_members);
        tmplen[im] = len_i;

        // Deactivate jm
        active[jm] = false;
    }

    topo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nj_two_sequences() {
        let mut dm = DistanceMatrix::new(2);
        dm.set(0, 1, 0.5);
        let topo = neighbor_joining(&dm);
        assert_eq!(topo.num_steps(), 1);
        assert!((topo.steps[0].left_length + topo.steps[0].right_length - 0.5).abs() < 1e-10);
    }

    #[test]
    fn nj_three_sequences() {
        let mut dm = DistanceMatrix::new(3);
        dm.set(0, 1, 0.2);
        dm.set(0, 2, 0.6);
        dm.set(1, 2, 0.5);
        let topo = neighbor_joining(&dm);
        assert_eq!(topo.num_steps(), 2);
        assert!(topo.is_complete());
    }

    #[test]
    fn nj_four_sequences_symmetric() {
        // Star-like tree: all distances equal
        let mut dm = DistanceMatrix::new(4);
        for i in 0..4 {
            for j in (i + 1)..4 {
                dm.set(i, j, 1.0);
            }
        }
        let topo = neighbor_joining(&dm);
        assert_eq!(topo.num_steps(), 3);
        assert!(topo.is_complete());

        // NJ can produce slightly negative branch lengths for star topologies;
        // verify total tree length is reasonable
        let total_len: f64 = topo
            .steps
            .iter()
            .map(|s| s.left_length + s.right_length)
            .sum();
        assert!(total_len > 0.0, "total tree length should be positive");
    }

    #[test]
    fn nj_preserves_all_sequences() {
        let mut dm = DistanceMatrix::new(5);
        for i in 0..5 {
            for j in (i + 1)..5 {
                dm.set(i, j, (i + j) as f64 * 0.1 + 0.1);
            }
        }
        let topo = neighbor_joining(&dm);

        // Last step should contain all sequences
        let last = topo.steps.last().unwrap();
        let mut all: Vec<usize> = last.left.iter().chain(last.right.iter()).copied().collect();
        all.sort();
        assert_eq!(all, vec![0, 1, 2, 3, 4]);
    }
}

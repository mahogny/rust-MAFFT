//! Identical sequence pairs share one pairwise alignment. Not in C MAFFT.
//!
//! `build_homology_table_with_unalign` aligns every pair `(i, j)`, and everything it derives
//! from a pair (distance, homology regions) is a function of the two sequences' bytes alone;
//! `i` and `j` only label the result. Inputs such as one gene across many genomes repeat the
//! same few alleles, so most pairs repeat an earlier pair's contents. Each distinct ordered
//! pair of contents is aligned once (as its first occurrence, with that pair's real indices)
//! and copied to the others, so the results are the same values in the same order.

use std::collections::HashMap;

pub(crate) struct DistinctPairs {
    /// first occurrence of each distinct ordered pair of contents, in `pairs` order
    pub distinct: Vec<(usize, usize)>,
    /// for each entry of `pairs`, its index in `distinct`
    pub owner: Vec<usize>,
}

pub(crate) fn distinct_pairs(sequences: &[&[u8]], pairs: &[(usize, usize)]) -> DistinctPairs {
    let mut first_copy: HashMap<&[u8], usize> = HashMap::new();
    let content: Vec<usize> = sequences.iter().enumerate().map(|(i, s)| *first_copy.entry(s).or_insert(i)).collect();
    let mut seen: HashMap<(usize, usize), usize> = HashMap::new();
    let mut distinct = Vec::new();
    let owner = pairs
        .iter()
        .map(|&(i, j)| {
            *seen.entry((content[i], content[j])).or_insert_with(|| {
                distinct.push((i, j));
                distinct.len() - 1
            })
        })
        .collect();
    DistinctPairs { distinct, owner }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_contents_share_the_first_pair() {
        let seqs: Vec<&[u8]> = vec![b"AC", b"GT", b"AC", b"GT"];
        let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let d = distinct_pairs(&seqs, &pairs);
        // (AC,GT) (AC,AC) (GT,AC) (GT,GT): order matters, so (1,2) is not (0,1)
        assert_eq!(d.distinct, vec![(0, 1), (0, 2), (1, 2), (1, 3)]);
        assert_eq!(d.owner, vec![0, 1, 0, 2, 3, 0]);
    }
}

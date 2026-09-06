/// Add sequences to an existing alignment.
///
/// Ports C `disttbfast.c`'s `--add` (`-K -I N`) flow: concatenate the
/// existing alignment with the new unaligned sequences, build a guide
/// tree over all (existing + new) sequences, compute `mergeoralign[]`
/// to mark each branch as existing-only ('n', skip), or
/// new-touching ('1'/'2'/'w', do the alignment), then run progressive
/// alignment with the existing alignment columns preserved at 'n'
/// branches.
///
/// The existing input sequences must already be aligned (all the same
/// width). The new sequences are treated as raw (gap chars stripped).
use rayon::prelude::*;

use mafft_tree::{ClusterMethod, DistanceMatrix, Topology, ktuple_distance, musclesupg};
use mafft_types::ScoringContext;

use crate::progressive::{MergeOrAlign, MultipleAlignment};

/// Add new sequences to an existing alignment.
///
/// `existing` is the already-aligned MSA (with gaps).
/// `new_sequences` are the unaligned sequences to add.
/// `new_names` are their names.
/// `scoring` is the scoring context.
/// `use_fft` controls whether FFT acceleration is used for the merge DPs.
///
/// Returns a new MSA containing all sequences (existing + new), aligned.
pub fn add_sequences(
    existing: &MultipleAlignment,
    new_sequences: &[Vec<u8>],
    new_names: &[String],
    scoring: &ScoringContext,
    use_fft: bool,
) -> MultipleAlignment {
    if new_sequences.is_empty() {
        return existing.clone();
    }
    let n_existing = existing.nseq();
    let n_new = new_sequences.len();
    if n_existing == 0 {
        // No anchor; just align the new sequences from scratch.
        return crate::progressive::progressive_align(
            new_sequences,
            new_names,
            &musclesupg(&compute_ktuple_dm(new_sequences), ClusterMethod::default()),
            scoring,
            use_fft,
            None,
        );
    }

    // 1. Strip common-gap columns from the existing alignment. C does
    //    `commongappick(njob-nadd, seq)` at `disttbfast.c:4319` before
    //    the per-step pairalign loop. With our 30-seq existing (width
    //    598), this keeps the columns where at least one existing
    //    sequence has a non-gap residue, dropping the all-gap columns.
    let stripped_existing = commongappick(&existing.sequences);

    // 2. Concatenate stripped existing + new (raw).
    let mut all_seqs: Vec<Vec<u8>> = stripped_existing;
    for s in new_sequences {
        // Strip any gap chars from incoming new sequences (defensive).
        let raw: Vec<u8> = s
            .iter()
            .filter(|&&c| c != b'-' && c != b'.')
            .copied()
            .collect();
        all_seqs.push(raw);
    }
    let mut all_names: Vec<String> = existing.names.clone();
    all_names.extend(new_names.iter().cloned());

    // 3. Guide tree on the all-N+nadd sequences. Distance is computed
    //    on the UNGAPPED form (since stripped existing still has
    //    inter-sequence gaps from the alignment, but the topology
    //    should reflect ungapped similarity).
    let ungapped: Vec<Vec<u8>> = all_seqs
        .iter()
        .map(|s| s.iter().filter(|&&c| c != b'-').copied().collect())
        .collect();
    let dm = compute_ktuple_dm(&ungapped);
    let topo = musclesupg(&dm, ClusterMethod::default());

    // 4. Compute mergeoralign[] tagging each branch.
    let mergeoralign = compute_mergeoralign(&topo, n_existing, n_new);

    // 5. Run progressive alignment with mergeoralign-aware skipping
    //    + new-gap propagation to non-active existing rows.
    crate::progressive::progressive_align_with_mergeoralign_n(
        &all_seqs,
        &all_names,
        &topo,
        &mergeoralign,
        scoring,
        use_fft,
        n_existing,
    )
}

/// Add sequences while preserving the existing alignment's column
/// structure (`--add --keeplength`). After running the standard
/// `add_sequences`, deletes any columns that exist only because of
/// new-sequence insertions, restoring the existing alignment to its
/// original width. Columns where ANY new sequence had a residue but ALL
/// existing sequences had gaps get the new-sequence residue dropped
/// (truncated at that position).
pub fn add_sequences_keeplength(
    existing: &MultipleAlignment,
    new_sequences: &[Vec<u8>],
    new_names: &[String],
    scoring: &ScoringContext,
    use_fft: bool,
) -> MultipleAlignment {
    let (msa, _) =
        add_sequences_keeplength_with_map(existing, new_sequences, new_names, scoring, use_fft);
    msa
}

/// Like [`add_sequences_keeplength`] but also returns the per-added-
/// sequence list of dropped insertion runs (`(start_in_addbk_0based,
/// run_length)` pairs). Used by `--mapout` / `--compactmapout`.
///
/// Each entry corresponds to a maximal run of residues in the
/// ORIGINAL (pre-alignment, gap-free) added sequence that the
/// `--keeplength` column filter dropped — these are the
/// "insertions" the user asked for the map of. Position is
/// 0-indexed in `addbk[i]`.
///
/// C parity: matches the `deletelist` populated by
/// `deletenewinsertions_whole` (`addfunctions.c`) that C feeds to
/// `reconstructdeletemap` / `reconstructdeletemap_compact`.
pub fn add_sequences_keeplength_with_map(
    existing: &MultipleAlignment,
    new_sequences: &[Vec<u8>],
    new_names: &[String],
    scoring: &ScoringContext,
    use_fft: bool,
) -> (MultipleAlignment, Vec<Vec<(usize, usize)>>) {
    if new_sequences.is_empty() {
        return (existing.clone(), Vec::new());
    }
    let target_width = existing.sequences.first().map(|s| s.len()).unwrap_or(0);
    let n_existing = existing.nseq();
    let nadd = new_sequences.len();

    let mut full = add_sequences(existing, new_sequences, new_names, scoring, use_fft);

    let width = full.sequences.first().map(|s| s.len()).unwrap_or(0);
    let mut keep = vec![false; width];
    for col in 0..width {
        let mut any_existing_residue = false;
        for s in full.sequences.iter().take(n_existing) {
            if let Some(&c) = s.get(col) {
                if c != b'-' && c != b'.' {
                    any_existing_residue = true;
                    break;
                }
            }
        }
        keep[col] = any_existing_residue;
    }

    // Build per-added-seq deletelist BEFORE column filtering — we
    // need the post-add (pre-filter) sequence to know which residue
    // of `addbk[i]` lives at each dropped column.
    let mut deletelist: Vec<Vec<(usize, usize)>> = Vec::with_capacity(nadd);
    for i in 0..nadd {
        let aligned = &full.sequences[n_existing + i];
        let mut entries: Vec<(usize, usize)> = Vec::new();
        let mut addbk_pos: usize = 0;
        let mut run_start: usize = 0;
        let mut run_len: usize = 0;
        for (col, &c) in aligned.iter().enumerate() {
            if c == b'-' || c == b'.' {
                continue;
            }
            if keep[col] {
                if run_len > 0 {
                    entries.push((run_start, run_len));
                    run_len = 0;
                }
            } else {
                if run_len == 0 {
                    run_start = addbk_pos;
                }
                run_len += 1;
            }
            addbk_pos += 1;
        }
        if run_len > 0 {
            entries.push((run_start, run_len));
        }
        deletelist.push(entries);
    }

    for s in full.sequences.iter_mut() {
        let mut filtered: Vec<u8> = Vec::with_capacity(target_width);
        for col in 0..s.len() {
            if keep[col] {
                filtered.push(s[col]);
            }
        }
        *s = filtered;
    }

    (full, deletelist)
}

/// Compute `mergeoralign[]` for each branch in the guide tree, mirroring
/// C `disttbfast.c:4282-4304` (`--add` non-profile path).
///
/// C's tag uses `includemember(localmem, addmem)` which returns true iff
/// EVERY member of `localmem` is also in `addmem` (i.e., the subtree is
/// *entirely* composed of new sequences). So:
/// - 'n': neither subtree is entirely new (the merge involves at least
///   one existing-or-mixed branch on each side — could be all-existing or
///   a mix of existing + already-merged-new).
/// - '1': LEFT subtree is entirely new, right has at least one
///   non-new (existing or mixed) member.
/// - '2': RIGHT subtree is entirely new, left has at least one
///   non-new member.
/// - 'w': BOTH subtrees are entirely new (a merge of two new-only
///   subclusters).
///
/// For '1' and '2' merges, C strips common gaps from the *non-all-new*
/// (= mixed) side; that's the side that has accumulated gaps from
/// progressive merges.
pub fn compute_mergeoralign(
    topo: &Topology,
    n_existing: usize,
    _n_new: usize,
) -> Vec<MergeOrAlign> {
    let mut tags: Vec<MergeOrAlign> = Vec::with_capacity(topo.steps.len());
    for step in &topo.steps {
        let left_all_new = step.left.iter().all(|&i| i >= n_existing);
        let right_all_new = step.right.iter().all(|&i| i >= n_existing);
        let tag = match (left_all_new, right_all_new) {
            (false, false) => MergeOrAlign::SkipExisting, // 'n'
            (true, false) => MergeOrAlign::NewLeft,       // '1'
            (false, true) => MergeOrAlign::NewRight,      // '2'
            (true, true) => MergeOrAlign::Wide,           // 'w'
        };
        tags.push(tag);
    }
    tags
}

/// `commongappick`: drop columns where ALL `sequences` are gap chars.
/// Mirrors C `addfunctions.c::commongappick`'s filter behavior.
pub fn commongappick(sequences: &[Vec<u8>]) -> Vec<Vec<u8>> {
    if sequences.is_empty() {
        return Vec::new();
    }
    let width = sequences[0].len();
    let mut keep = vec![false; width];
    for col in 0..width {
        for s in sequences {
            let c = s.get(col).copied().unwrap_or(b'-');
            if c != b'-' && c != b'.' {
                keep[col] = true;
                break;
            }
        }
    }
    sequences
        .iter()
        .map(|s| {
            let mut out = Vec::with_capacity(width);
            for col in 0..s.len() {
                if keep[col] {
                    out.push(s[col]);
                }
            }
            out
        })
        .collect()
}

fn compute_ktuple_dm(sequences: &[Vec<u8>]) -> DistanceMatrix {
    let nseq = sequences.len();
    let pairs: Vec<(usize, usize, f64)> = (0..nseq)
        .into_par_iter()
        .flat_map(|i| {
            let seqs = sequences;
            ((i + 1)..nseq).into_par_iter().map(move |j| {
                let d = ktuple_distance(&seqs[i], &seqs[j], 6);
                (i, j, d)
            })
        })
        .collect();

    let mut dm = DistanceMatrix::new(nseq);
    for (i, j, d) in pairs {
        dm.set(i, j, d);
    }
    dm
}

#[cfg(test)]
mod tests {
    use super::*;
    use mafft_scoring::build_context;
    use mafft_tree::JoinStep;
    use mafft_types::{ScoringModel, SeqType};

    fn make_existing_alignment() -> MultipleAlignment {
        MultipleAlignment {
            sequences: vec![
                b"ACDEFGHIK".to_vec(),
                b"ACDEF-HIK".to_vec(),
                b"ACD---HIK".to_vec(),
            ],
            names: vec!["s1".into(), "s2".into(), "s3".into()],
            score: 0.0,
            step_trace: Vec::new(),
            guide_tree: None,
            first_pass_sequences: None,
            distance_matrix: None,
        }
    }

    #[test]
    fn add_single_sequence() {
        let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);
        let existing = make_existing_alignment();
        let new_seqs = vec![b"ACDEFGHIKLM".to_vec()];
        let new_names = vec!["new1".into()];
        let result = add_sequences(&existing, &new_seqs, &new_names, &scoring, false);
        assert_eq!(result.nseq(), 4);
        let w = result.sequences[0].len();
        for s in &result.sequences {
            assert_eq!(s.len(), w);
        }
    }

    #[test]
    fn add_no_sequences_returns_existing() {
        let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);
        let existing = make_existing_alignment();
        let result = add_sequences(&existing, &[], &[], &scoring, false);
        assert_eq!(result.nseq(), 3);
    }

    #[test]
    fn keeplength_preserves_existing_width() {
        let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);
        let existing = make_existing_alignment();
        let target_width = existing.sequences[0].len();
        let new_seqs = vec![b"ACDEFGHIK".to_vec()];
        let new_names = vec!["new1".into()];
        let result = add_sequences_keeplength(&existing, &new_seqs, &new_names, &scoring, false);
        assert_eq!(result.nseq(), 4);
        for s in &result.sequences {
            assert_eq!(s.len(), target_width);
        }
    }

    #[test]
    fn mergeoralign_three_existing_one_new() {
        // Topology: 4 leaves, 3 join steps. Existing = [0, 1, 2], new = [3].
        let mut topo = Topology::new(4);
        // Step 0: join 0 and 1 (existing-only, 'n').
        topo.steps.push(JoinStep {
            left: vec![0],
            right: vec![1],
            left_length: 0.0,
            right_length: 0.0,
        });
        // Step 1: join (0,1) with 2 (existing-only, 'n').
        topo.steps.push(JoinStep {
            left: vec![0, 1],
            right: vec![2],
            left_length: 0.0,
            right_length: 0.0,
        });
        // Step 2: join (0,1,2) with 3 (right is all-new, '2').
        topo.steps.push(JoinStep {
            left: vec![0, 1, 2],
            right: vec![3],
            left_length: 0.0,
            right_length: 0.0,
        });

        let tags = compute_mergeoralign(&topo, 3, 1);
        assert_eq!(tags.len(), 3);
        assert!(matches!(tags[0], MergeOrAlign::SkipExisting));
        assert!(matches!(tags[1], MergeOrAlign::SkipExisting));
        assert!(matches!(tags[2], MergeOrAlign::NewRight));
    }

    #[test]
    fn mergeoralign_mixed_subtree_is_n_not_2() {
        // C's `includemember` (mltaln9.c:15053) returns true iff EVERY
        // member of the subtree is in addmem. So a subtree with a MIX
        // of existing and new members is NOT classified as all-new.
        // Topology: 4 leaves; existing = [0,1], new = [2,3].
        let mut topo = Topology::new(4);
        // Step 0: join 2 and 3 (both new, 'w').
        topo.steps.push(JoinStep {
            left: vec![2],
            right: vec![3],
            left_length: 0.0,
            right_length: 0.0,
        });
        // Step 1: join 0 and (2,3) — left is existing, right is all-new ('2').
        topo.steps.push(JoinStep {
            left: vec![0],
            right: vec![2, 3],
            left_length: 0.0,
            right_length: 0.0,
        });
        // Step 2: join (0,2,3) and 1 — left is MIXED, right is existing.
        // Neither side is all-new, so 'n'.
        topo.steps.push(JoinStep {
            left: vec![0, 2, 3],
            right: vec![1],
            left_length: 0.0,
            right_length: 0.0,
        });

        let tags = compute_mergeoralign(&topo, 2, 2);
        assert!(matches!(tags[0], MergeOrAlign::Wide));
        assert!(matches!(tags[1], MergeOrAlign::NewRight));
        assert!(matches!(tags[2], MergeOrAlign::SkipExisting));
    }
}

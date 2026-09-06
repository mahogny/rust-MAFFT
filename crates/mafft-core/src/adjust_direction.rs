//! Port of C MAFFT's `--adjustdirection` strand-detection preprocessing.
//!
//! The C pipeline (`scripts/mafft:2323-2342`) runs two helper binaries
//! before any alignment:
//! 1. `makedirectionlist` — k-mer-based scoring of every DNA sequence
//!    against forward AND reverse-complement orientations of already
//!    examined sequences. Emits a per-sequence `_F_`/`_R_` direction
//!    file.
//! 2. `setdirection` — reads that file, reverse-complements sequences
//!    flagged `R`, prefixes their names with `_R_`.
//!
//! This module reproduces both steps in one function:
//! [`adjust_direction`], operating directly on the [`SequenceSet`] the
//! engine receives. Protein inputs pass through unchanged. The 6-mer
//! mode (default `--adjustdirection`, C's `makedirectionlist -m -o a`)
//! is the only mode implemented here — the DP-based `-d` mode is
//! `--adjustdirectionaccurately` and tracked separately.
//!
//! ## Algorithm (6-mer, `mode='a'` averaging — `makedirectionlist.c:1217-1226`)
//!
//! ```text
//! contrastorder = argsort(forward_self - reverse_self) desc        // "most directional first"
//! direction[contrastorder[0]] = F                                  // pivot
//! for i in 1..N (in contrastorder):
//!     ic = contrastorder[i]
//!     for each previously-decided j (limit reflim, default 5000):
//!         resf[j] = common_6mers(comp_table(forward(ic)),  chosen_pointt(j))
//!         resr[j] = common_6mers(comp_table(reverse(ic)),  chosen_pointt(j))
//!     if mean(resr) > mean(resf):
//!         direction[ic] = R
//!     else:
//!         direction[ic] = F
//! if direction[0] == R:                                            // makedirectionlist.c:1261-1270
//!     flip all
//! ```
//!
//! The k-mer encoding and `common_sextets_p` primitive are reused from
//! [`mafft_tree::parttree_dist`].

use mafft_align::{GapModel, local_align};
use mafft_scoring::build_context;
use mafft_tree::parttree_dist::{common_sextets_p, composition_table, encode_points_dna};
use mafft_types::{ScoringModel, SeqType, Sequence, SequenceSet};

/// `--adjustdirection` reference cap (`scripts/mafft:2331` `-r 5000`)
/// for the 6-mer mode.
const REFERENCE_LIMIT_KMER: usize = 5000;

/// `--adjustdirectionaccurately` reference cap (`scripts/mafft:2333`
/// `-r 100`) for the DP mode.
const REFERENCE_LIMIT_DP: usize = 100;

/// `mafft-upstream/core/makedirectionlist.c:464` — tsize for DNA, `pow(4, 6) = 4096`.
const TSIZE: usize = 4096;

/// Which scoring mode the direction adjustment uses.
///
/// - [`Kmer`] (`--adjustdirection`): 6-mer composition overlap via
///   `common_sextets_p` — fast, default.
/// - [`Dp`] (`--adjustdirectionaccurately`): local pairwise alignment
///   score via `local_align` — slower (~N^2 DP) but uses the
///   real substitution scores so it's more robust on divergent
///   sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdjustMode {
    Kmer,
    Dp,
}

/// `mafft-upstream/core/io.c::creverse` complement table (DNA + RNA + IUPAC).
/// Identity for everything outside the table; gap chars left alone.
fn creverse(c: u8) -> u8 {
    match c {
        b'A' => b'T',
        b'C' => b'G',
        b'G' => b'C',
        b'T' => b'A',
        b'U' => b'A',
        b'M' => b'K',
        b'R' => b'Y',
        b'W' => b'W',
        b'S' => b'S',
        b'Y' => b'R',
        b'K' => b'M',
        b'V' => b'B',
        b'H' => b'D',
        b'D' => b'H',
        b'B' => b'V',
        b'N' => b'N',
        b'a' => b't',
        b'c' => b'g',
        b'g' => b'c',
        b't' => b'a',
        b'u' => b'a',
        b'm' => b'k',
        b'r' => b'y',
        b'w' => b'w',
        b's' => b's',
        b'y' => b'r',
        b'k' => b'm',
        b'v' => b'b',
        b'h' => b'd',
        b'd' => b'h',
        b'b' => b'v',
        b'n' => b'n',
        other => other,
    }
}

/// Reverse-complement a DNA sequence (mirrors `io.c::sreverse`).
///
/// C also flips `T`↔`U` when the input had more U than T (treats it as
/// RNA), but the adjustdirection caller has already gappicked and
/// case-preserved its input via `mafft-io`, and we keep the original
/// alphabet to stay byte-identical with C's setdirection output.
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(seq.len());
    for &c in seq.iter().rev() {
        out.push(creverse(c));
    }
    let num_t = seq.iter().filter(|&&c| c == b't' || c == b'T').count();
    let num_u = seq.iter().filter(|&&c| c == b'u' || c == b'U').count();
    if num_u > num_t {
        // `io.c::ttou`: RNA input — convert T→U on the complement.
        for c in &mut out {
            if *c == b't' {
                *c = b'u';
            } else if *c == b'T' {
                *c = b'U';
            }
        }
    }
    out
}

/// Strip gap characters AND normalize to lowercase. Port of
/// `mltaln9.c::gappick0` with an added case-normalization that
/// matters when this function is fed mixed-case input (e.g.,
/// `--add` combines an existing alignment read with
/// `--preservecase` and an addfile read without it). The DNA
/// scoring matrix only populates the lowercase 0-4 and uppercase
/// 5-9 sub-blocks separately — cross-case cells `[0..5][5..10]`
/// are zero — so without normalization `local_align` between
/// lowercase and uppercase versions of the same sequence returns
/// score 0. C MAFFT's `getnumlen_casepreserve` only flips case on
/// recognised gap chars, so cross-case is also possible in C
/// but the makedirectionlist invocation runs `getnumlen` (not
/// the case-preserving variant) which uppercases everything.
fn gappick0(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .copied()
        .filter(|&c| c != b'-' && c != b'.')
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Per-position decision result, plus the orientation actually fed to
/// the alignment engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Reverse,
}

/// Convenience wrapper for the default `--adjustdirection` (k-mer)
/// mode. See [`adjust_direction_mode`] for the full API.
pub fn adjust_direction(input: &SequenceSet) -> SequenceSet {
    adjust_direction_mode(input, AdjustMode::Kmer)
}

/// Like [`adjust_direction_mode`] but only re-orients the LAST `nadd`
/// sequences of the input — the first `nseq - nadd` are anchored
/// as Forward and serve as references. Used when the caller is
/// preparing input for `--add` / `--addfragments`: C MAFFT's
/// `makedirectionlist.c:881-917` slices the per-sequence pointt
/// build by `nadd`, and `makedirectionlist.c:925-936` constrains
/// contrastsort to the added subset.
///
/// `nadd == 0` is equivalent to plain [`adjust_direction_mode`].
pub fn adjust_direction_mode_add(
    input: &SequenceSet,
    mode: AdjustMode,
    nadd: usize,
) -> SequenceSet {
    adjust_direction_with(input, mode, nadd)
}

/// Run direction adjustment over `input` in the requested mode and
/// return a new [`SequenceSet`] with chosen orientations applied
/// and names prefixed by `_R_` where reversed. Protein inputs are
/// returned unchanged.
///
/// On the typical DNA flow, only sequence 0 stays with its original
/// name (matching `setdirection.c:138-155`).
pub fn adjust_direction_mode(input: &SequenceSet, mode: AdjustMode) -> SequenceSet {
    adjust_direction_with(input, mode, 0)
}

/// Implementation backing [`adjust_direction_mode`] and
/// [`adjust_direction_mode_add`].
fn adjust_direction_with(input: &SequenceSet, mode: AdjustMode, nadd: usize) -> SequenceSet {
    if !input.seq_type.is_nucleotide() {
        return input.clone();
    }

    let nseq = input.nseq();
    if nseq == 0 {
        return input.clone();
    }

    // Gappicked forward sequences (analogous to C's `gappick0` pre-pass).
    let forward: Vec<Vec<u8>> = input.sequences.iter().map(|s| gappick0(&s.data)).collect();
    let reverse: Vec<Vec<u8>> = forward.iter().map(|s| reverse_complement(s)).collect();

    // For the DP mode (`--adjustdirectionaccurately`) we need a DNA
    // scoring context. The k-mer mode never touches scoring.
    let scoring = if mode == AdjustMode::Dp {
        Some(build_context(ScoringModel::Dna, SeqType::Dna))
    } else {
        None
    };
    // C `Lalign11.c::L__align11_noalign` reads `penalty`/`penalty_ex`
    // globals set by `constants()`. For the makedirectionlist DNA
    // path that's `DEFAULTGOP_N` / `DEFAULTGEP_N` post-scaling — the
    // same values `build_context` populates into `scoring.gap`.
    let gap_dp = scoring
        .as_ref()
        .map(|s| GapModel::new(s.gap.open as f64, s.gap.extend as f64));

    // Step 1: build forward + reverse-complement 6-mer point vectors
    // (port of `makedirectionlist.c::makepointtable_nuc` calls at lines
    // 905-909). Empty point vector means the sequence had fewer than
    // 6 unambiguous bases; treat that sequence as forward. Used by
    // BOTH modes — the DP mode still uses k-mer contrast for the
    // initial ordering of sequences (C `makedirectionlist.c:937-941`
    // dispatches `makecontrastorder` for `dodp`, but that function
    // also reduces to a forward-vs-reverse self-score difference,
    // and is independent of the per-pair scoring used in step 3).
    let points_fwd: Vec<Vec<u32>> = forward.iter().map(|s| encode_points_dna(s)).collect();
    let points_rev: Vec<Vec<u32>> = reverse.iter().map(|s| encode_points_dna(s)).collect();

    // `n_anchor` = number of existing (pre-added) sequences that are
    // FORCED forward and used as references. For `--add` mode this
    // is `njob - nadd`; without `--add` it's 0 (so step 0 is the
    // pivot — matches C `makedirectionlist.c:881-984`'s
    // `if (nadd) ... else iend = 0/1` slicing).
    let n_anchor = if nadd > 0 && nadd <= nseq {
        nseq - nadd
    } else {
        0
    };

    // Step 2: contrastsort over the testable subset only. C
    // `makedirectionlist.c:925-941` runs `makecontrastorder*` on
    // `contrastorder + istart`, where `istart = njob - nadd`
    // (or 0 without --add). Anchors keep their natural index
    // order at the front of `contrast_order`.
    let mut contrast_order: Vec<(usize, f64)> = Vec::with_capacity(nseq);
    for i in 0..n_anchor {
        contrast_order.push((i, 0.0));
    }
    let mut testable: Vec<(usize, f64)> = match mode {
        AdjustMode::Kmer => (n_anchor..nseq)
            .map(|i| {
                let p_fwd = &points_fwd[i];
                let p_rev = &points_rev[i];
                let t_fwd = composition_table(p_fwd, TSIZE);
                let t_rev = composition_table(p_rev, TSIZE);
                let dif = (common_sextets_p(&t_fwd, p_fwd, TSIZE)
                    - common_sextets_p(&t_rev, p_fwd, TSIZE)) as f64;
                (i, dif)
            })
            .collect(),
        AdjustMode::Dp => {
            let sc = scoring.as_ref().unwrap();
            let gap = gap_dp.as_ref().unwrap();
            (n_anchor..nseq)
                .map(|i| {
                    let fwd_self = local_align(
                        &forward[i],
                        &forward[i],
                        &sc.consweight_matrix,
                        &sc.amino_map,
                        gap,
                        0.0,
                    )
                    .alignment
                    .score;
                    let rev_self = local_align(
                        &forward[i],
                        &reverse[i],
                        &sc.consweight_matrix,
                        &sc.amino_map,
                        gap,
                        0.0,
                    )
                    .alignment
                    .score;
                    (i, fwd_self - rev_self)
                })
                .collect()
        }
    };
    // C uses `qsort` (glibc, unstable) with a `b - a` comparator →
    // descending. To match C's tie-break behaviour as closely as a
    // standard library allows, use rust's `sort_unstable_by`
    // (pdqsort). Rust's stable sort would lock the *original input*
    // order on ties, while qsort's tie-break is data-dependent.
    // Neither matches the other bit-exactly on every conceivable
    // tied input, but unstable matches the spirit better. On the
    // committed fixtures the contrast key
    // (forward_self − reverse_self) does not produce ties, so the
    // choice has no observed effect; the regression test sweep
    // covering 8 + 5 + 17 + 36 sequences is byte-identical to C
    // under both stable and unstable here.
    testable.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    contrast_order.extend(testable);
    let order: Vec<usize> = contrast_order.iter().map(|(i, _)| *i).collect();

    // Step 3: decide orientation for each sequence in contrast order.
    // Without `--add` the first contrast-order entry is forced to F
    // (the pivot); with `--add` the first `n_anchor` entries are
    // forced to F (the existing sequences). Subsequent sequences
    // compare against every previously-decided sequence and pick
    // the higher-mean direction.
    let mut direction = vec![Direction::Forward; nseq];
    let mut chosen_points: Vec<&[u32]> = Vec::with_capacity(nseq);
    let mut chosen_seqs: Vec<&[u8]> = Vec::with_capacity(nseq);

    // Seed `chosen_*` with the anchors (or the single pivot in the
    // no-add case). `pivot_count = max(n_anchor, 1)` matches C's
    // `iend = 1` fallback when `nadd == 0`.
    let pivot_count = n_anchor.max(1);
    for step in 0..pivot_count.min(nseq) {
        chosen_points.push(&points_fwd[order[step]]);
        chosen_seqs.push(&forward[order[step]]);
        // direction[*] is already Forward by default.
    }

    let reflim = match mode {
        AdjustMode::Kmer => REFERENCE_LIMIT_KMER,
        AdjustMode::Dp => REFERENCE_LIMIT_DP,
    };

    for step in pivot_count..nseq {
        let ic = order[step];
        let iend = step.min(reflim);

        let (res_forward, res_reverse) = match mode {
            AdjustMode::Kmer => {
                // Composition tables for forward and reverse candidate
                // orientations of sequence `ic` (C lines 1014-1019).
                let table_fwd = composition_table(&points_fwd[ic], TSIZE);
                let table_rev = composition_table(&points_rev[ic], TSIZE);
                let mut sum_f: f64 = 0.0;
                let mut sum_r: f64 = 0.0;
                for j in 0..iend {
                    let ref_points = chosen_points[j];
                    sum_f += common_sextets_p(&table_fwd, ref_points, TSIZE) as f64;
                    sum_r += common_sextets_p(&table_rev, ref_points, TSIZE) as f64;
                }
                (sum_f / iend as f64, sum_r / iend as f64)
            }
            AdjustMode::Dp => {
                // Per-pair local alignment score for both orientations
                // (C `makedirectionlist.c::directionthread` lines
                // 637-647 in the `dodp` branch).
                let sc = scoring.as_ref().unwrap();
                let gap = gap_dp.as_ref().unwrap();
                let mut sum_f: f64 = 0.0;
                let mut sum_r: f64 = 0.0;
                for j in 0..iend {
                    let r = chosen_seqs[j];
                    sum_f += local_align(
                        &forward[ic],
                        r,
                        &sc.consweight_matrix,
                        &sc.amino_map,
                        gap,
                        0.0,
                    )
                    .alignment
                    .score;
                    sum_r += local_align(
                        &reverse[ic],
                        r,
                        &sc.consweight_matrix,
                        &sc.amino_map,
                        gap,
                        0.0,
                    )
                    .alignment
                    .score;
                }
                (sum_f / iend as f64, sum_r / iend as f64)
            }
        };

        // C `makedirectionlist.c:1234`: strict `>` — ties default to F.
        if res_reverse > res_forward {
            direction[ic] = Direction::Reverse;
            chosen_points.push(&points_rev[ic]);
            chosen_seqs.push(&reverse[ic]);
        } else {
            direction[ic] = Direction::Forward;
            chosen_points.push(&points_fwd[ic]);
            chosen_seqs.push(&forward[ic]);
        }
    }

    // Step 4: if the original-index-0 sequence ended up Reverse,
    // flip every direction (C `makedirectionlist.c:1261-1270`).
    // This ensures the first output sequence is always forward.
    if direction[0] == Direction::Reverse {
        for d in direction.iter_mut() {
            *d = match *d {
                Direction::Forward => Direction::Reverse,
                Direction::Reverse => Direction::Forward,
            };
        }
    }

    // Step 5: materialise the decisions back into a SequenceSet. For
    // reversed sequences, swap the data for the reverse-complement
    // (sreverse on the ORIGINAL non-gappicked data, matching
    // `setdirection.c:142`) and prefix the name with `_R_`. Forward
    // sequences keep their original data and name unchanged.
    let mut adjusted = input.clone();
    for (i, dir) in direction.iter().enumerate() {
        if *dir == Direction::Reverse {
            // setdirection.c operates on the original (possibly
            // gap-containing) sequence. Mirror that — though our
            // engine usually receives already-gappicked input, the
            // call should be safe either way.
            let rc = reverse_complement(&input.sequences[i].data);
            adjusted.sequences[i] = Sequence {
                name: format!("_R_{}", input.sequences[i].name),
                data: rc,
            };
        }
    }
    adjusted
}

#[cfg(test)]
mod tests {
    use super::*;
    use mafft_types::{SeqType, Sequence};

    fn dna_set(seqs: &[(&str, &str)]) -> SequenceSet {
        SequenceSet {
            sequences: seqs
                .iter()
                .map(|(n, s)| Sequence {
                    name: (*n).to_string(),
                    data: s.as_bytes().to_vec(),
                })
                .collect(),
            seq_type: SeqType::Dna,
        }
    }

    #[test]
    fn reverse_complement_basic() {
        assert_eq!(reverse_complement(b"ACGT"), b"ACGT");
        assert_eq!(reverse_complement(b"AAAA"), b"TTTT");
        assert_eq!(reverse_complement(b"AcGt"), b"aCgT");
    }

    #[test]
    fn rna_t_to_u_when_majority_u() {
        // mostly U input → output should also be U-form (creverse U→A,
        // then ttou converts T→U on the complement).
        assert_eq!(reverse_complement(b"uuuu"), b"aaaa");
        assert_eq!(reverse_complement(b"auug"), b"caau");
    }

    #[test]
    fn protein_passthrough() {
        let set = SequenceSet {
            sequences: vec![Sequence {
                name: "p1".into(),
                data: b"MKLVN".to_vec(),
            }],
            seq_type: SeqType::Protein,
        };
        let adjusted = adjust_direction(&set);
        assert_eq!(adjusted.sequences[0].data, b"MKLVN");
        assert_eq!(adjusted.sequences[0].name, "p1");
    }

    #[test]
    fn detects_reversed_sequence() {
        // Build a clearly directional sequence and its reverse-complement.
        // Need ≥6 unambiguous bases for the 6-mer encoder to see it.
        let fwd = "atggcaattcgcatggcaattcgcatggcaattcgc";
        let rc: String = fwd
            .chars()
            .rev()
            .map(|c| match c {
                'a' => 't',
                'c' => 'g',
                'g' => 'c',
                't' => 'a',
                _ => c,
            })
            .collect();
        // Two more forward copies + one reverse → adjust should flip the reverse one.
        let set = dna_set(&[("s1", fwd), ("s2", fwd), ("s3_rc", &rc)]);
        let adjusted = adjust_direction(&set);
        // s3 should be reverse-complemented back to forward, name prefixed with _R_.
        assert!(
            adjusted.sequences[2].name.starts_with("_R_"),
            "expected _R_ prefix, got {}",
            adjusted.sequences[2].name
        );
        assert_eq!(
            adjusted.sequences[2].data,
            fwd.as_bytes(),
            "reversed sequence should now equal forward"
        );
        // s1, s2 unchanged.
        assert_eq!(adjusted.sequences[0].name, "s1");
        assert_eq!(adjusted.sequences[1].name, "s2");
    }
}

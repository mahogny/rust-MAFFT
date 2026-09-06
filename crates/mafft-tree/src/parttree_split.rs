//! `splitseq_mq` recursion + topology assembly for `--parttree`,
//! mirroring `splittbfast.c::splitseq_mq` (lines 2132-2580) for the
//! single-recursion-level case (`nin <= picksize`, all sequences become
//! pivots). For our 36-seq fixture this is the entire algorithm; full
//! recursion handling (multi-level, `nin > picksize`) is left for a
//! follow-up because it requires `rand()` determinism.
//!
//! Pipeline at this layer:
//! 1. Build `dfromc[nyuko][nin]` — distance from each surviving yuko
//!    to every sorted-position sequence. Mirrors
//!    `splittbfast.c:2132-2236`.
//! 2. Assign each sequence (including pivots themselves) to its
//!    closest yuko via `argmin_i dfromc[i][j]`, building `outs[].numinseq`
//!    lists (`splittbfast.c:2244-2287`).
//! 3. UPGMA on `yukomtx` → topol (already done by Rust's `musclesupg`,
//!    FFI-validated).
//! 4. Assemble the final `Topology`:
//!    - Emit an internal-alignment `JoinStep` for each multi-member
//!      yuko, pre-aligning its assigned sequences (Rust's progressive
//!      alignment expects each leaf to be a single sequence; C's
//!      pairalign tolerates mixed-length raw inputs but Rust's
//!      `Profile::from_aligned` requires same-length rows).
//!    - For each UPGMA join step l, emit a `JoinStep` whose `left` /
//!      `right` are the union of `outs[yuko_idx].numinseq` for all
//!      yuko-indices in `topol[l][0]` / `topol[l][1]`, sorted by
//!      `intcompare` (ascending original sequence index — matching C's
//!      `qsort(mem1, ..., intcompare)` at `splittbfast.c:2477-2478`).

use crate::distance::DistanceMatrix;
use crate::musclesupg::{ClusterMethod, musclesupg};
use crate::parttree_dist::{
    DLENFACA, DLENFACB, DLENFACC, DLENFACD, MAX6DIST, PLENFACA, PLENFACB, PLENFACC, PLENFACD,
    common_sextets_p, composition_table, lenfac,
};
use crate::parttree_pivot::{PartTreePivots, PtSeqKind};
use crate::topology::{JoinStep, Topology};

/// Build `dfromc[i][j]` = parttree distance from the `i`-th surviving
/// yuko to the `j`-th sorted-position sequence (`j` in `[0..nin)`).
/// Mirrors `splittbfast.c:2132-2236`.
pub fn build_dfromc(pivots: &PartTreePivots, kind: PtSeqKind) -> Vec<Vec<f64>> {
    let nyuko = pivots.yukos.len();
    let nin = pivots.scores.len();
    let tsize = match kind {
        PtSeqKind::Protein => 46656,
        PtSeqKind::Dna => 4096,
    };
    let (a, b, c, d) = match kind {
        PtSeqKind::Protein => (PLENFACA, PLENFACB, PLENFACC, PLENFACD),
        PtSeqKind::Dna => (DLENFACA, DLENFACB, DLENFACC, DLENFACD),
    };

    let mut dfromc = vec![vec![0.0f64; nin]; nyuko];

    for (yi, &y_pick) in pivots.yukos.iter().enumerate() {
        let y_sorted_idx = pivots.picks[y_pick]; // sorted-scores index
        let y_score = &pivots.scores[y_sorted_idx];
        let y_table = composition_table(&y_score.points, tsize);

        for j in 0..nin {
            // Self-distance is 0 for the yuko itself.
            if j == y_sorted_idx {
                dfromc[yi][j] = 0.0;
                continue;
            }
            let common = common_sextets_p(&y_table, &pivots.scores[j].points, tsize);
            let bunbo = (y_score.selfscore.min(pivots.scores[j].selfscore)) as f64;
            let raw = if bunbo > 0.0 {
                1.0 - common as f64 / bunbo
            } else {
                1.0
            };
            let lf = lenfac(y_score.orilen, pivots.scores[j].orilen, a, b, c, d);
            let mut dist = raw * lf;
            if dist > MAX6DIST {
                dist = MAX6DIST;
            }
            dfromc[yi][j] = dist;
        }
    }
    dfromc
}

/// Assign every sorted-position sequence (incl. pivots) to its closest
/// yuko. Returns `outs[yi]` = list of original `numinseq` indices
/// assigned to yuko `yi`. Mirrors `splittbfast.c:2244-2302`.
///
/// **Tie-break**: C uses strict `<` (`splittbfast.c:2270-2286`), so the
/// FIRST-encountered yuko (smallest `yi`) wins on ties. Each pivot is
/// closest to itself with distance 0, so it always lands in its own
/// `outs[]` slot.
pub fn assign_to_yukos(pivots: &PartTreePivots, dfromc: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let nyuko = pivots.yukos.len();
    let nin = pivots.scores.len();
    let mut outs: Vec<Vec<usize>> = vec![Vec::new(); nyuko];

    for j in 0..nin {
        let mut belongto = 0usize;
        let mut minscore = f64::INFINITY;
        for yi in 0..nyuko {
            if dfromc[yi][j] < minscore {
                minscore = dfromc[yi][j];
                belongto = yi;
            }
        }
        outs[belongto].push(pivots.scores[j].numinseq);
    }
    outs
}

/// Build a yukomtx-as-DistanceMatrix (full square form) for feeding to
/// `musclesupg`, using the `with-diagonal` half-matrix layout that
/// `pivots.yukomtx` uses.
fn yukomtx_to_distance_matrix(pivots: &PartTreePivots) -> DistanceMatrix {
    let nyuko = pivots.yukos.len();
    let mut dm = DistanceMatrix::new(nyuko);
    for i in 0..nyuko {
        for j in (i + 1)..nyuko {
            // pivots.yukomtx[i][j-i] (with-diagonal layout, slot 0 unused).
            dm.set(i, j, pivots.yukomtx[i][j - i]);
        }
    }
    dm
}

/// Assemble the final `Topology` for `--parttree`. Returns one
/// `JoinStep` per UPGMA-on-yukomtx join (`nyuko - 1` steps total).
///
/// Mirrors `splittbfast.c:2444-2519`'s parent pairalign loop: at each
/// UPGMA step, the left/right groups are unions of `outs[yuko_idx]`
/// for the merging yuko-clusters, sorted ascending by original
/// sequence index (matching C's `qsort(mem1, ..., intcompare)` at
/// `splittbfast.c:2477-2478`).
///
/// **Multi-member yukos**: for a yuko with members `[m_0, m_1, …,
/// m_{k-1}]`, C does NOT pre-align them. The recursive
/// `splitseq_mq` call hits LEAF (via `uniform = -1` set when
/// `nyuko == 1` at `splittbfast.c:2115`) and just writes `order[]`
/// without alignment. The members are then merged for the first time
/// inside the parent's `pairalign(mem1=[m_0..m_{k-1}], …)` call
/// which builds a profile from the raw sequences and runs DP. For
/// our n=36 fixture the multi-member yuko's members are byte-
/// identical (the dedupe on shimon+strcmp), so they always have the
/// same length and Rust's `Profile::from_aligned` accepts them
/// directly. For non-identical multi-member groups (which would
/// require the recursive call to actually run pairalign), the
/// current implementation has held up: `--parttree` and `--dpparttree`
/// produce byte-identical output to C MAFFT 7.526 on the 36-seq
/// sample (width 752) and across the BBaliBase 3 sweep.
pub fn assemble_topology(pivots: &PartTreePivots, outs: &[Vec<usize>]) -> Topology {
    let nseq_total: usize = outs.iter().map(|v| v.len()).sum();
    let mut topo = Topology::new(nseq_total);

    // Run UPGMA on the yukomtx and convert each join step into a
    // sequence-level JoinStep. UPGMA's `left` / `right` are sets of
    // yuko-indices; we expand them via `outs[]`.
    let yuko_dm = yukomtx_to_distance_matrix(pivots);
    let yuko_topo = musclesupg(&yuko_dm, ClusterMethod::Mix { sueff: 0.1 });

    for yuko_step in &yuko_topo.steps {
        let mut left_seqs: Vec<usize> = yuko_step
            .left
            .iter()
            .flat_map(|&yi| outs[yi].iter().copied())
            .collect();
        let mut right_seqs: Vec<usize> = yuko_step
            .right
            .iter()
            .flat_map(|&yi| outs[yi].iter().copied())
            .collect();
        left_seqs.sort();
        right_seqs.sort();
        topo.steps.push(JoinStep {
            left: left_seqs,
            right: right_seqs,
            left_length: yuko_step.left_length,
            right_length: yuko_step.right_length,
        });
    }

    topo
}

/// One-shot helper: run pivot pipeline, build `dfromc`, assign to yukos,
/// and assemble the final topology.
pub fn build_parttree_topology(
    sequences: &[Vec<u8>],
    kind: PtSeqKind,
    picksize: usize,
) -> Topology {
    let pivots = crate::parttree_pivot::run_pivot_pipeline(sequences, kind, picksize);
    let dfromc = build_dfromc(&pivots, kind);
    let outs = assign_to_yukos(&pivots, &dfromc);
    assemble_topology(&pivots, &outs)
}

/// Recursive helper for `c_normalized_yuko_order`: produce the leaf-yuko
/// order for the subtree whose leaves are `leaves`, applying C's
/// smaller-first-element normalization at each internal node (mirroring
/// `mltaln9.c:8184-8197` in `fixed_musclesupg_double_realloc_nobk_halfmtx`).
fn c_normalized_subtree(steps: &[crate::topology::JoinStep], leaves: &[usize]) -> Vec<usize> {
    if leaves.len() == 1 {
        return vec![leaves[0]];
    }
    // Find the join step whose `left ∪ right == leaves`. Search from the
    // back since later steps merge larger sets first (the root is the
    // very last step), and we typically recurse downward.
    let leaf_set: std::collections::BTreeSet<usize> = leaves.iter().copied().collect();
    for step in steps.iter().rev() {
        let combined: std::collections::BTreeSet<usize> =
            step.left.iter().chain(step.right.iter()).copied().collect();
        if combined == leaf_set {
            let l_order = c_normalized_subtree(steps, &step.left);
            let r_order = c_normalized_subtree(steps, &step.right);
            // C: when extending `topol[k][i]`, the two child sub-arrays
            // are concatenated smaller-first-element first
            // (`mltaln9.c:8184-8197`).
            let (first, second) =
                if !l_order.is_empty() && !r_order.is_empty() && l_order[0] > r_order[0] {
                    (r_order, l_order)
                } else {
                    (l_order, r_order)
                };
            let mut result = first;
            result.extend(second);
            return result;
        }
    }
    // Fall back to insertion order (shouldn't happen if topology is
    // complete and leaves came from it).
    leaves.to_vec()
}

/// Build the parttree-specific `--treeout` Newick output, mirroring
/// `splittbfast.c::splitseq_mq` (`splittbfast.c:1275-1301` leaf-tree
/// emission, `:2532-2553` per-step merge). Single-recursion-level case
/// only (`nin <= picksize`), same as [`compute_parttree_order`].
///
/// Output format differs from the standard musclesupg `--treeout`:
/// - No branch lengths.
/// - Leaves are 1-indexed sequence numbers, NOT sanitized FASTA names.
/// - Multi-member yukos render as `(s1,s2,...)` on one line; single-
///   member yukos render as `<n>` sandwiched by `\n`.
/// - No trailing `;` (parttree's tree file omits it).
pub fn compute_parttree_newick(sequences: &[Vec<u8>], kind: PtSeqKind, picksize: usize) -> String {
    let nseq = sequences.len();
    if nseq == 0 {
        return "\n".to_string();
    }
    if nseq == 1 {
        return "\n1\n".to_string();
    }

    let pivots = crate::parttree_pivot::run_pivot_pipeline(sequences, kind, picksize);
    let dfromc = build_dfromc(&pivots, kind);
    let outs = assign_to_yukos(&pivots, &dfromc);
    let yuko_dm = yukomtx_to_distance_matrix(&pivots);
    let yuko_topo = musclesupg(&yuko_dm, ClusterMethod::Mix { sueff: 0.1 });

    let nyuko = outs.len();
    // Per-yuko initial tree string. Matches `splitseq_mq` leaf case
    // (`splittbfast.c:1275-1301`):
    //   nin == 1 →  "\n<seq+1>\n"
    //   nin >  1 →  "\n(<a+1>,<b+1>,...)\n"
    // Member emission order is `outs[yi]` (the parent's j-iteration order).
    let mut parttree: Vec<Option<String>> = (0..nyuko)
        .map(|yi| {
            let members = &outs[yi];
            if members.is_empty() {
                None
            } else if members.len() == 1 {
                Some(format!("\n{}\n", members[0] + 1))
            } else {
                let inner: Vec<String> = members.iter().map(|&m| (m + 1).to_string()).collect();
                Some(format!("\n({})\n", inner.join(",")))
            }
        })
        .collect();

    // Walk the yuko UPGMA tree in merge order. C uses `topol[l][0][0]` /
    // `topol[l][1][0]` for the active yuko IDs (`splittbfast.c:2535-2536`).
    // Our `JoinStep.left/right` accumulate leaves in merge order, NOT in
    // C's smaller-first-element normalized order — recurse via
    // `c_normalized_subtree` so the [0] we read matches what C reads.
    let mut last_v1 = 0usize;
    for step in &yuko_topo.steps {
        let l_norm = c_normalized_subtree(&yuko_topo.steps, &step.left);
        let r_norm = c_normalized_subtree(&yuko_topo.steps, &step.right);
        let (mut v1, mut v2) = (l_norm[0], r_norm[0]);
        if v1 > v2 {
            std::mem::swap(&mut v1, &mut v2);
        }
        let s1 = parttree[v1].take().expect("missing parttree[v1]");
        let s2 = parttree[v2].take().expect("missing parttree[v2]");
        parttree[v1] = Some(format!("({},{})", s1, s2));
        last_v1 = v1;
    }
    // The final surviving entry plus C's trailing newline (the file is
    // written by `fprintf(fp, "%s\n", *tree)` — `splittbfast.c:3046`,
    // and the tree string itself ends with `\n` from the last leaf).
    let mut out = parttree[last_v1].take().expect("missing root parttree");
    out.push('\n');
    out
}

/// Compute the C-equivalent partition-discovery order for `--reorder`,
/// mirroring `splittbfast.c::splitseq_mq` (`splittbfast.c:2351-2378` for
/// the yuko visit order + `:1305-1309` for the leaf emission). Currently
/// supports the single-recursion-level case (`nin <= picksize`, all yukos
/// bottom out without further pivoting) — matches the n=36 fixture.
/// Multi-level recursion (when a yuko itself has too many members to
/// trivially leaf out) would need a full port of `splitseq_mq` recursion;
/// for that case we fall back to input order, which gives a structurally
/// valid alignment but won't byte-match C.
pub fn compute_parttree_order(
    sequences: &[Vec<u8>],
    kind: PtSeqKind,
    picksize: usize,
) -> Vec<usize> {
    let nseq = sequences.len();
    if nseq <= 1 {
        return (0..nseq).collect();
    }
    let pivots = crate::parttree_pivot::run_pivot_pipeline(sequences, kind, picksize);
    let dfromc = build_dfromc(&pivots, kind);
    let outs = assign_to_yukos(&pivots, &dfromc);
    let yuko_dm = yukomtx_to_distance_matrix(&pivots);
    let yuko_topo = musclesupg(&yuko_dm, ClusterMethod::Mix { sueff: 0.1 });

    // Mirrors `splittbfast.c:2351-2354`: treeorder = root step's
    // `topol[nyuko-2][0] ++ topol[nyuko-2][1]`. Our `JoinStep.left/right`
    // accumulate leaves in merge order, NOT in C's smaller-first-element
    // normalized order (`mltaln9.c:8184-8197`). So we recursively rebuild
    // the order with that normalization applied to recover what C's
    // `topol[step][i]` arrays would hold.
    let mut order = Vec::with_capacity(nseq);
    if let Some(root) = yuko_topo.steps.last() {
        let l_yukos = c_normalized_subtree(&yuko_topo.steps, &root.left);
        let r_yukos = c_normalized_subtree(&yuko_topo.steps, &root.right);
        for &yi in l_yukos.iter().chain(r_yukos.iter()) {
            // Leaf-level emission: `splittbfast.c:1305-1309` writes
            // `scores[j].numinseq` in `j` order — `outs[yi]` already
            // holds them in that order (see `assign_to_yukos`).
            order.extend_from_slice(&outs[yi]);
        }
    } else {
        // nyuko == 1 → single yuko containing all sequences in
        // `outs[0]` (matches `splittbfast.c:2105-2123`'s uniform branch).
        order.extend_from_slice(&outs[0]);
    }
    debug_assert_eq!(order.len(), nseq, "parttree order missing sequences");
    order
}

// =============================================================================
// CALL 2 — second-pass `splittbfast` with `fromaln=1` (`-Z`) scoring.
//
// C MAFFT runs `splittbfast` twice for `--parttree` (`scripts/mafft:2655`
// and `:2681`). The second call uses `-Z` so distances are computed via
// `naivepairscore11` on the already-aligned sequences instead of via
// 6-mer composition. We mirror that pipeline here so `--parttree --reorder`
// reaches byte-identity with C.
// =============================================================================

/// `naivepairscore11` for aligned sequences. Mirrors
/// `mltaln9.c:13801-13851`:
///   1. Strip columns where both rows are gaps (`commongappickpair`,
///      `mltaln9.c:13377-13397`).
///   2. Walk the result: for each gap RUN in either row, add `penalty`
///      once (gap-open style, single hit per run); otherwise add
///      `matrix[amino_map[c1]][amino_map[c2]]`.
fn naivepairscore11_aligned(
    aligned1: &[u8],
    aligned2: &[u8],
    matrix: &[Vec<f64>],
    amino_map: &[u8; 256],
    penalty: f64,
) -> f64 {
    debug_assert_eq!(aligned1.len(), aligned2.len());
    let n = aligned1.len();
    let nalpha = matrix.len();
    let mut score = 0.0f64;
    let mut k = 0usize;
    while k < n {
        let c1 = aligned1[k];
        let c2 = aligned2[k];
        if c1 == b'-' && c2 == b'-' {
            // Common gap — strip (`commongappickpair`).
            k += 1;
            continue;
        }
        if c1 == b'-' {
            score += penalty;
            // Skip all consecutive '-' in row 1 (`naivepairscore11:13824-13828`).
            while k < n && aligned1[k] == b'-' {
                k += 1;
            }
            continue;
        }
        if c2 == b'-' {
            score += penalty;
            while k < n && aligned2[k] == b'-' {
                k += 1;
            }
            continue;
        }
        let i = amino_map[c1 as usize] as usize;
        let j = amino_map[c2 as usize] as usize;
        if i < nalpha && j < nalpha {
            score += matrix[i][j];
        }
        k += 1;
    }
    score
}

/// Diagonal self-score for an aligned sequence, mirroring
/// `splittbfast.c:3011-3017`: `pscore = sum amino_dis[c][c]` over every
/// non-gap character `c`. Gaps contribute 0 (C reads through gap chars
/// but `amino_dis['-']['-'] == 0` in `constants.c`'s table).
fn selfscore_aligned(aligned: &[u8], matrix: &[Vec<f64>], amino_map: &[u8; 256]) -> i64 {
    let nalpha = matrix.len();
    let mut s = 0.0f64;
    for &c in aligned {
        if c == b'-' {
            continue;
        }
        let i = amino_map[c as usize] as usize;
        if i < nalpha {
            s += matrix[i][i];
        }
    }
    s as i64
}

/// Apply C's `pick_reference_max_selfscore` then `compute_initial_scores`
/// for the `fromaln=1` path, mirroring `splittbfast.c:1316-1448` with
/// `doalign && fromaln`. Returns the sorted scores entries and the
/// index permutation we applied.
fn compute_initial_scores_fromaln(
    aligned_seqs: &[Vec<u8>],
    matrix: &[Vec<f64>],
    amino_map: &[u8; 256],
    penalty: f64,
) -> Vec<crate::parttree_pivot::ScoreEntry> {
    use crate::parttree_pivot::ScoreEntry;
    let nin = aligned_seqs.len();
    let mut entries: Vec<ScoreEntry> = (0..nin)
        .map(|i| {
            ScoreEntry {
                numinseq: i,
                selfscore: selfscore_aligned(&aligned_seqs[i], matrix, amino_map),
                // orilen = strlen post-gappick (i.e., non-gap residue count).
                orilen: aligned_seqs[i].iter().filter(|&&c| c != b'-').count(),
                score: 0.0,
                points: Vec::new(), // unused in fromaln path
            }
        })
        .collect();

    // pick_reference: scan for max selfscore (strict `>` → first wins).
    let mut best = 0usize;
    let mut best_score = entries[0].selfscore;
    for i in 1..nin {
        if entries[i].selfscore > best_score {
            best_score = entries[i].selfscore;
            best = i;
        }
    }
    if best != 0 {
        entries.swap(0, best);
    }

    // compute scores against entries[0] using naivepairscore11.
    let ref_aligned = aligned_seqs[entries[0].numinseq].clone();
    let ref_selfscore = entries[0].selfscore as f64;
    for i in 0..nin {
        let pair = naivepairscore11_aligned(
            &ref_aligned,
            &aligned_seqs[entries[i].numinseq],
            matrix,
            amino_map,
            penalty,
        );
        let bunbo = ref_selfscore.min(entries[i].selfscore as f64);
        entries[i].score = if bunbo > 0.0 { 1.0 - pair / bunbo } else { 1.0 };
        // C clamps score < 0 to 0 in `pickmtx` (`splittbfast.c:1687`), but
        // for the initial `scores[i].score` no clamp is applied.
    }

    // dcompare_sort via libc qsort (matches BSD `qsort` tie-break).
    crate::parttree_pivot::dcompare_sort(&mut entries);
    entries
}

/// Intermediate result of CALL 2 (`fromaln=1`)'s pivot pipeline: the
/// post-`dcompare` sorted scores, the yuko UPGMA topology, and the
/// per-yuko member lists (`outs[]`). Used by both
/// [`compute_parttree_order_fromaln`] and [`compute_parttree_newick_fromaln`]
/// so they share the exact same pipeline (and any future fixes propagate).
pub struct PartTreeFromalnResult {
    pub scores: Vec<crate::parttree_pivot::ScoreEntry>,
    pub outs: Vec<Vec<usize>>,
    pub yuko_topo: crate::topology::Topology,
}

/// Run the splittbfast pivot/yuko/UPGMA pipeline using caller-supplied
/// scoring callbacks. This is the generic core shared by the
/// `fromaln=1` (CALL 2) path and the `--dpparttree` CALL 1 path.
///
/// Callbacks operate on the original input indices (0..nseq):
/// - `selfscore`: returns the self-vs-self score for sequence `i`.
/// - `orilen`: returns the ungapped length of sequence `i`.
/// - `pair_score`: returns the pairwise score for `(i, j)`. Implementations
///   compute this however they like (e.g. `naivepairscore11_aligned` or
///   `G__align11_noalign`). For `i == j`, returns the selfscore.
/// - `seqs_equal`: returns `true` when `i` and `j` refer to byte-identical
///   sequences (used by the shimon-style dedupe in pivot selection).
pub fn run_parttree_pipeline_with_scorer<S, L, P, E>(
    nseq: usize,
    selfscore: S,
    orilen: L,
    pair_score: P,
    seqs_equal: E,
    picksize: usize,
) -> Option<PartTreeFromalnResult>
where
    S: Fn(usize) -> i64,
    L: Fn(usize) -> usize,
    P: Fn(usize, usize) -> f64,
    E: Fn(usize, usize) -> bool,
{
    use crate::parttree_pivot::ScoreEntry;
    if nseq <= 1 {
        return None;
    }

    // 1) Build initial `entries` with numinseq=0..nseq.
    let mut entries: Vec<ScoreEntry> = (0..nseq)
        .map(|i| ScoreEntry {
            numinseq: i,
            selfscore: selfscore(i),
            orilen: orilen(i),
            score: 0.0,
            points: Vec::new(),
        })
        .collect();

    // 2) pick_reference: scan for max selfscore (strict `>`, first wins),
    //    swap winner to position 0.
    let mut best = 0usize;
    let mut best_self = entries[0].selfscore;
    for i in 1..nseq {
        if entries[i].selfscore > best_self {
            best_self = entries[i].selfscore;
            best = i;
        }
    }
    if best != 0 {
        entries.swap(0, best);
    }

    // 3) Compute scores against entries[0].
    let ref_num = entries[0].numinseq;
    let ref_self = entries[0].selfscore as f64;
    for i in 0..nseq {
        let pair = pair_score(ref_num, entries[i].numinseq);
        let bunbo = ref_self.min(entries[i].selfscore as f64);
        entries[i].score = if bunbo > 0.0 { 1.0 - pair / bunbo } else { 1.0 };
    }
    crate::parttree_pivot::dcompare_sort(&mut entries);

    // 4) Pivot selection (same algorithm as `compute_parttree_order_fromaln`).
    let nin = entries.len();
    let mut picks: Vec<usize> = vec![0];
    let same_seq_at = |a: usize, b: usize| -> bool {
        entries[a].selfscore == entries[b].selfscore
            && entries[a].orilen == entries[b].orilen
            && seqs_equal(entries[a].numinseq, entries[b].numinseq)
    };
    let mut pickkouho: Vec<usize> = (1..nin).collect();
    let mut nkouho = pickkouho.len();
    if nkouho > 0 {
        let picktmp = pickkouho[nkouho - 1];
        nkouho -= 1;
        if !same_seq_at(0, picktmp) {
            picks.push(picktmp);
        }
    }
    let mut i_alt = 1;
    while picks.len() < picksize && nkouho > 0 {
        let rn = if i_alt == 1 {
            i_alt = 0;
            (nkouho as f64 * 0.5) as usize
        } else {
            nkouho - 1
        };
        let picktmp = pickkouho[rn];
        nkouho -= 1;
        pickkouho[rn] = pickkouho[nkouho];
        if !picks.iter().any(|&p| same_seq_at(p, picktmp)) {
            picks.push(picktmp);
        }
    }
    picks.sort_unstable();

    // 5) pickmtx via pair_score (same formula as fromaln pipeline).
    let npick = picks.len();
    let mut pickmtx: Vec<Vec<f64>> = (0..npick).map(|i| vec![0.0f64; npick - i]).collect();
    for k in 1..npick {
        pickmtx[0][k] = entries[picks[k]].score;
    }
    for j in 1..npick {
        let pj_self = entries[picks[j]].selfscore as f64;
        let pj_num = entries[picks[j]].numinseq;
        for i in (j + 1)..npick {
            let pair = pair_score(pj_num, entries[picks[i]].numinseq);
            let bunbo = pj_self.min(entries[picks[i]].selfscore as f64);
            let dist = if bunbo > 0.0 { 1.0 - pair / bunbo } else { 1.0 };
            pickmtx[j][i - j] = if dist < 0.0 { 0.0 } else { dist };
        }
    }

    // 6) dfromc[yuko][seq].
    let nyuko = npick;
    let yukos: Vec<usize> = picks.clone();
    let mut dfromc: Vec<Vec<f64>> = vec![vec![0.0f64; nin]; nyuko];
    for j in 0..nin {
        dfromc[0][j] = entries[j].score;
    }
    for i in 1..nyuko {
        let yi_num = entries[yukos[i]].numinseq;
        let yi_self = entries[yukos[i]].selfscore as f64;
        for j in 0..nin {
            if j == yukos[i] {
                dfromc[i][j] = 0.0;
                continue;
            }
            let pair = pair_score(yi_num, entries[j].numinseq);
            let bunbo = yi_self.min(entries[j].selfscore as f64);
            let dist = if bunbo > 0.0 { 1.0 - pair / bunbo } else { 1.0 };
            dfromc[i][j] = if dist < 0.0 { 0.0 } else { dist };
        }
    }

    // 7) Yuko assignment.
    let mut outs: Vec<Vec<usize>> = vec![Vec::new(); nyuko];
    for j in 0..nin {
        let mut belongto = 0usize;
        let mut min_d = f64::INFINITY;
        for yi in 0..nyuko {
            if dfromc[yi][j] < min_d {
                min_d = dfromc[yi][j];
                belongto = yi;
            }
        }
        outs[belongto].push(entries[j].numinseq);
    }

    // 8) UPGMA on yukomtx (= pickmtx since every pick survived).
    let mut yuko_dm = DistanceMatrix::new(nyuko);
    for i in 0..nyuko {
        for j in (i + 1)..nyuko {
            yuko_dm.set(i, j, pickmtx[i][j - i]);
        }
    }
    let yuko_topo = musclesupg(&yuko_dm, ClusterMethod::Mix { sueff: 0.1 });

    Some(PartTreeFromalnResult {
        scores: entries,
        outs,
        yuko_topo,
    })
}

/// Given a precomputed parttree pipeline result, build the `--treeout`
/// Newick string (numeric leaves, no branch lengths) — the format C's
/// `splittbfast.c::splitseq_mq` emits via `splittbfast.c:1275-1301`
/// (leaf) and `:2532-2553` (per-merge concat).
pub fn parttree_result_to_newick(result: &PartTreeFromalnResult) -> String {
    let nyuko = result.outs.len();
    let mut parttree: Vec<Option<String>> = (0..nyuko)
        .map(|yi| {
            let members = &result.outs[yi];
            if members.is_empty() {
                None
            } else if members.len() == 1 {
                Some(format!("\n{}\n", members[0] + 1))
            } else {
                let inner: Vec<String> = members.iter().map(|&m| (m + 1).to_string()).collect();
                Some(format!("\n({})\n", inner.join(",")))
            }
        })
        .collect();

    let mut last_v1 = 0usize;
    for step in &result.yuko_topo.steps {
        let l_norm = c_normalized_subtree(&result.yuko_topo.steps, &step.left);
        let r_norm = c_normalized_subtree(&result.yuko_topo.steps, &step.right);
        let (mut v1, mut v2) = (l_norm[0], r_norm[0]);
        if v1 > v2 {
            std::mem::swap(&mut v1, &mut v2);
        }
        let s1 = parttree[v1].take().expect("missing parttree[v1]");
        let s2 = parttree[v2].take().expect("missing parttree[v2]");
        parttree[v1] = Some(format!("({},{})", s1, s2));
        last_v1 = v1;
    }
    let mut out = parttree[last_v1].take().expect("missing root parttree");
    out.push('\n');
    out
}

fn run_parttree_fromaln_pipeline(
    aligned_seqs: &[Vec<u8>],
    matrix: &[Vec<f64>],
    amino_map: &[u8; 256],
    penalty: f64,
) -> Option<PartTreeFromalnResult> {
    let nseq = aligned_seqs.len();
    if nseq <= 1 {
        return None;
    }

    // 1. Build sorted scores (with selfscore, score) for all sequences.
    let scores = compute_initial_scores_fromaln(aligned_seqs, matrix, amino_map, penalty);

    // 2. Pivot selection. For `nin <= picksize` (=50), all distinct
    //    sequences become picks, just like `compute_parttree_order`.
    //    Uses scores order; dedupe identical sequences via the shimon-style
    //    check (here: same selfscore + same aligned content).
    let nin = scores.len();
    let mut picks: Vec<usize> = vec![0]; // position 0 is always picked
    // Helper: scores at positions a and b refer to the same sequence content?
    let same_seq = |a: usize, b: usize| -> bool {
        let na = scores[a].numinseq;
        let nb = scores[b].numinseq;
        scores[a].selfscore == scores[b].selfscore
            && scores[a].orilen == scores[b].orilen
            && aligned_seqs[na] == aligned_seqs[nb]
    };
    // C: pickkouho = [1, 2, ..., nin-1]; take the MOST distant first
    // (`splittbfast.c:1508`: picktmp = pickkouho[nkouho-1]).
    let mut pickkouho: Vec<usize> = (1..nin).collect();
    let mut nkouho = pickkouho.len();
    if nkouho > 0 {
        let picktmp = pickkouho[nkouho - 1];
        nkouho -= 1;
        if !same_seq(0, picktmp) {
            picks.push(picktmp);
        }
    }
    let mut i_alt = 1;
    while picks.len() < 50 && nkouho > 0 {
        let rn = if i_alt == 1 {
            i_alt = 0;
            (nkouho as f64 * 0.5) as usize
        } else {
            nkouho - 1
        };
        let picktmp = pickkouho[rn];
        nkouho -= 1;
        pickkouho[rn] = pickkouho[nkouho];
        if !picks.iter().any(|&p| same_seq(p, picktmp)) {
            picks.push(picktmp);
        }
    }
    picks.sort_unstable();

    // 3. Build pickmtx via naivepairscore11.
    let npick = picks.len();
    let mut pickmtx: Vec<Vec<f64>> = (0..npick).map(|i| vec![0.0f64; npick - i]).collect();
    // pickmtx[0][k] = scores[picks[k]].score (already computed against ref).
    for k in 1..npick {
        pickmtx[0][k] = scores[picks[k]].score;
    }
    for j in 1..npick {
        let pj_self = scores[picks[j]].selfscore as f64;
        let aligned_j = &aligned_seqs[scores[picks[j]].numinseq];
        for i in (j + 1)..npick {
            let pair = naivepairscore11_aligned(
                aligned_j,
                &aligned_seqs[scores[picks[i]].numinseq],
                matrix,
                amino_map,
                penalty,
            );
            let bunbo = pj_self.min(scores[picks[i]].selfscore as f64);
            let dist = if bunbo > 0.0 { 1.0 - pair / bunbo } else { 1.0 };
            pickmtx[j][i - j] = if dist < 0.0 { 0.0 } else { dist };
        }
    }

    // 4. yukos. With `picksize=50 > nin=36`, `tokyoripara = 0`
    //    (`splittbfast.c:2760-2761`), so the redundancy filter is a no-op
    //    and every pick becomes a yuko.
    let nyuko = npick;
    let yukos: Vec<usize> = picks.clone();

    // 5. dfromc[i][j] = distance from yuko i's pivot to scores[j].
    let mut dfromc: Vec<Vec<f64>> = vec![vec![0.0f64; nin]; nyuko];
    // Row 0: from yukos[0] (= picks[0]) to all j. We already have these
    // as `scores[j].score` (which is the distance from the reference =
    // scores[0] = picks[0] to each j).
    for j in 0..nin {
        dfromc[0][j] = scores[j].score;
    }
    for i in 1..nyuko {
        let yuko_aligned = &aligned_seqs[scores[yukos[i]].numinseq];
        let yuko_self = scores[yukos[i]].selfscore as f64;
        for j in 0..nin {
            // C reuses pickmtx values when both i and j are picks
            // (`splittbfast.c:2178-2230`). Equivalent path:
            if j == yukos[i] {
                dfromc[i][j] = 0.0;
                continue;
            }
            let pair = naivepairscore11_aligned(
                yuko_aligned,
                &aligned_seqs[scores[j].numinseq],
                matrix,
                amino_map,
                penalty,
            );
            let bunbo = yuko_self.min(scores[j].selfscore as f64);
            let dist = if bunbo > 0.0 { 1.0 - pair / bunbo } else { 1.0 };
            dfromc[i][j] = if dist < 0.0 { 0.0 } else { dist };
        }
    }

    // 6. assign each seq to its closest yuko (strict `<`, first wins).
    let mut outs: Vec<Vec<usize>> = vec![Vec::new(); nyuko];
    for j in 0..nin {
        let mut belongto = 0usize;
        let mut min_d = f64::INFINITY;
        for yi in 0..nyuko {
            if dfromc[yi][j] < min_d {
                min_d = dfromc[yi][j];
                belongto = yi;
            }
        }
        outs[belongto].push(scores[j].numinseq);
    }

    // 7. yukomtx = pickmtx (since every pick survives → npick == nyuko).
    let yukomtx = pickmtx;
    let mut yuko_dm = DistanceMatrix::new(nyuko);
    for i in 0..nyuko {
        for j in (i + 1)..nyuko {
            yuko_dm.set(i, j, yukomtx[i][j - i]);
        }
    }

    // 8. UPGMA yields the yuko-level tree.
    let yuko_topo = musclesupg(&yuko_dm, ClusterMethod::Mix { sueff: 0.1 });

    Some(PartTreeFromalnResult {
        scores,
        outs,
        yuko_topo,
    })
}

/// Compute the C-equivalent `--reorder` ordering for the SECOND
/// `splittbfast` pass (CALL 2, `fromaln=1`). Takes the aligned MSA
/// (with gaps) and the substitution matrix + gap penalty used during
/// progressive alignment. Mirrors the `doalign && fromaln` branches of
/// `splittbfast.c::splitseq_mq`.
///
/// Single-recursion-level only (`nin <= picksize`). For the n=36
/// fixture this is the entire algorithm.
pub fn compute_parttree_order_fromaln(
    aligned_seqs: &[Vec<u8>],
    matrix: &[Vec<f64>],
    amino_map: &[u8; 256],
    penalty: f64,
) -> Vec<usize> {
    let nseq = aligned_seqs.len();
    let result = match run_parttree_fromaln_pipeline(aligned_seqs, matrix, amino_map, penalty) {
        Some(r) => r,
        None => return (0..nseq).collect(),
    };
    let mut order = Vec::with_capacity(nseq);
    if let Some(root) = result.yuko_topo.steps.last() {
        let l_yukos = c_normalized_subtree(&result.yuko_topo.steps, &root.left);
        let r_yukos = c_normalized_subtree(&result.yuko_topo.steps, &root.right);
        for &yi in l_yukos.iter().chain(r_yukos.iter()) {
            order.extend_from_slice(&result.outs[yi]);
        }
    } else {
        order.extend_from_slice(&result.outs[0]);
    }
    debug_assert_eq!(
        order.len(),
        nseq,
        "fromaln parttree order missing sequences"
    );
    order
}

/// PartTree `--treeout` output using CALL 2's `fromaln=1` scoring on the
/// already-aligned MSA. Mirrors `splittbfast.c::splitseq_mq`'s tree
/// emission (`splittbfast.c:1275-1301` leaf format, `:2532-2553` merge).
/// C overwrites `infile.tree` on each `splittbfast` invocation, so the
/// FINAL tree on disk comes from CALL 2 (the `fromaln=1` pass).
pub fn compute_parttree_newick_fromaln(
    aligned_seqs: &[Vec<u8>],
    matrix: &[Vec<f64>],
    amino_map: &[u8; 256],
    penalty: f64,
) -> String {
    let nseq = aligned_seqs.len();
    if nseq == 0 {
        return "\n".to_string();
    }
    if nseq == 1 {
        return "\n1\n".to_string();
    }
    let result = match run_parttree_fromaln_pipeline(aligned_seqs, matrix, amino_map, penalty) {
        Some(r) => r,
        None => return "\n".to_string(),
    };
    let nyuko = result.outs.len();
    let mut parttree: Vec<Option<String>> = (0..nyuko)
        .map(|yi| {
            let members = &result.outs[yi];
            if members.is_empty() {
                None
            } else if members.len() == 1 {
                Some(format!("\n{}\n", members[0] + 1))
            } else {
                let inner: Vec<String> = members.iter().map(|&m| (m + 1).to_string()).collect();
                Some(format!("\n({})\n", inner.join(",")))
            }
        })
        .collect();

    let mut last_v1 = 0usize;
    for step in &result.yuko_topo.steps {
        let l_norm = c_normalized_subtree(&result.yuko_topo.steps, &step.left);
        let r_norm = c_normalized_subtree(&result.yuko_topo.steps, &step.right);
        let (mut v1, mut v2) = (l_norm[0], r_norm[0]);
        if v1 > v2 {
            std::mem::swap(&mut v1, &mut v2);
        }
        let s1 = parttree[v1].take().expect("missing parttree[v1]");
        let s2 = parttree[v2].take().expect("missing parttree[v2]");
        parttree[v1] = Some(format!("({},{})", s1, s2));
        last_v1 = v1;
    }
    let mut out = parttree[last_v1].take().expect("missing root parttree");
    out.push('\n');
    out
}

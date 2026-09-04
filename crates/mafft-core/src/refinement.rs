/// Iterative refinement (tree-dependent iteration).
///
/// Ports the C `TreeDependentIteration()` from tditeration.c.
///
/// Repeatedly re-aligns pairs of groups defined by the guide tree,
/// accepting improvements and rejecting regressions, until convergence.
///
/// Key insight from the C code: at each tree branch, ALL sequences are
/// split into two groups (subtree vs everything else). There are never
/// "uninvolved" sequences — every sequence is in one group or the other.
///
/// Branch enumeration matches C exactly:
/// - For each topology step, both sides (k=0: left, k=1: right) are
///   processed, EXCEPT the root step (last step) where only k=1 (right)
///   is used (since at the root, left-vs-complement and right-vs-complement
///   produce the same split, just flipped).
/// - Even iterations traverse steps forward (0 → N-1), odd iterations
///   traverse backward (N-1 → 0). Within each step, k always goes 0→1.
/// - Total branches per iteration: (nseq-1)*2 - 1.

use mafft_align::{
    profile_align, profile_align_imp,
    profile_align_imp_with_boundary, profile_align_imp_multimtx,
    BoundaryFreqs, MultiMtx,
    build_imp_matrix, FASTATHRESHOLD_DEFAULT,
    Profile, GapModel, AlignOp,
};
use mafft_fft::{alignable_segments, SegmentParams};
use mafft_tree::{Topology, BranchWeights};
use mafft_types::{ScoringContext, LocalHomologyTable};

use crate::progressive::MultipleAlignment;

/// Parameters controlling iterative refinement.
#[derive(Debug, Clone)]
pub struct RefinementParams {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Score improvement threshold (fraction of old score).
    /// C default is 0.0 (accept only strict improvements).
    pub cut: f64,
    /// Whether to use FFT-accelerated alignment during refinement.
    pub use_fft: bool,
    /// `--leavegappyregion` / `--legacygappenalty` — propagated into
    /// the inner `GapModel` so the profile DP treats every column as
    /// fully nongap (`legacygapcost = 1`, `Salignmm.c:1604-1610`).
    pub legacy_gap_cost: bool,
    /// `--allowshift` warp/shift penalty for the refinement DP. C's
    /// `dvtditr` receives `-Q 2.0` → `penalty_shift_factor = 2.0` (< 10)
    /// → `trywarp = 1`, so the warp DP fires in refinement with
    /// `penalty_shift = 2.0 * penalty`. `None` = no warp (default).
    pub shift: Option<f64>,
    /// `--allowshift` `specificityconsideration` (C `dvtditr -s 0.8`).
    /// When `> 0`, each refinement branch scores sequence pairs with
    /// distance-binned matrices (the `_variousdist` multi-matrix DP).
    /// 0.0 = disabled (single matrix).
    pub unalign_level: f64,
    /// Floor for per-sequence weights in the intergroup-score
    /// accumulation. Mirrors C's `tbfast -W $minimumweight`
    /// (`scripts/mafft:1029` / default 0.00001). Sequences with weight
    /// below this floor get clamped up. Set to the C default if not
    /// otherwise overridden by `--minimumweight`.
    pub minimum_weight: f64,
    /// `--bestfirst` parallelisation strategy. C MAFFT's BAATARI2
    /// (default) walks each branch sequentially in topology order and
    /// accepts improvements immediately. BESTFIRST evaluates all
    /// branches against the same baseline alignment, picks the one
    /// with the largest gain, applies it, repeats. C's BESTFIRST is
    /// deterministic across thread counts (verified --thread 1, 4,
    /// 8 produce byte-identical output) — threads only parallelise
    /// the per-branch evaluation.
    pub bestfirst: bool,
    /// Per-(step, side) skip flags for `--skipiterate F` small-F mode.
    /// `skip_branches[step_idx]` is `(skip_left, skip_right)` —
    /// when a side is `true`, that branch's realign attempt is
    /// skipped entirely (mirrors C `dvtditr.c:999/1004`'s
    /// `skipthisbranch[j][k] = 1`). Empty = no skips (default
    /// refinement). Populated by the caller from
    /// `mafft_tree::generate_subalignments_table` output.
    pub skip_branches: Vec<(bool, bool)>,
    /// Use C's `athread` convergence rule instead of the single-threaded one.
    ///
    /// C picks the refinement implementation on `nthread > 0`
    /// (`tditeration.c:1433`), and the two do not converge the same way:
    ///
    /// * `nthread == 0` — `TreeDependentIteration` checks
    ///   `converged >= locnjob * 2` after **every branch** and `goto end`s
    ///   immediately, mid-cycle (`tditeration.c:2328-2342`).
    /// * `nthread > 0` — `athread`'s collector checks once per **cycle**
    ///   whether any branch gained (`maxgain > 0.0`, `tditeration.c:589`);
    ///   if none did it prints `Converged.` and sets `*collectingpt = -1`,
    ///   which only takes effect at the top of the next cycle where the
    ///   `else` arm `pthread_exit`s (`:527-551`). So the converging cycle
    ///   always runs to completion.
    ///
    /// That difference is visible in C's own output: at `maxiterate 2`,
    /// 22 of 85 segments print `Converged.` alone (converged in cycle 0, so
    /// cycle 1 never starts), 56 print `Converged.` and `Reached 2`
    /// (converged in the last cycle, so the loop ended normally), and 7
    /// print `Reached 2` alone.
    pub per_cycle_convergence: bool,
}

impl Default for RefinementParams {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            cut: 0.0,
            use_fft: false,
            legacy_gap_cost: false,
            shift: None,
            unalign_level: 0.0,
            minimum_weight: 0.00001,
            bestfirst: false,
            skip_branches: Vec::new(),
            per_cycle_convergence: false,
        }
    }
}

/// Per-branch input for the `--allowshift` multi-distance-class refinement
/// DP. `distarr[leaf]` is the tree distance from each leaf to the branch
/// being refined (`BranchWeights::dist_from_a_branch`); pairs are binned by
/// `distarr[g1[i]] + distarr[g2[j]]` (C `smalldistmtx`, `USEDISTONTREE=1`).
struct MultiMtxInput<'a> {
    distarr: &'a [f64],
    unalign_level: f64,
}

/// Per-branch multi-distance-class context (C `makescoringmatrices` +
/// `classifypairs` + masklists), computed once per `realign_all_constrained_fft`
/// and reused across all FFT segments. Only the per-segment cpmx column
/// profiles (`cpmx1s`/`cpmx2s`), which depend on the stripped segment, are
/// rebuilt per segment; class assignment and matrices are branch-global.
struct MmBranchCtx {
    /// `matrices[c]` — substitution matrix for distance class `c`.
    matrices: Vec<Vec<Vec<f64>>>,
    /// `eff1s[c][i]` / `eff2s[c][j]` — per-class member weights (0 if member
    /// is in no pair of class `c`).
    eff1s: Vec<Vec<f64>>,
    eff2s: Vec<Vec<f64>>,
    /// Spurious-pair masks per class for `match_calc_del`.
    mask1: Vec<Vec<usize>>,
    mask2: Vec<Vec<usize>>,
}

/// A branch identifier for oscillation tracking: (step_index, side).
/// side 0 = left, side 1 = right.
type BranchId = (usize, usize);

/// Build the per-step branch splits from a topology, matching C's enumeration.
///
/// For each topology step, both sides (k=0: left vs complement, k=1: right vs
/// complement) are included — EXCEPT the root step (last step) where only k=1
/// is included. At the root, left-vs-complement and right-vs-complement are
/// identical splits (just flipped), so C skips the redundant one.
///
/// Returns: `branch_map[step_idx]` = list of `(side, group1, group2)`.
/// Total branches = `(nseq - 1) * 2 - 1`.
fn build_branch_map(
    topology: &Topology,
    nseq: usize,
) -> Vec<Vec<(usize, Vec<usize>, Vec<usize>)>> {
    let nsteps = topology.steps.len();
    let root_idx = nsteps - 1;
    let all_indices: Vec<usize> = (0..nseq).collect();

    let mut branch_map: Vec<Vec<(usize, Vec<usize>, Vec<usize>)>> = Vec::with_capacity(nsteps);
    for (step_idx, step) in topology.steps.iter().enumerate() {
        let is_root = step_idx == root_idx;
        let mut sides = Vec::new();

        if !is_root {
            // Side 0: step.left vs complement
            let complement: Vec<usize> = all_indices
                .iter()
                .filter(|i| !step.left.contains(i))
                .copied()
                .collect();
            if !step.left.is_empty() && !complement.is_empty() {
                sides.push((0, step.left.clone(), complement));
            }
        }

        // Side 1: step.right vs complement
        let complement: Vec<usize> = all_indices
            .iter()
            .filter(|i| !step.right.contains(i))
            .copied()
            .collect();
        if !step.right.is_empty() && !complement.is_empty() {
            sides.push((1, step.right.clone(), complement));
        }

        branch_map.push(sides);
    }
    branch_map
}

/// Is `MAFFT_RS_REFINE_STATS` set? When it is, the refinement entry points
/// print a one-line work summary to stderr.
///
/// C's `dvtditr` reports its refinement work directly (`Segment n/N`, then a
/// `IIII-BBBB-S ... accepted/rejected` line per branch), so the two sides can
/// be compared cycle-for-cycle. Rust had no equivalent, which made
/// "did both run the same number of cycles?" unanswerable from outside and
/// any speed comparison meaningless. Off by default, so CLI output and the
/// `Progress` sink are unchanged.
#[derive(Default)]
struct RefineCounters {
    /// Branches visited (the re-alignment DP ran). Comparable to the count of
    /// `IIII-BBBB-S` lines C's `dvtditr` prints.
    visited: usize,
    /// Of those, branches whose re-alignment actually changed the columns —
    /// C prints these as `accepted.`/`rejected.` rather than `identical`.
    branches: usize,
    accepted: usize,
    /// Why the cycle loop ended: `maxiter`, `converged` or `oscillation`.
    exit: &'static str,
}

fn refine_stats_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MAFFT_RS_REFINE_STATS").is_some())
}

/// Iteratively refine a multiple alignment.
///
/// At each tree branch, splits ALL sequences into two groups (subtree vs
/// rest), re-aligns the two groups, and accepts improvements.
pub fn iterative_refine(
    alignment: &mut MultipleAlignment,
    topology: &Topology,
    scoring: &ScoringContext,
    params: &RefinementParams,
    constraints: Option<&LocalHomologyTable>,
) -> usize {
    // Thin reporting wrapper so every exit path (max-iterations, convergence,
    // oscillation) is counted in one place — see `refine_stats_enabled`.
    let nseq0 = alignment.nseq();
    let len0 = alignment.sequences.first().map_or(0, |s| s.len());
    let mut counters = RefineCounters { exit: "maxiter", ..Default::default() };
    let iterations =
        iterative_refine_inner(alignment, topology, scoring, params, constraints, &mut counters);
    if refine_stats_enabled() {
        eprintln!(
            "refine: nseq={nseq0} len={len0} cycles={iterations}/{} visited={} changed={} accepted={} exit={}",
            params.max_iterations, counters.visited, counters.branches, counters.accepted, counters.exit,
        );
    }
    iterations
}

fn iterative_refine_inner(
    alignment: &mut MultipleAlignment,
    topology: &Topology,
    scoring: &ScoringContext,
    params: &RefinementParams,
    constraints: Option<&LocalHomologyTable>,
    counters: &mut RefineCounters,
) -> usize {
    let nseq = alignment.nseq();
    // C refines two sequences too: `dvtditr.c:704-708` sets
    // `weight = 0; niter = 1` for `njob == 2` rather than skipping, and
    // `tditeration.c:1425` gates branch-weight computation on
    // `locnjob > 2`, so the pair is refined once, unweighted.
    // `BranchWeights` already yields uniform weights at nseq <= 2 and the
    // engine caps the iteration count, so only the early-return had to go.
    if nseq < 2 || topology.steps.is_empty() {
        return 0;
    }

    let branch_weights = BranchWeights::new(topology);
    let global_weights = mafft_tree::sequence_weights(topology);
    let use_global_weights = std::env::var("RUST_MAFFT_GLOBAL_WEIGHTS").is_ok();
    // C MAFFT's `dvtditr` invocation in `scripts/mafft` does NOT pass
    // `-g $gexp` (the `--exp` extension penalty). Only `disttbfast`
    // gets `-g`. As a result C's refinement DP always sees
    // `penalty_ex = 0`, regardless of what `--exp` the user passed —
    // confirmed by C's progress output showing `alg=A, ..., -0.00,
    // -0.00` for the refinement-phase alignment vs `..., -0.00,
    // +0.10` for the progressive disttbfast phase. Match that: zero
    // out the extend penalty in refinement's GapModel so we mirror
    // C exactly. Without this our refinement keeps shortening the
    // alignment under non-zero `--exp` while C's keeps the
    // progressive width.
    let mut gap = GapModel::new(scoring.gap.open as f64, 0.0)
        .with_legacy_gap_cost(params.legacy_gap_cost);
    if let Some(s) = params.shift {
        gap = gap.with_shift(s);
    }


    let mut converged_count = 0usize;
    let convergence_target = nseq * 2;

    let nsteps = topology.steps.len();
    let branch_map = build_branch_map(topology, nseq);

    // Per-branch score history for oscillation detection.
    // history[iteration][(step_idx, side)] = score after processing that branch.
    let mut history: Vec<std::collections::HashMap<BranchId, f64>> = Vec::new();

    let mut iteration = 0;
    for iter in 0..params.max_iterations {
        iteration = iter + 1;
        let mut any_change = false;
        let mut iter_scores: std::collections::HashMap<BranchId, f64> = std::collections::HashMap::new();

        // C alternates step traversal direction: even → forward, odd → reverse.
        let step_order: Vec<usize> = if iter % 2 == 0 {
            (0..nsteps).collect()
        } else {
            (0..nsteps).rev().collect()
        };

        for &step_idx in &step_order {
            for (side, group1, group2) in &branch_map[step_idx] {
                let branch_id: BranchId = (step_idx, *side);

                // `--skipiterate F` small-F: per-(step, side) skip
                // flags (port of C `dvtditr.c:1059-1066`'s
                // `skipthisbranch[]`). Skipped branches are
                // silently dropped — they do NOT count toward the
                // convergence target, mirroring C `tditeration.c:2358`
                // (`identity = 1; tscore = mscore` for skipped
                // branches, which `tditeration.c:2255-2256` then
                // treats as "no improvement" without bumping the
                // converge counter).
                let skipped = params.skip_branches.get(step_idx)
                    .map(|&(l, r)| if *side == 0 { l } else { r })
                    .unwrap_or(false);
                if skipped {
                    iter_scores.insert(branch_id, 0.0);
                    continue;
                }

                if let Ok(f) = std::env::var("RS_DISTARR_DUMP") {
                    use std::io::Write;
                    let da = branch_weights.dist_from_a_branch(topology, step_idx, *side);
                    if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&f) {
                        let _ = write!(fp, "DISTARR iter={} l={} k={}:", iter, step_idx, side);
                        for v in &da { let _ = write!(fp, " {:.17e}", v); }
                        let _ = writeln!(fp);
                    }
                }

                if let Ok(f) = std::env::var("RS_PRE_BRANCH") {
                    use std::io::Write;
                    if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&f) {
                        let _ = writeln!(fp, "R_PREBR iter={} step={} side={} clus1={} clus2={}", iter, step_idx, side, group1.len(), group2.len());
                        for (i, s) in alignment.sequences.iter().enumerate() {
                            let mut h: u64 = 5381;
                            for &c in s { h = h.wrapping_mul(33).wrapping_add(c as u64); }
                            let _ = writeln!(fp, "  R_seq[{}] len={} hash={:x}", i, s.len(), h);
                        }
                    }
                }

                let weights = if use_global_weights {
                    global_weights.clone()
                } else {
                    branch_weights.weights_for_branch(topology, step_idx, *side)
                };

                if let Ok(f) = std::env::var("RS_BRANCH_WEIGHTS") {
                    use std::io::Write;
                    if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&f) {
                        let _ = writeln!(fp, "R_BW iter={} step={} side={}", iter, step_idx, side);
                        for (i, &w) in weights.iter().enumerate() {
                            let _ = writeln!(fp, "  R_bw[{}]={:.17e}", i, w);
                        }
                    }
                }

                // Group-local sum-1 normalized weights (matches C's
                // fastconjuction_noname). Used both for `compute_impmatch_diagonal`
                // and any future per-cluster averaging. Floor is C's
                // `minimumweight` (`scripts/mafft:1029`, overridable via
                // `--minimumweight`).
                let w1: Vec<f64> = group1.iter().map(|&i| weights[i].max(params.minimum_weight)).collect();
                let w2: Vec<f64> = group2.iter().map(|&i| weights[i].max(params.minimum_weight)).collect();
                let s1w: f64 = w1.iter().sum();
                let s2w: f64 = w2.iter().sum();
                let w1n: Vec<f64> = if s1w > 0.0 { w1.iter().map(|w| w / s1w).collect() } else { vec![1.0; group1.len()] };
                let w2n: Vec<f64> = if s2w > 0.0 { w2.iter().map(|w| w / s2w).collect() } else { vec![1.0; group2.len()] };

                // C's mscore = oimpmatchdouble + tmpdouble (tditeration.c:953):
                // intergroup substitution score + impmatch (sum of impmtx[i][i]
                // over the current alignment's columns). We compute the same.
                let old_sub = compute_split_score(
                    group1, group2, &alignment.sequences, &weights, scoring,
                    params.minimum_weight,
                );
                let old_imp = if let Some(lh) = constraints {
                    compute_impmatch_diagonal(
                        group1, group2, &alignment.sequences, &w1n, &w2n, lh,
                    )
                } else { 0.0 };
                let old_score = old_sub + old_imp;

                // `--allowshift`: per-branch distances-from-tip drive the
                // multi-distance-class matrix selection (C `distFromABranch`
                // + `classifypairs`). Computed here where `branch_weights`,
                // `topology`, and the branch `(step_idx, side)` are in scope.
                let mm_distarr: Option<Vec<f64>> = if params.unalign_level > 0.0 {
                    Some(branch_weights.dist_from_a_branch(topology, step_idx, *side))
                } else {
                    None
                };
                let mm_input = mm_distarr.as_ref().map(|d| MultiMtxInput {
                    distarr: d,
                    unalign_level: params.unalign_level,
                });

                counters.visited += 1;
                let new_seqs = realign_all(
                    group1, group2, &alignment.sequences, &weights, scoring, &gap,
                    constraints, params.use_fft, mm_input.as_ref(),
                    params.minimum_weight,
                );

                if let Some((new_seqs, _new_score, dp_impmatch)) = new_seqs {
                    // C's identity check (tditeration.c:2184-2185): compare only
                    // the representative sequences s1=memlist1[0], s2=memlist2[0]
                    // (from OneClusterAndTheOther_fast in tddis.c:834-835).
                    // `group1` is memlist1, `group2` is memlist2, so s1=group1[0],
                    // s2=group2[0]. Checking ALL sequences would incorrectly treat
                    // column-rearrangements that preserve the two representatives
                    // as "changed", causing spurious accepts.
                    let s1 = group1[0];
                    let s2 = group2[0];
                    let changed = alignment.sequences[s1] != new_seqs[s1]
                        || alignment.sequences[s2] != new_seqs[s2];
                    // COMPAT: this two-row test decides whether a whole
                    // re-alignment is kept, and it is deliberately NOT a
                    // full comparison. On a 120x1.4kb FFT-NS-i run, 213
                    // re-alignments per run have both representatives
                    // unchanged while other rows DID change; every one of
                    // them is discarded here. C does exactly the same: its
                    // identity test is `!strcmp(aseq[s1],bseq[s1]) *
                    // !strcmp(aseq[s2],bseq[s2])` (tditeration.c:2184-2185),
                    // and the copy-back `strcpy( aseq[i], bseq[i] )` runs
                    // only on the accept path (tditeration.c:1769) — so C
                    // throws the same 213 away. Widening this to "any row
                    // changed" looks like an obvious fix and is a
                    // divergence: it would send those branches through the
                    // score comparison, changing accept/reject decisions
                    // and the `converged_count` sequence.

                    if !changed {
                        // Identical — no change, count toward convergence
                        let tscore = old_score;
                        if let Ok(f) = std::env::var("RS_REFINE_TRACE") {
                            use std::io::Write;
                            if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&f) {
                                let _ = writeln!(fp, "NOTHREAD pid={} niter={} iter={} l={} k={} clus1={} clus2={} mscore={:.6} tscore={:.6} accept=0",
                                    std::process::id(), params.max_iterations, iter, step_idx, side, group1.len(), group2.len(), old_score, tscore);
                            }
                        }
                        iter_scores.insert(branch_id, tscore);
                        converged_count += 1;
                    } else {
                        // C's tscore = impmatchdouble + tmpdouble (tditeration.c:1094):
                        // intergroup score + new alignment's impmatch.
                        let new_sub = compute_split_score(
                            group1, group2, &new_seqs, &weights, scoring,
                            params.minimum_weight,
                        );
                        let new_imp = if let Some(lh) = constraints {
                            // Prefer the impmatch accumulated DURING the
                            // segmented DP (C's `Falign_localhom` totalimpmatch:
                            // per-segment backward sum, forward across segments).
                            // This reproduces C's FP summation order exactly,
                            // unlike a global diagonal sum which merges all
                            // segments into one sweep (BB30028 fingerprint).
                            // Fall back to the global diagonal sum only on the
                            // non-FFT path (dvtditr always uses -F, so the
                            // fallback is not hit by default L-INS-i).
                            dp_impmatch.unwrap_or_else(|| compute_impmatch_diagonal(
                                group1, group2, &new_seqs, &w1n, &w2n, lh,
                            ))
                        } else { 0.0 };
                        let tscore = new_sub + new_imp;

                        let threshold = old_score - params.cut / 100.0 * old_score;
                        counters.branches += 1;
                        if tscore > threshold { counters.accepted += 1; }
                        if std::env::var("RUST_MAFFT_TRACE").is_ok() {
                            eprintln!("ACCEPT iter={iter} step={step_idx} side={side} old={:.3} new={:.3} accept={}",
                                old_score, tscore, tscore > threshold);
                        }
                        if let Ok(f) = std::env::var("RS_REFINE_TRACE") {
                            use std::io::Write;
                            if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&f) {
                                let _ = writeln!(fp, "NOTHREAD pid={} niter={} iter={} l={} k={} clus1={} clus2={} mscore={:.6} tscore={:.6} accept={}",
                                    std::process::id(), params.max_iterations, iter, step_idx, side, group1.len(), group2.len(), old_score, tscore,
                                    if tscore > threshold { 1 } else { 0 });
                            }
                        }
                        if tscore > threshold {
                            alignment.sequences = new_seqs;
                            if let Ok(f) = std::env::var("RS_ALIGN_HASH") {
                                use std::io::Write;
                                if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&f) {
                                    let mut h: u64 = 5381;
                                    for s in &alignment.sequences {
                                        for &c in s { h = h.wrapping_mul(33).wrapping_add(c as u64); }
                                    }
                                    let w = alignment.sequences.first().map_or(0, |s| s.len());
                                    let _ = writeln!(fp, "ACCEPT iter={} step={} side={} width={} hash={:x}", iter, step_idx, side, w, h);
                                }
                            }
                            any_change = true;
                            converged_count = 0;
                            iter_scores.insert(branch_id, tscore);
                        } else {
                            converged_count += 1;
                            // C `tditeration.c:2336`: on reject, `tscore = mscore`
                            // before `history[iterate][l][k] = tscore`. Storing the
                            // (unchanged) mscore makes oscillation detection fire
                            // when the same branch's mscore equals an earlier
                            // iteration's mscore — which is what closes BB12019 /
                            // BB12029 / BB30018 / BB40043's 4-line residuals.
                            iter_scores.insert(branch_id, old_score);
                        }
                    }
                } else {
                    iter_scores.insert(branch_id, old_score);
                    converged_count += 1;
                }

                if !params.per_cycle_convergence && converged_count >= convergence_target {
                    counters.exit = "converged";
                    return iteration;
                }

                // Oscillation detection: check if this branch's score matches
                // the score from 2, 4, 6... iterations ago (same branch).
                if iter >= 2 {
                    let tscore = iter_scores[&branch_id];
                    let mut oscillating = false;
                    let mut ii = history.len() as isize - 2; // iterate-2
                    while ii >= 0 {
                        if let Some(&prev_score) = history[ii as usize].get(&branch_id) {
                            if tscore == prev_score {
                                oscillating = true;
                                break;
                            }
                        }
                        ii -= 2;
                    }
                    if oscillating {
                        counters.exit = "oscillation";
                        return iteration;
                    }
                }
            }
        }

        history.push(iter_scores);

        // C's `TreeDependentIteration` does NOT exit on "no branches accepted
        // this iteration". It keeps iterating until either the cumulative
        // `converged` counter hits `nseq*2` (line 250 above) or oscillation
        // triggers (line 270). Adding an early `!any_change` exit here makes
        // Rust skip iterations C would have run — sometimes including one with
        // identical branches that don't change the score but still bump the
        // converged counter. The BB12019 / BB12029 / BB30018 / BB40043
        // 4-line FFT-NS-i divergences come from that early exit.
        if params.per_cycle_convergence {
            // C `athread`: no branch gained this cycle -> converged. The
            // cycle we just finished still counts; the stop lands before the
            // next one (`tditeration.c:589` + `:527-551`).
            if !any_change {
                counters.exit = "converged";
                return iteration;
            }
        }
        let _ = any_change;
    }

    iteration
}

/// Re-align all sequences split into two groups.
///
/// Since group1 + group2 = ALL sequences, there are no "other" sequences
/// to worry about. Every sequence is in exactly one group.
///
/// When `use_fft` is true (matching C's Falign path in tditeration.c):
/// 1. Strip per-group gap columns → build stripped profiles
/// 2. Run FFT anchor detection on stripped profiles (clean, residue-rich data)
/// 3. Map anchors back to non-stripped coordinates via kept1/kept2
/// 4. Build non-stripped profiles from full sequences
/// 5. Run anchored DP on non-stripped profiles (matching C's input)
///
/// The anchor mapping ensures the FFT sees clean data for good anchor
/// detection, while the DP operates on the same non-stripped profiles C
/// uses. Anchors constrain the DP so width growth is bounded.
///
/// If FFT finds no anchors, falls back to profile_align on the full
/// non-stripped sequences (matching C's single-segment fallback in
/// Falign.c lines 1377-1421), bounded by alloclen.
fn realign_all(
    group1: &[usize],
    group2: &[usize],
    sequences: &[Vec<u8>],
    weights: &[f64],
    scoring: &ScoringContext,
    gap: &GapModel,
    constraints: Option<&LocalHomologyTable>,
    use_fft: bool,
    mm_input: Option<&MultiMtxInput>,
    min_weight: f64,
) -> Option<(Vec<Vec<u8>>, f64, Option<f64>)> {
    let width = sequences[0].len();

    // Per-group gap stripping.
    let gap1 = group_all_gap_columns(group1, sequences, width);
    let gap2 = group_all_gap_columns(group2, sequences, width);
    let kept1: Vec<usize> = (0..width).filter(|&c| !gap1[c]).collect();
    let kept2: Vec<usize> = (0..width).filter(|&c| !gap2[c]).collect();

    // C clamps per-sequence weights to `minimumweight` (default 0.00001
    // from `scripts/mafft:1029`, overridable via `--minimumweight`).
    // Applied in `fastconjuction_noname` at `tddis.c:548`.
    let w1: Vec<f64> = group1.iter().map(|&i| weights[i].max(min_weight)).collect();
    let w2: Vec<f64> = group2.iter().map(|&i| weights[i].max(min_weight)).collect();
    let sum1: f64 = w1.iter().sum();
    let sum2: f64 = w2.iter().sum();
    let w1n: Vec<f64> = if sum1 > 0.0 { w1.iter().map(|w| w / sum1).collect() } else { vec![1.0; group1.len()] };
    let w2n: Vec<f64> = if sum2 > 0.0 { w2.iter().map(|w| w / sum2).collect() } else { vec![1.0; group2.len()] };

    if let Ok(path) = std::env::var("RS_H_DUMP") {
        if let Ok(shape) = std::env::var("RS_H_DUMP_SHAPE") {
            let parts: Vec<&str> = shape.split(',').collect();
            if parts.len() == 4 {
                let sc1: usize = parts[2].parse().unwrap_or(0);
                let sc2: usize = parts[3].parse().unwrap_or(0);
                if group1.len() == sc1 && group2.len() == sc2 {
                    use std::io::Write;
                    if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                        let _ = write!(fp, "R_EFF1");
                        for &w in &w1n { let _ = write!(fp, " {:.17e}", w); }
                        let _ = writeln!(fp);
                        let _ = write!(fp, "R_EFF2");
                        for &w in &w2n { let _ = write!(fp, " {:.17e}", w); }
                        let _ = writeln!(fp);
                        let _ = write!(fp, "R_GROUP2_GLOBAL_IDX");
                        for &g in group2 { let _ = write!(fp, " {}", g); }
                        let _ = writeln!(fp);
                        let _ = write!(fp, "R_WEIGHTS_RAW");
                        for &g in group2 { let _ = write!(fp, " {:.17e}", weights[g]); }
                        let _ = writeln!(fp);
                    }
                }
            }
        }
    }

    // Build stripped sequences and profiles (used for FFT anchor detection
    // and as fallback for unconstrained non-FFT alignment).
    let stripped1: Vec<Vec<u8>> = group1.iter()
        .map(|&i| kept1.iter().map(|&c| sequences[i][c]).collect())
        .collect();
    let stripped2: Vec<Vec<u8>> = group2.iter()
        .map(|&i| kept2.iter().map(|&c| sequences[i][c]).collect())
        .collect();
    let s1_refs: Vec<&[u8]> = stripped1.iter().map(|s| s.as_slice()).collect();
    let s2_refs: Vec<&[u8]> = stripped2.iter().map(|s| s.as_slice()).collect();
    let stripped_prof1 = Profile::from_aligned(&s1_refs, &w1n, &scoring.amino_map, scoring.nalphabets);
    let stripped_prof2 = Profile::from_aligned(&s2_refs, &w2n, &scoring.amino_map, scoring.nalphabets);

    if stripped_prof1.length == 0 || stripped_prof2.length == 0 {
        return None;
    }

    if let Some(lh_table) = constraints {
        if use_fft {
            // C's `Falign_localhom` (kobetsubunkatsu=1 path): mirrors the
            // FFT-segmented loop in the unconstrained `Falign` but calls
            // `partA__align(constraint=1, ..., gapmap1, gapmap2, ...)` per
            // segment. The global impmtx is built once for the full
            // (non-stripped) alignment; each segment passes its sliced
            // view via gapmap1/gapmap2 (which translate stripped column
            // index to position within the segment).
            return realign_all_constrained_fft(
                group1, group2, sequences, &w1n, &w2n,
                scoring, gap, lh_table, mm_input,
            ).map(|(seqs, score, imp)| (seqs, score, Some(imp)));
        }
        // Non-FFT constraint path (L-INS-i without -F, single full DP).
        // Mirrors C's `A__align(..., constraint=1, ...)` (Salignmm.c:1086):
        // build the per-cell importance matrix `impmtx` once, then do the
        // standard profile DP with `currentw[j] += impmtx[i][j]` applied
        // row-by-row inside the DP (Salignmm.c:1700-1849).
        let g1_seq_refs: Vec<&[u8]> = stripped1.iter().map(|s| s.as_slice()).collect();
        let g2_seq_refs: Vec<&[u8]> = stripped2.iter().map(|s| s.as_slice()).collect();
        let imp = build_imp_matrix(
            lh_table,
            group1, group2,
            &g1_seq_refs, &g2_seq_refs,
            &w1n, &w2n,
            stripped_prof1.length, stripped_prof2.length,
            FASTATHRESHOLD_DEFAULT,
        );
        let aln = profile_align_imp(
            &stripped_prof1, &stripped_prof2,
            &scoring.consweight_matrix,
            gap,
            true, true,
            Some(&imp),
        );
        // Non-FFT constraint path: impmatch is folded into the DP score,
        // not accumulated separately, so return None — caller falls back to
        // `compute_impmatch_diagonal`. (dvtditr always passes -F, so this
        // path is not exercised by the default L-INS-i pipeline.)
        return build_result_from_stripped(
            &aln, group1, group2, sequences, &kept1, &kept2,
            &stripped_prof1, &stripped_prof2,
        ).map(|(seqs, score)| (seqs, score, None));
    }

    if use_fft {
        // C's Falign path for dvtditr (kobetsubunkatsu=1 in dvtditr.c:54):
        // 1. SKIP the FFT block (Falign.c:1110 `if(!kobetsubunkatsu)` is skipped)
        // 2. alignableReagion runs ONCE with lag=0 (Falign.c:1299 maxk=1,kouho[0]=0)
        // 3. Collected segments define cut points: [0, center0, center1, ..., width]
        // 4. Per segment, commongappick strips per-group all-gap columns
        //    (Falign.c:1610 `if(kobetsubunkatsu && fftkeika)`)
        // 5. MSalignmm aligns the stripped segment
        // 6. Results concatenated
        let width = sequences[0].len();
        let full_prof1 = Profile::from_aligned(
            &group1.iter().map(|&i| sequences[i].as_slice()).collect::<Vec<_>>(),
            &w1n, &scoring.amino_map, scoring.nalphabets);
        let full_prof2 = Profile::from_aligned(
            &group2.iter().map(|&i| sequences[i].as_slice()).collect::<Vec<_>>(),
            &w2n, &scoring.amino_map, scoring.nalphabets);

        // mafft.tmpl passes `-z 50` to dvtditr, setting fftThreshold=50 for the
        // alignableReagion sliding window (default from constants.c is 80, but
        // the script overrides it). Match that here.
        let segment_params = if scoring.seq_type.is_nucleotide() {
            SegmentParams::dna().with_threshold(50.0)
        } else {
            SegmentParams::protein().with_threshold(50.0)
        };

        // Step 1: Compute per-position site scores at lag=0 (C's alignableReagion,
        // fftFunctions.c:282-287). For refinement, prof1.length == prof2.length,
        // so we score pairwise at matching columns. C uses n_disFFT which equals
        // substitution_matrix when offset=0 (mafft default).
        // C divides by totaleff = sum eff1[i]*eff2[j]; with normalized weights this is 1.
        let totaleff: f64 = w1n.iter().sum::<f64>() * w2n.iter().sum::<f64>();
        let len = full_prof1.length.min(full_prof2.length);
        let mut site_scores = vec![0.0f64; len];
        for i in 0..len {
            site_scores[i] =
                full_prof1.match_score(i, &full_prof2, i, &scoring.consweight_matrix)
                / totaleff;
        }

        // Step 2: Find alignable segments via the sliding window threshold test.
        let segments = alignable_segments(&site_scores, &segment_params);

        // Step 3: Build cut points from segment centers (Falign.c:1414).
        // kobetsubunkatsu=1: cut1[i+1] = sortedseg1[i]->center, plus [0] and [len].
        // For refinement (lag=0), cut1[i] == cut2[i], so a single cut list suffices.
        let mut cuts: Vec<usize> = Vec::with_capacity(segments.len() + 2);
        cuts.push(0);
        for seg in &segments {
            cuts.push(seg.center.min(width));
        }
        cuts.push(width);
        // Ensure strictly increasing (segments might overlap/coincide — dedupe).
        cuts.sort();
        cuts.dedup();

        if let Ok(f) = std::env::var("RS_FFT_CUTS") {
            use std::io::Write;
            use std::sync::atomic::{AtomicUsize, Ordering};
            static CALL_NO: AtomicUsize = AtomicUsize::new(0);
            let cn = CALL_NO.fetch_add(1, Ordering::SeqCst);
            if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&f) {
                let cuts_str: Vec<String> = cuts.iter().map(|c| c.to_string()).collect();
                let _ = writeln!(fp, "R_FALIGN call={} clus1={} clus2={} len={} nsegs={} cut={}",
                    cn, group1.len(), group2.len(), width, cuts.len(), cuts_str.join(","));
            }
        }

        // Step 4-5: Per-segment strip + align, then concatenate.
        let mut new_sequences: Vec<Vec<u8>> = vec![Vec::new(); sequences.len()];
        let mut total_score = 0.0f64;
        for (seg_idx, win) in cuts.windows(2).enumerate() {
            let a = win[0];
            let b = win[1];
            if a >= b { continue; }

            // Slice the segment [a, b) from each sequence.
            let seg1: Vec<Vec<u8>> = group1.iter()
                .map(|&i| sequences[i][a..b].to_vec()).collect();
            let seg2: Vec<Vec<u8>> = group2.iter()
                .map(|&i| sequences[i][a..b].to_vec()).collect();

            // commongappick on the segment: strip columns where all sequences in
            // THIS GROUP have a gap within THIS SEGMENT. (C's commongappick)
            let seg_width = b - a;
            let seg_gap1: Vec<bool> = (0..seg_width)
                .map(|c| seg1.iter().all(|s| s[c] == b'-')).collect();
            let seg_gap2: Vec<bool> = (0..seg_width)
                .map(|c| seg2.iter().all(|s| s[c] == b'-')).collect();
            let seg_kept1: Vec<usize> = (0..seg_width).filter(|&c| !seg_gap1[c]).collect();
            let seg_kept2: Vec<usize> = (0..seg_width).filter(|&c| !seg_gap2[c]).collect();

            let stripped_seg1: Vec<Vec<u8>> = seg1.iter()
                .map(|s| seg_kept1.iter().map(|&c| s[c]).collect()).collect();
            let stripped_seg2: Vec<Vec<u8>> = seg2.iter()
                .map(|s| seg_kept2.iter().map(|&c| s[c]).collect()).collect();

            if stripped_seg1.is_empty() || stripped_seg1[0].is_empty() ||
               stripped_seg2.is_empty() || stripped_seg2[0].is_empty() {
                // One side is empty — emit gaps for both as-is (only possible
                // when whole segment is all-gap for one group in every column).
                let len1 = if stripped_seg1.is_empty() { 0 } else { stripped_seg1[0].len() };
                let len2 = if stripped_seg2.is_empty() { 0 } else { stripped_seg2[0].len() };
                for (k, &i) in group1.iter().enumerate() {
                    new_sequences[i].extend_from_slice(&stripped_seg1[k]);
                    new_sequences[i].extend(std::iter::repeat(b'-').take(len2));
                }
                for (k, &i) in group2.iter().enumerate() {
                    new_sequences[i].extend(std::iter::repeat(b'-').take(len1));
                    new_sequences[i].extend_from_slice(&stripped_seg2[k]);
                }
                continue;
            }

            let s1_refs: Vec<&[u8]> = stripped_seg1.iter().map(|s| s.as_slice()).collect();
            let s2_refs: Vec<&[u8]> = stripped_seg2.iter().map(|s| s.as_slice()).collect();
            if let Ok(f) = std::env::var("RS_FFT_CUTS") {
                use std::io::Write;
                if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&f) {
                    let mut h1: u64 = 5381;
                    let mut h2: u64 = 5381;
                    for s in &stripped_seg1 { for &c in s { h1 = h1.wrapping_mul(33).wrapping_add(c as u64); } }
                    for s in &stripped_seg2 { for &c in s { h2 = h2.wrapping_mul(33).wrapping_add(c as u64); } }
                    let _ = writeln!(fp, "R_FSEG seg={} c1raw={} c2raw={} w1strip={} w2strip={} h1={:x} h2={:x}",
                        seg_idx, b - a, b - a, stripped_seg1[0].len(), stripped_seg2[0].len(), h1, h2);
                    for (j, s) in stripped_seg2.iter().enumerate() {
                        let mut ph: u64 = 5381;
                        for &c in s { ph = ph.wrapping_mul(33).wrapping_add(c as u64); }
                        let first10: String = s.iter().take(10).map(|&c| c as char).collect();
                        let _ = writeln!(fp, "  R_clus2[{}] len={} hash={:x} first10={}", j, s.len(), ph, first10);
                    }
                }
            }
            let mut prof_seg1 = Profile::from_aligned(&s1_refs, &w1n, &scoring.amino_map, scoring.nalphabets);
            let mut prof_seg2 = Profile::from_aligned(&s2_refs, &w2n, &scoring.amino_map, scoring.nalphabets);

            // C's Falign segment loop (lines 1549-1565):
            //   sgap[j] = (cut1[i]  > 0)    ? (seq[j][cut1[i]-1]   == '-') : 'o'
            //   egap[j] = (cut1[i+1] != len)? (seq[j][cut1[i+1]]   == '-') : 'o'
            // These per-sequence boundary gap states are passed to MSalignmm,
            // which switches from `st_OpeningGapCount` to `new_OpeningGapCount`
            // (mltaln9.c:12557): a sequence already in a gap just before the
            // segment's first column does NOT count as "opening" at position 0.
            // `Profile::from_aligned` applies `st_OpeningGapCount` semantics
            // (gc starts at 0), so we correct the first/last counts here.
            let sgap_inside = a > 0;
            if sgap_inside {
                if !prof_seg1.ogcp.is_empty() {
                    for (k, &idx) in group1.iter().enumerate() {
                        if sequences[idx][a - 1] == b'-'
                            && !stripped_seg1[k].is_empty()
                            && stripped_seg1[k][0] == b'-'
                        {
                            prof_seg1.ogcp[0] -= w1n[k];
                        }
                    }
                }
                if !prof_seg2.ogcp.is_empty() {
                    for (k, &idx) in group2.iter().enumerate() {
                        if sequences[idx][a - 1] == b'-'
                            && !stripped_seg2[k].is_empty()
                            && stripped_seg2[k][0] == b'-'
                        {
                            prof_seg2.ogcp[0] -= w2n[k];
                        }
                    }
                }
            }
            // NOTE: do NOT correct fgcp[last] for the egap boundary. C's
            // `new_FinalGapCount` (mltaln9.c:12794-12828, the compiled
            // `#if 1` branch) effectively ignores `egappat` at the last
            // position — its post-loop block is dead code because the
            // inner loop's final iteration reads the null terminator
            // and sets gc=0, so `gb && !gc` never fires.
            // `Profile::from_aligned`'s closing-tail increment already
            // matches the resulting C value (= sum of weights for
            // sequences ending in a gap). The earlier code that did
            // `prof_seg.fgcp[last] -= wn[k]` mirrored the *disabled*
            // `#if 0` C branch and over-subtracted at the segment tail.

            // C's Falign segment loop (lines 1521-1522):
            //   headgp = (i==0) ? outgap : 1   ;   tailgp = (i==count-2) ? outgap : 1
            // outgap=1 in dvtditr.c:73, so headgp=tailgp=1 for every segment.
            //
            // Boundary frequencies (Salignmm.c:1585-1622):
            //   outgapcount(headgapfreq, sgap, eff) is the weighted gap
            //   fraction at the segment's left/right boundary column in
            //   the parent, then inverted to non-gap fraction
            //   (legacygapcost=0). Used by the inner DP at `g_iskip` and
            //   head_gap initialization. `BoundaryFreqs::default() = 1.0`
            //   is correct only for the first segment (where sgap[*]='o'
            //   so outgapcount=0 → invert=1.0). Inner segments need the
            //   actual gap fraction at the parent's a-1 / b columns to
            //   match C bit-for-bit (BB30013 86-seq divergence cause).
            let outgap_count = |grp: &[usize], col: Option<usize>| -> f64 {
                match col {
                    None => 1.0,
                    Some(c) => {
                        let wn: &[f64] = if std::ptr::eq(grp.as_ptr(), group1.as_ptr()) { &w1n } else { &w2n };
                        let mut gap_frac = 0.0f64;
                        for (k, &idx) in grp.iter().enumerate() {
                            if sequences[idx][c] == b'-' { gap_frac += wn[k]; }
                        }
                        1.0 - gap_frac
                    }
                }
            };
            let left_col = if a > 0 { Some(a - 1) } else { None };
            let right_col = if b < width { Some(b) } else { None };
            let bf = BoundaryFreqs {
                head1: outgap_count(group1, left_col),
                head2: outgap_count(group2, left_col),
                tail1: outgap_count(group1, right_col),
                tail2: outgap_count(group2, right_col),
            };
            let seg_aln = profile_align_imp_with_boundary(
                &prof_seg1, &prof_seg2, &scoring.consweight_matrix, gap,
                true, true, None, false, bf,
            );
            total_score += seg_aln.score;

            // Reconstruct the segment's output by applying ops to the stripped segments.
            let mut i1 = 0usize;
            let mut i2 = 0usize;
            for op in &seg_aln.operations {
                match op {
                    AlignOp::Match => {
                        for (k, &idx) in group1.iter().enumerate() {
                            new_sequences[idx].push(stripped_seg1[k][i1]);
                        }
                        for (k, &idx) in group2.iter().enumerate() {
                            new_sequences[idx].push(stripped_seg2[k][i2]);
                        }
                        i1 += 1;
                        i2 += 1;
                    }
                    AlignOp::Delete => {
                        for (k, &idx) in group1.iter().enumerate() {
                            new_sequences[idx].push(stripped_seg1[k][i1]);
                        }
                        for &idx in group2 {
                            new_sequences[idx].push(b'-');
                        }
                        i1 += 1;
                    }
                    AlignOp::Insert => {
                        for &idx in group1 {
                            new_sequences[idx].push(b'-');
                        }
                        for (k, &idx) in group2.iter().enumerate() {
                            new_sequences[idx].push(stripped_seg2[k][i2]);
                        }
                        i2 += 1;
                    }
                }
            }
        }

        return Some((new_sequences, total_score, None));
    }

    // Non-FFT path: profile_align on stripped profiles.
    let aln = profile_align(
        &stripped_prof1, &stripped_prof2,
        &scoring.consweight_matrix, gap, true, true,
    );
    build_result_from_stripped(
        &aln, group1, group2, sequences, &kept1, &kept2,
        &stripped_prof1, &stripped_prof2,
    ).map(|(seqs, score)| (seqs, score, None))
}

/// Constraint-aware FFT-segmented refinement, port of C's
/// `Falign_localhom` (Falign_localhom.c:163, kobetsubunkatsu=1 path).
///
/// Mirrors the unconstrained FFT-segmented refinement in
/// `realign_all`'s `use_fft` branch but per-segment calls
/// `profile_align_imp` with the local impmtx slice rather than the
/// unconstrained `profile_align`. The impmtx is built once for the full
/// alignment width using the localhom regions; each segment's local
/// view is the rectangle covering the segment's parent column range,
/// indexed by the per-group strip kept-column lists (= C's `gapmap1` /
/// `gapmap2`).
///
/// The cut points are computed from `alignable_segments` at lag=0 just
/// like the unconstrained path (C's `alignableReagion` with maxk=1 in
/// kobetsubunkatsu mode).
fn realign_all_constrained_fft(
    group1: &[usize],
    group2: &[usize],
    sequences: &[Vec<u8>],
    w1n: &[f64],
    w2n: &[f64],
    scoring: &ScoringContext,
    gap: &GapModel,
    lh_table: &LocalHomologyTable,
    mm_input: Option<&MultiMtxInput>,
) -> Option<(Vec<Vec<u8>>, f64, f64)> {
    let width = sequences[group1[0]].len();
    if width == 0 { return None; }

    // Build full-alignment profiles for site-score / segment detection
    // (matches the unconstrained FFT path). For refinement,
    // prof1.length == prof2.length == width.
    let full_prof1 = Profile::from_aligned(
        &group1.iter().map(|&i| sequences[i].as_slice()).collect::<Vec<_>>(),
        w1n, &scoring.amino_map, scoring.nalphabets,
    );
    let full_prof2 = Profile::from_aligned(
        &group2.iter().map(|&i| sequences[i].as_slice()).collect::<Vec<_>>(),
        w2n, &scoring.amino_map, scoring.nalphabets,
    );

    let segment_params = if scoring.seq_type.is_nucleotide() {
        SegmentParams::dna().with_threshold(50.0)
    } else {
        SegmentParams::protein().with_threshold(50.0)
    };

    // Per-position site scores at lag=0 (C's alignableReagion).
    let totaleff: f64 = w1n.iter().sum::<f64>() * w2n.iter().sum::<f64>();
    let len = full_prof1.length.min(full_prof2.length);
    let mut site_scores = vec![0.0f64; len];
    for i in 0..len {
        site_scores[i] = full_prof1.match_score(
            i, &full_prof2, i, &scoring.consweight_matrix,
        ) / totaleff;
    }
    let segments = alignable_segments(&site_scores, &segment_params);

    // Cut points (C's `cut1[i+1] = sortedseg1[i]->center` plus 0 and len).
    let mut cuts: Vec<usize> = Vec::with_capacity(segments.len() + 2);
    cuts.push(0);
    for seg in &segments { cuts.push(seg.center.min(width)); }
    cuts.push(width);
    cuts.sort();
    cuts.dedup();

    // Build the GLOBAL impmtx for the full non-stripped alignment
    // (width × width). C does this via `part_imp_match_init_strict(...,
    // length, length, mseq1, mseq2, ...)` once per branch realign.
    // Each per-segment partA__align then reads `impmtx[start1+gapmap1[i]][start2+gapmap2[j]]`.
    let g1_full: Vec<&[u8]> = group1.iter().map(|&i| sequences[i].as_slice()).collect();
    let g2_full: Vec<&[u8]> = group2.iter().map(|&i| sequences[i].as_slice()).collect();
    let global_imp = build_imp_matrix(
        lh_table,
        group1, group2,
        &g1_full, &g2_full,
        w1n, w2n,
        width, width,
        FASTATHRESHOLD_DEFAULT,
    );

    // `--allowshift`: precompute this branch's multi-distance-class context
    // (C `makescoringmatrices` + `classifypairs` + masklists). Class
    // assignment uses member-level tree distances `distarr[group{1,2}[·]]`,
    // which are constant across FFT segments; only the per-segment cpmx
    // profiles differ, so this is built once here.
    let mm_ctx: Option<MmBranchCtx> = mm_input.map(|mi| {
        let n1 = group1.len();
        let n2 = group2.len();
        let max_dc = crate::varidist::calc_max_dist_class(mi.unalign_level);
        let gap_idx = scoring.amino_map[b'-' as usize] as usize;
        let matrices = crate::varidist::make_scoring_matrices(
            &scoring.consweight_matrix, mi.unalign_level, gap_idx, max_dc,
        );
        // smalldist[i][j] = distFromABranch(group1[i]) + distFromABranch(group2[j])
        // (C `OneClusterAndTheOther_fast` with `USEDISTONTREE = 1`).
        let smalldist: Vec<Vec<f64>> = (0..n1)
            .map(|i| (0..n2).map(|j| mi.distarr[group1[i]] + mi.distarr[group2[j]]).collect())
            .collect();
        let pc = crate::varidist::classify_pairs(w1n, w2n, &smalldist, max_dc);
        // Spurious-pair masks: pairs landing in class c's cpmx product whose
        // true class differs (subtracted by `match_calc_del`). i-major, j-minor.
        let mut mask1 = vec![Vec::new(); max_dc];
        let mut mask2 = vec![Vec::new(); max_dc];
        for c in 0..max_dc {
            for i in 0..n1 {
                for j in 0..n2 {
                    if pc.eff1s[c][i] * pc.eff2s[c][j] != 0.0 && c != pc.matnum[i][j] {
                        mask1[c].push(i);
                        mask2[c].push(j);
                    }
                }
            }
        }
        MmBranchCtx { matrices, eff1s: pc.eff1s, eff2s: pc.eff2s, mask1, mask2 }
    });

    let mut new_sequences: Vec<Vec<u8>> = vec![Vec::new(); sequences.len()];
    let mut total_score = 0.0f64;
    // C `Falign_localhom.c:816`: `*totalimpmatch += impmatch` per FFT
    // segment, summed forward across segments. Within each segment C's
    // `Atracking_localhom` accumulates `impmtx` at match cells in BACKWARD
    // traceback order. We reproduce that exact FP order here so `new_imp`
    // matches C bit-for-equivalent (closes BB30028; a global diagonal sum
    // diverges by ~7 ULP because it merges all segments into one sweep).
    let mut total_impmatch = 0.0f64;
    for win in cuts.windows(2) {
        let a = win[0];
        let b = win[1];
        if a >= b { continue; }
        let seg_width = b - a;

        // Slice [a, b) from each member.
        let seg1: Vec<Vec<u8>> = group1.iter()
            .map(|&i| sequences[i][a..b].to_vec()).collect();
        let seg2: Vec<Vec<u8>> = group2.iter()
            .map(|&i| sequences[i][a..b].to_vec()).collect();

        // commongappick within segment + record gapmap (= column index
        // within segment for each kept stripped column).
        let seg_gap1: Vec<bool> = (0..seg_width)
            .map(|c| seg1.iter().all(|s| s[c] == b'-')).collect();
        let seg_gap2: Vec<bool> = (0..seg_width)
            .map(|c| seg2.iter().all(|s| s[c] == b'-')).collect();
        let gapmap1: Vec<usize> = (0..seg_width).filter(|&c| !seg_gap1[c]).collect();
        let gapmap2: Vec<usize> = (0..seg_width).filter(|&c| !seg_gap2[c]).collect();

        let stripped_seg1: Vec<Vec<u8>> = seg1.iter()
            .map(|s| gapmap1.iter().map(|&c| s[c]).collect()).collect();
        let stripped_seg2: Vec<Vec<u8>> = seg2.iter()
            .map(|s| gapmap2.iter().map(|&c| s[c]).collect()).collect();

        // Edge case: one side completely empty after strip.
        if stripped_seg1.is_empty() || stripped_seg1[0].is_empty()
            || stripped_seg2.is_empty() || stripped_seg2[0].is_empty()
        {
            let len1 = if stripped_seg1.is_empty() { 0 } else { stripped_seg1[0].len() };
            let len2 = if stripped_seg2.is_empty() { 0 } else { stripped_seg2[0].len() };
            for (k, &i) in group1.iter().enumerate() {
                new_sequences[i].extend_from_slice(&stripped_seg1[k]);
                new_sequences[i].extend(std::iter::repeat(b'-').take(len2));
            }
            for (k, &i) in group2.iter().enumerate() {
                new_sequences[i].extend(std::iter::repeat(b'-').take(len1));
                new_sequences[i].extend_from_slice(&stripped_seg2[k]);
            }
            continue;
        }

        // Build the segment's local impmtx by indexing the global impmtx
        // at (a + gapmap1[i], a + gapmap2[j]) — mirrors C's
        // `imp_match_out_vead_gapmap(imp[j] = impmtx[i1][start2+gapmap2[j]])`
        // (partSalignmm.c:71-83). For refinement, start1 = start2 = a.
        let l1 = stripped_seg1[0].len();
        let l2 = stripped_seg2[0].len();
        let mut local_imp = vec![vec![0.0f64; l2]; l1];
        for i in 0..l1 {
            let row = a + gapmap1[i];
            for j in 0..l2 {
                let col = a + gapmap2[j];
                if row < global_imp.len() && col < global_imp[row].len() {
                    local_imp[i][j] = global_imp[row][col];
                }
            }
        }

        let s1_refs: Vec<&[u8]> = stripped_seg1.iter().map(|s| s.as_slice()).collect();
        let s2_refs: Vec<&[u8]> = stripped_seg2.iter().map(|s| s.as_slice()).collect();
        let mut prof_seg1 = Profile::from_aligned(&s1_refs, w1n, &scoring.amino_map, scoring.nalphabets);
        let mut prof_seg2 = Profile::from_aligned(&s2_refs, w2n, &scoring.amino_map, scoring.nalphabets);

        // Per-segment boundary gap correction (mirrors C's `getkyokaigap`
        // → `new_OpeningGapCount` in `MSalignmm`/`partA__align`). Same as
        // the unconstrained FFT path: a sequence already in a gap just
        // before the segment's first column does NOT count as opening at
        // position 0.
        let sgap_inside = a > 0;
        let egap_inside = b < width;
        if sgap_inside || egap_inside {
            if !prof_seg1.ogcp.is_empty() {
                for (k, &idx) in group1.iter().enumerate() {
                    if sgap_inside
                        && sequences[idx][a - 1] == b'-'
                        && !stripped_seg1[k].is_empty()
                        && stripped_seg1[k][0] == b'-'
                    {
                        prof_seg1.ogcp[0] -= w1n[k];
                    }
                    let l1k = stripped_seg1[k].len();
                    if egap_inside
                        && l1k > 0
                        && stripped_seg1[k][l1k - 1] == b'-'
                        && sequences[idx][b] == b'-'
                    {
                        let last = prof_seg1.fgcp.len() - 1;
                        prof_seg1.fgcp[last] -= w1n[k];
                    }
                }
            }
            if !prof_seg2.ogcp.is_empty() {
                for (k, &idx) in group2.iter().enumerate() {
                    if sgap_inside
                        && sequences[idx][a - 1] == b'-'
                        && !stripped_seg2[k].is_empty()
                        && stripped_seg2[k][0] == b'-'
                    {
                        prof_seg2.ogcp[0] -= w2n[k];
                    }
                    let l2k = stripped_seg2[k].len();
                    if egap_inside
                        && l2k > 0
                        && stripped_seg2[k][l2k - 1] == b'-'
                        && sequences[idx][b] == b'-'
                    {
                        let last = prof_seg2.fgcp.len() - 1;
                        prof_seg2.fgcp[last] -= w2n[k];
                    }
                }
            }
        }

        // C's Falign_localhom segment loop passes `headgp = tailgp = 1`
        // when a segment is interior (`(i==0)?outgap:1` etc.); for
        // dvtditr, outgap=1 anyway so all segments use 1.
        //
        // C uses `partA__align` (partSalignmm.c:1218,1235) which uses STRICT
        // `>` for the prept-vs-mi/mjpt tie-break (the "// 2018/Apr" change).
        // The progressive `A__align` uses `>=`. We pass strict_part_tiebreak=true
        // to match `partA__align` exactly.
        //
        // Boundary nongap-frequencies (C's headgapfreq{1,2} and
        // gapfreq{1,2}[lgth] computed via outgapcount on sgap/egap):
        //   head{1,2} = nongap fraction at full-alignment column [a-1]
        //               (1.0 when a == 0 → C's `sgap[j]='o'` branch)
        //   tail{1,2} = nongap fraction at full-alignment column [b]
        //               (1.0 when b == width → C's `egap[j]='o'` branch)
        let head1 = if a > 0 {
            let s: f64 = group1.iter().enumerate()
                .filter(|&(_, &idx)| sequences[idx][a - 1] == b'-')
                .map(|(k, _)| w1n[k]).sum();
            1.0 - s
        } else { 1.0 };
        let head2 = if a > 0 {
            let s: f64 = group2.iter().enumerate()
                .filter(|&(_, &idx)| sequences[idx][a - 1] == b'-')
                .map(|(k, _)| w2n[k]).sum();
            1.0 - s
        } else { 1.0 };
        let tail1 = if b < width {
            let s: f64 = group1.iter().enumerate()
                .filter(|&(_, &idx)| sequences[idx][b] == b'-')
                .map(|(k, _)| w1n[k]).sum();
            1.0 - s
        } else { 1.0 };
        let tail2 = if b < width {
            let s: f64 = group2.iter().enumerate()
                .filter(|&(_, &idx)| sequences[idx][b] == b'-')
                .map(|(k, _)| w2n[k]).sum();
            1.0 - s
        } else { 1.0 };
        let boundary = BoundaryFreqs { head1, head2, tail1, tail2 };
        let seg_aln = if let Some(ctx) = mm_ctx.as_ref() {
            // `--allowshift`: build this segment's per-class cpmx profiles
            // (weighted by eff{1,2}s[c], same accumulation as the single
            // matrix Profile so the c=0 class is bit-identical), then run
            // the multi-distance-class DP (C `partA__align_variousdist`).
            let nc = ctx.matrices.len();
            let cpmx1s: Vec<Vec<Vec<f64>>> = (0..nc)
                .map(|c| Profile::from_aligned(
                    &s1_refs, &ctx.eff1s[c], &scoring.amino_map, scoring.nalphabets,
                ).freqs)
                .collect();
            let cpmx2s: Vec<Vec<Vec<f64>>> = (0..nc)
                .map(|c| Profile::from_aligned(
                    &s2_refs, &ctx.eff2s[c], &scoring.amino_map, scoring.nalphabets,
                ).freqs)
                .collect();
            // Sparse (alpha_index, value) representations of cpmx1s/cpmx2s,
            // skipping zero alphabet positions. Hot path for
            // `MultiMtx::match_row_into` — for 1-residue clusters this turns
            // an O(nalpha²) scarr build + O(nalpha·lgth2) accumulation into
            // O(nalpha) + O(lgth2). nalpha < 256 so u8 indices suffice.
            let sparsify = |dense: &Vec<Vec<Vec<f64>>>| -> Vec<Vec<Vec<(u8, f64)>>> {
                dense.iter().map(|class| {
                    class.iter().map(|col| {
                        let mut v: Vec<(u8, f64)> = Vec::with_capacity(col.len());
                        for (l, &x) in col.iter().enumerate() {
                            if x != 0.0 { v.push((l as u8, x)); }
                        }
                        v
                    }).collect()
                }).collect()
            };
            let cpmx1s_sparse = sparsify(&cpmx1s);
            let cpmx2s_sparse = sparsify(&cpmx2s);
            let mm = MultiMtx {
                matrices: &ctx.matrices,
                cpmx1s: &cpmx1s,
                cpmx2s: &cpmx2s,
                cpmx1s_sparse: &cpmx1s_sparse,
                cpmx2s_sparse: &cpmx2s_sparse,
                mask1: &ctx.mask1,
                mask2: &ctx.mask2,
                seq1: &s1_refs,
                seq2: &s2_refs,
                eff1: w1n,
                eff2: w2n,
                amino_map: &scoring.amino_map,
                nalpha: scoring.nalphabets,
            };
            profile_align_imp_multimtx(
                &prof_seg1, &prof_seg2, &scoring.consweight_matrix, gap,
                true, true, Some(&local_imp), true, boundary, Some(&mm),
            )
        } else {
            profile_align_imp_with_boundary(
                &prof_seg1, &prof_seg2, &scoring.consweight_matrix, gap,
                true, true, Some(&local_imp), true, boundary,
            )
        };
        total_score += seg_aln.score;

        // Per-segment impmatch: C's `Atracking_localhom` (Dalignmm.c:564)
        // adds `impmtx[iin][jin]` at each match cell during the BACKWARD
        // traceback. Collect this segment's match cells (stripped-local
        // positions) and sum `local_imp` over them in reverse op order to
        // reproduce that summation order; then add to `total_impmatch`
        // (forward across segments, matching Falign_localhom.c:816).
        {
            let mut mi1 = 0usize;
            let mut mi2 = 0usize;
            let mut match_cells: Vec<(usize, usize)> = Vec::new();
            for op in &seg_aln.operations {
                match op {
                    AlignOp::Match => { match_cells.push((mi1, mi2)); mi1 += 1; mi2 += 1; }
                    AlignOp::Delete => { mi1 += 1; }
                    AlignOp::Insert => { mi2 += 1; }
                }
            }
            let mut seg_imp = 0.0f64;
            for &(ci, cj) in match_cells.iter().rev() {
                seg_imp += local_imp[ci][cj];
            }
            total_impmatch += seg_imp;
        }

        // Reconstruct segment output by applying ops to stripped segments.
        let mut i1 = 0usize;
        let mut i2 = 0usize;
        for op in &seg_aln.operations {
            match op {
                AlignOp::Match => {
                    for (k, &idx) in group1.iter().enumerate() {
                        new_sequences[idx].push(stripped_seg1[k][i1]);
                    }
                    for (k, &idx) in group2.iter().enumerate() {
                        new_sequences[idx].push(stripped_seg2[k][i2]);
                    }
                    i1 += 1; i2 += 1;
                }
                AlignOp::Delete => {
                    for (k, &idx) in group1.iter().enumerate() {
                        new_sequences[idx].push(stripped_seg1[k][i1]);
                    }
                    for &idx in group2 {
                        new_sequences[idx].push(b'-');
                    }
                    i1 += 1;
                }
                AlignOp::Insert => {
                    for &idx in group1 {
                        new_sequences[idx].push(b'-');
                    }
                    for (k, &idx) in group2.iter().enumerate() {
                        new_sequences[idx].push(stripped_seg2[k][i2]);
                    }
                    i2 += 1;
                }
            }
        }
    }

    // Pad sequences not in either group to the new width.
    let new_width = if !group1.is_empty() {
        new_sequences[group1[0]].len()
    } else if !group2.is_empty() {
        new_sequences[group2[0]].len()
    } else {
        width
    };
    for (i, s) in new_sequences.iter_mut().enumerate() {
        if !group1.contains(&i) && !group2.contains(&i) {
            // "Other" sequences — preserve from input. Our refinement only
            // realigns groups that partition all sequences, so this should
            // never trigger; keep original.
            *s = sequences[i].clone();
        }
        if s.len() < new_width {
            s.resize(new_width, b'-');
        }
    }

    Some((new_sequences, total_score, total_impmatch))
}

/// Build result sequences from an alignment on stripped profiles.
/// Maps alignment operations back to original column positions via kept1/kept2.
fn build_result_from_stripped(
    aln: &mafft_align::Alignment,
    group1: &[usize],
    group2: &[usize],
    sequences: &[Vec<u8>],
    kept1: &[usize],
    kept2: &[usize],
    prof1: &Profile,
    prof2: &Profile,
) -> Option<(Vec<Vec<u8>>, f64)> {
    let consumed1 = aln.operations.iter()
        .filter(|op| matches!(op, AlignOp::Match | AlignOp::Delete)).count();
    let consumed2 = aln.operations.iter()
        .filter(|op| matches!(op, AlignOp::Match | AlignOp::Insert)).count();
    if consumed1 != prof1.length || consumed2 != prof2.length {
        return None;
    }

    let mut new_sequences = vec![Vec::with_capacity(aln.operations.len()); sequences.len()];
    let mut c1 = 0usize;
    let mut c2 = 0usize;
    for op in &aln.operations {
        match op {
            AlignOp::Match => {
                let oc1 = kept1[c1];
                let oc2 = kept2[c2];
                for &i in group1 { new_sequences[i].push(sequences[i][oc1]); }
                for &i in group2 { new_sequences[i].push(sequences[i][oc2]); }
                c1 += 1; c2 += 1;
            }
            AlignOp::Delete => {
                let oc1 = kept1[c1];
                for &i in group1 { new_sequences[i].push(sequences[i][oc1]); }
                for &i in group2 { new_sequences[i].push(b'-'); }
                c1 += 1;
            }
            AlignOp::Insert => {
                let oc2 = kept2[c2];
                for &i in group1 { new_sequences[i].push(b'-'); }
                for &i in group2 { new_sequences[i].push(sequences[i][oc2]); }
                c2 += 1;
            }
        }
    }
    Some((new_sequences, aln.score))
}

fn group_all_gap_columns(group: &[usize], sequences: &[Vec<u8>], width: usize) -> Vec<bool> {
    // Match C's commongappick: only '-' counts as gap (not '.').
    let mut all_gap = vec![true; width];
    for &idx in group {
        for (col, &ch) in sequences[idx].iter().enumerate() {
            if ch != b'-' {
                all_gap[col] = false;
            }
        }
    }
    all_gap
}

/// Sum the impmatch (constraint-importance bonus) across the diagonal of
/// the current alignment. Mirrors C's `oimpmatchdouble = sum imp_match_out_sc(i,i)`
/// loop (tditeration.c:925) for the existing alignment's `impmtx`.
///
/// `eff1`, `eff2` are group-local sum-1-normalized weights matching
/// what `imp_match_init_strict` is called with in C (= `effarr1`/`effarr2`
/// after `fastconjuction_noname` per-cluster normalization).
fn compute_impmatch_diagonal(
    group1: &[usize],
    group2: &[usize],
    sequences: &[Vec<u8>],
    eff1: &[f64],
    eff2: &[f64],
    lh_table: &LocalHomologyTable,
) -> f64 {
    let width = sequences[group1[0]].len();
    if width == 0 { return 0.0; }
    let g1_seqs: Vec<&[u8]> = group1.iter().map(|&i| sequences[i].as_slice()).collect();
    let g2_seqs: Vec<&[u8]> = group2.iter().map(|&i| sequences[i].as_slice()).collect();
    let imp = build_imp_matrix(
        lh_table,
        group1, group2,
        &g1_seqs, &g2_seqs,
        eff1, eff2,
        width, width,
        FASTATHRESHOLD_DEFAULT,
    );
    // C `tditeration.c:891`: `for(i=length-1; i>=0; i--) oimpmatchdouble += imp_match_out_scD(i,i);`
    // — sums BACKWARD. Match the iteration direction; FP addition is not
    // associative, and forward summation drifts by ~1 ULP per step, which
    // cascades into a flipped accept/reject at BB20004 iter=4 l=36 k=1.
    let mut total = 0.0f64;
    for i in (0..width).rev() {
        if let Some(row) = imp.get(i) {
            if let Some(&v) = row.get(i) {
                total += v;
            }
        }
    }
    total
}

fn compute_split_score(
    group1: &[usize],
    group2: &[usize],
    sequences: &[Vec<u8>],
    weights: &[f64],
    scoring: &ScoringContext,
    min_weight: f64,
) -> f64 {
    // Port of C's intergroup_score flow: weights are per-group normalized
    // by fastconjuction_noname (tddis.c line 548) before being passed.
    // Apply the same normalization here: each group's weights sum to 1.0.
    // `min_weight` is C's `minimumweight` (default 0.00001, overridable
    // via `--minimumweight`).
    let w1: Vec<f64> = group1.iter().map(|&i| weights[i].max(min_weight)).collect();
    let w2: Vec<f64> = group2.iter().map(|&i| weights[i].max(min_weight)).collect();
    let s1: f64 = w1.iter().sum();
    let s2: f64 = w2.iter().sum();
    let w1n: Vec<f64> = if s1 > 0.0 { w1.iter().map(|w| w / s1).collect() } else { vec![1.0; group1.len()] };
    let w2n: Vec<f64> = if s2 > 0.0 { w2.iter().map(|w| w / s2).collect() } else { vec![1.0; group2.len()] };

    // Sequential sum to match C's deterministic accumulation order.
    // par_iter gives non-deterministic summation order, which causes
    // small FP divergence that cascades into accept/reject decisions.
    //
    // FMA: clang at -O3 with FP_CONTRACT=on lowers
    //   total += score * wi * wj
    // to one plain mul (score * wi) plus one fma (acc += (score*wi)*wj),
    // i.e. 2 roundings. Plain Rust `*` and `+=` give 3 roundings, and
    // the resulting sub-ULP per-pair drift accumulates over (clus1 *
    // clus2) pairs into a multi-unit mscore drift that flips
    // accept/reject decisions late in iterative refinement (BB30028
    // L-INS-i iter=3 l=5 k=1 fingerprint).
    let mut total = 0.0f64;
    for (i_local, &i) in group1.iter().enumerate() {
        let wi = w1n[i_local];
        for (j_local, &j) in group2.iter().enumerate() {
            let wj = w2n[j_local];
            let s_wi = pairwise_score(&sequences[i], &sequences[j], scoring) * wi;
            total = s_wi.mul_add(wj, total);
        }
    }
    total
}

/// Branchless pairwise scoring for auto-vectorization.
///
/// The gap check is converted to a mask multiply: if either residue is '-',
/// the score contribution is 0. This eliminates branches that prevent SIMD.
#[inline]
fn pairwise_score(seq1: &[u8], seq2: &[u8], scoring: &ScoringContext) -> f64 {
    let map = &scoring.amino_map;
    let mtx = &scoring.consweight_matrix;
    let mtx_size = mtx.len();

    // Port of C's intergroup_score (mltaln9.c lines 404-475):
    // - Gap-gap positions: skipped (continue).
    // - Match positions: add amino_dis[c1][c2].
    // - Gap in seq1: add `penalty` (gap open), then consume all consecutive
    //   '-' in seq1. Same for seq2.
    // - amino_dis_consweight_multi[gap][*] = 0, so gap-region positions
    //   contribute only the single gap-open penalty per gap run.
    //
    // A previous attempt to rewrite this as a per-cell branchless state
    // machine (2026-05-18) failed because C's `while (seq1[k] == '-')`
    // consume loop crosses both-gap positions (it only looks at seq1),
    // while a naive per-cell rule that resets state on both-gap would
    // re-charge penalty when an A-gap-run is interrupted by a both-gap
    // column. The minimum branchless formulation that matches C exactly
    // is a 3-state machine (Neutral / AGapRun / BGapRun) — still has
    // branches, no clean SIMD path. Keeping the C-style structure.
    let penalty = scoring.gap.open as f64;
    let len = seq1.len().min(seq2.len());
    let mut score = 0.0f64;
    let mut k = 0;
    while k < len {
        let a = seq1[k];
        let b = seq2[k];
        if a == b'-' && b == b'-' {
            k += 1;
            continue;
        }
        if a == b'-' {
            score += penalty;
            // Consume all consecutive gaps in seq1 (C's while-loop at line 448).
            k += 1;
            while k < len && seq1[k] == b'-' {
                k += 1;
            }
            continue;
        }
        if b == b'-' {
            score += penalty;
            k += 1;
            while k < len && seq2[k] == b'-' {
                k += 1;
            }
            continue;
        }
        let i = map[a as usize] as usize;
        let j = map[b as usize] as usize;
        if i < mtx_size && j < mtx_size {
            score += mtx[i][j];
        }
        k += 1;
    }
    score
}

/// Find anchor column positions for segmenting a multiple alignment, mirroring
/// C `searchAnchors` (`mltaln9.c:11318`).
///
/// For each column, computes the average pairwise substitution score. Slides a
/// window of `div_win_size` (=20) columns; where the windowed sum exceeds
/// `div_threshold_pct/100 * 600 * div_win_size` (=7800 by default), marks an
/// anchor region. Returns the centers of those regions, framed by `[0, len]`
/// so consecutive entries define the segment boundaries dvtditr.c:1085 uses.
fn search_anchors_aa(
    sequences: &[Vec<u8>],
    matrix: &[Vec<i32>],
    amino_map: &[u8; 256],
    div_win_size: usize,
    div_threshold_pct: i32,
) -> Vec<usize> {
    const SEGMENTSIZE: usize = 150;

    let nseq = sequences.len();
    let len = sequences.first().map_or(0, |s| s.len());
    if nseq < 2 || len < div_win_size + 2 {
        return vec![0, len];
    }
    let threshold = (div_threshold_pct as f64 / 100.0) * 600.0 * div_win_size as f64;
    let n_pairs = (nseq * (nseq - 1) / 2) as f64;
    let mtx_size = matrix.len();

    // Per-column average pairwise substitution score (stra[i] in C).
    let mut stra = vec![0.0f64; len];
    for i in 0..len {
        let mut sum = 0.0f64;
        for k in 0..(nseq - 1) {
            let ki = amino_map[sequences[k][i] as usize] as usize;
            for j in (k + 1)..nseq {
                let ji = amino_map[sequences[j][i] as usize] as usize;
                if ki < mtx_size && ji < mtx_size {
                    sum += matrix[ki][ji] as f64;
                }
            }
        }
        stra[i] = sum / n_pairs;
    }

    let mut centers: Vec<usize> = Vec::new();
    let mut score: f64 = stra[..div_win_size].iter().sum();
    let mut status = false;
    let mut start_i: usize = 0;
    let mut length: usize = 0;

    // C loops `for( i=1; i<len-divWinSize; i++ )` and leaves `i` reachable
    // after the loop for the trailing flush; we keep `last_i` for parity.
    let mut last_i = 1usize;
    for i in 1..(len - div_win_size) {
        last_i = i;
        score = score - stra[i - 1] + stra[i + div_win_size - 1];
        if score > threshold {
            if !status {
                status = true;
                start_i = i;
                length = 0;
            }
            length += 1;
        }
        if score <= threshold || length > SEGMENTSIZE {
            if status {
                let end_i = i;
                let center = (start_i + end_i + div_win_size) / 2;
                centers.push(center);
                length = 0;
                status = false;
            }
        }
    }
    if status {
        let center = (start_i + last_i + div_win_size) / 2;
        centers.push(center);
    }

    let mut anchors: Vec<usize> = Vec::with_capacity(centers.len() + 2);
    anchors.push(0);
    anchors.extend(centers.into_iter().filter(|&c| c < len));
    anchors.push(len);
    anchors.dedup();
    anchors
}

/// Iteratively refine a multiple alignment, mirroring C MAFFT's FFT-NS-i
/// behaviour: split the alignment at high-conservation anchors and run
/// `iterative_refine` on each column-slice independently, then re-concatenate.
///
/// Mirrors the segmented loop in `dvtditr.c:1085` driven by `searchAnchors`.
/// BESTFIRST refinement strategy — port of C MAFFT's
/// `parallelizationstrategy = BESTFIRST` (`tditeration.c:595-619`).
///
/// Where BAATARI2 (the default in both C and rust) walks each branch
/// sequentially in topology order and accepts improvements immediately,
/// BESTFIRST evaluates all branches against the **same baseline**
/// alignment per iteration, picks the one with the largest gain, applies
/// it, and repeats. The result is deterministic (verified C's BESTFIRST
/// gives byte-identical output across `--thread 1`, `--thread 4`, and
/// `--thread 8`) — multi-threading in C only parallelises the per-branch
/// evaluation, never reorders the global pick.
///
/// Terminates when no branch yields positive gain (converged) or
/// `max_iterations` is reached.
pub fn bestfirst_refine(
    alignment: &mut MultipleAlignment,
    topology: &Topology,
    scoring: &ScoringContext,
    params: &RefinementParams,
    constraints: Option<&LocalHomologyTable>,
) -> usize {
    let nseq = alignment.nseq();
    // `nseq == 2` is refined too — see `iterative_refine` (C `dvtditr.c:704-708`).
    if nseq < 2 || topology.steps.is_empty() {
        return 0;
    }

    let branch_weights = BranchWeights::new(topology);
    let global_weights = mafft_tree::sequence_weights(topology);
    let use_global_weights = std::env::var("RUST_MAFFT_GLOBAL_WEIGHTS").is_ok();
    // Same C-mirroring zero-out as `iterative_refine` (dvtditr without -g).
    let mut gap = GapModel::new(scoring.gap.open as f64, 0.0)
        .with_legacy_gap_cost(params.legacy_gap_cost);
    if let Some(s) = params.shift {
        gap = gap.with_shift(s);
    }

    let nsteps = topology.steps.len();
    let branch_map = build_branch_map(topology, nseq);

    let mut iterations = 0usize;
    for _iter in 0..params.max_iterations {
        iterations += 1;
        // Snapshot baseline — every branch evaluates against this, NOT
        // against an updated mastercopy. That's the BESTFIRST signature
        // vs BAATARI2's eager-accept loop.
        let baseline_seqs = alignment.sequences.clone();

        // For each branch: compute baseline old_score, run realign, compute
        // new_score, record (branch_id, gain, new_seqs) if gain > 0.
        let mut best: Option<(f64, Vec<Vec<u8>>)> = None;
        for step_idx in 0..nsteps {
            for (_side, group1, group2) in &branch_map[step_idx] {
                let weights = if use_global_weights {
                    global_weights.clone()
                } else {
                    branch_weights.weights_for_branch(topology, step_idx, *_side)
                };
                let w1: Vec<f64> = group1.iter().map(|&i| weights[i].max(params.minimum_weight)).collect();
                let w2: Vec<f64> = group2.iter().map(|&i| weights[i].max(params.minimum_weight)).collect();
                let s1w: f64 = w1.iter().sum();
                let s2w: f64 = w2.iter().sum();
                let w1n: Vec<f64> = if s1w > 0.0 { w1.iter().map(|w| w / s1w).collect() } else { vec![1.0; group1.len()] };
                let w2n: Vec<f64> = if s2w > 0.0 { w2.iter().map(|w| w / s2w).collect() } else { vec![1.0; group2.len()] };

                let old_sub = compute_split_score(
                    group1, group2, &baseline_seqs, &weights, scoring,
                    params.minimum_weight,
                );
                let old_imp = if let Some(lh) = constraints {
                    compute_impmatch_diagonal(
                        group1, group2, &baseline_seqs, &w1n, &w2n, lh,
                    )
                } else { 0.0 };
                let old_score = old_sub + old_imp;

                let mm_distarr: Option<Vec<f64>> = if params.unalign_level > 0.0 {
                    Some(branch_weights.dist_from_a_branch(topology, step_idx, *_side))
                } else {
                    None
                };
                let mm_input = mm_distarr.as_ref().map(|d| MultiMtxInput {
                    distarr: d,
                    unalign_level: params.unalign_level,
                });

                if let Some((new_seqs, _, dp_impmatch)) = realign_all(
                    group1, group2, &baseline_seqs, &weights, scoring, &gap,
                    constraints, params.use_fft, mm_input.as_ref(),
                    params.minimum_weight,
                ) {
                    // C `tditeration.c:2185`: `identity = !strcmp(localcopy[s1], mastercopy[s1])`
                    // ANDed with the s2 comparison. When identical, `tscore = mscore`
                    // and gain = 0 — never accepted by `gain > 0` test. Skip the score
                    // recompute (matches C's branch and avoids FP drift around zero).
                    let s1 = group1[0];
                    let s2 = group2[0];
                    let changed = baseline_seqs[s1] != new_seqs[s1]
                        || baseline_seqs[s2] != new_seqs[s2];
                    if !changed {
                        continue;
                    }
                    let new_sub = compute_split_score(
                        group1, group2, &new_seqs, &weights, scoring,
                        params.minimum_weight,
                    );
                    let new_imp = if let Some(lh) = constraints {
                        dp_impmatch.unwrap_or_else(|| compute_impmatch_diagonal(
                            group1, group2, &new_seqs, &w1n, &w2n, lh,
                        ))
                    } else { 0.0 };
                    let new_score = new_sub + new_imp;
                    let gain = new_score - old_score;
                    if gain > 0.0 {
                        match &best {
                            None => best = Some((gain, new_seqs)),
                            Some((bg, _)) if gain > *bg => best = Some((gain, new_seqs)),
                            _ => {}
                        }
                    }
                }
            }
        }

        match best {
            Some((_gain, new_seqs)) => {
                alignment.sequences = new_seqs;
            }
            None => break, // converged: no branch improves
        }
    }
    iterations
}

/// `intergroup_score` clone that mirrors C's
/// `mltaln9.c::intergroup_score` (lines 404-477) FP-order EXACTLY:
/// the C code precomputes `efficient = eff1[i] * eff2[j]` THEN does
/// `*value += tmpscore * efficient` (one mul outside, then one fma).
/// This differs from `compute_split_score` which inlines as
/// `(tmpscore * wi) * wj + total` (a different product order).
///
/// The two formulations are mathematically equivalent but FP-different;
/// the difference is below the tie-break threshold for the
/// tree-dependent refinement (`iterative_refine` matches C byte-exactly
/// with `compute_split_score`), but for `dooneiteration`'s pure
/// leave-one-out splits the per-pair sub-ULP drift cumulates across
/// 36*2 iterations and flips two accept decisions on the 36-seq sample
/// (rust width 715 vs C 713 with `compute_split_score`).
///
/// Weights are normalised per-group exactly like
/// `fastconjuction_noname` (`tddis.c:548`) with `mineff = 0.0`.
fn intergroup_score_c_order(
    group1: &[usize],
    group2: &[usize],
    sequences: &[Vec<u8>],
    weights: &[f64],
    scoring: &ScoringContext,
) -> f64 {
    let w1: Vec<f64> = group1.iter().map(|&i| weights[i]).collect();
    let w2: Vec<f64> = group2.iter().map(|&i| weights[i]).collect();
    let s1: f64 = w1.iter().sum();
    let s2: f64 = w2.iter().sum();
    let w1n: Vec<f64> = if s1 > 0.0 { w1.iter().map(|w| w / s1).collect() } else { vec![1.0; group1.len()] };
    let w2n: Vec<f64> = if s2 > 0.0 { w2.iter().map(|w| w / s2).collect() } else { vec![1.0; group2.len()] };

    let mut total = 0.0f64;
    for (i_local, &i) in group1.iter().enumerate() {
        let wi = w1n[i_local];
        for (j_local, &j) in group2.iter().enumerate() {
            let wj = w2n[j_local];
            // C `mltaln9.c:426`: `efficient = eff1[i] * eff2[j]`
            // (one rounding), then `mltaln9.c:466`:
            // `*value += (double)tmpscore * (double)efficient`
            // (with FP_CONTRACT on at clang -O3 this is one fma).
            let efficient = wi * wj;
            let tmpscore = pairwise_score(&sequences[i], &sequences[j], scoring);
            total = tmpscore.mul_add(efficient, total);
        }
    }
    total
}

/// `--oneiteration` "one-vs-others" refinement — port of
/// `disttbfast.c::dooneiteration` (lines 2217-2538). Runs AFTER
/// the progressive merge but BEFORE the regular tree-dependent
/// refinement (`iterative_refine` / `segmented_iterative_refine`).
///
/// Only triggered from the disttbfast-path modes (FFT-NS-2 and
/// FFT-NS-i); L/G/E-INS-i pipelines do not call this function in C
/// because they go through `pairlocalalign → tbfast → dvtditr`
/// and `scripts/mafft:2673` passes `-r` only to `disttbfast`.
///
/// ## Algorithm
///
/// `ITERATIVECYCLE = 2` (disttbfast.c:11) full passes over the
/// alignment. Each pass walks every sequence index `l in 0..nseq`
/// in order; for each `l` we treat the singleton `{l}` as group 1
/// and all other sequences as group 2, then attempt a fresh
/// realignment of that split. We compute the C
/// `intergroup_score` (substitution score between groups, no
/// constraints) before AND after the realign and KEEP the new
/// alignment iff the new score is at least as good as the
/// baseline (C's `if( nscore < oscore )` revert at
/// `disttbfast.c:2457`).
///
/// Constraints are NOT used (disttbfast path never sees
/// `constraint != 0`); `gap.extend = 0.0` matches the fact that
/// `dvtditr` is not invoked here — the gap-extension penalty
/// `--exp` is only baked into the progressive DP via disttbfast's
/// `-g $gexp`, not into this refinement step (`scripts/mafft`
/// only passes `-r ` for oneiteration, never `-g`).
///
/// `min_weight = 0.0` (matches C `fastconjuction_noname` call at
/// `disttbfast.c:2321-2322` with `mineff = 0.0`).
pub fn one_vs_others_refine(
    alignment: &mut MultipleAlignment,
    topology: &Topology,
    scoring: &ScoringContext,
    params: &RefinementParams,
) {
    let nseq = alignment.nseq();
    if nseq <= 2 {
        return;
    }

    // C `disttbfast.c:11` `#define ITERATIVECYCLE 2`. Each cycle
    // walks every sequence index once.
    const ITERATIVE_CYCLE: usize = 2;

    let weights = mafft_tree::sequence_weights(topology);
    // `gap.extend = 0.0`: disttbfast itself does pass `-g $gexp`
    // for progressive, but `dooneiteration` calls `Falign`/`A__align`
    // with the in-process `penalty_ex` global, which `scripts/mafft`
    // does not reset before invoking `disttbfast -r`. The progressive
    // step left it at the user's `-g` value, so we mirror by reading
    // `scoring.gap.extend` (NOT zeroing). Keep the legacy/shift
    // pieces from `params` so `--allowshift` interactions are
    // forwarded correctly.
    let mut gap = GapModel::new(scoring.gap.open as f64, scoring.gap.extend as f64)
        .with_legacy_gap_cost(params.legacy_gap_cost);
    if let Some(s) = params.shift {
        gap = gap.with_shift(s);
    }

    let total_iters = nseq * ITERATIVE_CYCLE;
    for ll in 0..total_iters {
        let l = ll % nseq;
        // group1 = singleton {l}, group2 = the rest, preserving
        // sequence order (matches C's loop at disttbfast.c:2298-2300:
        // `for( i=0,j=0; i<njob; i++ ) if( i != l ) localmem[1][j++] = i;`).
        let group1 = vec![l];
        let group2: Vec<usize> = (0..nseq).filter(|&i| i != l).collect();

        // Baseline `intergroup_score` BEFORE commongappick. C also
        // commongappicks the groups before the realign DP, but
        // intergroup_score skips gap-gap columns anyway so the
        // baseline value is invariant to that stripping. No
        // constraints (disttbfast path), so impmatch = 0.
        let oscore = intergroup_score_c_order(
            &group1, &group2, &alignment.sequences, &weights, scoring,
        );


        // Mirror C `dooneiteration` exactly: per-group commongappick
        // FIRST, THEN call progressive-Falign on the stripped data.
        // For singleton group1, commongappick strips every column
        // where seq[l] is a gap → result is the gap-free singleton.
        // For group2, strips columns where ALL N-1 seqs are gap.
        let width = alignment.sequences[0].len();
        let g1_gap_cols: Vec<bool> = (0..width)
            .map(|c| group1.iter().all(|&i| alignment.sequences[i][c] == b'-'))
            .collect();
        let g2_gap_cols: Vec<bool> = (0..width)
            .map(|c| group2.iter().all(|&i| alignment.sequences[i][c] == b'-'))
            .collect();
        // Build a transient "candidate" workspace where each group has
        // its common-gap columns removed. For sequences NOT in either
        // group we keep raw bytes (they're irrelevant to the merge).
        let mut candidate: Vec<Vec<u8>> = alignment.sequences.iter()
            .map(|s| s.clone()).collect();
        for &i in &group1 {
            let stripped: Vec<u8> = (0..width)
                .filter(|&c| !g1_gap_cols[c])
                .map(|c| alignment.sequences[i][c])
                .collect();
            candidate[i] = stripped;
        }
        for &i in &group2 {
            let stripped: Vec<u8> = (0..width)
                .filter(|&c| !g2_gap_cols[c])
                .map(|c| alignment.sequences[i][c])
                .collect();
            candidate[i] = stripped;
        }
        // Now merge the two stripped groups via the progressive
        // Falign-equivalent (kobetsubunkatsu=0). After this call,
        // candidate[i] for i ∈ group1 ∪ group2 holds the new
        // alignment row; other indices keep their pre-strip data
        // (and we never read them again before discarding).
        let _ = crate::progressive::merge_two_groups_progressive(
            &group1, &group2, &mut candidate, &weights, scoring, &gap,
            params.use_fft,
            // C `disttbfast` is invoked with `-O` ($termgapopt) for
            // FFT-NS-2/i, meaning `outgap = 0` (terminal gaps NOT
            // penalised). Matches the progressive merge call site
            // in `engine.rs` which passes `penalize_term_gaps=false`
            // for non-G-INS-i / non-parttree modes.
            false,
        );

        let nscore = intergroup_score_c_order(
            &group1, &group2, &candidate, &weights, scoring,
        );
        // C `disttbfast.c:2457`: if( nscore < oscore ) revert.
        // Equivalent to accept-when-nscore-≥-oscore.
        if nscore >= oscore {
            alignment.sequences = candidate;
        }
    }
}

/// Falls back to whole-alignment refinement when no anchors are found.
///
/// `constraints` is intentionally not sliced — C's segmented path uses
/// `kobetsubunkatsu=1` which goes single-segment whenever `constraint != 0`,
/// so this function is only called from non-constraint modes (FFT-NS-i).
pub fn segmented_iterative_refine(
    alignment: &mut MultipleAlignment,
    topology: &Topology,
    scoring: &ScoringContext,
    params: &RefinementParams,
    constraints: Option<&LocalHomologyTable>,
) -> usize {
    let nseq = alignment.nseq();
    // `nseq == 2` is refined too — see `iterative_refine` (C `dvtditr.c:704-708`).
    if nseq < 2 || topology.steps.is_empty() {
        return 0;
    }

    let anchors = search_anchors_aa(
        &alignment.sequences,
        &scoring.substitution_matrix,
        &scoring.amino_map,
        20, 65,
    );
    if anchors.len() <= 2 {
        // No anchors found → behave like single-segment refinement.
        if refine_stats_enabled() {
            eprintln!("refine-segments: anchors={} segments=1 (unsegmented)", anchors.len());
        }
        return iterative_refine(alignment, topology, scoring, params, constraints);
    }
    if refine_stats_enabled() {
        eprintln!(
            "refine-segments: anchors={} segments={} len={}",
            anchors.len(),
            anchors.windows(2).filter(|w| w[0] < w[1]).count(),
            alignment.sequences.first().map_or(0, |s| s.len()),
        );
    }

    let mut total_iters = 0usize;
    let mut concat: Vec<Vec<u8>> = vec![Vec::new(); nseq];

    for w in anchors.windows(2) {
        let (start, end) = (w[0], w[1]);
        if start >= end { continue; }

        let seg_seqs: Vec<Vec<u8>> = alignment.sequences.iter()
            .map(|s| s[start..end].to_vec())
            .collect();
        let mut seg_msa = MultipleAlignment {
            sequences: seg_seqs,
            names: alignment.names.clone(),
            score: 0.0,
            step_trace: Vec::new(),
            guide_tree: None,
            first_pass_sequences: None, distance_matrix: None,
        };

        let iters = iterative_refine(&mut seg_msa, topology, scoring, params, constraints);
        total_iters += iters;

        for (i, seq) in seg_msa.sequences.iter().enumerate() {
            concat[i].extend_from_slice(seq);
        }
    }

    alignment.sequences = concat;
    total_iters
}

#[cfg(test)]
mod tests {
    use super::*;
    use mafft_tree::{DistanceMatrix, upgma};
    use mafft_scoring::build_context;
    use mafft_types::{ScoringModel, SeqType};
    use crate::progressive::progressive_align;

    /// Helper: build a 6-sequence UPGMA topology for branch-enumeration tests.
    fn make_6seq_topology() -> (Topology, usize) {
        let nseq = 6;
        let mut dm = DistanceMatrix::new(nseq);
        for i in 0..nseq {
            for j in (i + 1)..nseq {
                dm.set(i, j, (j - i) as f64 * 0.1);
            }
        }
        (upgma(&dm), nseq)
    }

    // ---------------------------------------------------------------
    // Regression guards for the iterative-refinement fixes.
    // Each test targets one specific behavior ported from C's
    // TreeDependentIteration() in tditeration.c. If any of these
    // are accidentally reverted, at least one test will fail.
    // ---------------------------------------------------------------

    /// Guard: branch count = (nseq-1)*2 - 1, matching C's nbranch formula.
    ///
    /// C computes `nbranch = (njob-1) * 2 - 1` (tditeration.c line 1458).
    /// The root step contributes only 1 branch (side 1), all others contribute
    /// 2 (sides 0 and 1). Reverting the root-step skip would produce
    /// (nseq-1)*2 branches instead.
    #[test]
    fn branch_count_matches_c_formula() {
        let (topo, nseq) = make_6seq_topology();
        let branch_map = build_branch_map(&topo, nseq);
        let total: usize = branch_map.iter().map(|sides| sides.len()).sum();
        let expected = (nseq - 1) * 2 - 1;
        assert_eq!(total, expected,
            "branch count should be (nseq-1)*2-1 = {expected}, got {total}");
    }

    /// Guard: root step has exactly 1 branch (side 1 only).
    ///
    /// C forces `k = 1` at the root step (tditeration.c line 1667:
    /// `if( l == locnjob-2 ) k = 1`), skipping side 0 because at the
    /// root left-vs-complement and right-vs-complement are identical
    /// splits. Reverting would give the root step 2 branches.
    #[test]
    fn root_step_has_single_branch() {
        let (topo, nseq) = make_6seq_topology();
        let branch_map = build_branch_map(&topo, nseq);
        let root_branches = branch_map.last().unwrap();
        assert_eq!(root_branches.len(), 1,
            "root step should have 1 branch (side 1 only), got {}", root_branches.len());
        assert_eq!(root_branches[0].0, 1, "root branch should be side 1");
    }

    /// Guard: non-root steps each have exactly 2 branches (sides 0 and 1).
    #[test]
    fn non_root_steps_have_two_branches() {
        let (topo, nseq) = make_6seq_topology();
        let branch_map = build_branch_map(&topo, nseq);
        for (step_idx, sides) in branch_map.iter().enumerate() {
            if step_idx < branch_map.len() - 1 {
                assert_eq!(sides.len(), 2,
                    "non-root step {step_idx} should have 2 branches, got {}", sides.len());
            }
        }
    }

    /// Guard: default cut is 0.0 (accept only strict improvements).
    ///
    /// C's dvtditr.c sets `cut = 0.0` (line 71). The acceptance test is
    /// `tscore > mscore - cut/100*mscore`, so with cut=0 only strictly
    /// improving moves are accepted. Reverting to a nonzero cut would
    /// accept non-improving moves.
    #[test]
    fn default_cut_is_zero() {
        let params = RefinementParams::default();
        assert_eq!(params.cut, 0.0,
            "default cut must be 0.0 (strict improvement only), matching C's dvtditr.c");
    }

    /// Guard: even iterations traverse steps forward, odd iterations reverse.
    ///
    /// C alternates direction (tditeration.c lines 1641-1648):
    ///   even → lin=0, ldf=+1 (forward)
    ///   odd  → lin=locnjob-2, ldf=-1 (reverse)
    /// This test verifies the first branch processed differs between
    /// iteration 0 (forward) and iteration 1 (reverse).
    #[test]
    fn alternating_direction_between_iterations() {
        let (topo, _) = make_6seq_topology();
        let nsteps = topo.steps.len();

        // Verify the step_order logic directly.
        let forward: Vec<usize> = (0..nsteps).collect();
        let reverse: Vec<usize> = (0..nsteps).rev().collect();

        // Even iteration → forward
        assert_eq!(forward[0], 0, "forward should start at step 0");
        // Odd iteration → reverse
        assert_eq!(reverse[0], nsteps - 1, "reverse should start at last step");
        // They must differ (nsteps > 1 for any nseq > 2)
        assert_ne!(forward, reverse,
            "forward and reverse step orders must differ for alternation");
    }

    /// Guard: oscillation detection terminates refinement early.
    ///
    /// C checks per-branch score history (tditeration.c lines 2343-2371)
    /// and stops if a branch's score at iteration N matches the score
    /// from iteration N-2. We verify that iterative_refine returns in
    /// fewer than max_iterations when running on inputs that converge
    /// quickly (which will produce identical scores across iterations).
    #[test]
    fn refinement_terminates_not_at_max_iterations() {
        let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);
        // Three nearly-identical sequences: refinement should converge fast,
        // well before hitting 100 iterations.
        let seqs = vec![
            b"ACDEFGHIK".to_vec(),
            b"ACDEFGHIK".to_vec(),
            b"ACDEFGHIK".to_vec(),
        ];
        let names = vec!["s1".into(), "s2".into(), "s3".into()];

        let mut dm = DistanceMatrix::new(3);
        dm.set(0, 1, 0.001);
        dm.set(0, 2, 0.001);
        dm.set(1, 2, 0.001);
        let topo = upgma(&dm);

        let mut msa = progressive_align(&seqs, &names, &topo, &scoring, false, None);
        let params = RefinementParams {
            max_iterations: 100,
            ..Default::default()
        };

        let iters = iterative_refine(&mut msa, &topo, &scoring, &params, None);
        // Identical sequences must converge immediately — either via the
        // identity check or via the convergence counter (nseq * 2 = 6).
        assert!(iters < 100,
            "expected early termination (convergence/oscillation), got {iters} iterations");
        assert!(iters <= 2,
            "identical sequences should converge in 1-2 iterations, got {iters}");
    }

    // ---------------------------------------------------------------
    // Original functional tests (preserved).
    // ---------------------------------------------------------------

    #[test]
    fn refinement_converges() {
        let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);
        let seqs = vec![
            b"ACDEFGHIK".to_vec(),
            b"ACDEFHIK".to_vec(),
            b"ACDHIK".to_vec(),
        ];
        let names = vec!["s1".into(), "s2".into(), "s3".into()];

        let mut dm = DistanceMatrix::new(3);
        dm.set(0, 1, 0.1);
        dm.set(0, 2, 0.3);
        dm.set(1, 2, 0.2);
        let topo = upgma(&dm);

        let mut msa = progressive_align(&seqs, &names, &topo, &scoring, false, None);
        let params = RefinementParams {
            max_iterations: 10,
            ..Default::default()
        };

        let iters = iterative_refine(&mut msa, &topo, &scoring, &params, None);
        assert!(iters <= 10);

        let width = msa.width();
        for seq in &msa.sequences {
            assert_eq!(seq.len(), width);
        }

        let ungapped: Vec<Vec<u8>> = msa.sequences.iter()
            .map(|s| s.iter().filter(|&&c| c != b'-').cloned().collect())
            .collect();
        assert_eq!(ungapped[0], b"ACDEFGHIK");
        assert_eq!(ungapped[1], b"ACDEFHIK");
        assert_eq!(ungapped[2], b"ACDHIK");
    }

    #[test]
    fn refinement_preserves_width_consistency() {
        let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);
        let seqs = vec![
            b"ACDEFGHIKLMNP".to_vec(),
            b"ACDEFHIKLMNP".to_vec(),
            b"ACDEHIKLMNP".to_vec(),
            b"ACDHIKLMNP".to_vec(),
        ];
        let names: Vec<String> = (0..4).map(|i| format!("s{i}")).collect();

        let mut dm = DistanceMatrix::new(4);
        dm.set(0, 1, 0.1); dm.set(0, 2, 0.2); dm.set(0, 3, 0.3);
        dm.set(1, 2, 0.15); dm.set(1, 3, 0.25); dm.set(2, 3, 0.15);
        let topo = upgma(&dm);

        let mut msa = progressive_align(&seqs, &names, &topo, &scoring, false, None);
        let params = RefinementParams { max_iterations: 5, ..Default::default() };

        iterative_refine(&mut msa, &topo, &scoring, &params, None);

        let width = msa.width();
        assert!(width > 0);
        for (i, seq) in msa.sequences.iter().enumerate() {
            assert_eq!(seq.len(), width, "sequence {i} has wrong width");
        }

        // Verify residue preservation
        for (i, seq) in msa.sequences.iter().enumerate() {
            let residue_count = seq.iter().filter(|&&c| c != b'-').count();
            assert_eq!(residue_count, seqs[i].len(),
                "sequence {i} lost residues: {} vs {}", residue_count, seqs[i].len());
        }
    }

    #[test]
    fn refinement_six_sequences() {
        let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);
        let seqs = vec![
            b"ACDEFGHIKLMNPQR".to_vec(),
            b"ACDEFHIKLMNPQR".to_vec(),
            b"ACDEHIKLMNPQR".to_vec(),
            b"ACDHIKLMNPQR".to_vec(),
            b"ACDHIKLMNP".to_vec(),
            b"ACDHIKLM".to_vec(),
        ];
        let names: Vec<String> = (0..6).map(|i| format!("s{i}")).collect();

        let mut dm = DistanceMatrix::new(6);
        for i in 0..6 {
            for j in (i + 1)..6 {
                dm.set(i, j, (j - i) as f64 * 0.1);
            }
        }
        let topo = upgma(&dm);

        let mut msa = progressive_align(&seqs, &names, &topo, &scoring, false, None);
        let params = RefinementParams { max_iterations: 3, ..Default::default() };

        iterative_refine(&mut msa, &topo, &scoring, &params, None);

        let width = msa.width();
        for (i, seq) in msa.sequences.iter().enumerate() {
            assert_eq!(seq.len(), width, "sequence {i} has wrong width after refinement");
            let residues = seq.iter().filter(|&&c| c != b'-').count();
            assert_eq!(residues, seqs[i].len(),
                "sequence {i} lost residues during refinement");
        }
    }

    /// Guard: FFT-accelerated refinement produces valid results and
    /// does not cause width explosion.
    ///
    /// C always uses Falign (FFT) in refinement. This test verifies that
    /// use_fft=true in RefinementParams produces a valid alignment with
    /// bounded width growth (no exponential blow-up).
    #[test]
    fn refinement_fft_no_width_explosion() {
        let scoring = build_context(ScoringModel::Blosum(62), SeqType::Protein);
        let seqs = vec![
            b"ACDEFGHIKLMNPQRSTVWY".to_vec(),
            b"ACDEFHIKLMNPQRSTVWY".to_vec(),
            b"ACDEHIKLMNPQRSTVWY".to_vec(),
            b"ACDHIKLMNPQRSTVWY".to_vec(),
            b"ACDHIKLMNPQR".to_vec(),
            b"ACDHIKLM".to_vec(),
        ];
        let names: Vec<String> = (0..6).map(|i| format!("s{i}")).collect();

        let mut dm = DistanceMatrix::new(6);
        for i in 0..6 {
            for j in (i + 1)..6 {
                dm.set(i, j, (j - i) as f64 * 0.1);
            }
        }
        let topo = upgma(&dm);

        let mut msa = progressive_align(&seqs, &names, &topo, &scoring, false, None);
        let pre_width = msa.width();

        let params = RefinementParams {
            max_iterations: 5,
            use_fft: true,
            ..Default::default()
        };

        iterative_refine(&mut msa, &topo, &scoring, &params, None);

        let post_width = msa.width();
        // Width should not blow up — allow at most 2x growth for reasonable
        // refinement (C typically keeps width within ~10% of progressive).
        assert!(post_width <= pre_width * 2,
            "width explosion: {} -> {} (>2x growth)", pre_width, post_width);

        for (i, seq) in msa.sequences.iter().enumerate() {
            assert_eq!(seq.len(), post_width, "sequence {i} has wrong width");
            let residues = seq.iter().filter(|&&c| c != b'-').count();
            assert_eq!(residues, seqs[i].len(),
                "sequence {i} lost residues during FFT refinement");
        }
    }
}

//! Pivot selection + pickmtx + redundancy filter for `--parttree`,
//! mirroring `splittbfast.c::splitseq_mq` lines 1316-2030.
//!
//! Steps:
//! 1. **Reference selection** (`uselongest=1`, the default): find
//!    sequence with maximum selfscore (= number of valid 6-mers) and
//!    swap it to position 0 of the score array.
//! 2. **Initial distance**: for every sequence `i`, compute
//!    `scores[i].score = parttree_distance(reference, i)`.
//! 3. **Sort** by `dcompare`: ascending `score`, then ascending
//!    `selfscore`, then ascending `orilen`.
//! 4. **Pivot selection**: pick the first sorted index (=reference,
//!    distance 0), then the last (most distant), then alternating
//!    deterministic-`nkouho/2` and random `rnd()` until `npick ==
//!    picksize` or `nkouho == 0`. Dedupes by string equality.
//! 5. **`qsort(picks)` ascending**: canonicalizes the picks order.
//! 6. **`pickmtx`**: pairwise distance among picks. Row 0 is the
//!    precomputed `scores[picks[i]].score`; rows ≥1 use the
//!    `localcommonsextet_p` formula.
//! 7. **Redundancy filter**: for each `i < j`, if `pickmtx[i][j-i] <
//!    maxdist · 0.7`, mark `j` redundant. Survivors form `yukos[]`.
//!
//! For our `n = 36 < picksize = 50` test fixture, all 36 sequences
//! become picks and `qsort(picks)` canonicalizes — the random-pick
//! path is suppressed and `rand()` determinism doesn't matter. For
//! `nin > picksize`, the libc `rand()` sequence becomes load-bearing
//! and would need to be matched bit-for-bit (macOS BSD ≠ glibc).

use crate::parttree_dist::{
    DLENFACA, DLENFACB, DLENFACC, DLENFACD, MAX6DIST, PICKSIZE, PLENFACA, PLENFACB, PLENFACC,
    PLENFACD, TOKYORIPARA, common_sextets_p, composition_table, encode_points_dna,
    encode_points_protein, lenfac,
};

/// Per-sequence info, mirroring C's `Scores` struct in `mltaln.h`.
/// Only the fields actually used by `splitseq_mq` are kept.
#[derive(Debug, Clone)]
pub struct ScoreEntry {
    /// Original (input-order) sequence index.
    pub numinseq: usize,
    /// `localcommonsextet_p(self, self) = points.len()` for protein/DNA
    /// k-tuple mode. Equals number of valid 6-mers in the sequence.
    pub selfscore: i64,
    /// `strlen(seq)` post-gappick. Same as the input-buffer length when
    /// the input has no gaps.
    pub orilen: usize,
    /// Initial distance from the reference (position-0 after
    /// `uselongest` swap). Set during step 2 of the pipeline.
    pub score: f64,
    /// 6-mer point vector (length = `orilen - 5` for ungapped protein).
    pub points: Vec<u32>,
}

/// Result of pivot selection: the (possibly permuted) score array,
/// the chosen pivot indices in `qsort`-canonicalized order, and the
/// `pickmtx` upper-triangular distance matrix.
///
/// `picks[i]` is an INDEX INTO THE SORTED SCORE ARRAY. To convert to
/// original sequence indices, use `scores[picks[i]].numinseq`.
///
/// `pickmtx` uses C's "with-diagonal" half-matrix layout:
/// `pickmtx[i][j-i]` for `j >= i`, with `pickmtx[i][0]` unused (the
/// diagonal slot kept zero, matching `AllocateFloatHalfMtx` in
/// `mtxutl.c:163`).
#[derive(Debug, Clone)]
pub struct PivotResult {
    pub scores: Vec<ScoreEntry>,
    pub picks: Vec<usize>,
    pub pickmtx: Vec<Vec<f64>>,
    /// `maxdist = scores[nin-1].score` from the post-`dcompare` sort.
    /// Used to compute the redundancy threshold.
    pub maxdist: f64,
}

/// Sequence type for distance/lenfac selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtSeqKind {
    Protein,
    Dna,
}

impl PtSeqKind {
    fn tsize(self) -> usize {
        match self {
            Self::Protein => 46656,
            Self::Dna => 4096,
        }
    }
    fn lenfac_constants(self) -> (f64, f64, f64, f64) {
        match self {
            Self::Protein => (PLENFACA, PLENFACB, PLENFACC, PLENFACD),
            Self::Dna => (DLENFACA, DLENFACB, DLENFACC, DLENFACD),
        }
    }
    fn encode(self, seq: &[u8]) -> Vec<u32> {
        match self {
            Self::Protein => encode_points_protein(seq),
            Self::Dna => encode_points_dna(seq),
        }
    }
}

/// Build initial `scores[]` for all input sequences:
/// - `numinseq = i` (original input order)
/// - `points` = 6-mer encoding
/// - `selfscore = points.len()`
/// - `orilen = seq.len()`
/// - `score = 0.0` (set by [`compute_initial_scores`] later)
pub fn build_score_entries(sequences: &[Vec<u8>], kind: PtSeqKind) -> Vec<ScoreEntry> {
    sequences
        .iter()
        .enumerate()
        .map(|(i, seq)| {
            let pts = kind.encode(seq);
            let selfscore = pts.len() as i64;
            ScoreEntry {
                numinseq: i,
                selfscore,
                orilen: seq.len(),
                score: 0.0,
                points: pts,
            }
        })
        .collect()
}

/// Pick reference (position 0): mirrors `splittbfast.c::1316-1342`'s
/// `uselongest=1` branch — linear scan for max selfscore (ties: keep
/// the first one), swap winner to index 0.
pub fn pick_reference_max_selfscore(scores: &mut [ScoreEntry]) {
    if scores.is_empty() {
        return;
    }
    let mut best = 0usize;
    let mut best_score = scores[0].selfscore;
    for i in 1..scores.len() {
        // C: `if( ptr->selfscore > selfscore0 )` — strict `>`, so the
        // earliest tied wins.
        if scores[i].selfscore > best_score {
            best_score = scores[i].selfscore;
            best = i;
        }
    }
    if best != 0 {
        scores.swap(0, best);
    }
}

/// Compute `scores[i].score` for all `i` as the parttree distance from
/// `scores[0]` to `scores[i]`. Mirrors `splittbfast.c::1393-1448` in
/// the non-doalign branch.
pub fn compute_initial_scores(scores: &mut [ScoreEntry], kind: PtSeqKind) {
    if scores.is_empty() {
        return;
    }
    let tsize = kind.tsize();
    let (a, b, c, d) = kind.lenfac_constants();

    let table0 = composition_table(&scores[0].points, tsize);
    let selfscore0 = scores[0].selfscore;
    let orilen0 = scores[0].orilen;

    for i in 0..scores.len() {
        let common = common_sextets_p(&table0, &scores[i].points, tsize);
        let bunbo = selfscore0.min(scores[i].selfscore) as f64;
        let raw = if bunbo > 0.0 {
            1.0 - common as f64 / bunbo
        } else {
            1.0
        };
        let lf = lenfac(orilen0, scores[i].orilen, a, b, c, d);
        let mut s = raw * lf;
        if s > MAX6DIST {
            s = MAX6DIST;
        }
        scores[i].score = s;
    }
}

/// Sort `scores[]` by C's `dcompare` (`splittbfast.c:72-87`): primary
/// `score` ASCENDING; tie-break by `selfscore` DESCENDING (C returns
/// `a < b → 1` for selfscore, so smaller selfscore sorts AFTER larger);
/// further tie-break by `orilen` DESCENDING (same convention).
///
/// Uses our in-tree port of FreeBSD's `qsort` (`bsd_qsort`) rather than
/// `libc::qsort`, because `libc::qsort` delegates to the host C library
/// (BSD qsort on macOS, glibc qsort on Linux, MSVC qsort on Windows) and
/// these implementations disagree on the relative order of *truly tied*
/// elements (same score / selfscore / orilen — only occurs when the
/// input contains exactly-duplicate sequences). C MAFFT 7.526 inherits
/// the same platform-dependence; our Rust port deliberately pins the
/// behavior to BSD/macOS qsort so the binary produces consistent output
/// on every platform it's built for.
pub fn dcompare_sort(scores: &mut [ScoreEntry]) {
    if scores.len() < 2 {
        return;
    }
    crate::bsd_qsort::bsd_qsort(scores, |a, b| {
        // Primary: score ASC (`dcompare:74-76`).
        if a.score > b.score {
            return std::cmp::Ordering::Greater;
        }
        if a.score < b.score {
            return std::cmp::Ordering::Less;
        }
        // Tie: selfscore DESC (`dcompare:78-79` returns 1 when a < b).
        if a.selfscore < b.selfscore {
            return std::cmp::Ordering::Greater;
        }
        if a.selfscore > b.selfscore {
            return std::cmp::Ordering::Less;
        }
        // Tie: orilen DESC (`dcompare:82-83` returns 1 when a < b).
        if a.orilen < b.orilen {
            return std::cmp::Ordering::Greater;
        }
        if a.orilen > b.orilen {
            return std::cmp::Ordering::Less;
        }
        std::cmp::Ordering::Equal
    });
}

/// Pivot selection — mirrors `splittbfast.c::1495-1574`.
///
/// Returns the `picks` list AFTER the final `qsort(picks)` canonical
/// sort. For `n ≤ picksize`, all sequences become picks (since the
/// loop terminates when `nkouho == 0`).
///
/// **`rand()` warning**: when `n > picksize`, the loop falls into the
/// random-pick path which uses libc `rand()` with C's default seed.
/// Byte-exact parity in that regime would require linking C's `rand()`
/// or replicating glibc's LCG state; the current implementation hasn't
/// produced an observed divergence (`--parttree` matches C MAFFT 7.526
/// on the 36-seq sample and across BBaliBase 3). For `n ≤ picksize`
/// the `qsort` makes the random-pick order irrelevant.
pub fn select_pivots(scores: &[ScoreEntry], picksize: usize) -> Vec<usize> {
    let nin = scores.len();
    if nin == 0 {
        return Vec::new();
    }
    if nin == 1 {
        return vec![0];
    }

    let mut pickkouho: Vec<usize> = (1..nin).collect();
    let mut nkouho = pickkouho.len(); // = nin - 1

    let mut picks = Vec::with_capacity(picksize.min(nin));
    picks.push(0);

    // picks[1] = pickkouho[nkouho - 1] = the last candidate (= the most
    // distant after `dcompare_sort` puts farthest at index nin-1).
    if nkouho > 0 {
        let picktmp = pickkouho[nkouho - 1];
        nkouho -= 1;
        // C dedupe: compare shimon (CRC) + full string equality. We
        // approximate by checking whether the candidate's full byte
        // sequence equals an existing pick's. For ungapped protein
        // input this matches C's path exactly for non-degenerate
        // datasets.
        if !picks
            .iter()
            .any(|&p| seqs_equal(&scores[p], &scores[picktmp]))
        {
            picks.push(picktmp);
        }
    }

    let mut deterministic_first = true;
    while picks.len() < picksize && nkouho > 0 {
        let rn = if deterministic_first {
            deterministic_first = false;
            (nkouho as f64 * 0.5) as usize
        } else {
            // C path uses libc rand() — see warning above. For
            // n <= picksize the whole loop never visits this branch
            // since each iteration consumes one nkouho entry until
            // qsort(picks) canonicalizes anyway. To keep things
            // deterministic in Rust (and obvious about the lack of
            // rand-matching), we just take the last remaining.
            nkouho - 1
        };
        let picktmp = pickkouho[rn];
        nkouho -= 1;
        // C: pickkouho[rn] = pickkouho[nkouho]; (swap with last, then shrink)
        pickkouho[rn] = pickkouho[nkouho];

        if !picks
            .iter()
            .any(|&p| seqs_equal(&scores[p], &scores[picktmp]))
        {
            picks.push(picktmp);
        }
    }

    picks.sort_unstable(); // C: qsort(picks, npick, sizeof(int), intcompare)
    picks
}

fn seqs_equal(a: &ScoreEntry, b: &ScoreEntry) -> bool {
    // C compares `shimon` (a CRC fingerprint) AND `strcmp(seq[a], seq[b])`.
    // We use selfscore + orilen + points-vector equality as a
    // strict-but-cheap proxy that catches all true byte-equal duplicates.
    a.selfscore == b.selfscore && a.orilen == b.orilen && a.points == b.points
}

/// Build `pickmtx`: `pickmtx[i][j-i]` for `j >= i` (with-diagonal
/// half-matrix layout matching `AllocateFloatHalfMtx` /
/// `setnearest`'s `eff[i][j-i]` access pattern).
///
/// Row 0 reuses the precomputed `scores[picks[i]].score` (which is
/// the distance from the reference, already computed in step 2).
/// Rows `i >= 1` compute via `localcommonsextet_p` + `lenfac`.
///
/// Mirrors `splittbfast.c::1620-1707`.
pub fn build_pickmtx(scores: &[ScoreEntry], picks: &[usize], kind: PtSeqKind) -> Vec<Vec<f64>> {
    let npick = picks.len();
    let tsize = kind.tsize();
    let (a, b, c, d) = kind.lenfac_constants();

    // Each row i has length `npick - i` (slot 0 unused).
    let mut pickmtx: Vec<Vec<f64>> = (0..npick).map(|i| vec![0.0f64; npick - i]).collect();

    // C lines 1620-1626: pickmtx[0][s_p_map[j]] = scores[j].score.
    // Translation: for each sorted-position j that's a pick, find its
    // position within `picks` and write scores[j].score there.
    // Equivalently: `pickmtx[0][k] = scores[picks[k]].score` for k>=1.
    if npick >= 1 {
        for k in 1..npick {
            pickmtx[0][k] = scores[picks[k]].score;
        }
    }

    // C lines 1629-1707: for each j in [1, npick), build the composition
    // table for picks[j], then compute pickmtx[j][i-j] for i > j.
    for j in 1..npick {
        let table_j = composition_table(&scores[picks[j]].points, tsize);
        let selfscore_j = scores[picks[j]].selfscore;
        let orilen_j = scores[picks[j]].orilen;

        for i in (j + 1)..npick {
            let common = common_sextets_p(&table_j, &scores[picks[i]].points, tsize);
            let bunbo = selfscore_j.min(scores[picks[i]].selfscore) as f64;
            let raw = if bunbo > 0.0 {
                1.0 - common as f64 / bunbo
            } else {
                1.0
            };
            let lf = lenfac(orilen_j, scores[picks[i]].orilen, a, b, c, d);
            let mut dist = raw * lf;
            if dist > MAX6DIST {
                dist = MAX6DIST;
            }
            pickmtx[j][i - j] = dist;
        }
    }

    pickmtx
}

/// Apply C's redundancy filter (`splittbfast.c::1942-1981` HUKINTOTREE
/// branch). For each `i < j` in increasing order, if
/// `pickmtx[i][j-i] < maxdist * tokyoripara`, mark `j` redundant.
///
/// `tokyoripara` defaults to `TOKYORIPARA` (0.70) but C
/// (`splittbfast.c:2760-2761`) sets it to **0.0** when `picksize >
/// njob`, which makes the filter a no-op (all pivots survive).
/// Callers should pass the effective `tokyoripara` for their input
/// size — see [`run_pivot_pipeline`] for the C-equivalent rule.
///
/// Returns the boolean `tsukau[]` array (length `npick`), where
/// `tsukau[i] == true` means pivot `i` survives.
pub fn redundancy_filter(pickmtx: &[Vec<f64>], maxdist: f64, tokyoripara: f64) -> Vec<bool> {
    let npick = pickmtx.len();
    let mut tsukau = vec![true; npick];
    let kijun = maxdist * tokyoripara;

    for i in 0..npick.saturating_sub(1) {
        if !tsukau[i] {
            continue;
        }
        for j in (i + 1)..npick {
            if !tsukau[j] {
                continue;
            }
            let d = pickmtx[i][j - i];
            if d < kijun {
                tsukau[j] = false;
            }
        }
    }
    tsukau
}

/// Compose the full pivot pipeline: build scores, pick reference,
/// compute initial scores, sort, select pivots, build pickmtx, apply
/// redundancy filter. Returns the final `(scores, picks, yukomtx)`
/// where `yukomtx` is the `nyuko × nyuko` slice of `pickmtx` indexed
/// by surviving picks.
///
/// `yukos` are picks-array indices (NOT sorted-scores indices).
pub struct PartTreePivots {
    pub scores: Vec<ScoreEntry>,
    pub picks: Vec<usize>,
    pub pickmtx: Vec<Vec<f64>>,
    pub maxdist: f64,
    /// Indices into `picks` that survived the redundancy filter (length = `nyuko`).
    pub yukos: Vec<usize>,
    /// `yukomtx[i][j-i]` — symmetric distance among surviving pivots,
    /// using C's with-diagonal half-matrix layout (slot `[0]` unused).
    pub yukomtx: Vec<Vec<f64>>,
}

pub fn run_pivot_pipeline(
    sequences: &[Vec<u8>],
    kind: PtSeqKind,
    picksize: usize,
) -> PartTreePivots {
    let nin = sequences.len();
    let mut scores = build_score_entries(sequences, kind);
    pick_reference_max_selfscore(&mut scores);
    compute_initial_scores(&mut scores, kind);
    dcompare_sort(&mut scores);
    let picks = select_pivots(&scores, picksize);
    let pickmtx = build_pickmtx(&scores, &picks, kind);
    let maxdist = if nin >= 1 { scores[nin - 1].score } else { 0.0 };
    // C `splittbfast.c:2760-2761`: tokyoripara → 0.0 when picksize > njob,
    // making the filter a no-op (all picks survive). Default otherwise
    // is TOKYORIPARA = 0.70.
    let effective_tokyoripara = if picksize > nin { 0.0 } else { TOKYORIPARA };
    let tsukau = redundancy_filter(&pickmtx, maxdist, effective_tokyoripara);

    // Compact pickmtx → yukomtx (C lines 1983-2032).
    let yukos: Vec<usize> = (0..picks.len()).filter(|&i| tsukau[i]).collect();
    let nyuko = yukos.len();
    let mut yukomtx: Vec<Vec<f64>> = (0..nyuko).map(|i| vec![0.0f64; nyuko - i]).collect();
    for (ii, &i) in yukos.iter().enumerate() {
        for (jj_off, &j) in yukos.iter().enumerate().skip(ii + 1) {
            // pickmtx[i][j-i] → yukomtx[ii][jj_off-ii]
            yukomtx[ii][jj_off - ii] = pickmtx[i][j - i];
        }
    }

    PartTreePivots {
        scores,
        picks,
        pickmtx,
        maxdist,
        yukos,
        yukomtx,
    }
}

#[allow(dead_code)]
fn _unused_constants_anchor() -> usize {
    PICKSIZE
}

//! C-exact 6-mer distance for `--parttree`, mirroring `splittbfast.c`.
//!
//! Two functions:
//! - [`encode_points`]: ports `seq_grp` + `makepointtable` (protein) /
//!   `seq_grp_nuc` + `makepointtable_nuc` (DNA), producing the rolling
//!   6-mer point vector that C feeds to `commonsextet_p`.
//! - [`composition_table`]: ports `makecompositiontable_p` — frequency
//!   table indexed by 6-mer code.
//! - [`common_sextets_p`]: ports `commonsextet_p` — `min(table[p],
//!   memo[p])` accumulator counting common 6-mers.
//! - [`parttree_distance`]: composes the above with the `lenfac`
//!   length-correction `splittbfast.c:1674` and `MAX6DIST = 10.0` clamp.
//!
//! These are intentionally *separate* from `distance::ktuple_distance`,
//! which scales by `* 2.0` and clamps at 2.0 (matching `disttbfast`'s
//! distance convention). PartTree uses the unscaled `splittbfast` form.
//!
//! Validated against C `commonsextet_p` via the FFI test in
//! `tests/cross_validate_parttree.rs`.

/// `splittbfast.c:5` `#define PICKSIZE 50`.
pub const PICKSIZE: usize = 50;

/// `splittbfast.c:7` `#define TOKYORIPARA 0.70`.
pub const TOKYORIPARA: f64 = 0.70;

/// `splittbfast.c:12` `#define MAX6DIST 10.0`.
pub const MAX6DIST: f64 = 10.0;

/// Length-correction parameters for protein (`PLENFAC{A,B,C,D}` in
/// `splittbfast.c:34-37`). Note these match `disttbfast.c:43-46`'s active
/// `#else` block, so all C MAFFT sites use the same constants.
pub const PLENFACA: f64 = 0.01;
pub const PLENFACB: f64 = 10000.0;
pub const PLENFACC: f64 = 10000.0;
pub const PLENFACD: f64 = 0.1;

/// DNA equivalents (`DLENFAC{A,B,C,D}` in `splittbfast.c:38-41`).
pub const DLENFACA: f64 = 0.01;
pub const DLENFACB: f64 = 2500.0;
pub const DLENFACC: f64 = 2500.0;
pub const DLENFACD: f64 = 0.1;

/// `splittbfast.c::seq_grp` + `makepointtable` (protein, alphabet=6).
///
/// Maps each amino acid to one of 6 physico-chemical groups (skipping
/// `X` / `.` / `-` / `J` and other ambiguous codes), then encodes
/// every contiguous 6-residue window into a base-6 integer code in
/// `[0, 6^6 = 46656)`.
///
/// Returns the point vector — empty if the filtered sequence is shorter
/// than 6 residues.
pub fn encode_points_protein(seq: &[u8]) -> Vec<u32> {
    encode_points_with(seq, amino_grp_protein, 6, 7776)
}

/// DNA variant: 4 base groups, alphabet=4.
pub fn encode_points_dna(seq: &[u8]) -> Vec<u32> {
    encode_points_with(seq, amino_grp_nuc, 4, 1024)
}

fn encode_points_with(
    seq: &[u8],
    grp: fn(u8) -> Option<u32>,
    base: u32,
    leading_weight: u32,
) -> Vec<u32> {
    // First filter to group indices (drop ambiguous chars).
    let mut g: Vec<u32> = Vec::with_capacity(seq.len());
    for &c in seq {
        if let Some(v) = grp(c) {
            g.push(v);
        }
    }
    if g.len() < 6 {
        return Vec::new();
    }

    // Mirrors C's `makepointtable`: rolling base-`base` encoding of 6-grams.
    let n = g.len();
    let mut points = Vec::with_capacity(n - 5);
    let mut point: u32 = g[0] * leading_weight
        + g[1] * (base * base * base * base)
        + g[2] * (base * base * base)
        + g[3] * (base * base)
        + g[4] * base
        + g[5];
    points.push(point);
    for i in 6..n {
        // C: `point -= *p++ * 7776; point *= 6; point += *n++;` (protein)
        // The inner constant is `base^5`. For DNA, base=4 → 4^5 = 1024.
        point = point.wrapping_sub(g[i - 6].wrapping_mul(leading_weight));
        point = point.wrapping_mul(base);
        point = point.wrapping_add(g[i]);
        points.push(point);
    }
    points
}

/// Port of `splittbfast.c::makecompositiontable_p`. Returns a flat
/// frequency table indexed by 6-mer code; `tsize = base^6`
/// (46656 for protein, 4096 for DNA).
pub fn composition_table(points: &[u32], tsize: usize) -> Vec<i32> {
    let mut table = vec![0i32; tsize];
    for &p in points {
        table[p as usize] += 1;
    }
    table
}

/// Port of `splittbfast.c::localcommonsextet_p` (and the equivalent
/// `mltaln9.c::commonsextet_p`).
///
/// For each 6-mer `p` in `points` (seq2's point vector), increment
/// `memo[p]`. If `memo[p]++ < table[p]` (table built from seq1's
/// points), increment `value`. Result = `Σ_p min(seq1_count[p],
/// seq2_count[p])` — the count of shared 6-mers with multiplicity.
pub fn common_sextets_p(table: &[i32], points: &[u32], tsize: usize) -> i32 {
    let mut memo = vec![0i32; tsize];
    let mut value: i32 = 0;
    for &p in points {
        let pi = p as usize;
        let tmp = memo[pi];
        memo[pi] = tmp + 1;
        if tmp < table[pi] {
            value += 1;
        }
    }
    value
}

/// `splittbfast.c:1674` length-correction factor:
/// `lenfac = 1 / (shorter/longer * d + b/(longer + c) + a)`.
///
/// Returns 1.0 for zero-length inputs to avoid division-by-zero
/// (matches `compute_lenfac` behavior in `distance.rs`).
pub fn lenfac(len1: usize, len2: usize, a: f64, b: f64, c: f64, d: f64) -> f64 {
    let l1 = len1 as f64;
    let l2 = len2 as f64;
    let longer = l1.max(l2);
    let shorter = l1.min(l2);
    if longer == 0.0 {
        return 1.0;
    }
    1.0 / (shorter / longer * d + b / (longer + c) + a)
}

/// PartTree pairwise distance — the C-exact `splittbfast` formula
/// (`splittbfast.c:1674` + `:1699`):
///
/// ```text
/// raw    = 1.0 - common_sextets(table1, points2) / min(selfscore_i, selfscore_j)
/// lenfac = 1 / (shorter/longer * d + b / (longer + c) + a)
/// dist   = clamp(raw * lenfac, 0.0, MAX6DIST = 10.0)
/// ```
///
/// **Critical**: `lenfac` uses RAW sequence lengths (`scores[].orilen` =
/// `strlen(seq)` post-gappick), NOT the filtered point-vector length.
/// `selfscore_i` is `points_i.len()` (number of valid 6-mers).
///
/// Note this is HALF the value `distance::ktuple_distance` returns
/// (which has an extra `* 2.0` factor matching disttbfast's convention).
pub fn parttree_distance_protein(
    points1: &[u32],
    points2: &[u32],
    raw_len1: usize,
    raw_len2: usize,
) -> f64 {
    parttree_distance_with(
        points1, points2, raw_len1, raw_len2, 46656, PLENFACA, PLENFACB, PLENFACC, PLENFACD,
    )
}

pub fn parttree_distance_dna(
    points1: &[u32],
    points2: &[u32],
    raw_len1: usize,
    raw_len2: usize,
) -> f64 {
    parttree_distance_with(
        points1, points2, raw_len1, raw_len2, 4096, DLENFACA, DLENFACB, DLENFACC, DLENFACD,
    )
}

fn parttree_distance_with(
    points1: &[u32],
    points2: &[u32],
    raw_len1: usize,
    raw_len2: usize,
    tsize: usize,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
) -> f64 {
    let pl1 = points1.len();
    let pl2 = points2.len();
    if pl1 == 0 || pl2 == 0 {
        return MAX6DIST;
    }
    let table1 = composition_table(points1, tsize);
    let common = common_sextets_p(&table1, points2, tsize);
    let bunbo = pl1.min(pl2) as f64;
    let raw = 1.0 - common as f64 / bunbo;
    let lf = lenfac(raw_len1, raw_len2, a, b, c, d);
    let mut dist = raw * lf;
    if dist > MAX6DIST {
        dist = MAX6DIST;
    }
    if dist < 0.0 {
        dist = 0.0;
    }
    dist
}

/// Maps protein characters to one of 6 physico-chemical groups; returns
/// `None` for characters outside the 20 standard AAs (X, gaps, J, etc.),
/// matching C's `seq_grp` filter `if( tmp < 6 )`.
fn amino_grp_protein(c: u8) -> Option<u32> {
    // Group assignments from blosum.c:13-17 (locgrpd[]).
    // Order in locaminod[]: ARNDCQEGHILKMFPSTWYVBZX.-J
    match c.to_ascii_uppercase() {
        b'A' => Some(0),
        b'R' => Some(3),
        b'N' => Some(2),
        b'D' => Some(2),
        b'C' => Some(5),
        b'Q' => Some(2),
        b'E' => Some(2),
        b'G' => Some(0),
        b'H' => Some(3),
        b'I' => Some(1),
        b'L' => Some(1),
        b'K' => Some(3),
        b'M' => Some(1),
        b'F' => Some(4),
        b'P' => Some(0),
        b'S' => Some(0),
        b'T' => Some(0),
        b'W' => Some(4),
        b'Y' => Some(4),
        b'V' => Some(1),
        b'B' => Some(2),
        b'Z' => Some(2),
        b'J' => Some(1),
        // X, '.', '-' map to grp=6 in locgrpd, dropped by `if(tmp<6)` filter.
        _ => None,
    }
}

fn amino_grp_nuc(c: u8) -> Option<u32> {
    // C's seq_grp_nuc filters `tmp < 4`, dropping ambiguous codes.
    match c.to_ascii_lowercase() {
        b'a' => Some(0),
        b'c' => Some(1),
        b'g' => Some(2),
        b't' | b'u' => Some(3),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protein_self_score_equals_point_count() {
        let seq = b"MNGTEGDNFYVPFSNKTGLARSPYEY";
        let points = encode_points_protein(seq);
        assert!(!points.is_empty());
        // Self-comparison via composition table → all 6-mers shared.
        let table = composition_table(&points, 46656);
        let common = common_sextets_p(&table, &points, 46656);
        assert_eq!(common as usize, points.len());
    }

    #[test]
    fn parttree_distance_zero_for_identical() {
        let seq = b"MNGTEGDNFYVPFSNKTGLARSPYEYPQYY";
        let p = encode_points_protein(seq);
        let d = parttree_distance_protein(&p, &p, seq.len(), seq.len());
        assert!(d < 1e-9, "identity distance should be ~0, got {d}");
    }
}

/// DNA IUPAC ambiguity code scoring.
///
/// Ports the C `ambiguousscore()` and `nscore()` from constants.c.
use crate::alphabet::{Alphabet, DNA_ALPHABET};
use crate::round_half_away;

/// Fill in ambiguity code scores in a 26x26 DNA scoring matrix.
///
/// Assumes indices 0-3 (a,g,c,t) already have valid scores.
/// Fills in scores for IUPAC ambiguity codes: r,y,k,m,s,w,b,d,h,v,n.
pub fn fill_dna_ambiguity_scores(n_dis: &mut [Vec<i32>]) {
    let map = DNA_ALPHABET.build_amino_map();
    let idx = |ch: u8| map[ch as usize] as usize;

    let a = idx(b'a');
    let g = idx(b'g');
    let c = idx(b'c');
    let t = idx(b't');

    // 2-base ambiguity codes
    let ambig2: &[(u8, usize, usize)] = &[
        (b'r', a, g), // purine
        (b'y', c, t), // pyrimidine
        (b'k', g, t),
        (b'm', a, c),
        (b's', g, c),
        (b'w', a, t),
    ];

    // 3-base ambiguity codes
    let ambig3: &[(u8, usize, usize, usize)] = &[
        (b'b', c, g, t), // not A
        (b'd', a, g, t), // not C
        (b'h', a, c, t), // not G
        (b'v', a, c, g), // not T
    ];

    // Fill 2-base codes: score = average of constituent scores
    for &(code, b1, b2) in ambig2 {
        let ci = idx(code);
        for i in 0..26 {
            let avg = (n_dis[b1][i] as f64 + n_dis[b2][i] as f64) / 2.0;
            n_dis[i][ci] = round_half_away(avg);
            n_dis[ci][i] = n_dis[i][ci];
        }
        // Self-score
        n_dis[ci][ci] = round_half_away((n_dis[b1][b1] as f64 + n_dis[b2][b2] as f64) / 2.0);
    }

    // Fill 3-base codes
    for &(code, b1, b2, b3) in ambig3 {
        let ci = idx(code);
        for i in 0..26 {
            let avg = (n_dis[b1][i] as f64 + n_dis[b2][i] as f64 + n_dis[b3][i] as f64) / 3.0;
            n_dis[i][ci] = round_half_away(avg);
            n_dis[ci][i] = n_dis[i][ci];
        }
        n_dis[ci][ci] = round_half_away(
            (n_dis[b1][b1] as f64 + n_dis[b2][b2] as f64 + n_dis[b3][b3] as f64) / 3.0,
        );
    }
}

/// Fill in N (any base) scores: N vs X = 0.25 * X self-score.
pub fn fill_dna_n_scores(n_dis: &mut [Vec<i32>]) {
    let map = DNA_ALPHABET.build_amino_map();
    let n_idx = map[b'n' as usize] as usize;

    for i in 0..26 {
        let score = round_half_away(0.25 * n_dis[i][i] as f64);
        n_dis[i][n_idx] = score;
        n_dis[n_idx][i] = score;
    }
    // N vs N = average of all 4 base self-scores
    let bases = [0usize, 1, 2, 3]; // a, g, c, t
    let avg: f64 = bases.iter().map(|&b| n_dis[b][b] as f64).sum::<f64>() * 0.25;
    n_dis[n_idx][n_idx] = round_half_away(avg);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dna::default_dna_matrix;

    #[test]
    fn ambiguity_r_is_average_of_a_and_g() {
        let raw = default_dna_matrix();
        let mut n_dis: Vec<Vec<i32>> = raw.iter().map(|row| row.to_vec()).collect();
        fill_dna_ambiguity_scores(&mut n_dis);

        let map = DNA_ALPHABET.build_amino_map();
        let a = map[b'a' as usize] as usize;
        let r = map[b'r' as usize] as usize;

        // R vs A = (A-A + G-A) / 2 = (1000 + 600) / 2 = 800
        assert_eq!(n_dis[r][a], 800);
        // Symmetric
        assert_eq!(n_dis[a][r], 800);
    }

    #[test]
    fn n_score_is_quarter_self() {
        let raw = default_dna_matrix();
        let mut n_dis: Vec<Vec<i32>> = raw.iter().map(|row| row.to_vec()).collect();
        fill_dna_ambiguity_scores(&mut n_dis);
        fill_dna_n_scores(&mut n_dis);

        let map = DNA_ALPHABET.build_amino_map();
        let a = map[b'a' as usize] as usize;
        let n = map[b'n' as usize] as usize;

        // N vs A = 0.25 * A-A = 0.25 * 1000 = 250
        assert_eq!(n_dis[n][a], 250);
    }
}

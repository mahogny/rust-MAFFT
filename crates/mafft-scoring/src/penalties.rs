/// Gap penalty parameters and defaults.
///
/// Ports the C `DEFAULTGOP_*`, `DEFAULTGEP_*`, `DEFAULTOFS_*` constants
/// and the penalty scaling logic from `constants()`.

/// Complete gap parameter set (matching the C globals).
#[derive(Debug, Clone)]
pub struct GapParams {
    /// Scaled gap opening penalty.
    pub penalty: i32,
    /// Scaled gap extension penalty.
    pub penalty_ex: i32,
    /// Scaled offset.
    pub offset: i32,
    /// Penalty for LN (log-normal) scoring.
    pub penalty_ln: i32,
    /// Extension for LN scoring.
    pub penalty_ex_ln: i32,
    /// Offset for LN scoring.
    pub offset_ln: i32,
    /// Offset for FFT scoring.
    pub offset_fft: i32,
}

// C defaults (pre-scaling "p" values)
const DEFAULTGOP_N: f64 = -1530.0;
const DEFAULTGEP_N: f64 = 0.0;

const DEFAULTGOP_B: f64 = -1530.0;
const DEFAULTGEP_B: f64 = 0.0;

/// Scaling factor for DNA penalties: 3 * 600 / 1000.
const DNA_SCALE: f64 = 3.0 * 600.0 / 1000.0;

/// Scaling factor for protein penalties: 600 / 1000.
const PROTEIN_SCALE: f64 = 600.0 / 1000.0;

fn scale(value: f64, factor: f64) -> i32 {
    (factor * value + 0.5) as i32
}

/// Default gap parameters for DNA alignment.
///
/// Matches what the MAFFT shell script (`mafft.tmpl`, `defaultaof="0.000"`)
/// passes to the C binaries via `-h 0.000`: `poffset = 0`, giving `offset = 0`.
/// C's `constants()` defaults `poffset = DEFAULTOFS_N = -369` (→ offset = -221)
/// when invoked without the shell script, but that path never runs in practice.
pub fn default_dna_gap_params() -> GapParams {
    GapParams {
        penalty: scale(DEFAULTGOP_N, DNA_SCALE),
        penalty_ex: scale(DEFAULTGEP_N, DNA_SCALE),
        offset: 0, // matches mafft.tmpl defaultaof="0.000" (-h 0.000)
        penalty_ln: scale(-2000.0, DNA_SCALE),
        penalty_ex_ln: scale(-100.0, DNA_SCALE),
        offset_ln: scale(100.0, 1.0 * 600.0 / 1000.0),
        offset_fft: 0,
    }
}

/// Default gap parameters for protein alignment (BLOSUM / JTT).
///
/// Matches what the MAFFT shell script (`mafft.tmpl`, `defaultaof="0.000"`)
/// passes to the C binaries via `-h 0.000`: `poffset = 0`, giving `offset = 0`.
/// C's `constants()` defaults `poffset = DEFAULTOFS_B = -123` (→ offset = -73)
/// when invoked without the shell script, but that path never runs in practice.
pub fn default_protein_gap_params() -> GapParams {
    GapParams {
        penalty: scale(DEFAULTGOP_B, PROTEIN_SCALE),
        penalty_ex: scale(DEFAULTGEP_B, PROTEIN_SCALE),
        offset: 0, // matches mafft.tmpl defaultaof="0.000" (-h 0.000)
        penalty_ln: scale(-2000.0, PROTEIN_SCALE),
        penalty_ex_ln: scale(-100.0, PROTEIN_SCALE),
        offset_ln: scale(100.0, PROTEIN_SCALE),
        offset_fft: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dna_penalty_values() {
        let p = default_dna_gap_params();
        // penalty = (int)(3 * 600.0 / 1000.0 * -1530 + 0.5) = (int)(-2754 + 0.5)
        // In C: shishagonyuu(-2754) = -2754
        assert_eq!(p.penalty, -2753); // scale rounds toward zero
    }

    /// Full per-alphabet audit of everything `constants()` derives, so a
    /// future edit cannot silently give one alphabet the other's constant.
    /// Three bugs of that exact shape have been found in this tree.
    /// C: nucleotide `constants.c:308-326`, protein `:672-682`.
    #[test]
    fn every_constants_c_value_matches_for_both_alphabets() {
        let n = default_dna_gap_params();
        let p = default_protein_gap_params();
        // gap penalties: nucleotide carries C's `3 *`, protein does not
        assert_eq!(n.penalty, -2753, "nuc penalty = (int)(3*0.6*-1530+0.5)");
        assert_eq!(p.penalty, -917, "protein penalty = (int)(0.6*-1530+0.5)");
        assert_eq!(n.penalty_ex, 0, "nuc DEFAULTGEP_N = 0");
        assert_eq!(p.penalty_ex, 0, "protein DEFAULTGEP_B = 0");
        assert_eq!(
            n.penalty_ln, -3599,
            "nuc penaltyLN = (int)(3*0.6*-2000+0.5)"
        );
        assert_eq!(
            p.penalty_ln, -1199,
            "protein penaltyLN = (int)(0.6*-2000+0.5)"
        );
        assert_eq!(
            n.penalty_ex_ln, -179,
            "nuc penalty_exLN = (int)(3*0.6*-100+0.5)"
        );
        assert_eq!(
            p.penalty_ex_ln, -59,
            "protein penalty_exLN = (int)(0.6*-100+0.5)"
        );
        // offsets: C uses `1 *` for nucleotide here, NOT `3 *`
        assert_eq!(
            n.offset_ln, 60,
            "nuc offsetLN = (int)(1*0.6*100+0.5) -- 1x, not 3x"
        );
        assert_eq!(p.offset_ln, 60, "protein offsetLN = (int)(0.6*100+0.5)");
        assert_eq!(
            n.offset_ln, p.offset_ln,
            "offsetLN is the same for both alphabets"
        );
        assert_eq!(n.offset_fft, 0);
        assert_eq!(p.offset_fft, 0);
        // the script passes -h 0.000, so poffset = 0 on both paths
        assert_eq!(n.offset, 0);
        assert_eq!(p.offset, 0);
    }

    #[test]
    fn protein_penalty_values() {
        let p = default_protein_gap_params();
        // penalty = (int)(600.0 / 1000.0 * -1530 + 0.5) = (int)(-918 + 0.5)
        assert_eq!(p.penalty, -917);
        // offset = 0 because mafft.tmpl sets defaultaof="0.000" (poffset = 0).
        assert_eq!(p.offset, 0);
    }
}

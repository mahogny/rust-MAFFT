/// Shared types for dynamic programming alignment.

/// An individual alignment operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignOp {
    /// Match or mismatch (consume one residue from each sequence).
    Match,
    /// Gap in sequence 1 (insertion in seq2).
    Insert,
    /// Gap in sequence 2 (deletion from seq1).
    Delete,
}

/// A pairwise alignment result.
#[derive(Debug, Clone)]
pub struct Alignment {
    /// Aligned sequence 1 (with gap characters '-' inserted).
    pub seq1: Vec<u8>,
    /// Aligned sequence 2 (with gap characters '-' inserted).
    pub seq2: Vec<u8>,
    /// The alignment score.
    pub score: f64,
    /// The sequence of alignment operations.
    pub operations: Vec<AlignOp>,
}

impl Alignment {
    /// Length of the alignment (including gaps).
    pub fn len(&self) -> usize {
        self.seq1.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seq1.is_empty()
    }

    /// Count the number of identical positions.
    pub fn identity_count(&self) -> usize {
        self.seq1
            .iter()
            .zip(self.seq2.iter())
            .filter(|(a, b)| a == b && **a != b'-')
            .count()
    }

    /// Fractional identity (identical positions / aligned length excluding gaps).
    pub fn identity(&self) -> f64 {
        let aligned = self
            .seq1
            .iter()
            .zip(self.seq2.iter())
            .filter(|(a, b)| **a != b'-' && **b != b'-')
            .count();
        if aligned == 0 {
            0.0
        } else {
            self.identity_count() as f64 / aligned as f64
        }
    }
}

/// Gap penalty model.
#[derive(Debug, Clone)]
pub struct GapModel {
    /// Gap opening penalty (negative value).
    pub open: f64,
    /// Gap extension penalty (negative value).
    pub extend: f64,
    /// Shift/warp penalty for long-range gap jumps (--allowshift).
    /// None = disabled (default). Some(penalty) = enabled.
    pub shift: Option<f64>,
    /// `--leavegappyregion` / `--legacygappenalty` (C `legacygapcost = 1`).
    /// When true, the profile-DP treats every column as fully nongap
    /// (`gapfreq[i] = 1.0`), disabling the gap-aware reweighting
    /// introduced in MAFFT 7.110. Mirrors `Salignmm.c:1592-1610`.
    pub legacy_gap_cost: bool,
}

impl GapModel {
    pub fn new(open: f64, extend: f64) -> Self {
        Self {
            open,
            extend,
            shift: None,
            legacy_gap_cost: false,
        }
    }

    /// Create with shift/warp enabled.
    pub fn with_shift(mut self, shift_penalty: f64) -> Self {
        self.shift = Some(shift_penalty);
        self
    }

    /// Create with legacy (pre-7.110) gap-cost behaviour.
    pub fn with_legacy_gap_cost(mut self, legacy: bool) -> Self {
        self.legacy_gap_cost = legacy;
        self
    }
}

impl Default for GapModel {
    /// PROTEIN defaults. C scales gap penalties per alphabet
    /// (`constants.c:672` vs `:316`), so this is **not** a safe default on a
    /// nucleotide path — DNA's gap-open is `-2753`, three times this. Callers
    /// aligning nucleotides must build the model from
    /// `mafft_scoring::default_dna_gap_params()` (or pass one in) rather than
    /// take this. Two bugs of exactly that shape have already been found and
    /// fixed elsewhere in the tree.
    fn default() -> Self {
        Self {
            // C: penalty = (int)( 600/1000 * DEFAULTGOP_B + 0.5 )
            //            = (int)( -918 + 0.5 ) = (int)( -917.5 ) = -917
            // (C casts toward zero, so this is -917, not -918.)
            open: -917.0,
            extend: 0.0, // C: DEFAULTGEP_B = 0
            shift: None,
            legacy_gap_cost: false,
        }
    }
}

/// Convert an integer substitution matrix to f64 (used to bridge from
/// `ScoringContext.substitution_matrix: Vec<Vec<i32>>` to the DP layer
/// which now operates on `Vec<Vec<f64>>` for the per-step / per-pair
/// dynamic-matrix path that requires sub-integer precision).
pub fn matrix_i32_to_f64(matrix: &[Vec<i32>]) -> Vec<Vec<f64>> {
    matrix
        .iter()
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect()
}

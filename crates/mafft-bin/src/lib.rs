use std::io::{self, BufReader, Write};
use std::path::PathBuf;

use clap::Parser;

use mafft_core::{MafftEngine, AlignmentMode};
use mafft_io::{read_fasta, read_fasta_from_reader, read_fasta_casepreserve, read_fasta_from_reader_casepreserve};
use mafft_types::{Sequence, SequenceSet, ScoringModel};

pub mod builder;
pub mod progress;
pub use builder::Mafft;
pub use progress::{Progress, SilentProgress, StderrProgress};

/// MAFFT-rs: Multiple sequence alignment (Rust implementation)
#[derive(Parser, Debug)]
#[command(name = "mafft-rs", version, about)]
struct Args {
    /// Input FASTA file (reads from stdin if omitted)
    #[arg(value_name = "INPUT")]
    input: Option<PathBuf>,

    /// Output file (writes to stdout if omitted)
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    // --- Algorithm selection ---
    /// Use L-INS-i (local, iterative; most accurate for <200 seqs)
    #[arg(long)]
    localpair: bool,

    /// Use G-INS-i (global, iterative; for globally alignable seqs)
    #[arg(long)]
    globalpair: bool,

    /// Use E-INS-i (generalized affine, iterative; for seqs with large gaps)
    #[arg(long)]
    genafpair: bool,

    /// Add new sequences to an existing alignment (provide aligned FASTA as INPUT,
    /// new sequences as --add FILE)
    #[arg(long, value_name = "FILE")]
    add: Option<PathBuf>,

    /// Add fragment sequences to an existing alignment (same as --add but for short fragments)
    #[arg(long, value_name = "FILE")]
    addfragments: Option<PathBuf>,

    /// Preserve existing alignment column structure when adding sequences
    #[arg(long)]
    keeplength: bool,

    /// Enable per-step dynamic matrix scaling (sets unalignlevel=0.8). Allows
    /// divergent regions to stay unaligned at shallow merges. Requires
    /// `--globalpair`.
    #[arg(long)]
    allowshift: bool,

    /// Per-step substitution-score offset = (distfromtip - unalignlevel) * 600
    /// when distfromtip < unalignlevel, else 0. Default 0 = no scaling.
    /// `--allowshift` sets this to 0.8 if not explicitly given.
    #[arg(long, value_name = "F")]
    unalignlevel: Option<f64>,

    /// Maximum number of iterative refinement cycles. Unset → mode default
    /// (1000 for INS-i modes, 0 for FFT-NS-2). Explicit 0 disables refinement.
    #[arg(long)]
    maxiterate: Option<usize>,

    /// Number of guide tree rebuilds [default: 2]
    #[arg(long, default_value_t = 2)]
    retree: usize,

    /// Disable FFT: force pure DP for all alignment steps (NW-NS-2 mode)
    #[arg(long)]
    nofft: bool,

    /// Use PartTree guide tree for large datasets (10K+ sequences)
    #[arg(long)]
    parttree: bool,

    /// Use DP-based PartTree (more accurate than --parttree, slower)
    #[arg(long)]
    dpparttree: bool,

    /// Group size for PartTree partitioning [default: 150]
    #[arg(long)]
    groupsize: Option<usize>,

    /// Use Q-INS-i: RNA secondary structure from McCaskill base-pair probabilities
    #[arg(long)]
    qinsi: bool,

    /// Use X-INS-i: RNA secondary structure from CONTRAfold predictions
    #[arg(long)]
    xinsi: bool,

    /// Use SCARNA-like structural alignment via DASH
    #[arg(long)]
    scarnalike: bool,

    /// PDB ID list file for structure-aware alignment.
    /// **Non-functional**: matches C MAFFT 7.526's behaviour — the
    /// upstream `scripts/mafft:969-983` disables this flag with
    /// "temporarily unavailable, 2018/Dec." and `exit`s before any
    /// structural alignment runs. Rust accepts the flag and exits
    /// with the same message rather than reporting an unknown arg.
    #[arg(long, value_name = "FILE")]
    pdbidlist: Option<std::path::PathBuf>,

    /// PDB file list for structure-aware alignment.
    /// **Non-functional**: same status as `--pdbidlist` — C MAFFT
    /// 7.526 (`scripts/mafft:980-990`) disables this with
    /// "temporarily unavailable, 2018/Dec." and `exit`s. Rust
    /// matches that behaviour.
    #[arg(long, value_name = "FILE")]
    pdbfilelist: Option<std::path::PathBuf>,

    /// Output format: fasta (default), clustal, phylip
    #[arg(long, default_value = "fasta")]
    format: String,

    /// FASTA line width (0 for unlimited) [default: 60]
    #[arg(long, default_value_t = 60)]
    linewidth: usize,

    /// Name field width in CLUSTAL/PHYLIP output. Default: 15 for
    /// CLUSTAL, 10 for PHYLIP (matches C MAFFT's `clustalout_pointer` /
    /// `phylipout_pointer`). Names longer than the field are truncated.
    #[arg(long, value_name = "N")]
    namelength: Option<usize>,

    // --- Scoring parameters ---
    /// Gap opening penalty (positive float, e.g. 1.53) [default: 1.53]
    #[arg(long)]
    op: Option<f64>,

    /// Offset (gap extension-like penalty, positive float, e.g. 0.123) [default: 0.123]
    #[arg(long)]
    ep: Option<f64>,

    /// Gap extension penalty (`--exp`). Positive float (e.g. 0.1); negated
    /// internally to match C's `gexp = -1.0 * arg` convention. Default 0
    /// (no per-residue extension cost).
    #[arg(long)]
    exp: Option<f64>,

    /// L-INS-i pairwise gap-open (`--lop`). Signed float, no negation
    /// (matches C: `lgop=-2.00` default). `allow_hyphen_values` so
    /// negative numbers like `-3.0` are parsed as the value, not a flag.
    #[arg(long, allow_hyphen_values = true)]
    lop: Option<f64>,

    /// L-INS-i pairwise offset (`--lep`). Signed float, no negation
    /// (matches C: `laof=0.100` default).
    #[arg(long, allow_hyphen_values = true)]
    lep: Option<f64>,

    /// L-INS-i pairwise gap-extend (`--lexp`). Signed float, no negation
    /// (matches C: `lexp=-0.100` default).
    #[arg(long, allow_hyphen_values = true)]
    lexp: Option<f64>,

    /// X-INS-i / Q-INS-i generalized-affine pair gap-open (`--gop`).
    /// Inert in protein/DNA pipelines (only used by RNA structure
    /// modes — see `TODO.md` external-dep limitations). Default
    /// `pggop=-1.53` in C.
    #[arg(long, value_name = "N", allow_hyphen_values = true)]
    gop: Option<f64>,

    /// X-INS-i / Q-INS-i generalized-affine pair offset (`--gep`).
    /// Inert in protein/DNA pipelines. Default `pgaof=0.10`.
    #[arg(long, value_name = "N", allow_hyphen_values = true)]
    gep: Option<f64>,

    /// X-INS-i / Q-INS-i generalized-affine pair gap-extend (`--gexp`).
    /// Inert in protein/DNA pipelines. Default `pgexp=-0.10`.
    #[arg(long, value_name = "N", allow_hyphen_values = true)]
    gexp: Option<f64>,

    /// RNA-only: ribosum gap-open penalty (`--rop`, C variable `rgop`,
    /// default `-1.530`). Forwarded to `rnaopt` in C MAFFT only for
    /// the RNA-structure paths (mccaskill / contrafold / dafs /
    /// rnaalifold), all of which depend on external binaries we don't
    /// ship. Accepted at the CLI for compatibility but emits a
    /// "no-op without --xinsi/--qinsi" warning if used.
    #[arg(long, value_name = "N", allow_hyphen_values = true)]
    rop: Option<f64>,

    /// RNA-only: ribosum gap-extend penalty (`--rep`, C variable `rgep`,
    /// default `-0.000`). Same RNA-structure-only gating as `--rop`.
    #[arg(long, value_name = "N", allow_hyphen_values = true)]
    rep: Option<f64>,

    /// LARA-only: gap-open penalty for the LARA RNA path (`--LOP`,
    /// C variable `LGOP`, default `-6.00`). LARA mode requires the
    /// `lara` binary which is not shipped here. Accepted at the CLI
    /// with a warning if used outside a LARA-mode flag.
    #[arg(long = "LOP", value_name = "N", allow_hyphen_values = true)]
    lop_lara: Option<f64>,

    /// LARA-only: gap-extend penalty for the LARA RNA path
    /// (`--LEXP`, C variable `LEXP`). Same gating as `--LOP`.
    #[arg(long = "LEXP", value_name = "N", allow_hyphen_values = true)]
    lexp_lara: Option<f64>,

    /// LARA-only: alternative-format gap-open penalty (`--GOP`,
    /// C variable `GGOP`, default `-6.00`). Same gating as `--LOP`.
    #[arg(long = "GOP", value_name = "N", allow_hyphen_values = true)]
    gop_lara: Option<f64>,

    /// LARA-only: alternative-format gap-extend penalty (`--GEXP`,
    /// C variable `GEXP`). Same gating as `--LOP`.
    #[arg(long = "GEXP", value_name = "N", allow_hyphen_values = true)]
    gexp_lara: Option<f64>,

    /// Shift penalty factor for `--allowshift` (`--shiftpenalty`).
    /// Multiplied by the gap-open penalty to get the per-cell shift cost.
    /// Default 2.0 (matches C `spfactor=2.0` when `--allowshift` is on).
    #[arg(long)]
    shiftpenalty: Option<f64>,

    /// BLOSUM matrix number (30, 45, 50, 62, 80). Only used with --localpair/--globalpair when scoring with BLOSUM.
    #[arg(long)]
    bl: Option<i32>,

    /// JTT PAM number for substitution scoring (e.g. 100, 200). Mirrors `mafft --jtt N` (default PAM 200).
    #[arg(long, conflicts_with_all = ["bl", "tm"])]
    jtt: Option<i32>,

    /// Transmembrane (TM) PAM number for substitution scoring (e.g. 100, 200). Mirrors `mafft --tm N` (default PAM 200).
    #[arg(long, conflicts_with_all = ["bl", "jtt"])]
    tm: Option<i32>,

    /// Kimura R parameter for DNA distance model [default: 2]
    #[arg(long)]
    kimura: Option<i32>,

    /// Number of threads (0 = use all available cores) [default: 0]
    #[arg(long, default_value_t = 0)]
    thread: usize,

    /// Quiet mode: suppress progress messages
    #[arg(long, short)]
    quiet: bool,

    /// Print the citation for MAFFT (and rust-MAFFT, once published) and
    /// exit. rust-MAFFT is a port of MAFFT by Kazutaka Katoh et al.; the
    /// scientific contribution is theirs and must be cited in any
    /// published work.
    #[arg(long)]
    cite: bool,

    /// Output sequences in guide-tree DFS order (matching C MAFFT `--reorder`).
    #[arg(long, conflicts_with = "inputorder")]
    reorder: bool,

    /// Output sequences in input order (default; matches C MAFFT `--inputorder`).
    #[arg(long)]
    inputorder: bool,

    /// Write the guide tree to `<INPUT>.tree` in Newick format (matches
    /// C MAFFT `--treeout`). Ignored when input is read from stdin.
    #[arg(long)]
    treeout: bool,

    /// Write the pairwise distance matrix used by the guide-tree
    /// construction to `<INPUT>.hat2` (matches C MAFFT `--distout`).
    /// Ignored when reading from stdin (no path to derive the output
    /// name from). The matrix is whatever the engine actually used
    /// for tree construction — k-mer for FFT-NS-2/FFT-NS-i, pairwise
    /// alignment-score-derived for L-INS-i / G-INS-i / E-INS-i.
    #[arg(long)]
    distout: bool,

    /// Print the unweighted sum-of-pairs score of the final alignment
    /// to stderr (matches C MAFFT `--scoreout`'s
    /// `Unweighted sum-of-pairs score = N.NNNNN` line).
    #[arg(long)]
    scoreout: bool,

    /// Write the guide tree to `<INPUT>.tree` followed by a per-leaf
    /// `Density:` section and a `Node info:` section (matches C MAFFT
    /// `--nodeout` → `treeout==2` path in `mltaln9.c:6492-6518`).
    /// Implies `--treeout`.
    #[arg(long)]
    nodeout: bool,

    /// Pileup alignment strategy: build a comb-tree guide
    /// (sequence 0 joins 1, then that pair joins 2, etc.) instead
    /// of UPGMA, and run a single progressive pass with no
    /// refinement. Mirrors C MAFFT's `--pileup` ("Pileup-NS-1"
    /// strategy, `scripts/mafft:2169` + `mltaln9.c::createchain`).
    #[arg(long)]
    pileup: bool,

    /// Write the per-input-column mapping (gap-insertion positions)
    /// to `<ADDFILE>.map`. Requires `--add` / `--addfragments` plus
    /// `--keeplength` (matches C MAFFT `--mapout` → internal
    /// `-Z -Y`). Mirrors `reconstructdeletemap` (addfunctions.c:1985).
    #[arg(long)]
    mapout: bool,

    /// Compact form of `--mapout`. Same requirements. Mirrors C's
    /// `reconstructdeletemap_compact` (`addfunctions.c:2047`).
    #[arg(long)]
    compactmapout: bool,

    /// Filter input sequences whose ambiguous-residue fraction exceeds
    /// N (range 0.0–1.0). Mirrors C MAFFT's `--maxambiguous`
    /// (`filter.c`): for protein, ambiguous = anything outside
    /// `ARNDCQEGHILKMFPSTWYV` (case-insensitive). For DNA/RNA,
    /// ambiguous = anything outside `ATGCU`. Sequences exceeding the
    /// threshold are removed before alignment; runs of N/X are
    /// collapsed to a single character (`shortenN`). Default 1.0 (no
    /// filtering).
    #[arg(long, value_name = "F")]
    maxambiguous: Option<f64>,

    /// Floor for per-sequence weights used in refinement and
    /// progressive merge. Mirrors C's `tbfast -W $minimumweight`
    /// (`scripts/mafft:1029`). Sequences with weight below this floor
    /// are clamped up to it. Default 0.00001.
    #[arg(long, value_name = "F")]
    minimumweight: Option<f64>,

    /// Treat N (DNA/RNA ambiguous) as a wildcard that matches anything
    /// positively (mirrors C MAFFT's `--nwildcard`, internal flag
    /// `-:`). Currently accepted but the N-row scoring tweak is not
    /// yet wired; affects only DNA workflows. See `TODO.md`.
    #[arg(long)]
    nwildcard: bool,

    /// Treat N (DNA/RNA ambiguous) as scoring 0 against everything
    /// (mirrors C MAFFT's `--nzero`). Currently accepted but the
    /// N-row scoring tweak is not yet wired. See `TODO.md`.
    #[arg(long)]
    nzero: bool,

    /// Exclude near-identical sequences during alignment (mirrors
    /// `--excludehomologs`). Documented in C as "works with --dash
    /// only"; we don't support `--dash`, so this flag is a no-op
    /// for now.
    #[arg(long)]
    excludehomologs: bool,

    /// Output only the original input sequences (mirrors C's
    /// `--originalseqonly`). Documented as "works with --dash only";
    /// no-op without `--dash`.
    #[arg(long)]
    originalseqonly: bool,

    /// Use pure average linkage for UPGMA cluster joining (sueff = 1.0,
    /// matches C `--averagelinkage` / `tbfast -X 1.0`). Mutually
    /// exclusive with `--minimumlinkage` and `--mixedlinkage`.
    #[arg(long, conflicts_with_all = ["minimumlinkage", "mixedlinkage"])]
    averagelinkage: bool,

    /// Use pure single-linkage (minimum) for UPGMA cluster joining
    /// (sueff = 0.0, matches C `--minimumlinkage` / `tbfast -X 0.0`).
    #[arg(long, conflicts_with_all = ["averagelinkage", "mixedlinkage"])]
    minimumlinkage: bool,

    /// Use a weighted mix of minimum and average linkage for UPGMA
    /// cluster joining (`--mixedlinkage F` → sueff = F, matches C
    /// `tbfast -X F`). Range 0.0–1.0. The C default is 0.1; this
    /// flag is for explicit overrides.
    #[arg(long, value_name = "F", conflicts_with_all = ["averagelinkage", "minimumlinkage"])]
    mixedlinkage: Option<f64>,

    /// Use the "youngest" linkage scheme (matches C
    /// `--youngestlinkage`). C's `youngestlinkage` is a separate
    /// algorithm, not just a `sueff` value, and is NOT yet ported.
    /// Flag accepted as a no-op for compatibility. See `TODO.md`.
    #[arg(long)]
    youngestlinkage: bool,

    /// Use C MAFFT's `BESTFIRST` parallelisation strategy for
    /// iterative refinement — score every branch first, then refine
    /// in score-order. Default is `BAATARI2` (simple hill climbing).
    /// Currently accepted as a no-op; the BESTFIRST refinement loop
    /// architecture is a separate port. See `TODO.md`.
    #[arg(long)]
    bestfirst: bool,

    /// Use the simple hill-climbing refinement strategy (matches C
    /// `--simplehillclimbing` → `parallelizationstrategy=BAATARI2`).
    /// This IS the default in both C MAFFT and mafft-rs, so the flag
    /// is a true no-op except when overriding a prior `--bestfirst`.
    #[arg(long)]
    simplehillclimbing: bool,

    /// Skip refinement of branches whose distance-from-tip exceeds F
    /// (matches C `--skipiterate F` → `dvtditr -E $fixthreshold`).
    /// Currently accepted as a no-op; the branch-skip gate is not
    /// yet wired into the refinement loop. See `TODO.md`.
    #[arg(long, value_name = "F", allow_hyphen_values = true)]
    skipiterate: Option<f64>,

    /// Run only one iteration of the disttbfast distance refinement
    /// (matches C `--oneiteration` → `disttbfast -r`). Currently
    /// accepted as a no-op; our distance recomputation already
    /// runs once per retree pass — closer investigation needed
    /// before forcing a single iteration here. See `TODO.md`.
    #[arg(long)]
    oneiteration: bool,

    /// Auto-detect input DNA strand orientation and reverse-complement
    /// sequences on the wrong strand before alignment (matches C
    /// `--adjustdirection`). Implemented: the k-mer-based detection
    /// algorithm is a port of
    /// `mafft-upstream/core/makedirectionlist.c` (see
    /// `mafft_core::adjust_direction`, `TODO.md` R-5, resolved
    /// 2026-06-03). Affects DNA workflows only; protein inputs
    /// silently bypass it.
    #[arg(long)]
    adjustdirection: bool,

    /// Slower, more accurate variant of `--adjustdirection` (matches
    /// C `--adjustdirectionaccurately`, internally
    /// `adjustdirection=2`). Also implemented — it selects the DP
    /// scorer (`AdjustMode::Dp`) instead of the k-mer one.
    #[arg(long, conflicts_with = "adjustdirection")]
    adjustdirectionaccurately: bool,

    /// Use a user-supplied guide tree (matches C MAFFT `--treein FILE`).
    /// FILE must be in MAFFT's internal tree format: nseq-1 lines of
    /// `im jm len0 len1` (1-indexed sequence numbers, im < jm). Convert
    /// a standard Newick file with `mafft-upstream/core/newick2mafft.rb`.
    #[arg(long, value_name = "FILE")]
    treein: Option<std::path::PathBuf>,

    /// Automatically select alignment strategy based on input size, matching
    /// C MAFFT `--auto` (`scripts/mafft:1290-1343`). Picks L-INS-i, FFT-NS-i,
    /// FFT-NS-2, FFT-NS-1, --dpparttree, or --parttree depending on the
    /// number of sequences and the longest sequence length. Overrides any
    /// other algorithm-selection flag.
    #[arg(long)]
    auto: bool,

    /// Use the memory-saving guide-tree algorithm (matches C MAFFT
    /// `--memsavetree`). Builds a UPGMA-like tree using k-mer distances
    /// computed on the fly, avoiding the O(N²) memory cost of a full
    /// distance matrix. Recommended for very large inputs (100k+ seqs);
    /// also enabled automatically by `--auto` in that bracket.
    #[arg(long)]
    memsavetree: bool,

    /// Allow any non-standard characters in input (matches C MAFFT
    /// `--anysymbol`). Before alignment, non-standard residues are
    /// replaced with `X` (protein) or `n` (DNA) so the alignment DP
    /// can score them; the alignment is then post-processed to
    /// restore the original characters (and case). Mirrors C's
    /// `replaceu` + `restoreu` external steps.
    #[arg(long)]
    anysymbol: bool,

    /// Alias for `--anysymbol` (matches C MAFFT `--preservecase`).
    /// C maps both flags to the same internal `anysymbol=1` variable.
    #[arg(long)]
    preservecase: bool,

    /// Restore the pre-7.110 gap-cost behaviour (matches C MAFFT
    /// `--leavegappyregion` / `--legacygappenalty`). The profile DP
    /// stops down-weighting columns by their gap fraction
    /// (`legacygapcost = 1` — `Salignmm.c:1604-1610`), which lets
    /// alignments leave heavily-gapped regions untouched instead of
    /// inserting more gaps to align around them.
    #[arg(long, alias = "legacygappenalty")]
    leavegappyregion: bool,

    /// Use a pre-aligned seed alignment as a strong-importance
    /// constraint (matches C MAFFT `--seed FILE`). The flag is
    /// repeatable — every seed file's sequences are prepended to the
    /// user input with a `_seed_` name prefix, and all in-group
    /// pairs are written to a `hat3.seed`-style local-homology table
    /// with `opt` multiplied by `tsuyosa = user_nseq² * 100` so the
    /// refinement DP follows them tightly. Forces
    /// `maxiterate ≥ 2` (`scripts/mafft:1911-1923`).
    #[arg(long = "seed", value_name = "FILE")]
    seed_files: Vec<PathBuf>,

    /// Pre-computed seed local-homology table (matches C MAFFT
    /// `--seedtable FILE`, `scripts/mafft:1021-1024`,
    /// `scripts/mafft:2437-2438`). The file follows the same 9-field text
    /// format `multi2hat3s.c:214` writes for `--seed`:
    ///   `i j overlapaa opt start1 end1 start2 end2 k`
    /// (one record per line, 0-based sequence indices and inclusive
    /// residue positions; `opt` is the already-`tsuyosa`-boosted score).
    /// Unlike `--seed`, no sequences are prepended — the file's `i`/`j`
    /// reference whatever indices the user's input FASTA contains.
    /// Mutually exclusive with `--seed`, `--add`/`--addfragments`,
    /// `--parttree`/`--dpparttree`, and `--memsave`. Forces
    /// `maxiterate ≥ 2` (`scripts/mafft:1911-1923`).
    #[arg(long = "seedtable", value_name = "FILE")]
    seedtable: Option<PathBuf>,

    /// Memory-saving mode (matches C MAFFT `--memsave` →
    /// `tbfast -M -B`, `scripts/mafft:543-544`). In C, this routes the
    /// profile DP through `MSalignmm` (Hirschberg-style linear-space
    /// divide-and-conquer) for the group merge, avoiding the O(N×M)
    /// allocation. For sequences that fit in memory (≤ 30000 residues)
    /// the alignment is byte-identical to default-mode output — C's
    /// auto-switch at `len > 30000` (`tbfast.c:1096`) makes the two
    /// paths converge for typical inputs. Our engine uses full-memory
    /// DP regardless; the flag is accepted (for CLI parity) and
    /// validated against the same script-level gating as C. NOTE: the
    /// actual Hirschberg DP is not yet ported — long sequences that
    /// would auto-trigger C's memsave path may OOM here. Tracked in
    /// TODO §B.3.
    #[arg(long)]
    memsave: bool,

    /// Disable the auto-switch to memory-saving DP for long sequences
    /// (matches C MAFFT `--nomemsave` → `tbfast -N`,
    /// `scripts/mafft:545-546`). In C this sets `nevermemsave = 1` so
    /// long-sequence inputs use the full-memory DP. We always use the
    /// full DP, so the flag is accepted for CLI parity and has no
    /// runtime effect.
    #[arg(long)]
    nomemsave: bool,

    /// Force the input to be treated as nucleotide, overriding the
    /// ATGC-frequency auto-detection (matches C MAFFT `--nuc` →
    /// `seqtype="-D"`, `scripts/mafft:547-548`). Mutually exclusive with
    /// `--amino`.
    #[arg(long, conflicts_with = "amino")]
    nuc: bool,

    /// Force the input to be treated as protein, overriding the
    /// ATGC-frequency auto-detection (matches C MAFFT `--amino` →
    /// `seqtype="-P"`, `scripts/mafft:549-550`). Mutually exclusive with
    /// `--nuc`.
    #[arg(long)]
    amino: bool,

    /// Replicate C MAFFT's static-TLS `reuseprofiles` memoization
    /// (`Salignmm.c:1446-1450`) so tied-DP-cell choices match C
    /// byte-for-byte. Off by default — the stateless progressive
    /// engine is the design goal. Enable when downstream byte-equality
    /// with C MAFFT 7.526 is hard-required on inputs that surface the
    /// `A__align` static-state artifact (BB20027-class cases, see
    /// `MAFFT_UPSTREAM_REPORT.md`).
    #[arg(long = "c-compat")]
    c_compat: bool,
}

/// Write the canonical MAFFT citation block to the output sink. Invoked
/// by the `--cite` flag (an early-exit path in `run_from()` that writes
/// this and returns; `run()` then exits 0). The sink is stdout for the
/// CLI, so the printed bytes are unchanged.
///
/// rust-MAFFT is a port — the science is by Katoh et al. The block
/// names Katoh & Standley 2013 as the primary citation (canonical for
/// MAFFT v7, which this is a port of) and points at the docs site for
/// mode-specific references and BibTeX.
fn print_citation(out: &mut dyn Write) -> io::Result<()> {
    writeln!(out, "rust-MAFFT v{} — port of MAFFT 7.526", env!("CARGO_PKG_VERSION"))?;
    writeln!(out)?;
    writeln!(out, "If you use this software in published work, please cite the")?;
    writeln!(out, "original MAFFT paper. The scientific contribution is by")?;
    writeln!(out, "Kazutaka Katoh and colleagues at CBRC; this Rust port preserves")?;
    writeln!(out, "their algorithm byte-for-byte.")?;
    writeln!(out)?;
    writeln!(out, "  Katoh, K., & Standley, D. M. (2013).")?;
    writeln!(out, "  MAFFT multiple sequence alignment software version 7:")?;
    writeln!(out, "  improvements in performance and usability.")?;
    writeln!(out, "  Molecular Biology and Evolution, 30(4), 772-780.")?;
    writeln!(out, "  doi: 10.1093/molbev/mst010")?;
    writeln!(out)?;
    writeln!(out, "Mode-specific references (cite additionally when relevant):")?;
    writeln!(out)?;
    writeln!(out, "  FFT-NS-1/2:        Katoh et al. 2002, NAR 30(14):3059-3066")?;
    writeln!(out, "                     doi: 10.1093/nar/gkf436")?;
    writeln!(out, "  L/G/E-INS-i:       Katoh et al. 2005, NAR 33(2):511-518")?;
    writeln!(out, "                     doi: 10.1093/nar/gki198")?;
    writeln!(out, "  --parttree:        Katoh & Toh 2007, Bioinformatics 23(3):372-374")?;
    writeln!(out, "                     doi: 10.1093/bioinformatics/btl592")?;
    writeln!(out)?;
    writeln!(out, "Full citation guidance and BibTeX entries:")?;
    writeln!(out, "  https://luksgrin.github.io/rust-MAFFT/citation/")?;
    writeln!(out)?;
    writeln!(out, "Machine-readable form (CITATION.cff):")?;
    writeln!(out, "  https://github.com/luksgrin/rust-MAFFT/blob/main/CITATION.cff")?;
    Ok(())
}

/// Apply C MAFFT shell-script defaults based on `argv[0]` basename.
///
/// C MAFFT ships symlinks (`linsi`, `ginsi`, `einsi`, `fftns`, `fftnsi`,
/// `nwns`, `nwnsi`, `qinsi`, `xinsi`) plus the `mafft-` prefixed forms.
/// Each symlink invocation sets a different combination of mode and
/// iteration defaults — see `scripts/mafft` (the C `if [ $progname = ... ]`
/// case-cascade) for the exact mapping. We mirror it here.
///
/// Reads the current executable's basename (`std::env::current_exe()`),
/// strips an optional `mafft-` prefix, and dispatches to
/// `apply_progname_dispatch`. The latter is a pure function so it can
/// be unit-tested without touching the process state.
fn apply_progname_defaults(args: &mut Args) {
    let progname = std::env::current_exe()
        .ok()
        .as_deref()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .map(|s| s.strip_prefix("mafft-").unwrap_or(s))
        .map(|s| s.to_string())
        .unwrap_or_default();
    apply_progname_dispatch(&progname, args);
}

/// Pure dispatch step for `apply_progname_defaults`. Given the (possibly
/// prefix-stripped) basename, override `args` fields that the user did
/// *not* explicitly pass. For `Option<T>` fields we check `is_none()`;
/// for boolean mode flags we check that *no* alternative mode is already
/// on (so `linsi --globalpair` correctly switches to global pairwise).
fn apply_progname_dispatch(progname: &str, args: &mut Args) {
    let no_pair_mode_set =
        !args.localpair && !args.globalpair && !args.genafpair
        && !args.qinsi && !args.xinsi && !args.scarnalike;
    let set_maxit_if_default = |m: &mut Option<usize>, v: usize| {
        if m.is_none() { *m = Some(v); }
    };

    match progname {
        "linsi" => {
            if no_pair_mode_set { args.localpair = true; }
            set_maxit_if_default(&mut args.maxiterate, 1000);
        }
        "ginsi" => {
            if no_pair_mode_set { args.globalpair = true; }
            set_maxit_if_default(&mut args.maxiterate, 1000);
        }
        "einsi" => {
            if no_pair_mode_set { args.genafpair = true; }
            set_maxit_if_default(&mut args.maxiterate, 1000);
        }
        "fftns" => {
            // FFT-NS-2 = default; nothing to set.
        }
        "fftnsi" => {
            // C: defaultiterate=2 (NOT 100). Verified
            // `fftnsi sample` byte-identical to `mafft --maxiterate 2 sample`.
            set_maxit_if_default(&mut args.maxiterate, 2);
        }
        "nwns" => {
            if !args.nofft { args.nofft = true; }
        }
        "nwnsi" => {
            if !args.nofft { args.nofft = true; }
            set_maxit_if_default(&mut args.maxiterate, 2);
        }
        "qinsi" => {
            if no_pair_mode_set { args.qinsi = true; }
            set_maxit_if_default(&mut args.maxiterate, 1000);
        }
        "xinsi" => {
            if no_pair_mode_set { args.xinsi = true; }
            set_maxit_if_default(&mut args.maxiterate, 1000);
        }
        _ => {} // Not a recognised shortcut (likely "mafft-rs" or unrelated)
    }
}

/// Error returned by [`run_from`] wherever [`run`] would call
/// `std::process::exit(N)`.
///
/// It carries the exact stderr text and exit code the CLI uses, so [`run`]
/// can reproduce the command-line behaviour byte-for-byte while a library
/// caller gets a value it can inspect instead of a dead process.
///
/// A *successful* early exit is reported as an error too: `--pdbidlist`
/// and `--pdbfilelist` print a message and exit 0 in C MAFFT (and here)
/// without producing an alignment, so [`MafftError::code`] is `0` while the
/// call still returns `Err`. `--cite` is the exception — its citation block
/// is written to the output sink and the call returns `Ok`.
#[derive(Debug)]
pub struct MafftError {
    code: i32,
    message: String,
    /// Set only for argv-parsing failures (including `--help` / `--version`),
    /// so [`run`] can hand the error straight back to clap and get
    /// byte-identical output, stream selection and exit code.
    clap: Option<Box<clap::Error>>,
}

impl MafftError {
    fn new(code: i32, message: impl Into<String>) -> Self {
        Self { code, message: message.into(), clap: None }
    }

    fn from_clap(err: clap::Error) -> Self {
        Self {
            code: err.exit_code(),
            message: err.render().to_string(),
            clap: Some(Box::new(err)),
        }
    }

    /// Exit code the CLI would terminate with for this failure.
    pub fn code(&self) -> i32 {
        self.code
    }

    /// Exact text the CLI would print, without the trailing newline
    /// `eprintln!` adds. (For argv-parsing failures this is clap's own
    /// rendered usage/help text, which does carry its own newlines.)
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Reproduce the CLI's reporting exactly and terminate the process.
    fn report_and_exit(self) -> ! {
        match self.clap {
            // `clap::Error::exit` is what `Parser::parse` itself calls, so
            // `--help` / `--version` / usage errors keep their stream,
            // colouring and exit code unchanged.
            Some(e) => e.exit(),
            None => {
                eprintln!("{}", self.message);
                std::process::exit(self.code)
            }
        }
    }
}

impl std::fmt::Display for MafftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MafftError {}

/// Run `f` inside `pool` when `--thread N` built one, otherwise directly
/// (which leaves rayon's default global pool in charge). See the
/// `--thread` handling in [`run_from`] for why the pool is local.
fn in_pool<R: Send>(pool: Option<&rayon::ThreadPool>, f: impl FnOnce() -> R + Send) -> R {
    match pool {
        Some(p) => p.install(f),
        None => f(),
    }
}

/// `--nuc` / `--amino`: force the sequence type, overriding the
/// ATGC-frequency auto-detection in `mafft_io::detect_seq_type`.
///
/// Mirrors C MAFFT `scripts/mafft:547-550`, where `--nuc` sets
/// `seqtype="-D"` and `--amino` sets `seqtype="-P"`; `$seqtype` is then
/// passed verbatim to every downstream binary (`tbfast`, `pairlocalalign`,
/// `filter`, `replaceu`, …). Forcing the type is all these flags do — the
/// only other place `$seqtype` is read is the `--dash` guard at
/// `scripts/mafft:2441`, and `--dash` is not supported here.
///
/// Because `$seqtype` fixes C's `dorp` *before* any sequence is read, the
/// forced type also drives the residue-case fold C applies at read time
/// (`io.c:1462-1467`). Re-apply it here so `--nuc` lowercases and
/// `--amino` uppercases, matching C. `casepreserve` skips that: on the
/// `--anysymbol` / `--preservecase` path C keeps the original characters
/// and restores them after alignment.
///
/// Inert unless one of the flags is present, so auto-detection and the
/// reader's own case fold are unchanged for every existing command line.
fn force_seq_type(mut set: SequenceSet, args: &Args, casepreserve: bool) -> SequenceSet {
    let forced = if args.nuc {
        Some(mafft_types::SeqType::Dna)
    } else if args.amino {
        Some(mafft_types::SeqType::Protein)
    } else {
        None
    };
    if let Some(seq_type) = forced {
        set.seq_type = seq_type;
        if !casepreserve {
            mafft_io::apply_case_convention(&mut set);
        }
    }
    set
}

/// MAFFT-rs CLI entry point.
///
/// Parses `std::env::args_os()`, runs the alignment per the flags, and
/// writes the result to stdout (or `--output`). Any error path calls
/// `std::process::exit(N)` directly — this function does not return on
/// failure.
///
/// This is a thin wrapper around [`run_from`]: every flag's meaning is
/// decided there, so the CLI and the library entry point cannot drift.
///
/// Reused by:
/// * `crates/mafft-bin/src/main.rs` (the `mafft-rs` binary)
/// * `crates/pymafft` (bundled into the wheel so `pip install pymafft`
///   puts `mafft-rs` on `$PATH`)
pub fn run() {
    let mut stdout = io::stdout();
    if let Err(e) = run_from(std::env::args_os(), &mut stdout) {
        e.report_and_exit();
    }
}

/// Argv-driven, non-exiting entry point — the library form of [`run`].
///
/// `argv` is parsed with exactly the same clap definition as the command
/// line (`argv[0]` is the program name, as usual), so argv stays the single
/// source of truth for flag semantics: `--auto` still picks the strategy
/// from sequence count and length, `--adjustdirection` still runs strand
/// detection before alignment, and so on. Every failure path that [`run`]
/// turns into a `std::process::exit(N)` is returned here as a
/// [`MafftError`] carrying the same message text and the same exit code.
///
/// The alignment is written to `out`. `--output FILE` still redirects it to
/// that file (matching the CLI exactly), in which case `out` receives
/// nothing. Progress and diagnostic messages still go to stderr, as on the
/// command line; pass `--quiet` to suppress them.
///
/// ```no_run
/// let mut aligned = Vec::new();
/// mafft_rs::run_from(
///     ["mafft-rs", "--auto", "--adjustdirection", "--thread", "1", "--nuc", "in.fasta"],
///     &mut aligned,
/// )?;
/// # Ok::<(), mafft_rs::MafftError>(())
/// ```
pub fn run_from<I, T>(argv: I, out: &mut dyn std::io::Write) -> Result<(), MafftError>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    run_from_with_progress(argv, out, &StderrProgress)
}

/// [`run_from`] with the progress messages redirected to `progress`.
///
/// The CLI reports what it is doing on stderr (`mafft-rs v0.1.2`,
/// `8 sequences (nuc), strategy: FFT-NS-2`, `Alignment: 398 columns`, …).
/// A caller running thousands of alignments from a worker pool wants those
/// somewhere other than the user's terminal; pass [`SilentProgress`] to drop
/// them, or any `Fn(&str)` to forward them into a logger.
///
/// `progress` is taken as `&(dyn Progress + Sync)` so a single sink can be
/// shared by concurrent runs. Messages arrive without a trailing newline.
///
/// Only progress is routed. Anything that aborts the run is returned as a
/// [`MafftError`], and non-fatal `Warning:` / `Could not …` diagnostics stay
/// on stderr, so a silent sink cannot hide a problem. `--scoreout`'s
/// `Unweighted sum-of-pairs score = …` line also stays on stderr: it is
/// output the user explicitly asked for, not progress.
///
/// ```no_run
/// use mafft_rs::SilentProgress;
///
/// let mut aligned = Vec::new();
/// mafft_rs::run_from_with_progress(
///     ["mafft-rs", "--auto", "--thread", "1", "in.fasta"],
///     &mut aligned,
///     &SilentProgress,
/// )?;
/// # Ok::<(), mafft_rs::MafftError>(())
/// ```
pub fn run_from_with_progress<I, T>(
    argv: I,
    out: &mut dyn std::io::Write,
    progress: &(dyn Progress + Sync),
) -> Result<(), MafftError>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let mut args = Args::try_parse_from(argv).map_err(MafftError::from_clap)?;
    apply_progname_defaults(&mut args);

    // --cite: print the citation block and exit cleanly. Comes before
    // every other flag handler so `--cite` is safe to combine with any
    // input or to invoke without one.
    if args.cite {
        return print_citation(&mut *out)
            .map_err(|e| MafftError::new(1, format!("Error writing output: {e}")));
    }

    // C `scripts/mafft:969-990` disables `--pdbidlist` and
    // `--pdbfilelist` with "temporarily unavailable, 2018/Dec." and
    // `exit`s before any structural alignment runs. Match that exact
    // behaviour and message verbatim — these flags have been
    // non-functional in upstream MAFFT since 2018.
    if args.pdbidlist.is_some() {
        return Err(MafftError::new(0, "--pdbidlist is temporarily unavailable, 2018/Dec.\n"));
    }
    if args.pdbfilelist.is_some() {
        return Err(MafftError::new(0, "--pdbfilelist is temporarily unavailable, 2018/Dec.\n"));
    }

    // C `scripts/mafft:1807-1810` rejects `--nodeout` combined with
    // `--maxiterate > 0` at the shell-script level (BEFORE any
    // alignment runs). Mirror the early exit and verbatim error.
    if args.nodeout && args.maxiterate.unwrap_or(0) > 0 {
        return Err(MafftError::new(1,
            "The --nodeout option supports only progressive method (--maxiterate 0) for now."));
    }

    // Configure thread pool. This builds a LOCAL rayon pool and installs
    // the alignment into it (see `in_pool` below) rather than calling
    // `build_global()`: a process-global pool can only be initialised
    // once, so a library caller invoking `run_from` repeatedly would have
    // been stuck with the first call's `--thread` value forever. The
    // remaining `.ok()` is not the "already initialised" swallow it used
    // to be — a local `build()` can only fail if the OS refuses to spawn
    // threads, and falling back to rayon's default pool there is exactly
    // what the previous code did.
    let pool = if args.thread > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.thread)
            .build()
            .ok()
    } else {
        None
    };

    // Read input. `--anysymbol`/`--preservecase` need every original
    // character preserved (case + non-standard residues) so the
    // post-alignment restore pass can put them back; the default
    // reader normalizes (`* → -`, drops non-alpha) which would lose
    // exactly the chars we need.
    let anysymbol_read = args.anysymbol || args.preservecase;
    let input = match &args.input {
        Some(path) => {
            let result = if anysymbol_read {
                read_fasta_casepreserve(path)
            } else {
                read_fasta(path)
            };
            result.map_err(|e|
                MafftError::new(1, format!("Error reading {}: {e}", path.display())))?
        }
        None => {
            let stdin = io::stdin();
            let reader = BufReader::new(stdin.lock());
            let result = if anysymbol_read {
                read_fasta_from_reader_casepreserve(reader)
            } else {
                read_fasta_from_reader(reader)
            };
            result.map_err(|e| MafftError::new(1, format!("Error reading stdin: {e}")))?
        }
    };

    // `--nuc` / `--amino` force the sequence type before anything that
    // branches on `is_nucleotide()` runs (mirrors C `scripts/mafft:547-550`,
    // where `$seqtype` is fixed at argument-parsing time). No-op unless one
    // of the flags was given.
    let input = force_seq_type(input, &args, anysymbol_read);

    // `--adjustdirection` / `--adjustdirectionaccurately`: detect DNA
    // strand orientation and reverse-complement sequences on the
    // wrong strand BEFORE any alignment runs. C runs this through
    // the `makedirectionlist`+`setdirection` helper pipeline in
    // `scripts/mafft:2323-2342` on the COMBINED existing+added
    // input (with `nadd` slicing so only the added sequences are
    // orientation-tested). Without `--add`, every sequence is
    // testable. The actual call is deferred until after the
    // add-file is read so we can pass the combined set; see
    // `apply_adjust_direction` near the `--add` handler below.

    // `--maxambiguous F`: validate range only here. The filter itself
    // runs against the `--add` / `--addfragments` file (matching C
    // MAFFT's `scripts/mafft:1132-1140` — it only filters the addfile,
    // never the primary input). Filter is applied below where the
    // addfile is read.
    if let Some(thresh) = args.maxambiguous {
        if !(0.0..=1.0).contains(&thresh) {
            return Err(MafftError::new(1,
                "The argument of --maxambiguous must be between 0.0 and 1.0"));
        }
    }

    let user_nseq = input.nseq();
    if user_nseq == 0 {
        return Err(MafftError::new(1, "Error: no sequences found in input"));
    }

    // `--memsave` gating: C MAFFT rejects `--memsave` for every
    // non-ktuples distance mode (`scripts/mafft:1866-1869`). That
    // covers `--localpair`, `--globalpair`, `--genafpair`,
    // `--lastpair`, `--multipair`, etc. — the seed `Impossible`
    // diagnostic. Mirror that here so users get the same error
    // surface.
    if args.memsave
        && (args.localpair || args.globalpair || args.genafpair
            || args.qinsi || args.xinsi || args.scarnalike)
    {
        return Err(MafftError::new(1, "Impossible"));
    }
    if args.memsave && !args.seed_files.is_empty() {
        // C rejects --seed + --memsave: MSalignmm doesn't accept
        // local-homology constraints (`tbfast.c:1117-1118`).
        return Err(MafftError::new(1, "Impossible"));
    }
    if args.memsave && args.seedtable.is_some() {
        // Same gating as --seed + --memsave: hat3.seed is plumbed into
        // tbfast through localhomtable, which `MSalignmm` doesn't read
        // (`tbfast.c:1117-1118`).
        return Err(MafftError::new(1, "Impossible"));
    }
    if !args.seed_files.is_empty() && args.seedtable.is_some() {
        // `scripts/mafft:1963-1965`: "Use either one of seedtable and seed.
        // Not both."
        return Err(MafftError::new(1, "Use either one of seedtable and seed.  Not both."));
    }
    let add_arg = args.add.as_ref().or(args.addfragments.as_ref());
    if args.seedtable.is_some() && add_arg.is_some() {
        // `scripts/mafft:1281-1284`: "Use either ONE of --seed,
        // --seedtable, --addprofile and --add."
        return Err(MafftError::new(1,
            "Impossible\nUse either ONE of --seed, --seedtable, --addprofile and --add."));
    }
    if args.seedtable.is_some() && (args.parttree || args.dpparttree) {
        // `scripts/mafft:1880-1883`: parttree + seed/seedtable is Impossible.
        return Err(MafftError::new(1, "Impossible"));
    }

    // `--seed FILE` (repeatable): read each pre-aligned seed file with
    // gaps preserved (the seed-pair LH extraction needs the gap
    // pattern), prepend the gap-stripped seed sequences to the user
    // input with `_seed_` name prefixes, and build the seed local-
    // homology table. Mirrors C MAFFT `scripts/mafft:2400-2436` +
    // `multi2hat3s.c`.
    //
    // The combined sequence list (seeds then user input) is what the
    // engine sees; the original user_nseq is preserved here so we can
    // restore the input subset and report mode names accurately.
    let mut input = input;
    let seed_groups_aligned: Vec<Vec<Vec<u8>>>;
    let mut seed_seq_count: usize = 0;
    if !args.seed_files.is_empty() {
        let mut groups: Vec<Vec<Vec<u8>>> = Vec::with_capacity(args.seed_files.len());
        for path in &args.seed_files {
            let seed_set = read_fasta_casepreserve(path).map_err(|e|
                MafftError::new(1, format!("Error reading {}: {e}", path.display())))?;
            groups.push(seed_set.sequences.iter().map(|s| s.data.clone()).collect());
            // Prepend renamed (gap-stripped) seed sequences to the input
            // ahead of the user data — matching C's `multi2hat3s` output
            // followed by `cat infile2 >> infile` (`scripts/mafft:2435`).
            for s in &seed_set.sequences {
                let ungapped: Vec<u8> = s.data.iter().copied()
                    .filter(|&c| c != b'-' && c != b'.').collect();
                let renamed = Sequence {
                    name: format!("_seed_{}", s.name),
                    data: ungapped,
                };
                input.sequences.insert(seed_seq_count, renamed);
                seed_seq_count += 1;
            }
        }
        seed_groups_aligned = groups;
    } else {
        seed_groups_aligned = Vec::new();
    }
    let total_nseq = input.nseq();
    if !args.quiet && seed_seq_count > 0 {
        progress.message(&format!("--seed: {} seed sequences across {} file(s)",
                  seed_seq_count, args.seed_files.len()));
    }

    // `--anysymbol` / `--preservecase`: snapshot the originals (case and
    // non-standard chars intact) and substitute X (protein) / n (DNA)
    // for any character outside the alignment alphabet before passing
    // the sequences to the DP. After alignment we restore the original
    // characters via name-keyed lookup. Mirrors C `replaceu` +
    // `restoreu` (`mafft-upstream/core/replaceu.c`, `restoreu.c`).
    let anysymbol = args.anysymbol || args.preservecase;
    let originals: Option<std::collections::HashMap<String, Vec<u8>>> = if anysymbol {
        let is_dna = input.seq_type.is_nucleotide();
        let map: std::collections::HashMap<String, Vec<u8>> = input.sequences.iter()
            .map(|s| (s.name.clone(), s.data.clone())).collect();
        for s in input.sequences.iter_mut() {
            replace_unusual(&mut s.data, is_dna);
        }
        Some(map)
    } else {
        None
    };

    // Check SCARNA-like mode (requires DASH client — network service)
    if args.scarnalike {
        return Err(MafftError::new(1,
            "SCARNA-like mode requires the DASH structural alignment client.\n\
             Install dash_client and ensure it is in your PATH or set MAFFT_BINARIES.\n\
             See: https://mafft.cbrc.jp/alignment/software/source.html"));
    }

    // `--auto`: pick mode + retree based on input size, mirroring
    // `scripts/mafft:1290-1343`. Overrides --localpair / --globalpair /
    // --genafpair / --parttree / --dpparttree / --maxiterate / --retree.
    let auto_choice = if args.auto {
        let nlen = input.sequences.iter().map(|s| s.data.len()).max().unwrap_or(0);
        Some(decide_auto(total_nseq, nlen))
    } else {
        None
    };

    // Determine alignment mode
    let mut mode = if let Some(ref a) = auto_choice {
        a.mode.clone()
    } else {
        determine_mode(&args)
    };

    // `--seed` / `--seedtable`: C MAFFT forces `iterate ≥ 2` when seed
    // alignments are present (`scripts/mafft:1911-1923`) — the seed-pair
    // `hat3.seed` constraints only fire during refinement. Lift `0`/`1`
    // iteration counts to 2, and promote progressive-only FFT-NS-2 to
    // FFT-NS-i with 2 iterations.
    if !args.seed_files.is_empty() || args.seedtable.is_some() {
        mode = match mode {
            AlignmentMode::FftNs2 => AlignmentMode::FftNsi { iterations: 2 },
            AlignmentMode::FftNsi { iterations } => {
                AlignmentMode::FftNsi { iterations: iterations.max(2) }
            }
            AlignmentMode::LInsi { iterations } => {
                AlignmentMode::LInsi { iterations: iterations.max(2) }
            }
            AlignmentMode::GInsi { iterations } => {
                AlignmentMode::GInsi { iterations: iterations.max(2) }
            }
            AlignmentMode::EInsi { iterations } => {
                AlignmentMode::EInsi { iterations: iterations.max(2) }
            }
            AlignmentMode::QInsi { iterations } => {
                AlignmentMode::QInsi { iterations: iterations.max(2) }
            }
            AlignmentMode::XInsi { iterations } => {
                AlignmentMode::XInsi { iterations: iterations.max(2) }
            }
        };
    }

    if !args.quiet {
        let mode_name = match &mode {
            AlignmentMode::FftNs2 => "FFT-NS-2",
            AlignmentMode::FftNsi { .. } => "FFT-NS-i",
            AlignmentMode::GInsi { .. } => "G-INS-i",
            AlignmentMode::LInsi { .. } => "L-INS-i",
            AlignmentMode::EInsi { .. } => "E-INS-i",
            AlignmentMode::QInsi { .. } => "Q-INS-i",
            AlignmentMode::XInsi { .. } => "X-INS-i",
        };
        let seq_type = if input.seq_type.is_nucleotide() { "nuc" } else { "aa" };
        progress.message(&format!("mafft-rs v{}", env!("CARGO_PKG_VERSION")));
        progress.message(&format!("{total_nseq} sequences ({seq_type}), strategy: {mode_name}"));
    }

    // Build engine. With `--auto`, the retree count comes from the size
    // heuristic; otherwise the CLI `--retree` value (default 2) wins.
    let retree = auto_choice.as_ref().map(|a| a.retree).unwrap_or(args.retree);
    let mut engine = MafftEngine::new(mode).with_retree(retree);
    if let Some(op) = args.op {
        engine = engine.with_gap_open(op);
    }
    if let Some(ep) = args.ep {
        engine = engine.with_gap_offset(ep);
    }
    // Fine-grained gap penalty overrides (--exp / --shiftpenalty /
    // pair-phase variants). Each is `Option<f64>`; `None` keeps the C
    // default applied inside the engine.
    engine.gap_extend = args.exp;
    engine.shift_penalty_factor = args.shiftpenalty;
    engine.pair_lop = args.lop;
    engine.pair_lep = args.lep;
    engine.pair_lexp = args.lexp;
    engine.pair_gop = args.gop;
    engine.pair_gep = args.gep;
    engine.pair_gexp = args.gexp;
    engine.minimum_weight = args.minimumweight;
    // Tree-linkage selection. Default (no override) keeps the engine's
    // C-matching `Mix { sueff: 0.1 }`. `--averagelinkage` → sueff 1.0
    // (= Average); `--minimumlinkage` → sueff 0.0 (= Minimum);
    // `--mixedlinkage F` → sueff F. `--youngestlinkage` is a separate
    // algorithm (not exposed here, see TODO).
    if args.averagelinkage {
        engine.cluster_method = mafft_tree::ClusterMethod::Mix { sueff: 1.0 };
    } else if args.minimumlinkage {
        engine.cluster_method = mafft_tree::ClusterMethod::Mix { sueff: 0.0 };
    } else if let Some(s) = args.mixedlinkage {
        if !(0.0..=1.0).contains(&s) {
            return Err(MafftError::new(1,
                "The argument of --mixedlinkage must be between 0.0 and 1.0"));
        }
        engine.cluster_method = mafft_tree::ClusterMethod::Mix { sueff: s };
    }
    // RNA-only gap-penalty knobs (`--rop`, `--rep`, `--LOP`,
    // `--LEXP`, `--GOP`, `--GEXP`). C forwards `rgop`/`rgep` to
    // `rnaopt` (mccaskill / contrafold / dafs / rnaalifold paths,
    // all external-dep blocked here) and `LGOP`/`LEXP`/`GEXP`/`GGOP`
    // to the LARA RNA path (also external-dep blocked). Accept the
    // values for CLI compatibility but warn that they're inert.
    let rna_knob_used = args.rop.is_some()
        || args.rep.is_some()
        || args.lop_lara.is_some()
        || args.lexp_lara.is_some()
        || args.gop_lara.is_some()
        || args.gexp_lara.is_some();
    if rna_knob_used && !args.quiet {
        progress.message(
            "Note: --rop/--rep/--LOP/--LEXP/--GOP/--GEXP only affect C MAFFT's RNA-structure \
             paths (X-INS-i contrafold, Q-INS-i mccaskill, LARA, DAFS), which require external \
             binaries not shipped with rust-MAFFT. Flag values accepted for compatibility but \
             have no runtime effect."
        );
    }

    // `--youngestlinkage` (C `treeext=youngestlinkage` →
    // `compacttree=4` → `compacttree_memsaveselectable(howcompact=2)`)
    // uses a per-step cluster-distance recompute via k-mer tables.
    // Rust now has a dedicated port (`youngestlinkage_tree`) — see
    // `crates/mafft-tree/src/memsavetree.rs`. Byte-identical to C on
    // first14 (small inputs where initial mindist[] survives); partial
    // closure on larger inputs (subtle tie-break / iteration-order
    // differences remain).
    engine.youngestlinkage = args.youngestlinkage;
    // Iteration-strategy stubs (gap #3 in TODO.md). All accepted at
    // the CLI for compatibility; the actual algorithm changes are
    // tracked separately. `--simplehillclimbing` is a TRUE no-op
    // (matches the default in both C MAFFT and us), so emit no note.
    // --bestfirst is now wired to the engine's BESTFIRST refinement
    // path (see refinement.rs::bestfirst_refine).
    engine.bestfirst = args.bestfirst;
    // `--thread N`: C's script forwards this to `dvtditr -C N`, and C selects
    // a different refinement implementation on `nthread > 0`
    // (`tditeration.c:1433`) whose convergence rule differs. Both no
    // `--thread` and `--thread 0` give C `-C 0`, so 0 means the
    // single-threaded rule here too.
    engine.nthread = args.thread;
    // --skipiterate is now partially wired: the "skip-refinement-entirely
    // when F is large" path matches C; the sub-alignment-aware partial
    // refinement is still TBD. Engine emits its own diagnostic when
    // appropriate; no extra note needed here.
    engine.skipiterate = args.skipiterate;
    // `--oneiteration`: C's `disttbfast -r` triggers a `dooneiteration`
    // "one-vs-others" refinement step after the progressive merge and
    // before regular refinement. Only effective in disttbfast-path
    // modes (FFT-NS-2, FFT-NS-i); see `refinement.rs::one_vs_others_refine`.
    engine.oneiteration = args.oneiteration;
    // `--nwildcard` (and the implicit case from `--allowshift` /
    // unalignlevel > 0 — `scripts/mafft:1437` sets `nmodel=" -: "`
    // when `unalignlevel != 0.0`, which the engine handles
    // internally via `self.unalign_level`). `--nzero` is the
    // documented default (N-row stays zero) — no wiring needed.
    engine.nwildcard = args.nwildcard;
    let _ = args.nzero; // documented-default no-op
    // `--pileup`: comb-tree guide + single progressive pass with
    // no refinement. C strategy "Pileup-NS-1"
    // (`scripts/mafft:2169` + `mltaln9.c::createchain`).
    engine.pileup = args.pileup;
    // --adjustdirectionaccurately is now implemented via the DP-mode
    // dispatch above in the adjust_direction call site.
    if let Some(bl) = args.bl {
        engine = engine.with_scoring_model(ScoringModel::Blosum(bl));
    }
    if let Some(pam) = args.jtt {
        engine = engine.with_scoring_model(ScoringModel::Jtt(pam));
    }
    if let Some(pam) = args.tm {
        engine = engine.with_scoring_model(ScoringModel::Tm(pam));
    }
    if let Some(kr) = args.kimura {
        engine = engine.with_kimura(kr);
    }
    if args.nofft {
        engine = engine.with_nofft(true);
    }
    // `--allowshift` sets unalignlevel=0.8 unless `--unalignlevel` is also
    // given (mirrors `scripts/mafft:1424-1428`).
    let unalign_level = match (args.unalignlevel, args.allowshift) {
        (Some(v), _) => v,
        (None, true) => 0.8,
        (None, false) => 0.0,
    };
    if args.allowshift {
        engine = engine.with_allowshift(true);
    }
    if unalign_level > 0.0 {
        engine = engine.with_unalign_level(unalign_level);
    }
    // `--auto` may override parttree/dpparttree based on the size heuristic.
    let parttree = auto_choice.as_ref().map(|a| a.parttree).unwrap_or(args.parttree);
    let dpparttree = auto_choice.as_ref().map(|a| a.dpparttree).unwrap_or(args.dpparttree);
    if parttree {
        engine = engine.with_parttree(true);
    }
    if dpparttree {
        engine = engine.with_dpparttree(true);
    }
    if let Some(gs) = args.groupsize {
        engine = engine.with_groupsize(gs);
    }
    if args.reorder {
        engine = engine.with_reorder(true);
    }
    if let Some(ref tree_path) = args.treein {
        if !tree_path.exists() {
            return Err(MafftError::new(1, format!("Cannot open {}", tree_path.display())));
        }
        engine.treein_path = Some(tree_path.clone());
    }
    if args.leavegappyregion {
        engine.legacy_gap_cost = true;
    }
    // `--memsave`: route the non-FFT progressive merge through
    // `mafft_align::msalignmm` (Hirschberg DP, linear-space). Verified
    // byte-identical to C MSalignmm via FFI cross-validation including
    // the asymmetric-length case closed 2026-05-16 (midw indexing fix
    // — `midw[j] += wm`, not `j+1`).
    if args.memsave {
        engine.memsave_dp = true;
    }
    if args.c_compat {
        engine.c_compat = true;
    }
    // `--memsavetree` overrides distance-based UPGMA tree construction with
    // C MAFFT's compacttree_memsaveselectable algorithm. `--auto` may also
    // request memsavetree in the 100k+ bracket — pass that through too.
    let memsavetree_active = args.memsavetree
        || auto_choice.as_ref().map(|a| a.memsavetree).unwrap_or(false);
    if memsavetree_active {
        engine.memsavetree = true;
    }

    // `--seed`: build the seed local-homology table. The engine will
    // (a) merge it into the L-INS-i/G-INS-i/E-INS-i pairwise table
    //     before `recompute_importance`, or
    // (b) use it directly when the chosen mode has no pairwise step
    //     (FFT-NS-i with `--seed`).
    if seed_seq_count > 0 {
        let scoring_model = if input.seq_type.is_nucleotide() {
            ScoringModel::Dna
        } else {
            engine.scoring_model
        };
        let scoring = mafft_scoring::build_context(scoring_model, input.seq_type);
        let mut seed_groups: Vec<mafft_align::SeedGroup> = Vec::new();
        let mut next_idx = 0usize;
        for group in &seed_groups_aligned {
            let n = group.len();
            seed_groups.push(mafft_align::SeedGroup {
                aligned: group.iter().map(|s| s.as_slice()).collect(),
                global_indices: (next_idx..next_idx + n).collect(),
            });
            next_idx += n;
        }
        let seed_table = mafft_align::build_seed_homology_table(
            &seed_groups,
            total_nseq,
            user_nseq,
            &scoring.consweight_matrix,
            &scoring.amino_map,
        );
        engine.seed_homology = Some(seed_table);
    } else if let Some(path) = &args.seedtable {
        // `--seedtable FILE`: parse the pre-computed hat3.seed file and
        // hand it to the engine like `--seed` would. No sequences are
        // prepended — the file's `i`/`j` reference indices into the user
        // input as supplied.
        let text = std::fs::read_to_string(path).map_err(|e|
            MafftError::new(1, format!("Error reading {}: {e}", path.display())))?;
        let seed_table = mafft_align::parse_hat3_seed(&text, total_nseq)
            .map_err(|e|
                MafftError::new(1, format!("Error parsing {}: {e}", path.display())))?;
        if !args.quiet {
            progress.message(&format!("--seedtable: loaded {}", path.display()));
        }
        engine.seed_homology = Some(seed_table);
    }

    // Handle --add / --addfragments. The alignment itself (and every
    // rayon-parallel step it drives) runs inside the local `--thread`
    // pool; see the `pool` construction above.
    let add_file = args.add.as_ref().or(args.addfragments.as_ref());
    // The closure below is the previous `let mut msa = ...` block, wrapped
    // so it can be `install`ed into the local `--thread` pool. Its body is
    // deliberately left at the original indentation (and `args` / `engine`
    // / `input` are rebound to shared/exclusive borrows) so this stays a
    // small, reviewable diff rather than a reindent of ~90 unchanged lines.
    let (args_ref, engine_ref, input_ref) = (&args, &engine, &mut input);
    let mut msa = in_pool(pool.as_ref(), move || -> Result<mafft_core::MultipleAlignment, MafftError> {
        let (args, engine, input) = (args_ref, engine_ref, input_ref);
        Ok(if let Some(add_path) = add_file {
        let new_input = read_fasta(add_path).map_err(|e|
            MafftError::new(1, format!("Error reading {}: {e}", add_path.display())))?;
        // `--nuc` / `--amino` force the addfile's type too — C passes the
        // same `$seqtype` to `filter` (`scripts/mafft:1140`) and to every
        // downstream binary.
        let new_input = force_seq_type(new_input, args, false);
        // `--maxambiguous F`: drop noisy sequences from the addfile
        // before they reach the alignment. C `scripts/mafft:1132-1140`
        // runs `filter -m F` only on `_addfile`, never on the primary
        // input — we mirror that gating exactly.
        let new_input = if let Some(thresh) = args.maxambiguous {
            let seq_type = new_input.seq_type;
            let (filtered, dropped) = apply_maxambiguous_filter(new_input, thresh);
            if dropped > 0 && !args.quiet {
                let kind = if matches!(seq_type, mafft_types::SeqType::Dna | mafft_types::SeqType::Rna) {
                    "nucleotides"
                } else {
                    "amino acids"
                };
                progress.message(&format!(
                    "\n\nRemoved {dropped} sequence(s) where the frequency of ambiguous {kind} > {thresh:.3}\n\n"
                ));
            }
            filtered
        } else {
            new_input
        };
        // `--adjustdirection` / `--adjustdirectionaccurately` on the
        // combined (existing + added) set, with `nadd` slicing so
        // only the added sequences get orientation-tested. C
        // `makedirectionlist.c:881-941` mirror.
        let new_input = if args.adjustdirection || args.adjustdirectionaccurately {
            use mafft_core::adjust_direction::{adjust_direction_mode_add, AdjustMode};
            let mode = if args.adjustdirectionaccurately {
                AdjustMode::Dp
            } else {
                AdjustMode::Kmer
            };
            // Combine existing + added, run adjust with nadd, split back.
            let nadd = new_input.nseq();
            let mut combined = input.clone();
            combined.sequences.extend(new_input.sequences.iter().cloned());
            let adjusted = adjust_direction_mode_add(&combined, mode, nadd);
            mafft_types::SequenceSet {
                sequences: adjusted.sequences.into_iter().skip(input.nseq()).collect(),
                seq_type: new_input.seq_type,
            }
        } else {
            new_input
        };

        if !args.quiet {
            progress.message(&format!("Adding {} sequences to existing alignment", new_input.nseq()));
        }
        // `--mapout` / `--compactmapout` both imply `--keeplength` in
        // C (`scripts/mafft:699-710` set `-Y` along with `-z`/`-Z`).
        // Use the with-map variant so we can write the `.map` file
        // below. The keeplength alignment itself is identical.
        if args.keeplength && (args.mapout || args.compactmapout) {
            let (msa, deletelist) = engine.add_to_alignment_with_map(input, &new_input);
            // Write .map file alongside the addfile, mirroring C
            // `scripts/mafft:2833-2837` (`cp _deletemap "$addfile.map"`).
            let map_path = {
                let mut p = add_path.clone();
                p.as_mut_os_string().push(".map");
                p
            };
            let map_content = if args.compactmapout {
                build_compact_map(&deletelist, &new_input, &msa, input.nseq())
            } else {
                build_full_map(&deletelist, &new_input, &msa, input.nseq())
            };
            match std::fs::write(&map_path, map_content) {
                Ok(_) if !args.quiet =>
                    progress.message(&format!("Wrote insertion map to {}", map_path.display())),
                Ok(_) => {}
                Err(e) => eprintln!("Warning: could not write {}: {e}", map_path.display()),
            }
            msa
        } else {
            engine.add_to_alignment(input, &new_input, args.keeplength)
        }
    } else {
        if args.maxambiguous.is_some() && !args.quiet {
            // Match C's behaviour: --maxambiguous without --add is a
            // no-op (the filter only runs on the addfile). Warn so
            // users don't expect main-input filtering.
            progress.message("Note: --maxambiguous has no effect without --add / --addfragments");
        }
        // `--adjustdirection` without `--add`: every sequence is
        // orientation-tested (n_anchor = 0 in the algorithm).
        if args.adjustdirection || args.adjustdirectionaccurately {
            use mafft_core::adjust_direction::{adjust_direction_mode, AdjustMode};
            let mode = if args.adjustdirectionaccurately {
                AdjustMode::Dp
            } else {
                AdjustMode::Kmer
            };
            *input = adjust_direction_mode(input, mode);
        }
        engine.align(input)
    })
    })?;

    // --distout: write the engine's distance matrix to `<INPUT>.hat2`,
    // mirroring C MAFFT's `cp $TMPFILE/hat2 $infilename.hat2`
    // (`scripts/mafft:2824-2826`). Requires a file-backed input — when
    // reading from stdin we have no path to derive the output name.
    // The matrix is whatever the engine actually used (k-mer for
    // FFT-NS-2 / FFT-NS-i, pairwise-score-derived for L/G/E-INS-i).
    if args.distout {
        match (&args.input, msa.distance_matrix.as_ref()) {
            (Some(input_path), Some(dm)) => {
                let hat2_path = {
                    let mut p = input_path.clone();
                    p.as_mut_os_string().push(".hat2");
                    p
                };
                let names: Vec<String> = input.sequences.iter()
                    .map(|s| s.name.clone()).collect();
                let mut distances: Vec<Vec<f64>> = Vec::with_capacity(dm.nseq);
                for i in 0..dm.nseq {
                    let row_len = dm.nseq - i - 1;
                    let mut row = Vec::with_capacity(row_len);
                    for j in (i + 1)..dm.nseq {
                        row.push(dm.get(i, j));
                    }
                    distances.push(row);
                }
                let hat2 = mafft_io::Hat2Matrix { names, distances };
                match std::fs::File::create(&hat2_path) {
                    Ok(mut f) => {
                        if let Err(e) = mafft_io::write_hat2(&hat2, &mut f) {
                            eprintln!("Error writing {}: {e}", hat2_path.display());
                        } else if !args.quiet {
                            progress.message(&format!("Wrote distance matrix to {}", hat2_path.display()));
                        }
                    }
                    Err(e) => eprintln!("Could not create {}: {e}", hat2_path.display()),
                }
            }
            (None, _) => {
                eprintln!("Warning: --distout requires a file input (stdin not supported)");
            }
            (_, None) => {
                eprintln!("Warning: --distout: engine did not produce a distance matrix \
                          (likely --parttree or --treein path)");
            }
        }
    }

    // --scoreout: print the unweighted sum-of-pairs score to stderr,
    // mirroring C MAFFT's `Unweighted sum-of-pairs score = N.NNNNN`
    // line from the `-S -B` tbfast args (`scripts/mafft:1466-1467`).
    // Computed over the final aligned MSA; gap columns contribute 0.
    //
    // NOT routed through the progress sink: this is output the user asked
    // for with `--scoreout`, not progress, so a silent sink must not be able
    // to swallow it. Same reasoning as the `Warning:` / `Could not …`
    // diagnostics below.
    if args.scoreout {
        let scoring_model = if input.seq_type.is_nucleotide() {
            mafft_types::ScoringModel::Dna
        } else {
            match args.bl {
                Some(n) => mafft_types::ScoringModel::Blosum(n),
                None => match args.jtt {
                    Some(p) => mafft_types::ScoringModel::Jtt(p),
                    None => match args.tm {
                        Some(p) => mafft_types::ScoringModel::Tm(p),
                        None => mafft_types::ScoringModel::Blosum(62),
                    },
                },
            }
        };
        let scoring = mafft_scoring::build_context(scoring_model, input.seq_type);
        let sp = compute_unweighted_sp_score(&msa.sequences, &scoring);
        eprintln!("Unweighted sum-of-pairs score = {sp:.5}");
    }

    // `--anysymbol`: restore each aligned row to its original characters
    // (case and non-standard residues intact). Mirrors C `restoreu`
    // (`mafft-upstream/core/restoreu.c::fillorichar`): for each aligned
    // sequence, walk every non-gap position and copy the next character
    // from the gap-stripped original.
    if let Some(orig_map) = originals {
        for i in 0..msa.sequences.len() {
            let Some(orig) = orig_map.get(&msa.names[i]) else { continue };
            let orig_no_gaps: Vec<u8> = orig.iter().copied()
                .filter(|c| *c != b'-' && *c != b'.').collect();
            let mut k = 0;
            for c in msa.sequences[i].iter_mut() {
                if *c != b'-' && *c != b'.' && k < orig_no_gaps.len() {
                    *c = orig_no_gaps[k];
                    k += 1;
                }
            }
        }
    }

    if !args.quiet {
        progress.message(&format!("Alignment: {} columns", msa.width()));
    }

    // --treeout: write the guide tree to `<INPUT>.tree` in Newick format,
    // mirroring C MAFFT's `cp $TMPFILE/infile.tree $infilename.tree`
    // (`scripts/mafft:2817-2819`). Requires a file-backed input — when
    // reading from stdin we have no path to derive the output name.
    //
    // PartTree uses a distinct tree format: numeric leaves only, no
    // branch lengths (`splittbfast.c:1275-1301,2532-2553`).
    if args.treeout || args.nodeout {
        if let Some(input_path) = &args.input {
            let tree_path = {
                let mut p = input_path.clone();
                p.as_mut_os_string().push(".tree");
                p
            };
            let seq_type = input.seq_type;
            let scoring_model = if seq_type.is_nucleotide() {
                mafft_types::ScoringModel::Dna
            } else {
                match args.bl {
                    Some(n) => mafft_types::ScoringModel::Blosum(n),
                    None => match args.jtt {
                        Some(p) => mafft_types::ScoringModel::Jtt(p),
                        None => match args.tm {
                            Some(p) => mafft_types::ScoringModel::Tm(p),
                            None => mafft_types::ScoringModel::Blosum(62),
                        },
                    },
                }
            };
            let scoring = mafft_scoring::build_context(scoring_model, seq_type);

            let newick_opt: Option<String> = if args.dpparttree {
                // `--dpparttree` uses cycle=1 (one `splittbfast` call) with
                // `-U` (`doalign=1`) so distances are computed via
                // `G__align11_noalign( n_disLN, -1200, -60, ... )` on the
                // RAW sequences (`splittbfast.c:1700`). `n_disLN` is the
                // base substitution matrix shifted by `offset - offsetLN`
                // (`constants.c:1431-1437`): for protein default with
                // `poffset = 0` this is `n_dis - 60` (offsetLN = 60).
                //
                // Selfscore uses the BASE matrix diagonal (no offsetLN shift)
                // per `splittbfast.c:3011-3017`.
                let raw_seqs: Vec<Vec<u8>> = input.sequences.iter()
                    .map(|s| s.data.clone()).collect();
                let base_matrix = &scoring.consweight_matrix;
                let amino_map = &scoring.amino_map;
                let nalpha = base_matrix.len();
                let offset_ln = 60.0f64;
                // n_disLN-equivalent: shift residue×residue cells by
                // `-offsetLN`. Cells involving non-residue indices stay 0.
                let nscored = scoring.nscoredalphabets;
                let mut dist_matrix: Vec<Vec<f64>> = (0..nalpha).map(|i| (0..nalpha).map(|j| {
                    if i < nscored && j < nscored {
                        base_matrix[i][j] - offset_ln
                    } else {
                        0.0
                    }
                }).collect()).collect();
                // Mirror C's `makedynamicmtx` '−' row/col skip
                // (`mltaln9.c:15197-15203`): the gap-index row/col is NOT
                // shifted so it stays zero, matching C's amino_dynamicmtx.
                let gap_idx = amino_map[b'-' as usize] as usize;
                if gap_idx < nalpha {
                    for j in 0..nalpha { dist_matrix[gap_idx][j] = 0.0; }
                    for i in 0..nalpha { dist_matrix[i][gap_idx] = 0.0; }
                }
                let selfscore_diag = |i: usize| -> i64 {
                    let mut s = 0.0f64;
                    for &c in &raw_seqs[i] {
                        let idx = amino_map[c as usize] as usize;
                        if idx < nalpha { s += base_matrix[idx][idx]; }
                    }
                    s as i64
                };
                let gap = mafft_align::GapModel::new(-1200.0, -60.0);
                let dist_matrix_ref = &dist_matrix;
                // `splittbfast.c:560` sets `outgap = 1` by default, so
                // `G__align11_noalign` penalizes BOTH terminal gaps —
                // equivalent to head_gap=true, tail_gap=true.
                let pair_dp = |i: usize, j: usize| -> f64 {
                    if i == j { return selfscore_diag(i) as f64; }
                    let aln = mafft_align::global_align(
                        &raw_seqs[i], &raw_seqs[j], dist_matrix_ref, amino_map, &gap, true, true,
                    );
                    aln.score
                };
                let seqs_equal = |i: usize, j: usize| -> bool {
                    raw_seqs[i] == raw_seqs[j]
                };
                let orilen = |i: usize| -> usize { raw_seqs[i].len() };
                mafft_tree::parttree_split::run_parttree_pipeline_with_scorer(
                    raw_seqs.len(), selfscore_diag, orilen, pair_dp, seqs_equal, 50,
                ).map(|r| mafft_tree::parttree_split::parttree_result_to_newick(&r))
            } else if args.parttree {
                // PartTree (cycle=2): C overwrites `infile.tree` with CALL 2's
                // (`fromaln=1`) tree, so we use the same fromaln scoring
                // on the FIRST-pass aligned MSA.
                let source_msa: &Vec<Vec<u8>> = msa.first_pass_sequences
                    .as_ref().unwrap_or(&msa.sequences);
                Some(mafft_tree::parttree_split::compute_parttree_newick_fromaln(
                    source_msa,
                    &scoring.consweight_matrix,
                    &scoring.amino_map,
                    scoring.gap.open as f64,
                ))
            } else {
                msa.guide_tree.as_ref().map(|t|
                    mafft_tree::topology_to_newick(t, &msa.names))
            };
            if let Some(mut newick) = newick_opt {
                // C `mltaln9.c:2818` appends `#by loadtree\n` to the
                // tree file when `--treein` was used. This is a comment
                // line (not part of the Newick string itself) but we
                // mirror it for byte-identical `--treeout` parity.
                if args.treein.is_some() {
                    newick.push_str("#by loadtree\n");
                }
                if args.nodeout {
                    if let (Some(topo), Some(dm)) =
                        (msa.guide_tree.as_ref(), msa.distance_matrix.as_ref())
                    {
                        newick.push_str(&build_nodeout_density_section(topo, dm));
                    } else if !args.quiet {
                        eprintln!(
                            "Warning: --nodeout requested but distance matrix or guide \
                             tree is unavailable for this mode (try --maxiterate 0); \
                             writing Newick only"
                        );
                    }
                }
                match std::fs::write(&tree_path, newick) {
                    Ok(_) if !args.quiet =>
                        progress.message(&format!("Wrote guide tree to {}", tree_path.display())),
                    Ok(_) => {}
                    Err(e) => eprintln!("Warning: could not write {}: {e}", tree_path.display()),
                }
            }
        } else {
            eprintln!("Warning: --treeout/--nodeout requires a file input (stdin not supported)");
        }
    }

    // Build output SequenceSet (with gaps)
    let output_seqs = SequenceSet {
        sequences: msa.sequences.iter().zip(msa.names.iter()).map(|(seq, name)| {
            Sequence {
                name: name.clone(),
                data: seq.clone(),
            }
        }).collect(),
        seq_type: input.seq_type,
    };

    // Write output. `--output FILE` still goes to that file, exactly as on
    // the command line; otherwise the alignment goes to `out` (stdout for
    // `run`, a caller-supplied sink for a library call).
    let write_result: Result<(), String> = match &args.output {
        Some(path) => {
            let file = std::fs::File::create(path).map_err(|e|
                MafftError::new(1, format!("Error creating {}: {e}", path.display())))?;
            let mut writer = io::BufWriter::new(file);
            write_output(&output_seqs, &mut writer, &args)
                .map_err(|e| e.to_string())
                .and_then(|()| writer.flush().map_err(|e| e.to_string()))
        }
        None => {
            let mut writer = io::BufWriter::new(&mut *out);
            write_output(&output_seqs, &mut writer, &args)
                .map_err(|e| e.to_string())
                .and_then(|()| writer.flush().map_err(|e| e.to_string()))
        }
    };

    if let Err(e) = write_result {
        return Err(MafftError::new(1, format!("Error writing output: {e}")));
    }
    Ok(())
}

/// `--anysymbol` preprocessor — substitute every character outside
/// the alignment alphabet with the appropriate "unknown" symbol, then
/// canonicalize case to match `replaceu.c::replace_unusual`:
/// - Protein: usual = "ARNDCQEGHILKMFPSTWYVarndcqeghilkmfpstwyv-.";
///   unknown = 'X', case = `toupper`.
/// - DNA:     usual = "ATGCUatgcuBDHKMNRSVWYXbdhkmnrsvwyx-";
///   unknown = 'n', case = `tolower`.
fn replace_unusual(seq: &mut [u8], is_dna: bool) {
    let usual_protein: &[u8] = b"ARNDCQEGHILKMFPSTWYVarndcqeghilkmfpstwyv-.";
    let usual_dna: &[u8] = b"ATGCUatgcuBDHKMNRSVWYXbdhkmnrsvwyx-";
    let (usual, unknown) = if is_dna {
        (usual_dna, b'n')
    } else {
        (usual_protein, b'X')
    };
    for c in seq.iter_mut() {
        if !usual.contains(c) {
            *c = unknown;
        } else if is_dna {
            *c = c.to_ascii_lowercase();
        } else {
            *c = c.to_ascii_uppercase();
        }
    }
}

/// Resolved alignment strategy for `--auto`.
#[derive(Debug, Clone)]
struct AutoChoice {
    mode: AlignmentMode,
    retree: usize,
    parttree: bool,
    dpparttree: bool,
    /// C MAFFT sets `treeext="memsavetree"` (and thus `compacttree=2`) for
    /// the 100k-200k brackets — see `scripts/mafft:1319-1328`. Surface it
    /// so the CLI can flip on `engine.memsavetree`.
    memsavetree: bool,
}

/// Mirror C `scripts/mafft:1290-1343` `--auto` heuristic. Picks mode and
/// retree count from `nseq` (sequence count) and `nlen` (longest input
/// sequence length, ungapped).
///
/// C uses `memsavetree` (large-N tree algorithm) at nseq ≥ 100k; we
/// don't have that yet, so for the 100k-200k bracket we fall through to
/// FFT-NS-2 / FFT-NS-1 (the alignment phase is the same; only the tree
/// construction differs). Output for those sizes may diverge from C.
fn decide_auto(nseq: usize, nlen: usize) -> AutoChoice {
    if nlen < 3000 && nseq < 100 {
        AutoChoice { mode: AlignmentMode::LInsi { iterations: 1000 }, retree: 1, parttree: false, dpparttree: false, memsavetree: false }
    } else if nlen < 1000 && nseq < 200 {
        AutoChoice { mode: AlignmentMode::LInsi { iterations: 2 }, retree: 1, parttree: false, dpparttree: false, memsavetree: false }
    } else if nlen < 10000 && nseq < 500 {
        AutoChoice { mode: AlignmentMode::FftNsi { iterations: 2 }, retree: 2, parttree: false, dpparttree: false, memsavetree: false }
    } else if nseq < 20000 {
        AutoChoice { mode: AlignmentMode::FftNs2, retree: 2, parttree: false, dpparttree: false, memsavetree: false }
    } else if nseq < 100000 {
        // C: cycle=2, memsavetree on. See `scripts/mafft:1315-1321`.
        AutoChoice { mode: AlignmentMode::FftNs2, retree: 2, parttree: false, dpparttree: false, memsavetree: true }
    } else if nseq < 200000 {
        // C: cycle=1, memsavetree on. See `scripts/mafft:1322-1328`.
        AutoChoice { mode: AlignmentMode::FftNs2, retree: 1, parttree: false, dpparttree: false, memsavetree: true }
    } else if nlen < 3000 {
        // PartTree + localalign distance (= --dpparttree).
        AutoChoice { mode: AlignmentMode::FftNs2, retree: 1, parttree: true, dpparttree: true, memsavetree: false }
    } else {
        // PartTree + ktuple distance.
        AutoChoice { mode: AlignmentMode::FftNs2, retree: 1, parttree: true, dpparttree: false, memsavetree: false }
    }
}

fn determine_mode(args: &Args) -> AlignmentMode {
    // C's `defaultiterate` (`scripts/mafft:86`) is 0 when invoked as
    // `mafft` — the bare flags `--localpair`/`--globalpair`/`--genafpair`
    // do NOT change it. Only the script-name aliases `linsi` / `ginsi` /
    // `einsi` (lines 142-156) set `defaultiterate=1000`. So `mafft
    // --localpair` runs progressive-only; `linsi` runs 1000-iter refinement.
    //
    // Explicit `--maxiterate N` always wins. `args.maxiterate` is
    // `Option<usize>`, so `None` means unset and `Some(n)` is the user's
    // choice (including `Some(0)`).
    let iters_for = |default: usize| args.maxiterate.unwrap_or(default);
    if args.qinsi {
        AlignmentMode::QInsi { iterations: iters_for(1000) }
    } else if args.xinsi {
        AlignmentMode::XInsi { iterations: iters_for(1000) }
    } else if args.localpair {
        AlignmentMode::LInsi { iterations: iters_for(0) }
    } else if args.globalpair {
        AlignmentMode::GInsi { iterations: iters_for(0) }
    } else if args.genafpair {
        AlignmentMode::EInsi { iterations: iters_for(0) }
    } else if let Some(n) = args.maxiterate.filter(|&n| n > 0) {
        AlignmentMode::FftNsi { iterations: n }
    } else {
        AlignmentMode::FftNs2
    }
}

/// Derive the human-readable strategy label (e.g. `FFT-NS-2`,
/// `L-INS-i`) for the CLUSTAL header. Mirrors what C's `scripts/mafft`
/// passes to `f2cl -c LABEL`.
///
/// The label depends on the combination of pair-mode flag, `--nofft`,
/// and `--maxiterate`. The mapping reproduces the `defaultprogname`
/// case-cascade in `scripts/mafft`:
///   FFT-NS-2 — default
///   NW-NS-2  — `--nofft`
///   FFT-NS-i — `--maxiterate N` (N > 0), no `--nofft`
///   NW-NS-i  — `--maxiterate N` (N > 0), `--nofft`
///   L-INS-1  — `--localpair --maxiterate 0`
///   L-INS-i  — `--localpair --maxiterate N` (N > 0)
///   G-INS-1  — `--globalpair --maxiterate 0`
///   G-INS-i  — `--globalpair --maxiterate N` (N > 0)
///   E-INS-1  — `--genafpair --maxiterate 0`
///   E-INS-i  — `--genafpair --maxiterate N` (N > 0)
///   Q-INS-i  — `--qinsi` (--maxiterate auto-defaults to 1000)
///   X-INS-i  — `--xinsi` (--maxiterate auto-defaults to 1000)
fn clustal_strategy_label(args: &Args) -> &'static str {
    let iter = args.maxiterate.unwrap_or(0);
    if args.localpair {
        if iter == 0 { "L-INS-1" } else { "L-INS-i" }
    } else if args.globalpair {
        if iter == 0 { "G-INS-1" } else { "G-INS-i" }
    } else if args.genafpair {
        if iter == 0 { "E-INS-1" } else { "E-INS-i" }
    } else if args.qinsi {
        "Q-INS-i"
    } else if args.xinsi {
        "X-INS-i"
    } else if iter > 0 {
        if args.nofft { "NW-NS-i" } else { "FFT-NS-i" }
    } else {
        if args.nofft { "NW-NS-2" } else { "FFT-NS-2" }
    }
}

/// Filter input sequences whose ambiguous-residue fraction (after gap
/// stripping) exceeds `threshold` (range 0.0–1.0). Direct port of
/// C MAFFT's `filter.c` algorithm. Also collapses consecutive runs of
/// the unknown character (`X` for protein, `n` for DNA) to a single
/// character, matching C's `shortenN`.
///
/// Returns the filtered SequenceSet plus the number of sequences
/// removed (for the stderr report `Removed N sequence(s) where the
/// frequency of ambiguous ... > F`).
fn apply_maxambiguous_filter(
    input: SequenceSet,
    threshold: f64,
) -> (SequenceSet, usize) {
    use mafft_types::SeqType;
    let (usual, unknown): (&[u8], u8) = match input.seq_type {
        SeqType::Dna | SeqType::Rna => (b"ATGCUatgcu-", b'n'),
        _ => (b"ARNDCQEGHILKMFPSTWYVarndcqeghilkmfpstwyv-", b'X'),
    };
    let mut kept: Vec<Sequence> = Vec::with_capacity(input.sequences.len());
    let mut dropped = 0usize;
    for seq in input.sequences {
        // gappick0: strip gaps before counting.
        let ungapped: Vec<u8> = seq.data.iter()
            .copied()
            .filter(|&b| b != b'-')
            .collect();
        if ungapped.is_empty() {
            // Empty after gap strip → unusual fraction = 0/0 = NaN; C
            // treats this as "all ambiguous" via division-by-zero
            // behavior, but in practice would-be-empty sequences are
            // dropped by upstream code. Drop conservatively.
            dropped += 1;
            continue;
        }
        let unusual_count = ungapped.iter()
            .filter(|&&b| !usual.contains(&b))
            .count();
        let frac = unusual_count as f64 / ungapped.len() as f64;
        if frac > threshold {
            dropped += 1;
            continue;
        }
        // shortenN: collapse runs of the unknown character.
        let mut collapsed: Vec<u8> = Vec::with_capacity(ungapped.len());
        let unknown_u = unknown.to_ascii_uppercase();
        let mut prev_was_unknown = false;
        for b in ungapped {
            if b.to_ascii_uppercase() == unknown_u {
                if !prev_was_unknown {
                    collapsed.push(unknown);
                    prev_was_unknown = true;
                }
            } else {
                collapsed.push(b);
                prev_was_unknown = false;
            }
        }
        kept.push(Sequence { name: seq.name, data: collapsed });
    }
    let filtered = SequenceSet { sequences: kept, seq_type: input.seq_type };
    (filtered, dropped)
}

/// Unweighted sum-of-pairs score, matching C MAFFT's `sumofpairsscore`
/// (`mltaln9.c:15411`): for each (i<j) pair, runs C's `naivepairscore11`
/// (gap-run consume loop, single penalty per gap run, common-gap
/// columns contribute 0) and sums the result divided by 600.
///
/// The divisor 600 unwinds C's scoring-matrix scaling (`consweight_matrix`
/// values are 600× the canonical BLOSUM/JTT/etc. integers), so the
/// reported number lines up with the canonical scoring-matrix units.
fn compute_unweighted_sp_score(
    seqs: &[Vec<u8>],
    scoring: &mafft_types::ScoringContext,
) -> f64 {
    let nseq = seqs.len();
    if nseq < 2 { return 0.0; }
    let mut total = 0.0f64;
    for i in 1..nseq {
        for j in 0..i {
            total += naivepairscore11(&seqs[i], &seqs[j], scoring) / 600.0;
        }
    }
    total
}

/// Port of C's `naivepairscore11` (`mltaln9.c:13851`). Walks both
/// sequences; common-gap columns are skipped; a gap in just one
/// sequence charges `penalty` (`scoring.gap.open`) once and consumes
/// the entire gap-run in THAT sequence (not the other — the asymmetry
/// matches C's `while (*p1 == '-')` loop). Matches mostly use
/// `consweight_matrix` (f64) for byte-identity with C's
/// `(double)amino_dis[c1][c2]` cast.
fn naivepairscore11(
    seq1: &[u8],
    seq2: &[u8],
    scoring: &mafft_types::ScoringContext,
) -> f64 {
    let map = &scoring.amino_map;
    let mtx = &scoring.consweight_matrix;
    let mtx_size = mtx.len();
    let penalty = scoring.gap.open as f64;
    let len = seq1.len().min(seq2.len());
    let mut score = 0.0f64;
    let mut k = 0;
    while k < len {
        let a = seq1[k];
        let b = seq2[k];
        if a == b'-' && b == b'-' { k += 1; continue; }
        if a == b'-' {
            score += penalty;
            k += 1;
            while k < len && seq1[k] == b'-' { k += 1; }
            continue;
        }
        if b == b'-' {
            score += penalty;
            k += 1;
            while k < len && seq2[k] == b'-' { k += 1; }
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

/// Build the `--nodeout` `Density:` + `Node info:` sections appended
/// after the Newick tree body, mirroring C
/// `mltaln9.c::fixed_musclesupg_double_realloc_nobk_halfmtx_treeout`
/// at lines 6495-6518 (the `treeout == 2` branch). Format:
///
/// ```text
/// (newick;\n)
/// \nDensity:
/// \nSequence {i+1}, {density:7.4f}
/// ... (per leaf)
/// \n\nNode info:
/// \nNode {k+1}, Height={height:f}
/// \n{densest_left+1}: {leaf list}
/// \n{densest_right+1}: {leaf list}
/// ... (per internal node)
/// ```
///
/// Density per leaf: `Σ_{j ≠ i, d(i,j) < 1.0} (2.0 - d(i,j))` —
/// port of `mltaln9.c::setdensity` (lines 1366-1395). Uses the
/// initial pairwise distance matrix (C calls `setdensity` once
/// before any UPGMA merge starts).
///
/// `densest` per subtree: the leaf in that subtree with the highest
/// density (first-encountered wins on tie — `mltaln9.c::getdensest`
/// strict `>` at line 1357).
fn build_nodeout_density_section(
    topology: &mafft_tree::Topology,
    dm: &mafft_tree::DistanceMatrix,
) -> String {
    use std::fmt::Write;

    let nseq = topology.nseq;

    // Per-leaf density (port of setdensity, mltaln9.c:1366-1395).
    let density: Vec<f64> = (0..nseq).map(|i| {
        let mut s = 0.0f64;
        for j in 0..nseq {
            if j == i { continue; }
            let d = dm.get(i, j);
            if d < 1.0 {
                s += 2.0 - d;
            }
        }
        s
    }).collect();

    let mut out = String::new();
    out.push_str("\nDensity:");
    for k in 0..nseq {
        // C uses `%7.4f` — width 7, 4 decimals, right-aligned.
        let _ = write!(out, "\nSequence {}, {:7.4}", k + 1, density[k]);
    }

    out.push_str("\n\nNode info:");

    let step_heights = mafft_tree::compute_distfromtip(topology);
    let getdensest = |mem: &[usize]| -> usize {
        // C `mltaln9.c:1350-1364`: first-encountered max wins (strict `>`).
        let mut best = mem[0];
        let mut best_v = density[best];
        for &m in &mem[1..] {
            if density[m] > best_v {
                best_v = density[m];
                best = m;
            }
        }
        best
    };
    for (k, step) in topology.steps.iter().enumerate() {
        let _ = write!(
            out,
            "\nNode {}, Height={:.6}\n",
            k + 1, step_heights[k]
        );
        let left = &step.left;
        let densest_left = getdensest(left);
        let _ = write!(out, "{}:", densest_left + 1);
        for &m in left { let _ = write!(out, " {}", m + 1); }
        out.push('\n');

        let right = &step.right;
        let densest_right = getdensest(right);
        let _ = write!(out, "{}:", densest_right + 1);
        for &m in right { let _ = write!(out, " {}", m + 1); }
        out.push('\n');
    }
    out
}

/// Build the C `--mapout` "full" map: one row per added-sequence
/// letter, with the original 1-indexed position and the
/// post-keeplength 1-indexed alignment column (`-` if dropped).
/// Port of `addfunctions.c::reconstructdeletemap` (lines 1985-2045).
///
/// `deletelist[i]` carries `(addbk_pos_0based, run_len)` runs of
/// dropped insertion residues for added-sequence `i`.
/// `new_input.sequences[i].data` is the gap-stripped `addbk[i]`.
fn build_full_map(
    deletelist: &[Vec<(usize, usize)>],
    new_input: &mafft_types::SequenceSet,
    msa: &mafft_core::MultipleAlignment,
    n_existing: usize,
) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for (i, dl) in deletelist.iter().enumerate() {
        let addbk: &[u8] = &new_input.sequences[i].data;
        let len = addbk.len();
        // Mark dropped positions of addbk[i] from the (pos, len) runs.
        let mut dropped = vec![false; len];
        for &(p, run) in dl {
            for k in 0..run {
                if p + k < len { dropped[p + k] = true; }
            }
        }
        let _ = writeln!(out, ">{}", new_input.sequences[i].name);
        let _ = writeln!(
            out,
            "# letter, position in the original sequence, position in the reference alignment"
        );
        // C `addfunctions.c:2024-2042`: `p` is the COLUMN index in the
        // post-keeplength aligned added sequence `realn[i]`. The loop
        // skips over '-' columns then emits `p+1` (1-indexed alignment
        // column) for each kept residue.
        let realn: &[u8] = &msa.sequences[n_existing + i];
        let mut p: usize = 0;
        for j in 0..len {
            // Advance past any gaps in realn before reading the next
            // residue position (mirrors C `while (realn[i][p] == '-') p++;`).
            while p < realn.len() && realn[p] == b'-' {
                p += 1;
            }
            let ch = addbk[j];
            if dropped[j] {
                let _ = writeln!(out, "{}, {}, -", ch as char, j + 1);
            } else {
                let _ = writeln!(out, "{}, {}, {}", ch as char, j + 1, p + 1);
                p += 1;
            }
        }
    }
    out
}

/// Build the C `--compactmapout` map: one block per added sequence
/// listing maximal runs of dropped insertions as
/// `<start_pos><residue> - <end_pos><residue> > <prev>v<next>`.
/// Sequences with NO insertions are skipped entirely. Port of
/// `addfunctions.c::reconstructdeletemap_compact` (lines 2047-2167).
fn build_compact_map(
    deletelist: &[Vec<(usize, usize)>],
    new_input: &mafft_types::SequenceSet,
    _msa: &mafft_core::MultipleAlignment,
    _n_existing: usize,
) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    out.push_str("# Insertion in added sequence > Position in reference\n");
    for (i, dl) in deletelist.iter().enumerate() {
        if dl.is_empty() { continue; }
        let addbk: &[u8] = &new_input.sequences[i].data;
        let len = addbk.len();
        let mut dropped = vec![false; len];
        for &(p, run) in dl {
            for k in 0..run {
                if p + k < len { dropped[p + k] = true; }
            }
        }
        let _ = writeln!(out, ">{}", new_input.sequences[i].name);
        // C `addfunctions.c:2074` calls this with `realn = seq+njob-nadd`
        // — the RAW (gap-free) added sequences, NOT the aligned ones.
        // The `while (realn[i][p] == '-')` skip is therefore a no-op,
        // and `p` is effectively a count of kept residues so far.
        // C compact's `vN` numbers are "between kept residue N and
        // N+1 of the post-keeplength addbk[i]" — NOT alignment columns.
        let mut p: usize = 0;
        let mut status: i32 = -1; // -1 = none, 0 = kept, 1 = in-run
        for j in 0..len {
            let ch = addbk[j];
            if dropped[j] {
                if status != 1 {
                    status = 1;
                    // C `addfunctions.c:2122` opens the run.
                    let _ = write!(out, "{}{} - ", j + 1, ch as char);
                }
                // dropped residues do NOT advance p.
            } else {
                if status == 1 {
                    // Close run: C `addfunctions.c:2137` emits
                    // `j addbk[j-1] > p v p+1` where `p` is the
                    // count of kept residues so far (before the
                    // `p++` that follows).
                    let prev_ch = addbk[j - 1];
                    let _ = writeln!(out, "{}{} > {}v{}", j, prev_ch as char, p, p + 1);
                }
                status = 0;
                p += 1;
            }
        }
        if status == 1 {
            // Run extends to end-of-sequence (C `addfunctions.c:2147-2153`).
            let j = len;
            let prev_ch = addbk[j - 1];
            let _ = writeln!(out, "{}{} > {}v{}", j, prev_ch as char, p, p + 1);
        }
    }
    out
}

fn write_output<W: Write>(
    seqs: &SequenceSet,
    writer: &mut W,
    args: &Args,
) -> Result<(), mafft_io::IoError> {
    match args.format.as_str() {
        "clustal" | "clw" => {
            // C MAFFT computes per-column conservation marks
            // (`setmark_clustal`, f2cl.c:22) and embeds the
            // alignment-mode label in the header line. Match both.
            let marks = mafft_io::compute_clustal_marks(seqs);
            let label = clustal_strategy_label(args);
            mafft_io::write_clustal_full(
                seqs, writer, None, args.namelength,
                Some(marks.as_str()), Some(label),
            )
        }
        "phylip" | "phy" => {
            mafft_io::write_phylip(seqs, writer, None, args.namelength)
        }
        _ => {
            mafft_io::write_fasta_to_writer_with_width(seqs, writer, args.linewidth)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mode_name(m: &AlignmentMode) -> &'static str {
        match m {
            AlignmentMode::FftNs2 => "FFT-NS-2",
            AlignmentMode::FftNsi { .. } => "FFT-NS-i",
            AlignmentMode::LInsi { .. } => "L-INS-i",
            AlignmentMode::GInsi { .. } => "G-INS-i",
            AlignmentMode::EInsi { .. } => "E-INS-i",
            AlignmentMode::QInsi { .. } => "Q-INS-i",
            AlignmentMode::XInsi { .. } => "X-INS-i",
        }
    }

    #[test]
    fn auto_small_picks_linsi_1000() {
        let a = decide_auto(50, 500);
        assert_eq!(mode_name(&a.mode), "L-INS-i");
        if let AlignmentMode::LInsi { iterations } = a.mode {
            assert_eq!(iterations, 1000);
        }
        assert_eq!(a.retree, 1);
        assert!(!a.parttree && !a.dpparttree);
    }

    #[test]
    fn auto_medium_picks_linsi_2() {
        // nlen<1000, nseq<200 but not <100 → L-INS-i iterate=2.
        let a = decide_auto(150, 800);
        assert_eq!(mode_name(&a.mode), "L-INS-i");
        if let AlignmentMode::LInsi { iterations } = a.mode {
            assert_eq!(iterations, 2);
        }
    }

    #[test]
    fn auto_large_picks_fft_nsi() {
        // nlen<10000, nseq<500 but not the LInsi brackets → FFT-NS-i iter=2.
        let a = decide_auto(300, 5000);
        assert_eq!(mode_name(&a.mode), "FFT-NS-i");
        if let AlignmentMode::FftNsi { iterations } = a.mode {
            assert_eq!(iterations, 2);
        }
        assert_eq!(a.retree, 2);
    }

    #[test]
    fn auto_xlarge_picks_fft_ns2() {
        // nseq<20000 → FFT-NS-2 retree=2.
        let a = decide_auto(5000, 5000);
        assert_eq!(mode_name(&a.mode), "FFT-NS-2");
        assert_eq!(a.retree, 2);
        assert!(!a.parttree);
    }

    #[test]
    fn auto_huge_picks_fft_ns1() {
        // nseq>=100000 (but <200000) → FFT-NS-2 retree=1.
        let a = decide_auto(150_000, 500);
        assert_eq!(mode_name(&a.mode), "FFT-NS-2");
        assert_eq!(a.retree, 1);
    }

    #[test]
    fn auto_giant_short_picks_dpparttree() {
        // nseq>=200000, nlen<3000 → --parttree --dpparttree retree=1.
        let a = decide_auto(250_000, 500);
        assert!(a.parttree);
        assert!(a.dpparttree);
        assert_eq!(a.retree, 1);
    }

    #[test]
    fn auto_giant_long_picks_parttree() {
        // nseq>=200000, nlen>=3000 → --parttree (ktuple) retree=1.
        let a = decide_auto(250_000, 5000);
        assert!(a.parttree);
        assert!(!a.dpparttree);
        assert_eq!(a.retree, 1);
    }

    #[test]
    fn replace_unusual_protein_canonicalizes() {
        // Protein usual set: 20 AA + lowercase + '-' '.'.
        // Lowercase gets uppercased; '*', '@', 'U' (selenocys), 'x',
        // 'B', 'J', 'Z' (extended set absent from `usual_protein`) →
        // 'X'.
        let mut seq = b"MKAUlsgVPxxBJL@&*fkdgna-.".to_vec();
        replace_unusual(&mut seq, false);
        assert_eq!(&seq, b"MKAXLSGVPXXXXLXXXFKDGNA-.");
    }

    #[test]
    fn replace_unusual_dna_canonicalizes() {
        // DNA usual set: ATGCU + lowercase + IUPAC ambig + 'X' + '-'.
        // Uppercase ATGC gets lowercased; '@' / '*' / digits-equivalents
        // → 'n'. IUPAC `R`, `Y`, `N` stay (lowercased).
        let mut seq = b"ATGCUNnxx@*RYBDHKMSVW-".to_vec();
        replace_unusual(&mut seq, true);
        // Every alphabetic char in `usual_dna` lowercased; unknowns → 'n'.
        assert_eq!(&seq, b"atgcunnxxnnrybdhkmsvw-");
    }

    #[test]
    fn replace_unusual_preserves_length() {
        // The function operates in place and must not change length.
        let original = b"AaBbCc@*xx.-".to_vec();
        let mut seq = original.clone();
        replace_unusual(&mut seq, false);
        assert_eq!(seq.len(), original.len());
        let mut seq2 = original.clone();
        replace_unusual(&mut seq2, true);
        assert_eq!(seq2.len(), original.len());
    }

    /// Parse a default `Args` (mafft-rs default behaviour) and apply
    /// progname dispatch for the given name. Returns the mutated args.
    fn dispatch(name: &str) -> Args {
        let mut a = Args::parse_from(["mafft-rs"]);
        apply_progname_dispatch(name, &mut a);
        a
    }

    #[test]
    fn progname_linsi_sets_localpair_and_iter_1000() {
        let a = dispatch("linsi");
        assert!(a.localpair);
        assert_eq!(a.maxiterate, Some(1000));
    }

    #[test]
    fn progname_ginsi_sets_globalpair_and_iter_1000() {
        let a = dispatch("ginsi");
        assert!(a.globalpair);
        assert_eq!(a.maxiterate, Some(1000));
    }

    #[test]
    fn progname_einsi_sets_genafpair_and_iter_1000() {
        let a = dispatch("einsi");
        assert!(a.genafpair);
        assert_eq!(a.maxiterate, Some(1000));
    }

    #[test]
    fn progname_fftnsi_sets_iter_2_not_100() {
        // C `scripts/mafft`: defaultiterate=2 for fftnsi. README previously
        // said --maxiterate 100; that was wrong.
        let a = dispatch("fftnsi");
        assert_eq!(a.maxiterate, Some(2));
        assert!(!a.localpair && !a.globalpair && !a.genafpair);
    }

    #[test]
    fn progname_nwns_sets_nofft_only() {
        let a = dispatch("nwns");
        assert!(a.nofft);
        assert_eq!(a.maxiterate, None);
    }

    #[test]
    fn progname_nwnsi_sets_nofft_and_iter_2() {
        let a = dispatch("nwnsi");
        assert!(a.nofft);
        assert_eq!(a.maxiterate, Some(2));
    }

    #[test]
    fn progname_qinsi_sets_qinsi_mode_and_iter_1000() {
        let a = dispatch("qinsi");
        assert!(a.qinsi);
        assert_eq!(a.maxiterate, Some(1000));
    }

    #[test]
    fn progname_xinsi_sets_xinsi_mode_and_iter_1000() {
        let a = dispatch("xinsi");
        assert!(a.xinsi);
        assert_eq!(a.maxiterate, Some(1000));
    }

    #[test]
    fn progname_unknown_leaves_args_untouched() {
        let a = dispatch("mafft-rs");
        assert!(!a.localpair && !a.globalpair && !a.genafpair);
        assert!(!a.nofft);
        assert_eq!(a.maxiterate, None);
    }

    #[test]
    fn user_maxiterate_overrides_progname_default() {
        let mut a = Args::parse_from(["mafft-rs", "--maxiterate", "5"]);
        apply_progname_dispatch("linsi", &mut a);
        assert!(a.localpair); // pair mode still applied
        assert_eq!(a.maxiterate, Some(5)); // but user iter wins
    }

    #[test]
    fn user_pair_mode_overrides_progname_default() {
        // `linsi --globalpair` should run G-INS-i, not L-INS-i.
        let mut a = Args::parse_from(["mafft-rs", "--globalpair"]);
        apply_progname_dispatch("linsi", &mut a);
        assert!(a.globalpair);
        assert!(!a.localpair);
        assert_eq!(a.maxiterate, Some(1000)); // iter still applied
    }

    /// All eight fine-grained gap-penalty flags parse correctly and
    /// land in their respective `Args` fields. Default is `None`.
    #[test]
    fn gap_penalty_flags_default_to_none() {
        let a = Args::parse_from(["mafft-rs"]);
        assert!(a.exp.is_none());
        assert!(a.shiftpenalty.is_none());
        assert!(a.lop.is_none());
        assert!(a.lep.is_none());
        assert!(a.lexp.is_none());
        assert!(a.gop.is_none());
        assert!(a.gep.is_none());
        assert!(a.gexp.is_none());
    }

    #[test]
    fn gap_penalty_flags_parse_signed_floats() {
        let a = Args::parse_from([
            "mafft-rs",
            "--exp", "0.1",
            "--shiftpenalty", "3.0",
            "--lop", "-3.0",
            "--lep", "0.2",
            "--lexp", "-0.2",
            "--gop", "-1.53",
            "--gep", "0.15",
            "--gexp", "-0.05",
        ]);
        assert_eq!(a.exp, Some(0.1));
        assert_eq!(a.shiftpenalty, Some(3.0));
        assert_eq!(a.lop, Some(-3.0));
        assert_eq!(a.lep, Some(0.2));
        assert_eq!(a.lexp, Some(-0.2));
        assert_eq!(a.gop, Some(-1.53));
        assert_eq!(a.gep, Some(0.15));
        assert_eq!(a.gexp, Some(-0.05));
    }

    // --- `--nuc` / `--amino`, `run_from` and the builder ---------------

    /// Small DNA input written to a unique temp file, so the tests below
    /// stay self-contained (no fixture files, no CWD assumptions).
    const DNA_FASTA: &str = "\
>a
ATGGCTAGCTTGGACCATTGCAGGTACCCATGGAACTTGGGCCATTAGGCATTGACCTAG
>b
ATGGCTAGCTTGGACCATTGCAGGTACCCTTGGAACTTGGCCATTAGGCATTGACCTAGG
>c
ATGGCAAGCTTAGACCTTTGCAGGTACGCATGGAACTAGGGCCTTTAGGCATTGACCTAG
>d
TTGGCTAGCTTGGACCATTGCAGCTACCCATGGAACTTGGGCCATTAGGCTTTGACGTAG
";

    fn write_tmp_fasta(tag: &str, body: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("mafft-rs-test-{}-{tag}.fa", std::process::id()));
        std::fs::write(&path, body).expect("write temp fasta");
        path
    }

    fn protein_set() -> SequenceSet {
        SequenceSet {
            sequences: vec![Sequence { name: "p".into(), data: b"MKALVWQHY".to_vec() }],
            seq_type: mafft_types::SeqType::Protein,
        }
    }

    fn dna_set() -> SequenceSet {
        SequenceSet {
            sequences: vec![Sequence { name: "d".into(), data: b"ACGTACGTAC".to_vec() }],
            seq_type: mafft_types::SeqType::Dna,
        }
    }

    #[test]
    fn nuc_forces_nucleotide_over_autodetection() {
        let a = Args::parse_from(["mafft-rs", "--nuc"]);
        let forced = force_seq_type(protein_set(), &a, false);
        assert_eq!(forced.seq_type, mafft_types::SeqType::Dna);
        // C's `dorp` is fixed by `$seqtype` before reading, so the forced
        // type also drives the read-time case fold (`io.c:1462-1467`).
        assert_eq!(forced.sequences[0].data, b"mkalvwqhy".to_vec());
    }

    #[test]
    fn amino_forces_protein_over_autodetection() {
        let a = Args::parse_from(["mafft-rs", "--amino"]);
        let forced = force_seq_type(dna_set(), &a, false);
        assert_eq!(forced.seq_type, mafft_types::SeqType::Protein);
        assert_eq!(forced.sequences[0].data, b"ACGTACGTAC".to_vec());
    }

    #[test]
    fn without_type_flags_autodetection_is_untouched() {
        // The flags must be completely inert when absent, so every
        // existing command line keeps its detected type.
        let a = Args::parse_from(["mafft-rs"]);
        let p = force_seq_type(protein_set(), &a, false);
        let d = force_seq_type(dna_set(), &a, false);
        assert_eq!(p.seq_type, mafft_types::SeqType::Protein);
        assert_eq!(d.seq_type, mafft_types::SeqType::Dna);
        // and the residues are untouched — the reader already folded them
        assert_eq!(p.sequences[0].data, b"MKALVWQHY".to_vec());
        assert_eq!(d.sequences[0].data, b"ACGTACGTAC".to_vec());
    }

    #[test]
    fn forced_type_does_not_recase_on_the_casepreserve_path() {
        // `--anysymbol`/`--preservecase` read case-preserving and restore
        // the originals after alignment, so the forced type must not fold
        // the residues here (C: `replaceu` + `restoreu`).
        let a = Args::parse_from(["mafft-rs", "--nuc"]);
        let kept = force_seq_type(protein_set(), &a, true);
        assert_eq!(kept.seq_type, mafft_types::SeqType::Dna);
        assert_eq!(kept.sequences[0].data, b"MKALVWQHY".to_vec());
    }

    #[test]
    fn nuc_and_amino_are_mutually_exclusive() {
        assert!(Args::try_parse_from(["mafft-rs", "--nuc", "--amino"]).is_err());
    }

    /// `run_from` returns the CLI's error instead of exiting the process.
    #[test]
    fn run_from_returns_err_for_nodeout_with_maxiterate() {
        let mut out = Vec::new();
        let err = run_from(
            ["mafft-rs", "--nodeout", "--maxiterate", "5", "unread.fa"],
            &mut out,
        ).expect_err("--nodeout with --maxiterate > 0 must fail");
        assert_eq!(err.code(), 1);
        assert_eq!(
            err.message(),
            "The --nodeout option supports only progressive method (--maxiterate 0) for now."
        );
        assert!(out.is_empty());
    }

    #[test]
    fn run_from_returns_err_for_unreadable_input() {
        let mut out = Vec::new();
        let err = run_from(["mafft-rs", "/nonexistent/mafft-rs-test-input.fa"], &mut out)
            .expect_err("a missing input file must fail");
        assert_eq!(err.code(), 1);
        assert!(
            err.message().starts_with("Error reading /nonexistent/mafft-rs-test-input.fa: "),
            "unexpected message: {}", err.message()
        );
        assert!(out.is_empty());
    }

    #[test]
    fn run_from_returns_err_for_impossible_flag_combination() {
        // `--memsave` is only rejected once the input has been read, so
        // point it at a real file.
        let path = write_tmp_fasta("memsave", DNA_FASTA);
        let mut out = Vec::new();
        let err = run_from(
            [
                std::ffi::OsString::from("mafft-rs"),
                std::ffi::OsString::from("--memsave"),
                std::ffi::OsString::from("--localpair"),
                path.clone().into_os_string(),
            ],
            &mut out,
        ).expect_err("--memsave with --localpair must fail");
        assert_eq!(err.code(), 1);
        assert_eq!(err.message(), "Impossible");
        assert!(out.is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn run_from_returns_err_for_unknown_flag_instead_of_exiting() {
        let mut out = Vec::new();
        let err = run_from(["mafft-rs", "--no-such-flag"], &mut out)
            .expect_err("clap parse failures must be returned, not `exit`ed");
        assert_eq!(err.code(), 2);
        assert!(err.message().contains("--no-such-flag"));
    }

    /// End-to-end: C MAFFT lowercases nucleotide output and uppercases
    /// protein output, and `--nuc` / `--amino` decide which applies
    /// (`io.c:1462-1467`, `scripts/mafft:547-550`).
    #[test]
    fn output_case_follows_the_c_mafft_convention() {
        let path = write_tmp_fasta("case", DNA_FASTA);
        let residues = |argv: &[&str]| -> String {
            let mut argv: Vec<std::ffi::OsString> =
                argv.iter().map(std::ffi::OsString::from).collect();
            argv.push(path.clone().into_os_string());
            let mut out = Vec::new();
            run_from(argv, &mut out).expect("alignment should succeed");
            String::from_utf8(out)
                .unwrap()
                .lines()
                .filter(|l| !l.starts_with('>'))
                .collect()
        };

        // Auto-detected nucleotide -> lowercase.
        let auto = residues(&["mafft-rs", "--quiet"]);
        assert!(!auto.is_empty());
        assert!(
            !auto.chars().any(|c| c.is_ascii_uppercase()),
            "nucleotide output must be lowercase: {auto}"
        );
        // Forced nucleotide -> still lowercase.
        assert_eq!(residues(&["mafft-rs", "--quiet", "--nuc"]), auto);
        // Forced protein -> uppercase.
        let amino = residues(&["mafft-rs", "--quiet", "--amino"]);
        assert!(
            !amino.chars().any(|c| c.is_ascii_lowercase()),
            "protein output must be uppercase: {amino}"
        );
        std::fs::remove_file(&path).ok();
    }

    /// `--preservecase` keeps the input's own case for nucleotides, so a
    /// mixed-case input survives the round trip verbatim.
    #[test]
    fn preservecase_keeps_input_case_for_nucleotides() {
        const MIXED: &str = "\
>a
ATGGCtagcTTGGACCATTGCAGGTACCCATGGAACTTGGGCCATTAGGCATTGACCTAG
>b
ATGGCTAGCTTGGACCATTGCAGGTACCCTTGGAACTTGGCCATTAGGCATTGACCTAGG
>c
atggcaagcttagacctttgcaggtacgcatggaactagggcctttaggcattgacctag
";
        let path = write_tmp_fasta("preservecase", MIXED);
        let out = Mafft::new().quiet().arg("--preservecase").input(&path).run_to_vec().unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("ATGGCtagcTTGG"), "original case must survive: {text}");
        assert!(text.contains("atggcaagctta"), "original case must survive: {text}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn run_from_writes_alignment_to_the_supplied_sink() {
        let path = write_tmp_fasta("sink", DNA_FASTA);
        let mut out = Vec::new();
        run_from(
            [
                std::ffi::OsString::from("mafft-rs"),
                std::ffi::OsString::from("--quiet"),
                path.clone().into_os_string(),
            ],
            &mut out,
        ).expect("alignment should succeed");
        assert!(out.starts_with(b">"));
        assert_eq!(out.iter().filter(|&&c| c == b'>').count(), 4);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn amino_changes_the_result_for_nucleotide_looking_input() {
        // CLUSTAL conservation marks are computed from the sequence type
        // (`mafft_io::compute_clustal_marks`), so they are a direct,
        // deterministic read-out of what `--amino` forced.
        let path = write_tmp_fasta("amino-e2e", DNA_FASTA);
        let auto = Mafft::new().quiet().format("clustal").input(&path).run_to_vec().unwrap();
        let forced =
            Mafft::new().quiet().format("clustal").amino().input(&path).run_to_vec().unwrap();
        assert!(!auto.is_empty());
        assert_ne!(
            auto, forced,
            "--amino must force the protein scoring path for DNA-looking input"
        );
        std::fs::remove_file(&path).ok();
    }

    /// The builder is only an argv constructor, so it must agree with the
    /// equivalent command line byte-for-byte.
    #[test]
    fn builder_matches_equivalent_argv() {
        let path = write_tmp_fasta("builder-e2e", DNA_FASTA);
        let mut via_argv = Vec::new();
        run_from(
            [
                std::ffi::OsString::from("mafft-rs"),
                std::ffi::OsString::from("--auto"),
                std::ffi::OsString::from("--adjustdirection"),
                std::ffi::OsString::from("--thread"),
                std::ffi::OsString::from("1"),
                std::ffi::OsString::from("--nuc"),
                std::ffi::OsString::from("--quiet"),
                path.clone().into_os_string(),
            ],
            &mut via_argv,
        ).expect("argv run should succeed");
        let via_builder = Mafft::new()
            .auto()
            .adjust_direction()
            .thread(1)
            .nuc()
            .quiet()
            .input(&path)
            .run_to_vec()
            .expect("builder run should succeed");
        assert!(!via_argv.is_empty());
        assert_eq!(via_argv, via_builder);
        std::fs::remove_file(&path).ok();
    }

    // --- progress sink --------------------------------------------------

    #[test]
    fn progress_sink_receives_the_progress_lines() {
        let path = write_tmp_fasta("progress", DNA_FASTA);
        let seen = std::sync::Mutex::new(Vec::new());
        let sink = |m: &str| seen.lock().unwrap().push(m.to_string());
        let mut out = Vec::new();
        run_from_with_progress(
            [
                std::ffi::OsString::from("mafft-rs"),
                path.clone().into_os_string(),
            ],
            &mut out,
            &sink,
        ).expect("alignment should succeed");
        let msgs = seen.lock().unwrap().clone();
        assert!(
            msgs.iter().any(|m| m.starts_with("mafft-rs v")),
            "expected the version banner: {msgs:?}"
        );
        assert!(
            msgs.iter().any(|m| m.contains("strategy: FFT-NS-2")),
            "expected the strategy line: {msgs:?}"
        );
        assert!(
            msgs.iter().any(|m| m.starts_with("Alignment: ")),
            "expected the column count: {msgs:?}"
        );
        // Messages arrive without a trailing newline; the sink adds it.
        assert!(msgs.iter().all(|m| !m.ends_with('\n')), "{msgs:?}");
        assert!(out.starts_with(b">"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn progress_sink_does_not_change_the_alignment() {
        let path = write_tmp_fasta("progress-eq", DNA_FASTA);
        let argv = || {
            [
                std::ffi::OsString::from("mafft-rs"),
                path.clone().into_os_string(),
            ]
        };
        let mut loud = Vec::new();
        run_from_with_progress(argv(), &mut loud, &StderrProgress).unwrap();
        let mut silent = Vec::new();
        run_from_with_progress(argv(), &mut silent, &SilentProgress).unwrap();
        let mut default = Vec::new();
        run_from(argv(), &mut default).unwrap();
        assert_eq!(loud, silent);
        assert_eq!(loud, default);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn quiet_suppresses_progress_even_with_a_sink() {
        // `--quiet` gates the messages at source, so a sink sees nothing.
        let path = write_tmp_fasta("progress-quiet", DNA_FASTA);
        let seen = std::sync::Mutex::new(Vec::new());
        let sink = |m: &str| seen.lock().unwrap().push(m.to_string());
        let mut out = Vec::new();
        run_from_with_progress(
            [
                std::ffi::OsString::from("mafft-rs"),
                std::ffi::OsString::from("--quiet"),
                path.clone().into_os_string(),
            ],
            &mut out,
            &sink,
        ).unwrap();
        assert!(seen.lock().unwrap().is_empty(), "{:?}", seen.lock().unwrap());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn builder_progress_sink_is_used() {
        let path = write_tmp_fasta("progress-builder", DNA_FASTA);
        let silent = Mafft::new().progress(SilentProgress).input(&path).run_to_vec().unwrap();
        let default = Mafft::new().input(&path).run_to_vec().unwrap();
        assert_eq!(silent, default);
        assert!(!silent.is_empty());
        std::fs::remove_file(&path).ok();
    }

    /// Requirement: only progress is routed. Diagnostics and explicitly
    /// requested output must stay on stderr so a silent sink cannot hide
    /// them.
    #[test]
    fn sink_receives_progress_but_not_scoreout_or_warnings() {
        let path = write_tmp_fasta("progress-scope", DNA_FASTA);
        let seen = std::sync::Mutex::new(Vec::new());
        let sink = |m: &str| seen.lock().unwrap().push(m.to_string());
        let mut out = Vec::new();
        run_from_with_progress(
            [
                std::ffi::OsString::from("mafft-rs"),
                // --scoreout prints a score line; --distout on a
                // file-backed input succeeds, but --nodeout with
                // refinement off warns when no matrix is available.
                std::ffi::OsString::from("--scoreout"),
                path.clone().into_os_string(),
            ],
            &mut out,
            &sink,
        ).expect("alignment should succeed");
        let msgs = seen.lock().unwrap().clone();
        assert!(msgs.iter().any(|m| m.starts_with("Alignment: ")), "{msgs:?}");
        assert!(
            !msgs.iter().any(|m| m.contains("sum-of-pairs score")),
            "--scoreout output must not go through the progress sink: {msgs:?}"
        );
        assert!(
            !msgs.iter().any(|m| m.starts_with("Warning:")),
            "warnings must not go through the progress sink: {msgs:?}"
        );
        std::fs::remove_file(&path).ok();
    }

    /// The intended caller drives one sink from a pool of worker threads.
    #[test]
    fn one_sink_serves_concurrent_runs() {
        let path = write_tmp_fasta("progress-threads", DNA_FASTA);
        let seen = std::sync::Mutex::new(Vec::new());
        let sink = |m: &str| seen.lock().unwrap().push(m.to_string());
        let dynref: &(dyn Progress + Sync) = &sink;
        let outs: Vec<Vec<u8>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4).map(|_| {
                let path = path.clone();
                scope.spawn(move || {
                    let mut out = Vec::new();
                    run_from_with_progress(
                        [
                            std::ffi::OsString::from("mafft-rs"),
                            path.into_os_string(),
                        ],
                        &mut out,
                        dynref,
                    ).expect("alignment should succeed");
                    out
                })
            }).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        // Every thread produced the same alignment...
        assert!(outs.iter().all(|o| *o == outs[0]));
        assert!(!outs[0].is_empty());
        // ...and every thread's progress reached the one shared sink.
        let msgs = seen.lock().unwrap();
        assert_eq!(msgs.iter().filter(|m| m.starts_with("Alignment: ")).count(), 4);
        std::fs::remove_file(&path).ok();
    }

    /// A library caller may run many alignments in one process; the
    /// `--thread` pool is local, so repeated calls with different thread
    /// counts all work and give the same answer.
    #[test]
    fn repeated_run_from_calls_honour_thread_counts() {
        let path = write_tmp_fasta("threads", DNA_FASTA);
        let mut prev: Option<Vec<u8>> = None;
        for threads in ["1", "2", "1"] {
            let mut out = Vec::new();
            run_from(
                [
                    std::ffi::OsString::from("mafft-rs"),
                    std::ffi::OsString::from("--quiet"),
                    std::ffi::OsString::from("--thread"),
                    std::ffi::OsString::from(threads),
                    path.clone().into_os_string(),
                ],
                &mut out,
            ).expect("alignment should succeed");
            if let Some(p) = &prev {
                assert_eq!(p, &out, "thread count must not change the alignment");
            }
            prev = Some(out);
        }
        std::fs::remove_file(&path).ok();
    }
}

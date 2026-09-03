# rust-MAFFT

A pure-Rust port of [**MAFFT**](https://mafft.cbrc.jp/alignment/software/), the
multiple sequence alignment software developed by Kazutaka Katoh and
colleagues at CBRC. The scientific contribution — the FFT-anchored
alignment algorithm, the iterative-refinement variants, the scoring
matrices — is theirs. This project re-engineers that work in Rust while
preserving **byte-identical output** to the C reference across the
BAliBASE 3 benchmark (1930/1930).

> **If you use this software in published work, please cite the
> original MAFFT paper (Katoh & Standley 2013) — see
> [Acknowledgments and Citation](#acknowledgments-and-citation) below
> for BibTeX entries.**

This project provides:
- **`mafft-rs`** — a single CLI binary replacing the original 5+ C binaries and shell script
- **`mafft`** / **`mafft-core`** — Rust library crates for programmatic use
- **`pymafft`** — Python bindings via PyO3

## Status

**Production-ready.** `mafft-rs` produces **byte-identical output to C MAFFT 7.526** across:

- All progressive modes: FFT-NS-2, NW-NS-2, FFT-NS-i, L-INS-1/i, G-INS-1/i, E-INS-1/i
- All scoring matrices: BLOSUM 30/45/50/62/80, JTT 100/200, TM 100/200 (FFT + NW + `--retree 1`)
- Tree variants: `--parttree`, `--dpparttree`, `--memsavetree`, `--youngestlinkage`, `--averagelinkage`, `--minimumlinkage`, `--mixedlinkage`, `--pileup`
- Iteration strategies: `--bestfirst`, `--simplehillclimbing`, `--skipiterate F` (both large- and small-F branches), `--oneiteration`
- Add-sequence modes: `--add`, `--addfragments`, `--keeplength` (including adversarial inputs with random insertions)
- Auxiliary outputs: `--distout`, `--scoreout`, `--nodeout`, `--mapout` / `--compactmapout`, `--treeout`
- DNA strand auto-detection: `--adjustdirection` (k-mer) and `--adjustdirectionaccurately` (DP)
- Refinement controls: `--allowshift`, `--unalignlevel`, `--maxambiguous`, `--minimumweight`, `--nwildcard`, `--leavegappyregion` / `--legacygappenalty`
- Memory strategy: `--memsave` / `--nomemsave` (Hirschberg DP), `--memsavetree`
- Seed alignment: `--seed FILE`, `--seedtable FILE` (single + multi-file)
- Fine-grained gap penalties: `--op`, `--ep`, `--exp`, `--shiftpenalty`, `--lop`, `--lep`, `--lexp`, `--gop`, `--gep`, `--gexp`
- Symlink dispatch via `argv[0]`: `linsi`, `ginsi`, `einsi`, `fftns`, `fftnsi`, `nwns`, `nwnsi`, `qinsi`, `xinsi`

Every progressive merge step matches in score and width; every refinement iteration converges to C's exact alignment.

### Parity matrix (36-seq protein sample, `mafft-upstream/test/sample`)

| Mode / flags                                | C width | Rust width | diff | Status |
|---------------------------------------------|---------|------------|------|--------|
| FFT-NS-2 (default)                          | 717     | 717        | 0    | ✓ byte-exact |
| NW-NS-2 (`--nofft`)                         | 717     | 717        | 0    | ✓ byte-exact |
| FFT-NS-i (`--maxiterate 100`)               | 721     | 721        | 0    | ✓ byte-exact |
| L-INS-1 (`--localpair --maxiterate 0`)      | 719     | 719        | 0    | ✓ byte-exact |
| L-INS-i (`linsi`)                           | 735     | 735        | 0    | ✓ byte-exact |
| G-INS-1 (`--globalpair --maxiterate 0`)     | 746     | 746        | 0    | ✓ byte-exact |
| G-INS-i (`ginsi`)                           | 737     | 737        | 0    | ✓ byte-exact |
| E-INS-1 (`--genafpair --maxiterate 0`)      | 729     | 729        | 0    | ✓ byte-exact |
| E-INS-i (`einsi`)                           | 729     | 729        | 0    | ✓ byte-exact |
| BLOSUM 30 / 45 / 50 / 62 / 80 (FFT + NW)    | match   | match      | 0    | ✓ byte-exact |
| JTT 100 / 200, TM 100 / 200 (FFT + NW)      | match   | match      | 0    | ✓ byte-exact |
| `--parttree`, `--dpparttree`, `--parttree --nofft`, `--parttree --reorder` | 752 | 752 | 0 | ✓ byte-exact |
| `--add` / `--add --nofft` / `--add --keeplength` (30+6) | match | match | 0 | ✓ byte-exact |
| `--add` adversarial (16-existing + 3-added with random insertions) | match | match | 0 | ✓ byte-exact |
| `--pileup`, `--youngestlinkage`, `--averagelinkage`, `--minimumlinkage` | match | match | 0 | ✓ byte-exact |
| `--bestfirst` (FFT-NS-i / L-INS-i / G-INS-i)  | match | match      | 0    | ✓ byte-exact |
| `--skipiterate F` (F ∈ {0.05…1.0})         | match   | match      | 0    | ✓ byte-exact |
| `--oneiteration` (FFT-NS-2 + FFT-NS-i)      | match   | match      | 0    | ✓ byte-exact |
| `--allowshift --globalpair --maxiterate 0`  | 1029    | 1029       | 0    | ✓ byte-exact |
| `--allowshift --globalpair --maxiterate 1000` | 1060  | 1060       | 0    | ✓ byte-exact (all 386 BBaliBase 3) |
| `--adjustdirection`, `--adjustdirectionaccurately` (mixed forward/RC DNA) | match | match | 0 | ✓ byte-exact |
| `--memsavetree`, `--memsavetree --treeout`  | match   | match      | 0    | ✓ byte-exact |
| `--auto` (small/medium/large brackets)      | match   | match      | 0    | ✓ byte-exact |
| `--seed FILE`, `--seedtable FILE`           | match   | match      | 0    | ✓ byte-exact |
| `--treein`, `--treeout`, `--treein --treeout` | match | match      | 0    | ✓ byte-exact |
| RNA NW (`--nofft samplerna`)                | 360     | 360        | 0    | ✓ byte-exact |
| Q-INS-i (`--qinsi samplerna`)               | 360     | 360        | 0‡   | ✓ byte-exact (needs `mxscarnamod`) |
| `--nuc`, `--amino` (forced sequence type)   | match   | match      | 0    | ✓ byte-exact |

‡ Q-INS-i requires the external `mxscarnamod` binary.

Residue case matches C MAFFT: nucleotide output is lowercased and protein
output uppercased at read time (C `io.c:1462-1467`, `io.c:1755`), driven by
the detected type or by `--nuc` / `--amino`; `--anysymbol` / `--preservecase`
keep the input's own case.

### BBaliBase 3 parity sweep

Sweep across BBaliBase 3 (RV11–RV50, 386 protein test sets) against MAFFT 7.526:

| Mode | Flags | Match | Total % |
|------|-------|-------|---------|
| FFT-NS-i | `--maxiterate 1000` | **386/386** | **100.0 %** |
| L-INS-i  | `--localpair --maxiterate 1000` | **386/386** | **100.0 %** |
| G-INS-i  | `--globalpair --maxiterate 1000` | **386/386** | **100.0 %** |
| E-INS-i  | `--genafpair --maxiterate 1000` | **386/386** | **100.0 %** |
| G-INS-i + allowshift | `--globalpair --maxiterate 1000 --allowshift` | **386/386** | **100.0 %** |

Total: **1930/1930 alignments byte-identical** to C MAFFT 7.526 across the BBaliBase 3 protein corpus.

The original MAFFT C code is included as a git submodule for testing and cross-validation.

## Building

### Prerequisites

- Rust 1.80+ (edition 2024)
- Git (for submodule checkout)
- GCC/Clang (only needed to build the C reference for FFI cross-validation tests)

### Build

```bash
git clone --recurse-submodules https://github.com/luksgrin/rust-MAFFT.git
cd rust-MAFFT
cargo build --release
```

The binary is at `target/release/mafft-rs`. To match C MAFFT's symlink-based shortcuts (`linsi` / `ginsi` / `einsi` / `fftns` / `fftnsi` / `nwns` / `nwnsi` / `qinsi` / `xinsi`), drop the binary on your `PATH` and create symlinks:

```bash
ln -s mafft-rs linsi
ln -s mafft-rs ginsi
ln -s mafft-rs einsi
# ... etc.
```

`mafft-rs` reads its `argv[0]` basename (stripping an optional `mafft-` prefix) and applies the same defaults as C's shell script. Explicit flags still override (e.g. `linsi --maxiterate 5 ...` honours the override).

### Run tests

```bash
# All integration tests across the workspace (419 tests)
cargo test --workspace --exclude pymafft --release

# Just unit tests (faster; no FFI / no C build)
cargo test --workspace --exclude pymafft --lib

# Build C reference for FFI cross-validation (optional; cargo test --tests
# triggers this automatically via mafft-c-bindings' build.rs)
make -C mafft-upstream/core
```

### Python bindings

```bash
cd crates/pymafft
uv venv .venv
uv pip install maturin pytest
maturin develop --release
uv run pytest tests/ -v   # 72 tests (32 API + 23 parity/shape + 17 Biopython interop)
```

## Usage

### CLI

```bash
# Basic usage (FFT-NS-2, fast default)
mafft-rs sequences.fasta > aligned.fasta

# Read from stdin
cat sequences.fasta | mafft-rs > aligned.fasta

# Write to file
mafft-rs -o aligned.fasta sequences.fasta

# Quiet mode (suppress progress messages)
mafft-rs -q sequences.fasta > aligned.fasta

# Control thread count (0 = all cores)
mafft-rs --thread 4 sequences.fasta > aligned.fasta
```

### Alignment strategies

```bash
# FFT-NS-2: fast progressive (default)
mafft-rs sequences.fasta

# FFT-NS-i: progressive + iterative refinement
mafft-rs --maxiterate 100 sequences.fasta

# G-INS-i: global iterative (accurate, for globally alignable sequences)
mafft-rs --globalpair --maxiterate 1000 sequences.fasta

# L-INS-i: local iterative (most accurate for <200 sequences)
mafft-rs --localpair --maxiterate 1000 sequences.fasta

# E-INS-i: generalized affine (for sequences with large internal gaps)
mafft-rs --genafpair --maxiterate 1000 sequences.fasta

# PartTree: divide-and-conquer guide tree (for 10K+ sequences)
mafft-rs --parttree sequences.fasta
```

### Adding sequences to an existing alignment

```bash
# Add new sequences to an existing aligned FASTA
mafft-rs existing_alignment.fasta --add new_sequences.fasta

# Add fragment sequences
mafft-rs existing_alignment.fasta --addfragments fragments.fasta

# Preserve existing alignment columns (no new gaps in original sequences)
mafft-rs existing_alignment.fasta --add new_sequences.fasta --keeplength
```

### Output formats

```bash
# FASTA (default)
mafft-rs sequences.fasta

# Clustal
mafft-rs --format clustal sequences.fasta

# PHYLIP
mafft-rs --format phylip sequences.fasta

# FASTA with custom line width (0 = no wrapping)
mafft-rs --linewidth 80 sequences.fasta
```

### Python

```python
import pymafft

# Align from list of sequences
result = pymafft.align(["ACDEFGHIK", "ACDEFHIK", "ACDHIK"])

# Align from named tuples
result = pymafft.align([("human", "ACDEFGHIK"), ("mouse", "ACDEFHIK")])

# Choose strategy
result = pymafft.align(seqs, strategy="linsi", maxiterate=1000)

# Align from file
result = pymafft.align_file("sequences.fasta")

# Access results
for seq in result:
    print(f"{seq.name}: {seq.sequence}")

result.to_fasta()    # FASTA string
result.to_tuples()   # list of (name, seq) tuples
```

#### Biopython interop

`pymafft.align` accepts `Bio.SeqRecord.SeqRecord` instances directly (duck-typed via `.id` and `.seq` attributes), and `result.to_fasta()` produces standard FASTA that `Bio.SeqIO` / `Bio.AlignIO` can read back. No glue code needed.

```python
import pymafft
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from io import StringIO

# Input: SeqRecord list (typical Biopython workflow)
records = list(SeqIO.parse("opsins.fasta", "fasta"))
result = pymafft.align(records, strategy="linsi", maxiterate=1000)

# Output → SeqRecord list
aligned_records = [
    SeqRecord(Seq(s.sequence), id=s.name)
    for s in result.sequences
]

# Output → MultipleSeqAlignment
msa = AlignIO.read(StringIO(result.to_fasta()), "fasta")
print(msa.get_alignment_length(), len(msa))
print(msa[:, 0])  # first column across all sequences

# Save back
SeqIO.write(aligned_records, "opsins.aligned.fasta", "fasta")
```

Notes on the conversion boundary:
- pymafft uses `SeqRecord.id` as the alignment name; `description` is dropped (intentional — matches what you'd expect from `mafft input.fa` on the CLI).
- pymafft outputs `result[i].sequence` as a plain `str`, not `Bio.Seq.Seq`. Wrap with `Seq(s.sequence)` if you need Biopython operations.
- The Biopython dependency is **optional** — pymafft has no runtime dep on Biopython. The interop happens via duck-typing.

## Migration from MAFFT (C)

The original MAFFT uses a shell script wrapper that invokes multiple C binaries. `mafft-rs` consolidates everything into a single binary with compatible flags.

### Command mapping

| Original MAFFT (C) | MAFFT-rs | Notes |
|---------------------|----------|-------|
| `mafft input.fa` | `mafft-rs input.fa` | FFT-NS-2 default |
| `mafft --maxiterate 100 input.fa` | `mafft-rs --maxiterate 100 input.fa` | FFT-NS-i |
| `mafft --globalpair input.fa` | `mafft-rs --globalpair input.fa` | G-INS-1 |
| `mafft --globalpair --maxiterate 1000 input.fa` | `mafft-rs --globalpair --maxiterate 1000 input.fa` | G-INS-i |
| `mafft --localpair input.fa` | `mafft-rs --localpair input.fa` | L-INS-1 |
| `mafft --localpair --maxiterate 1000 input.fa` | `mafft-rs --localpair --maxiterate 1000 input.fa` | L-INS-i |
| `mafft --ep 0 --genafpair --maxiterate 1000 input.fa` | `mafft-rs --genafpair --maxiterate 1000 input.fa` | E-INS-i |
| `linsi input.fa` | `linsi input.fa` (symlink) | L-INS-i shortcut |
| `ginsi input.fa` | `ginsi input.fa` (symlink) | G-INS-i shortcut |
| `einsi input.fa` | `einsi input.fa` (symlink) | E-INS-i shortcut |
| `fftnsi input.fa` | `fftnsi input.fa` (symlink) | FFT-NS-i shortcut (C default is `--maxiterate 2`) |
| `nwns input.fa` | `nwns input.fa` (symlink) | NW-NS-2 shortcut |
| `nwnsi input.fa` | `nwnsi input.fa` (symlink) | NW-NS-i shortcut |
| `qinsi input.fa` | `qinsi input.fa` (symlink) | Q-INS-i shortcut |
| `xinsi input.fa` | `xinsi input.fa` (symlink) | X-INS-i shortcut |
| `mafft --thread N input.fa` | `mafft-rs --thread N input.fa` | Multithreading |
| `mafft --op 1.53 input.fa` | `mafft-rs --op 1.53 input.fa` | Gap opening penalty |
| `mafft --ep 0.123 input.fa` | `mafft-rs --ep 0.123 input.fa` | Offset penalty |
| `mafft --bl 80 input.fa` | `mafft-rs --bl 80 input.fa` | BLOSUM matrix number |
| `mafft --jtt 200 input.fa` | `mafft-rs --jtt 200 input.fa` | JTT scoring matrix at PAM N |
| `mafft --tm 200 input.fa` | `mafft-rs --tm 200 input.fa` | Transmembrane scoring matrix at PAM N |
| `mafft --kimura 2 input.fa` | `mafft-rs --kimura 2 input.fa` | Kimura distance parameter |
| `mafft --allowshift --globalpair --maxiterate 1000 input.fa` | `mafft-rs --allowshift --globalpair --maxiterate 1000 input.fa` | G-INS-i with shift |
| `mafft --add new.fa existing.fa` | `mafft-rs existing.fa --add new.fa` | Add sequences to alignment |
| `mafft --addfragments frags.fa existing.fa` | `mafft-rs existing.fa --addfragments frags.fa` | Add fragment sequences |
| `mafft --nofft input.fa` | `mafft-rs --nofft input.fa` | Disable FFT (NW-NS-2) |
| `mafft --clustalout input.fa` | `mafft-rs --format clustal input.fa` | Clustal output |
| `mafft --phylipout input.fa` | `mafft-rs --format phylip input.fa` | PHYLIP output |
| `mafft --adjustdirection input.fa` | `mafft-rs --adjustdirection input.fa` | DNA strand auto-detect (k-mer) |
| `mafft --adjustdirectionaccurately input.fa` | `mafft-rs --adjustdirectionaccurately input.fa` | DNA strand auto-detect (DP) |
| `mafft --pileup input.fa` | `mafft-rs --pileup input.fa` | Comb-tree pile-up alignment |
| `mafft --bestfirst input.fa` | `mafft-rs --bestfirst input.fa` | Best-first refinement strategy |
| `mafft --skipiterate F input.fa` | `mafft-rs --skipiterate F input.fa` | Skip refinement for distant subtrees |
| `mafft --oneiteration input.fa` | `mafft-rs --oneiteration input.fa` | Stop after a single refinement iteration |
| `mafft --distout input.fa` | `mafft-rs --distout input.fa` | Write pairwise distance matrix |
| `mafft --scoreout input.fa` | `mafft-rs --scoreout input.fa` | Print sum-of-pairs score |
| `mafft --nodeout input.fa` | `mafft-rs --nodeout input.fa --maxiterate 0` | Write per-node density to .tree |
| `mafft --mapout` / `--compactmapout` | `mafft-rs --mapout` / `--compactmapout` | Write add-position map after `--add --keeplength` |
| `mafft --youngestlinkage` / `--averagelinkage` / `--minimumlinkage` / `--mixedlinkage F` | same | Tree-linkage variants |

## Architecture

The project is organized as a Cargo workspace with 12 crates:

```
crates/
  mafft-c-bindings/  Internal-only FFI shim that compiles MAFFT's C source
                     for cross-validation tests. Never shipped at runtime.
  mafft-sys/         Reserved name on crates.io (0.0.1 stub; future real FFI
                     shim with vendored C source will land here as 0.1.0+)
  mafft-types/       Shared Rust types (HomologyRegion, Sequence, ScoringContext, etc.)
  mafft-io/          FASTA, Clustal, PHYLIP, hat2 I/O
  mafft-scoring/     Substitution matrices (BLOSUM, JTT, TM, DNA) and gap penalties
  mafft-fft/         FFT-based homology detection (hand-ported Cooley-Tukey, bit-for-bit C-compat)
  mafft-align/       Pairwise and profile alignment algorithms (NW, SW, generalized affine, warp DP)
  mafft-tree/        Distance computation, NJ, UPGMA, guide tree construction, PartTree, memsavetree
  mafft-core/        Progressive alignment engine, iterative refinement, MafftEngine, `--add` machinery
  mafft/             Ergonomic top-level crate — `cargo add mafft` re-exports mafft-core/types/io
  mafft-bin/         CLI binary (`mafft-rs`; `cargo install mafft-rs`)
  pymafft/           Python bindings via PyO3 (`pip install pymafft`)
```

The release binary (`mafft-rs`) compiles with **zero C code** — `mafft-c-bindings` is only used as a dev-dependency for cross-validation tests.

### Key design decisions

- **No global state.** The C code uses ~400 `extern` globals. Rust modules use owned `ScoringContext`, `Topology`, `Profile` structs passed explicitly. This makes the engine reentrant, embeddable, and trivially thread-safe at the call-site boundary.

- **`num_complex::Complex64`** replaces the C `Fukusosuu` struct. A hand-ported bit-for-bit Cooley-Tukey FFT in `mafft-fft/src/fft_c_compat.rs` matches C's `fft.c` rounding exactly. Off-the-shelf FFT libraries vary in butterfly grouping order, producing 1-ULP correlation differences that flip FFT-anchor selection on flat-landscape matrices; the hand port is required for byte-identity.

- **`f64` DP matrices with `f64::mul_add` (FMA).** The substitution matrix is stored as `Vec<Vec<i32>>` and exposed as `consweight_matrix: Vec<Vec<f64>>` for DP arithmetic (matches C's `n_dis_consweight_multi`). gcc/clang `-O3` with `FP_CONTRACT=on` fuse `a + b * c` into a single-rounding FMA; plain Rust `+=` produces two roundings. Rust uses `mul_add` in `match_calc_row` and gap-frequency-modulated penalties to match C bit-for-bit.

- **Rayon parallelism** for pairwise distance computation, all-vs-all local alignments, and refinement scoring. Sequential float summation is preserved at accept/reject decision boundaries to keep refinement output deterministic across thread counts.

- **SIMD-friendly inner loops**: branchless patterns in `match_score()`, `pairwise_score()`, and `pairwise_identity_distance()` that LLVM auto-vectorizes to NEON/AVX instructions.

- **Three-way gap insertion** (`apply_c_insertnewgaps` in `mafft-core/src/progressive.rs`): full port of C `addfunctions.c::insertnewgaps` + `profilealignment` for the `--add` path, including profile-alignment-driven compression at adjacent new-merge-gap and common-gap restoration columns. Byte-identical to C on canonical and adversarial inputs.

- **Platform-independent sort tie-break.** C MAFFT calls `qsort()` directly inside `splitseq_mq` to sort sequences by their distance to the pivot. `qsort` is part of the host C library — BSD on macOS, glibc on Linux, MSVC on Windows — and its three implementations disagree on the relative order of *truly tied* elements. That makes C MAFFT's `--parttree --reorder` output platform-dependent on inputs containing exactly-duplicate sequences. Rust ships a hand-written BSD-qsort algorithm (`mafft-tree/src/bsd_qsort.rs`, Bentley-McIlroy) and uses it everywhere `dcompare_sort` is called, so the `mafft-rs` binary produces the **same output on every platform it's built for**, matching macOS C MAFFT 7.526 byte-for-byte.

- **Thread-local scratch pools.** The hot inner DP loop (`profile_align_imp_multimtx` in `crates/mafft-align/src/profile.rs`) is specialised at codegen time via a `#[inline(always)] fn j_loop<const TW: bool, const STRICT: bool>` helper, dispatched from the i-loop with a 4-way match — LLVM strips the warp arithmetic from the FFT-NS-i path and replaces per-cell branches on `strict_part_tiebreak` with const choices. Per-call scratch vectors come from thread-local pools (`DpScratch` / `DP_H_POOL` / `DP_IJP_POOL`) to amortise allocation across calls, matching C `A__align`'s static-TLS buffer strategy.

### Test suite

**419 Rust tests + 72 Python tests** (release build, all suites green):

| Suite | Count | What |
|-------|-------|------|
| Rust lib tests (`--lib`, all crates)  | ~170 | All `#[cfg(test)] mod tests` blocks |
| Rust binary tests (`mafft-rs`)        | 23   | CLI helper functions |
| Rust integration (`end_to_end`)       | 117  | Mode-level byte-identity vs C across every supported flag/mode combo |
| Rust FFI cross-validation             | ~80  | Cell-by-cell FFI equality: `G__align11`, `A__align`, `genL__align11`, `Falign`, `MSalignmm`, `compacttree_memsaveselectable`, `insertnewgaps`, weights, distances |
| `trace_refinement`, `integration`, `imp_*` | ~32 | End-to-end trajectory cross-checks |
| Python tests                          | 72   | API, strategies, file I/O, **parity with `mafft-rs` and C `mafft` binaries**, output-shape invariants, edge cases, **Biopython interop** |

Regression guards for C parity live in:
- `crates/mafft-core/tests/end_to_end.rs` — mode-level byte-identity
- `crates/mafft-core/tests/cross_validate_*.rs` — FFI-level cell/function equality
- `crates/mafft-tree/tests/cross_validate_*.rs` — PartTree, memsavetree, weights

Any regression in DP indexing, boundary handling, FFT anchor placement, refinement-tree distance, profile blending, or `--add` profilealignment fails at least one of them.

6 tests are `#[ignore]`-gated as deliberate diagnostic / bisection tools (annotated at the call site).

## Known limitations

These three categories cover everything `mafft-rs` does not currently produce byte-identical output to C MAFFT 7.526 for.

### A. External-dependency-blocked modes

These modes are wired correctly but require third-party binaries not shipped by upstream MAFFT; end-to-end validation depends on the binary being on `PATH`. Same limitation as C MAFFT itself faces.

| Flag | Requires | Status |
|------|----------|--------|
| `--xinsi` (X-INS-i) | Stanford [CONTRAfold v2.02+](http://contra.stanford.edu/contrafold/) | Wired; emits "contrafold not found" diagnostic if absent |
| `--qinsi` (Q-INS-i) | `mxscarnamod` built from `mafft-upstream/extensions` | Wired; byte-identical to C (mod RNA case) when present |
| `--scarnalike` | `dash_client` in PATH | Wired; untested without the binary |

### B. Intentional residual (pathological-value tied-trace)

A single mode-and-value combination is known to produce a tied-trace divergence from C MAFFT — same alignment width, same sum-of-pairs score, but a few residues placed at neighbouring columns due to a DP tie-break that depends on accumulated floating-point state. No realistic workflow hits this:

| Configuration | Effect |
|---------------|--------|
| `--exp ≥ 4.30` on FFT-NS-2 (no refinement) | 16-line content diff at same width (517) and same score on the 36-seq sample. Default `--exp` is 0; closes with `--nofft` or any `--maxiterate ≥ 1`. |

Full diagnostic infrastructure is in place (`RS_DP_DUMP` env var, `exp_residual_dump_compare` and `exp_residual_step10` FFI tests) for any future closure attempt. The divergence is below per-step FFI replay resolution: per-step `fft_profile_align` output is byte-identical to C's `Falign` on the same inputs; the end-to-end shift comes from accumulated FP order in tied DP cells that the per-step replay can't reproduce.

### C. Out-of-scope by design

These modes are intentionally not implemented. `mafft-rs` either rejects the flag or emits the same "non-functional" stub as C MAFFT does.

| Category | Flags | Why |
|----------|-------|-----|
| **PDB structure-aware alignment** | `--pdbidlist`, `--pdbfilelist` | Disabled in upstream C MAFFT itself since December 2018 (`scripts/mafft:969-990`: "temporarily unavailable, 2018/Dec." → `exit`). `mafft-rs` matches that exit byte-for-byte. |
| **RNA-structure alignment via external secondary-structure tools** | DAFS, FoldAlign, LARA, SCARNA, MCCASKILL, RIBOSUM, RNAALIFOLD | Require fundamental new infrastructure for RNA folding and secondary-structure parsing; out-of-scope for the rust reimplementation. |
| **Alternative pairwise aligners** | `--blastpair`, `--lastpair`, `--lastmultipair`, `--fastapair`, `--fastswpair`, `--hybridpair`, `--longshortpair`, `--shortlongpair` | Require external BLAST/LAST/FASTA binaries with non-trivial output-parsing glue. Use the built-in `--localpair` / `--globalpair` / `--genafpair` aligners instead. |
| **DASH-only sequence-filter flags** | `--excludehomologs`, `--originalseqonly` | C documents both as "works with `--dash` only"; the DASH structure-DB pipeline falls under the RNA-structure category above. Wired as no-ops, matching what C does without `--dash`. |
| **MPI parallelism** | `--mpi` | `mafft-rs` uses Rayon for in-process multithreading. |

### Case preservation

C preserves the input case of residues (e.g., lowercase RNA); `mafft-rs` uppercases all residues before alignment. The alignment itself (gap placement) is identical — `rna_nofft_case_insensitive_identical_to_c` verifies this with `diff -i`.

## Performance

Benchmark snapshot vs C MAFFT 7.526 (macOS arm64, alternating-run medians):

| Mode / input                                | C MAFFT | mafft-rs | Ratio              |
|---------------------------------------------|---------|----------|--------------------|
| default FFT-NS-2 (108-seq synthetic)        | 1.34s   | 1.16s    | **rust 1.16× faster** |
| `--maxiterate 100` FFT-NS-i (36-seq sample) | 0.785s  | 0.670s   | **rust 1.17× faster** |
| `--maxiterate 100` FFT-NS-i (BB12019)       | 0.333s  | 0.198s   | **rust 1.68× faster** |
| `--maxiterate 100` FFT-NS-i (BB30004)       | 1.366s  | 1.246s   | **rust 1.10× faster** |
| `--allowshift --globalpair --maxiterate 1000` (36-seq) | 2.586s | 1.756s | **rust 1.47× faster** |
| `--maxiterate 50 --localpair` (108-seq)     | 37.00s  | 22.93s   | **rust 1.61× faster** |
| `--maxiterate 50 --globalpair` (108-seq)    | 35.32s  | 22.62s   | **rust 1.56× faster** |
| `--parttree` (108-seq)                      | 1.70s   | 0.09s    | **rust 18× faster**   |
| `--dpparttree` (108-seq)                    | 7.24s   | 0.08s    | **rust 90× faster**   |

## Upstream MAFFT

The original MAFFT C code is included as a git submodule at `mafft-upstream/`, pinned to commit `ee97999` (version 7.526). To update:

```bash
cd mafft-upstream
git fetch origin
git checkout <new-commit-or-tag>
cd ..
git add mafft-upstream
cargo test --workspace  # verify nothing breaks
git commit -m "Bump MAFFT upstream to <version>"
```

## CI/CD

| Workflow | Triggers | What |
|----------|----------|------|
| **CI** (`ci.yml`) | push/PR to main/dev | Build + test Rust and C, smoke test binary |
| **Python** (`python.yml`) | push/PR + release | Build wheels (Linux, macOS, Windows), test on Python 3.9-3.13, publish to PyPI on release |
| **Release** (`release.yml`) | GitHub release | Build `mafft-rs` binaries for 4 platforms, attach to release |

## Acknowledgments and Citation

rust-MAFFT is a port — every algorithmic decision in this codebase
traces back to work by **Kazutaka Katoh** and colleagues. The FFT-
anchored alignment, the iterative-refinement strategies (L-INS-i /
G-INS-i / E-INS-i), the scoring matrices, the PartTree heuristic for
large datasets — all of it is theirs. This project is engineering on
top of their science, and it would not exist without two decades of
their public, open work on MAFFT. Thank you.

### How to cite

If you use rust-MAFFT in published work, **you must cite the original
MAFFT paper.** The rust-MAFFT manuscript (in preparation for
*Bioinformatics*) should be cited additionally once published — but
the algorithmic foundation always remains the Katoh et al. work, and
that citation never goes away.

**Primary citation — MAFFT v7 (always required):**

> Katoh, K., & Standley, D. M. (2013). MAFFT multiple sequence
> alignment software version 7: improvements in performance and
> usability. *Molecular Biology and Evolution*, 30(4), 772–780.
> https://doi.org/10.1093/molbev/mst010

```bibtex
@article{Katoh2013,
  author  = {Katoh, Kazutaka and Standley, Daron M.},
  title   = {{MAFFT} Multiple Sequence Alignment Software Version 7:
             Improvements in Performance and Usability},
  journal = {Molecular Biology and Evolution},
  volume  = {30},
  number  = {4},
  pages   = {772--780},
  year    = {2013},
  doi     = {10.1093/molbev/mst010}
}
```

**Foundational citation — original MAFFT (FFT-NS-1/2 modes):**

> Katoh, K., Misawa, K., Kuma, K., & Miyata, T. (2002). MAFFT: a novel
> method for rapid multiple sequence alignment based on fast Fourier
> transform. *Nucleic Acids Research*, 30(14), 3059–3066.
> https://doi.org/10.1093/nar/gkf436

```bibtex
@article{Katoh2002,
  author  = {Katoh, Kazutaka and Misawa, Kazuharu and
             Kuma, Kei-ichi and Miyata, Takashi},
  title   = {{MAFFT}: a novel method for rapid multiple sequence
             alignment based on fast {F}ourier transform},
  journal = {Nucleic Acids Research},
  volume  = {30},
  number  = {14},
  pages   = {3059--3066},
  year    = {2002},
  doi     = {10.1093/nar/gkf436}
}
```

See [`CITATION.cff`](CITATION.cff) for the machine-readable form and
[the citation page](https://luksgrin.github.io/rust-MAFFT/citation/) for
additional references (PartTree, MAFFT v5, etc.) you should cite when
using mode-specific features.

## License

This project is distributed under the SPDX expression `MIT AND
BSD-3-Clause`:

- **MIT** — original Rust code in this repository (the engine port,
  bindings, CLI, build system, tests). See [`LICENSE-MIT`](LICENSE-MIT).
- **BSD-3-Clause** — algorithmic constructs and any verbatim
  translations carried over from upstream MAFFT. Copyright Kazutaka
  Katoh. See [`LICENSE-BSD`](LICENSE-BSD).

The `mafft-upstream/` submodule contains the original C MAFFT under
its [own BSD-3-Clause license](mafft-upstream/license).

## Authors

- **Lucas Goiriz** ([@luksgrin](https://github.com/luksgrin)) — Rust
  port, library design, bindings, tests, documentation
- **Original MAFFT**: [**Kazutaka Katoh**](https://mafft.cbrc.jp/alignment/software/)
  and collaborators (CBRC, Osaka University). All scientific credit
  belongs to them.

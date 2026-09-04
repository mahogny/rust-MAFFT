# Changelog

All notable changes to rust-MAFFT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from 1.0.0 onwards. Pre-1.0 releases may contain breaking changes between
minor versions; the `mafft` and `mafft-rs` crates aim to keep their
public surfaces stable from 0.1.0 anyway.

## [Unreleased]

### Added

- `--nuc` / `--amino`: force the input sequence type, overriding the
  ATGC-frequency auto-detection (matches C MAFFT `scripts/mafft:547-550`,
  `seqtype="-D"` / `seqtype="-P"`). Mutually exclusive; inert when absent,
  so auto-detection is unchanged for every existing command line.
- `mafft_rs::run_from(argv, out)`: the argv-driven, non-exiting form of
  `run()`. Same clap definition and therefore exactly the same flag
  semantics (`--auto`'s size heuristic, `--adjustdirection`'s strand
  detection, …), but every path where `run()` calls `std::process::exit(N)`
  returns a `MafftError` carrying the same message text and exit code, and
  the alignment is written to a caller-supplied `io::Write` instead of
  stdout. `--output FILE` still writes to that file.
- `mafft_rs::MafftError`, with `code()` and `message()`.
- `mafft_rs::Mafft`: a typed builder that constructs an argv and hands it
  to `run_from`, so it cannot drift from the command line.
- `mafft_io::apply_case_convention`: applies C MAFFT's residue-case fold
  (lowercase nucleotide, uppercase protein) to a parsed `SequenceSet`.
- `MAFFT_RS_REFINE_STATS=1` prints one line per iterative-refinement call to
  stderr: `refine: nseq=.. len=.. cycles=n/max visited=.. changed=..
  accepted=.. exit=maxiter|converged|oscillation`, plus a
  `refine-segments: anchors=.. segments=..` line for the segmented
  (FFT-NS-i) path. C's `dvtditr` reports its refinement work directly
  (`Segment n/N`, then one line per branch), so this makes "did both sides
  run the same cycles?" answerable without a debugger — a speed comparison
  is meaningless otherwise. Off by default; CLI output is unchanged.
- Progress sink: `mafft_rs::Progress` (one method, `message(&self, &str)`)
  with `StderrProgress` (current behaviour) and `SilentProgress`, plus a
  blanket impl so any `Fn(&str)` is a sink.
  `mafft_rs::run_from_with_progress(argv, out, &(dyn Progress + Sync))`
  routes the run's 12 progress messages there instead of stderr, and
  `Mafft::progress(sink)` does the same for the builder. Both default to
  stderr, so `run_from`, `run` and the CLI are unchanged. The sink is taken
  by shared reference and called through `&self`, so one sink can serve a
  whole worker pool. Only progress is routed — failures come back as
  `MafftError`, and non-fatal `Warning:` / `Could not …` diagnostics plus
  `--scoreout`'s score line stay on stderr, so a silent sink cannot hide
  either a problem or requested output. The sink covers every progress
  message `mafft-rs` emits; two further lines in `mafft-core` (the
  Q-INS-i/X-INS-i BPP line and the `--skipiterate` banner) are not routed —
  see the `progress` module docs for why.

### Changed

- `--thread N` now builds a *local* rayon pool and `install()`s the
  alignment into it instead of calling `build_global()`. A process-global
  pool can only be initialised once, so an in-process caller running many
  alignments was previously stuck with the first call's thread count.
  No change for the CLI.
- `run()` is now a thin wrapper around `run_from` (unchanged signature and
  observable behaviour: same stdout, stderr, exit codes and messages).

### Fixed

- **DNA pairwise gap penalties were a third of C MAFFT's, so L-INS-i /
  G-INS-i / E-INS-i diverged on nucleotide input.** C scales the pair-phase
  gap penalties by `3 * 600/1000` for nucleotide and `600/1000` for protein
  (`constants.c:316-322` vs `:672-677`), keeping the offset at `1 * 600/1000`;
  the pair phase here applied the protein factor unconditionally (a variable
  named `scale_protein`). Gaps were too cheap, so the local/global pairwise
  step bought extra matches with gaps C refuses, and the `hat3` constraints
  and final alignment followed. Now `pair_penalty_scales(is_nucleotide)`,
  with a unit test pinning the `3 *`. Protein and FFT-NS-2 were never
  affected (protein has no `3 *` in C either; FFT-NS-2 does not use these
  penalties).

  **DNA pairwise alignments will change.** Any `--localpair`, `--globalpair`,
  `--genafpair` or `--auto` run on nucleotide input can now produce a
  different alignment than before. As with the case fold, this is a
  correction *toward* the C MAFFT 7.526 reference the crate claims
  byte-identity with, not a behaviour of our own: on 60 synthetic clusters
  L-INS-1 went from 24/60 to 60/60 byte-identical with C, and on 30
  clusters evolved from real biological ancestors `--auto` went from 14/30
  to 30/30. Minimal reproducer: two 15 bp sequences under
  `--localpair --maxiterate 0` (`crates/mafft-bin/tests/fixtures/dna_pair_gapscale_min.fa`).
- Two latent instances of the per-alphabet-constant class, found by the
  audit recorded under Notes rather than by a parity failure. Neither was
  reachable from the engine, so no output changes: `GapModel::default()` was
  protein-shaped *and* arithmetically wrong (`-918`; C truncates toward zero,
  giving `-917`) and is now correct and documented as protein-only; and the
  **public** `FftAlignParams::dna()` inherited that protein gap default
  instead of DNA's `-2753`, which would have mis-scaled gaps for any
  downstream caller using it. `GapPenalties::default()` holds unscaled
  `ppenalty`-style units unlike every other `GapPenalties` in the tree; it is
  unused, and now says so.
- **Removed every `f64::mul_add`; the reference C build emits no FMA.**
  74 call sites across the DP, FFT, constraints, refinement, UPGMA cluster
  distances and branch weights used fused multiply-add (one rounding) where
  C does a separate multiply and add (two). Measured rather than assumed:
  disassembling C MAFFT 7.526 — both the conda binary parity is defined
  against and a clean build from the pinned source with the project's own
  `-O3` flags — gives `vfmadd`/`vfmsub` counts of **0** across `disttbfast`,
  `dvtditr` and `tbfast`, against ~1250 `mulsd` and ~1550 `addsd`. Baseline
  x86-64 has no FMA, so gcc cannot contract. Several code comments asserting
  that `gcc -O3` fuses were simply wrong, and two more were calibrated
  against Apple clang on arm64 rather than the reference build; all are
  corrected, with the measurement recorded in `mafft-align`'s module docs.

  This closed every remaining nucleotide and protein parity residue on
  BAliBASE: DNA default **139/141 -> 141/141**, DNA FFT-NS-i
  **132/141 -> 141/141**, DNA `--auto` **137/140 -> 141/141**, protein
  default **379/386 -> 386/386**. It is also a large speedup, since the
  fused form was being emulated in software on a target without FMA: a
  120-sequence 1.4 kb FFT-NS-i run goes **22.9 s -> 15.1 s** (-34 %), from
  slower than C MAFFT to faster (C: 16.9 s).

  `tests/fixtures/sample.bl50.fftns2` was regenerated: it had been captured
  from a build that *did* contract, and a 2026-05 change had switched the DP
  to `mul_add` to match that fixture rather than the reference binary. The
  reference produces width 738, not the 712 recorded there.
- **`--thread N` (N >= 1) now uses C's `athread` convergence rule.** C picks
  its refinement implementation on `nthread > 0` (`tditeration.c:1433`) and
  the two do not converge alike: the single-threaded path tests
  `converged >= locnjob * 2` after **every branch** and stops immediately,
  mid-cycle (`:2328-2342`), while `athread`'s collector tests once per
  **cycle** whether any branch gained (`maxgain > 0.0`, `:589`) and only
  stops at the top of the next cycle, where the `else` arm `pthread_exit`s
  (`:527-551`) — so the converging cycle always completes. C's own output
  shows it: at `--maxiterate 2`, 22 of 85 segments print `Converged.` alone,
  56 print `Converged.` *and* `Reached 2`, and 7 print `Reached 2` alone.
  We modelled only the single-threaded rule. Now selected by
  `MafftEngine::nthread`, matching C's `-C` mapping exactly (both no
  `--thread` and `--thread 0` give C `-C 0`, i.e. the single-threaded rule).

  **DNA output changes for `--thread N >= 1` with refinement.** On the
  120-sequence reproducer, distance from C `--thread 1` goes 6 lines → 2,
  and the synthetic cluster corpus under
  `--auto --adjustdirection --thread 1 --nuc` goes 59/60 → **60/60**. The
  no-`--thread` path is untouched and remains byte-identical to C.
  A 2-line residue remains on the 120-sequence input (one sequence, a
  single-column gap shift at equal width); it is not yet explained.
- **DNA refinement guide trees used the protein `dndpre` offset, reordering
  UPGMA merges.** For modes with no `pairlocalalign` step (FFT-NS-i and
  friends) the refinement tree is rebuilt the way C's `dndpre` does. C's
  script does not pass `-h` to that `dndpre` call, so `constants()` uses its
  DEFAULT `poffset` — and the alphabets do not share one: `DEFAULTOFS_N =
  -369` (`DNA.h:3`) gives a matrix shift of 220, `DEFAULTOFS_B = -123`
  (`blosum.c:3`) gives 73. The shift was hardcoded to the protein 73.

  On DNA that produced refinement distances differing from C's `hat2`
  outright (on a 120-sequence 1.4 kb input, leaf pair (0,58): C 0.253, ours
  0.307), which swapped two UPGMA merges, which changed the group on 131 of
  237 refinement branches and so the final alignment. Now
  `dndpre_offset_shift(is_nucleotide)`, with a unit test pinning both values
  against the C constants.

  **DNA output changes for FFT-NS-i and any refinement mode without a
  pairwise phase.** Byte-parity with C MAFFT 7.526 over BAliBASE `bali2dna`
  (141 real DNA benchmark sets) under `--maxiterate 2` goes **67/141 → 132/141**,
  and the 120-sequence reproducer becomes byte-identical. Protein is
  unaffected (it already used 73), as are FFT-NS-2 and `--localpair`, which
  never reach this path.
- **Two-sequence inputs were never refined.** Every refinement entry point
  returned early at `nseq <= 2`. C does not skip a pair: `dvtditr.c:704-708`
  sets `weight = 0; niter = 1` for `njob == 2`, `tditeration.c:772` then
  uses uniform weights, and `:1425` gates branch-weight computation on
  `locnjob > 2`. So a pair is refined exactly once, unweighted — which can
  change it (e.g. `AA`/`CC` under `--maxiterate 1000`: C gives `aa`/`cc`,
  we gave `aa-`/`-cc`). Guards relaxed to `nseq < 2` and the iteration cap
  mirrors `niter = 1`; `BranchWeights` already yielded uniform weights at 2.
  Affects `--maxiterate N > 0` and every `*-INS-i` mode, including `--auto`,
  on exactly-two-sequence input, DNA and protein alike.
- **Nucleotide output case now matches C MAFFT.** DNA/RNA alignments were
  emitted in uppercase; C MAFFT emits them in lowercase. C folds residue
  case as it reads — `io.c:1462-1467` (`load1SeqWithoutName_realloc`) calls
  `onlyAlpha_lower` when `dorp == 'd'` and `onlyAlpha_upper` otherwise, and
  `readData_pointer` repeats the nucleotide pass with `seqLower`
  (`io.c:1755`; its `upperCase != -1` guard is only reachable from the
  legacy non-FASTA `FRead` header parser, so it is always true for FASTA
  input). rust-MAFFT's reader applied `onlyAlpha_upper` unconditionally.
  It now folds per the sequence type, and `--nuc` / `--amino` re-apply the
  fold for the type they force, because C's `$seqtype` fixes `dorp` before
  any sequence is read.

  **This changes existing output**: a DNA/RNA alignment that previously
  came back uppercase now comes back lowercase. That is the point — it is
  what C MAFFT 7.526 produces, and whole-file case flips were breaking
  byte-for-byte comparisons against a C-MAFFT reference. Protein output is
  unchanged (uppercase, as before and as in C). `--anysymbol` /
  `--preservecase` are unchanged too: they keep the input's own case, in
  both C and here.

  This closes the parity caveat the README recorded as
  "byte-exact (case-insensitive)"; `--nofft samplerna` is now byte-exact
  with `cmp`, not just with `diff -i`.
- `--adjustdirection` / `--adjustdirectionaccurately` help text claimed the
  k-mer strand detection was "not yet implemented"; it has been implemented
  since TODO R-5 (2026-06-03).

### Notes

- **Per-alphabet constant audit.** Three bugs were found reactively where a
  constant correct for one alphabet was applied on a path serving both
  (`scale_protein` in the pair phase, the `dndpre` offset, and the two latent
  ones above). Every value in the translation deriving from C's
  `constants()`, `DNA.h`, `blosum.c` or `JTT.c` has now been enumerated and
  checked against *both* C branches (nucleotide `constants.c:296-326`,
  protein `:664-682` / `:895-910`) — 17 live sites, all correct.
  `mafft_scoring::penalties` carries a test asserting every one of them for
  both alphabets at once, including that the `offset*` values take C's `1 *`
  factor on the nucleotide branch while the gap penalties take `3 *`, so a
  future edit cannot give one alphabet the other's constant unnoticed.

- C MAFFT 7.526 genuinely produces different output for *no* `--thread` than
  for `--thread 1` — it selects a different refinement implementation on
  `nthread > 0` (`tditeration.c:1433`), and the two converge by different
  rules. This was previously recorded here as "C-side sensitivity" that
  rust-MAFFT could not match; that was wrong, and rust-MAFFT now reproduces
  both paths (see the `--thread` entry under Fixed). Both are deterministic:
  5/5 identical over repeat runs.
- Still genuinely unmatchable: `--thread N` for **N >= 2**. C is
  nondeterministic there — the same binary on the same input produced 2
  distinct outputs over 3 runs at both `--thread 2` and `--thread 4` — so
  byte-identity with C is impossible in principle at those thread counts.
  rust-MAFFT remains deterministic across all thread counts.

## [0.1.2] - 2026-06-10

Metadata fixes. No engine, library, or CLI behaviour changes from
0.1.1 / 0.1.0; bumped so that all distribution channels can publish
together cleanly.

### Fixed

- `CITATION.cff`: dropped the SPDX expression `MIT AND BSD-3-Clause`
  in favour of the single SPDX identifier `MIT` (Zenodo's `cffconvert`
  pipeline rejected the expression form with "Citation metadata load
  failed"). The BSD-3-Clause attribution for the algorithmic constructs
  ported from upstream MAFFT remains in `LICENSE-BSD` and the workspace
  `Cargo.toml` `license = "MIT AND BSD-3-Clause"` (where cargo accepts
  expressions fine).
- Maintainer email updated to `lucas.goiriz@csic.es` in workspace
  authors and `pymafft` pyproject metadata.

### Notes

- v0.1.1 published partially: docs + GH-release binaries succeeded;
  crates.io got 5 of 10 crates before hitting the new-crate rate
  limit; PyPI publish was gated by a transient quay.io docker-pull
  flake on one Linux wheel. v0.1.2 retries all channels with a valid
  CITATION.cff.

## [0.1.1] - 2026-06-10

Release-pipeline fixes. No engine, library, or CLI behaviour changes
from `0.1.0`; bumped only because `pymafft 0.1.0` was published to
PyPI before the rest of the release-day workflows could be fixed.

### Fixed

- `release.yml`: added `permissions: contents: write` so
  `softprops/action-gh-release@v2` can attach the cross-compiled
  binaries to the GitHub release (previously failed with
  "Resource not accessible by integration").
- `docs.yml`: the `github-pages` deployment environment is now scoped
  to allow tag-triggered deploys (`v*`) in addition to `main` pushes.
- `python.yml`: x86_64-apple-darwin wheel is now cross-compiled from
  the Apple Silicon `macos-latest` runner (the previously-targeted
  `macos-13` Intel runner pool was retired by GitHub in 2026).
- `python.yml`: Linux wheel build no longer changes the cwd inside the
  manylinux container (`cd ../..` had broken maturin's manifest
  resolution; replaced with `--manifest-path ../../Cargo.toml`).

### Notes

- `pymafft 0.1.0` was yanked on PyPI after this release. Use 0.1.1+
  for any new installs. Programmatic and CLI behaviour is unchanged.

## [0.1.0] - 2026-06-10 [YANKED]

Initial release attempt. `pymafft` published to PyPI successfully but
the crates.io publish (email verification not yet completed), GitHub
release binary upload (missing `contents: write` permission), and
docs deploy (environment-rule rejected the tag) all failed. Superseded
by 0.1.1.

### Engine

- **Byte-identical to C MAFFT 7.526** across the full BAliBASE 3
  fixture set (1930/1930) for every supported mode: FFT-NS-2, FFT-NS-i,
  G-INS-i, L-INS-i, E-INS-i, PartTree, DPPartTree, plus `--add`,
  `--adjustdirection` (k-mer mode), `--oneiteration`, `--bestfirst`,
  `--allowshift`, `--unalignlevel`, `--skipiterate`, `--pileup`,
  `--youngestlinkage` / `--averagelinkage` / `--minimumlinkage` /
  `--mixedlinkage`.
- **Stub-parity** for `--pdbidlist` and `--pdbfilelist`: print the
  upstream "temporarily unavailable, 2018/Dec." message and exit 0 to
  match C MAFFT 7.526 behaviour verbatim (these were disabled
  upstream).
- **Pure Rust runtime**: the release binary compiles zero C code.
  `mafft-c-bindings` (the FFI cross-validation shim) is a dev-only
  internal crate.

### Distribution

Four parallel release channels off a single GitHub release tag:

- **Rust library**: `cargo add mafft` (top-level re-export of
  `mafft-core`, `mafft-types`, `mafft-io`). Sub-crates also published
  individually for finer-grained dependence.
- **Rust CLI**: `cargo install mafft-rs`.
- **Python package**: `pip install pymafft` ships 25 wheels (Linux
  x86_64 + aarch64, macOS Intel + Apple Silicon, Windows x86_64 ×
  Python 3.9–3.13) plus an sdist.
- **Python executable**: the `pymafft` wheel bundles the `mafft-rs`
  binary inside it; `pip install pymafft` puts `mafft-rs` on `$PATH`
  via a console-script entry. No Rust toolchain required.
- **Pre-built CLI binary**: attached to every GitHub release for 5
  targets (Linux x86_64 + aarch64, macOS Intel + Apple Silicon,
  Windows x86_64).

### Testing

- ~430 Rust tests including ~140 FFI cross-validation tests that
  compile MAFFT C source in-tree and compare per-function outputs
  byte-for-byte.
- 76 Python tests across binding parity, biopython interop, and the
  bundled-CLI smoke path.
- Full CI matrix on every push / PR: Linux byte-identity gate, plus
  macOS + Windows build + lib-test cross-platform sanity.

### Documentation

- Site at https://luksgrin.github.io/rust-MAFFT (Material for MkDocs)
- Per-crate `cargo doc` published to https://docs.rs

[Unreleased]: https://github.com/luksgrin/rust-MAFFT/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/luksgrin/rust-MAFFT/releases/tag/v0.1.2
[0.1.1]: https://github.com/luksgrin/rust-MAFFT/releases/tag/v0.1.1
[0.1.0]: https://github.com/luksgrin/rust-MAFFT/releases/tag/v0.1.0

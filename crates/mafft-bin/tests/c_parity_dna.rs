//! Byte-for-byte parity with C MAFFT 7.526 on nucleotide input.
//!
//! Every `.expected` file here is the verbatim stdout of C MAFFT 7.526
//! (`conda` build, `mafft --version` → `v7.526 (2024/Apr/26)`) for the
//! command named in the test, so these tests need no C installation.
//!
//! They pin two fixes and the class of input that detects them:
//!
//! * `dna_pair_gapscale_min` — the smallest input on which the pair-phase
//!   gap penalties for nucleotide were a third of C's (missing
//!   `constants.c:316-322` `3 *`). Two 15 bp sequences; C leaves the pair
//!   ungapped, the old code opened a gap.
//! * `refine_njob2_min` — the smallest input showing that C refines a
//!   *pair* (`dvtditr.c:704-708`: `njob == 2 → weight = 0; niter = 1`),
//!   where the old code skipped refinement at `nseq <= 2`.
//! * `r2_dna_clusters/` — 30 clusters of 2–6 sequences, 300–900 bp,
//!   85–99 % identity, evolved from real biological ancestors with
//!   substitutions and indels. Uniform-random or fixture-only corpora did
//!   not reliably surface the gap-scale bug; this shape did (16/30 before
//!   the fix), so it is the regression guard for that whole class.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
}

/// Run `mafft-rs <flags> <input>` in-process and return the FASTA bytes.
fn run(flags: &[&str], input: &Path) -> Vec<u8> {
    let mut argv: Vec<OsString> = vec![OsString::from("mafft-rs"), OsString::from("--quiet")];
    argv.extend(flags.iter().map(OsString::from));
    argv.push(input.as_os_str().to_os_string());
    let mut out = Vec::new();
    mafft_rs::run_from(argv, &mut out)
        .unwrap_or_else(|e| panic!("mafft-rs {flags:?} {}: {}", input.display(), e.message()));
    out
}

fn expect_identical(flags: &[&str], input: &Path, expected: &Path) {
    let got = run(flags, input);
    let want = std::fs::read(expected).expect("read expected");
    if got != want {
        panic!(
            "mafft-rs {:?} {} differs from C MAFFT 7.526 ({})\n--- C ---\n{}--- rust ---\n{}",
            flags,
            input.display(),
            expected.display(),
            String::from_utf8_lossy(&want),
            String::from_utf8_lossy(&got),
        );
    }
}

/// Bug: nucleotide pair-phase gap penalties lacked C's `3 *`
/// (`constants.c:316-322`), so L-INS-1 opened a gap C refuses.
#[test]
fn linsi1_dna_pair_gap_scale_matches_c() {
    let f = fixtures();
    expect_identical(
        &["--localpair", "--maxiterate", "0"],
        &f.join("dna_pair_gapscale_min.fa"),
        &f.join("dna_pair_gapscale_min.linsi1.expected"),
    );
}

/// Bug: refinement was skipped at `nseq <= 2`; C runs it once, unweighted
/// (`dvtditr.c:704-708`, `tditeration.c:772`). FFT-NS-i and `--auto`
/// (L-INS-i) both exercise it.
#[test]
fn fftnsi_refines_a_pair_like_c() {
    let f = fixtures();
    expect_identical(
        &["--maxiterate", "1000"],
        &f.join("refine_njob2_min.fa"),
        &f.join("refine_njob2_min.fftnsi.expected"),
    );
}

#[test]
fn auto_refines_a_pair_like_c() {
    let f = fixtures();
    expect_identical(
        &["--auto"],
        &f.join("refine_njob2_min.fa"),
        &f.join("refine_njob2_min.auto.expected"),
    );
}

/// Differential test on realistic clusters under the pipeline invocation
/// `--auto --adjustdirection --thread 1 --nuc`. Every cluster must be
/// byte-identical to C MAFFT 7.526.
#[test]
fn r2_real_ancestor_clusters_match_c_under_auto() {
    let dir = fixtures().join("r2_dna_clusters");
    let mut inputs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("r2_dna_clusters")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "fa"))
        .collect();
    inputs.sort();
    assert_eq!(inputs.len(), 30, "expected 30 clusters in {}", dir.display());
    let mut failures = Vec::new();
    for input in &inputs {
        let expected = input.with_extension("expected");
        let got = run(&["--auto", "--adjustdirection", "--thread", "1", "--nuc"], input);
        let want = std::fs::read(&expected).expect("read expected");
        if got != want {
            failures.push(input.file_name().unwrap().to_string_lossy().into_owned());
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} clusters differ from C MAFFT 7.526: {failures:?}",
        failures.len(),
        inputs.len()
    );
}

//! Typed builder for in-process MAFFT runs.
//!
//! [`Mafft`] is deliberately *not* a second implementation of the flag
//! logic: every method appends to an argv vector, and [`Mafft::run`] hands
//! that vector to [`crate::run_from`]. So there is exactly one place where
//! a flag's meaning is decided — the clap definition plus `run_from` — and
//! the builder cannot drift from the command line.
//!
//! ```no_run
//! use mafft_rs::Mafft;
//!
//! let aligned = Mafft::new()
//!     .auto()
//!     .adjust_direction()
//!     .thread(1)
//!     .nuc()
//!     .input("in.fasta")
//!     .run_to_vec()?;
//! # Ok::<(), mafft_rs::MafftError>(())
//! ```
//!
//! Flags without a dedicated method (and any future ones) are reachable
//! through [`Mafft::flag`], [`Mafft::option`] and [`Mafft::arg`], so the
//! builder never becomes a bottleneck on the CLI surface.

use std::ffi::OsString;
use std::path::Path;

use crate::{run_from, MafftError};

/// Builds a `mafft-rs` argv and runs it in-process via [`crate::run_from`].
///
/// The semantics of every flag are exactly the command line's, including
/// `--auto`'s size-based strategy choice and `--adjustdirection`'s
/// pre-alignment strand detection.
#[derive(Debug, Clone)]
pub struct Mafft {
    /// argv[0] followed by the flags, in the order they were added.
    argv: Vec<OsString>,
    /// Positional INPUT, appended last so it can never be swallowed by a
    /// preceding value-taking flag.
    input: Option<OsString>,
}

impl Default for Mafft {
    fn default() -> Self {
        Self::new()
    }
}

impl Mafft {
    /// Start a new invocation. `argv[0]` is `"mafft-rs"`.
    pub fn new() -> Self {
        Self { argv: vec![OsString::from("mafft-rs")], input: None }
    }

    // --- escape hatches -------------------------------------------------

    /// Append a raw argv entry (e.g. `.arg("--legacygappenalty")`).
    pub fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.argv.push(arg.into());
        self
    }

    /// Append a boolean flag by name, without the leading `--`
    /// (e.g. `.flag("reorder")` → `--reorder`).
    pub fn flag(self, name: &str) -> Self {
        self.arg(format!("--{name}"))
    }

    /// Append a flag that takes a value, by name without the leading `--`
    /// (e.g. `.option("maxiterate", "1000")` → `--maxiterate 1000`).
    pub fn option(self, name: &str, value: impl Into<OsString>) -> Self {
        self.arg(format!("--{name}")).arg(value)
    }

    // --- strategy -------------------------------------------------------

    /// `--auto`: pick the strategy from sequence count and length.
    pub fn auto(self) -> Self {
        self.flag("auto")
    }

    /// `--localpair` (L-INS-i).
    pub fn localpair(self) -> Self {
        self.flag("localpair")
    }

    /// `--globalpair` (G-INS-i).
    pub fn globalpair(self) -> Self {
        self.flag("globalpair")
    }

    /// `--genafpair` (E-INS-i).
    pub fn genafpair(self) -> Self {
        self.flag("genafpair")
    }

    /// `--nofft`.
    pub fn nofft(self) -> Self {
        self.flag("nofft")
    }

    /// `--maxiterate N`.
    pub fn maxiterate(self, n: usize) -> Self {
        self.option("maxiterate", n.to_string())
    }

    /// `--retree N`.
    pub fn retree(self, n: usize) -> Self {
        self.option("retree", n.to_string())
    }

    // --- sequence handling ----------------------------------------------

    /// `--adjustdirection`: k-mer strand detection before alignment.
    pub fn adjust_direction(self) -> Self {
        self.flag("adjustdirection")
    }

    /// `--adjustdirectionaccurately`: DP-based strand detection.
    pub fn adjust_direction_accurately(self) -> Self {
        self.flag("adjustdirectionaccurately")
    }

    /// `--nuc`: force nucleotide, overriding auto-detection.
    pub fn nuc(self) -> Self {
        self.flag("nuc")
    }

    /// `--amino`: force protein, overriding auto-detection.
    pub fn amino(self) -> Self {
        self.flag("amino")
    }

    /// `--anysymbol`.
    pub fn anysymbol(self) -> Self {
        self.flag("anysymbol")
    }

    // --- output / runtime -----------------------------------------------

    /// `--thread N` (0 = all cores).
    pub fn thread(self, n: usize) -> Self {
        self.option("thread", n.to_string())
    }

    /// `--quiet`: suppress progress messages on stderr.
    pub fn quiet(self) -> Self {
        self.flag("quiet")
    }

    /// `--reorder`: output in guide-tree order.
    pub fn reorder(self) -> Self {
        self.flag("reorder")
    }

    /// `--inputorder`: output in input order (the default).
    pub fn inputorder(self) -> Self {
        self.flag("inputorder")
    }

    /// `--format FMT` (`fasta`, `clustal`, `phylip`).
    pub fn format(self, fmt: &str) -> Self {
        self.option("format", fmt)
    }

    /// `--linewidth N` (0 = unlimited).
    pub fn linewidth(self, n: usize) -> Self {
        self.option("linewidth", n.to_string())
    }

    /// `--output FILE`. When set, the alignment goes to that file and the
    /// `out` sink passed to [`Mafft::run`] receives nothing — exactly as on
    /// the command line.
    pub fn output(self, path: impl AsRef<Path>) -> Self {
        self.option("output", path.as_ref().as_os_str().to_os_string())
    }

    /// The positional INPUT file. Without it, input is read from stdin
    /// (again matching the CLI).
    pub fn input(mut self, path: impl AsRef<Path>) -> Self {
        self.input = Some(path.as_ref().as_os_str().to_os_string());
        self
    }

    // --- execution ------------------------------------------------------

    /// The argv this builder will hand to [`crate::run_from`], including
    /// `argv[0]`. Useful for logging and for asserting in tests that the
    /// builder and a hand-written command line agree.
    pub fn to_argv(&self) -> Vec<OsString> {
        let mut argv = self.argv.clone();
        if let Some(input) = &self.input {
            argv.push(input.clone());
        }
        argv
    }

    /// Run the alignment, writing it to `out`.
    pub fn run(&self, out: &mut dyn std::io::Write) -> Result<(), MafftError> {
        run_from(self.to_argv(), out)
    }

    /// Run the alignment and return the output bytes (FASTA by default).
    pub fn run_to_vec(&self) -> Result<Vec<u8>, MafftError> {
        let mut buf = Vec::new();
        self.run(&mut buf)?;
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_produces_expected_argv() {
        let argv = Mafft::new()
            .auto()
            .adjust_direction()
            .thread(1)
            .nuc()
            .input("in.fasta")
            .to_argv();
        let argv: Vec<String> =
            argv.into_iter().map(|s| s.to_string_lossy().into_owned()).collect();
        assert_eq!(
            argv,
            vec![
                "mafft-rs",
                "--auto",
                "--adjustdirection",
                "--thread",
                "1",
                "--nuc",
                "in.fasta",
            ]
        );
    }

    #[test]
    fn input_is_always_last() {
        // `--output` takes a value; the positional must not be mistaken
        // for it, so it is appended after every flag regardless of the
        // order the builder methods were called in.
        let argv = Mafft::new()
            .input("in.fasta")
            .output("out.fasta")
            .quiet()
            .to_argv();
        let last = argv.last().unwrap().to_string_lossy().into_owned();
        assert_eq!(last, "in.fasta");
    }

    #[test]
    fn escape_hatches_append_verbatim() {
        let argv = Mafft::new()
            .flag("leavegappyregion")
            .option("bl", "45")
            .arg("--quiet")
            .to_argv();
        let argv: Vec<String> =
            argv.into_iter().map(|s| s.to_string_lossy().into_owned()).collect();
        assert_eq!(
            argv,
            vec!["mafft-rs", "--leavegappyregion", "--bl", "45", "--quiet"]
        );
    }
}

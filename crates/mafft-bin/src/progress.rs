//! Progress sink for in-process runs.
//!
//! The CLI reports what it is doing on stderr — `mafft-rs v0.1.2`,
//! `8 sequences (nuc), strategy: FFT-NS-2`, `Alignment: 398 columns`, and so
//! on. That is right for a terminal and wrong for a library caller running
//! thousands of alignments from a worker pool, where the same lines land
//! interleaved on the user's terminal and every thread contends for the
//! stderr lock.
//!
//! [`Progress`] lets such a caller redirect or drop those messages.
//! [`crate::run_from`] keeps the old behaviour by delegating to
//! [`StderrProgress`], so nothing changes for the command line.
//!
//! Only *progress* goes through the sink. Failures that abort the run are
//! returned as [`crate::MafftError`], and non-fatal `Warning:` /
//! `Could not …` diagnostics stay on stderr — a silent sink must never be
//! able to swallow the news that something went wrong. `--scoreout`'s
//! `Unweighted sum-of-pairs score = …` stays on stderr for the same reason:
//! it is output the user asked for, not progress.
//!
//! # The sink is not exhaustive
//!
//! It covers every progress message emitted by this crate. Two further
//! progress lines live in `mafft-core` and are deliberately NOT routed,
//! because reaching them would mean moving this trait into `mafft-types`,
//! adding a public field to `MafftEngine` and replacing its derived `Debug`
//! — a cost across two more published crates for two lines that are
//! unreachable on the common paths:
//!
//! * `mafft_core::engine` (`RNA structure: computed BPP for N sequences`) —
//!   only on the Q-INS-i / X-INS-i path, which needs external binaries that
//!   are not shipped. Its `quiet_mode` is a hardcoded `false`, so `--quiet`
//!   does not suppress it either.
//! * `mafft_core::engine` (the `--skipiterate` "Iterative refinment was not
//!   done" banner) — only with `--skipiterate`, and arguably a warning
//!   rather than progress.
//!
//! ```
//! use mafft_rs::{Progress, SilentProgress, StderrProgress};
//!
//! // Discard everything.
//! let sink = SilentProgress;
//! sink.message("ignored");
//!
//! // Or forward into your own logger — any `Fn(&str)` is a sink.
//! let collected = std::sync::Mutex::new(Vec::new());
//! let sink = |m: &str| collected.lock().unwrap().push(m.to_string());
//! sink.message("8 sequences (nuc), strategy: FFT-NS-2");
//! assert_eq!(collected.lock().unwrap().len(), 1);
//!
//! let _ = StderrProgress;
//! ```

/// Destination for the run's progress messages.
///
/// `msg` is a single line without a trailing newline; a sink that writes to
/// a stream is expected to add one (as [`StderrProgress`] does).
///
/// Implementors are shared across threads: `run_from_with_progress` takes
/// `&(dyn Progress + Sync)` so one sink can serve a whole worker pool.
/// `&self` (not `&mut self`) keeps that possible without forcing a lock on
/// callers that do not need one.
pub trait Progress {
    /// Report one progress line.
    fn message(&self, msg: &str);
}

/// Writes each message to stderr, exactly as the CLI always has.
///
/// This is what [`crate::run_from`] and [`crate::run`] use, so the
/// command-line output is byte-identical.
#[derive(Debug, Clone, Copy, Default)]
pub struct StderrProgress;

impl Progress for StderrProgress {
    fn message(&self, msg: &str) {
        eprintln!("{msg}");
    }
}

/// Discards every message.
#[derive(Debug, Clone, Copy, Default)]
pub struct SilentProgress;

impl Progress for SilentProgress {
    fn message(&self, _msg: &str) {}
}

/// Any `Fn(&str)` is a sink, so a closure can be passed directly:
/// `run_from_with_progress(argv, &mut out, &|m: &str| log::info!("{m}"))`.
impl<F: Fn(&str)> Progress for F {
    fn message(&self, msg: &str) {
        self(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn silent_progress_discards() {
        SilentProgress.message("nothing should happen");
    }

    #[test]
    fn closures_are_sinks() {
        let seen = Mutex::new(Vec::new());
        let sink = |m: &str| seen.lock().unwrap().push(m.to_string());
        sink.message("one");
        sink.message("two");
        assert_eq!(*seen.lock().unwrap(), vec!["one".to_string(), "two".to_string()]);
    }

    /// The sink is shared across a worker pool, so `&dyn Progress + Sync`
    /// must be usable from several threads at once.
    #[test]
    fn sink_is_usable_from_many_threads() {
        let seen = Mutex::new(Vec::new());
        let sink = |m: &str| seen.lock().unwrap().push(m.to_string());
        let dynref: &(dyn Progress + Sync) = &sink;
        std::thread::scope(|s| {
            for i in 0..8 {
                s.spawn(move || dynref.message(&format!("from {i}")));
            }
        });
        assert_eq!(seen.lock().unwrap().len(), 8);
    }
}

//! Python bindings for MAFFT-rs multiple sequence alignment.
//!
//! Exposes the core MAFFT alignment engine to Python via PyO3.
//!
//! ```python
//! import pymafft
//!
//! # Align sequences (FFT-NS-2 default)
//! result = pymafft.align(["ACDEFGHIK", "ACDEFHIK", "ACDHIK"])
//! for name, seq in result:
//!     print(f"{name}: {seq}")
//!
//! # Align with specific strategy
//! result = pymafft.align(sequences, strategy="ginsi", maxiterate=1000)
//!
//! # Read FASTA file and align
//! result = pymafft.align_file("sequences.fasta")
//! ```

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mafft_core::{AlignmentMode, MafftEngine};
use mafft_io::{read_fasta, read_fasta_from_reader};
use mafft_types::{Sequence, SequenceSet};

/// A single aligned sequence with name and gapped data.
#[pyclass]
#[derive(Clone)]
struct AlignedSequence {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    sequence: String,
}

#[pymethods]
impl AlignedSequence {
    fn __repr__(&self) -> String {
        format!(
            "AlignedSequence(name='{}', len={})",
            self.name,
            self.sequence.len()
        )
    }

    fn __str__(&self) -> String {
        format!(">{}\n{}", self.name, self.sequence)
    }

    /// Return the sequence without gap characters.
    fn ungapped(&self) -> String {
        self.sequence.chars().filter(|&c| c != '-').collect()
    }

    /// Length of the aligned sequence (including gaps).
    fn __len__(&self) -> usize {
        self.sequence.len()
    }
}

/// Result of a multiple sequence alignment.
#[pyclass]
#[derive(Clone)]
struct AlignmentResult {
    #[pyo3(get)]
    sequences: Vec<AlignedSequence>,
    #[pyo3(get)]
    width: usize,
    #[pyo3(get)]
    score: f64,
}

#[pymethods]
impl AlignmentResult {
    fn __repr__(&self) -> String {
        format!(
            "AlignmentResult(nseq={}, width={}, score={:.1})",
            self.sequences.len(),
            self.width,
            self.score,
        )
    }

    fn __len__(&self) -> usize {
        self.sequences.len()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<AlignmentResultIter>> {
        let iter = AlignmentResultIter {
            inner: slf.sequences.clone(),
            index: 0,
        };
        Py::new(slf.py(), iter)
    }

    /// Get a sequence by index.
    fn __getitem__(&self, idx: usize) -> PyResult<AlignedSequence> {
        self.sequences.get(idx).cloned().ok_or_else(|| {
            PyValueError::new_err(format!(
                "index {} out of range (nseq={})",
                idx,
                self.sequences.len()
            ))
        })
    }

    /// Return the alignment as a FASTA-formatted string.
    fn to_fasta(&self) -> String {
        self.sequences
            .iter()
            .map(|s| format!(">{}\n{}", s.name, s.sequence))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Return the alignment as a list of (name, sequence) tuples.
    fn to_tuples(&self) -> Vec<(String, String)> {
        self.sequences
            .iter()
            .map(|s| (s.name.clone(), s.sequence.clone()))
            .collect()
    }

    /// Number of sequences.
    #[getter]
    fn nseq(&self) -> usize {
        self.sequences.len()
    }
}

#[pyclass]
struct AlignmentResultIter {
    inner: Vec<AlignedSequence>,
    index: usize,
}

#[pymethods]
impl AlignmentResultIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<AlignedSequence> {
        if self.index < self.inner.len() {
            let item = self.inner[self.index].clone();
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
}

/// Parse the strategy string into an AlignmentMode.
fn parse_strategy(strategy: &str, maxiterate: usize) -> PyResult<AlignmentMode> {
    match strategy.to_lowercase().as_str() {
        "fftns2" | "default" | "" => Ok(AlignmentMode::FftNs2),
        "fftnsi" => Ok(AlignmentMode::FftNsi {
            iterations: if maxiterate > 0 { maxiterate } else { 100 },
        }),
        "ginsi" | "globalpair" => Ok(AlignmentMode::GInsi {
            iterations: if maxiterate > 0 { maxiterate } else { 1000 },
        }),
        "linsi" | "localpair" => Ok(AlignmentMode::LInsi {
            iterations: if maxiterate > 0 { maxiterate } else { 1000 },
        }),
        "einsi" | "genafpair" => Ok(AlignmentMode::EInsi {
            iterations: if maxiterate > 0 { maxiterate } else { 1000 },
        }),
        other => Err(PyValueError::new_err(format!(
            "unknown strategy '{}'. Valid: fftns2, fftnsi, ginsi, linsi, einsi",
            other
        ))),
    }
}

/// Build a SequenceSet from Python input.
fn build_input(sequences: Vec<(String, String)>) -> PyResult<SequenceSet> {
    if sequences.is_empty() {
        return Err(PyValueError::new_err("no sequences provided"));
    }

    let seqs: Vec<Sequence> = sequences
        .into_iter()
        .map(|(name, data)| Sequence {
            name,
            data: data.into_bytes(),
        })
        .collect();

    let seq_type =
        mafft_io::detect_seq_type(&seqs.iter().map(|s| s.data.clone()).collect::<Vec<_>>());

    Ok(SequenceSet {
        sequences: seqs,
        seq_type,
    })
}

fn run_alignment(input: SequenceSet, mode: AlignmentMode) -> AlignmentResult {
    let engine = MafftEngine::new(mode);
    let msa = engine.align(&input);

    let sequences: Vec<AlignedSequence> = msa
        .sequences
        .iter()
        .zip(msa.names.iter())
        .map(|(seq, name)| AlignedSequence {
            name: name.clone(),
            sequence: String::from_utf8_lossy(seq).to_string(),
        })
        .collect();

    AlignmentResult {
        width: msa.width(),
        score: msa.score,
        sequences,
    }
}

// ---------------------------------------------------------------------------
// Module functions
// ---------------------------------------------------------------------------

/// Try to convert a Python object with `.id` and `.seq` attributes (duck-
/// types `Bio.SeqRecord.SeqRecord` and similar) into a `(name, seq)` pair.
/// Returns `None` if either attribute is missing.
fn try_record_to_pair(obj: &Bound<'_, PyAny>) -> Option<(String, String)> {
    let id_obj = obj.getattr("id").ok()?;
    let seq_obj = obj.getattr("seq").ok()?;
    let id: String = id_obj.extract().ok()?;
    // `seq` may be a `Bio.Seq.Seq` (custom type) — `str()` always works.
    let seq_str = seq_obj.str().ok()?;
    let seq: String = seq_str.extract().ok()?;
    Some((id, seq))
}

/// Align sequences.
///
/// Args:
///     sequences: List of sequences. Accepted shapes:
///         - List of strings (auto-named "seq_1", "seq_2", ...)
///         - List of (name, sequence) tuples
///         - List of objects with `.id` and `.seq` attributes
///           (e.g. Biopython `SeqRecord`).
///     strategy: Alignment strategy. One of:
///         "fftns2" (default), "fftnsi", "ginsi", "linsi", "einsi"
///     maxiterate: Maximum refinement iterations (0 = default for strategy).
///
/// Returns:
///     AlignmentResult with aligned sequences.
#[pyfunction]
#[pyo3(signature = (sequences, strategy="fftns2", maxiterate=0))]
fn align(
    sequences: &Bound<'_, PyAny>,
    strategy: &str,
    maxiterate: usize,
) -> PyResult<AlignmentResult> {
    let mode = parse_strategy(strategy, maxiterate)?;

    // Accept (in order of try):
    //   1. list of strings
    //   2. list of (name, seq) tuples
    //   3. iterable of SeqRecord-like objects (`.id`, `.seq` attrs)
    let input = if let Ok(str_list) = sequences.extract::<Vec<String>>() {
        let pairs: Vec<(String, String)> = str_list
            .into_iter()
            .enumerate()
            .map(|(i, s)| (format!("seq_{}", i + 1), s))
            .collect();
        build_input(pairs)?
    } else if let Ok(tuple_list) = sequences.extract::<Vec<(String, String)>>() {
        build_input(tuple_list)?
    } else if let Ok(iter) = sequences.try_iter() {
        // Duck-type: try to read each item's `.id` and `.seq` attrs.
        let mut pairs: Vec<(String, String)> = Vec::new();
        for (i, item) in iter.enumerate() {
            let item = item?;
            match try_record_to_pair(&item) {
                Some(p) => pairs.push(p),
                None => {
                    return Err(PyValueError::new_err(format!(
                        "sequences[{}] is not a string, (name, seq) tuple, or \
                     object with .id and .seq attributes",
                        i
                    )));
                }
            }
        }
        build_input(pairs)?
    } else {
        return Err(PyValueError::new_err(
            "sequences must be a list of strings, (name, sequence) tuples, \
             or SeqRecord-like objects",
        ));
    };

    Ok(run_alignment(input, mode))
}

/// Align sequences from a FASTA file.
///
/// Args:
///     path: Path to FASTA file.
///     strategy: Alignment strategy (default: "fftns2").
///     maxiterate: Maximum refinement iterations.
///
/// Returns:
///     AlignmentResult with aligned sequences.
#[pyfunction]
#[pyo3(signature = (path, strategy="fftns2", maxiterate=0))]
fn align_file(path: &str, strategy: &str, maxiterate: usize) -> PyResult<AlignmentResult> {
    let mode = parse_strategy(strategy, maxiterate)?;

    let input = read_fasta(path).map_err(|e| {
        PyValueError::new_err(format!("error reading FASTA file '{}': {}", path, e))
    })?;

    Ok(run_alignment(input, mode))
}

/// Align sequences from a FASTA-formatted string.
///
/// Args:
///     fasta_string: FASTA-formatted string.
///     strategy: Alignment strategy (default: "fftns2").
///     maxiterate: Maximum refinement iterations.
///
/// Returns:
///     AlignmentResult with aligned sequences.
#[pyfunction]
#[pyo3(signature = (fasta_string, strategy="fftns2", maxiterate=0))]
fn align_fasta_string(
    fasta_string: &str,
    strategy: &str,
    maxiterate: usize,
) -> PyResult<AlignmentResult> {
    let mode = parse_strategy(strategy, maxiterate)?;

    let reader = std::io::Cursor::new(fasta_string.as_bytes());
    let input = read_fasta_from_reader(std::io::BufReader::new(reader))
        .map_err(|e| PyValueError::new_err(format!("error parsing FASTA string: {}", e)))?;

    Ok(run_alignment(input, mode))
}

/// pymafft: Python bindings for MAFFT-rs multiple sequence alignment.
#[pymodule]
fn pymafft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AlignedSequence>()?;
    m.add_class::<AlignmentResult>()?;
    m.add_function(wrap_pyfunction!(align, m)?)?;
    m.add_function(wrap_pyfunction!(align_file, m)?)?;
    m.add_function(wrap_pyfunction!(align_fasta_string, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

use mafft_types::{Sequence, SequenceSet};

use crate::detect::detect_seq_type;
use crate::error::IoError;

/// Default line width for FASTA output (matches C macro `C = 60`).
const DEFAULT_LINE_WIDTH: usize = 60;

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Read a FASTA file from a path into a `SequenceSet`.
///
/// Performs the same normalization as the C code:
/// - Strips non-alphabetic characters (except '-', '.') from sequences.
/// - Converts '*' to '-'.
/// - Auto-detects DNA vs protein via ATGC frequency.
/// - Canonicalises residue case per the detected type — lowercase for
///   DNA/RNA, uppercase for protein (see [`apply_case_convention`]).
///
/// Uses a lenient parser that handles MAFFT's non-standard headers
/// (e.g. `>     1== name ...` with leading spaces).
pub fn read_fasta(path: impl AsRef<Path>) -> Result<SequenceSet, IoError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    read_fasta_from_reader(reader)
}

/// Read a FASTA file preserving case and non-standard residues — used
/// by `--anysymbol`/`--preservecase`. Strips only whitespace and
/// digits (matching C MAFFT's `readData_pointer_casepreserve`); any
/// other character is kept verbatim so the post-alignment restore
/// pass can put the originals back.
pub fn read_fasta_casepreserve(path: impl AsRef<Path>) -> Result<SequenceSet, IoError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    read_fasta_from_reader_casepreserve(reader)
}

/// Like `read_fasta_from_reader` but preserves case and non-standard
/// residues (see `read_fasta_casepreserve`).
pub fn read_fasta_from_reader_casepreserve<R: BufRead>(reader: R) -> Result<SequenceSet, IoError> {
    let mut sequences = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_seq = Vec::new();

    for line_result in reader.lines() {
        let line = line_result?;
        if let Some(header) = line.strip_prefix('>') {
            if let Some(name) = current_name.take() {
                sequences.push(Sequence {
                    name,
                    data: normalize_sequence_casepreserve(&current_seq),
                });
                current_seq.clear();
            }
            current_name = Some(header.to_string());
        } else if current_name.is_some() {
            current_seq.extend_from_slice(line.as_bytes());
        }
    }
    if let Some(name) = current_name.take() {
        sequences.push(Sequence {
            name,
            data: normalize_sequence_casepreserve(&current_seq),
        });
    }
    if sequences.is_empty() {
        return Err(IoError::EmptyInput);
    }
    let seq_type = detect_seq_type(
        &sequences.iter().map(|s| s.data.clone()).collect::<Vec<_>>(),
    );
    Ok(SequenceSet { sequences, seq_type })
}

/// Case-preserving sequence normaliser — strips only whitespace and
/// ASCII digits; all other characters (including `*`, `@`, lowercase,
/// IUPAC) are kept so `--anysymbol`/`--preservecase` can replace then
/// restore them. Mirrors C `readData_pointer_casepreserve` reading
/// rules.
fn normalize_sequence_casepreserve(raw: &[u8]) -> Vec<u8> {
    raw.iter()
        .copied()
        .filter(|&c| !c.is_ascii_whitespace() && !c.is_ascii_digit())
        .collect()
}

/// Read FASTA from any buffered reader.
///
/// Handles non-standard headers with leading whitespace that strict parsers
/// (like `noodles-fasta`) reject. The full text after '>' is preserved as
/// the sequence name, matching MAFFT's C behavior.
pub fn read_fasta_from_reader<R: BufRead>(reader: R) -> Result<SequenceSet, IoError> {
    let mut sequences = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_seq = Vec::new();

    for line_result in reader.lines() {
        let line = line_result?;

        if let Some(header) = line.strip_prefix('>') {
            // Flush previous sequence
            if let Some(name) = current_name.take() {
                sequences.push(Sequence {
                    name,
                    data: normalize_sequence(&current_seq),
                });
                current_seq.clear();
            }
            current_name = Some(header.to_string());
        } else if current_name.is_some() {
            // Sequence data line
            current_seq.extend_from_slice(line.as_bytes());
        }
        // Lines before the first '>' are ignored
    }

    // Flush last sequence
    if let Some(name) = current_name.take() {
        sequences.push(Sequence {
            name,
            data: normalize_sequence(&current_seq),
        });
    }

    if sequences.is_empty() {
        return Err(IoError::EmptyInput);
    }

    let seq_type = detect_seq_type(
        &sequences.iter().map(|s| s.data.clone()).collect::<Vec<_>>(),
    );

    let mut set = SequenceSet { sequences, seq_type };
    apply_case_convention(&mut set);
    Ok(set)
}

/// Apply C MAFFT's residue-case convention to an already-parsed set:
/// lowercase for DNA/RNA, uppercase for everything else.
///
/// C canonicalises case as it reads — `io.c:1462-1467`
/// (`load1SeqWithoutName_realloc`) calls `onlyAlpha_lower` when
/// `dorp == 'd'` and `onlyAlpha_upper` otherwise, and `readData_pointer`
/// repeats the nucleotide pass with `seqLower` (`io.c:1755`). The
/// `upperCase != -1` guard there is only reachable from the legacy
/// non-FASTA `FRead` header parser (`io.c:1174-1184`), so for FASTA input
/// it is always true. Net effect: C MAFFT's default output is lowercase
/// for DNA/RNA and uppercase for protein, whatever case the input used.
///
/// The fold is idempotent, so it is safe to re-apply after `--nuc` /
/// `--amino` override the detected type — which is what C does, since
/// `$seqtype` fixes `dorp` before any sequence is read
/// (`scripts/mafft:547-550`).
///
/// Deliberately NOT applied by [`read_fasta_casepreserve`]: on the
/// `--anysymbol` / `--preservecase` path C reads with
/// `readData_pointer_casepreserve` and restores the original characters
/// after alignment (`replaceu` + `restoreu`), so the input case survives.
pub fn apply_case_convention(set: &mut SequenceSet) {
    let nucleotide = set.seq_type.is_nucleotide();
    for seq in set.sequences.iter_mut() {
        for ch in seq.data.iter_mut() {
            *ch = if nucleotide {
                ch.to_ascii_lowercase()
            } else {
                ch.to_ascii_uppercase()
            };
        }
    }
}

/// Normalize a raw sequence: keep only alpha + gap chars, convert '*' to '-'.
///
/// Mirrors the character-filtering half of C's `onlyAlpha_lower()` /
/// `onlyAlpha_upper()` plus `kake2hiku()` (`io.c:1425-1470`). Case is left
/// alone here because C picks the case fold from `dorp`, which is only
/// known once the sequence type has been detected (or forced by
/// `--nuc` / `--amino`); [`apply_case_convention`] applies it afterwards.
fn normalize_sequence(raw: &[u8]) -> Vec<u8> {
    raw.iter()
        .filter_map(|&ch| {
            if ch.is_ascii_alphabetic() {
                Some(ch)
            } else if ch == b'-' || ch == b'.' {
                Some(ch)
            } else if ch == b'*' {
                Some(b'-') // kake2hiku: * → -
            } else {
                None // strip digits, whitespace, etc.
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Write a `SequenceSet` as FASTA to a file path.
pub fn write_fasta(seqs: &SequenceSet, path: impl AsRef<Path>) -> Result<(), IoError> {
    let file = std::fs::File::create(path)?;
    let writer = io::BufWriter::new(file);
    write_fasta_to_writer(seqs, writer)
}

/// Write a `SequenceSet` as FASTA to any writer.
///
/// Uses 60-character line width by default (matching the C output).
pub fn write_fasta_to_writer<W: Write>(
    seqs: &SequenceSet,
    mut writer: W,
) -> Result<(), IoError> {
    write_fasta_to_writer_with_width(seqs, &mut writer, DEFAULT_LINE_WIDTH)
}

/// Write FASTA with a custom line width. Pass `0` for unlimited (single line).
pub fn write_fasta_to_writer_with_width<W: Write>(
    seqs: &SequenceSet,
    writer: &mut W,
    line_width: usize,
) -> Result<(), IoError> {
    for seq in &seqs.sequences {
        writeln!(writer, ">{}", seq.name)?;

        if line_width == 0 {
            writer.write_all(&seq.data)?;
            writeln!(writer)?;
        } else {
            for chunk in seq.data.chunks(line_width) {
                writer.write_all(chunk)?;
                writeln!(writer)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_and_converts() {
        let raw = b"MNG*T.E-G 123\n";
        let result = normalize_sequence(raw);
        assert_eq!(result, b"MNG-T.E-G");
    }

    #[test]
    fn roundtrip_fasta() {
        let original = SequenceSet {
            sequences: vec![
                Sequence {
                    name: "seq1 description".into(),
                    data: b"ACGTACGTACGT".to_vec(),
                },
                Sequence {
                    name: "seq2".into(),
                    data: b"MNGTEGDNFYVP".to_vec(),
                },
            ],
            seq_type: mafft_types::SeqType::Protein,
        };

        let mut buf = Vec::new();
        write_fasta_to_writer(&original, &mut buf).unwrap();

        let parsed = read_fasta_from_reader(io::Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.sequences.len(), 2);
        assert_eq!(parsed.sequences[0].name, "seq1 description");
        assert_eq!(parsed.sequences[0].data, b"ACGTACGTACGT");
        assert_eq!(parsed.sequences[1].name, "seq2");
        assert_eq!(parsed.sequences[1].data, b"MNGTEGDNFYVP");
    }

    #[test]
    fn handles_mafft_style_headers() {
        let input = b">     1== M63632 rhodopsin\nMNGTEGDNFYVP\n>     2== U22180 rat opsin\nACGT\n";
        let seqs = read_fasta_from_reader(io::Cursor::new(&input[..])).unwrap();
        assert_eq!(seqs.nseq(), 2);
        assert!(seqs.sequences[0].name.contains("M63632"));
        assert!(seqs.sequences[1].name.contains("U22180"));
    }

    // --- C MAFFT residue-case convention (io.c:1462-1467, io.c:1755) ---

    #[test]
    fn nucleotide_input_is_lowercased_whatever_the_input_case() {
        let input = b">a\nATGGCtagcTTGGACCATTGCAGG\n>b\nATGGCTAGCTTGGACCATTGCAGG\n";
        let seqs = read_fasta_from_reader(io::Cursor::new(&input[..])).unwrap();
        assert_eq!(seqs.seq_type, mafft_types::SeqType::Dna);
        assert_eq!(seqs.sequences[0].data, b"atggctagcttggaccattgcagg".to_vec());
        assert_eq!(seqs.sequences[1].data, b"atggctagcttggaccattgcagg".to_vec());
    }

    #[test]
    fn protein_input_is_uppercased_whatever_the_input_case() {
        let input = b">a\nMNGTegdnFYVPFSNKTGLARSPYEY\n>b\nMNGTEGDNFYVPFSNKTGLARSPYEY\n";
        let seqs = read_fasta_from_reader(io::Cursor::new(&input[..])).unwrap();
        assert_eq!(seqs.seq_type, mafft_types::SeqType::Protein);
        assert_eq!(seqs.sequences[0].data, b"MNGTEGDNFYVPFSNKTGLARSPYEY".to_vec());
    }

    #[test]
    fn casepreserve_reader_keeps_the_input_case() {
        // `--anysymbol` / `--preservecase` restore the originals after
        // alignment, so this reader must not fold anything.
        let input = b">a\nATGGCtagcTTGGACCATTGCAGG\n";
        let seqs = read_fasta_from_reader_casepreserve(io::Cursor::new(&input[..])).unwrap();
        assert_eq!(seqs.sequences[0].data, b"ATGGCtagcTTGGACCATTGCAGG".to_vec());
    }

    #[test]
    fn apply_case_convention_is_idempotent_and_follows_seq_type() {
        // Safe to re-apply after `--nuc` / `--amino` override the type.
        let mut set = SequenceSet {
            sequences: vec![Sequence { name: "a".into(), data: b"AtGc".to_vec() }],
            seq_type: mafft_types::SeqType::Dna,
        };
        apply_case_convention(&mut set);
        assert_eq!(set.sequences[0].data, b"atgc".to_vec());
        apply_case_convention(&mut set);
        assert_eq!(set.sequences[0].data, b"atgc".to_vec());

        set.seq_type = mafft_types::SeqType::Protein;
        apply_case_convention(&mut set);
        assert_eq!(set.sequences[0].data, b"ATGC".to_vec());
    }

    #[test]
    fn case_fold_does_not_disturb_type_detection() {
        // Detection runs on the pre-fold residues and is case-insensitive,
        // so a lowercase and an uppercase copy detect the same type.
        let upper = read_fasta_from_reader(io::Cursor::new(&b">a\nACGTACGTACGTACGT\n"[..])).unwrap();
        let lower = read_fasta_from_reader(io::Cursor::new(&b">a\nacgtacgtacgtacgt\n"[..])).unwrap();
        assert_eq!(upper.seq_type, lower.seq_type);
        assert_eq!(upper.sequences[0].data, lower.sequences[0].data);
    }
}

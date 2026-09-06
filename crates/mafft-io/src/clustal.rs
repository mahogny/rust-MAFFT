use std::io::Write;

use mafft_types::{SeqType, SequenceSet};

use crate::error::IoError;

/// Default block width for Clustal output (matches C code: 60).
const BLOCK_WIDTH: usize = 60;

/// Default name field width.
const DEFAULT_NAME_LEN: usize = 15;

/// Write an alignment in Clustal format.
///
/// Matches the output of the C `clustalout_pointer()` function with no
/// `comment` (header reads `CLUSTAL format alignment by MAFFT
/// (v7.526)`) and no conservation marks. For full-fidelity output use
/// `write_clustal_full`.
pub fn write_clustal<W: Write>(
    seqs: &SequenceSet,
    writer: &mut W,
    order: Option<&[usize]>,
    name_len: Option<usize>,
    conservation: Option<&str>,
) -> Result<(), IoError> {
    write_clustal_full(seqs, writer, order, name_len, conservation, None)
}

/// Write an alignment in Clustal format with an optional header
/// comment (e.g. the alignment-mode label `FFT-NS-2`).
///
/// Matches the output of the C `clustalout_pointer()` function
/// (io.c:5431) including the per-block conservation-mark line when
/// `conservation` is provided. Pass `comment = Some("FFT-NS-2")` to
/// produce C's `CLUSTAL format alignment by MAFFT FFT-NS-2 (v7.526)`
/// header.
///
/// - `order`: optional sequence ordering (indices into
///   `seqs.sequences`). If `None`, sequences are written in their
///   natural order.
/// - `name_len`: name field width. Pass `None` for the default (15).
/// - `conservation`: optional conservation mark string (`*` / `:` /
///   `.` / space, one byte per alignment column). Use
///   [`compute_clustal_marks`] to build it.
/// - `comment`: alignment-mode label embedded in the header line.
pub fn write_clustal_full<W: Write>(
    seqs: &SequenceSet,
    writer: &mut W,
    order: Option<&[usize]>,
    name_len: Option<usize>,
    conservation: Option<&str>,
    comment: Option<&str>,
) -> Result<(), IoError> {
    let nseq = seqs.nseq();
    if nseq == 0 {
        return Err(IoError::EmptyInput);
    }

    let name_len = name_len.unwrap_or(DEFAULT_NAME_LEN);
    let max_len = seqs.max_len();

    // Header. C `clustalout_pointer` (io.c:5437-5439):
    //   if (comment == NULL)
    //     fprintf(fp, "CLUSTAL format alignment by MAFFT (v%s)\n\n", VERSION);
    //   else
    //     fprintf(fp, "CLUSTAL format alignment by MAFFT %s (v%s)\n\n", comment, VERSION);
    match comment {
        Some(c) => writeln!(writer, "CLUSTAL format alignment by MAFFT {} (v7.526)", c)?,
        None => writeln!(writer, "CLUSTAL format alignment by MAFFT (v7.526)")?,
    }
    writeln!(writer)?;

    let default_order: Vec<usize> = (0..nseq).collect();
    let order = order.unwrap_or(&default_order);

    let mut pos = 0;
    while pos < max_len {
        writeln!(writer)?;

        let end = (pos + BLOCK_WIDTH).min(max_len);

        for &idx in order {
            let seq = &seqs.sequences[idx];
            let first_word = first_word_of(&seq.name);
            write_name_field(writer, first_word, name_len)?;
            writer.write_all(b" ")?;

            let chunk_end = end.min(seq.data.len());
            if pos < chunk_end {
                writer.write_all(&seq.data[pos..chunk_end])?;
            }
            writeln!(writer)?;
        }

        if let Some(marks) = conservation {
            let mark_bytes = marks.as_bytes();
            write_name_field(writer, "", name_len)?;
            writer.write_all(b" ")?;
            let chunk_end = end.min(mark_bytes.len());
            if pos < chunk_end {
                writer.write_all(&mark_bytes[pos..chunk_end])?;
            }
            writeln!(writer)?;
        }

        pos += BLOCK_WIDTH;
    }

    Ok(())
}

/// Conservation groups for protein CLUSTAL marks, from C's
/// `setmark_clustal` (f2cl.c:43-65). A `*` is full identity. A `:`
/// means every column residue is in at least one "strong" group.
/// A `.` means every residue is in at least one "weaker" group.
const PROTEIN_STRONG: &[&[u8]] = &[
    b"STA", b"NEQK", b"NHQK", b"NDEQ", b"QHRK", b"MILV", b"MILF", b"HY", b"FYW",
];
const PROTEIN_WEAKER: &[&[u8]] = &[
    b"CSA", b"ATV", b"SAG", b"STNK", b"STPA", b"SGND", b"SNDEQK", b"NDEQHK", b"NEQHRK", b"FVLIM",
    b"HFY",
];

/// Conservation groups for DNA/RNA CLUSTAL marks (f2cl.c:33-39).
const DNA_STRONG: &[&[u8]] = &[b"TU"];
const DNA_WEAKER: &[&[u8]] = &[b"AG", b"CT", b"CU"];

/// Set of canonical residues per type (f2cl.c:39/65 `nalpha`).
/// Columns containing a non-canonical residue at position 0 leave the
/// mark as space.
const PROTEIN_ALPHA: &[u8] = b"ARNDCQEGHILKMFPSTWYV";
const DNA_ALPHA: &[u8] = b"ATGCUNDHBVRYKMSW"; // 10-letter window in C; superset is fine for membership

/// Compute CLUSTAL-format conservation marks per column.
///
/// Direct port of C's `setmark_clustal` (f2cl.c:22-134). For each
/// alignment column:
/// - if any sequence has `-` or ` `, mark is space;
/// - else if all uppercase residues equal `seq[0][i]`, mark is `*`;
/// - else if all column residues are in one of the per-type "strong"
///   groups, mark is `:`;
/// - else if all are in one of the "weaker" groups, mark is `.`;
/// - else mark is space.
///
/// Returns a `String` of length `seqs.max_len()` (one mark byte per
/// column) suitable for the `conservation` parameter of
/// [`write_clustal_full`].
pub fn compute_clustal_marks(seqs: &SequenceSet) -> String {
    let nlen = seqs.max_len();
    let nseq = seqs.nseq();
    let mut marks = vec![b' '; nlen];

    let is_dna = matches!(seqs.seq_type, SeqType::Dna | SeqType::Rna);
    let (strong, weaker, alpha): (&[&[u8]], &[&[u8]], &[u8]) = if is_dna {
        (DNA_STRONG, DNA_WEAKER, DNA_ALPHA)
    } else {
        (PROTEIN_STRONG, PROTEIN_WEAKER, PROTEIN_ALPHA)
    };

    for i in 0..nlen {
        // Skip column if any seq has gap or space at position i.
        let mut any_gap = false;
        for j in 0..nseq {
            let s = &seqs.sequences[j].data;
            let c = if i < s.len() { s[i] } else { b'-' };
            if c == b'-' || c == b' ' {
                any_gap = true;
                break;
            }
        }
        if any_gap {
            continue;
        }

        // First-letter check (uppercase). Out-of-alpha → leave as space.
        let first = ascii_upper(seqs.sequences[0].data[i]);
        if !alpha.contains(&first) {
            continue;
        }

        // All same → '*'
        let all_same = (0..nseq).all(|j| ascii_upper(seqs.sequences[j].data[i]) == first);
        if all_same {
            marks[i] = b'*';
            continue;
        }

        // Strong group → ':'
        let in_strong = strong
            .iter()
            .any(|grp| (0..nseq).all(|j| grp.contains(&ascii_upper(seqs.sequences[j].data[i]))));
        if in_strong {
            marks[i] = b':';
            continue;
        }

        // Weaker group → '.'
        let in_weaker = weaker
            .iter()
            .any(|grp| (0..nseq).all(|j| grp.contains(&ascii_upper(seqs.sequences[j].data[i]))));
        if in_weaker {
            marks[i] = b'.';
        }
    }

    String::from_utf8(marks).expect("marks are ASCII")
}

fn ascii_upper(b: u8) -> u8 {
    if b.is_ascii_lowercase() { b - 32 } else { b }
}

/// Write a name field with C `%-*.*s` semantics: left-align, pad to
/// `width` chars with spaces, AND truncate if longer than `width`.
/// Rust's `{:<width$}` only pads — long strings flow past the field,
/// breaking column alignment vs C MAFFT.
fn write_name_field<W: Write>(writer: &mut W, name: &str, width: usize) -> Result<(), IoError> {
    let bytes = name.as_bytes();
    let len = bytes.len().min(width);
    writer.write_all(&bytes[..len])?;
    for _ in len..width {
        writer.write_all(b" ")?;
    }
    Ok(())
}

/// Extract the first whitespace-delimited word from a name string.
fn first_word_of(name: &str) -> &str {
    name.split_whitespace().next().unwrap_or(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mafft_types::{SeqType, Sequence};

    #[test]
    fn clustal_output_format() {
        let seqs = SequenceSet {
            sequences: vec![
                Sequence {
                    name: "seq1 some description".into(),
                    data: b"ACGT-ACGT".to_vec(),
                },
                Sequence {
                    name: "seq2".into(),
                    data: b"ACGTAAC-T".to_vec(),
                },
            ],
            seq_type: SeqType::Dna,
        };

        let mut buf = Vec::new();
        write_clustal(&seqs, &mut buf, None, None, None).unwrap();
        let output = String::from_utf8(buf).unwrap();

        assert!(output.starts_with("CLUSTAL format alignment by MAFFT"));
        assert!(output.contains("seq1"));
        assert!(output.contains("seq2"));
        assert!(output.contains("ACGT-ACGT"));
    }

    /// Long sequence names must be truncated to `namelen` (C's `%-*.*s`
    /// semantics) — earlier `{:<width$}` would let them overflow the
    /// column.
    #[test]
    fn name_field_truncates_overlong_names() {
        let seqs = SequenceSet {
            sequences: vec![
                Sequence {
                    name: "this_name_is_way_longer_than_15_chars".into(),
                    data: b"ACGT-ACGT".to_vec(),
                },
                Sequence {
                    name: "s2".into(),
                    data: b"ACGTAAC-T".to_vec(),
                },
            ],
            seq_type: SeqType::Dna,
        };
        let mut buf = Vec::new();
        write_clustal(&seqs, &mut buf, None, Some(15), None).unwrap();
        let out = String::from_utf8(buf).unwrap();
        // The truncated name should appear, but NOT the full overlong form.
        assert!(out.contains("this_name_is_wa "));
        assert!(!out.contains("this_name_is_way_longer_than_15_chars"));
    }

    /// CLUSTAL header carries the alignment-mode label when
    /// `comment = Some(...)` is passed to `write_clustal_full`.
    #[test]
    fn header_includes_strategy_comment() {
        let seqs = SequenceSet {
            sequences: vec![
                Sequence {
                    name: "s1".into(),
                    data: b"AC".to_vec(),
                },
                Sequence {
                    name: "s2".into(),
                    data: b"AC".to_vec(),
                },
            ],
            seq_type: SeqType::Protein,
        };
        let mut buf = Vec::new();
        write_clustal_full(&seqs, &mut buf, None, None, None, Some("FFT-NS-2")).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.starts_with("CLUSTAL format alignment by MAFFT FFT-NS-2 (v7.526)\n"));
    }

    /// Conservation marks: identical residues → `*`; otherwise space,
    /// `:`, or `.` per group membership. Direct port of C
    /// `setmark_clustal`.
    #[test]
    fn marks_star_for_full_identity_column() {
        let seqs = SequenceSet {
            sequences: vec![
                Sequence {
                    name: "a".into(),
                    data: b"MKT".to_vec(),
                },
                Sequence {
                    name: "b".into(),
                    data: b"MKT".to_vec(),
                },
                Sequence {
                    name: "c".into(),
                    data: b"MKT".to_vec(),
                },
            ],
            seq_type: SeqType::Protein,
        };
        let marks = compute_clustal_marks(&seqs);
        assert_eq!(marks, "***");
    }

    #[test]
    fn marks_space_for_gap_column() {
        let seqs = SequenceSet {
            sequences: vec![
                Sequence {
                    name: "a".into(),
                    data: b"M-T".to_vec(),
                },
                Sequence {
                    name: "b".into(),
                    data: b"MKT".to_vec(),
                },
            ],
            seq_type: SeqType::Protein,
        };
        let marks = compute_clustal_marks(&seqs);
        assert_eq!(marks, "* *"); // pos 0 identical, pos 1 has gap, pos 2 identical
    }

    /// `:` for residues that all fit a "strong" group. The protein
    /// strong group `MILV` covers M/I/L/V; an MILM column should mark.
    #[test]
    fn marks_strong_group_protein() {
        let seqs = SequenceSet {
            sequences: vec![
                Sequence {
                    name: "a".into(),
                    data: b"M".to_vec(),
                },
                Sequence {
                    name: "b".into(),
                    data: b"I".to_vec(),
                },
                Sequence {
                    name: "c".into(),
                    data: b"L".to_vec(),
                },
                Sequence {
                    name: "d".into(),
                    data: b"V".to_vec(),
                },
            ],
            seq_type: SeqType::Protein,
        };
        let marks = compute_clustal_marks(&seqs);
        assert_eq!(marks, ":");
    }

    /// `.` for residues that don't fit any strong group but all fit a
    /// weaker group. Protein weaker group `CSA` covers C/S/A.
    #[test]
    fn marks_weak_group_protein() {
        let seqs = SequenceSet {
            sequences: vec![
                Sequence {
                    name: "a".into(),
                    data: b"C".to_vec(),
                },
                Sequence {
                    name: "b".into(),
                    data: b"S".to_vec(),
                },
                Sequence {
                    name: "c".into(),
                    data: b"A".to_vec(),
                },
            ],
            seq_type: SeqType::Protein,
        };
        let marks = compute_clustal_marks(&seqs);
        assert_eq!(marks, ".");
    }
}

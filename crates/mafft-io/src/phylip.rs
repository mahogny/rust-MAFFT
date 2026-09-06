use std::io::Write;

use mafft_types::SequenceSet;

use crate::error::IoError;

/// Default name field width for PHYLIP (matches C code: 10).
const DEFAULT_NAME_LEN: usize = 10;

/// Residues per group in PHYLIP output.
const GROUP_WIDTH: usize = 10;

/// Number of groups per line.
const GROUPS_PER_LINE: usize = 5;

/// Total residues per line: 5 groups * 10 = 50.
const LINE_WIDTH: usize = GROUPS_PER_LINE * GROUP_WIDTH;

/// Write an alignment in interleaved PHYLIP format.
///
/// Matches the output of the C `phylipout_pointer()` function.
///
/// - `order`: optional sequence ordering. `None` = natural order.
/// - `name_len`: name field width. `None` = default (10).
pub fn write_phylip<W: Write>(
    seqs: &SequenceSet,
    writer: &mut W,
    order: Option<&[usize]>,
    name_len: Option<usize>,
) -> Result<(), IoError> {
    let nseq = seqs.nseq();
    if nseq == 0 {
        return Err(IoError::EmptyInput);
    }

    let name_len = name_len.unwrap_or(DEFAULT_NAME_LEN);
    let max_len = seqs.max_len();

    let default_order: Vec<usize> = (0..nseq).collect();
    let order = order.unwrap_or(&default_order);

    // Header
    writeln!(writer, " {} {}", nseq, max_len)?;

    let mut pos = 0;
    while pos < max_len {
        for &idx in order {
            let seq = &seqs.sequences[idx];

            // Name field on first block only. C `phylipout_pointer`
            // uses `%-*.*s` which truncates AND pads to `namelen`; the
            // Rust `{:<width$}` only pads, so we use a helper.
            if pos == 0 {
                let first_word = seq.name.split_whitespace().next().unwrap_or(&seq.name);
                write_name_field(writer, first_word, name_len)?;
            } else {
                write_name_field(writer, "", name_len)?;
            }

            // Write groups of 10 residues separated by spaces
            let mut p = pos;
            let end = (pos + LINE_WIDTH).min(max_len);
            while p < end {
                let group_end = (p + GROUP_WIDTH).min(end).min(seq.data.len());
                write!(writer, " ")?;
                if p < seq.data.len() {
                    writer.write_all(&seq.data[p..group_end])?;
                }
                p += GROUP_WIDTH;
            }
            writeln!(writer)?;
        }

        writeln!(writer)?; // blank line between blocks
        pos += LINE_WIDTH;
    }

    Ok(())
}

/// Write a name field with C `%-*.*s` semantics: left-align, pad to
/// `width` chars with spaces, AND truncate if longer than `width`.
fn write_name_field<W: Write>(writer: &mut W, name: &str, width: usize) -> Result<(), IoError> {
    let bytes = name.as_bytes();
    let len = bytes.len().min(width);
    writer.write_all(&bytes[..len])?;
    for _ in len..width {
        writer.write_all(b" ")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mafft_types::{SeqType, Sequence};

    #[test]
    fn phylip_header() {
        let seqs = SequenceSet {
            sequences: vec![
                Sequence {
                    name: "s1".into(),
                    data: b"ACGT".to_vec(),
                },
                Sequence {
                    name: "s2".into(),
                    data: b"TGCA".to_vec(),
                },
            ],
            seq_type: SeqType::Dna,
        };

        let mut buf = Vec::new();
        write_phylip(&seqs, &mut buf, None, None).unwrap();
        let output = String::from_utf8(buf).unwrap();

        let first_line = output.lines().next().unwrap();
        assert_eq!(first_line.trim(), "2 4");
    }
}

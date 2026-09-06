use std::io::{BufRead, BufReader, Read, Write};

use mafft_types::{HomologyRegion, LocalHomologyTable};

use crate::error::IoError;

/// Score scaling factor used by MAFFT: raw_score / 5.8 * 600.
const SCORE_SCALE_DIVISOR: f64 = 5.8;
const SCORE_SCALE_FACTOR: f64 = 600.0;

/// Read a local homology table (hat3 format).
///
/// Each line: `<i> <j> <overlapaa> <score> <start1> <end1> <start2> <end2> <korh>`
///
/// The score is scaled by `(raw / 5.8) * 600` to match the C code's convention.
/// Reciprocal entries (j, i) are created automatically with swapped coordinates.
pub fn read_localhom_table<R: Read>(reader: R, nseq: usize) -> Result<LocalHomologyTable, IoError> {
    let reader = BufReader::new(reader);
    let mut table = LocalHomologyTable::new(nseq);

    for line_result in reader.lines() {
        let line = line_result?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 9 {
            return Err(IoError::LocalHomFormat(format!(
                "expected 9 fields, got {}: '{line}'",
                fields.len()
            )));
        }

        let i: usize = fields[0]
            .parse()
            .map_err(|_| IoError::LocalHomFormat(format!("invalid seq index: '{}'", fields[0])))?;
        let j: usize = fields[1]
            .parse()
            .map_err(|_| IoError::LocalHomFormat(format!("invalid seq index: '{}'", fields[1])))?;
        let overlapaa: i32 = fields[2]
            .parse()
            .map_err(|_| IoError::LocalHomFormat(format!("invalid overlap: '{}'", fields[2])))?;
        let raw_score: f64 = fields[3]
            .parse()
            .map_err(|_| IoError::LocalHomFormat(format!("invalid score: '{}'", fields[3])))?;
        let start1: i32 = fields[4]
            .parse()
            .map_err(|_| IoError::LocalHomFormat(format!("invalid start1: '{}'", fields[4])))?;
        let end1: i32 = fields[5]
            .parse()
            .map_err(|_| IoError::LocalHomFormat(format!("invalid end1: '{}'", fields[5])))?;
        let start2: i32 = fields[6]
            .parse()
            .map_err(|_| IoError::LocalHomFormat(format!("invalid start2: '{}'", fields[6])))?;
        let end2: i32 = fields[7]
            .parse()
            .map_err(|_| IoError::LocalHomFormat(format!("invalid end2: '{}'", fields[7])))?;
        let korh: u8 = fields[8].as_bytes().first().copied().unwrap_or(b'h');

        let opt = (raw_score / SCORE_SCALE_DIVISOR) * SCORE_SCALE_FACTOR;

        // Forward entry (i → j)
        let region = HomologyRegion {
            start1,
            end1,
            start2,
            end2,
            opt,
            overlapaa,
            korh,
            ..Default::default()
        };
        table.push(i, j, region);

        // Reciprocal entry (j → i) with swapped coordinates
        if i != j {
            let reciprocal = HomologyRegion {
                start1: start2,
                end1: end2,
                start2: start1,
                end2: end1,
                opt,
                overlapaa,
                korh,
                ..Default::default()
            };
            table.push(j, i, reciprocal);
        }
    }

    Ok(table)
}

/// Write a local homology table (hat3 format).
///
/// Only writes the upper triangle (i < j) to avoid duplicate entries.
/// Score is written as the raw value (before scaling).
pub fn write_localhom_table<W: Write>(
    table: &LocalHomologyTable,
    writer: &mut W,
) -> Result<(), IoError> {
    for i in 0..table.nseq {
        for j in (i + 1)..table.nseq {
            let regions = table.get(i, j);
            for region in regions {
                // Reverse the score scaling: raw = opt * 5.8 / 600
                let raw_score = region.opt * SCORE_SCALE_DIVISOR / SCORE_SCALE_FACTOR;
                writeln!(
                    writer,
                    "{} {} {} {:.1} {} {} {} {} {}",
                    i,
                    j,
                    region.overlapaa,
                    raw_score,
                    region.start1,
                    region.end1,
                    region.start2,
                    region.end2,
                    region.korh as char,
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn roundtrip_localhom() {
        let input = "0 1 50 290.0 10 60 20 70 h\n0 2 30 145.0 5 35 15 45 k\n";
        let table = read_localhom_table(Cursor::new(input), 3).unwrap();

        // Forward entries
        let regions_01 = table.get(0, 1);
        assert_eq!(regions_01.len(), 1);
        assert_eq!(regions_01[0].start1, 10);
        assert_eq!(regions_01[0].end1, 60);
        assert_eq!(regions_01[0].start2, 20);
        assert_eq!(regions_01[0].end2, 70);

        // Reciprocal entries (coordinates swapped)
        let regions_10 = table.get(1, 0);
        assert_eq!(regions_10.len(), 1);
        assert_eq!(regions_10[0].start1, 20); // was start2
        assert_eq!(regions_10[0].end1, 70); // was end2
        assert_eq!(regions_10[0].start2, 10); // was start1
        assert_eq!(regions_10[0].end2, 60); // was end1

        // Score scaling: 290.0 / 5.8 * 600 = 30000.0
        let expected_opt = (290.0 / 5.8) * 600.0;
        assert!((regions_01[0].opt - expected_opt).abs() < 0.01);

        // Write and verify
        let mut buf = Vec::new();
        write_localhom_table(&table, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("0 1 50"));
        assert!(output.contains("0 2 30"));
    }
}

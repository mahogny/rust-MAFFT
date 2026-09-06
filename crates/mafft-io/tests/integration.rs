use std::io::Cursor;
use std::path::PathBuf;

use mafft_io::{detect_seq_type, read_fasta, write_clustal, write_fasta_to_writer, write_phylip};
use mafft_types::SeqType;

fn test_data_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../mafft-upstream/test")
        .join(name)
}

#[test]
fn read_sample_fasta() {
    let seqs = read_fasta(test_data_path("sample")).unwrap();

    assert_eq!(seqs.nseq(), 36);
    assert_eq!(seqs.seq_type, SeqType::Protein);

    // First sequence name should contain "M63632"
    assert!(
        seqs.sequences[0].name.contains("M63632"),
        "first seq name: {}",
        seqs.sequences[0].name
    );

    // All sequences should be non-empty
    for seq in &seqs.sequences {
        assert!(!seq.is_empty(), "empty sequence: {}", seq.name);
    }

    // Sequences should contain only uppercase alpha + gap chars
    for seq in &seqs.sequences {
        for &ch in &seq.data {
            assert!(
                ch.is_ascii_uppercase() || ch == b'-' || ch == b'.',
                "unexpected char {} in sequence {}",
                ch as char,
                seq.name
            );
        }
    }
}

#[test]
fn read_rna_sample() {
    let seqs = read_fasta(test_data_path("samplerna")).unwrap();

    assert!(seqs.nseq() > 0);
    // RNA sample should be detected as DNA/nucleotide
    assert!(
        seqs.seq_type.is_nucleotide(),
        "expected nucleotide, got {:?}",
        seqs.seq_type
    );
}

#[test]
fn fasta_roundtrip_preserves_content() {
    let original = read_fasta(test_data_path("sample")).unwrap();

    // Write to buffer
    let mut buf = Vec::new();
    write_fasta_to_writer(&original, &mut buf).unwrap();

    // Read back
    let reparsed = mafft_io::read_fasta_from_reader(Cursor::new(&buf)).unwrap();

    assert_eq!(reparsed.nseq(), original.nseq());
    for (orig, re) in original.sequences.iter().zip(reparsed.sequences.iter()) {
        assert_eq!(orig.data, re.data, "data mismatch for {}", orig.name);
    }
}

#[test]
fn clustal_output_has_correct_structure() {
    let seqs = read_fasta(test_data_path("sample")).unwrap();

    let mut buf = Vec::new();
    write_clustal(&seqs, &mut buf, None, None, None).unwrap();
    let output = String::from_utf8(buf).unwrap();

    // Must start with CLUSTAL header
    assert!(output.starts_with("CLUSTAL format alignment by MAFFT"));

    // Must contain sequence names (first word)
    let first_word = seqs.sequences[0].name.split_whitespace().next().unwrap();
    assert!(
        output.contains(first_word),
        "missing first sequence name: {first_word}"
    );
}

#[test]
fn phylip_output_has_correct_header() {
    let seqs = read_fasta(test_data_path("sample")).unwrap();

    let mut buf = Vec::new();
    write_phylip(&seqs, &mut buf, None, None).unwrap();
    let output = String::from_utf8(buf).unwrap();

    let first_line = output.lines().next().unwrap();
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    assert_eq!(parts.len(), 2);

    let nseq: usize = parts[0].parse().unwrap();
    assert_eq!(nseq, 36);

    let max_len: usize = parts[1].parse().unwrap();
    assert_eq!(max_len, seqs.max_len());
}

#[test]
fn detect_protein_from_sample() {
    let seqs = read_fasta(test_data_path("sample")).unwrap();
    let data: Vec<Vec<u8>> = seqs.sequences.iter().map(|s| s.data.clone()).collect();
    assert_eq!(detect_seq_type(&data), SeqType::Protein);
}

#[test]
fn detect_nucleotide_from_rna_sample() {
    let seqs = read_fasta(test_data_path("samplerna")).unwrap();
    let data: Vec<Vec<u8>> = seqs.sequences.iter().map(|s| s.data.clone()).collect();
    assert!(detect_seq_type(&data).is_nucleotide());
}

fn main() {
    use mafft_scoring::build_context;
    use mafft_types::{ScoringModel, SeqType};
    let blosum = build_context(ScoringModel::Blosum(62), SeqType::Protein);
    let tm = build_context(ScoringModel::Tm(200), SeqType::Protein);
    println!(
        "BLOSUM62: gap.open={}, gap.extend={}, gap.offset={}, fft_matrix[0][0]={}, sub[0][0]={}",
        blosum.gap.open,
        blosum.gap.extend,
        blosum.gap.offset,
        blosum.fft_matrix[0][0],
        blosum.substitution_matrix[0][0]
    );
    println!(
        "TM 200: gap.open={}, gap.extend={}, gap.offset={}, fft_matrix[0][0]={}, sub[0][0]={}",
        tm.gap.open,
        tm.gap.extend,
        tm.gap.offset,
        tm.fft_matrix[0][0],
        tm.substitution_matrix[0][0]
    );
}

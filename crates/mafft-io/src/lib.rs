//! I/O routines for MAFFT: FASTA, Clustal, PHYLIP, hat2, and localhom formats.
//!
//! FASTA reading/writing delegates to `noodles-fasta`. The other formats are
//! MAFFT-specific or simple enough that we implement them directly.

mod clustal;
mod detect;
mod error;
mod fasta;
mod hat2;
mod localhom;
mod phylip;

pub use clustal::{compute_clustal_marks, write_clustal, write_clustal_full};
pub use detect::detect_seq_type;
pub use error::IoError;
pub use fasta::{
    apply_case_convention, read_fasta, read_fasta_casepreserve, read_fasta_from_reader,
    read_fasta_from_reader_casepreserve, write_fasta, write_fasta_to_writer,
    write_fasta_to_writer_with_width,
};
pub use hat2::{Hat2Matrix, read_hat2, write_hat2};
pub use localhom::{read_localhom_table, write_localhom_table};
pub use phylip::write_phylip;

//! Distance computation and guide tree construction for MAFFT.
//!
//! Ports the C `nj.c` (neighbor-joining), UPGMA variants from `mltaln9.c`,
//! k-tuple distance from `tddis.c`, and sequence weighting from tree topology.

mod addonetip;
pub mod bsd_qsort;
mod distance;
pub mod memsavetree;
mod musclesupg;
pub mod newick;
mod nj;
pub mod parttree_dist;
pub mod parttree_pivot;
pub mod parttree_split;
mod topology;
pub mod treein;
mod upgma;
mod weighting;

pub use addonetip::{AddResult, addonetip, compute_distfromtip, generate_subalignments_table};
pub use distance::{
    DistanceMatrix, ktuple_distance, pairwise_identity_distance, scoring_matrix_distance,
};
pub use musclesupg::{ClusterMethod, musclesupg};
pub use newick::topology_to_newick;
pub use nj::neighbor_joining;
pub use topology::{JoinStep, Topology};
pub use treein::{parse_mafft_tree, parse_mafft_tree_str};
pub use upgma::{upgma, upgma_int};
pub use weighting::{BranchWeights, sequence_weights};

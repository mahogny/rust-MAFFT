//! Newick tree serialization for `--treeout`.
//!
//! Mirrors C MAFFT's tree-output format (`mltaln9.c:6190-6491` in
//! `fixed_musclesupg_double_realloc_nobk_halfmtx_treeout_memsave`):
//!
//! - Each leaf becomes `\n<i+1>_<sanitized_name>\n` where `i` is the
//!   0-indexed input position and `<sanitized_name>` is the FASTA
//!   header (without the leading `>`) with every char that is NOT
//!   alphanumeric, `/`, `=`, `-`, `{`, or `}` replaced by `_`.
//! - Each merge `(im, jm)` is formatted as
//!   `(<tree[im]>:<len[k][0]>,<tree[jm]>:<len[k][1]>)` with branch
//!   lengths printed via C's `%7.5f` (field width 7, 5 decimals).
//! - The final Newick string terminates with `;\n`.
//!
//! C's leaf naming has one subtlety: C reads FASTA names with the
//! leading `>` retained as `name[i][0]`. The sanitization loop converts
//! `>` to `_`, then `nameptr = nametmp + 1` skips that leading `_`. Our
//! `mafft_io` strips the `>` on read, so we sanitize the name directly
//! WITHOUT inserting a synthetic leading `_`.

use std::collections::BTreeMap;

use crate::topology::Topology;

/// Sanitize a FASTA name char-by-char, matching C's
/// `fixed_musclesupg_double_realloc_nobk_halfmtx_treeout_memsave`
/// (`mltaln9.c:6193-6202`).
fn sanitize_char(c: u8) -> u8 {
    if c.is_ascii_alphanumeric() || matches!(c, b'/' | b'=' | b'-' | b'{' | b'}') {
        c
    } else {
        b'_'
    }
}

fn sanitize_name(name: &str) -> String {
    name.bytes().map(sanitize_char).map(|b| b as char).collect()
}

/// Build a Newick tree string from a topology + leaf names, mirroring
/// C MAFFT 7.526's `--treeout` output exactly. Trailing `;\n` included.
///
/// Requirements:
/// - `topology.is_complete()` (i.e., `nseq - 1` merge steps).
/// - `names.len() == topology.nseq`.
pub fn topology_to_newick(topology: &Topology, names: &[String]) -> String {
    assert_eq!(names.len(), topology.nseq, "names length must equal nseq");
    if topology.nseq == 0 {
        return ";\n".to_string();
    }
    if topology.nseq == 1 {
        let leaf = format!("\n1_{}\n", sanitize_name(&names[0]));
        return format!("{};\n", leaf);
    }

    // Map each subtree (as a sorted set of leaf indices) to its current
    // Newick fragment. Initialize with one entry per leaf.
    let mut subtree: BTreeMap<Vec<usize>, String> = BTreeMap::new();
    for i in 0..topology.nseq {
        let frag = format!("\n{}_{}\n", i + 1, sanitize_name(&names[i]));
        subtree.insert(vec![i], frag);
    }

    for step in &topology.steps {
        let mut left_key = step.left.clone();
        left_key.sort_unstable();
        let mut right_key = step.right.clone();
        right_key.sort_unstable();
        let left_frag = subtree.remove(&left_key).expect("left subtree missing");
        let right_frag = subtree.remove(&right_key).expect("right subtree missing");
        let merged = format!(
            "({}:{:7.5},{}:{:7.5})",
            left_frag, step.left_length, right_frag, step.right_length,
        );
        let mut combined_key: Vec<usize> = left_key;
        combined_key.extend(right_key);
        combined_key.sort_unstable();
        subtree.insert(combined_key, merged);
    }

    let root_key: Vec<usize> = (0..topology.nseq).collect();
    let root = subtree.remove(&root_key).expect("root subtree missing");
    format!("{};\n", root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::JoinStep;

    #[test]
    fn sanitize_keeps_allowed_chars() {
        assert_eq!(sanitize_name("ABC123/=-{}"), "ABC123/=-{}");
        assert_eq!(sanitize_name("a b\tc"), "a_b_c");
        assert_eq!(sanitize_name(">seq1| name"), "_seq1__name");
    }

    #[test]
    fn newick_two_leaf() {
        let mut t = Topology::new(2);
        t.steps.push(JoinStep {
            left: vec![0],
            right: vec![1],
            left_length: 0.1,
            right_length: 0.2,
        });
        let names = vec!["A".to_string(), "B".to_string()];
        // %7.5f -> "0.10000" / "0.20000"
        let expected = "(\n1_A\n:0.10000,\n2_B\n:0.20000);\n";
        assert_eq!(topology_to_newick(&t, &names), expected);
    }

    #[test]
    fn newick_four_leaf_balanced() {
        let mut t = Topology::new(4);
        t.steps.push(JoinStep {
            left: vec![0],
            right: vec![1],
            left_length: 0.05,
            right_length: 0.05,
        });
        t.steps.push(JoinStep {
            left: vec![2],
            right: vec![3],
            left_length: 0.10,
            right_length: 0.10,
        });
        t.steps.push(JoinStep {
            left: vec![0, 1],
            right: vec![2, 3],
            left_length: 0.20,
            right_length: 0.15,
        });
        let names: Vec<String> = vec!["a", "b", "c", "d"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let nw = topology_to_newick(&t, &names);
        assert!(nw.ends_with(";\n"));
        assert!(nw.contains("(\n1_a\n:0.05000,\n2_b\n:0.05000)"));
        assert!(nw.contains("(\n3_c\n:0.10000,\n4_d\n:0.10000)"));
        assert!(nw.contains(":0.20000,"));
        assert!(nw.contains(":0.15000)"));
    }
}

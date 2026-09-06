use mafft_tree::ktuple_distance;
use std::fs;

fn parse_fasta(s: &str) -> Vec<(String, Vec<u8>)> {
    let mut r = vec![];
    let mut name = String::new();
    let mut buf = String::new();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix('>') {
            if !name.is_empty() {
                r.push((name.clone(), buf.bytes().collect()));
            }
            name = rest.split_whitespace().next().unwrap_or("").to_string();
            buf.clear();
        } else {
            buf.push_str(line.trim());
        }
    }
    if !name.is_empty() {
        r.push((name, buf.bytes().collect()));
    }
    r
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_ktuple_distance <fasta>");
    let s = fs::read_to_string(&path).unwrap();
    let seqs = parse_fasta(&s);
    println!("loaded {} seqs", seqs.len());
    for (i, (na, _)) in seqs.iter().enumerate() {
        println!("  {} {}", i + 1, na);
    }
    println!();
    let n = seqs.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = ktuple_distance(&seqs[i].1, &seqs[j].1, 6);
            println!(
                "d({},{}) = {:.10}  ({}<->{})",
                i + 1,
                j + 1,
                d,
                seqs[i].0,
                seqs[j].0
            );
        }
    }
}

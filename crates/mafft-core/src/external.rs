/// External tool integration for RNA and structure-based alignment modes.
///
/// Q-INS-i: McCaskill base-pair probabilities via `mxscarnamod`
/// X-INS-i: CONTRAfold structure predictions via `contrafold`
/// SCARNA-like: Structural homology via `dash_client`
///
/// These modes follow the same pattern as C MAFFT: call external tools
/// as subprocesses, parse their output, and convert to constraint tables
/// for use in the alignment DP.
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Base-pair probability: position i pairs with position j with given probability.
#[derive(Debug, Clone)]
pub struct BasePairProb {
    pub left: usize,
    pub right: usize,
    pub prob: f64,
}

/// Per-sequence base-pair probability table.
#[derive(Debug, Clone)]
pub struct SequenceBpp {
    pub seq_index: usize,
    pub pairs: Vec<BasePairProb>,
}

/// Find an external tool in PATH or MAFFT_BINARIES directory.
pub fn find_tool(name: &str) -> Option<PathBuf> {
    // Check MAFFT_BINARIES environment variable first
    if let Ok(bindir) = std::env::var("MAFFT_BINARIES") {
        let path = Path::new(&bindir).join(name);
        if path.exists() {
            return Some(path);
        }
    }

    // Check PATH
    if let Ok(output) = Command::new("which").arg(name).output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    None
}

/// Run McCaskill base-pair probability prediction on a single sequence.
///
/// Calls: `mxscarnamod -m -writebpp`
/// Input: FASTA (ungapped)
/// Output: `left right probability` per line
pub fn run_mccaskill(sequence: &[u8], tool_path: &Path) -> Result<Vec<BasePairProb>, String> {
    let tmpdir = std::env::temp_dir();
    let infile = tmpdir.join("_mafftrs_mccaskillin");
    let outfile = tmpdir.join("_mafftrs_mccaskillout");

    // Write input FASTA
    let mut f =
        std::fs::File::create(&infile).map_err(|e| format!("Cannot create temp file: {e}"))?;
    writeln!(f, ">seq").map_err(|e| format!("Write error: {e}"))?;
    f.write_all(sequence)
        .map_err(|e| format!("Write error: {e}"))?;
    writeln!(f).map_err(|e| format!("Write error: {e}"))?;
    drop(f);

    // Run mxscarnamod
    let output = Command::new(tool_path)
        .args(["-m", "-writebpp"])
        .stdin(std::fs::File::open(&infile).map_err(|e| format!("Cannot open temp: {e}"))?)
        .output()
        .map_err(|e| format!("Failed to run mxscarnamod: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "mxscarnamod failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // Write output to file for parsing
    std::fs::write(&outfile, &output.stdout).map_err(|e| format!("Cannot write output: {e}"))?;

    // Parse output: "left right probability" per line
    let content = String::from_utf8_lossy(&output.stdout);
    let pairs = parse_mccaskill_output(&content);

    // Cleanup
    let _ = std::fs::remove_file(&infile);
    let _ = std::fs::remove_file(&outfile);

    Ok(pairs)
}

/// Parse McCaskill output format: `left right probability` per line.
fn parse_mccaskill_output(content: &str) -> Vec<BasePairProb> {
    let mut pairs = Vec::new();
    for line in content.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            if let (Ok(left), Ok(right), Ok(prob)) = (
                parts[0].parse::<usize>(),
                parts[1].parse::<usize>(),
                parts[2].parse::<f64>(),
            ) {
                if prob >= 0.01 {
                    pairs.push(BasePairProb { left, right, prob });
                }
            }
        }
    }
    pairs
}

/// Run CONTRAfold structure prediction on a single sequence.
///
/// Calls: `contrafold predict <infile> --posteriors 0.01 <outfile>`
/// Input: FASTA (single sequence)
/// Output: `pos pair1:prob pair2:prob ...` per line (1-indexed)
pub fn run_contrafold(sequence: &[u8], tool_path: &Path) -> Result<Vec<BasePairProb>, String> {
    let tmpdir = std::env::temp_dir();
    let infile = tmpdir.join("_mafftrs_contrafoldin");
    let outfile = tmpdir.join("_mafftrs_contrafoldout");

    // Write input FASTA
    let mut f =
        std::fs::File::create(&infile).map_err(|e| format!("Cannot create temp file: {e}"))?;
    writeln!(f, ">seq").map_err(|e| format!("Write error: {e}"))?;
    f.write_all(sequence)
        .map_err(|e| format!("Write error: {e}"))?;
    writeln!(f).map_err(|e| format!("Write error: {e}"))?;
    drop(f);

    // Run contrafold
    let status = Command::new(tool_path)
        .args([
            "predict",
            infile.to_str().unwrap(),
            "--posteriors",
            "0.01",
            outfile.to_str().unwrap(),
        ])
        .status()
        .map_err(|e| format!("Failed to run contrafold: {e}"))?;

    if !status.success() {
        return Err("contrafold failed".to_string());
    }

    // Parse output
    let content = std::fs::read_to_string(&outfile)
        .map_err(|e| format!("Cannot read contrafold output: {e}"))?;
    let pairs = parse_contrafold_output(&content);

    // Cleanup
    let _ = std::fs::remove_file(&infile);
    let _ = std::fs::remove_file(&outfile);

    Ok(pairs)
}

/// Parse CONTRAfold output: `pos pair1:prob pair2:prob ...` (1-indexed).
fn parse_contrafold_output(content: &str) -> Vec<BasePairProb> {
    let mut pairs = Vec::new();
    for line in content.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        if let Ok(left) = parts[0].parse::<usize>() {
            let left = left.saturating_sub(1); // Convert 1-indexed to 0-indexed
            for &part in &parts[1..] {
                if let Some(colon) = part.find(':') {
                    if let (Ok(right), Ok(prob)) = (
                        part[..colon].parse::<usize>(),
                        part[colon + 1..].parse::<f64>(),
                    ) {
                        let right = right.saturating_sub(1);
                        if prob >= 0.01 {
                            pairs.push(BasePairProb { left, right, prob });
                        }
                    }
                }
            }
        }
    }
    pairs
}

/// Run DASH structural alignment client.
///
/// Calls: `dash_client -url <server> -i <infile> -hat3 <outfile>`
/// Input: FASTA (ungapped)
/// Output: hat3 format constraint table
pub fn run_dash(
    sequences: &[Vec<u8>],
    names: &[String],
    server_url: &str,
) -> Result<Vec<(usize, usize, f64, usize, usize, usize, usize)>, String> {
    let tool_path = find_tool("dash_client")
        .ok_or_else(|| "dash_client not found in PATH or MAFFT_BINARIES".to_string())?;

    let tmpdir = std::env::temp_dir();
    let infile = tmpdir.join("_mafftrs_dashin");
    let outfile = tmpdir.join("_mafftrs_hat3seed");

    // Write input FASTA (ungapped)
    let mut f =
        std::fs::File::create(&infile).map_err(|e| format!("Cannot create temp file: {e}"))?;
    for (name, seq) in names.iter().zip(sequences.iter()) {
        writeln!(f, ">{name}").map_err(|e| format!("Write error: {e}"))?;
        let ungapped: Vec<u8> = seq.iter().filter(|&&c| c != b'-').copied().collect();
        f.write_all(&ungapped)
            .map_err(|e| format!("Write error: {e}"))?;
        writeln!(f).map_err(|e| format!("Write error: {e}"))?;
    }
    drop(f);

    // Run dash_client
    let status = Command::new(&tool_path)
        .args([
            "-url",
            server_url,
            "-i",
            infile.to_str().unwrap(),
            "-hat3",
            outfile.to_str().unwrap(),
        ])
        .status()
        .map_err(|e| format!("Failed to run dash_client: {e}"))?;

    if !status.success() {
        return Err("dash_client failed".to_string());
    }

    // Parse hat3 output
    let content =
        std::fs::read_to_string(&outfile).map_err(|e| format!("Cannot read DASH output: {e}"))?;
    let constraints = parse_hat3(&content);

    // Cleanup
    let _ = std::fs::remove_file(&infile);
    let _ = std::fs::remove_file(&outfile);

    Ok(constraints)
}

/// Parse hat3 format: `i j overlapaa opt start1 end1 start2 end2`
fn parse_hat3(content: &str) -> Vec<(usize, usize, f64, usize, usize, usize, usize)> {
    let mut constraints = Vec::new();
    for line in content.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 8 {
            if let (Ok(i), Ok(j), Ok(opt), Ok(s1), Ok(e1), Ok(s2), Ok(e2)) = (
                parts[0].parse::<usize>(),
                parts[1].parse::<usize>(),
                parts[3].parse::<f64>(),
                parts[4].parse::<usize>(),
                parts[5].parse::<usize>(),
                parts[6].parse::<usize>(),
                parts[7].parse::<usize>(),
            ) {
                constraints.push((i, j, opt, s1, e1, s2, e2));
            }
        }
    }
    constraints
}

/// Compute base-pair probabilities for all sequences using McCaskill (Q-INS-i).
pub fn compute_bpp_mccaskill(sequences: &[Vec<u8>]) -> Result<Vec<SequenceBpp>, String> {
    let tool = find_tool("mxscarnamod").ok_or_else(|| {
        "mxscarnamod not found. Q-INS-i requires the McCaskill base-pair probability program.\n\
             Install it and ensure it is in your PATH or set MAFFT_BINARIES.\n\
             See: https://mafft.cbrc.jp/alignment/software/source.html"
            .to_string()
    })?;

    let mut results = Vec::with_capacity(sequences.len());
    for (idx, seq) in sequences.iter().enumerate() {
        // Strip gaps for BPP prediction
        let ungapped: Vec<u8> = seq.iter().filter(|&&c| c != b'-').copied().collect();
        let pairs = run_mccaskill(&ungapped, &tool)?;
        results.push(SequenceBpp {
            seq_index: idx,
            pairs,
        });
    }
    Ok(results)
}

/// Compute base-pair probabilities for all sequences using CONTRAfold (X-INS-i).
pub fn compute_bpp_contrafold(sequences: &[Vec<u8>]) -> Result<Vec<SequenceBpp>, String> {
    let tool = find_tool("contrafold").ok_or_else(|| {
        "contrafold not found. X-INS-i requires CONTRAfold.\n\
             Install CONTRAfold v2.02+ and ensure it is in your PATH or set MAFFT_BINARIES.\n\
             See: https://mafft.cbrc.jp/alignment/software/source.html"
            .to_string()
    })?;

    let mut results = Vec::with_capacity(sequences.len());
    for (idx, seq) in sequences.iter().enumerate() {
        let ungapped: Vec<u8> = seq.iter().filter(|&&c| c != b'-').copied().collect();
        let pairs = run_contrafold(&ungapped, &tool)?;
        results.push(SequenceBpp {
            seq_index: idx,
            pairs,
        });
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mccaskill_format() {
        let input = "0 5 0.85\n1 4 0.72\n2 3 0.005\n";
        let pairs = parse_mccaskill_output(input);
        assert_eq!(pairs.len(), 2); // 0.005 filtered out
        assert_eq!(pairs[0].left, 0);
        assert_eq!(pairs[0].right, 5);
        assert!((pairs[0].prob - 0.85).abs() < 1e-10);
    }

    #[test]
    fn parse_contrafold_format() {
        let input = "1 2:0.95 5:0.42\n3 4:0.31\n";
        let pairs = parse_contrafold_output(input);
        assert_eq!(pairs.len(), 3);
        // 1-indexed → 0-indexed
        assert_eq!(pairs[0].left, 0);
        assert_eq!(pairs[0].right, 1);
        assert!((pairs[0].prob - 0.95).abs() < 1e-10);
    }

    #[test]
    fn parse_hat3_format() {
        let input = "0 1 100 5.8 10 20 30 40 info\n2 3 50 2.9 15 25 35 45 info\n";
        let constraints = parse_hat3(input);
        assert_eq!(constraints.len(), 2);
        assert_eq!(constraints[0].0, 0); // i
        assert_eq!(constraints[0].1, 1); // j
        assert!((constraints[0].2 - 5.8).abs() < 1e-10); // opt
    }

    #[test]
    fn find_tool_nonexistent() {
        assert!(find_tool("nonexistent_tool_xyz_12345").is_none());
    }
}

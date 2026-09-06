/// Multi-channel sequence vectorization for FFT homology detection.
///
/// Ports the C `seq_vec_3()` and multi-channel correlation from Falign.c.
///
/// This is the core MAFFT innovation: each amino acid type (or biochemical
/// property) gets its own complex channel. The FFT correlation is computed
/// per-channel and summed, giving a position-specific similarity score that
/// accounts for all residue types simultaneously.
use num_complex::Complex64;

/// Number of channels for different sequence types.
pub const PROTEIN_CHANNELS: usize = 20;
pub const DNA_CHANNELS: usize = 4;

/// Convert a sequence group into per-channel complex vectors.
///
/// Each channel `k` (one per residue type) gets a vector where position `i`
/// has `R = sum of weights for sequences with residue k at position i`.
///
/// This is the multi-channel generalization of `seq_vec_3()` from C.
pub fn sequences_to_channels(
    sequences: &[&[u8]],
    weights: &[f64],
    amino_map: &[u8; 256],
    num_channels: usize,
    fft_size: usize,
) -> Vec<Vec<Complex64>> {
    let mut channels = vec![vec![Complex64::new(0.0, 0.0); fft_size]; num_channels];

    for (seq, &w) in sequences.iter().zip(weights.iter()) {
        for (pos, &ch) in seq.iter().enumerate() {
            if pos >= fft_size {
                break;
            }
            let idx = amino_map[ch as usize] as usize;
            if idx < num_channels {
                channels[idx][pos] += Complex64::new(w, 0.0);
            }
        }
    }

    channels
}

/// Compute multi-channel FFT cross-correlation.
///
/// For each channel, computes `conj(FFT(a_k)) * FFT(b_k)`, then sums
/// across all channels and inverse-FFTs to get the correlation profile.
///
/// Returns a real-valued correlation vector of length `fft_size`.
pub fn multichannel_correlate(
    channels_a: &[Vec<Complex64>],
    channels_b: &[Vec<Complex64>],
) -> Vec<f64> {
    assert_eq!(channels_a.len(), channels_b.len());
    let num_channels = channels_a.len();
    if num_channels == 0 || channels_a[0].is_empty() {
        return Vec::new();
    }

    let n = channels_a[0].len();

    // C's `fft()` (Cooley-Tukey radix-2, fft.c) and `rustfft` produce
    // bit-different correlation peaks for the same input. For most
    // scoring matrices the rounding doesn't shift the chosen anchor
    // lag, but `--tm 200` (FFT) has a flatter correlation landscape
    // that surfaces the difference as a 2-column shift in step 13's
    // anchor placement. Route through the bit-for-bit C-port in
    // `fft_c_compat` so anchor lags match C exactly.
    use crate::fft_c_compat::fft_inplace;

    // Accumulate correlation across all channels.
    // C convention: forward FFT applies 1/n scale. Two forward FFTs +
    // unnormalized inverse → result = correlation / n. Downstream peak
    // selection is scale-invariant, so we don't multiply back by n.
    let mut sum = vec![Complex64::new(0.0, 0.0); n];

    for k in 0..num_channels {
        let mut fa = channels_a[k].clone();
        let mut fb = channels_b[k].clone();
        fft_inplace(&mut fa, /* inverse = */ false);
        fft_inplace(&mut fb, /* inverse = */ false);
        for i in 0..n {
            sum[i] += fa[i].conj() * fb[i];
        }
    }

    fft_inplace(&mut sum, /* inverse = */ true);
    sum.iter().map(|c| c.re).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_sequences_peak_at_zero() {
        let mut map = [0xFFu8; 256];
        map[b'A' as usize] = 0;
        map[b'C' as usize] = 1;
        map[b'G' as usize] = 2;
        map[b'T' as usize] = 3;

        let seq = b"ACGTACGTACGTACGT";
        let seqs: Vec<&[u8]> = vec![seq];
        let weights = vec![1.0];
        let fft_size = 32;

        let ch_a = sequences_to_channels(&seqs, &weights, &map, DNA_CHANNELS, fft_size);
        let ch_b = sequences_to_channels(&seqs, &weights, &map, DNA_CHANNELS, fft_size);

        let corr = multichannel_correlate(&ch_a, &ch_b);

        // Autocorrelation should peak at lag 0
        let peak_idx = corr
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak_idx, 0);
    }

    #[test]
    fn shifted_sequences_detected() {
        let mut map = [0xFFu8; 256];
        map[b'A' as usize] = 0;
        map[b'C' as usize] = 1;
        map[b'G' as usize] = 2;
        map[b'T' as usize] = 3;

        let fft_size = 32;
        // seq_a has pattern starting at pos 4
        let seq_a = b"AAAACGTACGTAAAAAAAAAAAAAAAAAAAAAA";
        // seq_b has same pattern starting at pos 0
        let seq_b = b"CGTACGTAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let seqs_a: Vec<&[u8]> = vec![&seq_a[..fft_size]];
        let seqs_b: Vec<&[u8]> = vec![&seq_b[..fft_size]];

        let ch_a = sequences_to_channels(&seqs_a, &[1.0], &map, DNA_CHANNELS, fft_size);
        let ch_b = sequences_to_channels(&seqs_b, &[1.0], &map, DNA_CHANNELS, fft_size);

        let corr = multichannel_correlate(&ch_a, &ch_b);

        // Peak should be near lag 4 (or n-4 for negative wrap)
        let peak_idx = corr
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let lag = if peak_idx > fft_size / 2 {
            peak_idx as i32 - fft_size as i32
        } else {
            peak_idx as i32
        };
        assert!(lag.abs() <= 5, "expected lag near ±4, got {lag}");
    }

    #[test]
    fn weighted_sequences() {
        let mut map = [0xFFu8; 256];
        map[b'A' as usize] = 0;
        map[b'C' as usize] = 1;

        let fft_size = 8;
        let seqs: Vec<&[u8]> = vec![b"AACCAACC", b"CCAACCAA"];
        let weights = vec![0.7, 0.3];

        let channels = sequences_to_channels(&seqs, &weights, &map, 2, fft_size);
        // At position 0: A has weight 0.7 (from seq1), C has weight 0.3 (from seq2)
        assert!((channels[0][0].re - 0.7).abs() < 1e-10);
        assert!((channels[1][0].re - 0.3).abs() < 1e-10);
    }
}

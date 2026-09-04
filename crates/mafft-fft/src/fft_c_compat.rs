//! Bit-for-bit port of MAFFT's C FFT (`mafft-upstream/core/fft.c`).
//!
//! Reproduces C's `fft()` exactly, including:
//! - Same Cooley-Tukey radix-2 algorithm in iterative in-place form.
//! - Same `make_sintbl` / `make_bitrev` precomputed tables.
//! - Same forward/inverse semantics: forward applies `1/n` post-scale,
//!   inverse leaves data unscaled.
//! - Same multiply-add operation order in the butterfly:
//!     `dR = s * x[ik].I + c * x[ik].R`
//!     `dI = c * x[ik].I - s * x[ik].R`
//!
//! Used by `multichannel_correlate` so that the cross-correlation
//! peaks land at byte-identical lags to C MAFFT 7.526. `rustfft` uses
//! a different scheduling that gives 1-ULP-different correlation
//! values — for most scoring matrices this is invisible, but for
//! flat-landscape matrices like `--tm 200` and `--bl 50` it tips
//! anchor selection and produces multi-column alignment differences.
//!
//! References: `fft.c` lines 7-126 in MAFFT 7.526.

use num_complex::Complex64;
use std::f64::consts::PI;

/// Compute `sintbl[i]` for `i in 0..n + n/4`. Mirrors `make_sintbl`
/// in `fft.c:7-26`.
///
/// The table caches `sin(PI * i / (n/2))` for `i in 0..n` followed by
/// the negative half — used to look up sin/cos within the butterfly
/// loop without per-iteration `sin`/`cos` calls.
fn make_sintbl(n: usize, sintbl: &mut [f64]) {
    let n2 = n / 2;
    let n4 = n / 4;
    let n8 = n / 8;
    let mut t = (PI / n as f64).sin();
    let mut dc = 2.0 * t * t;
    let mut ds = (dc * (2.0 - dc)).sqrt();
    t = 2.0 * dc;
    let mut c = 1.0;
    sintbl[n4] = 1.0;
    let mut s = 0.0;
    sintbl[0] = 0.0;
    for i in 1..n8 {
        c -= dc;
        dc += t * c;
        s += ds;
        ds -= t * s;
        sintbl[i] = s;
        sintbl[n4 - i] = c;
    }
    if n8 != 0 {
        sintbl[n8] = (0.5_f64).sqrt();
    }
    for i in 0..n4 {
        sintbl[n2 - i] = sintbl[i];
    }
    for i in 0..(n2 + n4) {
        sintbl[i + n2] = -sintbl[i];
    }
}

/// Compute `bitrev[i]` for `i in 0..n`. Mirrors `make_bitrev` in
/// `fft.c:30-42`.
fn make_bitrev(n: usize, bitrev: &mut [usize]) {
    let n2 = n / 2;
    let mut i = 0;
    let mut j = 0;
    loop {
        bitrev[i] = j;
        i += 1;
        if i >= n {
            break;
        }
        let mut k = n2;
        while k <= j {
            j -= k;
            k /= 2;
        }
        j += k;
    }
}

/// In-place radix-2 Cooley-Tukey FFT, exact bit-for-bit port of C's
/// `fft()` from `fft.c:45-126`.
///
/// `inverse=false` does the forward DFT and applies the `1/n`
/// post-scale (matching C's `if (!inverse) for(i=0;i<n;i++) x[i] /= n`).
/// `inverse=true` does the inverse DFT without scaling (so the result
/// is `n * true_IDFT(input)` — same as C).
///
/// `n` (= `x.len()`) must be a power of 2.
pub fn fft_inplace(x: &mut [Complex64], inverse: bool) {
    let n = x.len();
    if n == 0 || (n & (n - 1)) != 0 {
        // Non-power-of-2 not supported by this port (matches C's
        // contract — Falign always rounds up to a power of 2 first).
        return;
    }
    let n4 = n / 4;

    // Precompute sintbl and bitrev (per-call; could be cached but the
    // overhead is small relative to the FFT itself).
    let mut sintbl = vec![0.0f64; n + n4];
    let mut bitrev = vec![0usize; n];
    make_sintbl(n, &mut sintbl);
    make_bitrev(n, &mut bitrev);

    // Bit-reverse permutation. Swap when i < j to avoid double-swap.
    for i in 0..n {
        let j = bitrev[i];
        if i < j {
            x.swap(i, j);
        }
    }

    // Iterative butterfly. `k` doubles each outer iteration: 1, 2, 4, ..., n/2.
    let mut k = 1usize;
    while k < n {
        let k2 = k + k;
        let d = n / k2;
        let mut h = 0usize;
        for j in 0..k {
            let c = sintbl[h + n4];
            let s = if inverse { -sintbl[h] } else { sintbl[h] };
            let mut i = j;
            while i < n {
                let ik = i + k;
                let xr = x[i].re;
                let xi = x[i].im;
                let xikr = x[ik].re;
                let xiki = x[ik].im;
                // Match C's exact multiply-add order (the reference build does NOT fuse;
                // these as `fmadd`):
                //   dR = s * x[ik].I + c * x[ik].R   →  fma(s, xiki, c * xikr)
                //   dI = c * x[ik].I - s * x[ik].R   →  fma(-s, xikr, c * xiki)
                //                                    or  fma(c, xiki, -(s * xikr))
                let d_r = s * xiki + (c * xikr);
                let d_i = c * xiki + (-(s * xikr));
                x[ik].re = xr - d_r;
                x[i].re = xr + d_r;
                x[ik].im = xi - d_i;
                x[i].im = xi + d_i;
                i += k2;
            }
            h += d;
        }
        k = k2;
    }

    // Forward applies the 1/n post-scale.
    if !inverse {
        let inv_n = 1.0 / n as f64;
        for v in x.iter_mut() {
            v.re *= inv_n;
            v.im *= inv_n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_inverse_roundtrip_identity() {
        // FFT(input) followed by IFFT(...) returns input scaled by 1/n
        // because C's forward applies 1/n and inverse doesn't.
        // Two forward applications would give 1/n², etc. Roundtrip
        // forward+inverse on a single signal should give input/1 = input.
        let n = 16;
        let signal: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect();
        let mut x = signal.clone();
        fft_inplace(&mut x, false);
        // After forward: x = (1/n) * DFT(signal)
        // To roundtrip, we need x = (1/n) * (1/n) * inv_DFT(DFT(signal)) = signal / n
        // Apply inverse: x = inv_DFT(x) (unnormalized) = N * true_inv_DFT(x)
        //               = N * true_inv_DFT((1/N) * DFT(signal))
        //               = N * (1/N) * signal = signal
        fft_inplace(&mut x, true);
        for (orig, got) in signal.iter().zip(x.iter()) {
            assert!((orig.re - got.re).abs() < 1e-10,
                "roundtrip failed: orig={} got={}", orig.re, got.re);
            assert!((orig.im - got.im).abs() < 1e-10);
        }
    }

    #[test]
    fn dc_signal_concentrates_at_zero_freq() {
        // Forward FFT of a constant signal should give 1.0 at index 0
        // (after the 1/n scale) and zero elsewhere.
        let n = 8;
        let mut x: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); n];
        fft_inplace(&mut x, false);
        // X[0] = sum(x[i]) / n = n / n = 1.0
        // X[k] = 0 for k != 0
        assert!((x[0].re - 1.0).abs() < 1e-10, "DC bin should be 1.0");
        for i in 1..n {
            assert!(x[i].re.abs() < 1e-10, "freq bin {} should be 0", i);
        }
    }
}

use num_complex::Complex64;
use rustfft::FftPlanner;

/// Compute the cross-correlation of two real-valued signals using FFT.
///
/// Returns a vector of length `n` (the FFT size, which is the next power
/// of 2 >= `max(a.len(), b.len()) * 2`). The correlation at lag `k` is
/// in `result[k]` for positive lags and `result[n-k]` for negative lags.
///
/// This replaces the C pattern of:
/// 1. Zero-pad sequences into `Fukusosuu` arrays
/// 2. Forward FFT both
/// 3. `calcNaiseki` (complex conjugate multiply)
/// 4. Inverse FFT the product
pub fn cross_correlate(a: &[f64], b: &[f64]) -> Vec<f64> {
    let min_size = a.len() + b.len();
    let n = min_size.next_power_of_two();

    let mut planner = FftPlanner::new();
    let fft_forward = planner.plan_fft_forward(n);
    let fft_inverse = planner.plan_fft_inverse(n);

    // Zero-pad into complex arrays
    let mut fa: Vec<Complex64> = a.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    fa.resize(n, Complex64::new(0.0, 0.0));

    let mut fb: Vec<Complex64> = b.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    fb.resize(n, Complex64::new(0.0, 0.0));

    // Forward FFT
    fft_forward.process(&mut fa);
    fft_forward.process(&mut fb);

    // Complex conjugate multiply (calcNaiseki in C):
    // correlation in frequency domain = conj(A) * B
    let mut product: Vec<Complex64> = fa
        .iter()
        .zip(fb.iter())
        .map(|(a, b)| a.conj() * b)
        .collect();

    // Inverse FFT
    fft_inverse.process(&mut product);

    // rustfft's inverse FFT does NOT normalize by 1/n, so we do it
    let scale = 1.0 / n as f64;
    product.iter().map(|c| c.re * scale).collect()
}

/// Compute the inner product (dot product) of two complex vectors.
///
/// This is the element-wise complex conjugate multiplication, equivalent
/// to the C `calcNaiseki` function: `result = conj(x) * y`.
pub fn inner_product(x: Complex64, y: Complex64) -> Complex64 {
    // C code: value.R = x.R * y.R + x.I * y.I
    //         value.I = -x.R * y.I + x.I * y.R
    // This is exactly conj(x) * y
    x.conj() * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inner_product_matches_c_calcnaiseki() {
        let x = Complex64::new(3.0, 4.0);
        let y = Complex64::new(1.0, 2.0);
        let result = inner_product(x, y);

        // conj(3+4i) * (1+2i) = (3-4i)(1+2i) = 3+6i-4i-8i² = 11+2i
        assert!((result.re - 11.0).abs() < 1e-10);
        assert!((result.im - 2.0).abs() < 1e-10);
    }

    #[test]
    fn autocorrelation_peak_at_zero() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let corr = cross_correlate(&signal, &signal);

        // Auto-correlation should peak at lag 0
        let max_idx = corr
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 0, "autocorrelation should peak at lag 0");
    }

    #[test]
    fn correlation_detects_shift() {
        // `a` has a peak at position 5, `b` has the same peak at position 2.
        // Cross-correlation conj(a)*b peaks at lag where a[i+lag] aligns b[i].
        // Shift = 5 - 2 = 3, but in circular correlation this appears at
        // index n - 3 (negative wrap) since conj(a)*b convention.
        let n = 16;
        let mut a = vec![0.0f64; n];
        let mut b = vec![0.0f64; n];
        a[5] = 10.0;
        a[6] = 5.0;
        b[2] = 10.0;
        b[3] = 5.0;

        let corr = cross_correlate(&a, &b);
        let fft_size = corr.len(); // next power of 2 >= 32

        let peak_idx = corr
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        // The lag is 3 positions, wrapped as n-3 in circular correlation
        let lag = if peak_idx > fft_size / 2 {
            peak_idx as i32 - fft_size as i32
        } else {
            peak_idx as i32
        };
        assert_eq!(
            lag.abs(),
            3,
            "should detect shift magnitude of 3, got {lag}"
        );
    }
}

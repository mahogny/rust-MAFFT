/// Verify our `fft_c_compat::fft_inplace` produces bit-identical
/// output to C MAFFT's `fft()` (`mafft-upstream/core/fft.c`). Used to
/// confirm the FFT port for the `--tm 200` (FFT) anchor-selection
/// fix in TODO §5.
use num_complex::Complex64;
use std::sync::Mutex;

use mafft_fft::fft_c_compat::fft_inplace;

static C_MUTEX: Mutex<()> = Mutex::new(());

#[test]
fn fft_port_matches_c_forward_dc() {
    let _guard = C_MUTEX.lock().unwrap();
    let n = 16;
    // Constant input: forward FFT should give 1.0 at index 0, 0 elsewhere.
    let signal: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); n];

    let mut rust_x = signal.clone();
    fft_inplace(&mut rust_x, /* inverse = */ false);

    let mut c_x: Vec<mafft_sys::Fukusosuu> = signal
        .iter()
        .map(|c| mafft_sys::Fukusosuu { R: c.re, I: c.im })
        .collect();
    unsafe {
        mafft_sys::fft(n as i32, c_x.as_mut_ptr(), 0);
    }

    for i in 0..n {
        assert!(
            (rust_x[i].re - c_x[i].R).abs() < 1e-15,
            "[{i}] re: rust={} c={}",
            rust_x[i].re,
            c_x[i].R
        );
        assert!(
            (rust_x[i].im - c_x[i].I).abs() < 1e-15,
            "[{i}] im: rust={} c={}",
            rust_x[i].im,
            c_x[i].I
        );
    }
    unsafe {
        mafft_sys::fft(0, std::ptr::null_mut(), 1);
    }
}

#[test]
fn fft_port_matches_c_random_input() {
    let _guard = C_MUTEX.lock().unwrap();
    // Use a deterministic "random" input that exercises real and
    // imaginary parts and varied magnitudes.
    let n = 64;
    let signal: Vec<Complex64> = (0..n)
        .map(|i| {
            let r = (i as f64 * 0.31415926).sin() * 7.0 + (i as f64).sqrt();
            let im = (i as f64 * 0.7).cos() * 3.0 - 0.5;
            Complex64::new(r, im)
        })
        .collect();

    // Forward FFT.
    let mut rust_fwd = signal.clone();
    fft_inplace(&mut rust_fwd, false);

    let mut c_fwd: Vec<mafft_sys::Fukusosuu> = signal
        .iter()
        .map(|c| mafft_sys::Fukusosuu { R: c.re, I: c.im })
        .collect();
    unsafe {
        mafft_sys::fft(n as i32, c_fwd.as_mut_ptr(), 0);
    }

    // Tolerance: a length-n radix-2 FFT does O(n log n) FP ops, each
    // adding ~1 ULP of error. For n=64 and |x| ≤ ~50 (so ULP(x) ≈ 1.1e-14),
    // the cross-platform difference budget is roughly n × ULP(max) ≈ 7e-13.
    // 1e-12 gives a small safety margin while still catching any genuine
    // algorithmic drift between our FFT port and C's `fft.c`.
    const FFT_ABS_TOL: f64 = 1e-12;

    let mut max_diff = 0.0_f64;
    for i in 0..n {
        let dr = (rust_fwd[i].re - c_fwd[i].R).abs();
        let di = (rust_fwd[i].im - c_fwd[i].I).abs();
        max_diff = max_diff.max(dr).max(di);
        assert!(
            dr < FFT_ABS_TOL && di < FFT_ABS_TOL,
            "forward FFT [{i}] differs: rust=({}, {}) c=({}, {}) dr={dr:e} di={di:e}",
            rust_fwd[i].re,
            rust_fwd[i].im,
            c_fwd[i].R,
            c_fwd[i].I
        );
    }
    eprintln!("Forward max diff: {max_diff:e}");

    // Inverse FFT.
    let mut rust_inv = signal.clone();
    fft_inplace(&mut rust_inv, true);

    let mut c_inv: Vec<mafft_sys::Fukusosuu> = signal
        .iter()
        .map(|c| mafft_sys::Fukusosuu { R: c.re, I: c.im })
        .collect();
    unsafe {
        mafft_sys::fft(-(n as i32), c_inv.as_mut_ptr(), 0);
    }

    let mut max_diff_inv = 0.0_f64;
    for i in 0..n {
        let dr = (rust_inv[i].re - c_inv[i].R).abs();
        let di = (rust_inv[i].im - c_inv[i].I).abs();
        max_diff_inv = max_diff_inv.max(dr).max(di);
        assert!(
            dr < FFT_ABS_TOL && di < FFT_ABS_TOL,
            "inverse FFT [{i}] differs: rust=({}, {}) c=({}, {}) dr={dr:e} di={di:e}",
            rust_inv[i].re,
            rust_inv[i].im,
            c_inv[i].R,
            c_inv[i].I
        );
    }
    eprintln!("Inverse max diff: {max_diff_inv:e}");

    unsafe {
        mafft_sys::fft(0, std::ptr::null_mut(), 1);
    }
}

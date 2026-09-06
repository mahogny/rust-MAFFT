/// Amino acid biochemical properties from Miyata et al.
///
/// Used for FFT-based homology detection (sequence → numerical vector).

/// Raw polarity values (order: ARNDCQEGHILKMFPSTWYVBZX.-J, first 20 used).
pub static POLARITY: [f64; 20] = [
    8.1, 10.5, 11.6, 13.0, 5.5, 10.5, 12.3, 9.0, 10.4, 5.2, 4.9, 11.3, 5.7, 5.2, 8.0, 9.2, 8.6,
    5.4, 6.2, 5.9,
];

/// Raw volume values (order: ARNDCQEGHILKMFPSTWYVBZX.-J, first 20 used).
pub static VOLUME: [f64; 20] = [
    31.0, 124.0, 56.0, 54.0, 55.0, 85.0, 83.0, 3.0, 96.0, 111.0, 111.0, 119.0, 105.0, 132.0, 32.5,
    32.0, 61.0, 170.0, 136.0, 84.0,
];

/// Z-score normalized polarity values.
pub fn normalized_polarity() -> [f64; 20] {
    z_normalize(&POLARITY)
}

/// Z-score normalized volume values.
pub fn normalized_volume() -> [f64; 20] {
    z_normalize(&VOLUME)
}

fn z_normalize(values: &[f64; 20]) -> [f64; 20] {
    let n = 20.0;
    let mean: f64 = values.iter().sum::<f64>() / n;
    let variance: f64 = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let sd = variance.sqrt();

    let mut result = [0.0; 20];
    for i in 0..20 {
        result[i] = (values[i] - mean) / sd;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_polarity_has_zero_mean() {
        let pol = normalized_polarity();
        let mean: f64 = pol.iter().sum::<f64>() / 20.0;
        assert!(mean.abs() < 1e-10, "mean = {mean}");
    }

    #[test]
    fn normalized_volume_has_zero_mean() {
        let vol = normalized_volume();
        let mean: f64 = vol.iter().sum::<f64>() / 20.0;
        assert!(mean.abs() < 1e-10, "mean = {mean}");
    }

    #[test]
    fn normalized_polarity_has_unit_variance() {
        let pol = normalized_polarity();
        let variance: f64 = pol.iter().map(|x| x * x).sum::<f64>() / 20.0;
        assert!((variance - 1.0).abs() < 1e-10, "variance = {variance}");
    }
}

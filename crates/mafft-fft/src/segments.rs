/// Segment detection: sliding-window scoring to find alignable regions.
///
/// Ports the C `alignableReagion()` function from fftFunctions.c.

/// Parameters controlling segment detection.
#[derive(Debug, Clone)]
pub struct SegmentParams {
    /// FFT window size for sliding-window scoring.
    /// Default: 20 for protein, 100 for DNA (FFT_WINSIZE_P / FFT_WINSIZE_D).
    pub window_size: usize,
    /// Threshold as a fraction of 600 * window_size.
    /// Default: fftThreshold / 100.0 * 600.0 * window_size.
    pub threshold: f64,
    /// Maximum segment length before forced split.
    pub max_segment_size: usize,
}

impl SegmentParams {
    pub fn protein() -> Self {
        Self {
            window_size: 20,
            threshold: 80.0 / 100.0 * 600.0 * 20.0,
            max_segment_size: 150,
        }
    }

    pub fn dna() -> Self {
        Self {
            window_size: 100,
            threshold: 80.0 / 100.0 * 600.0 * 100.0,
            max_segment_size: 150,
        }
    }

    /// Custom parameters with a given threshold percentage (0-100).
    pub fn with_threshold(mut self, pct: f64) -> Self {
        self.threshold = pct / 100.0 * 600.0 * self.window_size as f64;
        self
    }
}

/// A detected alignable segment.
#[derive(Debug, Clone)]
pub struct AlignableSegment {
    /// Start position in the scoring array.
    pub start: usize,
    /// End position in the scoring array.
    pub end: usize,
    /// Center position (midpoint + half window).
    pub center: usize,
    /// Cumulative score over the segment.
    pub score: f64,
    /// Whether this segment was forcibly split (exceeded max size).
    pub skip_forward: bool,
    /// Whether the previous segment was forcibly split.
    pub skip_backward: bool,
}

/// Detect alignable segments from a per-position scoring array.
///
/// `site_scores` contains the per-position match score between two groups
/// of sequences. This function applies a sliding window and returns segments
/// where the windowed score exceeds the threshold.
///
/// Ports the C `alignableReagion()` logic.
pub fn alignable_segments(site_scores: &[f64], params: &SegmentParams) -> Vec<AlignableSegment> {
    let len = site_scores.len();
    if len <= params.window_size {
        return Vec::new();
    }

    let mut segments = Vec::new();

    // Initial window sum
    let mut score: f64 = site_scores[..params.window_size].iter().sum();

    let mut status = false;
    let mut start_tmp = 0usize;
    let mut length = 0usize;
    let mut cumscore = 0.0f64;
    let mut prev_skip_forward = false;

    for i in 1..len.saturating_sub(params.window_size) {
        score = score - site_scores[i - 1] + site_scores[i + params.window_size - 1];

        if score > params.threshold {
            if !status {
                status = true;
                start_tmp = i;
                length = 0;
                cumscore = 0.0;
            }
            length += 1;
            cumscore += score;
        }

        if score <= params.threshold || length > params.max_segment_size {
            if status {
                if length > params.window_size {
                    let skip_forward = length > params.max_segment_size;
                    segments.push(AlignableSegment {
                        start: start_tmp,
                        end: i,
                        center: (start_tmp + i + params.window_size) / 2,
                        score: cumscore,
                        skip_forward,
                        skip_backward: prev_skip_forward,
                    });
                    prev_skip_forward = skip_forward;
                }
                length = 0;
                cumscore = 0.0;
                status = false;
                start_tmp = i;
            }
        }
    }

    // Flush final segment
    let i = len.saturating_sub(params.window_size);
    if status && length > params.window_size {
        segments.push(AlignableSegment {
            start: start_tmp,
            end: i,
            center: (start_tmp + i + params.window_size) / 2,
            score: cumscore,
            skip_forward: false,
            skip_backward: prev_skip_forward,
        });
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_segments_below_threshold() {
        let scores = vec![0.0; 100];
        let params = SegmentParams::protein();
        let segs = alignable_segments(&scores, &params);
        assert!(segs.is_empty());
    }

    #[test]
    fn detects_high_scoring_region() {
        let mut scores = vec![0.0; 200];
        // Create a high-scoring region from position 50 to 150
        for i in 50..150 {
            scores[i] = 1000.0;
        }
        let params = SegmentParams {
            window_size: 10,
            threshold: 5000.0,
            max_segment_size: 150,
        };
        let segs = alignable_segments(&scores, &params);
        assert!(!segs.is_empty(), "should detect at least one segment");
        assert!(segs[0].start >= 40 && segs[0].start <= 60);
    }

    #[test]
    fn short_input_returns_empty() {
        let scores = vec![1000.0; 5];
        let params = SegmentParams::protein();
        let segs = alignable_segments(&scores, &params);
        assert!(segs.is_empty());
    }
}

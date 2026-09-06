//! Cross-platform deterministic port of FreeBSD/macOS `qsort` (Bentley-
//! McIlroy three-way partitioning quicksort), used to keep tie-break
//! ordering byte-identical to C MAFFT 7.526 *as compiled on macOS*
//! regardless of the host platform the Rust binary runs on.
//!
//! Why this exists: `libc::qsort` delegates to the host C library. macOS
//! uses BSD qsort, Linux uses glibc qsort (merge-sort variant), Windows
//! uses MSVC's qsort. For *truly tied* elements (same score / selfscore /
//! orilen — only when the input contains exactly-duplicate sequences),
//! these implementations disagree on the final order. That makes the
//! Rust binary's output platform-dependent, which fails our parity tests
//! on Linux CI.
//!
//! Algorithm reference: FreeBSD `lib/libc/stdlib/qsort.c` (in turn from
//! Bentley & McIlroy, "Engineering a Sort Function", 1993). Faithfully
//! preserves swap order so tied elements move identically to the BSD
//! reference implementation.

use std::cmp::Ordering;

/// Sort `slice` in place using FreeBSD's qsort algorithm. The closure
/// `cmp` returns the standard `Ordering` (Less, Equal, Greater) used by
/// Rust's `sort_by`.
pub fn bsd_qsort<T, F>(slice: &mut [T], mut cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    let n = slice.len();
    qsort_impl(slice, 0, n, &mut cmp);
}

/// Median of three indices `a`, `b`, `c` — returns the index whose value
/// is the median under `cmp`. Mirrors FreeBSD's `med3` macro.
fn med3<T, F>(slice: &[T], a: usize, b: usize, c: usize, cmp: &mut F) -> usize
where
    F: FnMut(&T, &T) -> Ordering,
{
    let ab = cmp(&slice[a], &slice[b]);
    let bc = cmp(&slice[b], &slice[c]);
    if ab == Ordering::Less {
        if bc == Ordering::Less {
            b // a < b < c
        } else {
            // a < b, b >= c
            if cmp(&slice[a], &slice[c]) == Ordering::Less {
                c // a < c <= b
            } else {
                a // c <= a < b
            }
        }
    } else {
        if bc == Ordering::Greater {
            b // a >= b > c
        } else {
            // a >= b, b <= c
            if cmp(&slice[a], &slice[c]) == Ordering::Greater {
                c // b <= c < a
            } else {
                a // b <= a <= c
            }
        }
    }
}

/// In-place block swap of `n` elements starting at `a` with `n` elements
/// starting at `b`. Mirrors FreeBSD's `vecswap`.
fn vecswap<T>(slice: &mut [T], a: usize, b: usize, n: usize) {
    for i in 0..n {
        slice.swap(a + i, b + i);
    }
}

fn insertion_sort<T, F>(slice: &mut [T], start: usize, end: usize, cmp: &mut F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    for i in (start + 1)..end {
        let mut j = i;
        while j > start && cmp(&slice[j - 1], &slice[j]) == Ordering::Greater {
            slice.swap(j - 1, j);
            j -= 1;
        }
    }
}

fn qsort_impl<T, F>(slice: &mut [T], mut start: usize, mut len: usize, cmp: &mut F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    loop {
        if len < 7 {
            insertion_sort(slice, start, start + len, cmp);
            return;
        }

        // Pivot selection: median-of-3 (or median-of-9 for n > 40).
        let pl0 = start;
        let pm0 = start + len / 2;
        let pn0 = start + len - 1;
        let pm = if len > 7 {
            if len > 40 {
                let d = len / 8;
                let pl_m = med3(slice, pl0, pl0 + d, pl0 + 2 * d, cmp);
                let pm_m = med3(slice, pm0 - d, pm0, pm0 + d, cmp);
                let pn_m = med3(slice, pn0 - 2 * d, pn0 - d, pn0, cmp);
                med3(slice, pl_m, pm_m, pn_m, cmp)
            } else {
                med3(slice, pl0, pm0, pn0, cmp)
            }
        } else {
            pm0
        };

        // Move pivot to slice[start].
        slice.swap(start, pm);

        // 3-way partition. pa..pb is the "equal to pivot" region growing
        // from the left; pd..pc is the same growing from the right.
        let mut pa = start + 1;
        let mut pb = start + 1;
        let mut pc = start + len - 1;
        let mut pd = start + len - 1;
        let mut swap_cnt = false;

        loop {
            // Scan from left.
            while pb <= pc {
                let c = cmp(&slice[pb], &slice[start]);
                if c == Ordering::Greater {
                    break;
                }
                if c == Ordering::Equal {
                    swap_cnt = true;
                    slice.swap(pa, pb);
                    pa += 1;
                }
                pb += 1;
            }
            // Scan from right.
            while pb <= pc {
                let c = cmp(&slice[pc], &slice[start]);
                if c == Ordering::Less {
                    break;
                }
                if c == Ordering::Equal {
                    swap_cnt = true;
                    slice.swap(pc, pd);
                    if pd == 0 {
                        break;
                    }
                    pd -= 1;
                }
                if pc == 0 {
                    break;
                }
                pc -= 1;
            }
            if pb > pc {
                break;
            }
            slice.swap(pb, pc);
            swap_cnt = true;
            pb += 1;
            if pc == 0 {
                break;
            }
            pc -= 1;
        }

        if !swap_cnt {
            // No swaps in partition → already nearly sorted. Insertion sort.
            insertion_sort(slice, start, start + len, cmp);
            return;
        }

        // vecswap: move the "equal to pivot" elements from the ends into
        // the middle so the array is laid out [< pivot | = pivot | > pivot].
        let pn_end = start + len;
        let r1 = (pa - start).min(pb - pa);
        if r1 > 0 {
            vecswap(slice, start, pb - r1, r1);
        }
        let r2 = (pd - pc).min(pn_end - pd - 1);
        if r2 > 0 {
            vecswap(slice, pb, pn_end - r2, r2);
        }

        // Recurse on the smaller partition, iterate on the larger
        // (tail-call elimination to keep stack usage O(log n)).
        let left_len = pb - pa;
        let right_len = pd - pc;
        if left_len > 1 {
            qsort_impl(slice, start, left_len, cmp);
        }
        if right_len > 1 {
            start = pn_end - right_len;
            len = right_len;
        } else {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_simple_ints() {
        let mut v = vec![5, 3, 8, 1, 4, 7, 2, 6];
        bsd_qsort(&mut v, |a, b| a.cmp(b));
        assert_eq!(v, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn sorts_with_duplicates() {
        let mut v = vec![3, 1, 2, 3, 1, 2, 3, 1];
        bsd_qsort(&mut v, |a, b| a.cmp(b));
        // All sorted, exact tie-break order doesn't matter for primitive ints
        // since equal values are indistinguishable.
        for i in 1..v.len() {
            assert!(v[i - 1] <= v[i]);
        }
    }

    #[test]
    fn handles_small_input() {
        for n in 0..=6 {
            let mut v: Vec<i32> = (0..n).rev().collect();
            bsd_qsort(&mut v, |a, b| a.cmp(b));
            assert_eq!(v, (0..n).collect::<Vec<i32>>());
        }
    }

    #[test]
    fn matches_macos_qsort_on_partial_tie_pattern() {
        // Mimic the parttree CALL 1 input layout: 36 entries, almost all
        // distinct primary keys, with two entries at indices 0 and 34
        // tied. BSD qsort's median-of-3 + 3-way partition swaps the tied
        // pair so the originally-index-34 element ends up first.
        //
        // Each element is `(key, tag)`; `key` is the primary sort key,
        // `tag` is the "numinseq"-equivalent identity we use to verify
        // post-sort ordering.
        let mut v: Vec<(f64, usize)> = (0..36)
            .map(|i| {
                if i == 0 || i == 34 {
                    (0.0, i)
                } else {
                    (i as f64, i)
                }
            })
            .collect();
        // Replicate pick_reference swap: scan for max selfscore, swap to
        // index 0. Here we pre-set the tied entries; just sort.
        bsd_qsort(&mut v, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
        });
        // Both tied entries should be at the front.
        assert_eq!(v[0].0, 0.0);
        assert_eq!(v[1].0, 0.0);
        // BSD qsort's specific partition order: the originally-index-34
        // element comes first, originally-index-0 second.
        assert_eq!(v[0].1, 34, "BSD qsort puts originally-index-34 first");
        assert_eq!(v[1].1, 0, "BSD qsort puts originally-index-0 second");
    }
}

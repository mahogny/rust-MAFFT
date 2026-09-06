//! Cell-level FFI parity harness for C's `insertnewgaps` and the
//! supporting `findcommongaps`/`adjustgapmap`/`restorecommongaps`/
//! `findnewgaps` chain (`addfunctions.c:445-650, 327-456, 386-444,
//! 1453-1521`). Used to close R-6's `--add` multi-added divergence:
//! rust's `build_other_post_restore_row` does flat '-' padding while
//! C's `insertnewgaps` runs `profilealignment` at each new-merge-gap
//! region, sometimes compressing the alignment by 1+ columns.
//!
//! This harness sets up hand-crafted scenarios, runs the full C chain
//! via FFI, and reports the exact output. Used to:
//!   1. Discover the input shapes that trigger profilealignment.
//!   2. Capture C's reference output for the rust port to match.
//!   3. Diff rust's flat-padding output against C's compressed output.
//!
//! Run with:
//!     cargo test -p mafft-core --release --test \
//!         cross_validate_insertnewgaps -- --nocapture

use std::os::raw::{c_char, c_int};

unsafe fn init_c_protein() {
    unsafe {
        mafft_sys::initglobalvariables();
        std::ptr::addr_of_mut!(mafft_sys::ppenalty).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::ppenalty_ex).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::poffset).write(0);
        std::ptr::addr_of_mut!(mafft_sys::kimuraR).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::pamN).write(mafft_sys::NOTSPECIFIED);
        std::ptr::addr_of_mut!(mafft_sys::dorp).write(b'p' as i32);
        std::ptr::addr_of_mut!(mafft_sys::scoremtx).write(1);
        std::ptr::addr_of_mut!(mafft_sys::nblosum).write(62);
        std::ptr::addr_of_mut!(mafft_sys::fmodel).write(0);
        let seq_data = b"ACDEFGHIKLMNPQRSTVWY\0";
        let mut seq_ptr = seq_data.as_ptr() as *mut i8;
        let seq_arr: *mut *mut i8 = &mut seq_ptr;
        mafft_sys::constants(1, seq_arr);
    }
}

unsafe fn set_row(mtx: *mut *mut c_char, i: usize, content: &[u8]) {
    unsafe {
        let row = *mtx.add(i);
        for (k, &b) in content.iter().enumerate() {
            *row.add(k) = b as c_char;
        }
        *row.add(content.len()) = 0;
    }
}

unsafe fn read_row(mtx: *mut *mut c_char, i: usize) -> Vec<u8> {
    unsafe {
        let row = *mtx.add(i);
        let mut out = Vec::new();
        let mut k = 0;
        loop {
            let c = *row.add(k);
            if c == 0 {
                break;
            }
            out.push(c as u8);
            k += 1;
        }
        out
    }
}

/// Compute rust's flat-padding equivalent of `insertnewgaps` for
/// OTHER rows. Mirrors `progressive.rs::build_other_post_restore_row`
/// behavior. Used to diff against C's output.
fn rust_flat_padding(
    other_pre: &[u8],
    anchor_positions: &[usize],
    gap_cols_before: &[Vec<usize>],
    new_merge_gap_set: &std::collections::HashSet<usize>,
    post_merge_width: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        post_merge_width + gap_cols_before.iter().map(|v| v.len()).sum::<usize>(),
    );
    let mut s = 0usize;
    for q in 0..post_merge_width {
        if new_merge_gap_set.contains(&q) {
            out.push(b'-');
        } else {
            for &k_pre in &gap_cols_before[s] {
                out.push(other_pre.get(k_pre).copied().unwrap_or(b'-'));
            }
            out.push(other_pre.get(anchor_positions[s]).copied().unwrap_or(b'-'));
            s += 1;
        }
    }
    for &k_pre in &gap_cols_before[anchor_positions.len()] {
        out.push(other_pre.get(k_pre).copied().unwrap_or(b'-'));
    }
    out
}

/// A scenario for the parity harness: pre-merge state, post-merge
/// state (with `=` markers), and the group assignments. Drives both
/// the C chain and the rust port for comparison.
struct Scenario {
    name: &'static str,
    /// Pre-strip state of the existing alignment. Index = row, value
    /// = bytes. group2 (added) is given UNGAPPED here for clarity.
    pre_merge: Vec<(usize, Vec<u8>)>,
    /// Post-merge state for active rows ONLY. group1 has `=` markers
    /// where the merge DP inserted new gaps; group2 has its inserted
    /// residues with `-` at positions where group1 had pre-strip
    /// residues. Index in the tuple is the global row id.
    post_merge_active: Vec<(usize, Vec<u8>)>,
    /// Indices in `existing_grp` (= ex1) and `new_grp` (= ex2).
    existing_grp: Vec<usize>,
    new_grp: Vec<usize>,
    /// Row indices for OTHER (alreadyaligned[i]=1 && not in active).
    /// Their pre-merge content stays in `pre_merge`.
    other_grp: Vec<usize>,
}

/// Run both C and the rust port `apply_c_insertnewgaps`, compare
/// outputs row-by-row. Returns (c_outputs, rust_outputs).
unsafe fn run_c_chain(sc: &Scenario) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let njob = (sc.pre_merge.len()) as c_int;
    let alloclen: c_int = 1024;

    // Build aseq[njob][alloclen]. Initialize to post-merge state for
    // active rows; pre-merge state for OTHER rows (untouched by the
    // merge).
    let aseq = unsafe { mafft_sys::AllocateCharMtx(njob, alloclen) };
    for (i, content) in &sc.pre_merge {
        // OTHER rows: pre-merge.
        unsafe {
            set_row(aseq, *i, content);
        }
    }
    for (i, content) in &sc.post_merge_active {
        // Active rows: overwrite with post-merge.
        unsafe {
            set_row(aseq, *i, content);
        }
    }

    // For findcommongaps: build mseq1 = pre-strip group1 rows
    // (snapshot). C's call site:
    //   findcommongaps( clus1, mseq1, gapmap ) on pre-strip.
    let mseq1_prestrip =
        unsafe { mafft_sys::AllocateCharMtx(sc.existing_grp.len() as c_int, alloclen) };
    for (k, &i) in sc.existing_grp.iter().enumerate() {
        let content = sc
            .pre_merge
            .iter()
            .find(|(idx, _)| *idx == i)
            .map(|(_, v)| v.clone())
            .unwrap();
        unsafe {
            set_row(mseq1_prestrip, k, &content);
        }
    }
    let gapmap = unsafe { mafft_sys::AllocateIntVec(alloclen) };
    for k in 0..alloclen {
        unsafe {
            *gapmap.add(k as usize) = 0;
        }
    }
    unsafe {
        mafft_sys::findcommongaps(sc.existing_grp.len() as c_int, mseq1_prestrip, gapmap);
    }

    // Now we'd commongappick group1 and run the actual merge to get
    // post-merge state. We SKIP both steps and use the user-supplied
    // post-merge content directly (so the test is deterministic).

    // adjustgapmap: walks newlen of post-merge mseq1[0] with `=` chars.
    // For our test, post-merge active group1[0] is what was supplied.
    let mseq1_post =
        unsafe { mafft_sys::AllocateCharMtx(sc.existing_grp.len() as c_int, alloclen) };
    for (k, &i) in sc.existing_grp.iter().enumerate() {
        let content = sc
            .post_merge_active
            .iter()
            .find(|(idx, _)| *idx == i)
            .map(|(_, v)| v.clone())
            .unwrap();
        unsafe {
            set_row(mseq1_post, k, &content);
        }
    }
    let post_merge_width = sc.post_merge_active[0].1.len() as c_int;
    let gapmaplen = post_merge_width + 1;
    unsafe {
        mafft_sys::adjustgapmap(post_merge_width, gapmap, *mseq1_post.add(0));
    }
    let _ = gapmaplen;

    // restorecommongaps: expands active rows by inserting common-gap chars.
    // Also updates gapmap to post-restore indexing.
    let ex1 = unsafe { mafft_sys::AllocateIntVec(njob) };
    let ex2 = unsafe { mafft_sys::AllocateIntVec(njob) };
    for (k, &i) in sc.existing_grp.iter().enumerate() {
        unsafe {
            *ex1.add(k) = i as c_int;
        }
    }
    unsafe {
        *ex1.add(sc.existing_grp.len()) = -1;
    }
    for (k, &i) in sc.new_grp.iter().enumerate() {
        unsafe {
            *ex2.add(k) = i as c_int;
        }
    }
    unsafe {
        *ex2.add(sc.new_grp.len()) = -1;
    }

    let n0 = sc.other_grp.len() as c_int;
    unsafe {
        mafft_sys::restorecommongaps(njob, n0, aseq, ex1, ex2, gapmap, alloclen, b'-' as c_char);
    }

    // findnewgaps: on the post-restore mseq1.
    let mseq1_restored =
        unsafe { mafft_sys::AllocateCharMtx(sc.existing_grp.len() as c_int, alloclen) };
    for (k, &i) in sc.existing_grp.iter().enumerate() {
        let content = unsafe { read_row(aseq, i) };
        unsafe {
            set_row(mseq1_restored, k, &content);
        }
    }
    let gaplen_len = (unsafe { mafft_sys::seqlen(*mseq1_restored.add(0)) } + 1) as c_int;
    let gaplen = unsafe { mafft_sys::AllocateIntVec(gaplen_len.max(alloclen)) };
    for k in 0..gaplen_len {
        unsafe {
            *gaplen.add(k as usize) = 0;
        }
    }
    unsafe {
        mafft_sys::findnewgaps(sc.existing_grp.len() as c_int, 0, mseq1_restored, gaplen);
    }

    // alreadyaligned: all rows are alreadyaligned for --add.
    let alreadyaligned = unsafe { mafft_sys::AllocateIntVec(njob) };
    for i in 0..njob as usize {
        unsafe {
            *alreadyaligned.add(i) = 1;
        }
    }

    eprintln!("\n--- Scenario: {} ---", sc.name);
    eprintln!("After restorecommongaps, before insertnewgaps:");
    for i in 0..njob as usize {
        let row = unsafe { read_row(aseq, i) };
        eprintln!(
            "  aseq[{}] = {} (len={})",
            i,
            String::from_utf8_lossy(&row),
            row.len()
        );
    }
    let mut gaplen_vec = Vec::new();
    for k in 0..gaplen_len {
        gaplen_vec.push(unsafe { *gaplen.add(k as usize) });
    }
    eprintln!("  gaplen = {:?}", gaplen_vec);

    // Call insertnewgaps.
    unsafe {
        mafft_sys::insertnewgaps(
            njob,
            alreadyaligned,
            aseq,
            ex1,
            ex2,
            gaplen,
            gapmap,
            alloclen,
            b'A' as c_char,
            b'-' as c_char,
        );
    }

    eprintln!("After C insertnewgaps:");
    let mut c_outputs: Vec<Vec<u8>> = Vec::with_capacity(njob as usize);
    for i in 0..njob as usize {
        let row = unsafe { read_row(aseq, i) };
        eprintln!(
            "  aseq[{}] = {} (len={})",
            i,
            String::from_utf8_lossy(&row),
            row.len()
        );
        c_outputs.push(row);
    }

    // Compute rust's flat-padding prediction for OTHER rows for comparison.
    // We need anchor_positions, gap_cols_before, new_merge_gap_set from the
    // pre-merge group1 state.
    let pre_strip_group1 = sc
        .pre_merge
        .iter()
        .find(|(i, _)| *i == sc.existing_grp[0])
        .map(|(_, v)| v.clone())
        .unwrap();
    let pre_width = pre_strip_group1.len();
    let pre_class: Vec<bool> = (0..pre_width)
        .map(|k| {
            sc.existing_grp.iter().all(|&i| {
                let c = sc
                    .pre_merge
                    .iter()
                    .find(|(idx, _)| *idx == i)
                    .map(|(_, v)| v[k])
                    .unwrap_or(b'-');
                c == b'-' || c == b'.'
            })
        })
        .collect();
    let anchor_positions: Vec<usize> = (0..pre_width).filter(|&k| !pre_class[k]).collect();
    let stripped_width = anchor_positions.len();
    let mut gap_cols_before: Vec<Vec<usize>> = vec![Vec::new(); stripped_width + 1];
    {
        let mut s = 0usize;
        for k in 0..pre_width {
            if pre_class[k] {
                gap_cols_before[s].push(k);
            } else {
                s += 1;
            }
        }
    }
    let post_active_group1: Vec<u8> = sc
        .post_merge_active
        .iter()
        .find(|(i, _)| *i == sc.existing_grp[0])
        .map(|(_, v)| v.clone())
        .unwrap();
    let stripped_no_eq: Vec<u8> = post_active_group1
        .iter()
        .filter(|&&c| c != b'=')
        .copied()
        .collect();
    let new_merge_gap_set: std::collections::HashSet<usize> = {
        let mut set = std::collections::HashSet::new();
        let mut s = 0;
        for (k, &c) in post_active_group1.iter().enumerate() {
            if c == b'=' {
                set.insert(k);
            } else {
                s += 1;
                let _ = s;
            }
        }
        set
    };
    let _ = stripped_no_eq;
    let post_merge_w = post_active_group1.len();

    let mut rust_other_outputs: Vec<Vec<u8>> = Vec::new();
    for &i in &sc.other_grp {
        let other_pre = sc
            .pre_merge
            .iter()
            .find(|(idx, _)| *idx == i)
            .map(|(_, v)| v.clone())
            .unwrap();
        let rust_out = rust_flat_padding(
            &other_pre,
            &anchor_positions,
            &gap_cols_before,
            &new_merge_gap_set,
            post_merge_w,
        );
        rust_other_outputs.push(rust_out);
    }

    eprintln!("Rust flat-padding prediction for OTHER:");
    for (k, &i) in sc.other_grp.iter().enumerate() {
        eprintln!(
            "  rust_flat[{}] = {} (len={})",
            i,
            String::from_utf8_lossy(&rust_other_outputs[k]),
            rust_other_outputs[k].len()
        );
    }
    // Compare each OTHER row's C output with rust prediction.
    for (k, &i) in sc.other_grp.iter().enumerate() {
        let c_row = &c_outputs[i];
        let r_row = &rust_other_outputs[k];
        if c_row == r_row {
            eprintln!("  flat-pad OTHER {}: MATCH", i);
        } else {
            eprintln!("  flat-pad OTHER {}: DIVERGE", i);
        }
    }

    // Also test rust's apply_c_insertnewgaps port — should match C
    // for non-compression scenarios.
    let mut rust_port_input: Vec<Vec<u8>> = (0..sc.pre_merge.len()).map(|_| Vec::new()).collect();
    // Pre-state of OTHER (pre-merge) + active (post-restore-state from C
    // chain). For this test, capture the state by re-running the C prep
    // pipeline; or just use what C produced just-before insertnewgaps.
    // Since we already wrote that to aseq before calling insertnewgaps,
    // we need to re-create it. Easiest: just hand-build it again.
    {
        // Build OTHER pre-merge state.
        for (i, content) in &sc.pre_merge {
            if sc.other_grp.contains(i) {
                rust_port_input[*i] = content.clone();
            }
        }
        // Build active post-restore state by re-running restorecommongaps
        // on the post_merge_active values.
        // For simplicity here: use the gap_cols_before/anchor_positions
        // derived earlier to expand post_merge into post-restore for active.
        for (i, content) in &sc.post_merge_active {
            let mut restored: Vec<u8> = Vec::new();
            let mut s = 0;
            for q in 0..content.len() {
                let c = content[q];
                if c != b'=' {
                    // Anchor col q in post-merge. Emit common-gaps before s+1.
                    for _ in 0..gap_cols_before[s].len() {
                        restored.push(b'-'); // active rows get '-' at common-gap positions
                    }
                    restored.push(c);
                    s += 1;
                } else {
                    // New-merge-gap col: emit as '=' for group1, take from content for group2.
                    restored.push(c);
                }
            }
            for _ in 0..gap_cols_before[stripped_width].len() {
                restored.push(b'-');
            }
            rust_port_input[*i] = restored;
        }
    }
    // Build gaplen and gapmap for rust port using the same chain C used.
    let active_post_restore = rust_port_input
        .get(sc.existing_grp[0])
        .cloned()
        .unwrap_or_default();
    let gaplen_rust = mafft_core::progressive::findnewgaps(&active_post_restore);
    // gapmap_rust: indexed by post-restore position. Non-zero where the
    // common-gap restoration inserted '-'. For our scenarios: derived
    // from gap_cols_before counts at the post-merge anchor positions.
    let mut gapmap_rust = vec![0usize; active_post_restore.len() + 2];
    {
        // Walk active_post_restore (with '=' chars). At each non-'=' col,
        // count how many '-' came BEFORE it from common-gap restoration.
        // Actually adjusted gapmap places the common-gap COUNT at the
        // post-merge col INDEX in post-restore coords. For our test
        // scenarios this is captured by gap_cols_before[s] at residue
        // boundaries.
        let mut s = 0usize;
        let mut q = 0usize;
        let post_active_grp1: Vec<u8> = sc
            .post_merge_active
            .iter()
            .find(|(i, _)| *i == sc.existing_grp[0])
            .map(|(_, v)| v.clone())
            .unwrap();
        for &c in &post_active_grp1 {
            if c == b'=' {
                // post_restore position of '=' = current q + accumulated '-'.
                // gapmap[q] = 0 at '=' positions (per adjustgapmap).
                q += 1;
            } else {
                // Anchor. gapmap value at this anchor's post-restore position
                // = gap_cols_before[s].len().
                let n_common = gap_cols_before.get(s).map(|v| v.len()).unwrap_or(0);
                // The gapmap at the post-restore col preceding this anchor:
                q += n_common; // skip the restored '-' chars
                if q < gapmap_rust.len() {
                    gapmap_rust[q - n_common] = n_common; // gapmap[post_restore_col_before_anchor] = count
                }
                q += 1;
                s += 1;
            }
        }
        // Trailing
        if s == stripped_width {
            let n_common = gap_cols_before
                .get(stripped_width)
                .map(|v| v.len())
                .unwrap_or(0);
            if q < gapmap_rust.len() {
                gapmap_rust[q] = n_common;
            }
        }
    }

    // Build scoring context for the port.
    let scoring = mafft_scoring::build_context(
        mafft_types::ScoringModel::Blosum(62),
        mafft_types::SeqType::Protein,
    );
    let gap = mafft_align::GapModel {
        open: -1530.0,
        extend: 0.0,
        shift: None,
        legacy_gap_cost: false,
    };

    let mut rust_port_aseq = rust_port_input.clone();
    mafft_core::progressive::apply_c_insertnewgaps(
        &mut rust_port_aseq,
        &sc.existing_grp,
        &sc.new_grp,
        &sc.other_grp,
        &gaplen_rust,
        &gapmap_rust,
        &scoring,
        &gap,
    );

    eprintln!("Rust apply_c_insertnewgaps port:");
    for i in 0..sc.pre_merge.len() {
        let rust_row = &rust_port_aseq[i];
        let c_row = &c_outputs[i];
        if rust_row == c_row {
            eprintln!(
                "  rust_port[{}] = {} (len={}) MATCH",
                i,
                String::from_utf8_lossy(rust_row),
                rust_row.len()
            );
        } else {
            eprintln!(
                "  rust_port[{}] = {} (len={}) DIVERGE",
                i,
                String::from_utf8_lossy(rust_row),
                rust_row.len()
            );
            eprintln!(
                "           C  = {} (len={})",
                String::from_utf8_lossy(c_row),
                c_row.len()
            );
        }
    }

    unsafe {
        mafft_sys::FreeCharMtx(aseq);
        mafft_sys::FreeCharMtx(mseq1_prestrip);
        mafft_sys::FreeCharMtx(mseq1_post);
        mafft_sys::FreeCharMtx(mseq1_restored);
        mafft_sys::FreeIntVec(ex1);
        mafft_sys::FreeIntVec(ex2);
        mafft_sys::FreeIntVec(gaplen);
        mafft_sys::FreeIntVec(gapmap);
        mafft_sys::FreeIntVec(alreadyaligned);
    }

    (c_outputs, rust_other_outputs)
}

/// Both scenarios in one test to avoid C-global teardown issues
/// between test functions. Tests that call `init_c_protein()` twice
/// have shown SIGSEGV on freeconstants() between calls — the C
/// globals weren't designed for re-initialization.
#[test]
#[cfg_attr(target_os = "linux", ignore = "potential glibc teardown issue")]
fn dump_c_insertnewgaps_scenarios() {
    unsafe {
        init_c_protein();

        // Scenario 1: no common-gaps in group1 pre-merge.
        // OTHER must match group1 width (both come from the same
        // input alignment). 1 new-merge-gap inserted by the merge.
        let sc1 = Scenario {
            name: "no_compression — no common-gaps + 1 new-merge-gap",
            pre_merge: vec![
                (0, b"MNGTEGDNF".to_vec()),  // group1, 9 chars
                (1, b"MNGTEGDNF".to_vec()),  // OTHER, 9 chars — same as group1
                (2, b"MNGTPEGDNF".to_vec()), // group2 (added, 10 chars ungapped)
            ],
            // Merge stripped group1 (9) with added (10) → post-merge
            // 10 cols, with '=' for added's P at pos 4.
            post_merge_active: vec![(0, b"MNGT=EGDNF".to_vec()), (2, b"MNGTPEGDNF".to_vec())],
            existing_grp: vec![0],
            new_grp: vec![2],
            other_grp: vec![1],
        };
        let _ = run_c_chain(&sc1);

        // Scenario 2: group1 has a common-gap adjacent to new-merge-gap.
        // C's profilealignment COMPRESSES — OTHER's residue absorbs the
        // new-merge-gap col. Rust flat-padding diverges.
        let sc2 = Scenario {
            name: "common_gap_with_other_residue — triggers profilealignment",
            pre_merge: vec![
                (0, b"MNGT-EGDNF".to_vec()),
                (1, b"MNGTAEGDNF".to_vec()),
                (2, b"MNGTPEGDNF".to_vec()),
            ],
            post_merge_active: vec![(0, b"MNGT=EGDNF".to_vec()), (2, b"MNGTPEGDNF".to_vec())],
            existing_grp: vec![0],
            new_grp: vec![2],
            other_grp: vec![1],
        };
        let _ = run_c_chain(&sc2);

        // Do NOT call mafft_sys::freeconstants() — it has been observed
        // to SIGSEGV after consecutive use here on macOS. The test
        // process exits anyway, so the leak is harmless.
    }
}

// Additional scenarios to add when extending this harness:
//
// - multi-row OTHER (mirroring step 1 of the 3-added test where
//   OTHER=[1,3] contains an already-added row with residues at the
//   gap-region positions).
// - multi-col gap regions with overlapping common-gaps.
// - non-adjacent new-merge-gaps (e.g., two regions separated by
//   anchors).
//
// Each scenario should be appended INSIDE the single test function
// above to share the init_c_protein() call. Calling init twice
// SIGSEGVs on the second freeconstants().

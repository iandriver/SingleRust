//! Deterministic parallelism helpers.
//!
//! Floating-point addition is not associative, so a naive parallel reduction (rayon's work
//! stealing splits and combines partials in a scheduling-dependent order) produces results that
//! vary bit-for-bit between runs and thread counts. Everything here instead uses **fixed-size
//! blocks with an ordered merge**: row range `0..n` is cut into `ceil(n/BLOCK)` blocks at fixed
//! indices, each block is folded sequentially (deterministic), and the per-block partials are
//! merged in block order. The block boundaries and merge order depend only on `n` and the
//! constant block size — never on the thread count or scheduler — so the result is identical on
//! every run and on any machine.

use rayon::prelude::*;

/// Fixed block size (rows) for deterministic block reductions. Constant so the partition (and
/// hence the floating-point summation order) is reproducible across runs and machines.
pub(crate) const DET_BLOCK: usize = 4096;

/// Deterministically reduce `0..n` in parallel.
///
/// `init` makes a zero partial, `fold_row(&mut partial, row)` accumulates one row into a partial
/// (called sequentially within a block, in increasing row order), and `merge(&mut acc, partial)`
/// combines a block's partial into the accumulator (called in increasing block order). The result
/// is bit-identical regardless of how many threads rayon uses.
pub(crate) fn det_block_reduce<P, Init, Fold, Merge>(
    n: usize,
    init: Init,
    fold_row: Fold,
    merge: Merge,
) -> P
where
    P: Send,
    Init: Fn() -> P + Sync,
    Fold: Fn(&mut P, usize) + Sync,
    Merge: Fn(&mut P, P),
{
    if n == 0 {
        return init();
    }
    let nblocks = n.div_ceil(DET_BLOCK);
    // Parallel across blocks; `collect` preserves block order.
    let partials: Vec<P> = (0..nblocks)
        .into_par_iter()
        .map(|b| {
            let lo = b * DET_BLOCK;
            let hi = ((b + 1) * DET_BLOCK).min(n);
            let mut p = init();
            for row in lo..hi {
                fold_row(&mut p, row);
            }
            p
        })
        .collect();
    // Sequential, ordered merge -> fixed summation order.
    let mut acc = init();
    for p in partials {
        merge(&mut acc, p);
    }
    acc
}

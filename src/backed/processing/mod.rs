//! Out-of-core (disk-backed) processing — streaming transforms/metrics that never hold the full
//! expression matrix in memory.
//!
//! - [`transformation`]: streaming `normalize_total` / `log1p`.
//! - [`qc`]: streaming QC metrics, written into `obs`/`var` in place.
//! - [`pipeline`]: fused QC + normalize + log1p in one job.
//! - [`hvg`]: streaming highly variable genes (Seurat), written into `var` in place.
//!
//! Parallel passes use [`det`]'s fixed-block ordered reduction so results are bit-identical
//! regardless of thread count (floating-point summation order is pinned).
pub(crate) mod det;
pub mod hvg;
pub mod pca;
pub mod pipeline;
pub mod qc;
pub mod transformation;

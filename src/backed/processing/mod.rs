//! Out-of-core (disk-backed) processing — streaming transforms/metrics that never hold the full
//! expression matrix in memory.
//!
//! - [`transformation`]: streaming `normalize_total` / `log1p`.
//! - [`qc`]: streaming QC metrics, written into `obs`/`var` in place.
//! - [`pipeline`]: fused QC + normalize + log1p in one job (the per-cell total is reused as the
//!   normalization row-sum, so it is computed once).
pub mod pipeline;
pub mod qc;
pub mod transformation;

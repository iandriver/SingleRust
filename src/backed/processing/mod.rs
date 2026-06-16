//! Out-of-core (disk-backed) processing — streaming transforms/metrics that never hold the full
//! expression matrix in memory.
//!
//! - [`transformation`]: streaming `normalize_total` / `log1p` (read-chunk → transform → write-chunk).
//! - [`qc`]: streaming QC metrics, written back into `obs`/`var` in place.
pub mod qc;
pub mod transformation;

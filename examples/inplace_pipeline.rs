//! # In-place pipeline — how SingleRust mutates one AnnData, no copies
//!
//! The headline ergonomic of SingleRust's memory API: you **load the dataset once** and every
//! processing step writes its results *back into that same `IMAnnData`*. There is no
//! "transform A into B" copying between steps — QC columns, the normalized matrix, the HVG
//! mask, and the PCA embedding all accumulate on one object.
//!
//! Note that `adata` below is **not** declared `mut`. `IMAnnData` uses interior mutability
//! (the matrix and frames sit behind locks), so every step takes a shared `&adata` (or
//! `&adata.x()`) and mutates through it. That is what keeps the call sites this small — and
//! what avoids the per-step reallocations a copy-on-transform API would incur (see the
//! benchmark in `demo/scverse_benchmark.ipynb`).
//!
//! ```text
//! cargo run --release --features enrichment --example inplace_pipeline -- [input.h5ad] [out.h5ad]
//! ```

use single_algebra::dimred::pca::{PowerIterationNormalizer, SVDMethod};
use single_rust::io;
use single_rust::memory::processing::dimred::pca::run_pca_inplace;
use single_rust::memory::processing::dimred::FeatureSelectionMethod;
use single_rust::memory::processing::{
    compute_highly_variable_genes, log1p_expression, normalize_expression,
};
use single_rust::memory::statistics::qc_metrics;
use single_rust::shared::HVGParams;
use single_utilities::types::Direction;

fn main() -> anyhow::Result<()> {
    let input = std::env::args().nth(1).unwrap_or_else(|| "data/input.h5ad".into());
    let output = std::env::args().nth(2);

    // ── Load ONCE ────────────────────────────────────────────────────────────────────────
    // Everything after this mutates `adata`; the expression matrix is never copied between
    // steps. `adata` is a shared handle, not `mut` — interior mutability does the work.
    let adata = io::read_h5ad_memory(&input)?;
    println!("loaded {} cells × {} genes", adata.n_obs(), adata.n_vars());

    // ── QC ───────────────────────────────────────────────────────────────────────────────
    // Writes per-cell / per-gene metric columns into adata.obs and adata.var in place.
    qc_metrics(&adata)?;

    // ── Normalize + log1p ──────────────────────────────────────────────────────────────────
    // Both mutate adata.x() (the count matrix) directly — sparse or dense, no reallocation.
    normalize_expression(&adata.x(), 10_000, &Direction::ROW, None)?;
    log1p_expression(&adata.x(), None)?;

    // ── Highly variable genes ──────────────────────────────────────────────────────────────
    // Adds the boolean "highly_variable" column to adata.var in place.
    compute_highly_variable_genes(
        &adata,
        Some(HVGParams { n_top_genes: Some(2000), ..Default::default() }),
    )?;
    let hvg_mask: Vec<bool> = adata
        .var()
        .get_column_from_df("highly_variable")?
        .bool()?
        .into_iter()
        .map(|b| b.unwrap_or(false))
        .collect();

    // ── PCA ────────────────────────────────────────────────────────────────────────────────
    // Stores the embedding into adata.obsm["X_pca"] and variance ratios into adata.uns —
    // the scanpy slots, ready for the next tool to read.
    run_pca_inplace::<f64>(
        &adata,
        Some(FeatureSelectionMethod::HighlyVariableSelection(hvg_mask)),
        Some(true),  // center
        Some(false), // quiet
        Some(50),
        None,
        Some(42),
        Some(SVDMethod::Random {
            n_oversamples: 10,
            n_power_iterations: 7,
            normalizer: PowerIterationNormalizer::QR,
        }),
        None,
    )?;

    // ── Everything landed on the SAME object ───────────────────────────────────────────────
    println!("obs columns: {:?}", adata.obs().get_data().get_column_names_str());
    println!("var columns: {:?}", adata.var().get_data().get_column_names_str());
    println!("obsm keys:   {:?}", adata.obsm().keys());
    println!("uns keys:    {:?}", adata.uns().keys()?);

    // Optionally persist the fully-annotated object for scanpy/anndata.
    if let Some(out) = output {
        io::write_h5ad(&adata, &out)?;
        println!("wrote {out}");
    }
    Ok(())
}

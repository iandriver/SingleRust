//! # scverse interoperability demo
//!
//! End-to-end pipeline exercising the newer SingleRust features and the scverse
//! round-trip. Run it on a counts `.h5ad` and it writes a results `.h5ad` that can be
//! re-opened in scanpy/anndata:
//!
//! ```text
//! cargo run --release --features enrichment --example scverse_demo -- \
//!     input.h5ad output.h5ad markers.tsv
//! ```
//!
//! Steps:
//! 1. read `.h5ad` (counts, CSR)            -> `io::read_h5ad_memory`
//! 2. QC metrics (mito %, counts, genes)    -> `qc_metrics`            [obs/var]
//! 3. normalize to 10k + log1p              -> dense/sparse transforms [X]
//! 4. highly variable genes                 -> `compute_highly_variable_genes` [var]
//! 5. PCA into obsm["X_pca"], uns variance  -> `run_pca_inplace`       [obsm/uns]
//! 6. ORA over a marker network (optional)  -> `run_ora`               [obsm]
//! 7. write results `.h5ad`                 -> `io::write_h5ad`
//!
//! `markers.tsv` is an optional two-column (tab-separated) `pathway<TAB>gene` file. When
//! present, ORA scores land in `obsm["ora_scores"]` / `obsm["ora_pvals"]` and the pathway
//! column order is written next to the output as `<output>.ora_pathways.txt`.

use std::io::Write;

use anndata::data::DynArray;
use anndata::ArrayData;
use anndata_memory::{IMAnnData, IMArrayElement};
use ndarray::Array2;
use single_algebra::dimred::pca::{PowerIterationNormalizer, SVDMethod};
use single_rust::io;
use single_rust::memory::processing::dimred::FeatureSelectionMethod;
use single_rust::memory::processing::dimred::pca::run_pca_inplace;
use single_rust::memory::processing::enrichment::run_ora;
use single_rust::memory::processing::{
    compute_highly_variable_genes, log1p_expression, normalize_expression,
};
use single_rust::memory::statistics::qc_metrics;
use single_rust::shared::HVGParams;
use single_utilities::types::{Direction, PathwayNetwork};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: {} <input.h5ad> <output.h5ad> [markers.tsv]",
            args.first().map(String::as_str).unwrap_or("scverse_demo")
        );
        std::process::exit(2);
    }
    let input = &args[1];
    let output = &args[2];
    let markers = args.get(3);

    // 1. Load counts. SingleRust reads the standard AnnData on-disk layout.
    println!("→ reading {input}");
    let adata = io::read_h5ad_memory(input)?;
    println!("  loaded {} cells × {} genes", adata.n_obs(), adata.n_vars());

    // 2. QC metrics on the raw counts (adds pct_counts_mito, n_genes_by_counts, ... to obs).
    println!("→ qc_metrics");
    qc_metrics(&adata)?;

    // 3. Library-size normalize to 10k counts/cell, then log1p. (Order matters: scale
    //    before the log.) Works for both sparse and dense X now.
    println!("→ normalize_total(1e4) + log1p");
    normalize_expression(&adata.x(), 10_000, &Direction::ROW, None)?;
    log1p_expression(&adata.x(), None)?;

    // 4. Highly variable genes (Seurat flavor, top 2000) -> var["highly_variable"].
    println!("→ highly variable genes (top 2000)");
    compute_highly_variable_genes(
        &adata,
        Some(HVGParams {
            n_top_genes: Some(2000),
            ..Default::default()
        }),
    )?;
    let hvg_mask = read_bool_var(&adata, "highly_variable")?;
    let n_hvg = hvg_mask.iter().filter(|&&b| b).count();
    println!("  selected {n_hvg} HVGs");

    // 5. PCA restricted to HVGs, stored the scanpy way: obsm["X_pca"] + uns variance ratios.
    //    Use randomized SVD (like scanpy's default): it reliably returns the requested
    //    number of components, whereas the Lanczos default can under-converge on
    //    clustered single-cell data.
    let n_pcs = 50;
    println!("→ PCA ({n_pcs} comps, randomized SVD) into obsm[\"X_pca\"]");
    let svd = SVDMethod::Random {
        n_oversamples: 10,
        n_power_iterations: 7,
        normalizer: PowerIterationNormalizer::QR,
    };
    let pca = run_pca_inplace::<f64>(
        &adata,
        Some(FeatureSelectionMethod::HighlyVariableSelection(hvg_mask)),
        Some(true),  // center
        Some(false), // quiet
        Some(n_pcs),
        None,
        Some(42),
        Some(svd),
        None, // key_added -> "pca"
    )?;
    let cum = &pca.cumulative_explained_variance_ratio;
    if let Some(last) = cum.iter().last() {
        println!("  cumulative variance over 50 PCs: {:.1}%", last * 100.0);
    }

    // 6. ORA over a marker-gene network, if one was provided.
    if let Some(markers_path) = markers {
        println!("→ ORA over markers from {markers_path}");
        let net = build_network_from_tsv(markers_path, &adata.var_names())?;
        let n_paths = net.get_num_pathways();
        println!("  {n_paths} pathways");
        let res = run_ora(&adata, &net, Some(50), None, None)?;
        store_array(&adata, "ora_scores", &res.scores)?;
        store_array(&adata, "ora_pvals", &res.pvals)?;

        // Record the pathway column order alongside the output.
        let names: Vec<String> = (0..n_paths)
            .map(|p| net.get_pathway_name(p).to_string())
            .collect();
        let sidecar = format!("{output}.ora_pathways.txt");
        let mut f = std::fs::File::create(&sidecar)?;
        writeln!(f, "{}", names.join("\n"))?;
        println!("  ora scores -> obsm[\"ora_scores\"], names -> {sidecar}");
    }

    // 7. Write everything back out as .h5ad for scanpy/anndata.
    println!("→ writing {output}");
    io::write_h5ad(&adata, output)?;
    println!("done.");
    Ok(())
}

/// Read a boolean column from `var` as a `Vec<bool>` (missing entries -> false).
fn read_bool_var(adata: &IMAnnData, column: &str) -> anyhow::Result<Vec<bool>> {
    Ok(adata
        .var()
        .get_column_from_df(column)?
        .bool()?
        .into_iter()
        .map(|b| b.unwrap_or(false))
        .collect())
}

/// Store an `n_obs × k` f32 matrix into `obsm[key]`.
fn store_array(adata: &IMAnnData, key: &str, values: &Array2<f32>) -> anyhow::Result<()> {
    let arr: ArrayData = DynArray::from(values.clone()).into();
    adata.obsm().add_array(key.to_string(), IMArrayElement::new(arr))
}

/// Build a [`PathwayNetwork`] from a two-column `pathway<TAB>gene` TSV. Gene symbols are
/// resolved against `var_names` (the column order of `X`); unknown genes are dropped.
fn build_network_from_tsv(path: &str, var_names: &[String]) -> anyhow::Result<PathwayNetwork> {
    let text = std::fs::read_to_string(path)?;
    let mut sources = Vec::new();
    let mut targets = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut cols = line.split('\t');
        if let (Some(pathway), Some(gene)) = (cols.next(), cols.next()) {
            sources.push(pathway.trim().to_string());
            targets.push(gene.trim().to_string());
        }
    }
    Ok(PathwayNetwork::new_from_vec(
        sources,
        targets,
        None,
        var_names.to_vec(),
        1,
    ))
}

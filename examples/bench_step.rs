//! # Single-step benchmark harness
//!
//! Runs exactly **one** SingleRust pipeline step on an `.h5ad` and reports the wall-clock
//! time of *only that operation* (the `.h5ad` read and write are measured separately and
//! excluded), so the number is directly comparable to the equivalent scanpy call timed
//! in-process.
//!
//! ```text
//! cargo run --release --features enrichment --example bench_step -- \
//!     <input.h5ad> <step> <output.h5ad|-> [markers.tsv]
//! ```
//!
//! `step` is one of: `qc`, `normalize`, `log1p`, `hvg`, `pca`, `ora`.
//! Pass `-` as the output to skip writing. Machine-readable lines are printed to stdout:
//! `READ_SECONDS=…`, `STEP_SECONDS=…`, `WRITE_SECONDS=…`.

use std::time::Instant;

use anndata::data::DynArray;
use anndata::ArrayData;
use anndata_memory::{IMAnnData, IMArrayElement};
use ndarray::Array2;
use single_algebra::dimred::pca::{PowerIterationNormalizer, SVDMethod};
use single_rust::io;
use single_rust::memory::processing::dimred::pca::run_pca_inplace;
use single_rust::memory::processing::dimred::FeatureSelectionMethod;
use single_rust::memory::processing::enrichment::run_ora;
use single_rust::memory::processing::{
    compute_highly_variable_genes, log1p_expression, normalize_expression,
};
use single_rust::memory::statistics::qc_metrics;
use single_rust::shared::HVGParams;
use single_utilities::types::{Direction, PathwayNetwork};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: bench_step <input.h5ad> <step> <output.h5ad|-> [markers.tsv]");
        std::process::exit(2);
    }
    let (input, step, output) = (&args[1], args[2].as_str(), &args[3]);
    let markers = args.get(4);

    let t = Instant::now();
    let adata = io::read_h5ad_memory(input)?;
    println!("READ_SECONDS={}", t.elapsed().as_secs_f64());
    eprintln!("loaded {} × {} for step '{step}'", adata.n_obs(), adata.n_vars());

    let t = Instant::now();
    match step {
        "qc" => {
            qc_metrics(&adata)?;
        }
        "normalize" => {
            normalize_expression(&adata.x(), 10_000, &Direction::ROW, None)?;
        }
        "log1p" => {
            log1p_expression(&adata.x(), None)?;
        }
        "hvg" => {
            compute_highly_variable_genes(
                &adata,
                Some(HVGParams {
                    n_top_genes: Some(2000),
                    ..Default::default()
                }),
            )?;
        }
        "pca" => {
            let hvg_mask = read_bool_var(&adata, "highly_variable")?;
            run_pca_inplace::<f64>(
                &adata,
                Some(FeatureSelectionMethod::HighlyVariableSelection(hvg_mask)),
                Some(true),
                Some(false),
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
        }
        "ora" => {
            let markers_path = markers
                .ok_or_else(|| anyhow::anyhow!("step 'ora' requires a markers.tsv argument"))?;
            // Building the network is prep (not the scored compute), so do it before timing
            // — restart the clock around run_ora only.
            let net = build_network_from_tsv(markers_path, &adata.var_names())?;
            let t_ora = Instant::now();
            let res = run_ora(&adata, &net, Some(50), None, None)?;
            println!("STEP_SECONDS={}", t_ora.elapsed().as_secs_f64());
            store_array(&adata, "ora_scores", &res.scores)?;
            maybe_write(&adata, output)?;
            return Ok(());
        }
        // Full core pipeline in one process (for the cells-vs-runtime scaling sweep): each
        // step feeds the next in memory, and per-step + total compute times are printed.
        "all" => {
            let mut total = 0.0;
            let mut timed = |label: &str, dur: f64| {
                total += dur;
                println!("ALL_{label}={dur}");
            };

            let s = Instant::now();
            qc_metrics(&adata)?;
            timed("qc", s.elapsed().as_secs_f64());

            let s = Instant::now();
            normalize_expression(&adata.x(), 10_000, &Direction::ROW, None)?;
            timed("normalize", s.elapsed().as_secs_f64());

            let s = Instant::now();
            log1p_expression(&adata.x(), None)?;
            timed("log1p", s.elapsed().as_secs_f64());

            let s = Instant::now();
            compute_highly_variable_genes(
                &adata,
                Some(HVGParams {
                    n_top_genes: Some(2000),
                    ..Default::default()
                }),
            )?;
            timed("hvg", s.elapsed().as_secs_f64());

            let s = Instant::now();
            let hvg_mask = read_bool_var(&adata, "highly_variable")?;
            run_pca_inplace::<f64>(
                &adata,
                Some(FeatureSelectionMethod::HighlyVariableSelection(hvg_mask)),
                Some(true),
                Some(false),
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
            timed("pca", s.elapsed().as_secs_f64());

            println!("STEP_SECONDS={total}");
            maybe_write(&adata, output)?;
            return Ok(());
        }
        other => anyhow::bail!("unknown step '{other}'"),
    }
    println!("STEP_SECONDS={}", t.elapsed().as_secs_f64());

    maybe_write(&adata, output)?;
    Ok(())
}

fn maybe_write(adata: &IMAnnData, output: &str) -> anyhow::Result<()> {
    if output == "-" {
        println!("WRITE_SECONDS=0");
    } else {
        let t = Instant::now();
        io::write_h5ad(adata, output)?;
        println!("WRITE_SECONDS={}", t.elapsed().as_secs_f64());
    }
    Ok(())
}

fn read_bool_var(adata: &IMAnnData, column: &str) -> anyhow::Result<Vec<bool>> {
    Ok(adata
        .var()
        .get_column_from_df(column)?
        .bool()?
        .into_iter()
        .map(|b| b.unwrap_or(false))
        .collect())
}

fn store_array(adata: &IMAnnData, key: &str, values: &Array2<f32>) -> anyhow::Result<()> {
    let arr: ArrayData = DynArray::from(values.clone()).into();
    adata
        .obsm()
        .add_array(key.to_string(), IMArrayElement::new(arr))
}

fn build_network_from_tsv(path: &str, var_names: &[String]) -> anyhow::Result<PathwayNetwork> {
    let text = std::fs::read_to_string(path)?;
    let (mut sources, mut targets) = (Vec::new(), Vec::new());
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut cols = line.split('\t');
        if let (Some(p), Some(g)) = (cols.next(), cols.next()) {
            sources.push(p.trim().to_string());
            targets.push(g.trim().to_string());
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

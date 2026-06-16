//! # Out-of-core QC metrics (disk-backed, single streaming pass)
//!
//! Computes the same cell- and gene-level QC metrics as
//! [`crate::memory::statistics::qc::qc_metrics`], but streams `X` in row-chunks so peak memory
//! is bounded by one chunk plus a few per-cell / per-gene accumulator vectors. `X` is never
//! modified — the metrics are written back into `obs`/`var` of the same file in place.

use std::path::Path;

use anndata::data::DynCsrMatrix;
use anndata::{AnnData, AnnDataOp, ArrayData, ArrayElemOp, Backend};
use anndata_hdf5::H5;
use anyhow::bail;
use polars::prelude::Column;

use super::transformation::DEFAULT_CHUNK_SIZE;

const PERCENT_TOP: [usize; 4] = [50, 100, 200, 500];

/// Per-cell and per-gene accumulators filled during the streaming pass.
struct QcAcc {
    // cell-level (indexed by global obs index)
    n_genes: Vec<u32>,
    total: Vec<f64>,
    mito_total: Vec<f64>,
    pct_top: Vec<[f64; PERCENT_TOP.len()]>,
    // gene-level (indexed by var index, accumulated across chunks)
    col_total: Vec<f64>,
    col_nnz: Vec<u32>,
}

/// Compute standard QC metrics out-of-core and store them in `obs`/`var` of the file in place.
///
/// Mitochondrial genes are detected by an `MT-`/`mt-` var-name prefix (matching the in-memory
/// `qc_metrics`). Adds to `obs`: `n_genes_by_counts`, `total_counts`,
/// `pct_counts_in_top_{50,100,200,500}_genes`, `total_counts_mito`, `pct_counts_mito`
/// (+ `log1p_` variants); to `var`: `mito`, `n_cells_by_counts`, `mean_counts`,
/// `pct_dropout_by_counts`, `total_counts` (+ `log1p_` variants).
pub fn qc_metrics_backed(path: &Path, chunk_size: Option<usize>) -> anyhow::Result<()> {
    let chunk_size = chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);
    let adata = AnnData::<H5>::open(H5::open_rw(path)?)?;
    let (n_obs, n_vars) = (adata.n_obs(), adata.n_vars());

    let var_names = adata.var_names();
    let mito_mask: Vec<bool> = var_names
        .into_vec()
        .iter()
        .map(|n| n.starts_with("MT-") || n.starts_with("mt-"))
        .collect();

    let mut acc = QcAcc {
        n_genes: vec![0; n_obs],
        total: vec![0.0; n_obs],
        mito_total: vec![0.0; n_obs],
        pct_top: vec![[0.0; PERCENT_TOP.len()]; n_obs],
        col_total: vec![0.0; n_vars],
        col_nnz: vec![0; n_vars],
    };

    // Single streaming pass over X.
    for (chunk, start, _end) in adata.x().iter::<ArrayData>(chunk_size) {
        accumulate(&chunk, start, &mito_mask, &mut acc)?;
    }

    write_metrics(&adata, &mito_mask, &acc, n_obs)?;
    adata.close()?;
    Ok(())
}

fn accumulate(
    chunk: &ArrayData,
    start: usize,
    mito_mask: &[bool],
    acc: &mut QcAcc,
) -> anyhow::Result<()> {
    match chunk {
        ArrayData::CsrMatrix(DynCsrMatrix::F32(m)) => accumulate_rows(m, start, mito_mask, acc),
        ArrayData::CsrMatrix(DynCsrMatrix::F64(m)) => accumulate_rows(m, start, mito_mask, acc),
        other => bail!("OOC qc supports only F32/F64 CSR matrices, got {:?}", other),
    }
}

fn accumulate_rows<T>(
    m: &nalgebra_sparse::CsrMatrix<T>,
    start: usize,
    mito_mask: &[bool],
    acc: &mut QcAcc,
) -> anyhow::Result<()>
where
    T: Copy + num_traits::ToPrimitive,
{
    let mut scratch: Vec<f64> = Vec::new();
    for (i, row) in m.row_iter().enumerate() {
        let g = start + i;
        let cols = row.col_indices();
        let vals = row.values();

        let mut total = 0.0;
        let mut mito = 0.0;
        for (&c, &v) in cols.iter().zip(vals.iter()) {
            let v = v.to_f64().unwrap_or(0.0);
            total += v;
            if mito_mask[c] {
                mito += v;
            }
            acc.col_total[c] += v;
            acc.col_nnz[c] += 1;
        }
        acc.n_genes[g] = vals.len() as u32;
        acc.total[g] = total;
        acc.mito_total[g] = mito;

        // top-N segment proportions: sum of the largest N values / total.
        if total > 0.0 {
            scratch.clear();
            scratch.extend(vals.iter().map(|&v| v.to_f64().unwrap_or(0.0)));
            scratch.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let mut running = 0.0;
            let mut next = 0;
            for (rank, &v) in scratch.iter().enumerate() {
                running += v;
                while next < PERCENT_TOP.len() && rank + 1 == PERCENT_TOP[next].min(scratch.len()) {
                    acc.pct_top[g][next] = running / total * 100.0;
                    next += 1;
                }
            }
            // if a requested top-N exceeds nnz, it captures all counts -> 100%
            while next < PERCENT_TOP.len() {
                acc.pct_top[g][next] = 100.0;
                next += 1;
            }
        }
    }
    Ok(())
}

fn write_metrics(
    adata: &AnnData<H5>,
    mito_mask: &[bool],
    acc: &QcAcc,
    n_obs: usize,
) -> anyhow::Result<()> {
    let log1p = |xs: &[f64]| -> Vec<f64> { xs.iter().map(|&x| x.ln_1p()).collect() };

    // ----- obs -----
    let mut obs = adata.read_obs()?;
    let n_genes_f: Vec<f64> = acc.n_genes.iter().map(|&n| n as f64).collect();
    obs.with_column(Column::new("n_genes_by_counts".into(), acc.n_genes.clone()))?;
    obs.with_column(Column::new("log1p_n_genes_by_counts".into(), log1p(&n_genes_f)))?;
    obs.with_column(Column::new("total_counts".into(), acc.total.clone()))?;
    obs.with_column(Column::new("log1p_total_counts".into(), log1p(&acc.total)))?;
    for (k, n) in PERCENT_TOP.iter().enumerate() {
        let col: Vec<f64> = acc.pct_top.iter().map(|p| p[k]).collect();
        obs.with_column(Column::new(format!("pct_counts_in_top_{n}_genes").into(), col))?;
    }
    obs.with_column(Column::new("total_counts_mito".into(), acc.mito_total.clone()))?;
    obs.with_column(Column::new("log1p_total_counts_mito".into(), log1p(&acc.mito_total)))?;
    let pct_mito: Vec<f64> = acc
        .mito_total
        .iter()
        .zip(acc.total.iter())
        .map(|(&m, &t)| if t > 0.0 { m / t * 100.0 } else { 0.0 })
        .collect();
    obs.with_column(Column::new("pct_counts_mito".into(), pct_mito))?;
    adata.set_obs(obs)?;

    // ----- var -----
    let mut var = adata.read_var()?;
    let mean: Vec<f64> = acc.col_total.iter().map(|&t| t / n_obs as f64).collect();
    let pct_dropout: Vec<f64> = acc
        .col_nnz
        .iter()
        .map(|&n| (1.0 - n as f64 / n_obs as f64) * 100.0)
        .collect();
    var.with_column(Column::new("mito".into(), mito_mask.to_vec()))?;
    var.with_column(Column::new("n_cells_by_counts".into(), acc.col_nnz.clone()))?;
    var.with_column(Column::new("mean_counts".into(), mean.clone()))?;
    var.with_column(Column::new("log1p_mean_counts".into(), log1p(&mean)))?;
    var.with_column(Column::new("pct_dropout_by_counts".into(), pct_dropout))?;
    var.with_column(Column::new("total_counts".into(), acc.col_total.clone()))?;
    var.with_column(Column::new("log1p_total_counts".into(), log1p(&acc.col_total)))?;
    adata.set_var(var)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backed::processing::transformation::log1p_backed; // ensure module wiring
    use anndata::data::DynCsrMatrix;
    use nalgebra_sparse::{CooMatrix, CsrMatrix};

    fn write_fixture(path: &Path, rows: &[Vec<f64>], var_names: &[&str]) -> anyhow::Result<()> {
        let (nr, nc) = (rows.len(), rows[0].len());
        let mut coo = CooMatrix::<f32>::new(nr, nc);
        for (i, r) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                if v != 0.0 {
                    coo.push(i, j, v as f32);
                }
            }
        }
        let adata = AnnData::<H5>::new(path)?;
        adata.set_obs_names((0..nr).map(|i| format!("c{i}")).collect::<Vec<_>>().into())?;
        adata.set_var_names(var_names.iter().map(|s| s.to_string()).collect::<Vec<_>>().into())?;
        adata.set_x(ArrayData::CsrMatrix(DynCsrMatrix::F32(CsrMatrix::from(&coo))))?;
        adata.close()?;
        Ok(())
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(name);
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn ooc_qc_matches_hand_computed() -> anyhow::Result<()> {
        let _ = log1p_backed; // silence unused import if test layout changes
        let path = tmp("sr_ooc_qc.h5ad");
        // 3 cells × 4 genes; gene 0 is "MT-x" (mitochondrial).
        let rows = vec![
            vec![10.0, 0.0, 5.0, 5.0], // total 20, mito 10 -> 50%
            vec![0.0, 0.0, 0.0, 0.0],  // empty
            vec![1.0, 2.0, 3.0, 4.0],  // total 10, mito 1 -> 10%
        ];
        write_fixture(&path, &rows, &["MT-x", "g1", "g2", "g3"])?;
        qc_metrics_backed(&path, Some(2))?;

        let adata = AnnData::<H5>::open(H5::open(&path)?)?;
        let obs = adata.read_obs()?;
        let var = adata.read_var()?;

        let total = obs.column("total_counts")?.f64()?;
        assert_eq!(total.get(0), Some(20.0));
        assert_eq!(total.get(2), Some(10.0));
        let n_genes = obs.column("n_genes_by_counts")?.u32()?;
        assert_eq!(n_genes.get(0), Some(3));
        assert_eq!(n_genes.get(1), Some(0));
        let pct_mito = obs.column("pct_counts_mito")?.f64()?;
        assert!((pct_mito.get(0).unwrap() - 50.0).abs() < 1e-6);
        assert!((pct_mito.get(2).unwrap() - 10.0).abs() < 1e-6);

        // var: gene 0 total = 11 across cells, detected in 2 cells, mito=true
        let vtotal = var.column("total_counts")?.f64()?;
        assert_eq!(vtotal.get(0), Some(11.0));
        let ncells = var.column("n_cells_by_counts")?.u32()?;
        assert_eq!(ncells.get(0), Some(2));
        let mito = var.column("mito")?.bool()?;
        assert_eq!(mito.get(0), Some(true));
        assert_eq!(mito.get(1), Some(false));

        adata.close()?;
        std::fs::remove_file(path).ok();
        Ok(())
    }
}

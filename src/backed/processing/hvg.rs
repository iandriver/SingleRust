//! # Out-of-core highly variable genes (Seurat flavor)
//!
//! One streaming pass over `X` accumulates per-gene sum and sum-of-squares; from those we form
//! the per-gene mean and **sample** variance (`(Σx²/n − mean²)·n/(n−1)`) — identical to the
//! in-memory `var_col` — then hand off to the shared [`seurat_select`] for the dispersion
//! binning/normalization and top-N selection. Results are written into `var` in place.
//!
//! Memory is bounded by one chunk plus two `n_vars`-length f64 accumulators.

use std::path::Path;

use anndata::data::DynCsrMatrix;
use anndata::{AnnData, AnnDataOp, ArrayData, ArrayElemOp, Backend};
use anndata_hdf5::H5;
use anyhow::bail;
use polars::prelude::Column;

use crate::memory::processing::hvg::seurat_select;
use crate::shared::processing::HVGParams;

use super::det::det_block_reduce;
use super::transformation::DEFAULT_CHUNK_SIZE;

/// Compute highly variable genes (Seurat flavor) out-of-core and write the results
/// (`means`, `dispersions`, `dispersions_norm`, `highly_variable`) into `var` in place.
pub fn highly_variable_genes_backed(
    path: &Path,
    n_top_genes: Option<usize>,
    chunk_size: Option<usize>,
) -> anyhow::Result<()> {
    let chunk_size = chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);
    let adata = AnnData::<H5>::open(H5::open_rw(path)?)?;
    let (n_obs, n_vars) = (adata.n_obs(), adata.n_vars());

    // Streaming pass: per-gene sum and sum of squares. Each chunk is reduced with a deterministic
    // block reduction (parallel, fixed summation order), then added to the running totals.
    let mut col_sum = vec![0.0_f64; n_vars];
    let mut col_sumsq = vec![0.0_f64; n_vars];
    for (chunk, _start, _end) in adata.x().iter::<ArrayData>(chunk_size) {
        let (s, sq) = sumsq_chunk(&chunk, n_vars)?;
        for (a, b) in col_sum.iter_mut().zip(s.iter()) {
            *a += *b;
        }
        for (a, b) in col_sumsq.iter_mut().zip(sq.iter()) {
            *a += *b;
        }
    }

    let n = n_obs as f64;
    let raw_means: Vec<f64> = col_sum.iter().map(|&s| s / n).collect();
    let variances: Vec<f64> = col_sum
        .iter()
        .zip(col_sumsq.iter())
        .map(|(&s, &sq)| {
            let mean = s / n;
            let pop_var = sq / n - mean * mean;
            if n > 1.0 {
                pop_var * (n / (n - 1.0)) // Bessel-corrected, matches in-memory var_col
            } else {
                0.0
            }
        })
        .collect();

    let params = HVGParams {
        n_top_genes,
        ..Default::default()
    };
    let (log1p_means, log_dispersions, dispersions_norm, highly_variable) =
        seurat_select(&raw_means, &variances, &params)?;

    let mut var = adata.read_var()?;
    var.with_column(Column::new("means".into(), log1p_means))?;
    var.with_column(Column::new("dispersions".into(), log_dispersions))?;
    var.with_column(Column::new("dispersions_norm".into(), dispersions_norm))?;
    var.with_column(Column::new("highly_variable".into(), highly_variable))?;
    adata.set_var(var)?;

    adata.close()?;
    Ok(())
}

/// One chunk's per-gene `(sum, sum_of_squares)` via a deterministic block reduction over rows.
fn sumsq_chunk(chunk: &ArrayData, n_vars: usize) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    match chunk {
        ArrayData::CsrMatrix(DynCsrMatrix::F32(m)) => Ok(sumsq_rows(m, n_vars)),
        ArrayData::CsrMatrix(DynCsrMatrix::F64(m)) => Ok(sumsq_rows(m, n_vars)),
        other => bail!("OOC HVG supports only F32/F64 CSR matrices, got {:?}", other),
    }
}

fn sumsq_rows<T: Copy + num_traits::ToPrimitive + Sync>(
    m: &nalgebra_sparse::CsrMatrix<T>,
    n_vars: usize,
) -> (Vec<f64>, Vec<f64>) {
    det_block_reduce(
        m.nrows(),
        || (vec![0.0_f64; n_vars], vec![0.0_f64; n_vars]),
        |(sum, sumsq), r| {
            let row = m.row(r);
            for (&c, &v) in row.col_indices().iter().zip(row.values().iter()) {
                let v = v.to_f64().unwrap_or(0.0);
                sum[c] += v;
                sumsq[c] += v * v;
            }
        },
        |(asum, asq), (sum, sq)| {
            for (a, b) in asum.iter_mut().zip(sum.iter()) {
                *a += *b;
            }
            for (a, b) in asq.iter_mut().zip(sq.iter()) {
                *a += *b;
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndata::data::DynCsrMatrix;
    use nalgebra_sparse::{CooMatrix, CsrMatrix};

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(name);
        let _ = std::fs::remove_file(&p);
        p
    }

    /// OOC HVG must select exactly the same genes as the in-memory implementation on the same
    /// data (both share `seurat_select`; only the mean/variance computation differs in path).
    #[test]
    fn ooc_hvg_matches_in_memory() -> anyhow::Result<()> {
        // 40 cells × 12 genes, varied magnitudes so dispersion ranking is unambiguous.
        let (nr, nc) = (40usize, 12usize);
        let mut coo = CooMatrix::<f32>::new(nr, nc);
        for i in 0..nr {
            for j in 0..nc {
                let v = (((i * 13 + j * 7) % 11) as f32) * (1.0 + (j as f32) * 0.3);
                if v != 0.0 {
                    coo.push(i, j, v);
                }
            }
        }
        let path = tmp("sr_ooc_hvg.h5ad");
        let adata = AnnData::<H5>::new(&path)?;
        adata.set_obs_names((0..nr).map(|i| format!("c{i}")).collect::<Vec<_>>().into())?;
        adata.set_var_names((0..nc).map(|j| format!("g{j}")).collect::<Vec<_>>().into())?;
        adata.set_x(ArrayData::CsrMatrix(DynCsrMatrix::F32(CsrMatrix::from(&coo))))?;
        adata.close()?;

        // In-memory reference (does not mutate the file).
        let im = crate::io::read_h5ad_memory(&path)?;
        crate::memory::processing::compute_highly_variable_genes(
            &im,
            Some(HVGParams { n_top_genes: Some(5), ..Default::default() }),
        )?;
        let ref_mask: Vec<bool> = im
            .var()
            .get_column_from_df("highly_variable")?
            .bool()?
            .into_iter()
            .map(|b| b.unwrap_or(false))
            .collect();

        // Out-of-core (mutates the file's var), tiny chunk to exercise streaming.
        highly_variable_genes_backed(&path, Some(5), Some(7))?;
        let back = AnnData::<H5>::open(H5::open(&path)?)?;
        let ooc_mask: Vec<bool> = back
            .read_var()?
            .column("highly_variable")?
            .bool()?
            .into_iter()
            .map(|b| b.unwrap_or(false))
            .collect();
        back.close()?;

        assert_eq!(ref_mask, ooc_mask, "OOC HVG mask must equal in-memory HVG mask");
        assert!(ooc_mask.iter().any(|&b| b), "expected some HVGs selected");
        std::fs::remove_file(path).ok();
        Ok(())
    }

    /// Determinism: HVG run with 1 vs 8 threads must give bit-identical means/dispersions and the
    /// same selection (guards the parallel sum/sum-of-squares reduction).
    #[test]
    fn ooc_hvg_is_deterministic_across_thread_counts() -> anyhow::Result<()> {
        let (nr, nc) = (5000usize, 80usize);
        let build = |tag: &str| -> anyhow::Result<std::path::PathBuf> {
            let mut coo = CooMatrix::<f32>::new(nr, nc);
            for i in 0..nr {
                for j in 0..nc {
                    let v = (((i * 29 + j * 11) % 17) as f32) * (0.5 + (j as f32) * 0.1);
                    if v != 0.0 {
                        coo.push(i, j, v);
                    }
                }
            }
            let p = tmp(&format!("sr_hvg_det_{tag}.h5ad"));
            let a = AnnData::<H5>::new(&p)?;
            a.set_obs_names((0..nr).map(|i| format!("c{i}")).collect::<Vec<_>>().into())?;
            a.set_var_names((0..nc).map(|j| format!("g{j}")).collect::<Vec<_>>().into())?;
            a.set_x(ArrayData::CsrMatrix(DynCsrMatrix::F32(CsrMatrix::from(&coo))))?;
            a.close()?;
            Ok(p)
        };
        let run = |threads: usize, tag: &str| -> anyhow::Result<(Vec<f64>, Vec<bool>)> {
            let p = build(tag)?;
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
            pool.install(|| highly_variable_genes_backed(&p, Some(20), Some(1000)))?;
            let a = AnnData::<H5>::open(H5::open(&p)?)?;
            let var = a.read_var()?;
            let means = var.column("means")?.f64()?.into_iter().map(|x| x.unwrap_or(f64::NAN)).collect();
            let hv = var.column("highly_variable")?.bool()?.into_iter().map(|b| b.unwrap_or(false)).collect();
            a.close()?;
            std::fs::remove_file(p).ok();
            Ok((means, hv))
        };
        let (m1, h1) = run(1, "t1")?;
        let (m8, h8) = run(8, "t8")?;
        assert_eq!(m1, m8, "HVG means differ between 1 and 8 threads");
        assert_eq!(h1, h8, "HVG selection differs between 1 and 8 threads");
        Ok(())
    }
}

//! # Out-of-core transformations (disk-backed, streaming)
//!
//! These operate on a disk-backed `.h5ad` without ever holding the full expression matrix in
//! memory. The expression matrix `X` is streamed in row-chunks (`x().iter::<ArrayData>(n)`),
//! each chunk is transformed, and the result is streamed straight back out to a new `.h5ad`
//! (`set_x_from_iter`, which appends to an extendable HDF5 dataset). Peak memory is therefore
//! bounded by one chunk, not the dataset — this is the basis for processing larger-than-RAM data
//! natively, the niche scanpy reaches for Dask to fill.
//!
//! Only CSR `f32`/`f64` matrices are supported (the standard scRNA-seq layout).

use std::path::Path;

use anndata::data::DynCsrMatrix;
use anndata::{AnnData, AnnDataOp, ArrayData, ArrayElemOp, Backend};
use anndata_hdf5::H5;
use anyhow::bail;
use single_algebra::Log1P;

/// Default streaming chunk size (rows/cells per block).
pub const DEFAULT_CHUNK_SIZE: usize = 5_000;

/// Open a backed `.h5ad` for reading and create a fresh backed `.h5ad` for the result, copying
/// the (small) obs/var annotations and dimension names across. Returns `(input, output)`.
fn open_io(input: &Path, output: &Path) -> anyhow::Result<(AnnData<H5>, AnnData<H5>)> {
    let adata = AnnData::<H5>::open(H5::open(input)?)?;
    let out = AnnData::<H5>::new(output)?;
    out.set_obs_names(adata.obs_names())?;
    out.set_var_names(adata.var_names())?;
    out.set_obs(adata.read_obs()?)?;
    out.set_var(adata.read_var()?)?;
    Ok((adata, out))
}

/// Apply `log1p` (natural `ln(1 + x)`) to every value of `X`, streaming, writing the result to
/// `output`. Memory stays bounded by `chunk_size` rows regardless of dataset size.
pub fn log1p_backed(input: &Path, output: &Path, chunk_size: Option<usize>) -> anyhow::Result<()> {
    let chunk_size = chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);
    let (adata, out) = open_io(input, output)?;

    let iter = adata
        .x()
        .iter::<ArrayData>(chunk_size)
        .map(|(chunk, _start, _end)| {
            // The iterator can't yield a Result, so transform failures (unsupported dtype) panic
            // with context. We guarantee dtype up front in the public callers.
            log1p_chunk(chunk).expect("log1p chunk transform")
        });
    out.set_x_from_iter(iter)?;

    out.close()?;
    adata.close()?;
    Ok(())
}

/// Library-size normalize each cell to `target_sum`, then optionally `log1p`, streaming to
/// `output`. Two passes over `X`: pass 1 accumulates per-cell sums (one f64 per cell — tiny),
/// pass 2 streams chunks, scales each row, and writes. Memory stays bounded by `chunk_size`.
pub fn normalize_total_backed(
    input: &Path,
    output: &Path,
    target_sum: f64,
    log1p: bool,
    chunk_size: Option<usize>,
) -> anyhow::Result<()> {
    let chunk_size = chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);

    // Pass 1: per-cell totals (streamed). Keeps only n_obs f64s in memory.
    let sums = {
        let adata = AnnData::<H5>::open(H5::open(input)?)?;
        let n_obs = adata.n_obs();
        let mut sums = vec![0.0_f64; n_obs];
        for (chunk, start, _end) in adata.x().iter::<ArrayData>(chunk_size) {
            accumulate_row_sums(&chunk, start, &mut sums)?;
        }
        adata.close()?;
        sums
    };

    // Pass 2: scale (and optionally log1p) each row, streamed back out.
    let (adata, out) = open_io(input, output)?;
    let iter = adata
        .x()
        .iter::<ArrayData>(chunk_size)
        .map(move |(chunk, start, _end)| {
            normalize_chunk(chunk, start, &sums, target_sum, log1p)
                .expect("normalize chunk transform")
        });
    out.set_x_from_iter(iter)?;
    out.close()?;
    adata.close()?;
    Ok(())
}

/// `ln(1 + x)` over a chunk's stored values, preserving sparsity (zeros map to 0).
fn log1p_chunk(chunk: ArrayData) -> anyhow::Result<ArrayData> {
    match chunk {
        ArrayData::CsrMatrix(DynCsrMatrix::F32(mut m)) => {
            m.log1p_normalize()?;
            Ok(ArrayData::CsrMatrix(DynCsrMatrix::F32(m)))
        }
        ArrayData::CsrMatrix(DynCsrMatrix::F64(mut m)) => {
            m.log1p_normalize()?;
            Ok(ArrayData::CsrMatrix(DynCsrMatrix::F64(m)))
        }
        other => bail!("OOC log1p supports only F32/F64 CSR matrices, got {:?}", other),
    }
}

/// Add each row's value-sum into `sums[start + local_row]`.
fn accumulate_row_sums(chunk: &ArrayData, start: usize, sums: &mut [f64]) -> anyhow::Result<()> {
    match chunk {
        ArrayData::CsrMatrix(DynCsrMatrix::F32(m)) => {
            for (i, row) in m.row_iter().enumerate() {
                sums[start + i] = row.values().iter().map(|&v| v as f64).sum();
            }
            Ok(())
        }
        ArrayData::CsrMatrix(DynCsrMatrix::F64(m)) => {
            for (i, row) in m.row_iter().enumerate() {
                sums[start + i] = row.values().iter().sum();
            }
            Ok(())
        }
        other => bail!("OOC normalize supports only F32/F64 CSR matrices, got {:?}", other),
    }
}

/// Scale each row to `target_sum` (rows with zero total are left untouched), optionally log1p.
pub(crate) fn normalize_chunk(
    chunk: ArrayData,
    start: usize,
    sums: &[f64],
    target_sum: f64,
    log1p: bool,
) -> anyhow::Result<ArrayData> {
    match chunk {
        ArrayData::CsrMatrix(DynCsrMatrix::F32(mut m)) => {
            scale_rows_f32(&mut m, start, sums, target_sum, log1p);
            Ok(ArrayData::CsrMatrix(DynCsrMatrix::F32(m)))
        }
        ArrayData::CsrMatrix(DynCsrMatrix::F64(mut m)) => {
            scale_rows_f64(&mut m, start, sums, target_sum, log1p);
            Ok(ArrayData::CsrMatrix(DynCsrMatrix::F64(m)))
        }
        other => bail!("OOC normalize supports only F32/F64 CSR matrices, got {:?}", other),
    }
}

macro_rules! impl_scale_rows {
    ($name:ident, $t:ty) => {
        fn $name(
            m: &mut nalgebra_sparse::CsrMatrix<$t>,
            start: usize,
            sums: &[f64],
            target_sum: f64,
            log1p: bool,
        ) {
            let offsets = m.row_offsets().to_vec();
            let values = m.values_mut();
            for row in 0..offsets.len() - 1 {
                let total = sums[start + row];
                if total <= 0.0 {
                    continue;
                }
                let scale = target_sum / total;
                for v in &mut values[offsets[row]..offsets[row + 1]] {
                    let mut x = (*v as f64) * scale;
                    if log1p {
                        x = x.ln_1p();
                    }
                    *v = x as $t;
                }
            }
        }
    };
}
impl_scale_rows!(scale_rows_f32, f32);
impl_scale_rows!(scale_rows_f64, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use anndata::data::DynCsrMatrix;
    use anndata::ArrayData;
    use nalgebra_sparse::{CooMatrix, CsrMatrix};

    fn write_fixture(path: &Path, rows: &[Vec<f64>]) -> anyhow::Result<()> {
        let (nr, nc) = (rows.len(), rows[0].len());
        let mut coo = CooMatrix::<f32>::new(nr, nc);
        for (i, r) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                if v != 0.0 {
                    coo.push(i, j, v as f32);
                }
            }
        }
        let csr = CsrMatrix::from(&coo);
        let adata = AnnData::<H5>::new(path)?;
        adata.set_obs_names((0..nr).map(|i| format!("c{i}")).collect::<Vec<_>>().into())?;
        adata.set_var_names((0..nc).map(|j| format!("g{j}")).collect::<Vec<_>>().into())?;
        adata.set_x(ArrayData::CsrMatrix(DynCsrMatrix::F32(csr)))?;
        adata.close()?;
        Ok(())
    }

    fn read_dense(path: &Path) -> anyhow::Result<ndarray::Array2<f64>> {
        let adata = AnnData::<H5>::open(H5::open(path)?)?;
        let x = adata.x().get::<ArrayData>()?.unwrap();
        // Densify the CSR ourselves (independent of crate conversion helpers).
        let dense = match x {
            ArrayData::CsrMatrix(DynCsrMatrix::F32(m)) => {
                let mut d = ndarray::Array2::<f64>::zeros((m.nrows(), m.ncols()));
                for (i, row) in m.row_iter().enumerate() {
                    for (&j, &v) in row.col_indices().iter().zip(row.values().iter()) {
                        d[[i, j]] = v as f64;
                    }
                }
                d
            }
            other => anyhow::bail!("unexpected X type {:?}", other),
        };
        adata.close()?;
        Ok(dense)
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(name);
        let _ = std::fs::remove_file(&p);
        p
    }



    /// Streaming log1p over a backed file must equal elementwise ln(1+x), with a tiny chunk size
    /// so multiple chunks are exercised.
    #[test]
    fn ooc_log1p_matches_dense() -> anyhow::Result<()> {
        let inp = tmp("sr_ooc_log1p_in.h5ad");
        let out = tmp("sr_ooc_log1p_out.h5ad");
        let rows = vec![
            vec![1.0, 0.0, 3.0],
            vec![0.0, 7.0, 0.0],
            vec![2.0, 2.0, 2.0],
            vec![0.0, 0.0, 0.0],
            vec![5.0, 0.0, 1.0],
        ];
        write_fixture(&inp, &rows)?;
        log1p_backed(&inp, &out, Some(2))?; // chunk_size=2 -> 3 chunks

        let got = read_dense(&out)?;
        for (i, r) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                assert!((got[[i, j]] - (v).ln_1p()).abs() < 1e-6, "[{i},{j}] {}", got[[i, j]]);
            }
        }
        for p in [inp, out] {
            std::fs::remove_file(p).ok();
        }
        Ok(())
    }

    /// Streaming normalize_total(1e4)+log1p must match the row-scaled-then-log1p reference.
    #[test]
    fn ooc_normalize_total_matches_reference() -> anyhow::Result<()> {
        let inp = tmp("sr_ooc_norm_in.h5ad");
        let out = tmp("sr_ooc_norm_out.h5ad");
        let rows = vec![
            vec![1.0, 3.0, 0.0], // sum 4
            vec![0.0, 0.0, 0.0], // zero row -> untouched
            vec![2.0, 2.0, 6.0], // sum 10
        ];
        write_fixture(&inp, &rows)?;
        normalize_total_backed(&inp, &out, 1e4, true, Some(2))?;

        let got = read_dense(&out)?;
        for (i, r) in rows.iter().enumerate() {
            let total: f64 = r.iter().sum();
            for (j, &v) in r.iter().enumerate() {
                let expect = if total > 0.0 { (v / total * 1e4).ln_1p() } else { 0.0 };
                assert!((got[[i, j]] - expect).abs() < 1e-3, "[{i},{j}] {} vs {}", got[[i, j]], expect);
            }
        }
        for p in [inp, out] {
            std::fs::remove_file(p).ok();
        }
        Ok(())
    }
}

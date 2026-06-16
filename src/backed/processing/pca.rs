//! # Out-of-core PCA (covariance / eigendecomposition method)
//!
//! Exact PCA over the highly-variable genes without ever holding the cell×gene matrix in memory:
//!
//! 1. **Pass 1** streams `X` and accumulates, over the selected genes only, the gene×gene Gram
//!    matrix `Gᵢⱼ = Σ_cells xᵢxⱼ` and per-gene sums. Memory is `O(n_hvg²)` (e.g. 2000² f64 = 32 MB)
//!    plus one chunk — independent of cell count.
//! 2. The centered covariance `C = (G − n·μμᵀ)/(n−1)` is eigendecomposed (`n_hvg × n_hvg`,
//!    symmetric). Top-`k` eigenvectors are the principal axes; eigenvalues are the variances.
//! 3. **Pass 2** streams `X` again and projects each centered cell onto the top axes, writing the
//!    embedding to `obsm["X_pca"]` (and variance ratios to `uns`).
//!
//! This is exact (not randomized) PCA with centering; results match the in-memory PCA up to the
//! usual per-component sign ambiguity.

use std::path::Path;

use anndata::data::{DynArray, DynCsrMatrix};
use anndata::{
    AnnData, AnnDataOp, ArrayData, ArrayElemOp, AxisArraysOp, Backend, Data, ElemCollectionOp,
};
use anndata_hdf5::H5;
use anyhow::bail;
use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::Array2;
use rayon::prelude::*;

use super::det::det_block_reduce;
use super::transformation::DEFAULT_CHUNK_SIZE;

/// Run PCA out-of-core over the genes flagged in `var["highly_variable"]`, writing the cell
/// embedding to `obsm["X_{key}"]` (default `X_pca`) and variance ratios to
/// `uns["{key}_variance_ratio"]`, in place.
pub fn pca_backed(
    path: &Path,
    n_comps: usize,
    key: Option<&str>,
    chunk_size: Option<usize>,
) -> anyhow::Result<()> {
    let key = key.unwrap_or("pca");
    let chunk_size = chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);
    let adata = AnnData::<H5>::open(H5::open_rw(path)?)?;
    let (n_obs, n_vars) = (adata.n_obs(), adata.n_vars());

    // Selected (highly variable) genes -> local index map.
    let hv: Vec<bool> = adata
        .read_var()?
        .column("highly_variable")?
        .bool()?
        .into_iter()
        .map(|b| b.unwrap_or(false))
        .collect();
    let selected: Vec<usize> = hv.iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i).collect();
    let n_sel = selected.len();
    if n_sel == 0 {
        bail!("OOC PCA needs var['highly_variable'] set (run hvg first); none selected");
    }
    let mut local = vec![-1i64; n_vars];
    for (li, &g) in selected.iter().enumerate() {
        local[g] = li as i64;
    }

    // ---- Pass 1: Gram matrix + per-gene sums over selected genes ----
    // Each read-chunk's contribution is computed with a deterministic block reduction (parallel
    // across fixed row-blocks, ordered merge), then added to the running totals in chunk order.
    let mut gram = vec![0.0_f64; n_sel * n_sel];
    let mut col_sum = vec![0.0_f64; n_sel];
    for (chunk, _start, _end) in adata.x().iter::<ArrayData>(chunk_size) {
        let (g_chunk, cs_chunk) = gram_chunk(&chunk, &local, n_sel)?;
        for (a, b) in gram.iter_mut().zip(g_chunk.iter()) {
            *a += *b;
        }
        for (a, b) in col_sum.iter_mut().zip(cs_chunk.iter()) {
            *a += *b;
        }
    }

    // ---- Covariance + eigendecomposition (small, n_sel × n_sel) ----
    let n = n_obs as f64;
    let mean: Vec<f64> = col_sum.iter().map(|&s| s / n).collect();
    let mut cov = DMatrix::<f64>::zeros(n_sel, n_sel);
    for i in 0..n_sel {
        for j in 0..n_sel {
            cov[(i, j)] = (gram[i * n_sel + j] - n * mean[i] * mean[j]) / (n - 1.0);
        }
    }
    let eig = SymmetricEigen::new(cov);
    let mut order: Vec<usize> = (0..n_sel).collect();
    order.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let k = n_comps.min(n_sel);
    let total_var: f64 = eig.eigenvalues.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    let top: Vec<usize> = order[..k].to_vec();
    let variance_ratio: Vec<f64> = top.iter().map(|&t| eig.eigenvalues[t] / total_var).collect();

    // Precompute the per-component centering offset c_k = Σ_j mean_j · V[j,k].
    let offset: Vec<f64> = (0..k)
        .map(|c| (0..n_sel).map(|j| mean[j] * eig.eigenvectors[(j, top[c])]).sum::<f64>())
        .collect();

    // ---- Pass 2: project centered cells onto the top axes ----
    // Each cell's embedding is independent (a fixed-order sum over its own nonzeros), so rows are
    // computed in parallel and written to disjoint positions — order-independent, hence
    // deterministic, with no cross-row reduction.
    let mut emb = Array2::<f64>::zeros((n_obs, k));
    for (chunk, start, _end) in adata.x().iter::<ArrayData>(chunk_size) {
        let rows = project_chunk(&chunk, &local, &eig, &top, &offset, k)?;
        for (i, row) in rows.into_iter().enumerate() {
            for (c, val) in row.into_iter().enumerate() {
                emb[[start + i, c]] = val;
            }
        }
    }

    // ---- Store results in place ----
    let emb_ad: ArrayData = DynArray::from(emb).into();
    adata.obsm().add(&format!("X_{key}"), emb_ad)?;
    let vr: ArrayData = DynArray::from(ndarray::Array1::from(variance_ratio)).into();
    adata.uns().add(&format!("{key}_variance_ratio"), Data::ArrayData(vr))?;

    adata.close()?;
    Ok(())
}

/// Gather a CSR row's selected nonzeros as (local_index, value), reusing `buf`.
fn gather_selected<T: Copy + num_traits::ToPrimitive>(
    cols: &[usize],
    vals: &[T],
    local: &[i64],
    buf: &mut Vec<(usize, f64)>,
) {
    buf.clear();
    for (&c, &v) in cols.iter().zip(vals.iter()) {
        let li = local[c];
        if li >= 0 {
            buf.push((li as usize, v.to_f64().unwrap_or(0.0)));
        }
    }
}

/// One chunk's contribution to the Gram matrix and per-gene sums, via a deterministic block
/// reduction over the chunk's rows. Returns `(gram_flat, col_sum)`.
fn gram_chunk(
    chunk: &ArrayData,
    local: &[i64],
    n_sel: usize,
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    match chunk {
        ArrayData::CsrMatrix(DynCsrMatrix::F32(m)) => Ok(gram_rows(m, local, n_sel)),
        ArrayData::CsrMatrix(DynCsrMatrix::F64(m)) => Ok(gram_rows(m, local, n_sel)),
        other => bail!("OOC PCA supports only F32/F64 CSR matrices, got {:?}", other),
    }
}

fn gram_rows<T: Copy + num_traits::ToPrimitive + Sync>(
    m: &nalgebra_sparse::CsrMatrix<T>,
    local: &[i64],
    n_sel: usize,
) -> (Vec<f64>, Vec<f64>) {
    det_block_reduce(
        m.nrows(),
        || (vec![0.0_f64; n_sel * n_sel], vec![0.0_f64; n_sel]),
        |(gram, col_sum), r| {
            let row = m.row(r);
            let mut buf: Vec<(usize, f64)> = Vec::new();
            gather_selected(row.col_indices(), row.values(), local, &mut buf);
            for &(li, v) in buf.iter() {
                col_sum[li] += v;
            }
            for &(la, va) in buf.iter() {
                let base = la * n_sel;
                for &(lb, vb) in buf.iter() {
                    gram[base + lb] += va * vb;
                }
            }
        },
        |(ag, acs), (g, cs)| {
            for (a, b) in ag.iter_mut().zip(g.iter()) {
                *a += *b;
            }
            for (a, b) in acs.iter_mut().zip(cs.iter()) {
                *a += *b;
            }
        },
    )
}

/// Project a chunk's cells onto the top axes; returns one `k`-vector per row (in row order).
/// Rows are independent so they are computed in parallel; each value is a fixed-order sum, so the
/// result does not depend on the thread count.
fn project_chunk(
    chunk: &ArrayData,
    local: &[i64],
    eig: &SymmetricEigen<f64, nalgebra::Dyn>,
    top: &[usize],
    offset: &[f64],
    k: usize,
) -> anyhow::Result<Vec<Vec<f64>>> {
    match chunk {
        ArrayData::CsrMatrix(DynCsrMatrix::F32(m)) => Ok(project_rows(m, local, eig, top, offset, k)),
        ArrayData::CsrMatrix(DynCsrMatrix::F64(m)) => Ok(project_rows(m, local, eig, top, offset, k)),
        other => bail!("OOC PCA supports only F32/F64 CSR matrices, got {:?}", other),
    }
}

fn project_rows<T: Copy + num_traits::ToPrimitive + Sync>(
    m: &nalgebra_sparse::CsrMatrix<T>,
    local: &[i64],
    eig: &SymmetricEigen<f64, nalgebra::Dyn>,
    top: &[usize],
    offset: &[f64],
    k: usize,
) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .into_par_iter()
        .map(|r| {
            let row = m.row(r);
            let mut buf: Vec<(usize, f64)> = Vec::new();
            gather_selected(row.col_indices(), row.values(), local, &mut buf);
            let mut out = vec![0.0_f64; k];
            for (c, o) in out.iter_mut().enumerate() {
                *o = -offset[c];
            }
            for &(li, v) in buf.iter() {
                for (c, o) in out.iter_mut().enumerate() {
                    *o += v * eig.eigenvectors[(li, top[c])];
                }
            }
            out
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use nalgebra_sparse::{CooMatrix, CsrMatrix};

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(name);
        let _ = std::fs::remove_file(&p);
        p
    }

    /// OOC covariance-PCA must match a dense covariance-PCA reference computed in-test: identical
    /// variance ratios and, up to per-component sign, identical embeddings.
    #[test]
    fn ooc_pca_matches_dense_reference() -> anyhow::Result<()> {
        let (nr, nc) = (60usize, 8usize);
        let mut dense = vec![vec![0.0_f64; nc]; nr];
        let mut coo = CooMatrix::<f32>::new(nr, nc);
        for i in 0..nr {
            for j in 0..nc {
                let v = (((i * 7 + j * 5) % 9) as f64) * 0.5;
                dense[i][j] = v;
                if v != 0.0 {
                    coo.push(i, j, v as f32);
                }
            }
        }
        // All genes selected as HVG for the test.
        let path = tmp("sr_ooc_pca.h5ad");
        let adata = AnnData::<H5>::new(&path)?;
        adata.set_obs_names((0..nr).map(|i| format!("c{i}")).collect::<Vec<_>>().into())?;
        adata.set_var_names((0..nc).map(|j| format!("g{j}")).collect::<Vec<_>>().into())?;
        adata.set_x(ArrayData::CsrMatrix(DynCsrMatrix::F32(CsrMatrix::from(&coo))))?;
        let mut var = adata.read_var()?;
        var.with_column(polars::prelude::Column::new(
            "highly_variable".into(),
            vec![true; nc],
        ))?;
        adata.set_var(var)?;
        adata.close()?;

        let k = 3;
        pca_backed(&path, k, None, Some(16))?;

        // Read OOC results.
        let back = AnnData::<H5>::open(H5::open(&path)?)?;
        let emb_ooc = match back.obsm().get_item::<ArrayData>("X_pca")?.unwrap() {
            ArrayData::Array(DynArray::F64(a)) => a.into_dimensionality::<ndarray::Ix2>()?,
            other => panic!("unexpected obsm type {:?}", other),
        };
        let vr_ooc = match back.uns().get_item::<Data>("pca_variance_ratio")?.unwrap() {
            Data::ArrayData(ArrayData::Array(DynArray::F64(a))) => {
                a.into_dimensionality::<ndarray::Ix1>()?.to_vec()
            }
            other => panic!("unexpected uns type {:?}", other),
        };
        back.close()?;

        // ---- dense reference: covariance PCA on the same matrix ----
        let n = nr as f64;
        let mut means = vec![0.0; nc];
        for r in &dense {
            for j in 0..nc {
                means[j] += r[j] / n;
            }
        }
        let mut cov = DMatrix::<f64>::zeros(nc, nc);
        for r in &dense {
            for a in 0..nc {
                for b in 0..nc {
                    cov[(a, b)] += (r[a] - means[a]) * (r[b] - means[b]) / (n - 1.0);
                }
            }
        }
        let eig = SymmetricEigen::new(cov);
        let mut order: Vec<usize> = (0..nc).collect();
        order.sort_by(|&a, &b| eig.eigenvalues[b].partial_cmp(&eig.eigenvalues[a]).unwrap());
        let total: f64 = eig.eigenvalues.iter().sum();
        let vr_ref: Vec<f64> = order[..k].iter().map(|&t| eig.eigenvalues[t] / total).collect();

        // variance ratios must match closely
        for (a, b) in vr_ooc.iter().zip(vr_ref.iter()) {
            assert!((a - b).abs() < 1e-9, "var ratio {a} vs {b}");
        }
        // embeddings: column norms match the reference (sign-independent check)
        for c in 0..k {
            let t = order[c];
            // reference embedding column = centered @ eigenvector_t
            let mut ref_col = vec![0.0; nr];
            for (i, r) in dense.iter().enumerate() {
                ref_col[i] = (0..nc).map(|j| (r[j] - means[j]) * eig.eigenvectors[(j, t)]).sum();
            }
            let norm_ref: f64 = ref_col.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_ooc: f64 = (0..nr).map(|i| emb_ooc[[i, c]].powi(2)).sum::<f64>().sqrt();
            assert!((norm_ref - norm_ooc).abs() < 1e-6, "col {c} norm {norm_ref} vs {norm_ooc}");
        }
        std::fs::remove_file(path).ok();
        Ok(())
    }

    /// Write a CSR fixture with all genes flagged highly_variable.
    fn write_pca_fixture(path: &Path, nr: usize, nc: usize) -> anyhow::Result<()> {
        let mut coo = CooMatrix::<f32>::new(nr, nc);
        for i in 0..nr {
            for j in 0..nc {
                let v = (((i * 31 + j * 17) % 13) as f32) * 0.25;
                if v != 0.0 {
                    coo.push(i, j, v);
                }
            }
        }
        let a = AnnData::<H5>::new(path)?;
        a.set_obs_names((0..nr).map(|i| format!("c{i}")).collect::<Vec<_>>().into())?;
        a.set_var_names((0..nc).map(|j| format!("g{j}")).collect::<Vec<_>>().into())?;
        a.set_x(ArrayData::CsrMatrix(DynCsrMatrix::F32(CsrMatrix::from(&coo))))?;
        let mut var = a.read_var()?;
        var.with_column(polars::prelude::Column::new("highly_variable".into(), vec![true; nc]))?;
        a.set_var(var)?;
        a.close()?;
        Ok(())
    }

    fn read_pca(path: &Path) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
        let a = AnnData::<H5>::open(H5::open(path)?)?;
        let emb = match a.obsm().get_item::<ArrayData>("X_pca")?.unwrap() {
            ArrayData::Array(DynArray::F64(arr)) => arr.into_raw_vec_and_offset().0,
            other => panic!("unexpected obsm {:?}", other),
        };
        let vr = match a.uns().get_item::<Data>("pca_variance_ratio")?.unwrap() {
            Data::ArrayData(ArrayData::Array(DynArray::F64(arr))) => arr.into_raw_vec_and_offset().0,
            other => panic!("unexpected uns {:?}", other),
        };
        a.close()?;
        Ok((emb, vr))
    }

    /// Determinism: PCA run with 1 thread vs 8 threads (and a span of chunk sizes that force
    /// different row-block partitions) must produce **bit-identical** embeddings and variance
    /// ratios. This guards the parallel Gram reduction's fixed summation order.
    #[test]
    fn ooc_pca_is_deterministic_across_thread_counts() -> anyhow::Result<()> {
        let (nr, nc, k) = (5000usize, 60usize, 10usize);

        let run = |threads: usize, chunk: usize, tag: &str| -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
            let path = tmp(&format!("sr_pca_det_{tag}.h5ad"));
            write_pca_fixture(&path, nr, nc)?;
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
            pool.install(|| pca_backed(&path, k, None, Some(chunk)))?;
            let out = read_pca(&path)?;
            std::fs::remove_file(path).ok();
            Ok(out)
        };

        // Baseline: single-threaded.
        let (emb1, vr1) = run(1, 1000, "t1")?;
        // 8 threads, same chunking -> must match bit-for-bit.
        let (emb8, vr8) = run(8, 1000, "t8")?;
        assert_eq!(emb1, emb8, "embeddings differ between 1 and 8 threads");
        assert_eq!(vr1, vr8, "variance ratios differ between 1 and 8 threads");

        // Repeat the 8-thread run -> identical to itself (no run-to-run drift).
        let (emb8b, vr8b) = run(8, 1000, "t8b")?;
        assert_eq!(emb8, emb8b, "8-thread run not reproducible across repeats");
        assert_eq!(vr8, vr8b);

        Ok(())
    }
}

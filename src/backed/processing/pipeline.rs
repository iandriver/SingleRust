//! # Fused out-of-core preprocessing
//!
//! `preprocess_backed` runs QC + `normalize_total` + `log1p` against a disk-backed `.h5ad` in a
//! single fused job, writing one output file. The key saving over calling the three ops
//! separately: the per-cell total computed for QC **is** the row-sum the normalization needs, so
//! it is computed once. Two streamed passes total (QC accumulate, then normalize+log1p write),
//! instead of the ~4 read/write passes the separate CLI commands incur — roughly halving I/O.

use std::path::Path;

use anndata::{AnnData, AnnDataOp, ArrayData, ArrayElemOp, Backend};
use anndata_hdf5::H5;

use super::qc::{stream_qc, write_metrics};
use super::transformation::{normalize_chunk, DEFAULT_CHUNK_SIZE};

/// QC (on raw counts) + `normalize_total(target_sum)` + optional `log1p`, fused. Writes a new
/// `.h5ad` at `output` with QC metrics in obs/var and the normalized (log1p) matrix in `X`.
pub fn preprocess_backed(
    input: &Path,
    output: &Path,
    target_sum: f64,
    log1p: bool,
    chunk_size: Option<usize>,
) -> anyhow::Result<()> {
    let chunk_size = chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);
    let adata = AnnData::<H5>::open(H5::open(input)?)?;
    let n_obs = adata.n_obs();

    // Pass 1: QC on raw counts. acc.total holds per-cell totals == normalization row-sums.
    let (acc, mito_mask) = stream_qc(&adata, chunk_size)?;

    // Output: copy obs/var (+ names), then layer the QC metrics on top.
    let out = AnnData::<H5>::new(output)?;
    out.set_obs_names(adata.obs_names())?;
    out.set_var_names(adata.var_names())?;
    out.set_obs(adata.read_obs()?)?;
    out.set_var(adata.read_var()?)?;
    write_metrics(&out, &mito_mask, &acc, n_obs)?;

    // Pass 2: normalize (reusing the totals from pass 1) + log1p, streamed straight to X.
    let sums = acc.total;
    let iter = adata
        .x()
        .iter::<ArrayData>(chunk_size)
        .map(move |(chunk, start, _end)| {
            normalize_chunk(chunk, start, &sums, target_sum, log1p).expect("normalize chunk")
        });
    out.set_x_from_iter(iter)?;

    out.close()?;
    adata.close()?;
    Ok(())
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

    #[test]
    fn preprocess_fused_matches_separate_semantics() -> anyhow::Result<()> {
        let inp = tmp("sr_pre_in.h5ad");
        let out = tmp("sr_pre_out.h5ad");
        let rows = vec![
            vec![10.0_f64, 0.0, 5.0, 5.0],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ];
        let (nr, nc) = (rows.len(), rows[0].len());
        let mut coo = CooMatrix::<f32>::new(nr, nc);
        for (i, r) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                if v != 0.0 {
                    coo.push(i, j, v as f32);
                }
            }
        }
        let adata = AnnData::<H5>::new(&inp)?;
        adata.set_obs_names((0..nr).map(|i| format!("c{i}")).collect::<Vec<_>>().into())?;
        adata.set_var_names(["MT-a", "g1", "g2", "g3"].iter().map(|s| s.to_string()).collect::<Vec<_>>().into())?;
        adata.set_x(ArrayData::CsrMatrix(DynCsrMatrix::F32(CsrMatrix::from(&coo))))?;
        adata.close()?;

        preprocess_backed(&inp, &out, 1e4, true, Some(2))?;

        // Verify: obs has QC (total_counts on raw), X is normalize_total+log1p.
        let res = AnnData::<H5>::open(H5::open(&out)?)?;
        let obs = res.read_obs()?;
        let total = obs.column("total_counts")?.f64()?;
        assert_eq!(total.get(0), Some(20.0)); // raw total of row 0
        assert_eq!(total.get(1), Some(10.0));
        let pct_mito = obs.column("pct_counts_mito")?.f64()?;
        assert!((pct_mito.get(0).unwrap() - 50.0).abs() < 1e-6); // 10/20

        let x = res.x().get::<ArrayData>()?.unwrap();
        let m = match x {
            ArrayData::CsrMatrix(DynCsrMatrix::F32(m)) => m,
            _ => panic!("expected CSR f32"),
        };
        // row 0, col 0: raw 10, total 20 -> 10/20*1e4 = 5000 -> ln1p(5000)
        let row0: Vec<(usize, f32)> = m.row(0).col_indices().iter().zip(m.row(0).values()).map(|(&c, &v)| (c, v)).collect();
        let v00 = row0.iter().find(|(c, _)| *c == 0).unwrap().1 as f64;
        assert!((v00 - (5000.0_f64).ln_1p()).abs() < 1e-2, "v00={v00}");

        res.close()?;
        for p in [inp, out] {
            std::fs::remove_file(p).ok();
        }
        Ok(())
    }
}

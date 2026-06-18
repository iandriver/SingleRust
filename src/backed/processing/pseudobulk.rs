//! # Out-of-core pseudobulk (sample × group aggregation)
//!
//! decoupler-style pseudobulk: aggregate single-cell counts into one profile per
//! `sample × group` combination. decoupler's implementation loops the full sample×group cartesian
//! product and, for each, boolean-masks **all** cells and **densifies** the submatrix
//! (`X[mask].toarray()`), holding the whole input in memory — `O(n_obs · n_groups)` masking plus
//! per-group densification.
//!
//! This does it in a **single streaming pass**: each cell is read once (sparse) and scatter-added
//! into its group's accumulator. Cost is `O(nnz)`, input memory is one chunk, and only the small
//! `groups × genes` output is held. With `mode = "sum"` (the default, on raw counts) the result
//! matches decoupler's `psbulk` / `psbulk_cells` / `psbulk_counts` / `psbulk_props`.
//!
//! Aggregation is a sequential scatter-add (each cell touches one output row), which is
//! deterministic by construction — repeat runs are bit-identical.

// Several loops index flat (row*n_genes + col) buffers or build the sample×group cartesian
// product, where explicit range loops are the clearest form.
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::path::Path;

use anndata::data::{DynArray, DynCsrMatrix};
use anndata::{AnnData, AnnDataOp, ArrayData, ArrayElemOp, Backend};
use anndata_hdf5::H5;
use anyhow::{bail, Context};
use ndarray::Array2;
use num_traits::ToPrimitive;
use polars::prelude::{Column, DataType};

use super::transformation::DEFAULT_CHUNK_SIZE;

/// Aggregation mode. `Sum` (decoupler default, raw counts) and `Mean` (sum / n_cells).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PseudobulkMode {
    Sum,
    Mean,
}

/// Out-of-core pseudobulk by `sample_col` (and optional `group_col`), written to `output`.
///
/// Output rows are the `sample × group` cartesian product in decoupler order (group outer, sample
/// inner), indexed `"{sample}_{group}"`. `X` holds the aggregate (sum or mean); `obs` carries the
/// sample/group labels plus `psbulk_cells` and `psbulk_counts`; `layers["psbulk_props"]` holds the
/// per-gene fraction of cells that were non-zero.
pub fn pseudobulk_backed(
    input: &Path,
    output: &Path,
    sample_col: &str,
    group_col: Option<&str>,
    mode: PseudobulkMode,
    chunk_size: Option<usize>,
) -> anyhow::Result<()> {
    let chunk_size = chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);
    let adata = AnnData::<H5>::open(H5::open(input)?)?;
    let n_genes = adata.n_vars();
    let obs = adata.read_obs()?;

    let samples = col_as_strings(&obs, sample_col)?;
    let groups = match group_col {
        Some(g) => Some(col_as_strings(&obs, g)?),
        None => None,
    };

    // Unique, sorted sample (and group) labels -> stable row layout.
    let smp_labels = sorted_unique(&samples);
    let grp_labels = groups
        .as_ref()
        .map(|g| sorted_unique(g))
        .unwrap_or_else(|| vec![String::new()]); // single empty "group" when group_col is None
    let (n_smp, n_grp) = (smp_labels.len(), grp_labels.len());
    let n_rows = n_smp * n_grp;

    let smp_idx: HashMap<&str, usize> =
        smp_labels.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let grp_idx: HashMap<&str, usize> =
        grp_labels.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();

    // Per-cell output-row assignment (usize::MAX = skip: missing sample/group).
    let n_obs = samples.len();
    let mut cell_row = vec![usize::MAX; n_obs];
    for i in 0..n_obs {
        let si = match &samples[i] {
            Some(s) => smp_idx[s.as_str()],
            None => continue,
        };
        let gi = match &groups {
            Some(g) => match &g[i] {
                Some(v) => grp_idx[v.as_str()],
                None => continue,
            },
            None => 0,
        };
        cell_row[i] = gi * n_smp + si; // group-major (decoupler order)
    }

    // Accumulators (output is small: n_rows × n_genes).
    let mut psbulk = vec![0.0_f64; n_rows * n_genes];
    let mut nnz = vec![0.0_f64; n_rows * n_genes]; // non-zero counts -> props
    let mut ncells = vec![0.0_f64; n_rows];
    let mut counts = vec![0.0_f64; n_rows];

    // ---- single streaming pass: scatter-add each cell into its group row ----
    for (chunk, start, _end) in adata.x().iter::<ArrayData>(chunk_size) {
        scatter_add(&chunk, start, &cell_row, n_genes, &mut psbulk, &mut nnz, &mut ncells, &mut counts)?;
    }

    // Finalize: mean if requested, props = nnz / n_cells.
    let mut props = vec![0.0_f64; n_rows * n_genes];
    for r in 0..n_rows {
        let nc = ncells[r];
        if nc > 0.0 {
            for g in 0..n_genes {
                let idx = r * n_genes + g;
                props[idx] = nnz[idx] / nc;
                if mode == PseudobulkMode::Mean {
                    psbulk[idx] /= nc;
                }
            }
        }
    }

    write_output(
        &adata, output, sample_col, group_col, &smp_labels, &grp_labels, n_smp, n_grp, n_genes,
        psbulk, props, &ncells, &counts,
    )?;
    adata.close()?;
    Ok(())
}

/// Read an obs column as `Vec<Option<String>>` (casts categorical/other dtypes to string).
fn col_as_strings(obs: &polars::frame::DataFrame, name: &str) -> anyhow::Result<Vec<Option<String>>> {
    let col = obs
        .column(name)
        .with_context(|| format!("obs column '{name}' not found"))?
        .cast(&DataType::String)?;
    Ok(col.str()?.into_iter().map(|o| o.map(|s| s.to_string())).collect())
}

fn sorted_unique(vals: &[Option<String>]) -> Vec<String> {
    let mut set: Vec<String> = vals.iter().flatten().cloned().collect();
    set.sort_unstable();
    set.dedup();
    set
}

#[allow(clippy::too_many_arguments)]
fn scatter_add(
    chunk: &ArrayData,
    start: usize,
    cell_row: &[usize],
    n_genes: usize,
    psbulk: &mut [f64],
    nnz: &mut [f64],
    ncells: &mut [f64],
    counts: &mut [f64],
) -> anyhow::Result<()> {
    macro_rules! go {
        ($m:expr) => {{
            for (i, row) in $m.row_iter().enumerate() {
                let r = cell_row[start + i];
                if r == usize::MAX {
                    continue;
                }
                ncells[r] += 1.0;
                let base = r * n_genes;
                for (&c, &v) in row.col_indices().iter().zip(row.values().iter()) {
                    let v = v.to_f64().unwrap_or(0.0);
                    psbulk[base + c] += v;
                    nnz[base + c] += 1.0;
                    counts[r] += v;
                }
            }
        }};
    }
    match chunk {
        ArrayData::CsrMatrix(DynCsrMatrix::F32(m)) => go!(m),
        ArrayData::CsrMatrix(DynCsrMatrix::F64(m)) => go!(m),
        other => bail!("OOC pseudobulk supports only F32/F64 CSR matrices, got {:?}", other),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_output(
    adata: &AnnData<H5>,
    output: &Path,
    sample_col: &str,
    group_col: Option<&str>,
    smp_labels: &[String],
    grp_labels: &[String],
    n_smp: usize,
    n_grp: usize,
    n_genes: usize,
    psbulk: Vec<f64>,
    props: Vec<f64>,
    ncells: &[f64],
    counts: &[f64],
) -> anyhow::Result<()> {
    let n_rows = n_smp * n_grp;
    // Row metadata in (group outer, sample inner) order.
    let mut index = Vec::with_capacity(n_rows);
    let mut samp_col = Vec::with_capacity(n_rows);
    let mut grp_col = Vec::with_capacity(n_rows);
    for gi in 0..n_grp {
        for si in 0..n_smp {
            let s = &smp_labels[si];
            let g = &grp_labels[gi];
            index.push(if group_col.is_some() { format!("{s}_{g}") } else { s.clone() });
            samp_col.push(s.clone());
            grp_col.push(g.clone());
        }
    }

    let out = AnnData::<H5>::new(output)?;
    out.set_obs_names(index.into())?;
    out.set_var_names(adata.var_names())?;
    out.set_var(adata.read_var()?)?;

    let mut obs = polars::frame::DataFrame::default();
    obs.with_column(Column::new(sample_col.into(), samp_col))?;
    if let Some(g) = group_col {
        obs.with_column(Column::new(g.into(), grp_col))?;
    }
    obs.with_column(Column::new("psbulk_cells".into(), ncells.to_vec()))?;
    obs.with_column(Column::new("psbulk_counts".into(), counts.to_vec()))?;
    out.set_obs(obs)?;

    let x = Array2::from_shape_vec((n_rows, n_genes), psbulk)?;
    out.set_x(ArrayData::Array(DynArray::from(x)))?;
    let props_arr = Array2::from_shape_vec((n_rows, n_genes), props)?;
    out.set_layers([(
        "psbulk_props".to_string(),
        ArrayData::Array(DynArray::from(props_arr)),
    )])?;

    out.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::{CooMatrix, CsrMatrix};

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(name);
        let _ = std::fs::remove_file(&p);
        p
    }

    /// Pseudobulk sum must equal the per-group column sums; cells/counts/props correct.
    #[test]
    fn pseudobulk_sum_matches_manual() -> anyhow::Result<()> {
        // 5 cells × 3 genes; samples s1/s2, groups gA/gB.
        let rows = vec![
            (vec![1.0, 0.0, 2.0], "s1", "gA"),
            (vec![3.0, 1.0, 0.0], "s1", "gA"),
            (vec![0.0, 5.0, 0.0], "s1", "gB"),
            (vec![2.0, 2.0, 2.0], "s2", "gA"),
            (vec![0.0, 0.0, 9.0], "s2", "gB"),
        ];
        let (nr, nc) = (rows.len(), 3);
        let mut coo = CooMatrix::<f32>::new(nr, nc);
        for (i, (r, _, _)) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                if v != 0.0 {
                    coo.push(i, j, v as f32);
                }
            }
        }
        let inp = tmp("sr_psb_in.h5ad");
        let out = tmp("sr_psb_out.h5ad");
        let a = AnnData::<H5>::new(&inp)?;
        a.set_obs_names((0..nr).map(|i| format!("c{i}")).collect::<Vec<_>>().into())?;
        a.set_var_names(vec!["g0".to_string(), "g1".into(), "g2".into()].into())?;
        a.set_x(ArrayData::CsrMatrix(DynCsrMatrix::F32(CsrMatrix::from(&coo))))?;
        let mut obs = polars::frame::DataFrame::default();
        obs.with_column(Column::new(
            "sample".into(),
            rows.iter().map(|(_, s, _)| s.to_string()).collect::<Vec<_>>(),
        ))?;
        obs.with_column(Column::new(
            "celltype".into(),
            rows.iter().map(|(_, _, g)| g.to_string()).collect::<Vec<_>>(),
        ))?;
        a.set_obs(obs)?;
        a.close()?;

        pseudobulk_backed(&inp, &out, "sample", Some("celltype"), PseudobulkMode::Sum, Some(2))?;

        let res = AnnData::<H5>::open(H5::open(&out)?)?;
        let names = res.obs_names().into_vec();
        let x = match res.x().get::<ArrayData>()?.unwrap() {
            ArrayData::Array(DynArray::F64(a)) => a.into_dimensionality::<ndarray::Ix2>()?,
            other => panic!("unexpected X {:?}", other),
        };
        let obs = res.read_obs()?;
        let cells = obs.column("psbulk_cells")?.f64()?;
        let counts = obs.column("psbulk_counts")?.f64()?;

        // expected sums per (sample,group)
        let expect: std::collections::HashMap<&str, [f64; 3]> = [
            ("s1_gA", [4.0, 1.0, 2.0]), // cells 0+1
            ("s2_gA", [2.0, 2.0, 2.0]), // cell 3
            ("s1_gB", [0.0, 5.0, 0.0]), // cell 2
            ("s2_gB", [0.0, 0.0, 9.0]), // cell 4
        ]
        .into_iter()
        .collect();
        for (i, name) in names.iter().enumerate() {
            let e = expect.get(name.as_str()).unwrap_or(&[0.0, 0.0, 0.0]);
            for j in 0..3 {
                assert!((x[[i, j]] - e[j]).abs() < 1e-6, "{name}[{j}] {} vs {}", x[[i, j]], e[j]);
            }
        }
        // s1_gA has 2 cells summing to 7
        let pos = names.iter().position(|n| n == "s1_gA").unwrap();
        assert_eq!(cells.get(pos), Some(2.0));
        assert_eq!(counts.get(pos), Some(7.0));

        res.close()?;
        for p in [inp, out] {
            std::fs::remove_file(p).ok();
        }
        Ok(())
    }
}

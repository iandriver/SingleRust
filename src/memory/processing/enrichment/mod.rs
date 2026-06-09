//! # Gene Set Enrichment / Pathway Activity
//!
//! This module provides [decoupler](https://github.com/scverse/decoupler)-style enrichment
//! methods that score, for each cell, how active a set of pathways or gene programs is. The
//! prior knowledge is supplied as a [`PathwayNetwork`] (for example built from OmniPath via
//! the `connectors` module), whose feature indices must align with the columns of `adata.x()`.
//!
//! Currently implemented:
//! - **ORA** ([`ora_csr`] / [`run_ora`]): Over-Representation Analysis via the one-sided
//!   hypergeometric (Fisher) test.
//!
//! These methods are gated behind the `enrichment` cargo feature.

mod utils;

#[cfg(feature = "enrichment")]
mod ora;

#[cfg(feature = "enrichment")]
pub use ora::{ora_csr, OraResult};

#[cfg(feature = "enrichment")]
use anndata::data::DynCsrMatrix;
#[cfg(feature = "enrichment")]
use anndata::ArrayData;
#[cfg(feature = "enrichment")]
use anndata_memory::IMAnnData;
#[cfg(feature = "enrichment")]
use single_utilities::types::PathwayNetwork;
#[cfg(feature = "enrichment")]
use std::ops::Deref;

/// Run Over-Representation Analysis (ORA) for every cell in an [`IMAnnData`].
///
/// Convenience wrapper around [`ora_csr`] that reads the expression matrix from `adata.x()`.
/// Only F32/F64 CSR matrices are supported (convert first otherwise). The `net` feature
/// indices must refer to the columns (genes) of `adata.x()` in order — build the
/// [`PathwayNetwork`] with `features = adata.var_names()`.
///
/// ## Parameters
/// * `adata` - AnnData object containing the expression matrix.
/// * `net` - Pathway network (prior knowledge).
/// * `n_up_abs` - Absolute number of top genes to select per cell.
/// * `n_up_frac` - Fraction of genes to select per cell (mutually exclusive with `n_up_abs`).
/// * `n_background` - Background universe size (defaults to the number of genes).
///
/// ## Returns
/// An [`OraResult`] with `n_obs × n_pathways` score and p-value matrices, where column `p`
/// corresponds to `net.get_pathway_name(p)`.
#[cfg(feature = "enrichment")]
pub fn run_ora(
    adata: &IMAnnData,
    net: &PathwayNetwork,
    n_up_abs: Option<usize>,
    n_up_frac: Option<f32>,
    n_background: Option<usize>,
) -> anyhow::Result<OraResult> {
    let x = adata.x();
    let read_guard = x.0.read_inner();
    match read_guard.deref() {
        ArrayData::CsrMatrix(matrix) => match matrix {
            DynCsrMatrix::F32(csr) => ora_csr(csr, net, n_up_abs, n_up_frac, n_background),
            DynCsrMatrix::F64(csr) => ora_csr(csr, net, n_up_abs, n_up_frac, n_background),
            _ => anyhow::bail!(
                "ORA only supports F32 and F64 CSR matrices; convert the matrix first."
            ),
        },
        other => anyhow::bail!(
            "ORA only supports CSR matrices, got {:?}. Convert `X` to CSR first.",
            other
        ),
    }
}

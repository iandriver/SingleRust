//! # Over-Representation Analysis (ORA)
//!
//! ORA is the simplest of the [decoupler](https://github.com/scverse/decoupler) family of
//! enrichment methods. For each observation (cell) it takes the set of top-expressed genes
//! and asks, for every gene set / pathway in a [`PathwayNetwork`], whether that pathway's
//! genes are over-represented in the selection more than expected by chance.
//!
//! The test is the one-sided hypergeometric (Fisher's exact) test. Given:
//! - `N` — the background universe size (number of measured genes by default),
//! - `K` — the number of genes in the pathway,
//! - `n` — the number of selected (top) genes for the cell,
//! - `a` — the overlap between the selection and the pathway,
//!
//! the over-representation p-value is `P(X >= a)` for `X ~ Hypergeometric(N, K, n)`, and the
//! reported activity score is `-log10(p)` (so larger = more enriched), matching decoupler's
//! `run_ora` convention.

use anyhow::bail;
use nalgebra_sparse::CsrMatrix;
use ndarray::Array2;
use single_utilities::traits::FloatOpsTS;
use single_utilities::types::PathwayNetwork;
use statrs::distribution::{DiscreteCDF, Hypergeometric};
use std::cmp::Ordering;
use std::collections::HashSet;

/// Results of an ORA run over an expression matrix.
///
/// Both matrices are `n_obs × n_pathways`; column `p` corresponds to the pathway returned by
/// [`PathwayNetwork::get_pathway_name`]`(p)`.
pub struct OraResult {
    /// Activity scores, `-log10(p_value)` per (cell, pathway). Larger means more enriched.
    pub scores: Array2<f32>,
    /// Raw one-sided hypergeometric over-representation p-values per (cell, pathway).
    pub pvals: Array2<f32>,
}

/// One-sided hypergeometric over-representation p-value, `P(X >= overlap)`.
///
/// Returns `1.0` for the degenerate cases (no overlap, empty pathway, or empty selection)
/// and clamps the distribution parameters so they remain valid.
fn ora_pvalue(population: u64, successes: u64, draws: u64, overlap: u64) -> f64 {
    if overlap == 0 || successes == 0 || draws == 0 {
        return 1.0;
    }
    // `successes` and `draws` must not exceed the population for a valid distribution.
    let successes = successes.min(population);
    let draws = draws.min(population);
    match Hypergeometric::new(population, successes, draws) {
        // sf(x) = P(X > x), so P(X >= overlap) = sf(overlap - 1).
        Ok(dist) => dist.sf(overlap - 1),
        Err(_) => 1.0,
    }
}

/// Resolve the number of top genes to select per cell from the abs/frac options.
///
/// Exactly one of `n_up_abs` or `n_up_frac` may be supplied; if neither is given, the
/// default is the top 5% of genes (at least 1).
fn resolve_n_up(
    n_vars: usize,
    n_up_abs: Option<usize>,
    n_up_frac: Option<f32>,
) -> anyhow::Result<usize> {
    let n = match (n_up_abs, n_up_frac) {
        (Some(_), Some(_)) => {
            bail!("Specify only one of `n_up_abs` or `n_up_frac`, not both.")
        }
        (Some(abs), None) => abs,
        (None, Some(frac)) => {
            if !(0.0..=1.0).contains(&frac) {
                bail!("`n_up_frac` must be in [0, 1], got {frac}");
            }
            (frac as f64 * n_vars as f64).ceil() as usize
        }
        (None, None) => (0.05 * n_vars as f64).ceil() as usize,
    };
    Ok(n.clamp(1, n_vars.max(1)))
}

/// Run over-representation analysis for every cell against a pathway network.
///
/// For each row of `matrix` the `n_up` highest-valued genes (among the stored non-zeros)
/// form the selection, and every pathway is scored by the one-sided hypergeometric test.
///
/// ## Parameters
/// * `matrix` - Expression matrix (cells × genes) in CSR form.
/// * `net` - Pathway network whose feature indices refer to columns of `matrix`.
/// * `n_up_abs` - Absolute number of top genes to select per cell.
/// * `n_up_frac` - Fraction of genes to select per cell (mutually exclusive with `n_up_abs`).
/// * `n_background` - Background universe size `N` (defaults to the number of genes).
///
/// ## Returns
/// An [`OraResult`] with `n_obs × n_pathways` score and p-value matrices.
pub fn ora_csr<T: FloatOpsTS>(
    matrix: &CsrMatrix<T>,
    net: &PathwayNetwork,
    n_up_abs: Option<usize>,
    n_up_frac: Option<f32>,
    n_background: Option<usize>,
) -> anyhow::Result<OraResult> {
    let (n_obs, n_vars) = (matrix.nrows(), matrix.ncols());
    let n_paths = net.get_num_pathways();
    let n_up = resolve_n_up(n_vars, n_up_abs, n_up_frac)?;
    let population = n_background.unwrap_or(n_vars).min(n_vars).max(1) as u64;

    // Pre-materialize each pathway's gene set (restricted to valid column indices).
    let pathways: Vec<HashSet<usize>> = (0..n_paths)
        .map(|p| {
            net.get_pathway_features(p)
                .iter()
                .copied()
                .filter(|&g| g < n_vars)
                .collect()
        })
        .collect();

    let mut scores = Array2::<f32>::zeros((n_obs, n_paths));
    let mut pvals = Array2::<f32>::from_elem((n_obs, n_paths), 1.0);

    for (i, row) in matrix.row_iter().enumerate() {
        // Select the top `n_up` genes for this cell by expression value (descending).
        let mut entries: Vec<(usize, f64)> = row
            .col_indices()
            .iter()
            .zip(row.values().iter())
            .map(|(&c, v)| (c, v.to_f64().unwrap_or(0.0)))
            .collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let selection: HashSet<usize> =
            entries.iter().take(n_up).map(|(c, _)| *c).collect();
        let draws = selection.len() as u64;

        for (p, pset) in pathways.iter().enumerate() {
            let successes = pset.len() as u64;
            let overlap = selection.iter().filter(|g| pset.contains(g)).count() as u64;
            let pval = ora_pvalue(population, successes, draws, overlap);
            pvals[[i, p]] = pval as f32;
            // Guard the log against an exact-zero p-value (underflow).
            scores[[i, p]] = -(pval.max(f64::MIN_POSITIVE)).log10() as f32;
        }
    }

    Ok(OraResult { scores, pvals })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::{CooMatrix, CsrMatrix};
    use single_utilities::types::PathwayNetwork;

    // Two pathways over a 4-gene space: P0 = {g0, g1}, P1 = {g2, g3}.
    fn two_pathways() -> PathwayNetwork {
        let names = vec!["P0".to_string(), "P0".to_string(), "P1".to_string(), "P1".to_string()];
        let targets = vec![
            "g0".to_string(),
            "g1".to_string(),
            "g2".to_string(),
            "g3".to_string(),
        ];
        let features = vec![
            "g0".to_string(),
            "g1".to_string(),
            "g2".to_string(),
            "g3".to_string(),
        ];
        PathwayNetwork::new_from_vec(names, targets, None, features, 1)
    }

    fn csr_from_rows(rows: &[Vec<f64>]) -> CsrMatrix<f64> {
        let n_rows = rows.len();
        let n_cols = rows[0].len();
        let mut coo = CooMatrix::<f64>::new(n_rows, n_cols);
        for (i, r) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                if v != 0.0 {
                    coo.push(i, j, v);
                }
            }
        }
        CsrMatrix::from(&coo)
    }

    #[test]
    fn ora_pvalue_matches_known_hypergeometric() {
        // N=4, K=2, n=2, overlap=2 -> P(X>=2) = C(2,2)C(2,0)/C(4,2) = 1/6.
        let p = ora_pvalue(4, 2, 2, 2);
        assert!((p - 1.0 / 6.0).abs() < 1e-9, "got {p}");
        // No overlap -> p-value 1.
        assert_eq!(ora_pvalue(4, 2, 2, 0), 1.0);
    }

    #[test]
    fn ora_enriches_the_matching_pathway() -> anyhow::Result<()> {
        let net = two_pathways();
        // Cell 0 expresses g0,g1 highly -> should enrich P0 over P1.
        // Cell 1 expresses g2,g3 highly -> should enrich P1 over P0.
        let matrix = csr_from_rows(&[
            vec![10.0, 9.0, 0.0, 0.0],
            vec![0.0, 0.0, 8.0, 7.0],
        ]);

        let res = ora_csr(&matrix, &net, Some(2), None, None)?;
        assert_eq!(res.scores.dim(), (2, 2));

        // Locate pathway columns by name rather than assuming an order.
        let col = |name: &str| (0..net.get_num_pathways()).find(|&p| net.get_pathway_name(p) == name).unwrap();
        let (p0, p1) = (col("P0"), col("P1"));

        // Cell 0 (g0,g1 high): P0 strictly more enriched (lower p) than P1.
        assert!(res.scores[[0, p0]] > res.scores[[0, p1]]);
        assert!(res.pvals[[0, p0]] < res.pvals[[0, p1]]);
        // Cell 1 (g2,g3 high): P1 strictly more enriched than P0.
        assert!(res.scores[[1, p1]] > res.scores[[1, p0]]);
        Ok(())
    }
}

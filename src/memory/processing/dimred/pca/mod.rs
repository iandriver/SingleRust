//! # Principal Component Analysis (PCA) for Single-Cell Data
//!
//! This module provides high-performance PCA implementation optimized for single-cell RNA-seq data.
//! PCA is a fundamental dimensionality reduction technique that identifies the directions of maximum
//! variance in high-dimensional data.
//!
//! ## Overview
//!
//! PCA transforms the original gene expression space into a new coordinate system where:
//! - The first principal component captures the most variance
//! - Each subsequent component captures the most remaining variance
//! - Components are orthogonal (uncorrelated) to each other
//!
//! ## Key Features
//!
//! - **Sparse Matrix Support**: Optimized for sparse single-cell expression matrices
//! - **Feature Selection Integration**: Works with highly variable genes or custom gene selections
//! - **Multiple SVD Methods**: Choice of SVD algorithms for different performance characteristics
//! - **Memory Efficient**: Handles large datasets without excessive memory usage
//! - **Configurable Centering**: Option to center data (recommended for most analyses)
//!
//! ## When to Use PCA
//!
//! ✅ **Good for:**
//! - Initial exploration of dataset structure
//! - Noise reduction before clustering
//! - Input for other dimensionality reduction methods (t-SNE, UMAP)
//! - Quality control and batch effect detection
//! - Identifying major sources of variation
//!
//! ⚠️ **Limitations:**
//! - Linear method - may not capture complex non-linear relationships
//! - First components may be dominated by technical effects
//! - Interpretation can be challenging with many genes
//!
//! ## Typical Workflow
//!
//! ```rust,ignore
//! use single_rust::memory::processing::dimred::pca::run_pca_sparse_masked;
//! use single_rust::memory::processing::dimred::FeatureSelectionMethod;
//! use single_algebra::dimred::pca::SVDMethod;
//!
//! // 1. Select highly variable genes
//! let hvg_mask = compute_highly_variable_genes(&adata, None)?;
//! let feature_selection = FeatureSelectionMethod::HighlyVariableSelection(hvg_mask);
//!
//! // 2. Run PCA with 50 components
//! let pca_result = run_pca_sparse_masked::<f64>(
//!     &adata.x(),
//!     Some(feature_selection),
//!     Some(true),              // Center the data
//!     Some(false),             // Verbose output
//!     Some(50),                // Number of components
//!     Some(1.0),               // Regularization parameter
//!     Some(42),                // Random seed for reproducibility
//!     Some(SVDMethod::Randomized), // Fast randomized SVD
//! )?;
//!
//! // 3. Access results
//! let embeddings = pca_result.transformed;              // Cell embeddings in PC space
//! let variance_explained = pca_result.explained_variance_ratio;  // Variance per component
//! let loadings = pca_result.feature_importance;         // Gene loadings/weights
//! ```

use crate::memory::processing::dimred::FeatureSelectionMethod;
use crate::memory::utils::{arr1_conversion, arr2_conversion};
use anndata::data::{DynArray, DynCsrMatrix};
use anndata::{ArrayData, Data};
use anndata_memory::{IMAnnData, IMArrayElement, IMElement};
use anyhow::anyhow;
use ndarray::{Array1, Array2};
use rand::distr::Uniform;
use rand::prelude::Distribution;
use rand::rng;
use single_algebra::dimred::pca::{MaskedSparsePCABuilder, SVDMethod};
use single_utilities::traits::FloatOpsTS;
use std::ops::Deref;

/// Results from Principal Component Analysis.
///
/// Contains all the essential outputs from PCA analysis including the transformed data,
/// variance explained by each component, and feature importance scores.
///
/// ## Fields
///
/// - `transformed`: Cell embeddings in the principal component space (cells × components)
/// - `explained_variance_ratio`: Fraction of total variance explained by each component
/// - `cumulative_explained_variance_ratio`: Cumulative variance explained up to each component
/// - `feature_importance`: Gene loadings/weights for each component (genes × components)
///
/// ## Usage
///
/// ```rust,ignore
/// let pca_result = run_pca_sparse_masked(&matrix, ...)?;
///
/// // Get embeddings for visualization or clustering
/// let embeddings = pca_result.transformed;
///
/// // Check how much variance is captured
/// let total_variance = pca_result.cumulative_explained_variance_ratio[[49]]; // 50th component
///
/// // Find important genes for first component  
/// let pc1_loadings = pca_result.feature_importance.column(0);
/// ```
pub struct PCAResult<T>
where
    T: FloatOpsTS,
{
    /// Transformed data: cell embeddings in principal component space (n_cells × n_components)
    pub transformed: Array2<T>,
    /// Fraction of variance explained by each principal component
    pub explained_variance_ratio: Array1<T>,
    /// Cumulative fraction of variance explained up to each component
    pub cumulative_explained_variance_ratio: Array1<T>,
    /// Feature loadings/importance for each component (n_features × n_components)
    pub feature_importance: Array2<T>,
}

/// Perform Principal Component Analysis on sparse single-cell expression data.
///
/// This function provides a comprehensive PCA implementation optimized for single-cell data,
/// with support for feature selection, sparse matrices, and various SVD algorithms.
///
/// ## Algorithm Details
///
/// The implementation uses efficient sparse matrix operations and supports multiple SVD methods:
/// - **Randomized SVD**: Fast approximation, good for large datasets
/// - **Full SVD**: Exact computation, slower but more accurate
/// - **Truncated SVD**: Memory efficient for large matrices
///
/// ## Parameters
///
/// * `matrix` - The expression matrix (cells × genes) as IMArrayElement
/// * `feature_selection_method` - Method for selecting genes (HVGs recommended)
/// * `center` - Whether to center the data (recommended: true)
/// * `verbose` - Enable verbose output for debugging
/// * `n_components` - Number of principal components to compute (default: 50)
/// * `alpha` - Regularization parameter for numerical stability (default: 1.0)
/// * `random_seed` - Seed for reproducible results (default: 42)
/// * `svd_method` - SVD algorithm to use (default: Randomized)
///
/// ## Returns
///
/// Returns a `PCAResult` containing:
/// - Transformed cell embeddings
/// - Variance explained by each component
/// - Feature importance/loading scores
///
/// ## Examples
///
/// ### Basic Usage
/// ```rust,ignore
/// // Simple PCA with default parameters
/// let result = run_pca_sparse_masked::<f64>(
///     &adata.x(),
///     None,                    // Use default random selection
///     Some(true),              // Center the data
///     None,                    // No verbose output
///     Some(50),                // 50 components
///     None,                    // Default alpha
///     None,                    // Default seed
///     None,                    // Default SVD method
/// )?;
/// ```
///
/// ### Advanced Usage with HVGs
/// ```rust,ignore
/// // PCA using highly variable genes
/// let hvg_mask = compute_highly_variable_genes(&adata, None)?;
/// let feature_selection = FeatureSelectionMethod::HighlyVariableSelection(hvg_mask);
///
/// let result = run_pca_sparse_masked::<f64>(
///     &adata.x(),
///     Some(feature_selection),
///     Some(true),              // Center for better component interpretation
///     Some(false),             // Quiet mode
///     Some(30),                // Fewer components for speed
///     Some(0.1),               // Higher regularization
///     Some(123),               // Custom seed
///     Some(SVDMethod::Randomized), // Fast approximation
/// )?;
/// ```
///
/// ## Performance Considerations
///
/// - **Feature Selection**: Using 1000-5000 highly variable genes typically optimal
/// - **Components**: 30-50 components usually capture most biological variation
/// - **SVD Method**: Randomized SVD recommended for >10,000 cells
/// - **Centering**: Essential for proper component interpretation but increases memory usage
///
/// ## Errors
///
/// Returns error if:
/// - Matrix format is not supported (only CSR matrices supported)
/// - Data type is not F32 or F64
/// - Feature selection mask length doesn't match number of genes
/// - SVD computation fails (e.g., insufficient rank)
#[allow(clippy::too_many_arguments)]
pub fn run_pca_sparse_masked<T>(
    matrix: &IMArrayElement,
    feature_selection_method: Option<FeatureSelectionMethod>,
    center: Option<bool>,
    verbose: Option<bool>,
    n_components: Option<usize>,
    alpha: Option<f64>,
    random_seed: Option<u32>,
    svd_method: Option<SVDMethod>,
) -> anyhow::Result<PCAResult<T>>
where
    T: FloatOpsTS,
{
    let feature_selection_method =
        feature_selection_method.unwrap_or(FeatureSelectionMethod::RandomSelection(1000));
    let shape = matrix.get_shape()?;
    let ncols = shape[1];
    let center = center.unwrap_or(false);
    let verbose = verbose.unwrap_or(false);
    let n_components = n_components.unwrap_or(50);
    let random_seed = random_seed.unwrap_or(42);
    let svd_method = svd_method.unwrap_or_default();
    let selected = match feature_selection_method {
        FeatureSelectionMethod::FullFeatures => {
            vec![true; ncols]
        }
        FeatureSelectionMethod::HighlyVariableSelection(vec) => vec,
        FeatureSelectionMethod::RandomSelection(num_genes) => {
            generate_random_mask(ncols, num_genes)
        }
    };
    let read_guard = matrix.0.read_inner();
    let data = read_guard.deref();
    match data {
        ArrayData::CsrMatrix(dyn_csr) => {
            match dyn_csr {
                DynCsrMatrix::F32(csr) => {
                    let mut masked_pca = MaskedSparsePCABuilder::new()
                        .mask(selected)
                        .center(center)
                        .verbose(verbose)
                        .alpha(alpha.unwrap_or(1.0) as f32)
                        .n_components(n_components)
                        .random_seed(random_seed)
                        .svd_method(svd_method)
                        .build();
                    masked_pca.fit(csr)?;
                    let transformed = masked_pca.transform(csr)?;
                    let explained_variance_ratio = masked_pca.explained_variance_ratio()?;
                    let cumulative_explained_variance_ratio = masked_pca.cumulative_explained_variance_ratio()?;
                    let feature_importance = masked_pca.feature_importances()?;

                    let transformed: Array2<T> = arr2_conversion(transformed)?;
                    let explained_variance_ratio: Array1<T> = arr1_conversion(explained_variance_ratio)?;
                    let cumulative_explained_variance_ratio: Array1<T> = arr1_conversion(cumulative_explained_variance_ratio)?;
                    let feature_importance: Array2<T> = arr2_conversion(feature_importance)?;
                    let res = PCAResult {
                        transformed,
                        explained_variance_ratio,
                        cumulative_explained_variance_ratio,
                        feature_importance,
                    };
                    Ok(res)
                }
                DynCsrMatrix::F64(csr) => {
                    let mut masked_pca = MaskedSparsePCABuilder::new()
                        .mask(selected)
                        .center(center)
                        .verbose(verbose)
                        .alpha(alpha.unwrap_or(1.0))
                        .n_components(n_components)
                        .random_seed(random_seed)
                        .svd_method(svd_method)
                        .build();
                    masked_pca.fit(csr)?;
                    let transformed = masked_pca.transform(csr)?;
                    let explained_variance_ratio = masked_pca.explained_variance_ratio()?;
                    let cumulative_explained_variance_ratio = masked_pca.cumulative_explained_variance_ratio()?;
                    let feature_importance = masked_pca.feature_importances()?;

                    let transformed: Array2<T> = arr2_conversion(transformed)?;
                    let explained_variance_ratio: Array1<T> = arr1_conversion(explained_variance_ratio)?;
                    let cumulative_explained_variance_ratio: Array1<T> = arr1_conversion(cumulative_explained_variance_ratio)?;
                    let feature_importance: Array2<T> = arr2_conversion(feature_importance)?;
                    let res = PCAResult {
                        transformed,
                        explained_variance_ratio,
                        cumulative_explained_variance_ratio,
                        feature_importance,
                    };
                    Ok(res)
                }
                _ => Err(anyhow!("This datatype is currently not supported, please convert to F32 or F64 first, before running PCA!"))
            }
        }
        _ => Err(anyhow!("This anndata type is currently not supported. Only CSR matrices are supported for now!"))
    }
}

/// Run PCA and store the results in the AnnData object following scanpy conventions.
///
/// This is a convenience wrapper around [`run_pca_sparse_masked`] that, in addition to
/// returning the [`PCAResult`], writes the outputs back into `adata` using the same
/// slots scanpy/anndata expect. This makes the embedding directly consumable by
/// downstream steps (t-SNE, UMAP, clustering) and by scanpy itself once the object is
/// written to `.h5ad` via [`crate::io::write_h5ad`].
///
/// ## Stored outputs
///
/// * `adata.obsm["X_{key_added}"]` — cell embeddings (`n_obs × n_components`), as `f64`.
///   With the default `key_added = "pca"` this is `obsm["X_pca"]`, exactly the key
///   scanpy's neighbors/UMAP/t-SNE read from.
/// * `adata.uns["{key_added}_variance_ratio"]` — fraction of variance explained by each
///   component (for elbow/scree plots).
/// * `adata.uns["{key_added}_variance_ratio_cumulative"]` — cumulative variance explained.
///
/// ## Loadings (`varm["PCs"]`)
///
/// scanpy also stores signed principal axes in `varm["PCs"]`. The masked sparse PCA
/// backend currently only exposes *squared* feature importances over the **selected**
/// gene subset (not the signed loadings over all genes), so a faithful, scanpy-semantic
/// `varm["PCs"]` cannot be produced here yet. It is therefore intentionally omitted
/// rather than written with mismatched semantics. The squared importances remain
/// available on the returned [`PCAResult::feature_importance`].
///
/// ## Parameters
///
/// Identical to [`run_pca_sparse_masked`], plus:
/// * `key_added` - Base key for the stored results (default: `"pca"`).
///
/// ## Returns
///
/// The [`PCAResult`], so callers can still access loadings/variance directly. The
/// embedding and variance ratios are additionally persisted into `adata`.
///
/// ## Example
///
/// ```rust,ignore
/// use single_rust::memory::processing::dimred::pca::run_pca_inplace;
///
/// // Compute PCA and populate obsm["X_pca"] + uns variance ratios in one call.
/// run_pca_inplace::<f64>(&adata, Some(feature_selection), Some(true), None,
///                        Some(50), None, Some(42), None, None)?;
///
/// let x_pca = adata.obsm().get_array("X_pca")?; // ready for UMAP/clustering
/// ```
#[allow(clippy::too_many_arguments)]
pub fn run_pca_inplace<T>(
    adata: &IMAnnData,
    feature_selection_method: Option<FeatureSelectionMethod>,
    center: Option<bool>,
    verbose: Option<bool>,
    n_components: Option<usize>,
    alpha: Option<f64>,
    random_seed: Option<u32>,
    svd_method: Option<SVDMethod>,
    key_added: Option<&str>,
) -> anyhow::Result<PCAResult<T>>
where
    T: FloatOpsTS,
{
    let key = key_added.unwrap_or("pca");
    let result = run_pca_sparse_masked::<T>(
        &adata.x(),
        feature_selection_method,
        center,
        verbose,
        n_components,
        alpha,
        random_seed,
        svd_method,
    )?;
    store_pca_result(adata, &result, key)?;
    Ok(result)
}

/// Persist PCA embeddings and variance ratios into `adata` (scanpy slot conventions).
///
/// Embeddings are stored as `f64` in `obsm["X_{key}"]`; the variance-ratio vectors are
/// stored as `f64` arrays in `uns`.
fn store_pca_result<T>(
    adata: &IMAnnData,
    result: &PCAResult<T>,
    key: &str,
) -> anyhow::Result<()>
where
    T: FloatOpsTS,
{
    // Cell embeddings -> obsm["X_{key}"] (e.g. "X_pca").
    let embeddings: Array2<f64> = arr2_conversion(result.transformed.clone())?;
    let embeddings: ArrayData = DynArray::from(embeddings).into();
    adata
        .obsm()
        .add_array(format!("X_{}", key), IMArrayElement::new(embeddings))?;

    // Variance ratios -> uns (scanpy uses these for scree/elbow plots).
    let variance_ratio: Array1<f64> = arr1_conversion(result.explained_variance_ratio.clone())?;
    adata.uns().add_data(
        format!("{}_variance_ratio", key),
        IMElement::new(Data::ArrayData(DynArray::from(variance_ratio).into())),
    )?;

    let variance_ratio_cumulative: Array1<f64> =
        arr1_conversion(result.cumulative_explained_variance_ratio.clone())?;
    adata.uns().add_data(
        format!("{}_variance_ratio_cumulative", key),
        IMElement::new(Data::ArrayData(
            DynArray::from(variance_ratio_cumulative).into(),
        )),
    )?;

    Ok(())
}

/// Generate a random boolean mask for gene selection.
///
/// Creates a boolean vector where `num_random_selection` randomly chosen positions
/// are set to `true`, and all others are `false`. This is used for benchmarking
/// and testing purposes when you want to select a random subset of genes.
///
/// ## Parameters
///
/// * `n_genes` - Total number of genes in the dataset
/// * `num_random_selection` - Number of genes to randomly select
///
/// ## Returns
///
/// Boolean vector of length `n_genes` with exactly `num_random_selection` true values
/// at random positions.
///
/// ## Note
///
/// This function uses the default random number generator. For reproducible results,
/// set the global random seed before calling, or consider using the `random_seed`
/// parameter in `run_pca_sparse_masked`.
///
/// ## Example
///
/// ```rust,ignore
/// // Select 2000 random genes from 20000 total genes
/// let mask = generate_random_mask(20000, 2000);
/// assert_eq!(mask.len(), 20000);
/// assert_eq!(mask.iter().filter(|&&x| x).count(), 2000);
/// ```
fn generate_random_mask(n_genes: usize, num_random_selection: usize) -> Vec<bool> {
    let mut rng = rng();
    let uniform = Uniform::new(0, n_genes).unwrap();
    let mut vec = vec![false; num_random_selection];
    for _ in 0..num_random_selection {
        let v = uniform.sample(&mut rng);
        vec[v] = true;
    }
    vec
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndata_memory::IMAnnData;
    use nalgebra_sparse::{CooMatrix, CsrMatrix};

    /// Build a tiny CSR-backed AnnData fixture for PCA.
    fn fixture(n_obs: usize, n_vars: usize) -> anyhow::Result<IMAnnData> {
        // Deterministic dense-ish pattern, stored sparsely.
        let mut coo = CooMatrix::<f64>::new(n_obs, n_vars);
        for i in 0..n_obs {
            for j in 0..n_vars {
                let v = ((i * 7 + j * 3) % 5) as f64;
                if v != 0.0 {
                    coo.push(i, j, v);
                }
            }
        }
        let csr = CsrMatrix::from(&coo);
        let matrix: ArrayData = DynCsrMatrix::from(csr).into();
        let obs_names = (0..n_obs).map(|i| format!("cell{i}")).collect();
        let var_names = (0..n_vars).map(|j| format!("gene{j}")).collect();
        IMAnnData::new_basic(matrix, obs_names, var_names)
    }

    /// `run_pca_inplace` must populate `obsm["X_pca"]` with an (n_obs × n_components)
    /// embedding and record the variance ratios in `uns` — the scanpy slots that make
    /// the result consumable downstream.
    #[test]
    fn run_pca_inplace_populates_scanpy_slots() -> anyhow::Result<()> {
        let (n_obs, n_vars, n_components) = (8, 6, 3);
        let adata = fixture(n_obs, n_vars)?;

        run_pca_inplace::<f64>(
            &adata,
            Some(FeatureSelectionMethod::FullFeatures),
            Some(true),
            Some(false),
            Some(n_components),
            None,
            Some(42),
            None,
            None,
        )?;

        // obsm["X_pca"] exists with the expected shape.
        assert!(adata.obsm().keys().contains(&"X_pca".to_string()));
        let emb = adata.obsm().get_array("X_pca")?;
        let shape = emb.get_shape()?;
        assert_eq!(shape[0], n_obs);
        assert_eq!(shape[1], n_components);

        // Variance ratios recorded in uns.
        let uns_keys = adata.uns().keys()?;
        assert!(uns_keys.contains(&"pca_variance_ratio".to_string()));
        assert!(uns_keys.contains(&"pca_variance_ratio_cumulative".to_string()));

        Ok(())
    }

    /// A custom `key_added` should redirect the obsm/uns keys.
    #[test]
    fn run_pca_inplace_respects_key_added() -> anyhow::Result<()> {
        let adata = fixture(6, 5)?;
        run_pca_inplace::<f64>(
            &adata,
            Some(FeatureSelectionMethod::FullFeatures),
            Some(true),
            Some(false),
            Some(2),
            None,
            Some(7),
            None,
            Some("mypca"),
        )?;

        assert!(adata.obsm().keys().contains(&"X_mypca".to_string()));
        assert!(adata
            .uns()
            .keys()?
            .contains(&"mypca_variance_ratio".to_string()));
        Ok(())
    }
}

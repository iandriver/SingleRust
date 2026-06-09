use std::path::Path;

use anndata::{AnnData, Backend};
use anndata_hdf5::H5;
use anndata_memory::{convert_to_in_memory, convert_to_new_backed_h5, IMAnnData};

pub enum FileScope {
    Read = 0,
    ReadWrite = 1,
}

pub fn read_h5ad<P: AsRef<Path>>(
    path_to_file: P,
    scope: FileScope,
    enable_cache: bool,
) -> anyhow::Result<AnnData<H5>> {
    let h5_file = match scope {
        FileScope::Read => H5::open(path_to_file)?,
        FileScope::ReadWrite => H5::open_rw(path_to_file)?,
    };
    let adata = AnnData::<H5>::open(h5_file)?;
    if enable_cache {
        adata.get_x().inner().enable_cache();
    }
    Ok(adata)
}

pub fn read_h5ad_memory<P: AsRef<Path>>(path_to_file: P) -> anyhow::Result<IMAnnData> {
    let adata = read_h5ad(path_to_file, FileScope::Read, false)?;
    convert_to_in_memory(adata)
}

pub fn read_h5ad_fast_memory<P: AsRef<Path>>(path_to_file: P) -> anyhow::Result<IMAnnData> {
    anndata_memory::load_h5ad_fast(path_to_file)
}

/// Write an in-memory [`IMAnnData`] object to an `.h5ad` file on disk.
///
/// This serializes the full AnnData object — the expression matrix (`X`), `obs`/`var`
/// data frames, and every `obsm`/`varm`/`obsp`/`varp`/`layers`/`uns` slot — into a new
/// HDF5 file using the standard AnnData on-disk layout. The resulting file can be read
/// back by scanpy, anndata (Python), or any other tool in the AnnData ecosystem, which
/// makes this the primary mechanism for round-tripping SingleRust analysis results into
/// the broader single-cell toolchain.
///
/// Any existing file at `path_to_file` is overwritten.
///
/// ## Parameters
/// * `adata` - The in-memory AnnData object to serialize.
/// * `path_to_file` - Destination path for the `.h5ad` file.
///
/// ## Example
///
/// ```rust,ignore
/// use single_rust::io;
///
/// let adata = io::read_h5ad_memory("input.h5ad")?;
/// // ... run QC, normalization, HVG, PCA, differential expression ...
/// io::write_h5ad(&adata, "results.h5ad")?;  // re-open in scanpy for plotting
/// ```
pub fn write_h5ad<P: AsRef<Path>>(adata: &IMAnnData, path_to_file: P) -> anyhow::Result<()> {
    let backed = convert_to_new_backed_h5(adata, path_to_file)?;
    // Close flushes all buffered writes and releases the HDF5 file handle.
    backed.close()
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndata::data::DynArray;
    use anndata::ArrayData;
    use ndarray::Array2;
    use polars::prelude::{Column, DataFrame};

    /// Writing an `IMAnnData` to disk and reading it back must preserve shape,
    /// dimension names, and obs/var annotation columns — the round-trip that
    /// makes results usable from scanpy/anndata.
    #[test]
    fn write_h5ad_round_trips() -> anyhow::Result<()> {
        let x = Array2::from_shape_vec((3, 2), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let matrix: ArrayData = DynArray::from(x).into();

        // Realistic annotation frames (no column named "index", which would
        // collide with the AnnData index dataset on write).
        let obs_df = DataFrame::new(vec![Column::new(
            "cell_type".into(),
            ["T", "B", "T"].as_slice(),
        )])?;
        let var_df = DataFrame::new(vec![Column::new(
            "highly_variable".into(),
            [true, false].as_slice(),
        )])?;

        let adata = IMAnnData::new_extended(
            matrix,
            vec!["cell0".into(), "cell1".into(), "cell2".into()],
            vec!["geneA".into(), "geneB".into()],
            obs_df,
            var_df,
        )?;

        let mut path = std::env::temp_dir();
        path.push("single_rust_write_h5ad_round_trips.h5ad");
        let _ = std::fs::remove_file(&path);

        write_h5ad(&adata, &path)?;
        assert!(path.exists(), "output file was not created");

        let reloaded = read_h5ad_memory(&path)?;
        assert_eq!(reloaded.n_obs(), 3);
        assert_eq!(reloaded.n_vars(), 2);
        assert_eq!(reloaded.obs_names(), vec!["cell0", "cell1", "cell2"]);
        assert_eq!(reloaded.var_names(), vec!["geneA", "geneB"]);
        // Annotation columns survive the round-trip.
        let obs_back = reloaded.obs().get_data();
        assert!(obs_back.get_column_names_str().contains(&"cell_type"));

        std::fs::remove_file(&path).ok();
        Ok(())
    }
}

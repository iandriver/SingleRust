"""Fetch a stratified blood (PBMC-like) slice from the CZI CELLxGENE Census and save it as
a counts `.h5ad` that SingleRust can consume.

  python demo/prepare_data.py [--n-cells N] [--per-type K] [--out PATH]

Output: raw counts, CSR float32, var_names = gene symbols.
"""
import os
import argparse
import numpy as np
import scanpy as sc
import cellxgene_census
from scipy.sparse import csr_matrix

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "input.h5ad")
CENSUS_VERSION = "2023-12-15"  # pinned for reproducibility
SEED = 0

# Filter: healthy primary blood. The narrow default pins a single assay for homogeneity;
# `--all-assays` drops that clause (4M+ candidate cells) so very large samples are possible.
NARROW_FILTER = (
    "tissue_general == 'blood' and is_primary_data == True "
    "and disease == 'normal' and assay == '10x 3\\' v3'"
)
ALL_ASSAY_FILTER = (
    "tissue_general == 'blood' and is_primary_data == True and disease == 'normal'"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=3000, help="total cell cap")
    ap.add_argument("--per-type", type=int, default=250, help="cells sampled per cell type")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output .h5ad path")
    ap.add_argument("--all-assays", action="store_true",
                    help="drop the single-assay restriction (needed for very large samples)")
    a = ap.parse_args()
    run(out=a.out, total_cap=a.n_cells, per_type=a.per_type, all_assays=a.all_assays)


def run(out=DEFAULT_OUT, total_cap=3000, per_type=250, all_assays=False):
    OUT, TOTAL_CAP, PER_TYPE = out, total_cap, per_type
    obs_filter = ALL_ASSAY_FILTER if all_assays else NARROW_FILTER
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"opening census {CENSUS_VERSION} ...")
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        print("querying obs ...")
        obs = cellxgene_census.get_obs(
            census,
            "Homo sapiens",
            value_filter=obs_filter,
            column_names=["soma_joinid", "cell_type"],
        )
        print(f"  {len(obs)} candidate cells, {obs['cell_type'].nunique()} cell types")

        # Stratified subsample: up to PER_TYPE per cell type, capped at TOTAL_CAP.
        rng = np.random.default_rng(SEED)
        picks = []
        for ct, grp in obs.groupby("cell_type"):
            n = min(PER_TYPE, len(grp))
            picks.append(grp.sample(n=n, random_state=SEED))
        sampled = (
            __import__("pandas").concat(picks).sample(frac=1.0, random_state=SEED)
        )
        if len(sampled) > TOTAL_CAP:
            sampled = sampled.iloc[:TOTAL_CAP]
        ids = sampled["soma_joinid"].to_numpy()
        print(f"  sampled {len(ids)} cells")

        print("fetching counts ...")
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            measurement_name="RNA",
            X_name="raw",
            obs_coords=ids.tolist(),
            var_column_names=["feature_id", "feature_name"],
            obs_column_names=["cell_type", "assay", "sex", "disease"],
        )

    # var_names -> gene symbols (unique) so marker symbols match.
    adata.var["ensembl_id"] = adata.var["feature_id"].values
    adata.var_names = adata.var["feature_name"].astype(str).values
    adata.var_names_make_unique()

    # Drop genes detected in zero sampled cells to keep things lean.
    sc.pp.filter_genes(adata, min_cells=1)

    # SingleRust's PCA/ORA want CSR float32 counts in X.
    adata.X = csr_matrix(adata.X, dtype=np.float32)

    # Keep obs lean and predictable.
    adata.obs = adata.obs[["cell_type", "assay", "sex", "disease"]].copy()
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)

    print(f"writing {OUT}: {adata.n_obs} cells × {adata.n_vars} genes")
    adata.write_h5ad(OUT)
    print("cell types:")
    print(adata.obs["cell_type"].value_counts())


if __name__ == "__main__":
    main()

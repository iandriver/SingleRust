"""scanpy IN-MEMORY qc + normalize_total + log1p on an .h5ad (loads the whole matrix).

Used by the OOC benchmark as the baseline lane. Prints STEP_SECONDS=<total compute>.
Peak RSS (captured by the caller via /usr/bin/time -l) scales with the dataset — that's the
point of the comparison against SingleRust's bounded-memory OOC lane.
"""
import sys
import time

import hdf5plugin  # noqa: F401
import scanpy as sc
import anndata as ad


def main():
    path = sys.argv[1]
    adata = ad.read_h5ad(path)  # full load into memory
    adata.var["mito"] = adata.var_names.str.upper().str.startswith("MT-")
    t = time.perf_counter()
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mito"], percent_top=[50, 100, 200, 500], log1p=True, inplace=True
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f"STEP_SECONDS={time.perf_counter() - t}")


if __name__ == "__main__":
    main()

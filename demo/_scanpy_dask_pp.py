"""scanpy OUT-OF-CORE via Dask: read only X lazily as a dask array (obs/var eager + small), run
dask-enabled normalize_total + log1p (and attempt qc), then stream the result to disk to force
chunked execution.

Requires anndata>=0.11 + dask + xarray. Run with the isolated .venv-dask. Prints
STEP_SECONDS=<compute wall> and QC_DASK=ok|err:<msg>. Peak RSS is captured by the caller via
/usr/bin/time -l; if Dask's sparse path densifies/materializes, RSS spikes — itself the finding.

    python demo/_scanpy_dask_pp.py <file.h5ad> [chunk_rows]
"""
import sys
import time
import tempfile
import os

import hdf5plugin  # noqa: F401
import h5py
import anndata as ad
import scanpy as sc
from anndata.experimental import read_elem_lazy
from anndata.io import read_elem


def main():
    path = sys.argv[1]
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 20000

    # Keep the file open for the whole run — the lazy dask array reads from it on compute.
    f = h5py.File(path, "r")
    try:
        X = read_elem_lazy(f["X"], chunks=(chunk, -1))  # dask array, streamed in row blocks
        obs = read_elem(f["obs"])                       # eager, small
        var = read_elem(f["var"])
        adata = ad.AnnData(X=X, obs=obs, var=var)

        t = time.perf_counter()

        qc_status = "ok"
        try:
            adata.var["mito"] = adata.var_names.str.upper().str.startswith("MT-")
            sc.pp.calculate_qc_metrics(
                adata, qc_vars=["mito"], percent_top=[50, 100, 200, 500], log1p=True, inplace=True
            )
        except Exception as e:
            qc_status = f"err:{type(e).__name__}:{str(e)[:120]}"

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Force streaming execution by writing the transformed X out (chunked).
        out = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False).name
        try:
            adata.write_h5ad(out)
        finally:
            if os.path.exists(out):
                os.remove(out)

        print(f"STEP_SECONDS={time.perf_counter() - t}")
        print(f"QC_DASK={qc_status}")
    finally:
        f.close()


if __name__ == "__main__":
    main()

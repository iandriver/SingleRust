"""Reproduce group-count stress tests using raw Emont WAT counts."""

import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from partitioned import aggregate
from scipy import sparse


def fixture(source, path, cells, genes):
    with h5py.File(source) as f:
        x = f["raw/X"]
        take = min(cells, int(x.attrs["shape"][0]))
        ptr = x["indptr"][: take + 1]
        pick = np.linspace(0, x.attrs["shape"][1] - 1, genes, dtype=int)
        parts = []
        for i in range(0, take, 4096):
            q = ptr[i : min(i + 4096, take) + 1]
            sl = slice(q[0], q[-1])
            a = sparse.csr_matrix(
                (x["data"][sl], x["indices"][sl], q - q[0]),
                shape=(len(q) - 1, x.attrs["shape"][1]),
            )
            parts.append(a[:, pick])
        counts = sparse.vstack(parts, format="csr")
        counts = sparse.vstack(
            [counts] * (cells // take) + [counts[: cells % take]], format="csr"
        )
        ad.AnnData(counts).write_h5ad(path)


def execute(mode, path, dest, n, workers):
    import dask

    a = ad.read_h5ad(path, backed="r")
    if mode in ("scanpy", "decoupler") and n * a.n_vars * 8 > 6_000_000_000:
        print(
            f"Preflight memory limit: one dense float64 result needs {n * a.n_vars * 8:,} bytes",
            flush=True,
        )
        raise SystemExit(124)
    labels = np.arange(a.n_obs, dtype=np.int64) % n
    names = np.array([f"g{i:06d}" for i in range(n)])
    a.obs["group"] = pd.Categorical.from_codes(labels, names)
    if mode == "rust":
        a.file.close()
        aggregate(path, dest, labels, names, workers)
    elif mode in ("scanpy", "scanpy-pr"):
        import scanpy as sc
        from anndata.experimental import read_elem_lazy

        with (
            h5py.File(path) as f,
            dask.config.set(scheduler="threads", num_workers=workers),
        ):
            lazy = ad.AnnData(
                read_elem_lazy(f["X"], chunks=(4096, -1)), obs=a.obs, var=a.var
            )
            options = {"keep_sparse": True} if mode == "scanpy-pr" else {}
            result = sc.get.aggregate(
                lazy, "group", ["sum", "count_nonzero"], **options
            )
            if mode == "scanpy-pr":
                # At e92b11c the delayed arrays advertise dense meta but compute
                # CSR blocks. AnnData selects its writer from the advertised meta.
                for key, array in result.layers.items():
                    result.layers[key] = array.map_blocks(
                        lambda block: block,
                        meta=sparse.csr_matrix((0, 0), dtype=array.dtype),
                    )
            result.write_h5ad(dest)
    elif mode == "decoupler":
        import decoupler as dc

        result = dc.pp.pseudobulk(
            a, sample_col="group", groups_col=None, mode="sum", bsize=4096
        )
        result.write_h5ad(dest)
    elif mode == "check":
        from verify import verify

        a.file.close()
        verify(path, dest, n, names)
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    if sys.argv[1] == "fixture":
        fixture(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    else:
        execute(*sys.argv[1:4], int(sys.argv[4]), int(sys.argv[5]))

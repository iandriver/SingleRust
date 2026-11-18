"""Tile a raw count matrix using HDF5 virtual arrays, keeping the source read-only."""

import argparse

import anndata as ad
import h5py
import numpy as np
from scipy import sparse


def build(source, output, cells):
    from pathlib import Path

    source = str(Path(source).resolve())
    with h5py.File(source) as src:
        x = src["raw/X"]
        n, genes = map(int, x.attrs["shape"])
        ptr = x["indptr"][:]
        blocks = [n] * (cells // n) + ([cells % n] if cells % n else [])
        nnz = sum(int(ptr[b]) for b in blocks)
        var = ad.io.read_elem(src["raw/var"])
        ad.AnnData(
            sparse.csr_matrix((cells, genes), dtype=x["data"].dtype), var=var
        ).write_h5ad(output)
        with h5py.File(output, "r+") as out:
            g = out["X"]
            for name in ("indices", "data"):
                layout = h5py.VirtualLayout(shape=(nnz,), dtype=x[name].dtype)
                v = h5py.VirtualSource(source, x[name].name, shape=x[name].shape)
                start = 0
                for b in blocks:
                    end = start + int(ptr[b])
                    layout[start:end] = v[: ptr[b]]
                    start = end
                del g[name]
                g.create_virtual_dataset(name, layout)
            start = 0
            parts = [np.array([0], dtype="int64")]
            for b in blocks:
                parts.append(ptr[1 : b + 1] + start)
                start += int(ptr[b])
            del g["indptr"]
            g["indptr"] = np.concatenate(parts)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("output")
    p.add_argument("cells", type=int)
    build(**vars(p.parse_args()))

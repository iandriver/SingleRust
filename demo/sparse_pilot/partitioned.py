"""Count-only CSR H5AD pilot. Group codes index the full requested row axis."""

import ctypes
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def aggregate(
    path,
    dest,
    labels,
    names,
    workers,
    chunk=4096,
    library=None,
    mode="sum",
    metadata=None,
):
    if library is None:
        library = Path(__file__).with_name("libscatter.dylib")
    if Path(path).resolve() == Path(dest).resolve():
        raise ValueError("input and output must be distinct")
    if mode not in ("sum", "mean"):
        raise ValueError("expected sum or mean")
    if workers < 1 or chunk < 1:
        raise ValueError("workers and chunk must be positive")
    scatter = ctypes.CDLL(str(library)).scatter
    scatter.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    scatter.restype = ctypes.c_bool
    with (
        tempfile.TemporaryDirectory(dir=Path(dest).parent) as tmp,
        h5py.File(path) as src,
    ):
        x = src["X"]
        cells, genes = map(int, x.attrs["shape"])
        if x.attrs["encoding-type"] != "csr_matrix" or genes == 0:
            raise ValueError("expected CSR counts with a nonempty gene axis")
        labels = np.asarray(labels)
        if (
            labels.shape != (cells,)
            or labels.dtype.kind != "i"
            or np.any(labels < -1)
            or np.any(labels >= len(names))
            or not len(names)
        ):
            raise ValueError("invalid group codes or empty group axis")
        if cells > np.iinfo(np.uint32).max or genes > np.iinfo(np.int32).max:
            raise ValueError("matrix exceeds supported index range")
        width = max(1, min(len(names), 2_000_000 // genes))
        shards = range((len(names) + width - 1) // width)
        cells_per = np.bincount(labels[labels >= 0], minlength=len(names))
        for start in range(0, cells, chunk):
            ptr = x["indptr"][start : min(cells, start + chunk) + 1]
            sl = slice(ptr[0], ptr[-1])
            groups = np.repeat(labels[start : start + len(ptr) - 1], np.diff(ptr))
            cols = x["indices"][sl]
            counts = x["data"][sl]
            with np.errstate(invalid="ignore", over="ignore"):
                vals = counts.astype("uint32")
            if not np.array_equal(vals, counts):
                raise ValueError("expected nonnegative integer counts fitting uint32")
            # A duplicated CSR coordinate represents one cell-gene observation.
            block = sparse.csr_matrix(
                (vals.astype("uint64"), cols, ptr - ptr[0]), shape=(len(ptr) - 1, genes)
            )
            if not block.has_canonical_format:
                block.sum_duplicates()
                if block.data.size and block.data.max() > np.iinfo(np.uint32).max:
                    raise ValueError("duplicate counts exceed uint32")
                vals = block.data.astype("uint32")
                cols = block.indices
                groups = np.repeat(
                    labels[start : start + len(ptr) - 1], np.diff(block.indptr)
                )
            del block
            keep = (groups >= 0) & (vals > 0)
            groups = groups[keep]
            buckets = groups // width
            order = np.argsort(buckets, kind="stable")
            records = np.column_stack(
                ((groups % width) * genes + cols[keep], vals[keep])
            ).astype("uint32")[order]
            ids, starts = np.unique(buckets[order], return_index=True)
            for i, shard in enumerate(ids):
                with open(Path(tmp) / str(shard), "ab") as handle:
                    records[
                        starts[i] : starts[i + 1] if i + 1 < len(ids) else None
                    ].tofile(handle)

        def reduce(shard):
            start = shard * width
            shape = (min(width, len(names) - start), genes)
            sums = np.zeros(shape)
            hits = np.zeros(shape, dtype="uint32")
            p = Path(tmp) / str(shard)
            if p.exists():
                with p.open("rb") as handle:
                    while (
                        records := np.fromfile(handle, dtype="uint32", count=262144)
                    ).size:
                        if not scatter(
                            records.ctypes.data,
                            records.size // 2,
                            sums.ctypes.data,
                            hits.ctypes.data,
                            sums.size,
                        ):
                            raise ValueError("invalid partition index")
                p.unlink()
            rows, cols = np.nonzero(hits)
            offsets = np.bincount(rows, minlength=shape[0]).cumsum(dtype="int64")
            values = sums[rows, cols]
            if mode == "mean":
                values /= cells_per[start + rows]
            return (
                cols.astype("int32"),
                offsets,
                values,
                hits[rows, cols] / cells_per[start + rows],
                sums.sum(axis=1),
            )

        obs = pd.DataFrame(
            {"psbulk_cells": cells_per, "psbulk_counts": np.zeros(len(names))},
            index=names,
        )
        if metadata is not None:
            obs = pd.concat([metadata, obs], axis=1, verify_integrity=True)
        var = ad.io.read_elem(src["var"])
        empty = sparse.csr_matrix((len(names), genes), dtype="float64")
        staged = Path(tmp) / "result.h5ad"
        ad.AnnData(empty, obs=obs, var=var, layers={"psbulk_props": empty}).write_h5ad(
            staged
        )
        with h5py.File(staged, "r+") as out, ThreadPoolExecutor(workers) as pool:
            matrices = [out["X"], out["layers/psbulk_props"]]
            for m in matrices:
                for field in ("indices", "data", "indptr"):
                    del m[field]
                    m.create_dataset(
                        field,
                        (1 if field == "indptr" else 0,),
                        maxshape=(None,),
                        dtype={"indices": "i4", "data": "f8", "indptr": "i8"}[field],
                    )
            totals = []
            # Bound completed results awaiting the ordered writer.
            for batch in range(0, len(shards), workers):
                for cols, offsets, sums, props, counts in pool.map(
                    reduce, shards[batch : batch + workers]
                ):
                    totals.append(counts)
                    for m, values in zip(matrices, (sums, props)):
                        for field, data in (
                            ("indptr", offsets + len(m["data"])),
                            ("indices", cols),
                            ("data", values),
                        ):
                            ds = m[field]
                            pos = len(ds)
                            ds.resize((pos + len(data),))
                            ds[pos:] = data
            out["obs/psbulk_counts"][:] = np.concatenate(totals)
        os.replace(staged, dest)


def aggregate_by(path, dest, sample, group=None, *, workers=1, chunk=4096, mode="sum"):
    with h5py.File(path) as src:
        obs = ad.io.read_elem(src["obs"])

    def codes(key):
        series = obs[key].astype("string")
        levels = sorted(series.dropna().unique())
        return pd.Categorical(series, categories=levels).codes.astype("int64"), levels

    sample_codes, samples = codes(sample)
    if group is None:
        labels = sample_codes
        names = samples
        metadata = pd.DataFrame({sample: samples}, index=names)
    else:
        group_codes, groups = codes(group)
        labels = group_codes * len(samples) + sample_codes
        labels[(group_codes < 0) | (sample_codes < 0)] = -1
        sample_axis = np.tile(samples, len(groups))
        group_axis = np.repeat(groups, len(samples))
        names = [f"{s}_{g}" for s, g in zip(sample_axis, group_axis)]
        metadata = pd.DataFrame({sample: sample_axis, group: group_axis}, index=names)
    aggregate(path, dest, labels, names, workers, chunk, mode=mode, metadata=metadata)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("dest")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--group")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--mode", choices=("sum", "mean"), default="sum")
    aggregate_by(**vars(parser.parse_args()))

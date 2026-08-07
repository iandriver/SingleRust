"""Build a larger .h5ad by tiling an existing CSR .h5ad R times along obs.

Used to scale the pseudobulk benchmark past the largest real dataset on hand without
re-downloading. Cells are repeated verbatim (obs index gets a `-rN` suffix so it stays
unique), so the sample x group cardinality — and therefore the work decoupler does — is
identical to the source; only the cell count grows. A useful side effect: the pseudobulk
sums of the R-tiled file must equal R x the source's, which is an exact correctness check.

Runs out-of-core: X is copied in bounded slices, never materialized.

    python demo/_replicate_cells.py <src.h5ad> <out.h5ad> <replicas>
"""
import sys

import h5py
import numpy as np

SLICE = 64_000_000  # elements per copy step (~256 MB for float32/int32)


def copy_tiled(src, dst_name, dst_parent, replicas, dtype=None):
    """Create dst dataset = src repeated `replicas` times, copying in bounded slices."""
    n = src.shape[0]
    dt = dtype or src.dtype
    out = dst_parent.create_dataset(dst_name, shape=(n * replicas,), dtype=dt, chunks=src.chunks)
    for r in range(replicas):
        base = r * n
        for i in range(0, n, SLICE):
            j = min(i + SLICE, n)
            out[base + i:base + j] = src[i:j]
    return out


def main():
    src_path, out_path, replicas = sys.argv[1], sys.argv[2], int(sys.argv[3])

    with h5py.File(src_path, "r") as f, h5py.File(out_path, "w") as g:
        n_obs, n_var = (int(x) for x in f["X"].attrs["shape"])
        nnz = f["X"]["data"].shape[0]
        total_nnz = nnz * replicas
        n_out = n_obs * replicas
        # CSR column indices stay < n_var, but indptr counts up to total nnz.
        indptr_dtype = np.int32 if total_nnz < np.iinfo(np.int32).max else np.int64
        print(f"{n_obs} -> {n_out} cells x {n_var} genes | nnz {nnz} -> {total_nnz} "
              f"| indptr {np.dtype(indptr_dtype).name}", flush=True)

        g.attrs.update({k: v for k, v in f.attrs.items()})

        X = g.create_group("X")
        X.attrs.update({k: v for k, v in f["X"].attrs.items()})
        X.attrs["shape"] = np.array([n_out, n_var], dtype=np.int64)
        copy_tiled(f["X"]["data"], "data", X, replicas)
        print("  data copied", flush=True)
        copy_tiled(f["X"]["indices"], "indices", X, replicas)
        print("  indices copied", flush=True)

        src_indptr = f["X"]["indptr"][:].astype(np.int64)  # (n_obs+1,), small
        out_indptr = np.empty(n_out + 1, dtype=indptr_dtype)
        for r in range(replicas):
            out_indptr[r * n_obs:(r + 1) * n_obs] = src_indptr[:-1] + r * nnz
        out_indptr[-1] = total_nnz
        X.create_dataset("indptr", data=out_indptr)
        print("  indptr built", flush=True)

        # ---- obs: tile categorical codes, make the index unique ----
        obs = g.create_group("obs")
        obs.attrs.update({k: v for k, v in f["obs"].attrs.items()})
        idx_key = f["obs"].attrs["_index"]
        src_idx = f["obs"][idx_key].asstr()[:]
        out_idx = np.concatenate([np.char.add(src_idx, f"-r{r}") for r in range(replicas)])
        obs.create_dataset(idx_key, data=out_idx.astype(object),
                           dtype=h5py.special_dtype(vlen=str))
        obs[idx_key].attrs.update({k: v for k, v in f["obs"][idx_key].attrs.items()})

        for key in f["obs"]:
            if key == idx_key:
                continue
            s, d = f["obs"][key], obs.create_group(key)
            d.attrs.update({k: v for k, v in s.attrs.items()})
            d.create_dataset("categories", data=s["categories"][:])
            d["categories"].attrs.update({k: v for k, v in s["categories"].attrs.items()})
            d.create_dataset("codes", data=np.tile(s["codes"][:], replicas))
            d["codes"].attrs.update({k: v for k, v in s["codes"].attrs.items()})
        print("  obs built", flush=True)

        f.copy("var", g)
        print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

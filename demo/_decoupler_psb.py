"""Run decoupler.pp.pseudobulk (in-memory) and write the result. Prints STEP_SECONDS.

    python demo/_decoupler_psb.py <input.h5ad> <sample_col> <group_col> <out.h5ad>
"""
import sys
import time

import hdf5plugin  # noqa: F401
import anndata as ad
import decoupler as dc


def main():
    inp, sample, group, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    adata = ad.read_h5ad(inp)  # full load into memory (decoupler needs it in memory)
    t = time.perf_counter()
    pdata = dc.pp.pseudobulk(adata, sample_col=sample, groups_col=group, mode="sum")
    dt = time.perf_counter() - t
    print(f"STEP_SECONDS={dt}")
    pdata.write_h5ad(out)


if __name__ == "__main__":
    main()

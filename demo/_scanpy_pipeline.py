"""Run the scanpy core pipeline on an .h5ad and print per-step + total compute time.

Mirrors `examples/bench_step.rs all` so the two can be compared as subprocesses (same input
file, each measured with `/usr/bin/time -l` for peak RSS). An internal warm-up on a small slice
absorbs scanpy's one-time numba JIT / thread-pool startup so the timed run is steady-state.

    python demo/_scanpy_pipeline.py <input.h5ad>
"""
import sys
import time
import gc

import hdf5plugin  # noqa: F401  (harmless; needed only if the input is blosc-compressed)
import scanpy as sc
import anndata as ad


def pipeline(a):
    a.var["mt"] = a.var_names.str.upper().str.startswith("MT-")
    t = {}

    def step(name, fn):
        s = time.perf_counter()
        fn()
        t[name] = time.perf_counter() - s

    step("qc", lambda: sc.pp.calculate_qc_metrics(
        a, qc_vars=["mt"], percent_top=[50, 100, 200, 500], log1p=True, inplace=True))
    step("normalize", lambda: sc.pp.normalize_total(a, target_sum=1e4))
    step("log1p", lambda: sc.pp.log1p(a))
    step("hvg", lambda: sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat"))
    step("pca", lambda: sc.pp.pca(a, n_comps=50, svd_solver="randomized",
                                  use_highly_variable=True, random_state=42))
    return t


def main():
    path = sys.argv[1]
    adata = ad.read_h5ad(path)

    # Warm-up (untimed) on a slice — JIT numba, spin up thread pools.
    warm = adata[:800].copy()
    pipeline(warm)
    del warm
    gc.collect()

    times = pipeline(adata)
    for k, v in times.items():
        print(f"ALL_{k}={v}")
    print(f"STEP_SECONDS={sum(times.values())}")


if __name__ == "__main__":
    main()

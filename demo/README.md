# SingleRust performance & in-place ergonomics

This folder demonstrates two things:

1. **In-place operations** — load an AnnData once and mutate it through the whole pipeline,
   no copies (`examples/inplace_pipeline.rs`).
2. **Why that's fast** — a per-step runtime benchmark of SingleRust vs scanpy on a real
   ~50k-cell CZI CELLxGENE dataset (`demo/scverse_benchmark.ipynb`).

## In-place pipeline

SingleRust's memory API uses interior mutability: you pass a shared `&adata` to each step and
results accumulate on that same object — QC columns into `obs`/`var`, the normalized/log1p
matrix into `X`, the HVG mask into `var`, the PCA embedding into `obsm["X_pca"]`. No per-step
reallocation of the matrix.

```rust
let adata = io::read_h5ad_memory("input.h5ad")?;          // load ONCE (note: not `mut`)

qc_metrics(&adata)?;                                       // -> obs / var
normalize_expression(&adata.x(), 10_000, &Direction::ROW, None)?;  // -> X (in place)
log1p_expression(&adata.x(), None)?;                       // -> X (in place)
compute_highly_variable_genes(&adata, Some(HVGParams { n_top_genes: Some(2000), ..Default::default() }))?;  // -> var
run_pca_inplace::<f64>(&adata, Some(hvg_selection), Some(true), Some(false),
                       Some(50), None, Some(42), Some(svd), None)?;   // -> obsm["X_pca"], uns
```

Run it:

```bash
cargo run --release --features enrichment --example inplace_pipeline -- data/input.h5ad
```

It prints the obs/var columns, `obsm`, and `uns` keys that accumulated on the single object.

## Benchmark

`scverse_benchmark.ipynb` runs each step **one at a time** — the SingleRust implementation and
the equivalent scanpy function — and reports **compute time only** (the Rust `.h5ad` read/write
is measured separately and excluded). scanpy produces the state *before* each step and the Rust
binary (`examples/bench_step.rs`) runs that same step on that same state, so the comparison is
apples-to-apples.

A warm-up pass runs first so the timings aren't charged for scanpy's one-time numba JIT / thread
pool startup. Indicative result on an 18-core machine, **50,000 cells × 35,507 genes**:

| step      | scanpy | SingleRust | speedup |
|-----------|-------:|-----------:|--------:|
| qc        | 0.85s  | 1.13s      | 0.75×   |
| normalize | 0.10s  | 0.01s      | 8.2×    |
| log1p     | 0.20s  | 0.15s      | 1.4×    |
| hvg       | 0.31s  | 0.12s      | 2.5×    |
| pca       | 6.44s  | 1.06s      | 6.1×    |
| ora\*     | 1.47s  | 0.46s      | 3.2×    |
| **total** | 9.37s  | 2.93s      | **3.2×**|

SingleRust wins decisively on the heavy/vectorizable steps (PCA ~6×, normalize ~8×). QC is the
exception — scanpy's optimized C path slightly beats SingleRust's `qc_metrics` here (the top-N
segment proportions are the cost), an honest result worth flagging rather than hiding.

\* ORA has no scanpy equivalent; compared against a NumPy/SciPy implementation of the same
hypergeometric algorithm. Numbers are machine-dependent — re-run the notebook to get yours.

### Scaling: runtime vs cell count

`scverse_scaling.ipynb` sweeps the cell count (3k → 50k) and runs the full core pipeline
(QC → normalize → log1p → HVG → PCA) at each size with both tools. SingleRust scales close to
linearly with the number of non-zeros, while scanpy carries higher fixed per-step overhead:

| cells  | scanpy | SingleRust | speedup |
|--------|-------:|-----------:|--------:|
| 3,000  | 1.80s  | 0.21s      | 8.5×    |
| 6,000  | 3.38s  | 0.37s      | 9.1×    |
| 12,500 | 5.59s  | 0.69s      | 8.1×    |
| 25,000 | 6.12s  | 1.35s      | 4.5×    |
| 50,000 | 7.77s  | 2.62s      | 3.0×    |

SingleRust is faster at every size (3–9×). The margin is largest at moderate sizes — where
scanpy's fixed overhead dominates — and narrows by 50k as both become compute-bound (PCA, the
heaviest step, stays a steady ~6×).

## Running the benchmark

```bash
# Python side (data fetch + scanpy + plotting)
python3 -m venv .venv && . .venv/bin/activate
pip install -r demo/requirements.txt

# Rust side needs cmake (anndata-hdf5 builds HDF5 from source): brew install cmake

jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 demo/scverse_benchmark.ipynb   # per-step head-to-head
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 demo/scverse_scaling.ipynb     # runtime vs cell count
```

The notebooks fetch the dataset (first run only, via `demo/prepare_data.py --n-cells 50000`),
build the Rust binary, and run the comparison.

## Files

- `examples/inplace_pipeline.rs` — annotated in-place pipeline.
- `examples/bench_step.rs` — runs a single step (or `all`), prints `STEP_SECONDS` (compute only).
- `demo/scverse_benchmark.ipynb` (+ `_build_benchmark_notebook.py`) — per-step benchmark.
- `demo/scverse_scaling.ipynb` (+ `_build_scaling_notebook.py`) — runtime-vs-cell-count scaling.
- `demo/prepare_data.py` — fetch a stratified blood slice from the CELLxGENE Census
  (`--n-cells/--per-type/--out`).
- `demo/markers.tsv` — immune-lineage marker sets for ORA.

## Note on compression

SingleRust writes blosc/zstd-compressed `.h5ad`, so reading those files from Python needs
`import hdf5plugin` before `anndata` (the notebook does this).

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

Indicative result on an 18-core machine, **50,000 cells × 35,507 genes**:

| step      | scanpy | SingleRust | speedup |
|-----------|-------:|-----------:|--------:|
| qc        | 4.58s  | 1.07s      | 4.3×    |
| normalize | 0.08s  | 0.01s      | 7.8×    |
| log1p     | 0.19s  | 0.14s      | 1.4×    |
| hvg       | 0.83s  | 0.12s      | 7.0×    |
| pca       | 6.73s  | 1.04s      | 6.5×    |
| ora\*     | 1.43s  | 0.44s      | 3.2×    |
| **total** | 13.8s  | 2.8s       | **4.9×**|

\* ORA has no scanpy equivalent; compared against a NumPy/SciPy implementation of the same
hypergeometric algorithm. Numbers are machine-dependent — re-run the notebook to get yours.

## Running the benchmark

```bash
# Python side (data fetch + scanpy + plotting)
python3 -m venv .venv && . .venv/bin/activate
pip install -r demo/requirements.txt

# Rust side needs cmake (anndata-hdf5 builds HDF5 from source): brew install cmake

jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 demo/scverse_benchmark.ipynb
```

The notebook fetches the dataset (first run only, via `demo/prepare_data.py --n-cells 50000`),
builds the Rust binary, and runs the comparison.

## Files

- `examples/inplace_pipeline.rs` — annotated in-place pipeline.
- `examples/bench_step.rs` — runs a single step, prints `STEP_SECONDS` (compute only).
- `demo/scverse_benchmark.ipynb` (+ `_build_benchmark_notebook.py`) — the benchmark.
- `demo/prepare_data.py` — fetch a stratified blood slice from the CELLxGENE Census
  (`--n-cells/--per-type/--out`).
- `demo/markers.tsv` — immune-lineage marker sets for ORA.

## Note on compression

SingleRust writes blosc/zstd-compressed `.h5ad`, so reading those files from Python needs
`import hdf5plugin` before `anndata` (the notebook does this).

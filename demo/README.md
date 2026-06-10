# SingleRust × scverse demo

An end-to-end demo that runs the newer SingleRust features on a real
[CZI CELLxGENE](https://cellxgene.cziscience.com/) blood dataset and visualizes the results
in scanpy — showcasing the **scverse round-trip**: compute in Rust, write `.h5ad`, plot in Python.

## What it exercises

| Feature | Where |
|---|---|
| `io::write_h5ad` — write results back to `.h5ad` | round-trip in the notebook |
| `run_pca_inplace` — `obsm["X_pca"]` + `uns` variance ratios | PCA / UMAP steps |
| dense `normalize_expression` / `log1p_expression` (no panic) | pipeline step 3 |
| `run_ora` — decoupler-style ORA over a marker network | ORA step |
| `qc_metrics` (incl. the top-segment-proportions fix) | QC step |

## Layout

- `examples/scverse_demo.rs` — the Rust pipeline (read → QC → normalize/log1p → HVG → PCA → ORA → write).
- `examples/bench_step.rs` — runs a **single** step and prints its compute time (for benchmarking).
- `demo/prepare_data.py` — fetches a stratified blood slice (raw counts) from the CELLxGENE Census.
  `--n-cells N --per-type K --out PATH` controls the size (default 3000 → `data/input.h5ad`).
- `demo/markers.tsv` — immune lineage marker sets (`pathway<TAB>gene`) used for ORA.
- `demo/scverse_demo.ipynb` — the demo notebook: runs the Rust example and visualizes its output.
- `demo/scverse_benchmark.ipynb` — **per-step SingleRust vs scanpy runtime benchmark** (~50k cells).
- `demo/_build_notebook.py` / `demo/_build_benchmark_notebook.py` — regenerate the `.ipynb`s
  (edit cells here, not the JSON).

## Benchmark

`scverse_benchmark.ipynb` runs each step one at a time — the SingleRust implementation and the
equivalent scanpy function — on a bigger (~50k cell) CELLxGENE slice, reporting **compute time
only** (the Rust `.h5ad` read/write is measured separately and excluded). scanpy produces the
state before each step and the Rust binary runs that same step on that same state, so the
comparison is apples-to-apples. Indicative result on an 18-core machine, 50k cells × 35.5k genes:

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
algorithm. Numbers depend on the machine — re-run the notebook to get yours.

## Prerequisites

- Rust toolchain + **`cmake`** (the `anndata-hdf5` dependency builds HDF5 from source).
  - macOS: `brew install cmake`
- Python 3.9+ with the demo deps (a virtualenv is recommended).

## Run it

```bash
# from the repo root
python3 -m venv .venv && . .venv/bin/activate
pip install -r demo/requirements.txt

# Option A: just open and run the notebook top-to-bottom
jupyter lab demo/scverse_demo.ipynb

# Option B: headless execution
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 demo/scverse_demo.ipynb
```

The notebook fetches the data (first run only), builds + runs the Rust example
(`cargo run --release --features enrichment --example scverse_demo`), then re-opens the
SingleRust output and plots QC, PCA, UMAP, and ORA activity.

You can also run the pipeline directly:

```bash
python demo/prepare_data.py                      # -> data/input.h5ad
cargo run --release --features enrichment --example scverse_demo -- \
    data/input.h5ad data/output.h5ad demo/markers.tsv
```

## Note on compression

SingleRust writes blosc/zstd-compressed `.h5ad`. To read those files from Python you must
`import hdf5plugin` **before** `anndata`/`h5py` (the notebook does this). A future
`write_h5ad` option for gzip/uncompressed output would remove this requirement.

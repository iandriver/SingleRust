# Out-of-core benchmark — memory vs scanpy

Same work in both lanes (QC + `normalize_total(1e4)` + `log1p`) on the same `.h5ad`, each run as
a subprocess under `/usr/bin/time -l` to capture **peak RSS** and wall time.

```bash
cargo build --release --example sr_ooc
python demo/bench_ooc.py data/bench_input_500k.h5ad 20000
```

## Result (500,000 cells × 48,788 genes, 48 GB / 18-core)

| lane | time | peak RSS |
|---|---:|---:|
| scanpy in-memory  | 13.5 s (compute) | 11.7 GB |
| scanpy + Dask OOC | 92.2 s (compute) | 7.5 GB |
| **SingleRust OOC** | 128.1 s (wall, incl. I/O) | **2.1 GB** |

**SingleRust OOC uses ~5.6× less peak memory than scanpy in-memory and ~3.6× less than
scanpy+Dask**, and that footprint is chunk-bounded — it stays roughly flat as the cell count
grows, whereas scanpy in-memory scales linearly and eventually OOMs (≈46 GB at 2M cells, over a
48 GB machine). That is the point of out-of-core: process data that does not fit in RAM.

The standout finding is the **Dask lane**: its sparse out-of-core path only modestly reduces
memory (7.5 vs 11.7 GB) while costing **~7× the runtime** (92 vs 13 s). This is exactly the
Dask-sparse immaturity scanpy's own issues describe — Dask helps a bit on memory but doesn't reach
truly bounded memory for sparse single-cell data. Native Rust streaming does (2.1 GB).

### Caveats / honest reading

- **Time is not apples-to-apples.** scanpy's number is compute-only (excludes its initial load);
  the Dask lane includes a lazy read + chunked write-out; SingleRust's wall time includes a 6 GB
  working-copy plus two full read/write passes to disk as *separate* CLI processes. A fused
  single-pass pipeline (+ chunk-size tuning, parallel chunk processing) would cut SingleRust's time
  substantially. **Peak RSS is the fair, durable signal** and it is immune to thermal/throttling.
- SingleRust OOC RSS includes one chunk (here 20k rows) plus small per-cell / per-gene
  accumulators; smaller chunks lower it further.
- Both scanpy lanes are run with the newer `.venv-dask` stack (anndata 0.12, scanpy 1.12, dask)
  for consistency; SingleRust is the `sr_ooc` release binary.

### Projection to 2M cells (not run — memory safety)

At ~2M cells the count matrix is ≈46 GB dense-equivalent; scanpy in-memory would exceed the 48 GB
machine (OOM/heavy swap), and the Dask lane's ~0.6× memory ratio still lands near the ceiling.
SingleRust OOC stays ~2 GB (chunk-bounded) and is the only lane that completes comfortably. We
did **not** run 2M here to avoid destabilizing the machine; it needs a ~32 GB base dataset and a
box with headroom.

## Reproducing the Dask lane

The Dask lane needs `anndata>=0.11` (`experimental.read_elem_lazy`) + dask + xarray, which require
Python ≥3.10 — kept in an isolated venv so the main 3.9 demo env is untouched:

```bash
python3.12 -m venv .venv-dask
. .venv-dask/bin/activate
pip install "anndata>=0.11" "scanpy>=1.11" "dask[array]" xarray hdf5plugin h5py scipy numpy
```

`bench_ooc.py` auto-uses `.venv-dask/bin/python` for both scanpy lanes (override with `SR_PY`).
Relevant scanpy issues: [#4095](https://github.com/scverse/scanpy/issues/4095),
[dask#11880](https://github.com/dask/dask/issues/11880),
[pydata/sparse#860](https://github.com/pydata/sparse/issues/860).

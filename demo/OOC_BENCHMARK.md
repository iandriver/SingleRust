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
| scanpy in-memory | 12.3 s (compute) | **12.4 GB** |
| SingleRust OOC   | 44.9 s (wall, incl. I/O) | **2.3 GB** |

**SingleRust OOC uses ~5.3× less peak memory**, and that footprint is chunk-bounded — it stays
roughly flat as the cell count grows, whereas scanpy in-memory scales linearly and eventually
OOMs (≈50 GB at 2M cells > 48 GB RAM). That is the point of out-of-core: process data that does
not fit in RAM.

### Caveats / honest reading

- **Time is not apples-to-apples.** scanpy's number is compute-only (excludes its initial load);
  SingleRust's wall time includes a 6 GB working-copy plus two full read/write passes to disk as
  *separate* CLI processes. A fused single-pass pipeline (and chunk-size tuning, parallel chunk
  processing) would cut that substantially. The memory result is the durable, fair signal.
- SingleRust OOC RSS includes one chunk (here 20k rows) plus small per-cell / per-gene
  accumulators; smaller chunks lower it further.

## scanpy + Dask lane (pending)

A true scanpy out-of-core lane needs `anndata>=0.11`'s `read_elem_as_dask` (this env has 0.10.9,
which lacks it) and, per scanpy's own issues, the Dask **sparse** path is immature
([dask#11880](https://github.com/dask/dask/issues/11880),
[pydata/sparse#860](https://github.com/pydata/sparse/issues/860),
[scanpy#4095](https://github.com/scverse/scanpy/issues/4095)). Adding it cleanly means an isolated
venv with the newer stack. `demo/bench_ooc.py` leaves a hook for the lane; scaling to 1M–2M also
needs a larger base dataset.

# Out-of-core preprocessing — memory vs scanpy

> **Tier-1 OOC pipeline is complete.** All of QC, `normalize_total`, `log1p`, a fused
> `preprocess`, highly variable genes (Seurat), and PCA run disk-backed in bounded memory, exposed
> as `sr_ooc <cmd>` and `sr.pp.*` (drop-in for `sc.pp.*`). Each step is unit-tested against its
> in-memory / dense counterpart: log1p & normalize match scanpy's X exactly; QC matches
> (totals/mito/top-N/var); HVG selects the **same** genes as SingleRust's in-memory HVG; PCA
> matches a dense covariance-PCA reference (variance ratios exact, embedding norms match, up to
> per-component sign). End-to-end vs scanpy, the PCA *spectrum* tracks scanpy but the exact axes
> differ because SingleRust's Seurat HVG picks a different gene set than scanpy's (a pre-existing
> in-memory difference, not introduced by the OOC path).

The benchmark below covers the QC + normalize_total + log1p portion.

## Pseudobulk vs decoupler

`dc.pp.pseudobulk` aggregates single-cell counts per `sample × group`. decoupler loops the full
sample×group cartesian product and, for each, boolean-masks **all** cells and **densifies** the
submatrix (`X[mask].toarray()`) — `O(n_obs · n_groups)` masking plus per-group densification, with
the whole input resident. SingleRust does it in a **single streaming sparse scatter-add** (each
cell → its group accumulator): `O(nnz)`, one pass, only the small `groups × genes` output resident.

```bash
python demo/bench_pseudobulk.py data/psb_input.h5ad donor cell_type 10000
```

48,788 genes, **12 donors × 143 cell types = 1,716 groups** throughout (48 GB / 18-core, chunk
10,000). The 1M/2M inputs are the 500k dataset tiled 2×/4× along obs (`demo/_replicate_cells.py`),
so the group cardinality — and thus the work decoupler does — is held constant and only the cell
count grows.

| cells | decoupler time | decoupler peak RSS | SingleRust time | SingleRust peak RSS |
|---:|---:|---:|---:|---:|
| 500k | 29.9 s | 21.6 GB | 12.2 s | 3.4 GB |
| 1M | 167.8 s ⚠ | 19.2 GB ⚠ | 19.3 s | 2.4 GB |
| 2M | not run (see below) | — | 26.1 s | 3.6 GB |

**SingleRust's memory is flat (~2.4–3.6 GB) and its time is sub-linear** — 4× the cells costs 2.1×
the time. Both follow from the design: peak RSS is dominated by the fixed `groups × genes`
accumulator plus one chunk (neither depends on cell count), and the fixed output allocate/write cost
amortizes as cells grow. decoupler's cost is driven by `O(n_obs · n_groups)` masking plus per-group
densification with the whole input resident, so it grows with cells even at constant group count.

**Exactness at every size.** SingleRust vs decoupler at 1M: aggregate-sum max abs diff = **0** over
all 1,716 groups. Tiling also gives a free self-check — the 1M and 2M sums must equal exactly 2× and
4× the 500k sums, and they do (max abs diff = 0, `psbulk_cells` likewise).

`mode="sum"` (default) and `"mean"` are supported; `obs` carries `psbulk_cells`/`psbulk_counts` and
`layers["psbulk_props"]` holds the non-zero fraction — matching decoupler's outputs.

### Caveats — read before quoting these numbers

- **⚠ The 1M decoupler run is contaminated.** An unrelated 8–16 GB process was running on the same
  machine for part of it, and decoupler was paging: 185 s wall against only 122 s CPU
  (71 user + 51 sys). Its true quiet-machine time is lower than 167.8 s. Treat the 1M row as
  "decoupler degrades sharply once it no longer fits" — not as a precise 8.7× ratio.
- **⚠ Peak RSS understates decoupler's demand at 1M.** Resident set fell from 13.5 GB to ~5 GB
  mid-run as the OS evicted pages to swap, so 19.2 GB is a *ceiling on residency*, not on demand —
  which is why it reads lower than the 500k row despite twice the data. Once a lane swaps, RSS stops
  being a fair memory metric.
- **The 500k row is a fresh cold-cache re-measurement.** An earlier version of this file reported
  5.1 s / 4.6 GB for SingleRust at 500k. That run followed decoupler over the same file, so it read
  from a warm page cache. Cold, it is 12.2 s. The corrected gap at 500k is ~2.4×, not ~5.9×.
- **Time is still not apples-to-apples, in decoupler's favor on I/O:** decoupler's number is
  compute-only (`dc.pp.pseudobulk` alone, excluding its data load), while SingleRust's is wall time
  including reading the file and writing the result.
- **2M decoupler was not run.** It needs ~24 GB for the CSR matrix alone before any aggregation; at
  the time of the run a neighbor process held 16.2 GB and swap was 98% used, so attempting it risked
  destabilizing the machine rather than producing a usable number.
- **nnz overflows int32 past ~1.4M cells at this density.** The 2M file (nnz = 3.01e9) requires an
  int64 CSR `indptr`; SingleRust reads it without issue, but it is a real cliff for any 32-bit
  index path.

**Deterministic parallel scatter.** The scatter-add is parallelized by partitioning groups across
a fixed 16 buckets (`group % 16`); each bucket is owned by one thread, so every group is summed by
one thread in cell order — no cross-thread merge of any group, hence bit-identical regardless of
thread count (the partition buffers add ~1.4 GB vs the sequential version). Verified at scale:
`RAYON_NUM_THREADS=1` vs `18` on 500k gave bit-identical X, counts, and props.

## Deterministic parallelism

The compute-bound passes — QC accumulation, HVG sum/sum-of-squares, and PCA's gene×gene Gram
matrix — are parallelized with rayon. Naive parallel floating-point reduction is **not**
reproducible (FP addition isn't associative and rayon's work-stealing varies the summation order),
so all reductions use a **fixed-block, ordered-merge** scheme (`backed::processing::det`): rows are
cut into fixed-size blocks at fixed indices, each block is folded sequentially, and partials are
merged in block order. The partition and merge order depend only on the data size and a constant
block size — never on thread count or scheduling — so results are bit-identical on every run.

Verified two ways:
- **Unit tests** run QC / HVG / preprocess / PCA under rayon pools of 1 vs 8 threads and assert
  bit-identical metrics, masks, embeddings, and variance ratios.
- **At scale (50k cells, real data):** the full pipeline (`preprocess → hvg → pca`) run with
  `RAYON_NUM_THREADS=1` vs `=18` produced **bit-identical** X, obs QC columns, var HVG columns,
  `obsm["X_pca"]`, and `uns` variance ratios.

(The cheap elementwise transforms — normalize/log1p — have no cross-row reduction and are
deterministic by construction; PCA projection is per-cell independent, also order-free.)


Same work in both lanes (QC + `normalize_total(1e4)` + `log1p`) on the same `.h5ad`, each run as
a subprocess under `/usr/bin/time -l` to capture **peak RSS** and wall time.

```bash
cargo build --release --example sr_ooc
python demo/bench_ooc.py data/bench_input_500k.h5ad 20000
```

## Result (500,000 cells × 48,788 genes, 48 GB / 18-core)

| lane | time | peak RSS |
|---|---:|---:|
| scanpy in-memory  | 13.5 s (compute) | 13.6 GB |
| scanpy + Dask OOC | 35.4 s (compute) | 14.5 GB |
| **SingleRust OOC (fused)** | 45.0 s (wall, incl. I/O) | **2.5 GB** |

**SingleRust OOC uses ~5.4× less peak memory than scanpy in-memory and ~5.8× less than
scanpy+Dask**, and that footprint is chunk-bounded — it stays roughly flat as the cell count
grows, whereas scanpy in-memory scales linearly and eventually OOMs (≈46 GB at 2M cells, over a
48 GB machine). That is the point of out-of-core: process data that does not fit in RAM.

The standout finding is the **Dask lane**: its sparse out-of-core path gives essentially **no
memory benefit** — peak RSS lands at/above scanpy in-memory (it has measured 7.5–14.5 GB across
runs, i.e. ≥ in-memory) while still costing 2.6× the runtime. This is exactly the Dask-sparse
immaturity scanpy's own issues describe. Native Rust streaming is the only lane with truly bounded
memory (2.5 GB).

The SingleRust lane is a **single fused command** (`sr_ooc preprocess` = QC + normalize_total +
log1p in one job; the per-cell total computed for QC is reused as the normalization row-sum, so
it's computed once). Fusing cut its wall time from 128 s (three separate ops + a working copy) to
45 s — now in the same ballpark as Dask on time, at ~1/6th the memory.

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

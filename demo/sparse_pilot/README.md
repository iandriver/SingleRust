# Pseudobulk by disk partition

This experimental command reads a count matrix in cell order. It writes each observed gene and its count to a temporary file for a range of output groups. Each worker reads its assigned file into a bounded dense buffer. The writer emits CSR rows in group order, including empty donor × cell-type combinations. This trades a second disk pass for a peak accumulator size set by the partition width. The complete donor × gene matrix never needs to reside in RAM.

The Python driver handles H5AD, stable partitioning and CSR output. A Rust shared library accumulates counts and detections. This is a separate prototype from the crate's dense `sr_ooc pseudobulk` command.

## Run on a backed dataset

From this directory, on macOS with Rust and Python 3.12:

```sh
rustc -O --crate-type cdylib scatter.rs -o libscatter.dylib
python test_partitioned.py
python partitioned.py counts.h5ad donor-celltype.h5ad --sample donor_id --group cell_type --workers 4
```

The input must store nonnegative integer counts in CSR `X`. The command leaves the source open for reads only and replaces the destination after the writer closes. Sum is the default. Use `--mode mean` for a per-cell mean. Missing group labels skip the cell. A zero-count cell with valid labels contributes to its group's denominator. Stored zeros contribute zero detections. Duplicate entries for a gene in one cell count as a single detection.

Allow scratch space on the output filesystem for up to eight bytes per retained input nonzero, plus the new output and any existing destination. Sparse storage can take more space than dense storage for large groups that express most genes.

## Reproduce the benchmark

Use the Emont WAT reference H5AD with raw counts in `raw/X`. The full-width fixture refers back to that file via HDF5 virtual datasets. Keep the source at the same absolute path. This repeats 166,149 source cells to reach 500,000 and keeps source gene order. The optional 256-gene panel selects evenly spaced gene positions. Both fixtures assign cell i to group i modulo the requested number of groups, interleaving the groups throughout the input.

```sh
export NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python full_fixture.py emont2022_human_WAT.h5ad full.h5ad 500000
python sweep.py full.h5ad full-results 1000 10000 100000
python run_pilot.py fixture emont2022_human_WAT.h5ad panel.h5ad 500000 256
python sweep.py panel.h5ad panel-results 1000 10000 100000
```

The sweep measures wall time from Python startup through writing H5AD, including library imports. Scanpy computes sum and count of nonzero cells per gene with threaded Dask. Backed decoupler and the prototype also write proportions and per-group cell and count totals. All workers are threads in the measured process. The macOS high-water RSS includes the Python runtime. The runner checks a 6 GB RSS threshold and preflights dense results that cannot fit. A resource-limited run has no completion time. Set `BENCH_TIMEOUT` in seconds to change the default ten-minute wall budget per lane.

For verification, SciPy independently adds strided blocks of cells in their original order. The checker reads every result row, compares sums, detections or proportions, checks gene and group identities, and verifies cell and count totals where present. The sweep hashes both CSR structures, values and totals across one worker and two four-worker runs. The edge-case test also covers empty combinations, means, skipped cells, stored zeros, duplicate coordinates, invalid counts and preservation of prior output on rejection.

To test the proposed Scanpy sparse aggregation implementation, put its checkout's `src` directory first on `PYTHONPATH` and use the `scanpy-pr` lane:

```sh
PYTHONPATH=/path/to/scanpy/src python measure.py run_pilot.py scanpy-pr full.h5ad pr-output.h5ad 1000 4
python run_pilot.py check full.h5ad pr-output.h5ad 1000 4
```

The PR lane explicitly requests `keep_sparse=True`. At commit `e92b11cc353cdaa45ecb08892993217fad4b429c`, the delayed Dask arrays advertise dense metadata and return SciPy CSR blocks. With AnnData 0.13.4, a direct write failed on a 64-cell identity-matrix fixture. The benchmark supplies CSR block metadata with an identity `map_blocks` call so AnnData uses its sparse writer. This does not change any aggregate values.

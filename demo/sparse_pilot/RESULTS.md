# Disk-partitioned pseudobulk on 500,000 rows

The source was the Emont human adipose reference (`emont2022_human_WAT.h5ad`, SHA-256 `ccc6c21bcb53ca76f46d795972e587b49f174199de74598aee644c2867d84d29`). The full fixture contains 869,089,608 stored counts across 28,101 genes. The 256-gene fixture contains 7,658,459. The 500,000 rows repeat 166,149 original cells. Group membership is synthetic and interleaved. These fixtures measure software scaling; they contain no independent donor-level contrasts.

The measurements used macOS arm64, Rust 1.97.0, Scanpy 1.12.6, Dask 2026.8.0, decoupler 2.3.1, AnnData 0.13.4, NumPy 2.5.5, SciPy 1.18.1 and h5py 3.16.0. All results below include the output write. Units are decimal GB. The baseline memory threshold was 6 GB, sampled every 100 ms. An observed high-water mark can exceed the threshold before the guard gets CPU time.

## What the wide result costs

| Output groups | Cells/group | Nonzero group × gene entries | Density | Prototype H5AD size |
|---:|---:|---:|---:|---:|
| 1,000 | 500 | 22,561,455 | 80.29% | 0.549 GB |
| 10,000 | 50 | 150,866,845 | 53.69% | 3.649 GB |
| 100,000 | 5 | 587,954,070 | 20.92% | 14.216 GB |

Both sums and detection proportions have CSR data and indices. At 100,000 groups, one fully dense float64 matrix alone would use 22.48 GB. Conversely, when 80% of entries are present, writing sparse column indices makes the output larger than a dense result. At 1,000 groups decoupler wrote a 0.453 GB file. The input pass also needs temporary storage of up to 6.95 GB for partition records on this fixture.

## Correctness

The independent reference adds the rows for each group directly with SciPy. Every completed group was checked across all genes. Sums and cell/count totals must match exactly. Proportions may differ by at most 1e-7 from the reference. The repeatability check is stricter and requires exactly matching SHA-256 digests of the array structure and numerical data.

| Full-width groups | SHA-256 of canonical numeric output |
|---:|---|
| 1,000 | `c0a8d0d3600f7d0d5560ed9efcab83de3396e9729f98f36f74ec3788f31c9bdf` |
| 10,000 | `3b332b6aacf685b851ce1f4393c940cb1e5b1fa261749c63b25b50ae4d9971b4` |
| 100,000 | `a1e1875b29bb59486355d0404bd3870aa9ffa2b5a53d513041a015a2b71fcfb7` |

## Full-width prototype measurements

Each pair gives wall seconds / peak GB. Both four-worker runs and the single-worker run had the same digest.

| Groups | One worker | Four workers, run 1 | Four workers, run 2 |
|---:|---:|---:|---:|
| 1,000 | 46.0 / 4.33 | 43.3 / 4.58 | 40.6 / 4.67 |
| 10,000 | 50.9 / 3.80 | 47.2 / 4.43 | 45.0 / 4.65 |
| 100,000 | 75.1 / 4.18 | 69.2 / 4.08 | 70.2 / 4.48 |

## Baselines

At full width, backed decoupler completed the 1,000-group case in 152.2 s at 2.48 GB peak RSS. At 10,000 groups it crossed 6 GB. Released Scanpy with threaded Dask crossed 6 GB at four workers for both sizes. A one-worker retry completed the 1,000-group case in 57.7 s at 5.28 GB and crossed 6 GB at 10,000 groups. Both released dense-output lanes were skipped at 100,000 groups on the 22.48 GB single-matrix preflight. That entry has no observed runtime or RSS.

We also ran the proposed sparse-output change in [Scanpy PR #4073](https://github.com/scverse/scanpy/pull/4073), pinned at `e92b11c`, with the sparse-writer metadata adjustment described in the README. Its full-width runs crossed 6 GB at both one and four workers for all three group counts. This applies to the tested chunk size and Dask scheduler.

### Narrow panel, 500,000 × 256

Values give wall seconds / peak GB. Ranges cover two four-worker prototype runs. Baselines are single runs.

| Groups | Prototype | Released Scanpy + Dask | Scanpy sparse PR + Dask | Backed decoupler |
|---:|---:|---:|---:|---:|
| 1,000 | 0.95–0.96 / 0.28–0.30 | 2.41 / 0.59 | 2.40 / 0.53 | 8.16 / 0.53 |
| 10,000 | 1.08 / 0.42–0.45 | 2.90 / 1.21 | 3.13 / 0.83 | 36.68 / 0.60 |
| 100,000 | 1.75–1.85 / 0.67–0.70 | 18.02 / 3.98 (one worker) | 10.91 / 1.90 | stopped at 600 s / 0.83 |

The sparse PR completed the 100,000-group panel at four workers, where the released version crossed 6 GB at four workers. Its successful result passed the same full SciPy check. Decoupler reached the panel time limit with memory below 1 GB; this result provides only a lower bound on its completion time.

### Interpretation

Gene width changes the answer. The full input carries over 100 times as many stored entries as the narrow panel. The prototype pays for scratch writes and reads, plus column indices in each output layer. Its main benefit here was completing wide, high-group-count outputs under a fixed RSS threshold. Backed decoupler used less RAM in the full-width 1,000-group case. Individual wall times depend on software startup, input compression, page cache, the storage device, and Dask worker count. Cache state was uncontrolled. The benchmark supplies no estimate of cold-read throughput, biological replication, or performance on a different group-size distribution.

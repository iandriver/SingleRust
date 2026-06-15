"""Generate demo/scverse_scaling.ipynb — cells-vs-runtime scaling: SingleRust vs scanpy."""
import pathlib
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(new_code_cell(s.strip("\n")))

md(r"""
# Scaling: runtime vs number of cells — SingleRust vs scanpy

How does the **core pipeline** (QC → normalize → log1p → HVG → PCA) scale as the cell count
grows? We subsample a CZI CELLxGENE blood dataset to several sizes and, at each size, run the
full pipeline with both SingleRust and scanpy, timing **compute only**.

- SingleRust runs the whole pipeline in one process (`bench_step … all`), feeding each step into
  the next in memory (the in-place pattern) — per-step and total compute times are printed.
- scanpy runs the same five steps on the same raw subsample, timed in-process.
""")

md("## Setup")
code(r"""
import hdf5plugin
import os, sys, subprocess, pathlib, time
import numpy as np, pandas as pd
import scanpy as sc, anndata as ad
import matplotlib.pyplot as plt

ROOT = pathlib.Path.cwd()
if ROOT.name == "demo":
    ROOT = ROOT.parent
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
BENCH_INPUT = DATA / "bench_input.h5ad"
N_BASE = 50_000
SIZES = [3_000, 6_000, 12_500, 25_000, 50_000]   # cells-vs-runtime sweep points
SEED = 0

ENV = dict(os.environ)
ENV["PATH"] = "/opt/homebrew/bin:" + str(pathlib.Path.home()/".cargo"/"bin") + ":" + ENV.get("PATH","")
BIN = ROOT / "target" / "release" / "examples" / "bench_step"

import multiprocessing; print("CPU cores:", multiprocessing.cpu_count())
subprocess.run(["cargo","build","--release","--features","enrichment","--example","bench_step"],
               cwd=ROOT, env=ENV, check=True, capture_output=True, text=True)

def rust_all(path):
    # Run the full Rust pipeline once; return {step: seconds, ..., 'total': seconds}.
    p = subprocess.run([str(BIN), str(path), "all", "-"], cwd=ROOT, env=ENV,
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-800:]); print(p.stderr[-1500:]); raise RuntimeError("rust all failed")
    out = {}
    for line in p.stdout.splitlines():
        k, _, v = line.partition("=")
        if k.startswith("ALL_"): out[k[4:]] = float(v)
        elif k == "STEP_SECONDS": out["total"] = float(v)
    return out

def scanpy_all(a):
    # Same five steps, timed individually; returns {step: seconds, ..., 'total': seconds}.
    a.var["mt"] = a.var_names.str.upper().str.startswith("MT-")
    t = {}
    def step(name, fn):
        s = time.perf_counter(); fn(); t[name] = time.perf_counter() - s
    step("qc", lambda: sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=[50,100,200,500],
                                                  log1p=True, inplace=True))
    step("normalize", lambda: sc.pp.normalize_total(a, target_sum=1e4))
    step("log1p", lambda: sc.pp.log1p(a))
    step("hvg", lambda: sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat"))
    step("pca", lambda: sc.pp.pca(a, n_comps=50, svd_solver="randomized",
                                  use_highly_variable=True, random_state=42))
    t["total"] = sum(t.values())
    return t
""")

md("## Load the base dataset (raw counts)")
code(r"""
if not BENCH_INPUT.exists():
    subprocess.run([sys.executable, str(ROOT/"demo"/"prepare_data.py"),
                    "--n-cells", str(N_BASE), "--per-type", "2000", "--out", str(BENCH_INPUT)],
                   check=True)
raw = ad.read_h5ad(BENCH_INPUT)
print(raw.shape)
""")

md(r"""
## Sweep cell counts

At each size we write a raw-counts subsample for the Rust binary and run both pipelines. The
full 50k uses the base file directly (no resampling).
""")
code(r"""
rng = np.random.default_rng(SEED)

# Warm-up (untimed): scanpy's HVG/PCA paths JIT-compile and spin up thread pools on first call.
# Run both pipelines once on a tiny subsample so the first *timed* size isn't penalized.
_warm = raw[np.sort(rng.choice(raw.n_obs, size=min(800, raw.n_obs), replace=False))].copy()
scanpy_all(_warm.copy())
_wp = DATA / "scale_warmup.h5ad"; _warm.write_h5ad(_wp); rust_all(_wp)
print("warm-up done")

rows = []  # long form: n_cells, impl, step, seconds
for n in SIZES:
    if n >= raw.n_obs:
        sub, path = raw, BENCH_INPUT
    else:
        idx = np.sort(rng.choice(raw.n_obs, size=n, replace=False))
        sub = raw[idx].copy()
        path = DATA / f"scale_{n}.h5ad"
        sub.write_h5ad(path)
    rt = rust_all(path)
    st = scanpy_all(sub.copy())
    for step in ["qc","normalize","log1p","hvg","pca","total"]:
        rows.append(dict(n_cells=n, impl="SingleRust", step=step, seconds=rt[step]))
        rows.append(dict(n_cells=n, impl="scanpy", step=step, seconds=st[step]))
    print(f"{n:6d} cells | scanpy {st['total']:6.2f}s | rust {rt['total']:6.2f}s "
          f"| speedup {st['total']/rt['total']:4.2f}×")
df = pd.DataFrame(rows)
""")

md("## Total runtime vs cell count")
code(r"""
tot = df[df.step=="total"].pivot(index="n_cells", columns="impl", values="seconds")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(tot.index, tot["scanpy"], "o-", label="scanpy")
ax[0].plot(tot.index, tot["SingleRust"], "s-", label="SingleRust")
ax[0].set(xscale="log", yscale="log", xlabel="cells", ylabel="compute seconds",
          title="Full pipeline runtime vs cells (log–log)")
ax[0].legend(); ax[0].grid(True, which="both", alpha=.3)

speedup = tot["scanpy"] / tot["SingleRust"]
ax[1].plot(speedup.index, speedup.values, "D-", color="#2b8cbe")
ax[1].axhline(1.0, color="k", lw=1, ls="--")
ax[1].set(xscale="log", xlabel="cells", ylabel="speedup (scanpy / SingleRust)",
          title="Speedup vs cells")
ax[1].grid(True, which="both", alpha=.3)
plt.tight_layout(); plt.show()
tot.assign(speedup=speedup).style.format("{:.3f}")
""")

md("## Per-step scaling")
code(r"""
steps = ["qc","normalize","log1p","hvg","pca"]
fig, axes = plt.subplots(1, len(steps), figsize=(4*len(steps), 3.6), sharex=True)
for ax, step in zip(axes, steps):
    d = df[df.step==step].pivot(index="n_cells", columns="impl", values="seconds")
    ax.plot(d.index, d["scanpy"], "o-", label="scanpy")
    ax.plot(d.index, d["SingleRust"], "s-", label="SingleRust")
    ax.set(xscale="log", yscale="log", title=step, xlabel="cells")
axes[0].set_ylabel("seconds"); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md(r"""
## Notes

- Compute time only (Rust `.h5ad` read/write excluded); both sides use all cores.
- Each size runs both pipelines on the **same** raw subsample. Single run per size — indicative.
- ORA is omitted here (no scanpy equivalent); see `scverse_benchmark.ipynb` for the per-step
  head-to-head including ORA.
""")

nb = new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = pathlib.Path(__file__).resolve().parent / "scverse_scaling.ipynb"
nbf.write(nb, str(out))
print("wrote", out)

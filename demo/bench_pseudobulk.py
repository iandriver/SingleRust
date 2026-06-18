"""Benchmark pseudobulk: decoupler (in-memory) vs SingleRust OOC (streaming).

Aggregates a single-cell .h5ad by sample × group. Each lane runs under /usr/bin/time -l for
peak RSS + compute time; the resulting pseudobulk matrices are then checked bit/near-identical.

    python demo/bench_pseudobulk.py <input.h5ad> <sample_col> <group_col> [chunk]

If <input.h5ad> lacks <sample_col>, pass --make-donors N to synthesize N donors first via
demo/_prep_pseudobulk.py.
"""
import os
import re
import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "examples" / "sr_ooc"
ENV = dict(os.environ)
ENV["PATH"] = "/opt/homebrew/bin:" + str(pathlib.Path.home() / ".cargo" / "bin") + ":" + ENV.get("PATH", "")
ENV.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
PY = os.environ.get("SR_PY", str(ROOT / ".venv-dask" / "bin" / "python"))


def run_timed(cmd):
    p = subprocess.run(["/usr/bin/time", "-l", *cmd], cwd=ROOT, env=ENV, capture_output=True, text=True)
    if p.returncode:
        sys.stderr.write(p.stdout[-1500:] + "\n" + p.stderr[-2500:] + "\n")
        raise RuntimeError(" ".join(map(str, cmd)))
    secs = next((float(l.split("=", 1)[1]) for l in p.stdout.splitlines() if l.startswith("STEP_SECONDS=")), None)
    m = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
    rss = int(m.group(1)) / 1e9 if m else float("nan")
    w = re.search(r"([\d.]+)\s+real", p.stderr)
    wall = float(w.group(1)) if w else float("nan")
    return (secs if secs is not None else wall), rss


def main():
    inp = pathlib.Path(sys.argv[1])
    sample, group = sys.argv[2], sys.argv[3]
    chunk = sys.argv[4] if len(sys.argv) > 4 else None
    if not BIN.exists():
        raise SystemExit("build the CLI: cargo build --release --example sr_ooc")

    import h5py
    with h5py.File(inp) as f:
        shape = list(f["X"].attrs.get("shape", []))
    print(f"dataset: {inp.name}  shape={shape}  by {sample} × {group}\n")

    dc_out = inp.with_suffix(".dc_psb.h5ad")
    dt, drss = run_timed([PY, str(ROOT / "demo" / "_decoupler_psb.py"), str(inp), sample, group, str(dc_out)])
    print(f"decoupler (in-mem) : {dt:8.2f}s | peak RSS {drss:6.2f} GB")

    sr_out = inp.with_suffix(".sr_psb.h5ad")
    cmd = [str(BIN), "pseudobulk", str(inp), "--out", str(sr_out), "--sample-col", sample, "--group-col", group]
    if chunk:
        cmd += ["--chunk", str(chunk)]
    rt, rrss = run_timed(cmd)
    print(f"SingleRust OOC     : {rt:8.2f}s | peak RSS {rrss:6.2f} GB")

    print()
    if dt and rt:
        print(f"time:   SingleRust {dt / rt:.1f}× faster than decoupler")
    if drss and rrss:
        print(f"memory: SingleRust {drss / rrss:.1f}× less peak RAM than decoupler")

    # ---- correctness: aligned aggregate sums ----
    import hdf5plugin  # noqa: F401
    import anndata as ad
    import numpy as np
    A = ad.read_h5ad(dc_out)
    B = ad.read_h5ad(sr_out)
    Ax = A.X.toarray() if hasattr(A.X, "toarray") else np.asarray(A.X)
    Bx = B.X.toarray() if hasattr(B.X, "toarray") else np.asarray(B.X)
    # align rows by index, columns by var name
    import pandas as pd
    A_df = pd.DataFrame(Ax, index=A.obs_names, columns=A.var_names)
    B_df = pd.DataFrame(Bx, index=B.obs_names, columns=B.var_names)
    common = A_df.index.intersection(B_df.index)
    A_al = A_df.loc[common, A.var_names].values
    B_al = B_df.loc[common, A.var_names].values
    maxdiff = float(np.max(np.abs(A_al - B_al))) if len(common) else float("nan")
    print(f"\naggregate-sum max abs diff (aligned {len(common)} groups): {maxdiff:.6g}")
    print("rows: decoupler", A.n_obs, "| SingleRust", B.n_obs)
    for f in [dc_out, sr_out]:
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

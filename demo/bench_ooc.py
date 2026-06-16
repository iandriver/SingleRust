"""Out-of-core memory benchmark: scanpy in-memory vs SingleRust OOC.

Same work in both lanes — QC + normalize_total(1e4) + log1p — on the same .h5ad. Each lane runs
as a subprocess under /usr/bin/time -l so we capture **peak RSS** (the headline) and compute time:

  * scanpy in-memory : loads the whole matrix; peak RSS scales with the dataset.
  * SingleRust OOC   : streams the file in chunks; peak RSS stays ~bounded.

Usage:
    python demo/bench_ooc.py <file.h5ad> [chunk_size]

(A scanpy+Dask out-of-core lane needs anndata>=0.11's read_elem_as_dask; see notes in the
performance roadmap. This harness leaves a hook for it.)
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

# Python with the newer stack (anndata>=0.11 + dask) for both scanpy lanes; override with SR_PY.
PY = os.environ.get("SR_PY", str(ROOT / ".venv-dask" / "bin" / "python"))


def run_timed(cmd):
    """Run under /usr/bin/time -l; return (compute_seconds|None, peak_rss_gb)."""
    p = subprocess.run(["/usr/bin/time", "-l", *cmd], cwd=ROOT, env=ENV, capture_output=True, text=True)
    if p.returncode:
        sys.stderr.write(p.stdout[-1000:] + "\n" + p.stderr[-2000:] + "\n")
        raise RuntimeError(f"failed: {' '.join(map(str, cmd))}")
    secs = None
    for line in p.stdout.splitlines():
        if line.startswith("STEP_SECONDS="):
            secs = float(line.split("=", 1)[1])
    m = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
    rss_gb = int(m.group(1)) / 1e9 if m else float("nan")  # macOS: bytes
    w = re.search(r"([\d.]+)\s+real", p.stderr)  # wall time from /usr/bin/time -l
    wall = float(w.group(1)) if w else float("nan")
    return secs if secs is not None else wall, rss_gb


def scanpy_inmem(path):
    return run_timed([PY, str(ROOT / "demo" / "_scanpy_inmem_pp.py"), str(path)])


def scanpy_dask(path, chunk):
    args = [PY, str(ROOT / "demo" / "_scanpy_dask_pp.py"), str(path)]
    if chunk:
        args.append(str(chunk))
    return run_timed(args)


def singlerust_ooc(path, chunk):
    # Single fused pass (qc + normalize_total + log1p), reading source -> temp output. No 6 GB
    # copy and one process, so wall time and peak RSS are a single clean /usr/bin/time -l measure.
    out = path.with_suffix(".ooc_out.h5ad")
    chunk_args = ["--chunk", str(chunk)] if chunk else []
    t, r = run_timed([str(BIN), "preprocess", str(path), "--out", str(out),
                      "--target-sum", "10000.0", *chunk_args])
    out.unlink(missing_ok=True)
    return t, r


def main():
    if len(sys.argv) < 2:
        print("usage: python demo/bench_ooc.py <file.h5ad> [chunk_size]")
        sys.exit(2)
    path = pathlib.Path(sys.argv[1])
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if not BIN.exists():
        raise SystemExit(f"build the CLI first: cargo build --release --example sr_ooc  ({BIN} missing)")

    import h5py
    with h5py.File(path) as f:
        shape = list(f["X"].attrs.get("shape", []))
    print(f"dataset: {path.name}  shape={shape}  chunk={chunk or 'default'}\n")

    s_t, s_rss = scanpy_inmem(path)
    print(f"scanpy in-memory : {s_t:7.2f}s compute | peak RSS {s_rss:6.2f} GB")

    if pathlib.Path(PY).exists():
        try:
            d_t, d_rss = scanpy_dask(path, chunk)
            print(f"scanpy + Dask OOC: {d_t:7.2f}s compute | peak RSS {d_rss:6.2f} GB")
        except Exception as e:
            d_rss = None
            print(f"scanpy + Dask OOC: FAILED ({e})")
    else:
        d_rss = None
        print(f"scanpy + Dask OOC: skipped (no {PY}; create .venv-dask with anndata>=0.11 + dask)")

    r_t, r_rss = singlerust_ooc(path, chunk)
    print(f"SingleRust OOC   : {r_t:7.2f}s wall    | peak RSS {r_rss:6.2f} GB")

    print()
    if s_rss and r_rss:
        print(f"memory: SingleRust OOC uses {s_rss / r_rss:.1f}× less peak RAM than scanpy in-memory")
    if d_rss and r_rss:
        print(f"memory: SingleRust OOC uses {d_rss / r_rss:.1f}× less peak RAM than scanpy+Dask "
              f"(Dask's sparse path barely beats in-memory)")


if __name__ == "__main__":
    main()

"""singlerust — a near-drop-in `sc.pp.*` replacement backed by SingleRust's out-of-core engine.

The goal: take an existing scanpy preprocessing script and change as little as possible to run
the heavy steps in Rust, out-of-core, on a disk-backed `.h5ad`.

    import scanpy as sc                       import singlerust as sr
    adata = sc.read_h5ad(path, backed="r")    adata = sc.read_h5ad(path, backed="r")
    sc.pp.calculate_qc_metrics(adata, ...)    sr.pp.calculate_qc_metrics(adata)
    sc.pp.normalize_total(adata, 1e4)         sr.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)                        sr.pp.log1p(adata)

Each `sr.pp.*` call operates on the file the AnnData is backed by (or a path you pass directly),
streaming in bounded memory, and edits it **in place** — matching scanpy's inplace default. The
expression matrix is never fully loaded.

Accepted `adata` argument forms:
  * a path / `pathlib.Path` to a `.h5ad`
  * a backed AnnData (``sc.read_h5ad(path, backed="r"|"r+")``) — its ``.filename`` is used
An in-memory AnnData is rejected with a clear message (use scanpy directly, or write to disk).

This shells out to the ``sr_ooc`` example binary; build it once with
``cargo build --release --features enrichment --example sr_ooc``.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

# Locate the compiled CLI (override with SR_OOC_BIN).
_ROOT = pathlib.Path(__file__).resolve().parent.parent
_BIN = pathlib.Path(os.environ.get("SR_OOC_BIN", _ROOT / "target" / "release" / "examples" / "sr_ooc"))


def _resolve_path(adata) -> pathlib.Path:
    if isinstance(adata, (str, os.PathLike)):
        return pathlib.Path(adata)
    # Duck-type a backed AnnData: it exposes `.filename` (truthy when backed). The Rust process
    # opens the file read-write, so we must release Python's handle first (HDF5 file locking),
    # otherwise the open collides. The backed object is stale afterwards — re-read to see results.
    fn = getattr(adata, "filename", None)
    if fn:
        path = pathlib.Path(fn)
        f = getattr(adata, "file", None)
        if f is not None and hasattr(f, "close"):
            try:
                f.close()
            except Exception:
                pass
        return path
    raise TypeError(
        "singlerust operates on a disk-backed .h5ad. Pass a path, or open with "
        "sc.read_h5ad(path, backed='r+'). (Got an in-memory AnnData.)"
    )


def _run(args: list[str]) -> None:
    if not _BIN.exists():
        raise FileNotFoundError(
            f"sr_ooc binary not found at {_BIN}. Build it with:\n"
            "  cargo build --release --features enrichment --example sr_ooc"
        )
    env = dict(os.environ)
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # avoid stale-lock false positives
    proc = subprocess.run([str(_BIN), *args], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        raise RuntimeError(f"sr_ooc {' '.join(args)} failed (exit {proc.returncode})")


class _PP:
    """Mirror of the subset of ``scanpy.pp`` that SingleRust implements out-of-core."""

    def calculate_qc_metrics(self, adata, *, chunk=None, **_ignored) -> None:
        """≈ ``sc.pp.calculate_qc_metrics(adata, inplace=True)`` — writes obs/var in place.

        Mitochondrial genes are auto-detected by ``MT-``/``mt-`` prefix; top-N segments are
        [50,100,200,500]. Extra scanpy kwargs are accepted and ignored for compatibility.
        """
        path = _resolve_path(adata)
        _run(["qc", str(path), *(["--chunk", str(chunk)] if chunk else [])])

    def normalize_total(self, adata, *, target_sum=1e4, chunk=None, **_ignored) -> None:
        """≈ ``sc.pp.normalize_total(adata, target_sum=...)`` — rewrites X in place."""
        path = _resolve_path(adata)
        _run(["normalize_total", str(path), "--target-sum", repr(float(target_sum)),
              *(["--chunk", str(chunk)] if chunk else [])])

    def log1p(self, adata, *, chunk=None, **_ignored) -> None:
        """≈ ``sc.pp.log1p(adata)`` — rewrites X in place."""
        path = _resolve_path(adata)
        _run(["log1p", str(path), *(["--chunk", str(chunk)] if chunk else [])])

    def normalize_total_log1p(self, adata, *, target_sum=1e4, chunk=None, **_ignored) -> None:
        """Fused normalize_total + log1p in a single streaming pass (one rewrite of X)."""
        path = _resolve_path(adata)
        _run(["normalize_total", str(path), "--target-sum", repr(float(target_sum)), "--log1p",
              *(["--chunk", str(chunk)] if chunk else [])])

    def preprocess(self, adata, *, target_sum=1e4, log1p=True, chunk=None, **_ignored) -> None:
        """Fused QC + normalize_total + log1p in one out-of-core job (≈ sc.pp QC then
        normalize_total then log1p). Writes obs/var QC metrics + the normalized X in place."""
        path = _resolve_path(adata)
        _run(["preprocess", str(path), "--target-sum", repr(float(target_sum)),
              *([] if log1p else ["--no-log1p"]),
              *(["--chunk", str(chunk)] if chunk else [])])


pp = _PP()

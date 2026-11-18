"""Sequential lanes, repeated Rust hashes, complete independent output checks."""

import hashlib
import subprocess
import sys
from pathlib import Path

import h5py


def digest(path):
    hash_ = hashlib.sha256()
    with h5py.File(path) as f:
        for k in ("X", "layers/psbulk_props"):
            for field in ("indices", "indptr", "data"):
                a = f[k][field]
                for i in range(0, len(a), 500000):
                    hash_.update(a[i : i + 500000].tobytes())
        for field in ("psbulk_cells", "psbulk_counts"):
            hash_.update(f["obs"][field][:].tobytes())
    return hash_.hexdigest()


def main():
    path, work = sys.argv[1:3]
    Path(work).mkdir(parents=True, exist_ok=True)
    for n in map(int, sys.argv[3:]):
        hashes = []
        for mode, t in [
            ("rust", 1),
            ("rust", 4),
            ("rust", 4),
            ("scanpy", 4),
            ("decoupler", 4),
        ]:
            dest = Path(work) / f"{mode}-{n}.h5ad"
            args = ["run_pilot.py", mode, path, str(dest), str(n), str(t)]
            result = subprocess.run([sys.executable, "measure.py", *args], check=False)
            if result.returncode == 124:
                print("FAILED", args, result.returncode, flush=True)
                continue
            result.check_returncode()
            if mode == "rust":
                hashes.append(digest(dest))
                print(hashes[-1], flush=True)
                assert len(set(hashes)) == 1
                if len(hashes) > 1:
                    continue
            args[1] = "check"
            subprocess.run([sys.executable, *args], check=True)


if __name__ == "__main__":
    main()

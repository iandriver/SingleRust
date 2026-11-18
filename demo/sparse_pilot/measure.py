"""Time a threaded (single-process) lane through output close on macOS."""

import json
import os
import resource
import runpy
import sys
import threading
import time

start = time.monotonic()
sys.argv = sys.argv[1:]


def report(status):
    print(
        json.dumps(
            {
                "mode": sys.argv[1],
                "groups": sys.argv[-2],
                "threads": sys.argv[-1],
                "seconds": round(time.monotonic() - start, 3),
                "peak_MB": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1
                ),
                "status": status,
            }
        ),
        flush=True,
    )


def guard():
    limit = float(os.environ.get("BENCH_TIMEOUT", "600"))
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 6e9:
            report("memory limit")
            os._exit(124)
        if time.monotonic() - start > limit:
            report("time limit")
            os._exit(124)
        time.sleep(0.1)


threading.Thread(target=guard, daemon=True).start()
runpy.run_path(sys.argv[0], run_name="__main__")
report("complete")

"""Aligned CPU latency rerun for the Sprint 4 manuscript.

Uses the same prepared models, rows, exact batch sweep, randomized blocks, and
fresh/reused TableProfile protocol for exact inference and StratQ.  Frozen
live_latency.json is retained unchanged; this separate record supersedes the
cross-study CPU timing comparison in the manuscript.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
import live_latency


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with threadpool_limits(limits=1):
        datasets = [live_latency.benchmark(name) for name in ("sms", "fashion")]
    frozen = json.loads((ROOT / "results/live_latency.json").read_text(encoding="utf-8"))
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "duckdb": duckdb.__version__,
        "numpy": np.__version__,
        "arrow": pa.__version__,
        "threadpools": threadpool_info(),
        "protocol": "same live_latency.benchmark; 1 warm-up + 10 measured randomized blocks",
        "frozen_live_latency_sha256": sha(ROOT / "results/live_latency.json"),
        "current_code_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in (
            ROOT / "src/stratq/bounds.py", ROOT / "src/stratq/engine.py")},
        "datasets": datasets,
    }
    target = ROOT / "results/sprint4_cpu_latency_aligned.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for dataset in datasets:
        print(dataset["dataset"])
        for row in dataset["summary"]:
            print(row["mode"], row["batch"], "calls", row["median_calls"],
                  "wall_ms", round(row["median_wall_ms"], 2),
                  "query_ms", round(row["median_query_ms"], 2))
    print(target)


if __name__ == "__main__":
    main()

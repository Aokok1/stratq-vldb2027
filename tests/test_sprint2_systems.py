"""Unit tests for Sprint 2 systems evaluation helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import sprint2_systems as s2


def test_zipf_sizes_preserve_total_and_skew():
    sizes = s2.zipf_sizes(100000, 20, 1.2)
    assert sizes.sum() == 100000
    assert sizes[0] > sizes[-1]
    equal = s2.zipf_sizes(100000, 10, 0.0)
    assert np.ptp(equal) <= 1


def test_proxy_auc_targets_and_breakeven():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, 2000)
    for target in (0.95, 0.75, 0.50):
        score, auc = s2.proxy_with_target_auc(labels, target)
        assert abs(auc - target) < 0.03
    assert s2.breakeven_q([100, 200, 300], [150, 180, 210]) == 2
    assert s2.breakeven_q([100, 200], [150, 250]) is None


def test_memory_stage_schema_and_coverage_summary(tmp_path):
    path = tmp_path / "tiny.parquet"
    pq.write_table(pa.table({"grp": ["A", "A", "B", "B"], "label": [0, 1, 1, 1]}), path)
    record = s2.memory_stage("verify", 4, data_path=path)
    assert record["stage"] == "verify"
    assert record["tracemalloc_peak_bytes"] > 0
    assert record["peak_working_set_bytes"] is None or record["peak_working_set_bytes"] > 0

    class Result:
        groups = ["A", "B"]
        lower = {"A": 0, "B": 10}
        upper = {"A": 2, "B": 12}

    summary = s2.coverage_summary(Result(), {"A": 1, "B": 20}, {"A": 10, "B": 100})
    assert summary["all_covered"] is False
    assert summary["tail_covered"] == 1


def test_smoke_systems_pipeline(tmp_path):
    # The smoke functions are exercised through their public helpers; full
    # experiments are run by the script and write structured JSON.
    report = s2.run_profile_amortization(smoke=True)
    assert "records" in report

"""Lightweight tests for GPU workload harness helpers (no model download)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import sprint2_gpu_workloads as gpu


def test_gpu_reference_accounting_and_no_repeats():
    ref = gpu.GPUReference(lambda ids: np.ones(len(ids), dtype=np.uint8), 10)
    out = ref(np.array([0, 1, 2]))
    assert out.tolist() == [1, 1, 1]
    assert ref.calls == 3 and ref.batches == 1
    with pytest.raises(AssertionError):
        ref(np.array([2, 3]))
    ref.reset(np.ones(10, dtype=np.uint8))
    assert ref.calls == 0


def test_summarize_count_runs():
    workload = {"count_runs": [
        {"method": "A", "calls": 10, "all_covered": True, "mean_normalized_half_width": 0.1, "wall_ms": 5},
        {"method": "A", "calls": 20, "all_covered": False, "mean_normalized_half_width": 0.2, "wall_ms": 7},
    ]}
    summary = gpu.summarize(workload)
    assert summary["A"]["mean_calls"] == 15
    assert summary["A"]["all_covered_trials"] == 1

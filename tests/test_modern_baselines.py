"""Regression tests for literature-derived modern operator adaptations."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import stratq.modern_baselines as mb


def small_population(seed=7, n=24, groups=3):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n).astype(float)
    proxy = rng.uniform(size=n)
    indices = [np.arange(g, n, groups) for g in range(groups)]
    truth = {f"G{g}": int(labels[idx].sum()) for g, idx in enumerate(indices)}
    return indices, proxy, labels, truth


def test_celeus_uniform_signal_is_conditionally_unbiased():
    n = 4
    labels = np.array([1.0, 0.0, 1.0, 1.0])
    proxy = np.array([0.2, 0.7, 0.4, 0.9])
    mu = labels.mean()
    for t in range(1, n + 1):
        signals = []
        for perm in itertools.permutations(range(n)):
            seen = perm[:t - 1]
            row = perm[t - 1]
            seen_sum = labels[list(seen)].sum()
            rem_proxy = proxy[list(perm[t - 1:])].sum()
            q = 1.0 / (n - t + 1)
            signals.append(mb.celeus_signal(seen_sum, rem_proxy, labels[row], proxy[row], n, q))
        assert np.mean(signals) == pytest.approx(mu, abs=1e-12)


def test_all_modern_count_arms_census_exact_and_no_repeated_rows():
    indices, proxy, labels, truth = small_population()
    seen = []

    def oracle(rows):
        seen.extend(rows.tolist())
        return labels[rows]

    eps = {g: 0 for g in truth}
    factories = [
        lambda: mb.spertus_wor_count(indices, proxy, oracle, eps, pilot_per_group=2, batch_size=3, seed=3),
        lambda: mb.calibrated_fixed_proxy_count(indices, proxy, oracle, eps, pilot_per_group=2, batch_size=3, seed=3),
        lambda: mb.celeus_fixed_proxy_count(indices, proxy, oracle, eps, pilot_per_group=2, batch_size=3, seed=3, grid_size=101),
        lambda: mb.celeus_fixed_proxy_count(indices, proxy, oracle, eps, pilot_per_group=2, batch_size=3, seed=3, nonuniform=True, grid_size=101),
        lambda: mb.pilotdb_asymp_count(indices, oracle, eps, pilot_per_group=2, seed=3),
    ]
    for factory in factories:
        seen.clear()
        result = factory()
        assert len(seen) == len(set(seen)) == len(labels)
        assert result.total_oracle_calls == len(labels)
        assert result.lower == truth and result.upper == truth


def test_calibrated_predictor_is_fitted_once_per_group_and_frozen():
    indices, proxy, labels, truth = small_population(n=60)
    with patch.object(mb, "fit_fixed_linear_predictor", wraps=mb.fit_fixed_linear_predictor) as fit:
        result = mb.calibrated_fixed_proxy_count(
            indices, proxy, labels, {g: 0.05 * len(idx) for g, idx in zip(truth, indices)},
            pilot_per_group=4, batch_size=4, seed=11,
        )
    assert fit.call_count == len(indices)
    assert result.metadata["fixed_predictor"] is True
    assert result.metadata["alpha_g"] == pytest.approx(0.1 / len(indices))


def test_celeus_small_population_anytime_coverage():
    rng = np.random.default_rng(91)
    covered = 0
    trials = 80
    for trial in range(trials):
        n = int(rng.integers(6, 15))
        labels = rng.integers(0, 2, n).astype(float)
        proxy = rng.uniform(size=n)
        result = mb.celeus_fixed_proxy_count(
            [np.arange(n)], proxy, labels, {"G0": 0.25 * n}, delta=0.1,
            pilot_per_group=2, batch_size=2, seed=trial, grid_size=201,
            group_names=["G0"],
        )
        covered += int(result.lower["G0"] - 1e-9 <= labels.sum() <= result.upper["G0"] + 1e-9)
    assert covered / trials >= 0.90


def test_bargain_global_and_pilotdb_diagnostic_interfaces():
    indices, proxy, labels, truth = small_population(n=90)
    global_result = mb.bargain_global_count(
        indices, proxy, labels, eps_global=0.05 * len(labels), pilot=8, batch_size=8, seed=5,
        group_names=list(truth),
    )
    assert global_result.metadata["global_target_only"] is True
    assert set(global_result.group_estimates) == set(truth)
    pilot = mb.pilotdb_asymp_count(indices, labels, {g: 0.1 * len(idx) for g, idx in zip(truth, indices)},
                                   pilot_per_group=4, optional_stopping=True, seed=5,
                                   group_names=list(truth))
    assert pilot.metadata["optional_stopping"] is True
    assert pilot.metadata["clt"] is True


def test_decision_adapter_and_invalid_inputs():
    indices, proxy, labels, truth = small_population()
    result = mb.spertus_wor_count(indices, proxy, labels, {g: 0 for g in truth},
                                  pilot_per_group=2, batch_size=3, seed=3)
    having = mb.decisions_from_count_bounds(result, truth, tau=2)
    top = mb.decisions_from_count_bounds(result, truth, k=1)
    assert having["decision_correct"] and top["decision_correct"]
    with pytest.raises(ValueError):
        mb.decisions_from_count_bounds(result, truth)
    with pytest.raises(ValueError):
        mb.spertus_wor_count([np.array([], dtype=int)], proxy, labels, 0.1)

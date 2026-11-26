"""Safe-LOO gating, predictability, and outward-binary64 verification.

The Decimal replay treats every stored binary64 input as an exact real, uses
50 significant digits, and recomputes the complete betting recursion.  It is
an independent reference for the directed-rounding implementation.
"""
from __future__ import annotations

import math
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stratq.bounds import (
    WoRControlVariateCS,
    _down,
    _log_threshold_enclosure,
    _up,
    nonnegative_sum_interval,
    psi_e_enclosure,
    ridge_loo_residuals,
    select_optimal_shrinkage_theta,
)
from stratq.engine import StratQEngine


def dec(value) -> Decimal:
    return Decimal.from_float(float(value))


def dec_dot(theta, phi):
    return sum((dec(a) * dec(b) for a, b in zip(theta, phi)), Decimal(0))


def decimal_range(theta, features):
    values = [dec_dot(theta, row) for row in features]
    return min(values), max(values)


def decimal_psi(lam: Decimal) -> Decimal:
    return -(Decimal(1) - lam).ln() - lam


class DecimalCS:
    """Fifty-digit real-arithmetic replay with the same collision policy."""

    def __init__(self, n_total, alpha, phi_mean, y_min, y_max, theta, f_bounds,
                 phi_total=None):
        self.n_total = n_total
        self.alpha = dec(alpha)
        self.y_min, self.y_max = dec(y_min), dec(y_max)
        self.y_span = self.y_max - self.y_min
        self.phi_total = ([dec(v) for v in phi_total] if phi_total is not None
                          else [dec(v) * n_total for v in phi_mean])
        self.phi_seen = [Decimal(0)] * len(phi_mean)
        self.theta = [dec(v) for v in theta]
        self.f_min, self.f_max = f_bounds
        self.n_seen = 0
        self.y_sum = Decimal(0)
        self.resid_sum = Decimal(0)
        self.A = self.B = self.V = Decimal(0)
        self.lo, self.hi = Decimal(0), Decimal(1)
        self.restarts = 0
        self.last_restart_outside_coverage = False
        self.mu = None

    def set_truth(self, labels):
        normalized = [min(Decimal(1), max(Decimal(0), (dec(v) - self.y_min) / self.y_span))
                      for v in labels]
        self.mu = sum(normalized, Decimal(0)) / self.n_total

    def step(self, raw_y, phi, lam):
        y = min(Decimal(1), max(Decimal(0), (dec(raw_y) - self.y_min) / self.y_span))
        rem = self.n_total - self.n_seen
        phi_rem = [(a - b) / rem for a, b in zip(self.phi_total, self.phi_seen)]
        w = Decimal(self.n_total) / rem
        h = self.y_sum / rem
        f_rem = dec_dot(self.theta, phi_rem)
        f_i = dec_dot(self.theta, phi)
        a = f_rem - self.f_max
        Z = f_rem + y - f_i
        R = Decimal(1) + self.f_max - self.f_min
        tz = min(Decimal(1), max(Decimal(0), (Z - a) / R))
        if any(v != 0 for v in self.theta[1:]):
            avg_resid = self.resid_sum / self.n_seen if self.n_seen else Decimal(0)
            center = min(Decimal(1), max(Decimal(0), f_rem + Decimal("0.5") * avg_resid))
        else:
            center = (self.y_sum + Decimal("0.5")) / (self.n_seen + 1)
        tz_pred = min(Decimal(1), max(Decimal(0), (center - a) / R))
        lam = dec(lam)
        self.A += lam * w / R
        self.B += lam * (Z + h) / R
        self.V += decimal_psi(lam) * (tz - tz_pred) ** 2
        self.resid_sum += y - f_i
        self.n_seen += 1
        self.y_sum += y
        self.phi_seen = [a + dec(b) for a, b in zip(self.phi_seen, phi)]

        det_lo = self.y_sum / self.n_total
        det_hi = (self.y_sum + self.n_total - self.n_seen) / self.n_total
        self.last_restart_outside_coverage = False
        if self.n_seen == self.n_total:
            self.lo, self.hi = det_lo, det_hi
        elif det_lo > self.hi or det_hi < self.lo:
            previous = (self.lo, self.hi)
            self.restarts += 1
            self.last_restart_outside_coverage = not (
                previous[0] <= self.mu <= previous[1]
            )
            self.lo, self.hi = det_lo, det_hi
        else:
            penalty = (Decimal(2) / self.alpha).ln() + self.V
            lo_eb = (self.B - penalty) / self.A
            hi_eb = (self.B + penalty) / self.A
            new_lo, new_hi = max(self.lo, lo_eb, det_lo), min(self.hi, hi_eb, det_hi)
            if new_lo <= new_hi:
                self.lo, self.hi = new_lo, new_hi
            else:
                self.lo = min(self.lo, lo_eb, det_lo)
                self.hi = max(self.hi, hi_eb, det_hi)

    @property
    def bounds(self):
        # Preserve exact affine endpoints; otherwise a 50-digit Decimal add
        # can round an exact binary64 boundary to the wrong side of it.
        lo = self.y_min if self.lo == 0 else self.y_min + self.lo * self.y_span
        hi = self.y_max if self.hi == 1 else self.y_min + self.hi * self.y_span
        return lo, hi


def test_psi_enclosure_against_decimal():
    with localcontext() as ctx:
        ctx.prec = 50
        for lam in np.linspace(0.0001, 0.5, 101):
            low, high = psi_e_enclosure(float(lam))
            truth = decimal_psi(dec(lam))
            assert dec(low) <= truth <= dec(high)


def test_log_threshold_enclosure_against_high_precision_decimal():
    rng = np.random.default_rng(20261126)
    alphas = [0.1 / g for g in range(1, 101)]
    alphas += [float(v) for v in rng.uniform(1e-6, 0.999, 500)]
    with localcontext() as ctx:
        ctx.prec = 140
        for alpha in alphas:
            low, high = _log_threshold_enclosure(alpha)
            truth = (Decimal(2) / dec(alpha)).ln()
            assert dec(low) <= truth <= dec(high), alpha


def test_nonbinary_normalized_labels_remain_enclosed():
    # The exact real-arithmetic CS may legitimately exclude the truth on its
    # failure event.  The safe implementation must enclose that Decimal CS,
    # not necessarily the truth on every individual path.
    n = 66
    phi = np.column_stack([np.ones(n), np.linspace(0.0, 1.0, n)])
    theta = np.array([0.2, 0.6])
    f_bounds = decimal_range(theta, phi)
    exact_bounds = (_down(float(f_bounds[0])), _up(float(f_bounds[1])))
    exact_total = [sum((dec(v) for v in phi[:, j]), Decimal(0)) for j in range(2)]
    cs = WoRControlVariateCS(
        n, 0.1, phi.mean(axis=0),
        phi_min_g=phi.min(axis=0), phi_max_g=phi.max(axis=0),
        y_min=0.0, y_max=0.3, safe_rounding=True,
    )
    with localcontext() as ctx:
        ctx.prec = 60
        reference = DecimalCS(n, 0.1, phi.mean(axis=0), 0.0, 0.3, theta,
                              f_bounds, phi_total=exact_total)
        reference.set_truth(np.full(n, 0.2))
        for t in range(1, n + 1):
            cs.update_batch(np.array([0.2]), phi[t - 1:t], theta,
                            exact_f_bounds=exact_bounds)
            reference.step(0.2, phi[t - 1], cs.lambda_history[-1])
            safe_lo, safe_hi = cs.bounds
            ref_lo, ref_hi = reference.bounds
            assert dec(safe_lo) <= ref_lo, (t, safe_lo, ref_lo)
            assert dec(safe_hi) >= ref_hi, (t, safe_hi, ref_hi)
        # Census is exact and must contain the true non-binary mean.
        assert dec(cs.bounds[0]) <= dec(0.2) <= dec(cs.bounds[1])


def test_random_nonbinary_paths_contain_decimal_intervals():
    rng = np.random.default_rng(20261127)
    checked_steps = 0
    with localcontext() as ctx:
        ctx.prec = 60
        for path in range(300):
            n = int(rng.integers(2, 13))
            d = int(rng.integers(1, 5))
            features = rng.uniform(0.0, 1.0, (n, d))
            features[:, 0] = 1.0
            labels = rng.uniform(-0.7, 1.9, n)
            theta = rng.uniform(-0.8, 0.8, d)
            theta[0] = float(rng.uniform(-0.2, 0.8))
            phi_mean = features.mean(axis=0)
            sum_lo, sum_hi = nonnegative_sum_interval(features, axis=0)
            exact_total = [sum((dec(v) for v in features[:, j]), Decimal(0))
                           for j in range(d)]
            y_min, y_max = -0.7, 1.9
            f_bounds = decimal_range(theta, features)
            exact_bounds = (_down(float(f_bounds[0])), _up(float(f_bounds[1])))
            cs = WoRControlVariateCS(n, 0.1, phi_mean, y_min=y_min, y_max=y_max,
                                     safe_rounding=True, phi_sum_interval=(sum_lo, sum_hi))
            reference = DecimalCS(n, 0.1, phi_mean, y_min, y_max, theta, f_bounds,
                                  phi_total=exact_total)
            reference.set_truth(labels)
            for step, row in enumerate(rng.permutation(n)):
                cs.update_batch(labels[row:row + 1], features[row:row + 1], theta,
                                exact_f_bounds=exact_bounds)
                reference.step(labels[row], features[row], cs.lambda_history[-1])
                safe_lo, safe_hi = cs.bounds
                ref_lo, ref_hi = reference.bounds
                assert dec(safe_lo) <= ref_lo, (path, step, "lower", safe_lo, ref_lo)
                assert dec(safe_hi) >= ref_hi, (path, step, "upper", safe_hi, ref_hi)
                checked_steps += 1
    assert checked_steps > 1500


def test_loo_matches_explicit_refits():
    rng = np.random.default_rng(81)
    for n in (8, 17, 40):
        X = np.column_stack([np.ones(n), rng.uniform(-1, 1, (n, 3))])
        y = rng.normal(size=n)
        for kappa in (0.0, 0.1, 0.25, 0.5, 1.0):
            loo, _, _ = ridge_loo_residuals(X, y, kappa, 2.3)
            brute = []
            for i in range(n):
                mask = np.arange(n) != i
                Xc = X[mask, 1:] - X[mask, 1:].mean(axis=0)
                yc = y[mask] - y[mask].mean()
                beta = np.linalg.solve(kappa ** 2 * Xc.T @ Xc + 2.3 * np.eye(3),
                                       kappa * Xc.T @ yc)
                prediction = y[mask].mean() + kappa * (X[i, 1:] - X[mask, 1:].mean(axis=0)) @ beta
                brute.append(y[i] - prediction)
            np.testing.assert_allclose(loo, brute, rtol=2e-12, atol=2e-12)


def test_safe_gate_small_samples_and_weak_proxy_degenerate():
    rng = np.random.default_rng(12)
    X = np.column_stack([np.ones(200), rng.uniform(size=200)])
    y = 0.8 * X[:, 1] + rng.normal(0, 0.01, 200)
    theta, fb, diag = select_optimal_shrinkage_theta(
        X[:11], y[:11], np.array([0.0, 1.0]), 0.05,
        phi_min_g=np.array([1.0, 0.0]), phi_max_g=np.array([1.0, 1.0]),
        return_diagnostics=True,
    )
    assert diag["selected_kappa"] == 0.0
    assert diag["gate_reason"] == "insufficient_group_size"
    assert theta[1] == 0.0 and fb == (theta[0], theta[0])

    # Independently randomized labels carry no predictable proxy signal.
    y_noise = rng.uniform(size=200)
    _, fb, diag = select_optimal_shrinkage_theta(
        X, y_noise, np.array([0.0, 1.0]), 0.05,
        phi_min_g=np.array([1.0, 0.0]), phi_max_g=np.array([1.0, 1.0]),
        return_diagnostics=True,
    )
    assert diag["selected_kappa"] == 0.0
    assert fb[0] == fb[1]
    assert diag["selected_cost"] == pytest.approx(diag["baseline_cost"])


def test_safe_gate_selects_strong_proxy_and_is_predictable():
    rng = np.random.default_rng(19)
    X = np.column_stack([np.ones(160), rng.uniform(size=160)])
    y = np.clip(0.5 * X[:, 1] + rng.normal(0, 0.15, 160), 0, 1)
    kwargs = dict(alpha=0.05, target_eps=0.01,
                  phi_min_g=np.array([1.0, 0.0]),
                  phi_max_g=np.array([1.0, 1.0]), return_diagnostics=True)
    theta, fb, diag = select_optimal_shrinkage_theta(
        X[:120], y[:120], np.array([0.0, 0.3]), **kwargs
    )
    assert diag["selected_kappa"] > 0.0
    assert diag["selected_cost"] <= 0.95 * diag["baseline_cost"]
    # Changing unrevealed labels cannot change the selected predictable theta.
    changed = y.copy()
    changed[120:] = 1.0 - changed[120:]
    theta2, fb2, _ = select_optimal_shrinkage_theta(X[:120], changed[:120],
                                                   np.array([0.0, 0.3]), **kwargs)
    np.testing.assert_array_equal(theta, theta2)
    assert fb == fb2


def test_engine_batch_predictability_and_decision_eps():
    rng = np.random.default_rng(4)
    n = 80
    table = pa.table({"grp": ["A"] * n, "proxy": rng.uniform(size=n),
                      "oracle": rng.integers(0, 2, n)})
    engine = StratQEngine(table, group_col="grp", proxy_col="proxy", oracle_col="oracle")
    calls = []
    original = engine._get_predictable_theta

    def record(gi, phi_history, y_history, *args, **kwargs):
        calls.append((sum(map(len, phi_history[gi])), sum(map(len, y_history[gi]))))
        return original(gi, phi_history, y_history, *args, **kwargs)

    engine._get_predictable_theta = record
    engine.execute_grouped_agg(agg_type="count", eps=0, batch_size=4,
                               pilot_per_group=4, seed=3)
    assert calls == sorted(calls)
    assert all(phi_n == y_n for phi_n, y_n in calls)
    # Every fit precedes its batch: history lengths never include that batch.
    assert calls[0] == (0, 0)

    cs = WoRControlVariateCS(100, 0.05, np.array([1.0, 0.5]))
    cs.lo_norm, cs.hi_norm = 0.4, 0.6
    eps = engine._predictable_decision_eps(cs, 0.5, 0.05)
    assert 0.0 < eps <= 0.05
    assert eps == pytest.approx(0.025)  # interval contains boundary: width/4
    assert engine._predictable_decision_eps(cs, 0.5, 0.01) == pytest.approx(0.01)
    cs.lo_norm, cs.hi_norm = 0.7, 0.8
    assert engine._predictable_decision_eps(cs, 0.5, 0.01) == pytest.approx(0.01)


def test_1200_random_paths_contain_decimal_intervals():
    rng = np.random.default_rng(20261125)
    checked_steps = 0
    with localcontext() as ctx:
        ctx.prec = 50
        for path in range(1200):
            n = int(rng.integers(2, 10))
            d = int(rng.integers(1, 5))
            features = rng.uniform(0.0, 1.0, (n, d))
            features[:, 0] = 1.0
            labels = rng.integers(0, 2, n).astype(float)
            theta = rng.uniform(-0.8, 0.8, d)
            theta[0] = float(rng.uniform(-0.2, 0.8))
            phi_mean = features.mean(axis=0)
            sum_lo, sum_hi = nonnegative_sum_interval(features, axis=0)
            exact_total = [sum((dec(v) for v in features[:, j]), Decimal(0))
                           for j in range(d)]
            y_min, y_max = (-0.3, 1.7) if path % 3 == 0 else (0.0, 1.0)
            raw_labels = labels if path % 3 else labels + y_min
            f_bounds = decimal_range(theta, features)
            exact_bounds = (_down(float(f_bounds[0])), _up(float(f_bounds[1])))
            cs = WoRControlVariateCS(n, 0.1, phi_mean, y_min=y_min, y_max=y_max,
                                     safe_rounding=True, phi_sum_interval=(sum_lo, sum_hi))
            reference = DecimalCS(n, 0.1, phi_mean, y_min, y_max, theta, f_bounds,
                                  phi_total=exact_total)
            reference.set_truth(raw_labels)
            order = rng.permutation(n)
            for step, row in enumerate(order):
                cs.update_batch(np.array([raw_labels[row]]), features[row:row + 1],
                                theta, exact_f_bounds=exact_bounds)
                reference.step(raw_labels[row], features[row], cs.lambda_history[-1])
                safe_lo, safe_hi = cs.bounds
                ref_lo, ref_hi = reference.bounds
                assert dec(safe_lo) <= ref_lo, (path, step, "lower", safe_lo, ref_lo)
                assert dec(safe_hi) >= ref_hi, (path, step, "upper", safe_hi, ref_hi)
                assert cs.A_t <= reference.A
                checked_steps += 1
    assert checked_steps > 5000


def test_small_population_coverage_and_restart_event():
    rng = np.random.default_rng(771)
    covered_paths = 0
    paths = 0
    restart_checks = 0
    with localcontext() as ctx:
        ctx.prec = 50
        for n in range(2, 10):
            for _ in range(80):
                labels = rng.integers(0, 2, n).astype(float)
                features = rng.uniform(size=(n, 2))
                features[:, 0] = 1.0
                theta = np.array([0.2, 0.6])
                f_bounds = decimal_range(theta, features)
                exact_bounds = (_down(float(f_bounds[0])), _up(float(f_bounds[1])))
                sum_lo, sum_hi = nonnegative_sum_interval(features, axis=0)
                exact_total = [sum((dec(v) for v in features[:, j]), Decimal(0))
                               for j in range(features.shape[1])]
                cs = WoRControlVariateCS(n, 0.1, features.mean(axis=0), safe_rounding=True,
                                         phi_sum_interval=(sum_lo, sum_hi))
                ref = DecimalCS(n, 0.1, features.mean(axis=0), 0.0, 1.0, theta, f_bounds,
                                phi_total=exact_total)
                ref.set_truth(labels)
                truth = labels.sum() / n
                anytime_covered = True
                old_restarts = 0
                for row in rng.permutation(n):
                    cs.update_batch(labels[row:row + 1], features[row:row + 1], theta,
                                    exact_f_bounds=exact_bounds)
                    ref.step(labels[row], features[row], cs.lambda_history[-1])
                    lo, hi = cs.bounds
                    anytime_covered &= lo - 1e-14 <= truth <= hi + 1e-14
                    if cs.empty_intersection_restarts > old_restarts:
                        # The independent exact replay proves disjointness of
                        # the previous reported and deterministic intervals.
                        assert ref.last_restart_outside_coverage
                        restart_checks += 1
                        old_restarts = cs.empty_intersection_restarts
                covered_paths += int(anytime_covered)
                paths += 1
    # Random paths verify coverage; the explicit conflict below guarantees
    # that the conservative restart branch itself is exercised.
    assert covered_paths / paths >= 0.90

    cs = WoRControlVariateCS(4, 0.1, np.ones(1), safe_rounding=True)
    cs.hi_norm = 0.1
    cs.update_batch(np.ones(1), np.ones((1, 1)), np.zeros(1))
    assert cs.empty_intersection_restarts == 1
    assert cs.bounds == (0.25, 1.0)


def test_feature_total_enclosure_covers_exact_sum():
    values = np.array([0.6053276518340511, 0.7538466491583667, 0.7940177408735195,
                       0.418945405670905, 0.3220890502078766])
    with localcontext() as ctx:
        ctx.prec = 50
        truth = sum((dec(v) for v in values), Decimal(0))
        # Both the backward-compatible mean-only contract and the tight
        # profile sum interval must contain the exact feature total.
        cs = WoRControlVariateCS(len(values), 0.1, np.array([1.0, values.mean()]))
        assert dec(cs._phi_total_lo[1]) <= truth <= dec(cs._phi_total_hi[1])
        low, high = nonnegative_sum_interval(values)
        assert dec(low) <= truth <= dec(high)


def test_gate_effective_df_and_range_rejections():
    rng = np.random.default_rng(31)
    n, d = 12, 11
    X = np.column_stack([np.ones(n), rng.uniform(size=(n, d - 1))])
    y = rng.uniform(size=n)
    _, _, diag = select_optimal_shrinkage_theta(
        X, y, np.zeros(d), 0.05, ridge_lambda=1e-7,
        phi_min_g=np.r_[1.0, np.zeros(d - 1)],
        phi_max_g=np.ones(d), return_diagnostics=True,
    )
    assert diag["selected_kappa"] == 0.0
    assert diag["gate_reason"] == "insufficient_effective_df"

    X = np.column_stack([np.ones(60), rng.uniform(size=60)])
    y = rng.uniform(size=60)
    _, fb, diag = select_optimal_shrinkage_theta(
        X, y, np.array([0.0, 2.0]), 0.05,
        phi_min_g=np.array([1.0, 0.0]), phi_max_g=np.array([1.0, 1.0]),
        return_diagnostics=True,
    )
    assert diag["selected_kappa"] == 0.0
    assert diag["gate_reason"] == "range_constraint"
    assert fb[0] == fb[1]


def test_engine_future_label_invariance_and_decision_targets():
    rng = np.random.default_rng(4)
    n = 80
    proxy = rng.uniform(size=n)
    labels = rng.integers(0, 2, n)

    def prediction_after_pilot(table_labels):
        table = pa.table({"grp": ["A"] * n, "proxy": proxy, "oracle": table_labels})
        engine = StratQEngine(table, group_col="grp", proxy_col="proxy", oracle_col="oracle")
        original = engine._get_predictable_theta
        predictions = []

        def record(gi, phi_history, y_history, *args, **kwargs):
            theta, bounds = original(gi, phi_history, y_history, *args, **kwargs)
            if sum(map(len, phi_history[gi])) == 4:
                predictions.append(theta.copy())
            return theta, bounds

        engine._get_predictable_theta = record
        engine.execute_grouped_agg(agg_type="count", eps=0, batch_size=4,
                                   pilot_per_group=4, seed=3)
        assert len(predictions) == 1
        return predictions[0]

    baseline = prediction_after_pilot(labels)
    pilot_rows = np.random.default_rng(3).permutation(n)[:4]
    changed = labels.copy()
    future = np.setdiff1d(np.arange(n), pilot_rows)
    changed[future] = 1 - changed[future]
    np.testing.assert_array_equal(baseline, prediction_after_pilot(changed))

    small = pa.table({"grp": ["A"] * 20 + ["B"] * 20,
                      "proxy": rng.uniform(size=40), "oracle": rng.integers(0, 2, 40)})
    for query in ("having", "topk"):
        engine = StratQEngine(small, group_col="grp", proxy_col="proxy", oracle_col="oracle")
        original = engine._get_predictable_theta
        history, targets = [], []

        def record_decision(gi, phi_history, y_history, *args, **kwargs):
            history.append(sum(map(len, phi_history)))
            targets.append(kwargs.get("target_eps"))
            return original(gi, phi_history, y_history, *args, **kwargs)

        engine._get_predictable_theta = record_decision
        if query == "having":
            engine.execute_grouped_having(tau=10, agg_type="count", batch_size=2,
                                          pilot_per_group=2, seed=5)
        else:
            engine.execute_grouped_topk(k=1, agg_type="count", batch_size=2,
                                        pilot_per_group=2, seed=5)
        assert history == sorted(history)
        assert any(target is not None and target > 0 for target in targets)


def test_legacy_mode_remains_available():
    rng = np.random.default_rng(9)
    X = np.column_stack([np.ones(30), rng.uniform(size=30)])
    y = rng.uniform(size=30)
    safe = select_optimal_shrinkage_theta(X, y, np.array([0.0, 1.0]), 0.05,
                                           gating_mode="safe_loo")
    legacy = select_optimal_shrinkage_theta(X, y, np.array([0.0, 1.0]), 0.05,
                                             gating_mode="legacy")
    legacy_diag = select_optimal_shrinkage_theta(X, y, np.array([0.0, 1.0]), 0.05,
                                                  gating_mode="legacy", return_diagnostics=True)
    assert legacy_diag[2]["gating_mode"] == "legacy"
    assert safe[0].shape == legacy[0].shape == (2,)
    with pytest.raises(ValueError, match="gating_mode"):
        select_optimal_shrinkage_theta(X, y, np.array([0.0, 1.0]), 0.05,
                                       gating_mode="unknown")

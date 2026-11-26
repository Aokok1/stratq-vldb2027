"""Operator-level modern conference comparators for StratQ.

Every class here is a literature-derived *operator adaptation*, not a full
system replica.  The audit trail in ``BASELINE_AUDIT.md`` records the exact
source sections and the contract differences.  All random sampling uses the
revealed history only; fixed predictors are fitted once on a pilot prefix.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .bounds import (
    WoRControlVariateCS,
    linear_prediction_bounds,
    nonnegative_sum_interval,
)


@dataclass
class ModernCountResult:
    method: str
    groups: List[str]
    estimates: Dict[str, float]
    lower: Dict[str, float]
    upper: Dict[str, float]
    half_widths: Dict[str, float]
    oracle_calls_by_group: Dict[str, int]
    total_oracle_calls: int
    total_rows: int
    wall_ms: float
    oracle_ms: float
    optimizer_ms: float
    converged: bool
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class GlobalCountResult:
    method: str
    estimate: float
    lower: float
    upper: float
    total_oracle_calls: int
    total_rows: int
    wall_ms: float
    converged: bool
    group_estimates: Dict[str, float]
    group_lower: Dict[str, float]
    group_upper: Dict[str, float]
    metadata: Dict[str, object] = field(default_factory=dict)


def _oracle_eval(oracle: Union[Callable[[np.ndarray], np.ndarray], np.ndarray],
                 rows: np.ndarray) -> Tuple[np.ndarray, float]:
    start = time.perf_counter()
    values = np.asarray(oracle(rows) if callable(oracle) else oracle[rows], dtype=np.float64)
    if values.shape != rows.shape or not np.all((values == 0) | (values == 1)):
        raise ValueError("modern count comparators require one binary label per row")
    return values, (time.perf_counter() - start) * 1000.0


def _group_arrays(group_indices: Sequence[np.ndarray], group_names: Optional[Sequence[str]] = None):
    indices = [np.asarray(v, dtype=np.int64) for v in group_indices]
    if not indices or any(len(v) == 0 for v in indices):
        raise ValueError("every group must be nonempty")
    names = list(group_names) if group_names is not None else [f"G{i}" for i in range(len(indices))]
    if len(names) != len(indices):
        raise ValueError("group_names length mismatch")
    return indices, names


def fit_fixed_linear_predictor(phi_pilot: np.ndarray, y_pilot: np.ndarray,
                               ridge_lambda: float = 8.0) -> np.ndarray:
    """Fit one Ridge predictor on the pilot prefix and return a frozen theta."""
    X = np.asarray(phi_pilot, dtype=np.float64)
    y = np.asarray(y_pilot, dtype=np.float64)
    if X.ndim != 2 or len(y) != len(X):
        raise ValueError("pilot feature/label dimensions mismatch")
    d = X.shape[1]
    A = X.T @ X + ridge_lambda * np.eye(d)
    A[0, 0] -= 0.9 * ridge_lambda
    try:
        theta = np.linalg.solve(A, X.T @ y)
    except np.linalg.LinAlgError:
        theta = np.zeros(d)
        theta[0] = float(np.mean(y)) if len(y) else 0.0
    return WoRControlVariateCS.project_theta_span(theta, max_span=1.0)


def _run_cs_count(
    method: str,
    group_indices: Sequence[np.ndarray],
    proxy_scores: np.ndarray,
    oracle: Union[Callable[[np.ndarray], np.ndarray], np.ndarray],
    eps: Union[float, Dict[str, float]],
    delta: float,
    batch_size: int,
    pilot_per_group: int,
    seed: int,
    *,
    fixed_predictor: bool,
    max_oracle_budget: Optional[int] = None,
    group_names: Optional[Sequence[str]] = None,
) -> ModernCountResult:
    """Shared uniform-WoR/max-slack runner for raw-WoR and fixed-proxy arms."""
    start = time.perf_counter()
    indices, names = _group_arrays(group_indices, group_names)
    scores = np.asarray(proxy_scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    G = len(indices)
    alpha_g = delta / G
    sizes = np.array([len(v) for v in indices], dtype=np.int64)
    eps_norm = np.array([
        (eps[name] if isinstance(eps, dict) else float(eps)) / max(int(n), 1)
        for name, n in zip(names, sizes)
    ])
    perms = [rng.permutation(v) for v in indices]
    ptrs = np.zeros(G, dtype=np.int64)
    cs_list, theta_list, bounds_list = [], [], []
    phi_seen: List[List[np.ndarray]] = [[] for _ in range(G)]
    y_seen: List[List[np.ndarray]] = [[] for _ in range(G)]
    total_calls = 0
    oracle_ms = 0.0
    budget = max_oracle_budget if max_oracle_budget is not None else int(sizes.sum())

    def draw(gi: int, k_req: int) -> int:
        nonlocal total_calls, oracle_ms
        rem_budget = budget - total_calls
        rem_group = int(sizes[gi] - ptrs[gi])
        k = min(k_req, rem_group, rem_budget)
        if k <= 0:
            return 0
        rows = perms[gi][ptrs[gi]:ptrs[gi] + k]
        ptrs[gi] += k
        y, ms = _oracle_eval(oracle, rows)
        oracle_ms += ms
        total_calls += k
        phi = np.column_stack([np.ones(k), scores[rows]])
        cs_list[gi].update_batch(y, phi, theta_list[gi], exact_f_bounds=bounds_list[gi])
        phi_seen[gi].append(phi)
        y_seen[gi].append(y)
        return k

    for gi in range(G):
        group_scores = scores[indices[gi]]
        phi_mean = np.array([1.0, float(group_scores.mean())])
        sum_lo, sum_hi = nonnegative_sum_interval(group_scores)
        cs = WoRControlVariateCS(
            n_total=int(sizes[gi]), alpha=alpha_g, phi_mean_init=phi_mean,
            phi_sum_interval=(np.array([1.0 * sizes[gi], sum_lo]),
                              np.array([1.0 * sizes[gi], sum_hi])),
            phi_min_g=np.array([1.0, float(group_scores.min())]),
            phi_max_g=np.array([1.0, float(group_scores.max())]),
            target_eps=float(eps_norm[gi]), safe_rounding=True,
        )
        cs_list.append(cs)
        theta_list.append(np.zeros(2))
        bounds_list.append((0.0, 0.0))
        draw(gi, pilot_per_group)

    if fixed_predictor:
        for gi in range(G):
            if phi_seen[gi]:
                X = np.vstack(phi_seen[gi]); y = np.concatenate(y_seen[gi])
                theta = fit_fixed_linear_predictor(X, y)
            else:
                theta = np.array([0.0, 1.0])
            theta_list[gi] = theta
            group_phi = np.column_stack([np.ones(sizes[gi]), scores[indices[gi]]])
            bounds_list[gi] = linear_prediction_bounds(theta, group_phi, safe_rounding=True)
    # Raw-WoR keeps theta=0 forever.  Fixed-proxy keeps the pilot theta forever.

    while total_calls < budget:
        slacks = np.array([
            0.0 if (cs.exhausted or cs.half_width <= eps_norm[gi])
            else cs.half_width / max(eps_norm[gi], 1e-12)
            for gi, cs in enumerate(cs_list)
        ])
        if float(np.max(slacks)) <= 0.0:
            break
        gi = int(np.argmax(slacks))
        if draw(gi, batch_size) == 0:
            break

    estimates, lower, upper, widths, calls = {}, {}, {}, {}, {}
    converged = True
    for gi, name in enumerate(names):
        cs = cs_list[gi]
        if cs.exhausted and cs.deterministic_count_bounds is not None:
            lo, hi = cs.deterministic_count_bounds
        else:
            lo_norm, hi_norm = cs.bounds
            lo, hi = float(lo_norm * sizes[gi]), float(hi_norm * sizes[gi])
        estimates[name] = (lo + hi) / 2
        lower[name], upper[name] = lo, hi
        widths[name] = (hi - lo) / 2
        calls[name] = int(cs.n_seen)
        if not cs.exhausted and widths[name] > eps_norm[gi] * sizes[gi] + 1e-9:
            converged = False
    wall = (time.perf_counter() - start) * 1000
    return ModernCountResult(
        method=method, groups=names, estimates=estimates, lower=lower, upper=upper,
        half_widths=widths, oracle_calls_by_group=calls, total_oracle_calls=total_calls,
        total_rows=int(sizes.sum()), wall_ms=wall, oracle_ms=oracle_ms,
        optimizer_ms=max(0.0, wall - oracle_ms), converged=converged,
        metadata={"risk_allocation": "Bonferroni", "alpha_g": alpha_g,
                  "fixed_predictor": fixed_predictor,
                  "adaptation": "operator-level; not a full system replica"},
    )


def spertus_wor_count(group_indices, proxy_scores, oracle, eps, delta=0.1,
                      batch_size=32, pilot_per_group=8, seed=42, **kwargs):
    """Raw uniform-WoR stratified comparator, no ML control variate.

    Source: Spertus–Sridhar–Stark arXiv:2409.06680v5 §2.2, §2.3.1, §3.1.
    This uses the WoR remaining-mean betting engine and Bonferroni per-group
    risk.  It does not implement the UI-TS or the weighted global LCB of
    Proposition 2 as a full audit system.
    """
    return _run_cs_count("Spertus-WoR", group_indices, proxy_scores, oracle, eps,
                         delta, batch_size, pilot_per_group, seed,
                         fixed_predictor=False, **kwargs)


def calibrated_fixed_proxy_count(group_indices, proxy_scores, oracle, eps, delta=0.1,
                                 batch_size=32, pilot_per_group=8, seed=42, **kwargs):
    """One pilot-fitted frozen linear predictor, shared WoR CS and routing."""
    return _run_cs_count("Calibrated-FixedProxy", group_indices, proxy_scores, oracle,
                         eps, delta, batch_size, pilot_per_group, seed,
                         fixed_predictor=True, **kwargs)


def celeus_signal(seen_sum: float, remaining_proxy_sum: float, y: float,
                  proxy: float, n_total: int, q_obs: float) -> float:
    """CELEUS Eq. (2): surrogate completion plus realized IPW residual."""
    return (seen_sum + remaining_proxy_sum) / n_total + (y - proxy) / (n_total * q_obs)


class _CeleusPoolCS:
    """Uniform/nonuniform fixed-proxy hedged-capital inversion, Algorithm B."""

    def __init__(self, n_total: int, proxy: np.ndarray, alpha: float,
                 grid_size: int = 401, beta: float = 0.4, theta_hedge: float = 0.5,
                 c: float = 1e-3, population_alpha2: float = 0.0):
        self.n = int(n_total)
        self.proxy = np.asarray(proxy, dtype=np.float64)
        self.proxy_sum = float(np.sum(self.proxy))
        self.f_min, self.f_max = float(self.proxy.min()), float(self.proxy.max())
        self.alpha = float(alpha)
        self.log_thresh = math.log(1.0 / self.alpha)
        self.grid = np.linspace(0.0, 1.0, grid_size)
        self.spacing = 1.0 / (grid_size - 1)
        self.beta = float(beta)
        self.theta_hedge = float(theta_hedge)
        self.c = float(c)
        self.population_delta = (
            math.sqrt(math.log(2.0 / population_alpha2) / (2.0 * self.n))
            if population_alpha2 > 0 else 0.0
        )
        self.log_plus = np.zeros(grid_size)
        self.log_minus = np.zeros(grid_size)
        self.active = np.ones(grid_size, dtype=bool)
        self.seen_sum = 0.0
        self.seen_proxy_sum = 0.0
        self.n_seen = 0
        self.signal_history: List[float] = []
        self.signal_mean = 0.0
        self.signal_m2 = 0.0
        self.residuals: List[float] = []
        self.q_history: List[float] = []
        self.lo, self.hi = 0.0, 1.0

    def _bounds(self, q_obs: float) -> Tuple[float, float]:
        rem = self.n - self.n_seen
        base = (self.seen_sum + self.proxy_sum - self.seen_proxy_sum) / self.n
        weight = 1.0 / (self.n * q_obs)
        a = base - self.f_max * weight
        b = base + (1.0 - self.f_min) * weight
        return min(a, 0.0), max(b, 1.0)

    def update(self, y: float, proxy_value: float, q_obs: float) -> None:
        signal = celeus_signal(self.seen_sum, self.proxy_sum - self.seen_proxy_sum,
                               y, proxy_value, self.n, q_obs)
        a, b = self._bounds(q_obs)
        span = max(b - a, 1e-12)
        sbar = min(1.0, max(0.0, (signal - a) / span))
        t = self.n_seen + 1
        sig2 = max(self.signal_m2 / self.n_seen, 1e-4) if self.n_seen > 0 else 0.25
        gamma = min(10.0, max(0.1, math.sqrt(2.0 * math.log(2.0 / self.alpha) / sig2)))
        lam_tilde = gamma / math.sqrt(t * math.log(t + 1.0))
        vbar = (self.grid - a) / span
        lam_plus = np.minimum(lam_tilde, 1.0 / (vbar + self.c))
        lam_minus = np.minimum(lam_tilde, 1.0 / (1.0 - vbar + self.c))
        d = sbar - vbar
        self.log_plus += np.log1p(lam_plus * d)
        self.log_minus += np.log1p(-lam_minus * d)
        hedged = np.logaddexp(math.log(self.theta_hedge) + self.log_plus,
                              math.log1p(-self.theta_hedge) + self.log_minus)
        self.active &= hedged < self.log_thresh
        # Predictable history update for the next round's betting fraction.
        delta = signal - self.signal_mean
        self.signal_mean += delta / t
        self.signal_m2 += delta * (signal - self.signal_mean)
        self.signal_history.append(signal)
        self.residuals.append(y - proxy_value)
        self.q_history.append(q_obs)
        self.seen_sum += y
        self.seen_proxy_sum += proxy_value
        self.n_seen += 1
        det_lo = self.seen_sum / self.n
        det_hi = (self.seen_sum + self.n - self.n_seen) / self.n
        if self.n_seen == self.n:
            self.lo = self.hi = self.seen_sum / self.n
        elif np.any(self.active):
            surviving = self.grid[self.active]
            lo = max(det_lo, float(surviving.min()) - self.spacing - self.population_delta)
            hi = min(det_hi, float(surviving.max()) + self.spacing + self.population_delta)
            self.lo, self.hi = (lo, hi) if lo <= hi else (det_lo, det_hi)
        else:
            self.lo, self.hi = det_lo, det_hi

    @property
    def bounds(self):
        return self.lo, self.hi

    @property
    def half_width(self):
        return (self.hi - self.lo) / 2

    @property
    def exhausted(self):
        return self.n_seen >= self.n

    def estimate(self):
        if self.exhausted:
            return self.seen_sum / self.n
        t = self.n_seen
        weights = [1.0 + (self.n - t) / (self.n - m) *
                   (1.0 / ((self.n - m + 1) * self.q_history[m - 1]) - 1.0)
                   for m in range(1, t + 1)]
        return self.proxy_sum / self.n + sum(w * r for w, r in zip(weights, self.residuals)) / t


def celeus_fixed_proxy_count(group_indices, proxy_scores, oracle, eps, delta=0.1,
                             batch_size=32, pilot_per_group=8, seed=42, *,
                             nonuniform: bool = False, beta: float = 0.4,
                             grid_size: int = 401, max_oracle_budget=None,
                             group_names=None):
    """CELEUS fixed-proxy e-process adaptation (Eq. 2–4, Algorithm B).

    Uniform WoR is the default.  ``nonuniform=True`` uses the probability-floor
    proposal q_t(j)=beta/m_t+(1-beta)p_t(j) from Appendix B/C.1.  The target is
    finite-pool risk, so the population correction is disabled by default.
    Grid endpoints are padded by one spacing as a conservative numerical guard.
    """
    start = time.perf_counter()
    indices, names = _group_arrays(group_indices, group_names)
    scores = np.asarray(proxy_scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    G = len(indices)
    sizes = np.array([len(v) for v in indices], dtype=np.int64)
    eps_norm = np.array([(eps[name] if isinstance(eps, dict) else float(eps)) / max(int(n), 1)
                         for name, n in zip(names, sizes)])
    pools = [rng.permutation(v) for v in indices] if not nonuniform else [v.copy() for v in indices]
    ptrs = np.zeros(G, dtype=np.int64)
    cs_list = [_CeleusPoolCS(int(n), scores[idx], delta / G, grid_size=grid_size, beta=beta)
               for n, idx in zip(sizes, indices)]
    total_calls = 0
    oracle_ms = 0.0
    budget = max_oracle_budget if max_oracle_budget is not None else int(sizes.sum())

    def draw(gi: int, k_req: int) -> int:
        nonlocal total_calls, oracle_ms
        cs = cs_list[gi]
        rem_budget = budget - total_calls
        rem_group = int(sizes[gi] - ptrs[gi])
        k = min(k_req, rem_group, rem_budget)
        if k <= 0:
            return 0
        rows, q_obs = [], []
        if nonuniform:
            remaining = pools[gi] if nonuniform else pools[gi][ptrs[gi]:]
            mean_seen = cs.seen_sum / cs.n_seen if cs.n_seen else 0.5
            acquisition = np.abs(mean_seen - scores[remaining])
            active_mask = np.ones(len(remaining), dtype=bool)
            selected_mask = np.zeros(len(remaining), dtype=bool)
            for _ in range(k):
                m = int(active_mask.sum())
                s = acquisition[active_mask]
                p = s / s.sum() if s.sum() > 1e-12 else np.full(m, 1.0 / m)
                q = beta / m + (1.0 - beta) * p
                local = int(rng.choice(m, p=q))
                global_pos = np.flatnonzero(active_mask)[local]
                rows.append(int(remaining[global_pos]))
                q_obs.append(float(q[local]))
                active_mask[global_pos] = False
                selected_mask[global_pos] = True
            rows = np.asarray(rows, dtype=np.int64)
            pools[gi] = remaining[~selected_mask]
            ptrs[gi] = sizes[gi] - len(pools[gi])
        else:
            rows = pools[gi][ptrs[gi]:ptrs[gi] + k]
            ptrs[gi] += k
            q_obs = [1.0 / (sizes[gi] - cs.n_seen - j) for j in range(k)]
        y, ms = _oracle_eval(oracle, rows)
        oracle_ms += ms
        total_calls += k
        for j, row in enumerate(rows):
            cs.update(float(y[j]), float(scores[row]), float(q_obs[j]))
        return k

    for gi in range(G):
        draw(gi, pilot_per_group)
    while total_calls < budget:
        slacks = np.array([0.0 if (cs.exhausted or cs.half_width <= eps_norm[gi])
                           else cs.half_width / max(eps_norm[gi], 1e-12)
                           for gi, cs in enumerate(cs_list)])
        if float(np.max(slacks)) <= 0.0:
            break
        if draw(int(np.argmax(slacks)), batch_size) == 0:
            break
    estimates, lower, upper, widths, calls = {}, {}, {}, {}, {}
    converged = True
    for gi, name in enumerate(names):
        cs = cs_list[gi]
        lo, hi = cs.bounds
        lo_count, hi_count = lo * sizes[gi], hi * sizes[gi]
        if cs.exhausted:
            lo_count = hi_count = cs.seen_sum
        estimates[name] = cs.estimate() * sizes[gi]
        lower[name], upper[name] = lo_count, hi_count
        widths[name] = (hi_count - lo_count) / 2
        calls[name] = cs.n_seen
        if not cs.exhausted and widths[name] > eps_norm[gi] * sizes[gi] + 1e-9:
            converged = False
    wall = (time.perf_counter() - start) * 1000
    return ModernCountResult(
        method="CELEUS-FixedProxy" + ("-Nonuniform" if nonuniform else ""),
        groups=names, estimates=estimates, lower=lower, upper=upper,
        half_widths=widths, oracle_calls_by_group=calls, total_oracle_calls=total_calls,
        total_rows=int(sizes.sum()), wall_ms=wall, oracle_ms=oracle_ms,
        optimizer_ms=max(0.0, wall - oracle_ms), converged=converged,
        metadata={"nonuniform": nonuniform, "beta": beta, "grid_size": grid_size,
                  "population_correction": False,
                  "adaptation": "CELEUS Eq.2-4/Algorithm B operator adaptation"},
    )


def bargain_global_count(group_indices, proxy_scores, oracle, eps_global: float,
                         delta=0.1, batch_size=64, pilot=32, seed=42,
                         max_oracle_budget=None, group_names=None):
    """Single global anytime count target; group outputs are proportional plugs.

    BARGAIN source: arXiv:2509.02896 §4.1.2 / Appendix B.4.4 Lemma B.6.
    This is a global-quality operator adaptation, not the cascade system.
    """
    start = time.perf_counter()
    indices, names = _group_arrays(group_indices, group_names)
    all_rows = np.concatenate(indices)
    scores = np.asarray(proxy_scores, dtype=np.float64)
    n = len(all_rows)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(all_rows)
    cs = WoRControlVariateCS(n_total=n, alpha=delta,
                             phi_mean_init=np.array([1.0, float(scores[all_rows].mean())]),
                             target_eps=eps_global / n, safe_rounding=True)
    theta = np.zeros(2)
    total = 0
    oracle_ms = 0.0
    budget = max_oracle_budget if max_oracle_budget is not None else n

    def draw(k):
        nonlocal total, oracle_ms
        k = min(k, n - total, budget - total)
        if k <= 0:
            return 0
        rows = perm[total:total + k]
        total += k
        y, ms = _oracle_eval(oracle, rows)
        oracle_ms += ms
        cs.update_batch(y, np.column_stack([np.ones(k), scores[rows]]), theta)
        return k

    draw(pilot)
    while total < budget and not cs.exhausted and cs.half_width * n > eps_global:
        if draw(batch_size) == 0:
            break
    lo_norm, hi_norm = cs.bounds
    lo, hi = lo_norm * n, hi_norm * n
    if cs.exhausted and cs.deterministic_count_bounds is not None:
        lo, hi = cs.deterministic_count_bounds
    estimate = cs.estimate * n
    group_est, group_lo, group_hi = {}, {}, {}
    for name, idx in zip(names, indices):
        weight = len(idx) / n
        group_est[name] = estimate * weight
        group_lo[name] = lo * weight
        group_hi[name] = hi * weight
    wall = (time.perf_counter() - start) * 1000
    return GlobalCountResult(
        method="BARGAIN-Global", estimate=estimate, lower=lo, upper=hi,
        total_oracle_calls=total, total_rows=n, wall_ms=wall,
        converged=cs.exhausted or (hi - lo) / 2 <= eps_global + 1e-9,
        group_estimates=group_est, group_lower=group_lo, group_upper=group_hi,
        metadata={"global_target_only": True, "group_output": "proportional plug-in",
                  "adaptation": "BARGAIN global-quality operator, not cascade replica"},
    )


def pilotdb_asymp_count(group_indices, oracle, eps, delta=0.1, pilot_per_group=32,
                        seed=42, optional_stopping=True, looks=None, group_names=None):
    """Two-stage CLT simultaneous intervals; optional stopping is diagnostic.

    PilotDB source: arXiv:2503.21087 §2.4, §3.1 Theorem 3.1, Appendix B.1.
    The paper is a pilot→final a-priori design, not anytime.  When
    ``optional_stopping=True`` we deliberately reuse the fixed-look CLT radius
    at data-dependent looks to expose the coverage failure mode.
    """
    start = time.perf_counter()
    indices, names = _group_arrays(group_indices, group_names)
    rng = np.random.default_rng(seed)
    G = len(indices)
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1.0 - delta / (2.0 * G))
    estimates, lower, upper, widths, calls = {}, {}, {}, {}, {}
    total = 0
    oracle_ms = 0.0
    converged = True
    for gi, (name, idx) in enumerate(zip(names, indices)):
        n = len(idx)
        perm = rng.permutation(idx)
        n_pilot = min(pilot_per_group, n)
        y_pilot, ms = _oracle_eval(oracle, perm[:n_pilot])
        oracle_ms += ms
        total += n_pilot
        mean = float(np.mean(y_pilot))
        var = float(np.var(y_pilot, ddof=1)) if n_pilot > 1 else 0.25
        target_norm = (eps[name] if isinstance(eps, dict) else float(eps)) / n
        # Solve z*sqrt(var/m*(N-m)/(N-1)) <= eps for the final fixed look.
        if target_norm <= 0:
            n_final = n
        else:
            denom = target_norm ** 2 * (n - 1) + z ** 2 * var
            n_final = min(n, max(n_pilot, int(math.ceil(z ** 2 * var * n / denom))))
        if target_norm <= 0:
            look_sizes = [n]
        elif looks is None:
            look_sizes = [n_pilot, n_final] if not optional_stopping else sorted(set(
                [n_pilot] + [min(n, n_pilot * 2 ** k) for k in range(12)] + [n_final]))
        else:
            look_sizes = sorted(set(min(n, max(n_pilot, int(v))) for v in looks))
        seen = n_pilot
        y_sum = float(y_pilot.sum())
        y_sq = float(np.sum(y_pilot ** 2))
        lo = hi = mean
        for n_look in look_sizes:
            if n_look > seen:
                y_more, ms = _oracle_eval(oracle, perm[seen:n_look])
                oracle_ms += ms
                total += len(y_more)
                y_sum += float(y_more.sum())
                y_sq += float(np.sum(y_more ** 2))
                seen = n_look
            mean = y_sum / seen
            var = max(0.0, (y_sq - seen * mean ** 2) / max(seen - 1, 1))
            fpc = max(0.0, (n - seen) / max(n - 1, 1))
            radius = z * math.sqrt(var / seen * fpc)
            lo, hi = max(0.0, mean - radius), min(1.0, mean + radius)
            if optional_stopping and (hi - lo) / 2 <= target_norm:
                break
            if seen >= n:
                break
        if seen >= n:
            lo = hi = y_sum / n
        estimates[name] = (lo + hi) / 2 * n
        lower[name], upper[name] = lo * n, hi * n
        widths[name] = (hi - lo) / 2 * n
        calls[name] = seen
        if seen < n and widths[name] > target_norm * n + 1e-9:
            converged = False
    wall = (time.perf_counter() - start) * 1000
    return ModernCountResult(
        method="PilotDB-Asymp" + ("-OptionalStop" if optional_stopping else ""),
        groups=names, estimates=estimates, lower=lower, upper=upper,
        half_widths=widths, oracle_calls_by_group=calls, total_oracle_calls=total,
        total_rows=sum(len(v) for v in indices), wall_ms=wall, oracle_ms=oracle_ms,
        optimizer_ms=max(0.0, wall - oracle_ms), converged=converged,
        metadata={"optional_stopping": optional_stopping, "clt": True,
                  "adaptation": "PilotDB CLT operator; not an anytime certificate"},
    )


def decisions_from_count_bounds(result: ModernCountResult, truth: Dict[str, int],
                                tau: Optional[float] = None, k: Optional[int] = None):
    """Derive HAVING/TOP-k decisions from final count intervals (no extra draws)."""
    groups = result.groups if hasattr(result, "groups") else list(truth)
    lower = result.group_lower if hasattr(result, "group_lower") else result.lower
    upper = result.group_upper if hasattr(result, "group_upper") else result.upper
    if tau is not None:
        accepted = [g for g in groups if lower[g] >= tau]
        rejected = [g for g in groups if upper[g] < tau]
        uncertain = [g for g in groups if g not in accepted and g not in rejected]
        correct = all(truth[g] >= tau for g in accepted) and all(truth[g] < tau for g in rejected)
        return {"accepted": accepted, "rejected": rejected, "uncertain": uncertain,
                "decision_correct": correct}
    if k is not None:
        order = sorted(groups, key=lambda g: (-(lower[g] + upper[g]) / 2, g))
        selected = order[:k]
        rest = order[k:]
        certified = bool(selected and rest and
                         min(lower[g] for g in selected) > max(upper[g] for g in rest))
        expected = sorted(truth, key=lambda g: (-truth[g], g))[:k]
        return {"selected": selected, "certified": certified,
                "decision_correct": set(selected) == set(expected)}
    raise ValueError("provide tau or k")

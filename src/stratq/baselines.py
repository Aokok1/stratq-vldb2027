"""
Faithful implementations of PVLDB/SIGMOD baselines for head-to-head comparison:
  1. ExhaustiveOracle          (exact full oracle scan)
  2. UniformStratifiedAQP      (BlinkDB-style proportional/equal stratified sampling, no proxy)
  3. ABAEGlobal / ABAEGroup    (Kang et al., PVLDB 2021: proxy-assisted aggregate estimation)
  4. SUPGGlobal / SUPGGroup    (Kang et al., PVLDB 2020: importance-weighted selection)
  5. LOTUSCascadeGlobal        (Patel et al., PVLDB 2025: global two-threshold proxy cascade)
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .bounds import empirical_bernstein_radius
from .engine import GroupAggResult, GroupSelectResult, StratQEngine


def run_exhaustive_agg(engine: StratQEngine, agg_type: str = "mean") -> GroupAggResult:
    """Exact ground-truth baseline scanning 100% of tuples."""
    t0 = time.perf_counter()
    prof = engine.profile
    assert prof.oracle_true is not None
    y_all = prof.oracle_true * prof.measure_raw
    est, lo, hi, hw, calls, n_d = {}, {}, {}, {}, {}, {}
    for gi, g in enumerate(prof.groups):
        idxs = prof.group_row_indices[gi]
        n_g = len(idxs)
        val = float(np.sum(y_all[idxs])) if agg_type in ("count", "sum") else float(np.mean(y_all[idxs]))
        est[g] = val
        lo[g] = val
        hi[g] = val
        hw[g] = 0.0
        calls[g] = n_g
        n_d[g] = n_g
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return GroupAggResult(
        groups=prof.groups,
        n_total_by_group=n_d,
        estimates=est,
        lower=lo,
        upper=hi,
        half_widths=hw,
        oracle_calls_by_group=calls,
        total_oracle_calls=prof.n_total,
        total_rows=prof.n_total,
        duckdb_scan_ms=prof.duckdb_scan_ms,
        optimizer_ms=wall_ms,
        oracle_ms=0.0,
        wall_ms=wall_ms,
        converged=True,
        agg_type=agg_type,
    )


def run_uniform_stratified_agg(
    engine: StratQEngine,
    eps: Union[float, Dict[str, float]] = 0.05,
    delta: float = 0.05,
    agg_type: str = "mean",
    proportional: bool = False,
    batch_size: int = 16,
    max_oracle_budget: Optional[int] = None,
    seed: int = 42,
) -> GroupAggResult:
    """
    BlinkDB-style Stratified AQP without proxy or covariates.
    - proportional=False: Equal round-robin allocation across all G groups until all satisfy eps_g.
    - proportional=True: Proportional-to-N_g sampling (exposes minority-group starvation!).
    """
    if not proportional:
        return engine.execute_grouped_agg(
            eps=eps,
            delta=delta,
            agg_type=agg_type,
            batch_size=batch_size,
            pilot_per_group=8,
            max_oracle_budget=max_oracle_budget,
            use_control_variates=False,
            use_covariates=False,
            use_hierarchical=False,
            use_adaptive_alloc=False,
            use_wor=True,
            seed=seed,
        )

    # Proportional-to-N_g sampling under fixed budget (or until global eps met)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    prof = engine.profile
    G = len(prof.groups)
    alpha_g = delta / max(G, 1)
    y_span = max(prof.measure_max - prof.measure_min, 1e-12)
    assert prof.oracle_true is not None
    y_all = prof.oracle_true * prof.measure_raw

    budget = max_oracle_budget if max_oracle_budget is not None else max(G * 30, int(0.05 * prof.n_total))
    budget = min(budget, prof.n_total)
    chosen = rng.choice(prof.n_total, size=budget, replace=False)

    # Partition sampled rows by group
    g_mask_map = np.empty(prof.n_total, dtype=np.int32)
    for gi, idxs in enumerate(prof.group_row_indices):
        g_mask_map[idxs] = gi

    est_d, lo_d, hi_d, hw_d, calls_d, n_d = {}, {}, {}, {}, {}, {}
    converged = True
    for gi, g in enumerate(prof.groups):
        n_g = int(prof.n_by_group[gi])
        n_d[g] = n_g
        scale = float(n_g) if agg_type in ("count", "sum") else 1.0
        e_target = eps[g] if isinstance(eps, dict) else float(eps)
        e_mean = e_target / max(n_g, 1) if agg_type in ("count", "sum") else e_target

        s_rows = chosen[g_mask_map[chosen] == gi]
        k = len(s_rows)
        calls_d[g] = k
        if k == 0:
            est_d[g] = 0.5 * scale
            lo_d[g] = prof.measure_min * scale
            hi_d[g] = prof.measure_max * scale
            hw_d[g] = 0.5 * y_span * scale
            converged = False
        elif k >= n_g:
            m = float(np.mean(y_all[s_rows]))
            est_d[g] = lo_d[g] = hi_d[g] = m * scale
            hw_d[g] = 0.0
        else:
            vals = y_all[s_rows]
            m = float(np.mean(vals))
            var = float(np.var(vals, ddof=1)) if k > 1 else (0.25 * y_span * y_span)
            rad = empirical_bernstein_radius(var, k, alpha_g, y_span)
            lo_m = max(prof.measure_min, m - rad)
            hi_m = min(prof.measure_max, m + rad)
            est_d[g] = m * scale
            lo_d[g] = lo_m * scale
            hi_d[g] = hi_m * scale
            hw_d[g] = 0.5 * (hi_m - lo_m) * scale
            if 0.5 * (hi_m - lo_m) > e_mean:
                converged = False

    wall_ms = (time.perf_counter() - t0) * 1000.0
    return GroupAggResult(
        groups=prof.groups,
        n_total_by_group=n_d,
        estimates=est_d,
        lower=lo_d,
        upper=hi_d,
        half_widths=hw_d,
        oracle_calls_by_group=calls_d,
        total_oracle_calls=budget,
        total_rows=prof.n_total,
        duckdb_scan_ms=prof.duckdb_scan_ms,
        optimizer_ms=wall_ms,
        oracle_ms=0.0,
        wall_ms=wall_ms,
        converged=converged,
        agg_type=agg_type,
    )


def run_abae_global_agg(
    engine: StratQEngine,
    eps: float = 0.05,
    delta: float = 0.05,
    agg_type: str = "mean",
    max_oracle_budget: Optional[int] = None,
    seed: int = 42,
) -> GroupAggResult:
    """
    ABAE-Global (Kang et al., PVLDB 2021):
    Samples globally with proxy-guided importance sampling (defensive mixture
    q_i proportional to sqrt(p_i) + c0) until GLOBAL mean CI <= eps, then
    reports per-group Horvitz-Thompson / difference estimates from the same global sample.
    Demonstrates severe per-group error blowup on skewed GROUP BY queries.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    prof = engine.profile
    assert prof.oracle_true is not None
    y_all = prof.oracle_true * prof.measure_raw
    p_all = np.clip(prof.phi_all[:, 1], 0.0, 1.0)
    N = prof.n_total
    G = len(prof.groups)
    y_span = max(prof.measure_max - prof.measure_min, 1e-12)

    # Defensive importance weights q_i (ABAE / SUPG style: 0.9 * sqrt(p_i) + 0.1 * uniform)
    raw_w = 0.9 * np.sqrt(np.maximum(p_all, 1e-4)) + 0.1
    q = raw_w / np.sum(raw_w)

    # Difference estimator using proxy: y_i - p_i (scaled to measure)
    p_pred_all = p_all * prof.measure_raw
    budget_cap = max_oracle_budget if max_oracle_budget is not None else N

    # Sequential global check in batches until global EB radius <= eps
    sampled_idx: List[int] = []
    batch = 32
    est_resid_vals: List[float] = []
    # Maximum weight ratio for valid range bound on HT difference
    max_inv_nq = float(np.max(1.0 / (N * q)))
    ht_range = 2.0 * y_span * max_inv_nq

    while len(sampled_idx) < budget_cap:
        k = min(batch, budget_cap - len(sampled_idx))
        draw = rng.choice(N, size=k, replace=True, p=q)
        sampled_idx.extend(draw.tolist())
        # Horvitz-Thompson residual on global population
        res = (y_all[draw] - p_pred_all[draw]) / (N * q[draw])
        est_resid_vals.extend(res.tolist())
        n_cur = len(est_resid_vals)
        if n_cur >= 32:
            var = float(np.var(est_resid_vals, ddof=1))
            # CLT-style stopping as in ABAE (z * se <= eps)
            z = math.sqrt(2.0 * math.log(2.0 / delta))
            rad_glob = z * math.sqrt(var / n_cur) + (ht_range * math.log(2.0 / delta) / (3.0 * n_cur))
            if rad_glob <= eps:
                break

    sampled_arr = np.array(sampled_idx, dtype=np.int64)
    g_map = np.empty(N, dtype=np.int32)
    for gi, idxs in enumerate(prof.group_row_indices):
        g_map[idxs] = gi

    alpha_g = delta / max(G, 1)
    est_d, lo_d, hi_d, hw_d, calls_d, n_d = {}, {}, {}, {}, {}, {}
    n_tot_samp = len(sampled_arr)
    for gi, g in enumerate(prof.groups):
        idxs = prof.group_row_indices[gi]
        n_g = len(idxs)
        n_d[g] = n_g
        scale = float(n_g) if agg_type in ("count", "sum") else 1.0
        p_mean_g = float(np.mean(p_pred_all[idxs]))
        mask_g = g_map[sampled_arr] == gi
        calls_d[g] = int(np.sum(mask_g))
        # Unbiased group HT difference estimator from global sample:
        # mean_g = p_mean_g + (1 / n_tot_samp) * sum_{s: g_s = g} (y_s - p_s) / (n_g * q_s)
        ht_g_terms = np.zeros(n_tot_samp, dtype=np.float64)
        if calls_d[g] > 0:
            s_g = sampled_arr[mask_g]
            ht_g_terms[: len(s_g)] = (y_all[s_g] - p_pred_all[s_g]) / (n_g * q[s_g])
        m_g = float(np.clip(p_mean_g + np.mean(ht_g_terms), prof.measure_min, prof.measure_max))
        se_g = float(np.std(ht_g_terms, ddof=1) / math.sqrt(max(n_tot_samp, 1))) if n_tot_samp > 1 else y_span
        rad_g = math.sqrt(2.0 * math.log(2.0 / alpha_g)) * se_g
        lo_g = max(prof.measure_min, m_g - rad_g)
        hi_g = min(prof.measure_max, m_g + rad_g)
        est_d[g] = m_g * scale
        lo_d[g] = lo_g * scale
        hi_d[g] = hi_g * scale
        hw_d[g] = 0.5 * (hi_g - lo_g) * scale

    wall_ms = (time.perf_counter() - t0) * 1000.0
    return GroupAggResult(
        groups=prof.groups,
        n_total_by_group=n_d,
        estimates=est_d,
        lower=lo_d,
        upper=hi_d,
        half_widths=hw_d,
        oracle_calls_by_group=calls_d,
        total_oracle_calls=int(n_tot_samp),
        total_rows=N,
        duckdb_scan_ms=prof.duckdb_scan_ms,
        optimizer_ms=wall_ms,
        oracle_ms=0.0,
        wall_ms=wall_ms,
        converged=False,
        agg_type=agg_type,
    )


def run_abae_group_agg(
    engine: StratQEngine,
    eps: Union[float, Dict[str, float]] = 0.05,
    delta: float = 0.05,
    agg_type: str = "mean",
    batch_size: int = 16,
    max_oracle_budget: Optional[int] = None,
    seed: int = 42,
) -> GroupAggResult:
    """
    ABAE-Group (Independent per-group proxy-only control variate with Bonferroni delta/G,
    equal round-robin allocation, no relational covariates, no hierarchical shrinkage).
    """
    return engine.execute_grouped_agg(
        eps=eps,
        delta=delta,
        agg_type=agg_type,
        batch_size=batch_size,
        pilot_per_group=8,
        max_oracle_budget=max_oracle_budget,
        use_control_variates=True,
        use_covariates=False,
        use_hierarchical=False,
        use_adaptive_alloc=False,
        use_wor=True,
        seed=seed,
    )


def run_supg_global_selection(
    engine: StratQEngine,
    target_gamma: float = 0.90,
    target_mode: str = "recall",
    delta: float = 0.05,
    oracle_budget: int = 1000,
    seed: int = 42,
) -> GroupSelectResult:
    """
    SUPG-Global (Kang et al., PVLDB 2020):
    Importance-weighted global sampling (defensive mixture proportional to sqrt(p_i))
    choosing a single global threshold tau* satisfying global Recall >= target_gamma
    (or Precision >= target_gamma) with probability >= 1 - delta.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    prof = engine.profile
    assert prof.oracle_true is not None
    N = prof.n_total
    y_all = (prof.oracle_true >= 0.5).astype(np.float64)
    p_all = np.clip(prof.phi_all[:, 1], 0.0, 1.0)
    B = min(int(oracle_budget), N)

    # Defensive importance sampling weights w_i proportional to 0.9 * sqrt(p_i) + 0.1
    raw_w = 0.9 * np.sqrt(np.maximum(p_all, 1e-4)) + 0.1
    q = raw_w / np.sum(raw_w)
    samp = rng.choice(N, size=B, replace=False, p=q)
    inv_w = 1.0 / (N * q[samp])
    y_s = y_all[samp]
    p_s = p_all[samp]

    # Sort sampled points descending by proxy score
    ord_s = np.argsort(-p_s)
    p_ord = p_s[ord_s]
    wy_ord = inv_w[ord_s] * y_s[ord_s]
    w_ord = inv_w[ord_s]

    cum_wy = np.cumsum(wy_ord)
    tot_wy = float(cum_wy[-1]) if len(cum_wy) > 0 else 0.0
    z = math.sqrt(2.0 * math.log(2.0 / delta))

    if target_mode == "recall":
        # Pick highest threshold tau whose lower confidence bound on global recall >= target_gamma
        tau_star = 0.0
        for idx in range(len(p_ord)):
            if tot_wy <= 1e-12:
                tau_star = 0.0
                break
            # Variance of linearized recall margin h_i = (1-gamma)*y_i*1[i<=idx] - gamma*y_i*1[i>idx]
            h = np.empty(B, dtype=np.float64)
            h[: idx + 1] = (1.0 - target_gamma) * wy_ord[: idx + 1]
            h[idx + 1 :] = -target_gamma * wy_ord[idx + 1 :]
            mean_h = float(np.mean(h))
            se_h = float(np.std(h, ddof=1) / math.sqrt(B)) if B > 1 else 1.0
            if mean_h - z * se_h >= 0.0:
                tau_star = float(p_ord[idx])
                break
    else:
        # Precision target: largest prefix whose precision LCB >= target_gamma
        tau_star = 1.01
        for idx in range(len(p_ord) - 1, -1, -1):
            h = np.zeros(B, dtype=np.float64)
            h[: idx + 1] = w_ord[: idx + 1] * (y_s[ord_s][: idx + 1] - target_gamma)
            mean_h = float(np.mean(h))
            se_h = float(np.std(h, ddof=1) / math.sqrt(B)) if B > 1 else 1.0
            if mean_h - z * se_h >= 0.0:
                tau_star = float(p_ord[idx])
                break

    # Build output set (oracle positives among sampled + proxy >= tau_star among unsampled)
    sampled_mask = np.zeros(N, dtype=bool)
    sampled_mask[samp] = True
    sel_mask = (sampled_mask & (y_all >= 0.5)) | ((~sampled_mask) & (p_all >= tau_star))
    selected_idx = np.flatnonzero(sel_mask)

    sel_cnt, thr_d, calls_d = {}, {}, {}
    for gi, g in enumerate(prof.groups):
        idxs = prof.group_row_indices[gi]
        sel_cnt[g] = int(np.sum(sel_mask[idxs]))
        thr_d[g] = tau_star
        calls_d[g] = int(np.sum(sampled_mask[idxs]))

    wall_ms = (time.perf_counter() - t0) * 1000.0
    return GroupSelectResult(
        groups=prof.groups,
        selected_indices=selected_idx,
        selected_count_by_group=sel_cnt,
        thresholds_by_group=thr_d,
        oracle_calls_by_group=calls_d,
        total_oracle_calls=B,
        total_rows=N,
        duckdb_scan_ms=prof.duckdb_scan_ms,
        optimizer_ms=wall_ms,
        wall_ms=wall_ms,
        target_mode=target_mode,
        target_gamma=target_gamma,
    )


def run_supg_group_selection(
    engine: StratQEngine,
    target_gamma: float = 0.90,
    target_mode: str = "recall",
    delta: float = 0.05,
    oracle_budget: int = 1000,
    seed: int = 42,
) -> GroupSelectResult:
    """
    SUPG-Group: Naive independent per-group SUPG with equal budget B/G and Bonferroni delta/G,
    using raw proxy score only (no relational covariates, no adaptive cross-group routing).
    """
    return engine.execute_grouped_selection(
        target_gamma=target_gamma,
        target_mode=target_mode,
        delta=delta,
        oracle_budget=oracle_budget,
        pilot_frac=0.0,
        batch_size=16,
        use_covariates=False,
        use_adaptive_alloc=False,
        seed=seed,
    )


def run_lotus_cascade_global_selection(
    engine: StratQEngine,
    target_gamma: float = 0.90,
    target_mode: str = "recall",
    delta: float = 0.05,
    oracle_budget: int = 1000,
    pilot_frac: float = 0.25,
    seed: int = 42,
) -> GroupSelectResult:
    """
    LOTUS-Cascade Global (Patel et al., PVLDB 2025):
    Uses a global calibration pilot to select two global proxy thresholds (tau_low, tau_high):
      - p_i >= tau_high -> accept without oracle
      - p_i <= tau_low  -> reject without oracle
      - tau_low < p_i < tau_high -> route to oracle (up to remaining budget)
    Calibrated strictly for GLOBAL recall/precision target (no per-group guarantees).
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    prof = engine.profile
    assert prof.oracle_true is not None
    N = prof.n_total
    y_all = (prof.oracle_true >= 0.5).astype(np.float64)
    p_all = np.clip(prof.phi_all[:, 1], 0.0, 1.0)
    B = min(int(oracle_budget), N)

    n_pilot = max(32, min(int(B * pilot_frac), N // 2))
    pilot_idx = rng.choice(N, size=n_pilot, replace=False)
    rem_budget = max(0, B - n_pilot)

    # Choose global (tau_low, tau_high) on pilot to route at most rem_budget uncertain tuples
    # while achieving global target_gamma on pilot with conservative margin
    p_pil = p_all[pilot_idx]
    y_pil = y_all[pilot_idx]
    rem_frac = rem_budget / max(N - n_pilot, 1)

    # Search threshold band centered around uncertainty region
    best_pair = (0.0, 1.0)
    best_obj = -1.0
    qs = np.linspace(0.0, 1.0, 21)
    cand_vals = np.unique(np.quantile(p_pil, qs))
    z = math.sqrt(2.0 * math.log(4.0 / delta))

    for t_lo in cand_vals:
        for t_hi in cand_vals:
            if t_hi < t_lo:
                continue
            mid_frac = float(np.mean((p_pil > t_lo) & (p_pil < t_hi)))
            if mid_frac > rem_frac + 0.01:
                continue
            # Simulated cascade output on pilot: mid routed to exact oracle, >=t_hi accepted
            pred_pos = (p_pil >= t_hi) | ((p_pil > t_lo) & (p_pil < t_hi) & (y_pil >= 0.5))
            tp = float(np.sum(pred_pos & (y_pil >= 0.5)))
            pos_tot = float(np.sum(y_pil >= 0.5))
            pred_tot = float(np.sum(pred_pos))
            rec = tp / max(pos_tot, 1.0)
            prec = tp / max(pred_tot, 1.0)
            if target_mode == "recall":
                se = math.sqrt(max(rec * (1.0 - rec), 1e-4) / max(pos_tot, 1.0))
                if rec - 0.5 * z * se >= target_gamma and prec > best_obj:
                    best_obj = prec
                    best_pair = (float(t_lo), float(t_hi))
            else:
                se = math.sqrt(max(prec * (1.0 - prec), 1e-4) / max(pred_tot, 1.0))
                if prec - 0.5 * z * se >= target_gamma and rec > best_obj:
                    best_obj = rec
                    best_pair = (float(t_lo), float(t_hi))

    tau_lo, tau_hi = best_pair
    queried_mask = np.zeros(N, dtype=bool)
    queried_mask[pilot_idx] = True

    # Route uncertain middle tuples to oracle up to rem_budget
    mid_candidates = np.flatnonzero((~queried_mask) & (p_all > tau_lo) & (p_all < tau_hi))
    if len(mid_candidates) > rem_budget:
        chosen_mid = rng.choice(mid_candidates, size=rem_budget, replace=False)
    else:
        chosen_mid = mid_candidates
    queried_mask[chosen_mid] = True
    actual_calls = int(np.sum(queried_mask))

    # Final selection: exact on queried, p >= tau_hi on unqueried (or fallback >= tau_lo for RT)
    if target_mode == "recall":
        sel_mask = (queried_mask & (y_all >= 0.5)) | ((~queried_mask) & (p_all > tau_lo))
    else:
        sel_mask = (queried_mask & (y_all >= 0.5)) | ((~queried_mask) & (p_all >= tau_hi))
    selected_idx = np.flatnonzero(sel_mask)

    sel_cnt, thr_d, calls_d = {}, {}, {}
    for gi, g in enumerate(prof.groups):
        idxs = prof.group_row_indices[gi]
        sel_cnt[g] = int(np.sum(sel_mask[idxs]))
        thr_d[g] = tau_lo if target_mode == "recall" else tau_hi
        calls_d[g] = int(np.sum(queried_mask[idxs]))

    wall_ms = (time.perf_counter() - t0) * 1000.0
    return GroupSelectResult(
        groups=prof.groups,
        selected_indices=selected_idx,
        selected_count_by_group=sel_cnt,
        thresholds_by_group=thr_d,
        oracle_calls_by_group=calls_d,
        total_oracle_calls=actual_calls,
        total_rows=N,
        duckdb_scan_ms=prof.duckdb_scan_ms,
        optimizer_ms=wall_ms,
        wall_ms=wall_ms,
        target_mode=target_mode,
        target_gamma=target_gamma,
    )

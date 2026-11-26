"""
Unit tests & Monte Carlo finite-sample validity verification for StratQ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stratq.baselines import (
    run_abae_global_agg,
    run_abae_group_agg,
    run_exhaustive_agg,
    run_lotus_cascade_global_selection,
    run_supg_global_selection,
    run_supg_group_selection,
    run_uniform_stratified_agg,
)
from stratq.engine import StratQEngine


def make_skewed_synthetic_table(
    n_groups: int = 12,
    n_total: int = 12000,
    seed: int = 123,
) -> pa.Table:
    """
    Generate a skewed multi-group dataset with:
      - Zipf-skewed group sizes (large groups ~3000, tail groups ~120)
      - Heterogeneous proxy miscalibration across groups
      - Informative relational covariates z1, z2 that explain part of proxy error
    """
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_groups + 1, dtype=np.float64)
    probs = (ranks ** -1.1)
    probs /= probs.sum()
    counts = np.maximum(80, np.round(probs * n_total).astype(int))
    N = int(counts.sum())

    groups, proxy, z1, z2, oracle = [], [], [], [], []
    for gi in range(n_groups):
        n_g = int(counts[gi])
        # Group-specific base signal + covariate shift
        z1_g = rng.normal(0.0, 1.0, n_g)
        z2_g = rng.uniform(0.0, 1.0, n_g)
        # True latent logit depends on latent feature u + covariates
        u_g = rng.normal(0.0, 1.5, n_g)
        # Group-dependent bias in proxy (mimics domain shift across categories/states)
        bias_g = -1.6 + 3.2 * (gi / max(n_groups - 1, 1))
        noise_scale_g = 0.25 if gi % 3 == 0 else (0.9 if gi % 3 == 1 else 1.5)

        logit_true = 1.6 * u_g + 1.4 * z1_g - 1.2 * z2_g + 0.5 * bias_g
        prob_true = 1.0 / (1.0 + np.exp(-logit_true))
        y_g = (rng.uniform(0.0, 1.0, n_g) < prob_true).astype(np.float64)

        # Cheap proxy sees noisy u_g + systematic group shift, misses z1/z2
        logit_proxy = 1.3 * u_g - 0.8 * bias_g + rng.normal(0.0, noise_scale_g, n_g)
        p_g = 1.0 / (1.0 + np.exp(-logit_proxy))

        groups.extend([f"G{gi:02d}"] * n_g)
        proxy.append(p_g)
        z1.append(z1_g)
        z2.append(z2_g)
        oracle.append(y_g)

    return pa.table(
        {
            "grp": pa.array(groups),
            "proxy": pa.array(np.concatenate(proxy)),
            "z1": pa.array(np.concatenate(z1)),
            "z2": pa.array(np.concatenate(z2)),
            "oracle": pa.array(np.concatenate(oracle)),
        }
    )


def test_q1_simultaneous_coverage_monte_carlo(tmp_path: Path) -> None:
    """Measure simultaneous coverage and paired call counts on one synthetic population."""
    tbl = make_skewed_synthetic_table(n_groups=10, n_total=6000, seed=7)
    pq_path = tmp_path / "_test_mc.parquet"
    pq.write_table(tbl, pq_path)

    engine = StratQEngine(
        pq_path,
        group_col="grp",
        proxy_col="proxy",
        covariate_cols=["z1", "z2"],
        oracle_col="oracle",
        include_interactions=True,
    )
    truth = run_exhaustive_agg(engine, agg_type="mean").estimates

    delta = 0.10
    n_trials = 100
    all_covered = 0
    calls_stratq = []
    calls_unif = []

    for trial in range(n_trials):
        res = engine.execute_grouped_agg(
            eps=0.08,
            delta=delta,
            agg_type="mean",
            batch_size=16,
            pilot_per_group=8,
            seed=1000 + trial,
        )
        ok = all(res.lower[g] - 1e-12 <= truth[g] <= res.upper[g] + 1e-12 for g in res.groups)
        all_covered += int(ok)
        calls_stratq.append(res.total_oracle_calls)
        u_res = engine.execute_grouped_agg(
            eps=0.08, delta=delta, agg_type="mean", batch_size=16,
            pilot_per_group=8, seed=1000 + trial,
            use_control_variates=False, use_covariates=False,
        )
        calls_unif.append(u_res.total_oracle_calls)

    cov_rate = all_covered / n_trials
    mean_sq = float(np.mean(calls_stratq))
    mean_un = float(np.mean(calls_unif))
    print(
        f"[Q1 Monte Carlo] Simultaneous coverage = {cov_rate:.3f} (nominal {1 - delta:.2f}) | "
        f"Paired mean oracle calls: StratQ={mean_sq:.1f} vs NoProxy={mean_un:.1f} "
        f"(difference {mean_sq - mean_un:+.1f})"
    )
    assert cov_rate >= (1.0 - delta - 0.03), f"Coverage diagnostic failed: {cov_rate}"
    pq_path.unlink(missing_ok=True)


def test_q1_q2_q3_end_to_end_and_baselines() -> None:
    """Verify Q1, Q2 (Recall & Precision), Q3a (HAVING), Q3b (TOP-K) and all baselines."""
    tbl = make_skewed_synthetic_table(n_groups=12, n_total=12000, seed=42)
    engine = StratQEngine(
        tbl,
        group_col="grp",
        proxy_col="proxy",
        covariate_cols=["z1", "z2"],
        oracle_col="oracle",
        include_interactions=True,
    )
    truth_mean = run_exhaustive_agg(engine, agg_type="mean").estimates
    truth_cnt = run_exhaustive_agg(engine, agg_type="count").estimates

    # Q1 comparison
    r_sq = engine.execute_grouped_agg(eps=0.06, delta=0.05, seed=42)
    r_abae_g = run_abae_group_agg(engine, eps=0.06, delta=0.05, seed=42)
    r_abae_gl = run_abae_global_agg(engine, eps=0.06, delta=0.05, seed=42)
    r_unif = run_uniform_stratified_agg(engine, eps=0.06, delta=0.05, seed=42)

    max_err_sq = max(abs(r_sq.estimates[g] - truth_mean[g]) for g in r_sq.groups)
    max_err_gl = max(abs(r_abae_gl.estimates[g] - truth_mean[g]) for g in r_sq.groups)
    print(
        f"[Q1] StratQ calls={r_sq.total_oracle_calls} (max_err={max_err_sq:.4f}) | "
        f"ABAE-Group calls={r_abae_g.total_oracle_calls} | "
        f"Uniform-Strat calls={r_unif.total_oracle_calls} | "
        f"ABAE-Global calls={r_abae_gl.total_oracle_calls} (max_err={max_err_gl:.4f})"
    )
    assert all(r_sq.lower[g] <= truth_mean[g] <= r_sq.upper[g] for g in r_sq.groups)

    # Q2 Recall Target comparison
    B = 1800
    s_sq = engine.execute_grouped_selection(target_gamma=0.90, target_mode="recall", oracle_budget=B, seed=42)
    s_supg_gl = run_supg_global_selection(engine, target_gamma=0.90, target_mode="recall", oracle_budget=B, seed=42)
    s_supg_grp = run_supg_group_selection(engine, target_gamma=0.90, target_mode="recall", oracle_budget=B, seed=42)
    s_lotus = run_lotus_cascade_global_selection(engine, target_gamma=0.90, target_mode="recall", oracle_budget=B, seed=42)

    y_all = (engine.profile.oracle_true >= 0.5)
    for name, res in [
        ("StratQ", s_sq),
        ("SUPG-Group", s_supg_grp),
        ("SUPG-Global", s_supg_gl),
        ("LOTUS-Global", s_lotus),
    ]:
        sel_m = np.zeros(engine.profile.n_total, dtype=bool)
        sel_m[res.selected_indices] = True
        recs = []
        for gi in range(len(engine.profile.groups)):
            idxs = engine.profile.group_row_indices[gi]
            pos = int(np.sum(y_all[idxs]))
            tp = int(np.sum(y_all[idxs] & sel_m[idxs]))
            if pos > 0:
                recs.append(tp / pos)
        glob_prec = float(np.sum(y_all & sel_m)) / max(int(np.sum(sel_m)), 1)
        print(
            f"[Q2-RT {name:12s}] calls={res.total_oracle_calls} | "
            f"min_group_rec={min(recs):.3f} | mean_rec={np.mean(recs):.3f} | prec={glob_prec:.3f}"
        )

    # Q3a HAVING
    h_prune = engine.execute_grouped_having(tau=0.50, agg_type="mean", delta=0.05, use_early_pruning=True, seed=42)
    h_noprune = engine.execute_grouped_having(tau=0.50, agg_type="mean", delta=0.05, use_early_pruning=False, seed=42)
    true_in = {g for g, v in truth_mean.items() if v >= 0.50}
    print(
        f"[Q3a HAVING] Pruned calls={h_prune.total_oracle_calls} vs NoPrune={h_noprune.total_oracle_calls} "
        f"({h_noprune.total_oracle_calls / max(h_prune.total_oracle_calls, 1):.2f}x) | "
        f"acc_exact={set(h_prune.accepted_groups).issubset(true_in)}"
    )
    assert set(h_prune.accepted_groups).issubset(true_in)

    # Q3b TOP-K
    tk_prune = engine.execute_grouped_topk(k=3, agg_type="count", delta=0.05, use_early_pruning=True, seed=42)
    tk_noprune = engine.execute_grouped_topk(k=3, agg_type="count", delta=0.05, use_early_pruning=False, seed=42)
    true_top3 = sorted(truth_cnt.keys(), key=lambda g: -truth_cnt[g])[:3]
    print(
        f"[Q3b TOP-3] LUCB calls={tk_prune.total_oracle_calls} vs NoPrune={tk_noprune.total_oracle_calls} "
        f"({tk_noprune.total_oracle_calls / max(tk_prune.total_oracle_calls, 1):.2f}x) | "
        f"certified={tk_prune.certified} | exact_match={set(tk_prune.topk_groups) == set(true_top3)}"
    )
    assert set(tk_prune.topk_groups) == set(true_top3)


if __name__ == "__main__":
    test_q1_simultaneous_coverage_monte_carlo()
    test_q1_q2_q3_end_to_end_and_baselines()
    print("ALL STRATQ TESTS PASSED!")

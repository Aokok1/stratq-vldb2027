"""Sprint 2 systems hard evaluations: proxy sweep, Zipf stress, profile amortization, RSS.

Default run writes results/sprint2_systems.json.  ``--smoke`` reduces sizes/seeds
for interface validation only; smoke numbers are not the full study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import tracemalloc
from pathlib import Path
from time import perf_counter

import duckdb
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stratq.engine import StratQEngine
from stratq import modern_baselines as mb

CODE_FILES = ("src/stratq/bounds.py", "src/stratq/engine.py", "src/stratq/modern_baselines.py")


def code_hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in CODE_FILES}


def zipf_sizes(n: int, g: int, alpha: float, min_size: int = 8):
    weights = np.arange(1, g + 1, dtype=float) ** (-alpha)
    weights /= weights.sum()
    sizes = np.maximum(min_size, np.round(weights * n).astype(int))
    # Preserve total exactly by adjusting the largest group.
    sizes[0] += n - int(sizes.sum())
    if sizes[0] < min_size:
        raise ValueError("n too small for requested groups")
    return sizes


def make_synthetic(n: int, sizes: np.ndarray, seed: int = 1729):
    rng = np.random.default_rng(seed)
    g = len(sizes)
    prevalence = np.linspace(0.12, 0.62, g)
    groups = np.repeat(np.arange(g), sizes)
    latent = rng.random(n)
    labels = (latent < prevalence[groups]).astype(np.uint8)
    return groups, labels, prevalence


def proxy_with_target_auc(labels: np.ndarray, target_auc: float, seed: int = 99):
    """Construct a synthetic proxy and binary-search its signal strength."""
    rng = np.random.default_rng(seed)
    noise = rng.uniform(size=len(labels))
    lo, hi = 0.0, 1.0
    best = noise.copy()
    for _ in range(40):
        mid = (lo + hi) / 2
        score = np.clip(mid * labels + (1 - mid) * noise, 0, 1)
        auc = roc_auc_score(labels, score)
        best = score
        if auc < target_auc:
            lo = mid
        else:
            hi = mid
    return best, float(roc_auc_score(labels, best))


def coverage_summary(result, truth, sizes, tail_frac=0.2):
    groups = result.groups if hasattr(result, "groups") else list(truth)
    if hasattr(result, "group_lower"):
        lower, upper = result.group_lower, result.group_upper
    else:
        lower, upper = result.lower, result.upper
    covered = {g: bool(lower[g] - 1e-9 <= truth[g] <= upper[g] + 1e-9) for g in groups}
    order = sorted(groups, key=lambda g: sizes[g])
    tail = order[:max(1, int(math.ceil(tail_frac * len(order))))]
    return {
        "covered_groups": sum(covered.values()), "total_groups": len(groups),
        "all_covered": all(covered.values()),
        "tail_groups": tail, "tail_covered": sum(covered[g] for g in tail),
        "per_group_covered": covered,
    }


def run_proxy_sweep(smoke=False):
    n = 4000 if smoke else 12000
    sizes = zipf_sizes(n, 6 if smoke else 8, 1.0)
    groups, labels, prevalence = make_synthetic(n, sizes)
    indices = [np.flatnonzero(groups == g) for g in range(len(sizes))]
    names = [f"G{g}" for g in range(len(sizes))]
    truth = {name: int(labels[idx].sum()) for name, idx in zip(names, indices)}
    eps = {name: 0.04 * len(idx) for name, idx in zip(names, indices)}
    targets = [0.95, 0.75, 0.50] if smoke else [0.95, 0.85, 0.75, 0.65, 0.55, 0.50]
    seeds = (11,) if smoke else (11, 12, 13)
    runs = []
    for target in targets:
        proxy, auc = proxy_with_target_auc(labels, target)
        table = pa.table({"grp": np.asarray(names)[groups], "proxy": proxy, "label": labels})
        engines = {
            "no_proxy": StratQEngine(table, group_col="grp", proxy_col="proxy", oracle_col="label",
                                     gating_mode="safe_loo", safe_rounding=True),
            "legacy": StratQEngine(table, group_col="grp", proxy_col="proxy", oracle_col="label",
                                   gating_mode="legacy", safe_rounding=True),
            "safe_loo": StratQEngine(table, group_col="grp", proxy_col="proxy", oracle_col="label",
                                     gating_mode="safe_loo", safe_rounding=True),
        }
        for seed in seeds:
            for method in ("no_proxy", "legacy", "safe_loo"):
                r = engines[method].execute_grouped_agg(
                    eps=eps, delta=0.1, agg_type="count", batch_size=32, pilot_per_group=8,
                    seed=seed, use_control_variates=method != "no_proxy", use_covariates=False)
                runs.append({"target_auc": target, "actual_auc": auc, "method": method, "seed": seed,
                             "calls": r.total_oracle_calls, "wall_ms": r.wall_ms,
                             "optimizer_ms": r.optimizer_ms,
                             **coverage_summary(r, truth, dict(zip(names, sizes)))})
            r = mb.calibrated_fixed_proxy_count(indices, proxy, labels, eps, seed=seed,
                                                batch_size=32, pilot_per_group=8, group_names=names)
            runs.append({"target_auc": target, "actual_auc": auc, "method": "Calibrated-FixedProxy",
                         "seed": seed, "calls": r.total_oracle_calls, "wall_ms": r.wall_ms,
                         "optimizer_ms": r.optimizer_ms, **coverage_summary(r, truth, dict(zip(names, sizes)))})
            r = mb.celeus_fixed_proxy_count(indices, proxy, labels, eps, seed=seed,
                                            batch_size=32, pilot_per_group=8, group_names=names)
            runs.append({"target_auc": target, "actual_auc": auc, "method": "CELEUS-FixedProxy",
                         "seed": seed, "calls": r.total_oracle_calls, "wall_ms": r.wall_ms,
                         "optimizer_ms": r.optimizer_ms, **coverage_summary(r, truth, dict(zip(names, sizes)))})
    summary = {}
    for target in targets:
        for method in sorted({r["method"] for r in runs}):
            subset = [r for r in runs if r["target_auc"] == target and r["method"] == method]
            summary[f"{target:.2f}|{method}"] = {
                "actual_auc": subset[0]["actual_auc"], "trials": len(subset),
                "mean_calls": float(np.mean([r["calls"] for r in subset])),
                "all_covered_trials": sum(r["all_covered"] for r in subset),
                "mean_optimizer_ms": float(np.mean([r["optimizer_ms"] for r in subset])),
            }
    return {"n": n, "groups": len(sizes), "sizes": sizes.tolist(), "targets": targets,
            "seeds": list(seeds), "summary": summary, "runs": runs}


def run_zipf_stress(smoke=False):
    n = 10000 if smoke else 100000
    group_grid = (5, 10) if smoke else (5, 10, 20, 50, 100)
    alphas = (0.0, 1.2) if smoke else (0.0, 0.8, 1.2)
    seeds = (21,) if smoke else (21, 22)
    runs = []
    for g in group_grid:
        for alpha in alphas:
            sizes = zipf_sizes(n, g, alpha)
            groups, labels, _ = make_synthetic(n, sizes, seed=1729 + g)
            proxy, auc = proxy_with_target_auc(labels, 0.85, seed=99 + g)
            names = [f"G{i}" for i in range(g)]
            indices = [np.flatnonzero(groups == i) for i in range(g)]
            truth = {name: int(labels[idx].sum()) for name, idx in zip(names, indices)}
            eps = {name: 0.05 * len(idx) for name, idx in zip(names, indices)}
            table = pa.table({"grp": np.asarray(names)[groups], "proxy": proxy, "label": labels})
            engine = StratQEngine(table, group_col="grp", proxy_col="proxy", oracle_col="label")
            for seed in seeds:
                r = engine.execute_grouped_agg(eps=eps, delta=0.1, agg_type="count", batch_size=128,
                                               pilot_per_group=16, seed=seed, use_covariates=False)
                runs.append({"G": g, "alpha": alpha, "method": "StratQ-safe_loo", "seed": seed,
                             "calls": r.total_oracle_calls, "optimizer_ms": r.optimizer_ms,
                             **coverage_summary(r, truth, dict(zip(names, sizes)))})
                r = mb.spertus_wor_count(indices, proxy, labels, eps, seed=seed, batch_size=128,
                                         pilot_per_group=16, group_names=names)
                runs.append({"G": g, "alpha": alpha, "method": "Spertus-WoR", "seed": seed,
                             "calls": r.total_oracle_calls, "optimizer_ms": r.optimizer_ms,
                             **coverage_summary(r, truth, dict(zip(names, sizes)))})
                r = mb.bargain_global_count(indices, proxy, labels, eps_global=0.05 * n, seed=seed,
                                            batch_size=128, pilot=16, group_names=names)
                runs.append({"G": g, "alpha": alpha, "method": "BARGAIN-Global", "seed": seed,
                             "calls": r.total_oracle_calls, "optimizer_ms": r.wall_ms,
                             **coverage_summary(r, truth, dict(zip(names, sizes)))})
                r = mb.pilotdb_asymp_count(indices, labels, eps, seed=seed, pilot_per_group=16,
                                           optional_stopping=True, group_names=names)
                runs.append({"G": g, "alpha": alpha, "method": "PilotDB-Asymp", "seed": seed,
                             "calls": r.total_oracle_calls, "optimizer_ms": r.optimizer_ms,
                             **coverage_summary(r, truth, dict(zip(names, sizes)))})
    summary = {}
    for g in group_grid:
        for alpha in alphas:
            for method in sorted({r["method"] for r in runs}):
                subset = [r for r in runs if r["G"] == g and r["alpha"] == alpha and r["method"] == method]
                summary[f"G{g}|a{alpha}|{method}"] = {
                    "trials": len(subset), "mean_calls": float(np.mean([r["calls"] for r in subset])),
                    "mean_tail_covered": float(np.mean([r["tail_covered"] for r in subset])),
                    "all_covered_trials": sum(r["all_covered"] for r in subset),
                    "mean_optimizer_ms": float(np.mean([r["optimizer_ms"] for r in subset])),
                }
    return {"n": n, "group_grid": list(group_grid), "alphas": list(alphas), "seeds": list(seeds),
            "summary": summary, "runs": runs}


def breakeven_q(fresh_cumulative, reused_cumulative):
    for q, (f, r) in enumerate(zip(fresh_cumulative, reused_cumulative), start=1):
        if r < f:
            return q
    return None


def run_profile_amortization(smoke=False):
    scales = (100000,) if smoke else (1000000, 10000000)
    qs = (1, 2, 5) if smoke else (1, 2, 5, 10, 20)
    records = []
    for n in scales:
        path = ROOT / "data" / f"scale_{n}.parquet"
        if not path.exists():
            records.append({"rows": n, "status": "missing_parquet"})
            continue
        repeats = 1 if n >= 10000000 else (1 if smoke else 2)
        measure_q = min(max(qs), 5)
        for rep in range(repeats):
            fresh_times, reused_times, exact_times = [], [], []
            warm_engine = None
            profile_once_ms = 0.0
            for q in range(1, measure_q + 1):
                eps_rate = 0.03 + 0.005 * ((q - 1) % 5)
                # Fresh profile every query.
                start = perf_counter()
                engine = StratQEngine(path, group_col="grp", proxy_col="proxy", oracle_col="label")
                sizes = dict(zip(engine.profile.groups, engine.profile.n_by_group.tolist()))
                eps = {g: eps_rate * int(s) for g, s in sizes.items()}
                r = engine.execute_grouped_agg(eps=eps, delta=0.1, agg_type="count", batch_size=128,
                                               pilot_per_group=32, seed=3000 + q, use_covariates=False)
                fresh_times.append((perf_counter() - start) * 1000)
                # Reused profile.
                if warm_engine is None:
                    start = perf_counter()
                    warm_engine = StratQEngine(path, group_col="grp", proxy_col="proxy", oracle_col="label")
                    profile_ms = (perf_counter() - start) * 1000
                    profile_once_ms = profile_ms
                else:
                    profile_ms = 0.0
                start = perf_counter()
                r2 = warm_engine.execute_grouped_agg(eps=eps, delta=0.1, agg_type="count", batch_size=128,
                                                     pilot_per_group=32, seed=3000 + q, use_covariates=False)
                reused_times.append(profile_ms + (perf_counter() - start) * 1000)
                # Exact SQL verification query.
                start = perf_counter()
                with duckdb.connect() as con:
                    con.read_parquet(str(path)).aggregate("grp, sum(label)", "grp").fetchall()
                exact_times.append((perf_counter() - start) * 1000)
            fresh_med = float(np.median(fresh_times))
            reused_med = float(np.median(reused_times))
            exact_med = float(np.median(exact_times))
            query_list = list(range(1, max(qs) + 1))
            fresh_cum = [fresh_med * q for q in query_list]
            reused_cum = [profile_once_ms + reused_med * q for q in query_list]
            exact_cum = [exact_med * q for q in query_list]
            records.append({
                "rows": n, "replicate": rep, "queries": query_list,
                "measured_queries": measure_q,
                "extrapolated_beyond_measured": max(qs) > measure_q,
                "fresh_ms": fresh_times, "reused_ms": reused_times, "exact_ms": exact_times,
                "profile_once_ms": profile_once_ms,
                "median_fresh_ms": fresh_med, "median_reused_ms": reused_med,
                "median_exact_ms": exact_med,
                "fresh_cumulative_ms": fresh_cum, "reused_cumulative_ms": reused_cum,
                "exact_cumulative_ms": exact_cum,
                "breakeven_vs_fresh": breakeven_q(fresh_cum, reused_cum),
                "breakeven_vs_exact": breakeven_q(exact_cum, reused_cum),
            })
    return {"records": records}


def memory_stage(stage: str, rows: int, data_path: Path | None = None):
    """Run one isolated stage and return JSON-serializable memory measurements."""
    path = Path(data_path) if data_path is not None else ROOT / "data" / f"scale_{rows}.parquet"
    proc = psutil.Process()
    tracemalloc.start()
    start = perf_counter()
    extra = {}
    if stage == "generate":
        tmp = ROOT / "data" / f"_sprint2_memory_{rows}.parquet"
        from scale_count_replay import generate
        truth = generate(tmp, rows)
        extra["generated_bytes"] = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
    elif stage == "verify":
        with duckdb.connect() as con:
            rows_sql = con.read_parquet(str(path)).aggregate("grp, count(*), sum(label)", "grp").fetchall()
        extra["sql_groups"] = len(rows_sql)
    elif stage == "profile":
        engine = StratQEngine(path, group_col="grp", proxy_col="proxy", oracle_col="label")
        extra["profile_ms"] = engine.profile.duckdb_scan_ms
        extra["groups"] = len(engine.profile.groups)
    elif stage == "query":
        engine = StratQEngine(path, group_col="grp", proxy_col="proxy", oracle_col="label")
        rss_after_profile = proc.memory_info().rss
        tracemalloc.stop(); tracemalloc.start()
        sizes = dict(zip(engine.profile.groups, engine.profile.n_by_group.tolist()))
        eps = {g: 0.04 * int(s) for g, s in sizes.items()}
        r = engine.execute_grouped_agg(eps=eps, delta=0.1, agg_type="count", batch_size=128,
                                       pilot_per_group=32, seed=42, use_covariates=False)
        extra["calls"] = r.total_oracle_calls
        extra["query_ms"] = r.wall_ms
        extra["rss_after_profile_bytes"] = rss_after_profile
        extra["rss_query_delta_bytes"] = proc.memory_info().rss - rss_after_profile
    else:
        raise ValueError(stage)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem = proc.memory_info()
    return {
        "stage": stage, "rows": rows, "wall_ms": (perf_counter() - start) * 1000,
        "tracemalloc_peak_bytes": peak,
        "peak_working_set_bytes": getattr(mem, "peak_wset", None),
        "rss_bytes": mem.rss, **extra,
    }


def run_memory_breakdown(rows=10000000):
    records = []
    for stage in ("generate", "verify", "profile", "query"):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__)), "--memory-stage", stage, "--rows", str(rows)],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        if completed.returncode != 0:
            records.append({"stage": stage, "error": completed.stderr[-2000:]})
        else:
            records.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    return {"rows": rows, "stages": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--memory-stage", choices=("generate", "verify", "profile", "query"))
    parser.add_argument("--rows", type=int, default=10000000)
    parser.add_argument("--skip-memory", action="store_true")
    args = parser.parse_args()
    if args.memory_stage:
        print(json.dumps(memory_stage(args.memory_stage, args.rows), allow_nan=False))
        return
    with threadpool_limits(limits=1):
        report = {
            "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "smoke": args.smoke, "code_sha256": code_hashes(),
            "host": {"platform": platform.platform(), "python": platform.python_version(),
                     "cpu_count": os.cpu_count(), "ram_bytes": psutil.virtual_memory().total},
            "proxy_quality_sweep": run_proxy_sweep(args.smoke),
            "zipf_stress": run_zipf_stress(args.smoke),
            "profile_amortization": run_profile_amortization(args.smoke),
        }
        if not args.skip_memory:
            report["memory_10m"] = run_memory_breakdown(100000 if args.smoke else 10000000)
    out = ROOT / "results" / ("sprint2_systems_smoke.json" if args.smoke else "sprint2_systems.json")
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

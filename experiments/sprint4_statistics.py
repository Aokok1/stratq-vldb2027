"""Recompute Sprint 4 descriptive statistics directly from frozen result JSON.

This script never edits the source measurements.  It writes a separate derived
record with sample standard deviations, ranges, coverage denominators, timing
boundaries, and the exact definitions used by the manuscript.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str):
    raw = (RESULTS / f"{name}.json").read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def summary(values):
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "std_sample": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "range": max(vals) - min(vals),
        "q1": float(__import__("numpy").quantile(vals, 0.25)),
        "q3": float(__import__("numpy").quantile(vals, 0.75)),
    }


def group_stats(records, key_fn, value_fns):
    groups = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record)
    out = {}
    for key, rows in groups.items():
        out[key] = {"n": len(rows)}
        for name, fn in value_fns.items():
            out[key][name] = summary(fn(r) for r in rows)
    return out


def main():
    hashes = {}

    def record(name):
        obj, digest = load(name)
        hashes[name] = digest
        return obj

    gpu = record("sprint2_gpu_workloads")
    systems = record("sprint2_systems")
    memory = record("sprint2_memory_10m")
    s1 = record("sprint1_safe_gating_ablation")
    decisions = record("decision_boundaries")
    routing = record("adult_routing_capped")
    hyper = record("adult_hypergeom")

    out = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashes,
        "definitions": {
            "std": "sample standard deviation (ddof=1); descriptive only",
            "coverage": "observed covered runs / total runs on fixed snapshots",
            "max_hw": "mean over runs of each run's largest per-group normalized half-width",
            "gpu_latency": "ratio of medians; proxy scores precomputed; GPU model and OS cache warm",
            "memory_delta": "RSS immediately after query minus RSS after profile, before tracemalloc teardown",
        },
    }

    # GPU COUNT and decision statistics.
    out["gpu_count"] = {}
    out["gpu_decisions"] = {}
    out["gpu_latency"] = {}
    for workload in gpu["workloads"]:
        name = workload["workload"]
        out["gpu_count"][name] = group_stats(
            workload["count_runs"], lambda r: r["method"],
            {"calls": lambda r: r["calls"], "wall_ms": lambda r: r["wall_ms"],
             "inference_ms": lambda r: r["inference_ms"],
             "mean_normalized_half_width": lambda r: r["mean_normalized_half_width"]},
        )
        for method, rows in out["gpu_count"][name].items():
            rows["covered_runs"] = sum(r["all_covered"] for r in workload["count_runs"] if r["method"] == method)
            rows["total_runs"] = rows["n"]
        out["gpu_decisions"][name] = group_stats(
            workload["decision_runs"], lambda r: f'{r["query"]}|{r["method"]}',
            {"calls": lambda r: r["calls"]},
        )
        for key, rows in out["gpu_decisions"][name].items():
            query, method = key.split("|")
            subset = [r for r in workload["decision_runs"] if r["query"] == query and r["method"] == method]
            if query == "HAVING":
                rows["fully_resolved_runs"] = sum(not r["uncertain"] for r in subset)
                rows["correct_runs"] = sum(r["decision_correct"] for r in subset)
            else:
                rows["certified_runs"] = sum(r["certified"] for r in subset)
                rows["correct_runs"] = sum(r["decision_correct"] for r in subset)
        lat = workload["latency"]
        exact = lat["median_exact_ms"]
        fresh = lat["median_fresh_ms"]
        reused = lat["median_reused_query_ms"]
        rec = {
            "repeats": len(lat["fresh_runs"]),
            "query_seeds": list(range(5000, 5000 + len(lat["fresh_runs"]))),
            "tolerance_schedule": [0.03 + 0.01 * (i % 3) for i in range(len(lat["fresh_runs"]))],
            "exact_ms": summary(r["wall_ms"] for r in lat["exact_runs"] if r["batch_size"] == lat["best_exact_batch"]),
            "fresh_ms": summary(r["wall_ms"] for r in lat["fresh_runs"]),
            "reused_ms": summary(r["query_ms"] for r in lat["reused_runs"]),
            "speedup_fresh_ratio_of_medians": exact / fresh,
            "speedup_reused_ratio_of_medians": exact / reused,
            "per_repeat_speedup_fresh": [lat["median_exact_ms"] / r["wall_ms"] for r in lat["fresh_runs"]],
            "per_repeat_speedup_reused": [lat["median_exact_ms"] / r["query_ms"] for r in lat["reused_runs"]],
            "best_exact_batch": lat["best_exact_batch"],
            "proxy_train_ms": workload.get("proxy_train_ms"),
            "proxy_scan_ms": workload.get("proxy_scan_ms"),
            "count_wall_excludes_profile": True,
            "fresh_means_new_tableprofile_not_cold_process": True,
        }
        if rec["proxy_scan_ms"] is not None:
            rec["speedup_fresh_including_proxy_scan"] = exact / (fresh + rec["proxy_scan_ms"])
            rec["speedup_reused_including_proxy_scan"] = exact / (reused + rec["proxy_scan_ms"])
        if rec["proxy_train_ms"] is not None:
            rec["speedup_first_query_including_proxy_train_and_scan"] = exact / (
                fresh + rec["proxy_train_ms"] + (rec["proxy_scan_ms"] or 0.0)
            )
        out["gpu_latency"][name] = rec

    # Sprint 1 ablation, including the omitted SMS legacy arm.
    out["sprint1_ablation"] = {}
    blocks = [("Adult COUNT", s1["adult"]["runs"])]
    for query, block in s1["sms"]["workloads"].items():
        blocks.append((f"SMS {query}", block["runs"]))
    blocks.append(("Fashion COUNT", s1["fashion_mnist"]["runs"]))
    for label, runs in blocks:
        out["sprint1_ablation"][label] = group_stats(
            runs, lambda r: r["mode"],
            {"calls": lambda r: r["calls"],
             "mean_normalized_half_width": lambda r: r["mean_normalized_half_width"],
             "max_normalized_half_width": lambda r: r["max_normalized_half_width"]},
        )
        for mode, rows in out["sprint1_ablation"][label].items():
            subset = [r for r in runs if r["mode"] == mode]
            rows["covered_runs"] = sum(r["covered"] for r in subset)
            rows["target_met_runs"] = sum(r.get("target_met", r.get("decision_correct", False)) for r in subset)

    # Proxy sweep and Zipf denominators.
    ps = systems["proxy_quality_sweep"]["runs"]
    out["proxy_sweep"] = group_stats(
        ps, lambda r: f'{r["target_auc"]:.2f}|{r["method"]}',
        {"calls": lambda r: r["calls"]},
    )
    for key, rows in out["proxy_sweep"].items():
        target, method = key.split("|")
        subset = [r for r in ps if f'{r["target_auc"]:.2f}' == target and r["method"] == method]
        rows["covered_runs"] = sum(r["all_covered"] for r in subset)
    zr = systems["zipf_stress"]["runs"]
    configurations = len({(r["G"], r["alpha"]) for r in zr})
    out["zipf"] = {
        "configurations": configurations,
        "runs_per_method": len({r["seed"] for r in zr if r["method"] == "StratQ-safe_loo"}) * configurations,
        "methods": {},
    }
    for method in sorted({r["method"] for r in zr}):
        subset = [r for r in zr if r["method"] == method]
        tail_total = sum(len(r["tail_groups"]) for r in subset)
        out["zipf"]["methods"][method] = {
            "runs": len(subset),
            "all_group_covered_runs": sum(r["all_covered"] for r in subset),
            "tail_covered_instances": sum(r["tail_covered"] for r in subset),
            "tail_total_instances": tail_total,
            "tail_coverage_fraction": sum(r["tail_covered"] for r in subset) / tail_total,
            "g100_all_group_covered_runs": sum(r["all_covered"] for r in subset if r["G"] == 100),
            "g100_runs": sum(r["G"] == 100 for r in subset),
            "calls": summary(r["calls"] for r in subset),
        }
    # 14/15 mean-call comparison at configuration level.
    cells = sorted({(r["G"], r["alpha"]) for r in zr})
    wins = 0
    for G, alpha in cells:
        means = {}
        for method in ("StratQ-safe_loo", "Spertus-WoR"):
            means[method] = statistics.mean(r["calls"] for r in zr if r["G"] == G and r["alpha"] == alpha and r["method"] == method)
        wins += means["StratQ-safe_loo"] < means["Spertus-WoR"]
    out["zipf"]["stratq_mean_call_wins_vs_spertus"] = f"{wins}/{len(cells)}"

    # Classical Adult comparisons.
    out["adult_classical"] = {}
    for name in ("adult_fixed_budget", "adult_blazeit", "adult_senate", "adult_online_aggregation", "adult_lws", "adult_verdict"):
        data = record(name)
        truth = data["truth"]
        rows = [r for r in data["runs"] if r["budget"] == 4096]
        stats = group_stats(rows, lambda r: r.get("method", name),
                            {"worst_group_half_width_pp": lambda r: 100 * max(
                                (v["upper"] - v["lower"]) / (2 * truth[g]["size"]) for g, v in r["groups"].items())})
        for method, item in stats.items():
            subset = [r for r in rows if r.get("method", name) == method]
            item["covered_runs"] = sum(all(r["groups"][g]["lower"] <= truth[g]["positives"] <= r["groups"][g]["upper"] for g in truth) for r in subset)
        out["adult_classical"][name] = stats
    out["adult_hypergeometric"] = {
        "calls": summary(r["calls"] for r in hyper["runs"]),
        "covered_runs": sum(r["covered"] for r in hyper["runs"]),
        "total_runs": len(hyper["runs"]),
    }

    # Decision-boundary correctness and denominators.
    db = decisions["runs"]
    wrong_cert = 0
    wrong_having = 0
    interval_failures = 0
    topk_cert = 0
    topk_cert_correct = 0
    for r in db:
        truth = decisions["populations"][str(r["gap"])]
        interval_failures += int(not all(r["lower"][g] <= truth[g] <= r["upper"][g] for g in truth))
        if r["query"] == "having":
            wrong = any(truth[g] < decisions["threshold"] for g in r["accepted"]) or any(
                truth[g] >= decisions["threshold"] for g in r["rejected"])
            wrong_having += int(wrong)
        else:
            expected = sorted(truth, key=lambda g: (-truth[g], g))[:decisions["k"]]
            correct = set(r["selected"]) == set(expected)
            topk_cert += int(r["certified"])
            topk_cert_correct += int(r["certified"] and correct)
            wrong_cert += int(r["certified"] and not correct)
    out["decision_boundaries"] = {
        "design": "4 gaps x 3 budgets x 20 seeds x 2 proxy settings x 2 queries",
        "runs": len(db), "interval_coverage_failures": interval_failures,
        "wrong_having_labels": wrong_having, "wrong_certificates": wrong_cert,
        "topk_certified_runs": topk_cert, "topk_certified_correct_runs": topk_cert_correct,
        "positive_gap_topk_budget768": {
            "certified": sum(r["certified"] for r in db if r["query"] == "topk" and r["budget"] == 768 and r["gap"] > 0),
            "total": sum(r["query"] == "topk" and r["budget"] == 768 and r["gap"] > 0 for r in db),
        },
        "tied_topk_census": {
            "certified": sum(r["certified"] for r in db if r["query"] == "topk" and r["budget"] == 1024 and r["gap"] == 0),
            "total": sum(r["query"] == "topk" and r["budget"] == 1024 and r["gap"] == 0 for r in db),
        },
    }

    # Capped routing matched pairs.
    pairs = defaultdict(dict)
    truth = routing["truth"]
    for r in routing["runs"]:
        key = (r["budget"], r["seed"], r["proxy"])
        max_hw = 100 * max((v["upper"] - v["lower"]) / (2 * truth[g]["size"]) for g, v in r["groups"].items())
        below = sum((v["upper"] - v["lower"]) / 2 <= truth[g]["epsilon"] + 1e-9 for g, v in r["groups"].items())
        pairs[key][r["adaptive"]] = {"max_hw_pp": max_hw, "groups_below_target": below}
    strict = sum(v[True]["max_hw_pp"] < v[False]["max_hw_pp"] for v in pairs.values() if True in v and False in v)
    b4096 = [v for (budget, _, proxy), v in pairs.items() if budget == 4096 and proxy]
    out["capped_routing"] = {
        "matched_pairs": len(pairs), "strict_width_improvements": strict,
        "budget4096_proxy_mean_max_hw_pp": {
            "round_robin": statistics.mean(v[False]["max_hw_pp"] for v in b4096),
            "width_first": statistics.mean(v[True]["max_hw_pp"] for v in b4096),
            "round_robin_groups_below_target": statistics.mean(v[False]["groups_below_target"] for v in b4096),
            "width_first_groups_below_target": statistics.mean(v[True]["groups_below_target"] for v in b4096),
        },
    }

    # Memory definitions and modeled-vs-measured amortization.
    stages = {s["stage"]: s for s in memory["stages"]}
    q = stages["query"]
    p = stages["profile"]
    out["memory_10m"] = {
        "query_subprocess_total_wall_ms": q["wall_ms"],
        "query_only_wall_ms": q["query_ms"],
        "query_stage_minus_query_only_ms": q["wall_ms"] - q["query_ms"],
        "profile_stage_wall_ms": p["wall_ms"],
        "rss_after_profile_bytes": q["rss_after_profile_bytes"],
        "rss_immediately_after_query_bytes": q["rss_after_profile_bytes"] + q["rss_query_delta_bytes"],
        "rss_query_delta_bytes": q["rss_query_delta_bytes"],
        "rss_final_after_tracemalloc_teardown_bytes": q["rss_bytes"],
        "rss_final_minus_after_profile_bytes": q["rss_bytes"] - q["rss_after_profile_bytes"],
        "query_tracemalloc_peak_bytes": q["tracemalloc_peak_bytes"],
        "query_subprocess_peak_wset_includes_profile": True,
        "oracle_mode": "offline boolean-label replay",
        "repeats": 1,
    }
    rec10 = next(r for r in systems["profile_amortization"]["records"] if r["rows"] == 10_000_000)
    out["profile_amortization_10m"] = {
        "profile_once_ms": rec10["profile_once_ms"],
        "fresh_query_median_ms": rec10["median_fresh_ms"],
        "reused_query_median_ms": rec10["median_reused_ms"],
        "measured_queries": rec10["measured_queries"],
        "extrapolated_beyond_measured": rec10["extrapolated_beyond_measured"],
        "modeled_breakeven_vs_fresh": rec10["breakeven_vs_fresh"],
        "modeled_speedup_at_q10": rec10["fresh_cumulative_ms"][9] / rec10["reused_cumulative_ms"][9],
        "oracle_mode": "offline boolean-label replay",
    }

    cpu = record("sprint4_cpu_latency_aligned")
    out["cpu_latency_aligned"] = {}
    for dataset in cpu["datasets"]:
        exact_best = min((r for r in dataset["summary"] if r["mode"] == "exact"),
                         key=lambda r: r["median_wall_ms"])
        fresh = next(r for r in dataset["summary"] if r["mode"] == "fresh_proxy")
        reused = next(r for r in dataset["summary"] if r["mode"] == "reuse_proxy")
        out["cpu_latency_aligned"][dataset["dataset"]] = {
            "exact_best_batch": exact_best["batch"],
            "exact_best_median_wall_ms": exact_best["median_wall_ms"],
            "fresh_proxy_median_wall_ms": fresh["median_wall_ms"],
            "reused_proxy_median_wall_ms": reused["median_wall_ms"],
            "fresh_speedup": exact_best["median_wall_ms"] / fresh["median_wall_ms"],
            "reused_speedup": exact_best["median_wall_ms"] / reused["median_wall_ms"],
            "measured_blocks": 10,
            "model_fitting_and_data_loading_excluded": True,
        }

    # Current code hashes are provenance for this derived audit, not a claim
    # that frozen measurements were regenerated with the current code.
    out["current_code_sha256"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "src/stratq/bounds.py", ROOT / "src/stratq/engine.py", ROOT / "src/stratq/modern_baselines.py")
    }
    target = RESULTS / "sprint4_summary_statistics.json"
    target.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()

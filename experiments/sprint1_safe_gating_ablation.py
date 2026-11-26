"""Sprint 1 matched ablation: legacy vs safe-LOO vs no-proxy.

All three arms use outward binary64 intervals, identical row permutations,
batch sizes, risk budgets, tolerances and routing.  Only proxy selection changes.
The script never overwrites the frozen Adult/SMS/Fashion result files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from stratq.engine import StratQEngine
from adult_count_replay import records as adult_records
from sms_live import LENGTH_CUTS, SEEDS as SMS_SEEDS, prepare as prepare_sms
from fashion_live import SEEDS as FASHION_SEEDS, pooled, prepare as prepare_fashion

MODES = ("legacy", "safe_loo", "no_proxy")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_engines(table, *, group_col, proxy_col, covariate_cols, oracle_col=None, oracle_fn=None):
    engines = {}
    for mode in MODES:
        engines[mode] = StratQEngine(
            table, group_col=group_col, proxy_col=proxy_col,
            covariate_cols=covariate_cols, oracle_col=oracle_col, oracle_fn=oracle_fn,
            gating_mode="safe_loo" if mode == "no_proxy" else mode,
            safe_rounding=True,
        )
    return engines


def summarize_runs(runs):
    base = {r["seed"]: r["calls"] for r in runs if r["mode"] == "no_proxy"}
    summary = {}
    for mode in MODES:
        subset = [r for r in runs if r["mode"] == mode]
        calls = [r["calls"] for r in subset]
        item = {
            "trials": len(subset),
            "mean_calls": float(np.mean(calls)),
            "median_calls": float(np.median(calls)),
            "call_range": [min(calls), max(calls)],
            "mean_paired_call_delta_vs_no_proxy": float(np.mean(
                [r["calls"] - base[r["seed"]] for r in subset]
            )),
            "covered_trials": sum(r["covered"] for r in subset),
            "coverage_rate": float(np.mean([r["covered"] for r in subset])),
            "mean_normalized_half_width": float(np.mean(
                [r["mean_normalized_half_width"] for r in subset]
            )),
            "mean_max_normalized_half_width": float(np.mean(
                [r["max_normalized_half_width"] for r in subset]
            )),
            "mean_count_interval_width": float(np.mean(
                [r["mean_count_interval_width"] for r in subset]
            )),
        }
        for field in ("target_met", "decision_correct", "certified"):
            values = [r[field] for r in subset if field in r]
            if values:
                item[field + "_trials"] = sum(bool(v) for v in values)
                item[field + "_rate"] = float(np.mean([bool(v) for v in values]))
        summary[mode] = item
    return summary


def record_run(mode, seed, result, truth, sizes, **decision):
    groups = {}
    normalized_widths = []
    count_widths = []
    covered = True
    for group in result.groups:
        lo, hi = float(result.lower[group]), float(result.upper[group])
        true_count = float(truth[group])
        covered &= lo - 1e-9 <= true_count <= hi + 1e-9
        width = hi - lo
        count_widths.append(width)
        normalized_widths.append(width / (2.0 * sizes[group]))
        groups[group] = {
            "lower": lo, "upper": hi, "estimate": float(result.estimates[group]),
            "calls": int(result.oracle_calls_by_group[group]),
            "normalized_half_width": width / (2.0 * sizes[group]),
        }
    return {
        "mode": mode, "seed": seed, "calls": int(result.total_oracle_calls),
        "covered": bool(covered), "groups": groups,
        "mean_normalized_half_width": float(np.mean(normalized_widths)),
        "max_normalized_half_width": float(np.max(normalized_widths)),
        "mean_count_interval_width": float(np.mean(count_widths)),
        **decision,
    }


def run_adult(trials):
    archive_path = ROOT / "data/adult_uci.zip"
    train_counts = {}
    with ZipFile(archive_path) as archive:
        for row in adult_records(archive, "adult.data"):
            counts = train_counts.setdefault(int(row[4]), [0, 0])
            counts[0] += int(row[14] == ">50K")
            counts[1] += 1
        test = list(adult_records(archive, "adult.test"))
    if sum(n for _, n in train_counts.values()) != 32561 or len(test) != 16281:
        raise ValueError("Unexpected Adult split")
    education = [int(row[4]) for row in test]
    proxy = [(train_counts[e][0] + 1) / (train_counts[e][1] + 2) for e in education]
    table = pa.table({
        "grp": [row[1] for row in test], "proxy": proxy,
        "age": [int(row[0]) for row in test], "hours": [int(row[12]) for row in test],
        "oracle": [int(row[14].rstrip(".") == ">50K") for row in test],
    })
    snapshot = ROOT / "data/_sprint1_adult.parquet"
    pq.write_table(table, snapshot)
    try:
        with duckdb.connect() as con:
            con.register("adult_replay", table)
            rows = con.sql("SELECT grp, count(*), sum(oracle) FROM adult_replay GROUP BY 1 ORDER BY 1").fetchall()
        truth = {g: int(count) for g, _, count in rows}
        sizes = {g: int(n) for g, n, _ in rows}
        eps = {g: 0.04 * sizes[g] for g in sizes}
        engines = make_engines(snapshot, group_col="grp", proxy_col="proxy",
                               covariate_cols=["age", "hours"], oracle_col="oracle")
        runs = []
        for seed in range(1000, 1000 + trials):
            for mode in MODES:
                result = engines[mode].execute_grouped_agg(
                    eps=eps, delta=0.1, agg_type="count", batch_size=32, pilot_per_group=16,
                    seed=seed, use_control_variates=mode != "no_proxy",
                    use_covariates=mode != "no_proxy",
                )
                runs.append(record_run(
                    mode, seed, result, truth, sizes,
                    target_met=all(result.half_widths[g] <= eps[g] + 1e-9 for g in truth),
                ))
        return {
            "query": "COUNT", "seeds": [1000, 1000 + trials - 1], "delta": 0.1,
            "batch_size": 32, "pilot_per_group": 16, "rate_tolerance": 0.04,
            "proxy_covariates": ["age", "hours"], "truth": truth, "sizes": sizes,
            "source_sha256": sha256(archive_path), "summary": summarize_runs(runs), "runs": runs,
        }
    finally:
        snapshot.unlink(missing_ok=True)


def run_sms():
    train, test, holdout_labels, test_text, proxy_model, reference = prepare_sms()
    start = perf_counter()
    scores = proxy_model.predict_proba(test_text)[:, 1]
    proxy_scan_ms = 1000 * (perf_counter() - start)
    group_ids = np.searchsorted(LENGTH_CUTS, [len(text) for text in test_text], side="right")
    groups = np.asarray([f"L{group}" for group in group_ids])
    n = len(test)
    table = pa.table({"grp": groups, "proxy": scores, "row_id": np.arange(n)})
    engines = make_engines(table, group_col="grp", proxy_col="proxy",
                           covariate_cols=[], oracle_fn=reference)
    exhaustive = []
    exact = None
    for batch in (32, 128, n):
        reference.reset()
        start = perf_counter()
        predicted = np.concatenate([
            reference(np.arange(i, min(i + batch, n))) for i in range(0, n, batch)
        ])
        if exact is None:
            exact = predicted
        elif not np.array_equal(exact, predicted):
            raise AssertionError("SMS exhaustive labels depend on batch size")
        exhaustive.append({"batch_size": batch, "calls": reference.calls,
                           "wall_ms": 1000 * (perf_counter() - start)})
    sizes = dict(Counter(groups.tolist()))
    truth = {g: int(exact[groups == g].sum()) for g in sorted(sizes)}
    expected_top = sorted(sizes, key=lambda g: (-truth[g], g))[:2]
    eps = {g: 0.06 * sizes[g] for g in sizes}
    runs = []
    for query in ("COUNT", "HAVING", "TOP-2"):
        for seed in SMS_SEEDS:
            for mode in MODES:
                reference.reset()
                reference.expected = exact
                options = dict(seed=seed, agg_type="count", delta=0.1, batch_size=32,
                               pilot_per_group=8, use_control_variates=mode != "no_proxy",
                               use_covariates=False)
                if query == "COUNT":
                    result = engines[mode].execute_grouped_agg(eps=eps, **options)
                    decision = {"target_met": all(result.half_widths[g] <= eps[g] + 1e-9
                                                   for g in sizes)}
                elif query == "HAVING":
                    result = engines[mode].execute_grouped_having(tau=40, indifference_eps=0, **options)
                    decision = {
                        "accepted": result.accepted_groups, "rejected": result.rejected_groups,
                        "uncertain": result.uncertain_groups,
                        "decision_correct": all(truth[g] >= 40 for g in result.accepted_groups)
                        and all(truth[g] < 40 for g in result.rejected_groups),
                    }
                else:
                    result = engines[mode].execute_grouped_topk(k=2, **options)
                    decision = {"selected": result.topk_groups, "certified": result.certified,
                                "decision_correct": set(result.topk_groups) == set(expected_top)}
                if result.total_oracle_calls != reference.calls:
                    raise AssertionError("SMS call accounting mismatch")
                runs.append(record_run(mode, seed, result, truth, sizes, query=query, **decision))
    output = {}
    for query in ("COUNT", "HAVING", "TOP-2"):
        subset = [r for r in runs if r["query"] == query]
        output[query] = {
            "seeds": list(SMS_SEEDS), "delta": 0.1, "batch_size": 32,
            "pilot_per_group": 8, "truth": truth, "sizes": sizes,
            "summary": summarize_runs(subset), "runs": subset,
        }
    return {
        "source": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "source_sha256": sha256(ROOT / "data/uci_sms.zip"), "train_rows": len(train),
        "test_rows": n, "proxy_scan_ms": proxy_scan_ms, "exhaustive": exhaustive,
        "reference_accuracy_vs_annotations": float(np.mean(exact == holdout_labels)),
        "workloads": output,
    }


def run_fashion():
    n_train, n_fit, test, y_test, hashes, proxy, reference, cuts = prepare_fashion()
    start = perf_counter()
    scores = proxy.predict_proba(pooled(test))[:, 1]
    proxy_ms = 1000 * (perf_counter() - start)
    groups = np.array([f"B{i}" for i in np.searchsorted(cuts, test.mean(axis=(1, 2)), side="right")])
    n = len(test)
    table = pa.table({"grp": groups, "proxy": scores, "row_id": np.arange(n)})
    engines = make_engines(table, group_col="grp", proxy_col="proxy",
                           covariate_cols=[], oracle_fn=reference)
    exhaustive = []
    exact = None
    for batch in (64, 512, n):
        reference.reset(exact)
        start = perf_counter()
        predicted = np.concatenate([
            reference(np.arange(i, min(i + batch, n))) for i in range(0, n, batch)
        ])
        if exact is None:
            exact = predicted
        elif not np.array_equal(exact, predicted):
            raise AssertionError("Fashion exhaustive labels depend on batch size")
        exhaustive.append({"batch_size": batch, "calls": reference.calls,
                           "wall_ms": 1000 * (perf_counter() - start)})
    truth = {g: int(exact[groups == g].sum()) for g in sorted(set(groups.tolist()))}
    sizes = {g: int((groups == g).sum()) for g in truth}
    eps = {g: 0.04 * sizes[g] for g in sizes}
    runs = []
    for seed in FASHION_SEEDS:
        for mode in MODES:
            reference.reset(exact)
            result = engines[mode].execute_grouped_agg(
                eps=eps, delta=0.1, agg_type="count", batch_size=64, pilot_per_group=16,
                use_control_variates=mode != "no_proxy", use_covariates=False, seed=seed,
            )
            if result.total_oracle_calls != reference.calls:
                raise AssertionError("Fashion call accounting mismatch")
            runs.append(record_run(
                mode, seed, result, truth, sizes,
                target_met=all(result.half_widths[g] <= eps[g] + 1e-9 for g in truth),
            ))
    return {
        "source": "https://github.com/zalandoresearch/fashion-mnist", "sha256": hashes,
        "train_rows": n_train, "fit_rows": n_fit, "test_rows": n, "proxy_ms": proxy_ms,
        "reference_accuracy": float(np.mean(exact == y_test)), "delta": 0.1,
        "batch_size": 64, "pilot_per_group": 16, "rate_tolerance": 0.04,
        "truth": truth, "sizes": sizes, "exhaustive": exhaustive,
        "summary": summarize_runs(runs), "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adult-trials", type=int, default=30)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/sprint1_safe_gating_ablation.json")
    args = parser.parse_args()
    if args.adult_trials < 1:
        parser.error("--adult-trials must be positive")
    with threadpool_limits(limits=1):
        adult = run_adult(args.adult_trials)
        sms = run_sms()
        fashion = run_fashion()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "modes": list(MODES), "safe_rounding_all_arms": True,
            "gating_default": "safe_loo", "legacy_selector": "in_sample_variance",
            "no_proxy": "use_control_variates=False",
            "width_definition": "mean/max of (upper-lower)/(2*group_size) on count rate scale",
        },
        "code_sha256": {name: sha256(ROOT / "src/stratq" / name)
                        for name in ("bounds.py", "engine.py")},
        "script_sha256": sha256(Path(__file__)),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "duckdb": duckdb.__version__, "pyarrow": pa.__version__,
                     "sklearn": sklearn.__version__},
        "adult": adult, "sms": sms, "fashion_mnist": fashion,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Adult", json.dumps(adult["summary"], indent=2))
    for query, item in sms["workloads"].items():
        print("SMS", query, json.dumps(item["summary"], indent=2))
    print("Fashion-MNIST", json.dumps(fashion["summary"], indent=2))
    print(args.output)


if __name__ == "__main__":
    main()

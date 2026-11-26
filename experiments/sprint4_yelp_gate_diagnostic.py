"""CPU/GPU-light Yelp COUNT gate diagnostic for the Sprint 4 review.

Trains the same TF-IDF proxy as the frozen workload, performs one reference
pass over the 20K evaluation rows, verifies the frozen group totals, and traces
every Safe-LOO decision for the three matched COUNT seeds.  It does not rewrite
the Sprint 2 measurement JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from stratq.engine import StratQEngine
import stratq.engine as engine_module
from stratq import modern_baselines as mb
from sprint2_gpu_workloads import text_predict_fn


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    archive = ROOT / "data/sprint2/yelp_test.parquet"
    model_dir = ROOT / "models/sprint2/distilbert_sst2"
    frozen = json.loads((ROOT / "results/sprint2_gpu_workloads.json").read_text(encoding="utf-8"))
    frozen_yelp = next(w for w in frozen["workloads"] if w["workload"] == "Yelp-DistilBERT-SST2")
    table = pq.read_table(archive)
    labels_all = np.asarray(table["label"].to_numpy()) + 1
    texts_all = table["text"].to_pylist()
    train_texts, texts = texts_all[:18000], texts_all[18000:38000]
    train_labels = (labels_all[:18000] >= 4).astype(np.uint8)
    stars = labels_all[18000:38000]
    names = [str(s) for s in sorted(set(stars.tolist()))]
    groups = stars.astype(str)
    indices = [np.flatnonzero(groups == g) for g in names]
    sizes = {g: len(idx) for g, idx in zip(names, indices)}

    with threadpool_limits(limits=1):
        proxy_model = make_pipeline(
            TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True),
            LogisticRegression(max_iter=1000, random_state=1729),
        )
        proxy_model.fit(train_texts, train_labels)
        proxy = proxy_model.predict_proba(texts)[:, 1]

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    synchronize = torch.cuda.synchronize if device.type == "cuda" else None
    predict = text_predict_fn(model, tokenizer, texts, device)
    chunks = []
    for start in range(0, len(texts), 128):
        if synchronize is not None:
            synchronize()
        chunks.append(predict(np.arange(start, min(start + 128, len(texts)))))
        if synchronize is not None:
            synchronize()
    oracle = np.concatenate(chunks)
    truth = {g: int(oracle[idx].sum()) for g, idx in zip(names, indices)}
    if truth != frozen_yelp["truth"]:
        raise AssertionError(f"Reference totals changed: {truth} != {frozen_yelp['truth']}")

    trace = []
    current_group = [None]
    engine = StratQEngine(
        pa.table({"grp": groups, "proxy": proxy, "oracle": oracle}),
        group_col="grp", proxy_col="proxy", oracle_col="oracle",
    )
    original_selector = engine_module.select_optimal_shrinkage_theta

    def traced_selector(*a, **kw):
        kw["return_diagnostics"] = True
        theta, bounds, diag = original_selector(*a, **kw)
        trace.append({
            "group": current_group[0],
            "n_seen": int(diag["n_g"]),
            "selected_kappa": diag["selected_kappa"],
            "gate_reason": diag["gate_reason"],
            "prediction_min": float(bounds[0]),
            "prediction_max": float(bounds[1]),
            "prediction_span": float(bounds[1] - bounds[0]),
            "range_factor": 1.0 + float(bounds[1] - bounds[0]),
            "baseline_cost": diag["baseline_cost"],
            "selected_cost": diag["selected_cost"],
            "loo_variances": diag["loo_variances"],
            "loo_costs": diag["loo_costs"],
            "effective_df": diag["effective_df"],
        })
        return theta, bounds

    engine_module.select_optimal_shrinkage_theta = traced_selector
    original_theta = engine._get_predictable_theta

    def traced_theta(gi, *a, **kw):
        current_group[0] = names[gi]
        return original_theta(gi, *a, **kw)

    engine._get_predictable_theta = traced_theta
    eps = {g: 0.04 * sizes[g] for g in names}
    runs = []
    for seed in (6001, 6002, 6003):
        trace.clear()
        result = engine.execute_grouped_agg(
            eps=eps, delta=0.1, agg_type="count", batch_size=64,
            pilot_per_group=8, seed=seed, use_covariates=False,
        )
        runs.append({"seed": seed, "calls": result.total_oracle_calls,
                     "covered": all(result.lower[g] <= truth[g] <= result.upper[g] for g in names),
                     "gate_trace": list(trace)})

    spertus_calls = []
    for seed in (6001, 6002, 6003):
        result = mb.spertus_wor_count(indices, proxy, oracle, eps, seed=seed,
                                      batch_size=64, pilot_per_group=8, group_names=names)
        spertus_calls.append(result.total_oracle_calls)

    all_trace = [item for run in runs for item in run["gate_trace"]]
    reasons = Counter(item["gate_reason"] for item in all_trace)
    selected = [item for item in all_trace if item["selected_kappa"] > 0]
    selected_range = [item for item in selected if item["prediction_span"] > 0]
    variance_reductions = []
    for item in selected:
        base = item["loo_variances"].get(0.0)
        chosen = item["loo_variances"].get(item["selected_kappa"])
        if base and chosen is not None:
            variance_reductions.append(1.0 - chosen / base)
    group_variance = {g: float(np.mean((oracle[idx] - oracle[idx].mean()) ** 2)) for g, idx in zip(names, indices)}

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "python": platform.python_version(),
        "dataset": {"path": "data/sprint2/yelp_test.parquet", "sha256": sha(archive),
                    "revision": "c1f9ee939b7d05667af864ee1cb066393154bf85"},
        "reference_model": {"path": "models/sprint2/distilbert_sst2", "sha256": sha(model_dir / "model.safetensors"),
                            "revision": "714eb0fa89d2f80546fda750413ed43d93601a13"},
        "frozen_result_sha256": hashlib.sha256((ROOT / "results/sprint2_gpu_workloads.json").read_bytes()).hexdigest(),
        "current_code_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in (
            ROOT / "src/stratq/bounds.py", ROOT / "src/stratq/engine.py", ROOT / "src/stratq/modern_baselines.py")},
        "sizes": sizes,
        "truth": truth,
        "truth_matches_frozen": True,
        "group_label_variance": group_variance,
        "stratq_runs": runs,
        "spertus_calls": spertus_calls,
        "summary": {
            "stratq_mean_calls": float(np.mean([r["calls"] for r in runs])),
            "spertus_mean_calls": float(np.mean(spertus_calls)),
            "gate_calls": len(all_trace),
            "selected_proxy_calls": len(selected),
            "selected_proxy_fraction": len(selected) / len(all_trace),
            "gate_reasons": dict(reasons),
            "selected_with_nonzero_range": len(selected_range),
            "selected_range_factor_mean": float(np.mean([r["range_factor"] for r in selected])) if selected else None,
            "selected_prediction_span_max": max((r["prediction_span"] for r in selected), default=None),
            "selected_with_range_factor_above_1_01": sum(r["range_factor"] > 1.01 for r in selected),
            "selected_loo_variance_reduction_mean": float(np.mean(variance_reductions)) if variance_reductions else None,
            "selected_loo_variance_reduction_median": float(np.median(variance_reductions)) if variance_reductions else None,
            "call_gap_vs_spertus_pct": 100.0 * (np.mean([r["calls"] for r in runs]) / np.mean(spertus_calls) - 1.0),
            "range_penalty_explanation_supported": max((r["prediction_span"] for r in selected), default=0.0) > 1e-6,
        },
        "interpretation_boundary": (
            "The trace measures range and LOO variance at each gate. It can test whether the "
            "range-penalty explanation is consistent with the observed calls, but it does not "
            "identify a causal counterfactual for every batch."
        ),
    }
    target = ROOT / "results/sprint4_yelp_gate_diagnostic.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(target)


if __name__ == "__main__":
    main()

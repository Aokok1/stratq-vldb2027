"""Sprint 2 real GPU expensive-predicate workloads (text + vision).

Writes results/sprint2_gpu_workloads.json.  All StratQ runs use the Sprint 1
default safe_loo + safe_rounding engine.  Modern comparators are operator-level
adaptations documented in BASELINE_AUDIT.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import sys
import tarfile
from pathlib import Path
from time import perf_counter

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
from stratq.bounds import _psi_e_enclosure_cached
from stratq import modern_baselines as mb
from sprint2_systems import breakeven_q, code_hashes, coverage_summary

DATA = ROOT / "data" / "sprint2"
MODELS = ROOT / "models" / "sprint2"


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GPUReference:
    def __init__(self, predict_fn, n, synchronize=None):
        self.predict_fn = predict_fn
        self.n = n
        self.synchronize = synchronize
        self.reset()

    def reset(self, expected=None):
        self.calls = self.batches = 0
        self.inference_ms = 0.0
        self.seen = np.zeros(self.n, dtype=bool)
        self.expected = expected

    def __call__(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if self.seen[ids].any() or len(np.unique(ids)) != len(ids):
            raise AssertionError("repeated row in one query")
        if self.synchronize is not None:
            self.synchronize()
        start = perf_counter()
        out = self.predict_fn(ids)
        if self.synchronize is not None:
            self.synchronize()
        self.inference_ms += (perf_counter() - start) * 1000
        self.calls += len(ids)
        self.batches += 1
        self.seen[ids] = True
        if self.expected is not None and not np.array_equal(out, self.expected[ids]):
            raise AssertionError("reference output changed with batch")
        return out


def exact_pass(reference, n, batch):
    reference.reset()
    start = perf_counter()
    out = np.concatenate([reference(np.arange(i, min(i + batch, n))) for i in range(0, n, batch)])
    wall = (perf_counter() - start) * 1000
    return out, {"batch_size": batch, "calls": reference.calls, "batches": reference.batches,
                 "inference_ms": reference.inference_ms, "wall_ms": wall}


def run_count_comparison(engine, indices, proxy, reference, exact, truth, sizes, names, eps, seeds):
    runs = []
    count_results = {}
    for seed in seeds:
        for method in ("StratQ-safe_loo", "Spertus-WoR", "CELEUS-FixedProxy",
                       "Calibrated-FixedProxy", "BARGAIN-Global"):
            reference.reset(exact)
            start = perf_counter()
            if method == "StratQ-safe_loo":
                r = engine.execute_grouped_agg(eps=eps, delta=0.1, agg_type="count", batch_size=64,
                                               pilot_per_group=8, seed=seed, use_covariates=False)
            elif method == "Spertus-WoR":
                r = mb.spertus_wor_count(indices, proxy, reference, eps, seed=seed,
                                         batch_size=64, pilot_per_group=8, group_names=names)
            elif method == "CELEUS-FixedProxy":
                r = mb.celeus_fixed_proxy_count(indices, proxy, reference, eps, seed=seed,
                                                batch_size=64, pilot_per_group=8, group_names=names)
            elif method == "Calibrated-FixedProxy":
                r = mb.calibrated_fixed_proxy_count(indices, proxy, reference, eps, seed=seed,
                                                    batch_size=64, pilot_per_group=8, group_names=names)
            else:
                r = mb.bargain_global_count(indices, proxy, reference, eps_global=0.04 * sum(sizes.values()),
                                            seed=seed, batch_size=64, pilot=8, group_names=names)
            wall = (perf_counter() - start) * 1000
            cov = coverage_summary(r, truth, sizes)
            calls = r.total_oracle_calls
            assert calls == reference.calls
            if hasattr(r, "group_lower"):
                lower, upper = r.group_lower, r.group_upper
            else:
                lower, upper = r.lower, r.upper
            widths = {g: (upper[g] - lower[g]) / (2 * sizes[g]) for g in names}
            runs.append({"method": method, "seed": seed, "calls": calls, "wall_ms": wall,
                         "inference_ms": reference.inference_ms, "batches": reference.batches,
                         "mean_normalized_half_width": float(np.mean(list(widths.values()))),
                         **cov})
            count_results[(method, seed)] = r
    return runs, count_results


def run_decisions(engine, reference, exact, truth, names, seeds, count_results):
    runs = []
    tau = int(np.median(list(truth.values())))
    expected_top = sorted(truth, key=lambda g: (-truth[g], g))[:2]
    for seed in seeds:
        reference.reset(exact)
        h = engine.execute_grouped_having(tau=tau, agg_type="count", delta=0.1, batch_size=64,
                                          pilot_per_group=8, seed=seed, use_covariates=False)
        runs.append({"query": "HAVING", "method": "StratQ-safe_loo", "seed": seed, "tau": tau,
                     "calls": h.total_oracle_calls, "accepted": h.accepted_groups,
                     "rejected": h.rejected_groups, "uncertain": h.uncertain_groups,
                     "decision_correct": all(truth[g] >= tau for g in h.accepted_groups)
                     and all(truth[g] < tau for g in h.rejected_groups)})
        reference.reset(exact)
        t = engine.execute_grouped_topk(k=2, agg_type="count", delta=0.1, batch_size=64,
                                        pilot_per_group=8, seed=seed, use_covariates=False)
        runs.append({"query": "TOP-2", "method": "StratQ-safe_loo", "seed": seed,
                     "calls": t.total_oracle_calls, "selected": t.topk_groups,
                     "certified": t.certified,
                     "decision_correct": set(t.topk_groups) == set(expected_top)})
        for method, r in count_results.items():
            if method[1] != seed or method[0] == "StratQ-safe_loo":
                continue
            hd = mb.decisions_from_count_bounds(r, truth, tau=tau)
            td = mb.decisions_from_count_bounds(r, truth, k=2)
            runs.append({"query": "HAVING", "method": method[0], "seed": seed, "tau": tau,
                         "calls": r.total_oracle_calls, **hd})
            runs.append({"query": "TOP-2", "method": method[0], "seed": seed,
                         "calls": r.total_oracle_calls, **td})
    return runs


def run_latency(engine, table, reference, exact, n, sizes, truth, repeats=5,
                proxy_train_ms=None, proxy_scan_ms=None):
    exact_runs = []
    for batch in (64, 128, 256):
        for rep in range(repeats):
            out, rec = exact_pass(reference, n, batch)
            counts = {g: int(out[np.asarray(table["grp"].to_pylist()) == g].sum()) for g in truth}
            assert counts == truth
            exact_runs.append({"replicate": rep, **rec})
    best_batch = min({r["batch_size"] for r in exact_runs},
                     key=lambda b: np.median([r["wall_ms"] for r in exact_runs if r["batch_size"] == b]))
    fresh, reused = [], []
    warm = engine
    for rep in range(repeats):
        eps_rate = 0.03 + 0.01 * (rep % 3)
        eps = {g: eps_rate * sizes[g] for g in truth}
        # Fresh means a new TableProfile and a cold process-local psi cache.
        # The GPU model and OS cache remain warm after the exact batch sweep.
        _psi_e_enclosure_cached.cache_clear()
        start = perf_counter()
        fresh_engine = StratQEngine(table, group_col="grp", proxy_col="proxy", oracle_fn=reference)
        profile_ms = (perf_counter() - start) * 1000
        reference.reset(exact)
        start = perf_counter()
        r = fresh_engine.execute_grouped_agg(eps=eps, delta=0.1, agg_type="count", batch_size=64,
                                             pilot_per_group=8, seed=5000 + rep, use_covariates=False)
        query_ms = (perf_counter() - start) * 1000
        fresh.append({"replicate": rep, "profile_ms": profile_ms, "query_ms": query_ms,
                      "wall_ms": profile_ms + query_ms, "calls": r.total_oracle_calls})
        reference.reset(exact)
        start = perf_counter()
        r2 = warm.execute_grouped_agg(eps=eps, delta=0.1, agg_type="count", batch_size=64,
                                      pilot_per_group=8, seed=5000 + rep, use_covariates=False)
        reused.append({"replicate": rep, "profile_ms": 0.0,
                       "query_ms": (perf_counter() - start) * 1000, "calls": r2.total_oracle_calls})
    # Cumulative Q-query amortization using median per-query costs.
    fresh_med = float(np.median([r["wall_ms"] for r in fresh]))
    reused_med = float(np.median([r["query_ms"] for r in reused]))
    exact_med = float(np.median([r["wall_ms"] for r in exact_runs if r["batch_size"] == best_batch]))
    qs = list(range(1, 21))
    fresh_cum = [fresh_med * q for q in qs]
    reused_cum = [float(np.median([r["profile_ms"] for r in fresh])) + reused_med * q for q in qs]
    exact_cum = [exact_med * q for q in qs]
    return {"exact_runs": exact_runs, "best_exact_batch": best_batch,
            "fresh_runs": fresh, "reused_runs": reused,
            "median_fresh_ms": fresh_med, "median_reused_query_ms": reused_med,
            "median_exact_ms": exact_med,
            "breakeven_vs_fresh": breakeven_q(fresh_cum, reused_cum),
            "breakeven_vs_exact": breakeven_q(exact_cum, reused_cum),
            "tolerance_schedule": [0.03 + 0.01 * (rep % 3) for rep in range(repeats)],
            "query_seeds": list(range(5000, 5000 + repeats)),
            "fresh_psi_cache_cleared": True,
            "gpu_model_cache_state": "warm after exact batch sweep",
            "proxy_scores_precomputed": True,
            "excluded_proxy_train_ms": proxy_train_ms,
            "excluded_proxy_scan_ms": proxy_scan_ms}


def text_predict_fn(model, tokenizer, texts, device):
    import torch

    @torch.inference_mode()
    def predict(ids):
        batch_texts = [texts[i] for i in ids]
        enc = tokenizer(batch_texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        return torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
    return predict


def run_text(smoke=False):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synchronize = torch.cuda.synchronize if device.type == "cuda" else None
    model_dir = MODELS / "distilbert_sst2"
    archive = DATA / "yelp_test.parquet"
    n_eval = 2000 if smoke else 20000
    n_train = 1000 if smoke else 18000
    table = pq.read_table(archive)
    labels_all = np.asarray(table["label"].to_numpy()) + 1
    texts_all = table["text"].to_pylist()
    if len(texts_all) < n_train + n_eval:
        raise ValueError("Yelp test split too small for disjoint proxy/eval subsets")
    train_texts = texts_all[:n_train]
    train_labels = (labels_all[:n_train] >= 4).astype(np.uint8)
    texts = texts_all[n_train:n_train + n_eval]
    stars = labels_all[n_train:n_train + n_eval]
    names = [str(s) for s in sorted(set(stars.tolist()))]
    groups = stars.astype(str)
    indices = [np.flatnonzero(groups == g) for g in names]
    sizes = {g: len(idx) for g, idx in zip(names, indices)}
    start = perf_counter()
    proxy_model = make_pipeline(TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True),
                                LogisticRegression(max_iter=1000, random_state=1729))
    proxy_model.fit(train_texts, train_labels)
    proxy_train_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    proxy = proxy_model.predict_proba(texts)[:, 1]
    proxy_scan_ms = (perf_counter() - start) * 1000
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    reference = GPUReference(text_predict_fn(model, tokenizer, texts, device), n_eval,
                             synchronize=synchronize)
    exact, exact_rec = exact_pass(reference, n_eval, 128)
    truth = {g: int(exact[idx].sum()) for g, idx in zip(names, indices)}
    table = pa.table({"grp": groups, "proxy": proxy})
    engine = StratQEngine(table, group_col="grp", proxy_col="proxy", oracle_fn=reference)
    eps = {g: 0.04 * sizes[g] for g in names}
    seeds = (6001,) if smoke else (6001, 6002, 6003)
    count_runs, count_results = run_count_comparison(engine, indices, proxy, reference, exact,
                                                     truth, sizes, names, eps, seeds)
    decision_runs = run_decisions(engine, reference, exact, truth, names, seeds, count_results)
    latency = run_latency(engine, table, reference, exact, n_eval, sizes, truth,
                          repeats=2 if smoke else 5, proxy_train_ms=proxy_train_ms,
                          proxy_scan_ms=proxy_scan_ms)
    peak_gpu = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return {"workload": "Yelp-DistilBERT-SST2", "n_eval": n_eval, "n_train_proxy": n_train,
            "proxy_eval_split": "disjoint row prefixes of the official Yelp test parquet",
            "groups": names, "sizes": sizes, "truth": truth,
            "model_sha256": sha(model_dir / "model.safetensors"),
            "dataset_sha256": sha(archive),
            "proxy_train_ms": proxy_train_ms, "proxy_scan_ms": proxy_scan_ms,
            "exact_initial": exact_rec, "peak_gpu_bytes": peak_gpu,
            "count_runs": count_runs, "decision_runs": decision_runs, "latency": latency}


def load_cifar():
    archive = DATA / "cifar-10-python.tar.gz"
    out = DATA / "cifar10"
    if not (out / "cifar-10-batches-py" / "test_batch").exists():
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(out, filter="data")
    base = out / "cifar-10-batches-py"
    def load_batch(name):
        with open(base / name, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.asarray(d[b"labels"])
    test_x, test_y = load_batch("test_batch")
    train_x, train_y = [], []
    for i in range(1, 6):
        x, y = load_batch(f"data_batch_{i}")
        train_x.append(x); train_y.append(y)
    return np.concatenate(train_x), np.concatenate(train_y), test_x, test_y


def vision_batch_tensor(images, device):
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255).to(device)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def run_vision(smoke=False):
    import torch
    import torch.nn.functional as F
    from torchvision import models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synchronize = torch.cuda.synchronize if device.type == "cuda" else None
    train_x, train_y, test_x, test_y = load_cifar()
    n_train = 2000 if smoke else 20000
    n_eval = 1000 if smoke else 10000
    train_x, train_y = train_x[:n_train], train_y[:n_train]
    test_x, test_y = test_x[:n_eval], test_y[:n_eval]
    animal_classes = {2, 3, 4, 5, 6, 7}
    pos_train = np.isin(train_y, list(animal_classes)).astype(np.uint8)
    names = [str(c) for c in range(10)]
    indices = [np.flatnonzero(test_y == c) for c in range(10)]
    sizes = {g: len(idx) for g, idx in zip(names, indices)}
    # Cheap low-resolution (4x4 RGB + channel stats) CPU proxy.
    def _low_res_feats(arr):
        grid = arr.reshape(len(arr), 4, 8, 4, 8, 3).mean(axis=(2, 4)).reshape(len(arr), -1) / 255.0
        ch_std = arr.reshape(len(arr), -1, 3).std(axis=1) / 255.0
        return np.hstack([grid, ch_std])

    low = _low_res_feats(train_x)
    low_test = _low_res_feats(test_x)
    start = perf_counter()
    proxy_model = LogisticRegression(max_iter=500, random_state=1729).fit(low, pos_train)
    proxy_train_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    proxy = proxy_model.predict_proba(low_test)[:, 1]
    proxy_scan_ms = (perf_counter() - start) * 1000
    # Real GPU reference: cached ImageNet ResNet34 fine-tuned at standard 224x224 on CIFAR-10.
    weights_path = Path.home() / ".cache/torch/hub/checkpoints/resnet34-b627a593.pth"
    model = models.resnet34(weights=None)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    epochs = 1 if smoke else 2
    batch = 64
    train_times = []
    model.train()
    for epoch in range(epochs):
        start = perf_counter()
        perm = np.random.default_rng(1729 + epoch).permutation(n_train)
        for i in range(0, n_train, batch):
            ids = perm[i:i + batch]
            x = vision_batch_tensor(train_x[ids], device)
            y = torch.from_numpy(train_y[ids]).long().to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_times.append((perf_counter() - start) * 1000)
    model.eval()
    state_path = MODELS / "resnet34_cifar10.pth"
    torch.save(model.state_dict(), state_path)

    @torch.inference_mode()
    def predict(ids):
        x = vision_batch_tensor(test_x[ids], device)
        pred = torch.argmax(model(x), dim=1).cpu().numpy()
        return np.isin(pred, list(animal_classes)).astype(np.uint8)

    reference = GPUReference(predict, n_eval, synchronize=synchronize)
    exact, exact_rec = exact_pass(reference, n_eval, 256)
    truth = {g: int(exact[idx].sum()) for g, idx in zip(names, indices)}
    groups = test_y.astype(str)
    table = pa.table({"grp": groups, "proxy": proxy})
    engine = StratQEngine(table, group_col="grp", proxy_col="proxy", oracle_fn=reference)
    eps = {g: 0.04 * sizes[g] for g in names}
    seeds = (7001,) if smoke else (7001, 7002, 7003)
    count_runs, count_results = run_count_comparison(engine, indices, proxy, reference, exact,
                                                     truth, sizes, names, eps, seeds)
    decision_runs = run_decisions(engine, reference, exact, truth, names, seeds, count_results)
    latency = run_latency(engine, table, reference, exact, n_eval, sizes, truth,
                          repeats=2 if smoke else 5, proxy_train_ms=proxy_train_ms,
                          proxy_scan_ms=proxy_scan_ms)
    peak_gpu = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return {"workload": "CIFAR10-ResNet34-GPU", "n_eval": n_eval, "n_train_reference": n_train,
            "groups": names, "sizes": sizes, "truth": truth,
            "reference_train_epoch_ms": train_times, "model_sha256": sha(state_path),
            "backbone_sha256": sha(weights_path), "dataset_sha256": sha(DATA / "cifar-10-python.tar.gz"),
            "proxy_train_ms": proxy_train_ms, "proxy_scan_ms": proxy_scan_ms,
            "exact_initial": exact_rec,
            "peak_gpu_bytes": peak_gpu, "count_runs": count_runs,
            "decision_runs": decision_runs, "latency": latency}


def summarize(workload):
    summary = {}
    for method in sorted({r["method"] for r in workload["count_runs"]}):
        subset = [r for r in workload["count_runs"] if r["method"] == method]
        summary[method] = {"trials": len(subset),
                           "mean_calls": float(np.mean([r["calls"] for r in subset])),
                           "all_covered_trials": sum(r["all_covered"] for r in subset),
                           "mean_half_width": float(np.mean([r["mean_normalized_half_width"] for r in subset])),
                           "mean_wall_ms": float(np.mean([r["wall_ms"] for r in subset]))}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workload", choices=("text", "vision", "both"), default="both")
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Sprint 2 GPU workloads require CUDA")
    with threadpool_limits(limits=1):
        workloads = []
        if args.workload in ("text", "both"):
            workloads.append(run_text(args.smoke))
        if args.workload in ("vision", "both"):
            workloads.append(run_vision(args.smoke))
    report = {"generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
              "smoke": args.smoke, "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "code_sha256": code_hashes(), "workloads": workloads,
              "summary": {w["workload"]: summarize(w) for w in workloads}}
    for w in workloads:
        if w["peak_gpu_bytes"] > 6 * 2**30:
            raise RuntimeError(f"GPU peak exceeds 6 GiB: {w['peak_gpu_bytes']}")
    out = ROOT / "results" / ("sprint2_gpu_smoke.json" if args.smoke else "sprint2_gpu_workloads.json")
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(out)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

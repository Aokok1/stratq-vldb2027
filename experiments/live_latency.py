"""Repeated timing of exact inference and StratQ, with trained models in memory."""
import hashlib
import json
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import duckdb
import numpy as np
import pyarrow as pa
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits

import fashion_live
import sms_live
from stratq.engine import StratQEngine

ROOT = Path(__file__).resolve().parents[1]


def benchmark(workload):
    if workload == 'sms':
        _, _, _, rows, model, reference = sms_live.prepare()
        group_index = lambda: np.searchsorted(sms_live.LENGTH_CUTS, [len(s) for s in rows], side='right')
        score = lambda: model.predict_proba(rows)[:, 1]
        batch, pilot, tolerance = 32, 8, .06
        source_hash = hashlib.sha256(sms_live.ARCHIVE.read_bytes()).hexdigest()
    else:
        _, _, rows, _, source_hash, model, reference, cuts = fashion_live.prepare()
        group_index = lambda: np.searchsorted(cuts, rows.mean(axis=(1, 2)), side='right')
        score = lambda: model.predict_proba(fashion_live.pooled(rows))[:, 1]
        batch, pilot, tolerance = 64, 16, .04
    n = len(rows)
    ids = np.arange(n)
    expected = reference(ids)
    exact_groups = group_index()
    total = np.bincount(exact_groups, weights=expected).astype(int)
    sizes = np.bincount(exact_groups)
    truth = {str(i): int(v) for i, v in enumerate(total)}
    modes = [('exact', b) for b in sorted({batch, 128, 512, 1024, n})]
    modes += [('fresh_no_proxy', batch), ('fresh_proxy', batch), ('reuse_proxy', batch)]
    timing = []
    rng = np.random.default_rng(7401)
    eps = {str(i): tolerance * int(size) for i, size in enumerate(sizes)}

    def profile(use_proxy):
        t = perf_counter_ns()
        scores = score() if use_proxy else np.zeros(n)
        after_proxy = perf_counter_ns()
        table = pa.table({'grp': group_index().astype(str), 'proxy': scores,
                          'row_id': ids})
        engine = StratQEngine(table, group_col='grp', proxy_col='proxy',
                              covariate_cols=['row_id'], oracle_fn=reference)
        if not np.array_equal(engine.profile.phi_all[:, 2], ids / (n - 1)):
            raise RuntimeError('Callback row ordering changed')
        return engine, (after_proxy-t)/1e6, (perf_counter_ns()-after_proxy)/1e6

    warm_engine, setup_proxy_ms, setup_profile_ms = profile(True)
    for replicate in range(11):  # First randomized block warms all arms; exclude it from summaries.
        for ix in rng.permutation(len(modes)):
            mode, b = modes[ix]
            reference.reset()
            reference.expected = expected
            start = perf_counter_ns()
            proxy_ms = profile_ms = 0.
            if mode == 'exact':
                groups = group_index()
                labels = np.concatenate([reference(ids[i:i+b]) for i in range(0, n, b)])
                counts = np.bincount(groups, weights=labels, minlength=len(total))
                wall_ms = (perf_counter_ns()-start)/1e6
                assert np.array_equal(counts, total) and reference.calls == n
                bounds = {str(i): [int(count), int(count)] for i, count in enumerate(counts)}
                query_ms = wall_ms
            else:
                if mode == 'reuse_proxy':
                    engine = warm_engine
                else:
                    engine, proxy_ms, profile_ms = profile(mode == 'fresh_proxy')
                query_start = perf_counter_ns()
                result = engine.execute_grouped_agg(
                    eps=eps, delta=.1, agg_type='count', batch_size=b,
                    pilot_per_group=pilot, use_control_variates=mode != 'fresh_no_proxy',
                    use_covariates=False, seed=8000+replicate,
                )
                end = perf_counter_ns()
                wall_ms, query_ms = (end-start)/1e6, (end-query_start)/1e6
                assert reference.calls == result.total_oracle_calls
                bounds = {g: [result.lower[g], result.upper[g]] for g in result.groups}
                assert set(bounds) == set(truth) and all(
                    lo-1e-9 <= truth[g] <= hi+1e-9 and (hi-lo)/2 <= eps[g]+1e-9
                    for g, (lo, hi) in bounds.items())
            timing.append(dict(replicate=replicate, mode=mode, batch=b, calls=reference.calls,
                               batches=reference.batches, wall_ms=wall_ms, query_ms=query_ms,
                               proxy_ms=proxy_ms, profile_ms=profile_ms, bounds=bounds))
    summary = []
    for mode, b in modes:
        records = [r for r in timing if r['mode'] == mode and r['batch'] == b and r['replicate']]
        summary.append(dict(mode=mode, batch=b, n=len(records), **{
            'median_'+key: median(r[key] for r in records)
            for key in ('calls', 'wall_ms', 'proxy_ms', 'profile_ms', 'query_ms')
        }))
    return dict(dataset=workload, source_hash=source_hash, rows=n, tolerance=tolerance,
                truth=truth, sizes=sizes.tolist(), pilot=pilot, delta=.1,
                setup_proxy_ms=setup_proxy_ms, setup_profile_ms=setup_profile_ms,
                summary=summary, runs=timing)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        report = dict(platform=platform.platform(), cpu=platform.processor(),
                      python=platform.python_version(), sklearn=sklearn.__version__,
                      duckdb=duckdb.__version__, numpy=np.__version__, arrow=pa.__version__,
                      threadpools=threadpool_info(), datasets=[benchmark(x) for x in ('sms', 'fashion')])
    (ROOT/'results/live_latency.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for dataset in report['datasets']:
        print(dataset['dataset'], json.dumps(dataset['summary'], indent=2))

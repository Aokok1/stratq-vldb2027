"""Matched round-robin comparison: raw score vs predictable refitting."""
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import pyarrow as pa
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from stratq.engine import StratQEngine
from fashion_live import pooled, prepare


def source(name):
    blob = (ROOT / 'results' / f'{name}.json').read_bytes()
    return json.loads(blob), hashlib.sha256(blob).hexdigest()


def evaluate(engine, truth, labels, seeds, batch, delta, reference=None):
    assert set(engine.profile.groups) == set(truth)
    for group, ids in zip(engine.profile.groups, engine.profile.group_row_indices):
        assert (len(ids), labels[ids].sum()) == (truth[group]['size'], truth[group]['positives'])
    runs = []
    for seed in seeds:
        for refit in (False, True):
            if reference is not None:
                reference.reset(labels)
            result = engine.execute_grouped_agg(
                eps={g: v['size'] * .04 for g, v in truth.items()}, delta=delta, agg_type='count',
                batch_size=batch, pilot_per_group=16, use_control_variates=True, use_covariates=False,
                use_adaptive_alloc=False, seed=seed, refit_proxy=refit)
            if reference is not None and reference.calls != result.total_oracle_calls:
                raise ValueError('Reference and engine disagree on query call count')
            runs.append(dict(seed=seed, refit=refit, calls=int(result.total_oracle_calls),
                             groups={g: dict(lower=result.lower[g], upper=result.upper[g],
                                             calls=int(result.oracle_calls_by_group[g])) for g in truth}))
    return runs


def main():
    a, a_hash = source('adult_count_replay')
    adult = StratQEngine(ROOT / 'data/adult_test_replay.parquet', group_col='grp',
                         proxy_col='proxy', oracle_col='oracle')
    adults = sorted({r['seed'] for r in a['runs'] if r['mode'] == 'proxy_only'})
    out = {'adult': dict(source_sha256=a_hash, truth=a['truth'], delta=a['delta'], batch=32,
                         runs=evaluate(adult, a['truth'], adult.profile.oracle_true, adults, 32, a['delta']))}
    f, f_hash = source('fashion_live')
    _, _, test, _, hashes, proxy, reference, cuts = prepare()
    assert hashes == f['sha256']
    groups = np.array([f'B{i}' for i in np.searchsorted(cuts, test.mean(axis=(1, 2)), side='right')])
    table = pa.table({'grp': groups, 'proxy': proxy.predict_proba(pooled(test))[:, 1], 'row_id': np.arange(len(test))})
    fashion = StratQEngine(table, group_col='grp', proxy_col='proxy', covariate_cols=['row_id'], oracle_fn=reference)
    assert np.array_equal(fashion.profile.phi_all[:, 2], np.arange(len(test)) / (len(test) - 1))
    exact = np.concatenate([reference(np.arange(i, min(i+512, len(test)))) for i in range(0, len(test), 512)])
    out['fashion'] = dict(source_sha256=f_hash, truth=f['truth'], delta=f['delta'], batch=64,
                          runs=evaluate(fashion, f['truth'], exact, sorted({r['seed'] for r in f['runs']}),
                                        64, f['delta'], reference=reference))
    report = dict(pilot=16, rate_tolerance=.04, routing='round_robin', workloads=out)
    (ROOT / 'results/proxy_refit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for name, work in out.items():
        for refit in (False, True):
            rows = [r for r in work['runs'] if r['refit'] == refit]
            covered = sum(all(x['lower'] <= work['truth'][g]['positives'] <= x['upper']
                              for g, x in r['groups'].items()) for r in rows)
            print(name, refit, len(rows), mean(r['calls'] for r in rows), 'covered:', covered)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()

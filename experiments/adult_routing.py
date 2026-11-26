"""Paired allocation ablation on the Adult snapshot."""
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from stratq.engine import StratQEngine


def main():
    source = (ROOT / 'results/adult_count_replay.json').read_bytes()
    original = json.loads(source)
    truth = original['truth']
    engine = StratQEngine(ROOT / 'data/adult_test_replay.parquet', group_col='grp',
                          proxy_col='proxy', oracle_col='oracle')
    profile = engine.profile
    if set(profile.groups) != set(truth) or any(
        len(ids) != truth[g]['size'] or profile.oracle_true[ids].sum() != truth[g]['positives']
        for g, ids in zip(profile.groups, profile.group_row_indices)
    ):
        raise ValueError('Replay snapshot differs from the recorded reference')
    seeds = sorted({r['seed'] for r in original['runs'] if r['mode'] == 'proxy_only'})
    runs = []
    for seed in seeds:
        for adaptive in (False, True):
            result = engine.execute_grouped_agg(
                agg_type='count', eps={g: v['epsilon'] for g, v in truth.items()},
                batch_size=32, pilot_per_group=16, delta=original['delta'], seed=seed,
                use_control_variates=False, use_covariates=False, use_adaptive_alloc=adaptive,
            )
            bounds = {g: [result.lower[g], result.upper[g]] for g in truth}
            assert all(lo - 1e-9 <= truth[g]['positives'] <= hi + 1e-9 and
                       (hi - lo) / 2 <= truth[g]['epsilon'] + 1e-9
                       for g, (lo, hi) in bounds.items()), (seed, adaptive)
            runs.append({'seed': seed, 'adaptive': adaptive, 'calls': int(result.total_oracle_calls),
                         'group_calls': {g: int(v) for g, v in result.oracle_calls_by_group.items()},
                         'bounds': bounds})
    avg = {str(adaptive): mean(r['calls'] for r in runs if r['adaptive'] == adaptive)
           for adaptive in (False, True)}
    report = {'source_sha256': hashlib.sha256(source).hexdigest(), 'truth': truth,
              'delta': original['delta'], 'pilot': 16, 'batch_size': 32, 'proxy': False,
              'mean_calls': avg, 'runs': runs}
    (ROOT / 'results/adult_routing.json').write_text(json.dumps(report, allow_nan=False, indent=2)+'\n',
                                                        encoding='utf-8')
    print(avg)


if __name__ == '__main__':
    main()

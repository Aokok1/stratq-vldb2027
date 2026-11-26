"""Recompute Fashion-MNIST grouped query outcomes from saved records."""
import json
import math
from pathlib import Path
from statistics import mean, median


def audit(report):
    n = report['test_rows']
    truth = report['truth']
    if n != 10000 or len(truth) != 5 or sum(g['size'] for g in truth.values()) != n:
        raise ValueError('Unexpected test population')
    for scan in report['exhaustive']:
        if scan['calls'] != n or scan['batches'] != math.ceil(n / scan['batch_size']):
            raise ValueError('Incorrect exhaustive call accounting')
    pairs = {}
    for run in report['runs']:
        key = (run['proxy'], run['seed'])
        groups = run['groups']
        if key in pairs or groups.keys() != truth.keys():
            raise ValueError(f'{key}: duplicate trial or group mismatch')
        if sum(g['calls'] for g in groups.values()) != run['calls']:
            raise ValueError(f'{key}: inconsistent call totals')
        coverage = width_met = True
        for group, row in groups.items():
            size = truth[group]['size']
            count = truth[group]['positives']
            lo, hi = row['lower'], row['upper']
            if not (0 <= row['calls'] <= size and -1e-9 <= lo <= hi <= size + 1e-9):
                raise ValueError(f'{key} {group}: invalid calls or bounds')
            if not math.isclose(row['estimate'], (lo + hi) / 2, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f'{key} {group}: midpoint mismatch')
            coverage &= lo - 1e-9 <= count <= hi + 1e-9
            width_met &= (hi - lo) / 2 <= report['rate_tolerance'] * size + 1e-9
        pairs[key] = (run, coverage, width_met)
    seeds = range(2701, 2706)
    if pairs.keys() != {(proxy, seed) for proxy in (False, True) for seed in seeds}:
        raise ValueError('Trial schedule mismatch')
    summary = {}
    for proxy in (False, True):
        subset = [pairs[proxy, seed] for seed in seeds]
        summary[str(proxy)] = {'mean_calls': mean(r['calls'] for r, _, _ in subset),
                               'median_wall_ms': median(r['wall_ms'] for r, _, _ in subset),
                               'covered_trials': sum(covered for _, covered, _ in subset),
                               'width_met_trials': sum(met for _, _, met in subset)}
    return summary


if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1] / 'results' / 'fashion_live.json'
    print(json.dumps(audit(json.loads(path.read_text(encoding='utf-8'))), indent=2, allow_nan=False))

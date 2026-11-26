"""Recompute scale metrics from per-group results and reject inconsistent records."""
import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
SCALES = (100_000, 1_000_000, 10_000_000)


def require(condition, context):
    if not condition:
        raise ValueError(context)


def audit(path, row_count):
    report = json.loads(path.read_text(encoding='utf-8'))
    truth = report['truth']
    require(report['rows'] == row_count and len(truth) == 20, f'{path.name}: size or group count')
    require(sum(size for size, _ in truth.values()) == row_count, f'{path.name}: group sizes')
    runs = {}
    for run in report['runs']:
        key = run['use_proxy'], run['seed']
        context = f'{path.name} {key}'
        require(key not in runs, f'{context}: duplicate trial')
        require(run['groups'].keys() == truth.keys(), f'{context}: group mismatch')
        require(sum(row['calls'] for row in run['groups'].values()) == run['calls'], f'{context}: call total')
        require(math.isfinite(run['wall_ms']) and run['wall_ms'] >= 0, f'{context}: invalid query timer')
        covered = target_met = True
        for group, row in run['groups'].items():
            size, positive_count = truth[group]
            lower, upper = row['lower'], row['upper']
            require(0 <= row['calls'] <= size and 0 <= positive_count <= size, f'{context} {group}: count')
            require(math.isfinite(lower) and math.isfinite(upper) and lower <= upper, f'{context} {group}: bounds')
            covered &= lower - 1e-8 <= positive_count <= upper + 1e-8
            target_met &= (upper - lower) / 2 <= size * report['rate_tolerance'] + 1e-8
        require(covered == run['covered'] and target_met == run['target_met'], f'{context}: certificate flags')
        runs[key] = run
    seeds = (2401, 2402, 2403)
    require(set(runs) == {(proxy, seed) for proxy in (False, True) for seed in seeds}, f'{path.name}: trial pairing')
    baseline, proxy = ([runs[mode, seed] for seed in seeds] for mode in (False, True))
    base_calls, proxy_calls = (mean(r['calls'] for r in mode) for mode in (baseline, proxy))
    return {'rows': row_count, 'profile_ms': report['profile_ms'],
            'peak_working_set_bytes': report['peak_working_set_bytes'],
            'baseline_mean_calls': base_calls, 'proxy_mean_calls': proxy_calls,
            'relative_call_reduction': (base_calls - proxy_calls) / base_calls,
            'baseline_median_query_ms': median(r['wall_ms'] for r in baseline),
            'proxy_median_query_ms': median(r['wall_ms'] for r in proxy),
            'covered_trials': sum(r['covered'] for r in runs.values()),
            'target_met_trials': sum(r['target_met'] for r in runs.values()), 'trials': len(runs)}


if __name__ == '__main__':
    rows = [audit(ROOT / 'results' / f'scale_{size}.json', size) for size in SCALES]
    print(json.dumps(rows, indent=2, allow_nan=False))

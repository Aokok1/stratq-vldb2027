"""Validate the saved SMS benchmark certificate and recompute paired metrics."""
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def audit(report):
    truth = report['positives']
    require(sum(report['sizes'].values()) == report['test_rows'], 'Population size')
    qualifying = {g for g, count in truth.items() if count >= report['having_threshold']}
    top_groups = set(sorted(truth, key=lambda g: (-truth[g], g))[:report['topk']])
    indexed = {}
    for run in report['runs']:
        key = (run['query'], run['proxy'], run['seed'])
        require(key not in indexed, f'{key}: duplicate trial')
        indexed[key] = run
        lower, upper = run['lower'], run['upper']
        require(lower.keys() == upper.keys() == truth.keys(), f'{key}: group mismatch')
        covered = all(lower[g] - 1e-9 <= truth[g] <= upper[g] + 1e-9 for g in truth)
        require(run['covered'] == covered, f'{key}: incorrect coverage flag')
        require(0 < run['batches'] <= run['calls'] <= report['test_rows'], f'{key}: call accounting')
        decision = run['decision']
        if key[0] == 'count':
            attained = all((upper[g] - lower[g]) / 2 <= report['epsilon'][g] + 1e-9 for g in truth)
            require(decision['target_met'] == attained, f'{key}: incorrect width flag')
        elif key[0] == 'having':
            accepted = set(decision['accepted'])
            rejected = set(decision['rejected'])
            uncertain = set(decision['uncertain'])
            require(accepted == {g for g in truth if lower[g] >= report['having_threshold']}, f'{key}: acceptance')
            require(rejected == {g for g in truth if upper[g] < report['having_threshold']}, f'{key}: rejection')
            require(uncertain == set(truth) - (accepted | rejected), f'{key}: uncertain groups')
            require(decision['correct'] == (accepted <= qualifying and not rejected & qualifying), f'{key}: incorrect decision flag')
        elif key[0] == 'topk':
            selected = set(decision['selected'])
            require(len(selected) == report['topk'] and selected <= truth.keys(), f'{key}: invalid top-k members')
            require(decision['correct'] == (selected == top_groups), f'{key}: incorrect top-k flag')
            if decision['certified'] and run['calls'] < report['test_rows']:
                require(min(lower[g] for g in selected) > max(upper[g] for g in truth.keys() - selected), f'{key}: no strict separation')
    queries = ('count', 'having', 'topk')
    seeds = range(2601, 2611)
    require(set(indexed) == {(query, proxy, seed) for query in queries for proxy in (False, True) for seed in seeds}, 'Trial pairing')
    return {query: {str(proxy): summarize([indexed[query, proxy, seed] for seed in seeds])
                    for proxy in (False, True)} for query in queries}


def summarize(runs):
    return {'mean_calls': mean(r['calls'] for r in runs), 'median_wall_ms': median(r['wall_ms'] for r in runs),
            'covered': sum(r['covered'] for r in runs), 'trials': len(runs)}


if __name__ == '__main__':
    path = ROOT / 'results' / 'sms_live.json'
    print(json.dumps(audit(json.loads(path.read_text(encoding='utf-8'))), indent=2, allow_nan=False))

"""Recompute grouped decision certificates and trial summaries from raw records."""
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def audit(data):
    tau, k, n = (data[key] for key in ('threshold', 'k', 'group_size'))
    cells = defaultdict(list)
    seen = set()
    for r in data['runs']:
        gap, cap, proxy, query, seed = (r[key] for key in ('gap', 'budget', 'proxy', 'query', 'seed'))
        key = (gap, cap, proxy, query, seed)
        assert key not in seen, key
        seen.add(key)
        truth = data['populations'][str(gap)]
        order = sorted(truth, key=lambda g: (-truth[g], g))
        assert truth['C'] == tau - gap and truth[order[k-1]] - truth[order[k]] == gap
        lo, hi, counts = (r[key] for key in ('lower', 'upper', 'group_calls'))
        assert set(lo) == set(hi) == set(counts) == set(truth)
        assert 0 <= r['calls'] == sum(counts.values()) <= cap
        assert all(0 <= lo[g] <= hi[g] <= n and 0 <= counts[g] <= n for g in truth)
        assert all(lo[g] == hi[g] == truth[g] for g in truth if counts[g] == n)
        focal_resolved = None
        if query == 'having':
            accepted = {g for g in truth if lo[g] >= tau}
            rejected = {g for g in truth if hi[g] < tau}
            unresolved = set(truth) - (accepted | rejected)
            assert accepted == set(r['accepted']) and rejected == set(r['rejected'])
            assert unresolved == set(r['unresolved'])
            wrong = any(truth[g] < tau for g in accepted) or any(truth[g] >= tau for g in rejected)
            full = not unresolved
            focal_resolved = 'C' not in unresolved
        else:
            assert query == 'topk'
            selected = set(r['selected'])
            rest = set(truth) - selected
            assert len(selected) == k and selected <= set(truth)
            full = all(counts[g] == n for g in truth) or min(lo[g] for g in selected) > max(hi[g] for g in rest)
            assert full == r['certified']
            # Ties resolve by alphabetical profile order, including at census.
            wrong = full and selected != set(order[:k])
        covered = all(lo[g] - 1e-9 <= truth[g] <= hi[g] + 1e-9 for g in truth)
        if covered:
            assert not wrong, key
        label = 'wrong' if wrong else 'correct' if full else 'unresolved'
        cells[key[:-1]].append((label, r['calls'], covered, focal_resolved))
    expected = set(product(data['gaps'], data['budgets'], (False, True), ('having', 'topk'), data['seeds']))
    assert seen == expected
    return {key: dict(trials=len(rows), correct=sum(r[0] == 'correct' for r in rows),
                      wrong=sum(r[0] == 'wrong' for r in rows), unresolved=sum(r[0] == 'unresolved' for r in rows),
                      coverage=sum(r[2] for r in rows), mean_calls=mean(r[1] for r in rows),
                      focal_resolved=sum(r[3] is True for r in rows)) for key, rows in cells.items()}


if __name__ == '__main__':
    data = json.loads((ROOT / 'results/decision_boundaries.json').read_text(encoding='utf-8'))
    cells = audit(data)
    print('gap cap proxy having_correct focal_C_resolved topk_correct / trials')
    for gap, cap, proxy in product(data['gaps'], data['budgets'], (False, True)):
        having = cells[gap, cap, proxy, 'having']
        topk = cells[gap, cap, proxy, 'topk']
        print(gap, cap, proxy, having['correct'], having['focal_resolved'], topk['correct'], '/', having['trials'])
    print('total wrong:', sum(cell['wrong'] for cell in cells.values()),
          'covered:', sum(cell['coverage'] for cell in cells.values()), '/', len(data['runs']))

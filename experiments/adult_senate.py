"""Evaluate SENATE at the matched Adult replay budgets."""
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from stratq.engine import StratQEngine
from stratq.senate import senate_fixed_counts


def main():
    snapshot = (ROOT/'results/adult_fixed_budget.json').read_bytes()
    prior = json.loads(snapshot)
    profile = StratQEngine(ROOT/'data/adult_test_replay.parquet', group_col='grp',
                           proxy_col='proxy', oracle_col='oracle').profile
    groups = dict(zip(profile.groups, profile.group_row_indices))
    truth = prior['truth']
    assert groups.keys() == truth.keys() and all(
        len(ids) == truth[g]['size'] and profile.oracle_true[ids].sum() == truth[g]['positives']
        for g, ids in groups.items())
    runs = []
    for row in prior['runs']:
        if row['method'] != 'stratq':
            continue
        budget, seed = row['budget'], row['seed']
        groups_out = senate_fixed_counts(groups, lambda ids: profile.oracle_true[ids], budget=budget,
                                          seed=seed, delta=prior['delta'])
        assert sum(v['calls'] for v in groups_out.values()) == budget
        runs.append(dict(budget=budget, seed=seed, groups=groups_out))
    data = dict(truth=truth, delta=prior['delta'], source_sha256=hashlib.sha256(snapshot).hexdigest(), runs=runs)
    (ROOT/'results/adult_senate.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for budget in sorted({r['budget'] for r in runs}):
        trials = [r['groups'] for r in runs if r['budget'] == budget]
        widths = [max(100*(x['upper']-x['lower'])/(2*truth[g]['size']) for g, x in t.items()) for t in trials]
        covered = sum(all(v['lower'] <= truth[g]['positives'] <= v['upper'] for g, v in t.items()) for t in trials)
        print(budget, f'{mean(widths):.2f} pp', f'covered: {covered}/{len(trials)}')


if __name__ == '__main__':
    main()

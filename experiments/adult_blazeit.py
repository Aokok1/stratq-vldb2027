"""Compare the BlazeIt-derived fixed-look operator at the Adult call budgets."""
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from stratq.blazeit import blazeit_fixed_counts
from stratq.engine import StratQEngine


def main():
    prior_path = ROOT / 'results/adult_fixed_budget.json'
    prior_bytes = prior_path.read_bytes()
    prior = json.loads(prior_bytes)
    engine = StratQEngine(ROOT / 'data/adult_test_replay.parquet', group_col='grp',
                          proxy_col='proxy', oracle_col='oracle')
    profile = engine.profile
    groups = dict(zip(profile.groups, profile.group_row_indices))
    truth = prior['truth']
    if truth.keys() != groups.keys() or any(
        len(ids) != truth[group]['size'] or profile.oracle_true[ids].sum() != truth[group]['positives']
        for group, ids in groups.items()
    ):
        raise ValueError('Replay population differs from fixed-budget comparator population')
    runs = []
    for baseline in prior['runs']:
        if baseline['method'] != 'stratq':
            continue
        budget, seed = baseline['budget'], baseline['seed']
        actual_calls = 0

        def oracle(rows):
            nonlocal actual_calls
            actual_calls += len(rows)
            return profile.oracle_true[rows]

        outputs = blazeit_fixed_counts(profile.phi_all[:, 1], groups, oracle,
                                       budget=budget, delta=prior['delta'], seed=seed)
        assert actual_calls == sum(x['calls'] for x in outputs.values()) == budget
        covered = True
        widths = []
        for group, intervals in outputs.items():
            size, total = truth[group]['size'], truth[group]['positives']
            assert 0 <= intervals['lower'] <= intervals['upper'] <= size
            covered = covered and bool(intervals['lower'] <= total <= intervals['upper'])
            widths.append(float(intervals['upper'] - intervals['lower']) / (2 * size))
        runs.append({'seed': seed, 'budget': budget, 'calls': actual_calls, 'covered': covered,
                     'max_rate_half_width': max(widths), 'groups': outputs})
    summary = []
    for budget in sorted({r['budget'] for r in runs}):
        selected = [r for r in runs if r['budget'] == budget]
        assert len({r['seed'] for r in selected}) == len(selected) == 30
        summary.append({'budget': budget, 'trials': len(selected), 'covered': sum(r['covered'] for r in selected),
                        'mean_max_rate_half_width': mean(r['max_rate_half_width'] for r in selected)})
    report = {'method': 'blazeit_control_variate_fixed_look', 'delta': prior['delta'], 'pilot': 16,
              'source_doi': '10.14778/3372716.3372725', 'source_sha256': hashlib.sha256(prior_bytes).hexdigest(),
              'truth': truth, 'summary': summary, 'runs': runs}
    (ROOT / 'results/adult_blazeit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

"""Equal-budget comparison on the Adult label-replay workload."""
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from stratq.engine import StratQEngine
from stratq.proxy_stratified import fixed_budget_counts


def main():
    adult = json.loads((ROOT / 'results/adult_count_replay.json').read_text(encoding='utf-8'))
    engine = StratQEngine(ROOT / 'data/adult_test_replay.parquet', group_col='grp',
                          proxy_col='proxy', oracle_col='oracle')
    p = engine.profile
    groups = dict(zip(p.groups, p.group_row_indices))
    truth = adult['truth']
    if truth.keys() != groups.keys() or any(len(ids) != truth[g]['size'] or
                                           p.oracle_true[ids].sum() != truth[g]['positives']
                                           for g, ids in groups.items()):
        raise ValueError('Replay population does not match recorded counts')
    runs = []
    for budget in (2048, 4096, 8192):
        for seed in range(1000, 1030):
            methods = {'proxy_stratified_fixed': fixed_budget_counts(
                p.phi_all[:, 1], groups, lambda ids: p.oracle_true[ids], budget=budget,
                delta=adult['delta'], seed=seed)}
            result = engine.execute_grouped_agg(eps=0.0, delta=adult['delta'], agg_type='count',
                                                batch_size=32, pilot_per_group=16,
                                                max_oracle_budget=budget, seed=seed,
                                                use_control_variates=True, use_covariates=False)
            methods['stratq'] = {g: {'estimate': result.estimates[g], 'lower': result.lower[g],
                                    'upper': result.upper[g], 'calls': result.oracle_calls_by_group[g]}
                                 for g in result.groups}
            for method, outputs in methods.items():
                calls = sum(x['calls'] for x in outputs.values())
                if calls != budget or outputs.keys() != truth.keys():
                    raise ValueError(f'{method}, {seed}, {budget}: mismatched budget or groups')
                widths = [(x['upper'] - x['lower']) / (2 * truth[g]['size']) for g, x in outputs.items()]
                errors = [abs(x['estimate'] - truth[g]['positives']) / truth[g]['size'] for g, x in outputs.items()]
                covered = all(x['lower'] - 1e-9 <= truth[g]['positives'] <= x['upper'] + 1e-9 for g, x in outputs.items())
                runs.append({'method': method, 'budget': budget, 'seed': seed, 'calls': calls, 'covered': covered,
                             'max_rate_half_width': max(widths), 'max_rate_error': max(errors), 'groups': outputs})
    summary = []
    for budget in (2048, 4096, 8192):
        for method in ('proxy_stratified_fixed', 'stratq'):
            records = [r for r in runs if r['method'] == method and r['budget'] == budget]
            summary.append({'budget': budget, 'method': method, 'trials': len(records),
                            'covered': sum(r['covered'] for r in records),
                            'mean_max_rate_half_width': mean(r['max_rate_half_width'] for r in records),
                            'mean_max_rate_error': mean(r['max_rate_error'] for r in records)})
    report = {'source': adult['source'], 'archive_sha256': adult['archive_sha256'], 'truth': truth,
              'delta': adult['delta'], 'summary': summary, 'runs': runs}
    (ROOT / 'results/adult_fixed_budget.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

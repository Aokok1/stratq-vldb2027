"""Measure a globally randomized Online Aggregation scan on Adult replay."""
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from stratq.engine import StratQEngine
from stratq.online_aggregation import online_fixed_counts


def main():
    source = (ROOT/'results/adult_fixed_budget.json').read_bytes()
    base = json.loads(source)
    truth, delta = base['truth'], base['delta']
    profile = StratQEngine(ROOT/'data/adult_test_replay.parquet', group_col='grp', proxy_col='proxy', oracle_col='oracle').profile
    groups = dict(zip(profile.groups, profile.group_row_indices))
    if groups.keys() != truth.keys() or any((len(ids), int(profile.oracle_true[ids].sum())) !=
        (truth[g]['size'], truth[g]['positives']) for g, ids in groups.items()):
        raise ValueError('Mismatched population')
    runs = []
    for row in base['runs']:
        if row['method'] != 'stratq':
            continue
        b, s = row['budget'], row['seed']
        outputs = online_fixed_counts(groups, lambda ids: profile.oracle_true[ids], budget=b, seed=s, delta=delta)
        if sum(v['calls'] for v in outputs.values()) != b:
            raise ValueError('Mismatched call budget')
        runs.append(dict(budget=b, seed=s, groups=outputs))
    output = dict(truth=truth, delta=delta, source_sha256=hashlib.sha256(source).hexdigest(), runs=runs)
    (ROOT/'results/adult_online_aggregation.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for budget in sorted({r['budget'] for r in runs}):
        rows = [r['groups'] for r in runs if r['budget'] == budget]
        widths = [max(100*(v['upper']-v['lower'])/(2*truth[g]['size']) for g,v in row.items()) for row in rows]
        covered = sum(all(v['lower'] <= truth[g]['positives'] <= v['upper'] for g,v in row.items()) for row in rows)
        unsampled = sum(any(v['calls'] == 0 for v in row.values()) for row in rows)
        print(budget, f'{mean(widths):.2f} pp', f'covered {covered}/{len(rows)}', f'unsampled groups in {unsampled}/{len(rows)}')


if __name__ == '__main__':
    main()

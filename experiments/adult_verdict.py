"""Adult replay using VerdictDB-style error estimation on Senate samples."""
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from stratq.engine import StratQEngine
from stratq.verdict import verdict_fixed_counts


def main():
    source = (ROOT/'results/adult_senate.json').read_bytes()
    base = json.loads(source)
    truth, delta = base['truth'], base['delta']
    p = StratQEngine(ROOT/'data/adult_test_replay.parquet', group_col='grp', proxy_col='proxy', oracle_col='oracle').profile
    groups = dict(zip(p.groups, p.group_row_indices))
    if groups.keys() != truth.keys() or any((len(ids), int(p.oracle_true[ids].sum())) !=
        (truth[g]['size'], truth[g]['positives']) for g,ids in groups.items()):
        raise ValueError('Mismatched population')
    runs = []
    for row in base['runs']:
        budget, seed = row['budget'], row['seed']
        out = verdict_fixed_counts(groups, lambda ids: p.oracle_true[ids], budget=budget, delta=delta, seed=seed)
        runs.append(dict(budget=budget, seed=seed, groups=out))
    data = dict(truth=truth, delta=delta, source_sha256=hashlib.sha256(source).hexdigest(), runs=runs)
    (ROOT/'results/adult_verdict.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for budget in sorted({r['budget'] for r in runs}):
        rows = [r['groups'] for r in runs if r['budget'] == budget]
        widths = [max(100*(v['upper']-v['lower'])/(2*truth[g]['size']) for g,v in row.items()) for row in rows]
        covered = sum(all(v['lower'] <= truth[g]['positives'] <= v['upper'] for g,v in row.items()) for row in rows)
        print(budget, f'{mean(widths):.2f} pp', f'covered {covered}/{len(rows)}')


if __name__ == '__main__':
    main()

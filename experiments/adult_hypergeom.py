"""Compare a grouped hypergeometric confidence sequence to saved Adult runs."""
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import scipy
from zipfile import ZipFile

from adult_count_replay import records

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from stratq.hypergeom import count_interval


def main():
    reference = json.loads((ROOT / 'results/adult_count_replay.json').read_text(encoding='utf-8'))
    path = ROOT / 'data/adult_uci.zip'
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference['archive_sha256']:
        raise ValueError('Adult archive differs from paired runs')
    with ZipFile(path) as archive:
        columns = {}
        for row in records(archive, 'adult.test'):
            columns.setdefault(row[1], []).append(int(row[14].rstrip('.') == '>50K'))
    labels = {g: np.array(columns[g], dtype=np.int64) for g in sorted(columns)}
    truth = reference['truth']
    if {g: (len(y), int(y.sum())) for g, y in labels.items()} != {g: (v['size'], v['positives']) for g, v in truth.items()}:
        raise ValueError('Adult grouped truths differ')
    alpha = reference['delta'] / len(labels)
    batch, pilot = reference['batch_size'], reference['pilot_per_group']
    base = {r['seed']: r['calls'] for r in reference['runs'] if r['mode'] == 'no_proxy'}
    proxy = {r['seed']: r['calls'] for r in reference['runs'] if r['mode'] == 'proxy_only'}
    runs = []
    for seed in sorted(base):
        rng = np.random.default_rng(seed)
        groups = {}
        for g, outcomes in labels.items():
            n = len(outcomes)
            order = rng.permutation(n)
            low, high, successes, sampled = 0, n, 0, 0
            while (high - low) / 2 > truth[g]['epsilon']:
                end = min(n, sampled + (pilot if sampled == 0 else batch))
                successes += int(outcomes[order[sampled:end]].sum())
                sampled = end
                lo, hi = count_interval(n, sampled, successes, alpha)
                low, high = max(low, lo), min(high, hi)
                if low > high:
                    low, high = successes, successes + n - sampled
            groups[g] = {'lower': low, 'upper': high, 'calls': sampled}
        calls = sum(v['calls'] for v in groups.values())
        runs.append({'seed': seed, 'calls': calls, 'vs_no_proxy_calls': calls - base[seed],
                     'vs_proxy_only_calls': calls - proxy[seed], 'groups': groups,
                     'covered': all(v['lower'] <= truth[g]['positives'] <= v['upper'] for g, v in groups.items())})
    summary = {'mean_calls': mean(r['calls'] for r in runs), 'covered_trials': sum(r['covered'] for r in runs),
               'trials': len(runs), 'mean_delta_from_no_proxy': mean(r['vs_no_proxy_calls'] for r in runs),
               'mean_delta_from_proxy_only': mean(r['vs_proxy_only_calls'] for r in runs)}
    report = {k: reference[k] for k in ('source', 'archive_sha256', 'delta', 'batch_size', 'pilot_per_group', 'truth')}
    report.update({'method': 'Grouped hypergeometric prior-posterior confidence sequence', 'scipy': scipy.__version__,
                   'numpy': np.__version__, 'summary': summary, 'runs': runs})
    (ROOT / 'results/adult_hypergeom.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

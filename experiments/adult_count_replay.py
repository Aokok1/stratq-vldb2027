"""Paired grouped-count label replay using the official UCI Adult split."""
import argparse
import csv
import hashlib
import io
import json
import platform
import sys
from pathlib import Path
from zipfile import ZipFile

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from stratq.engine import StratQEngine

URL = 'https://archive.ics.uci.edu/static/public/2/adult.zip'


def records(archive, member):
    with archive.open(member) as raw:
        for row in csv.reader(io.TextIOWrapper(raw, encoding='ascii'), skipinitialspace=True):
            if not row or row[0].startswith('|'):
                continue
            if len(row) != 15 or row[14].rstrip('.') not in {'>50K', '<=50K'}:
                raise ValueError(f'Invalid Adult record in {member}')
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=30)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    archive_path = ROOT / 'data/adult_uci.zip'
    train_counts = {}
    with ZipFile(archive_path) as archive:
        for row in records(archive, 'adult.data'):
            counts = train_counts.setdefault(int(row[4]), [0, 0])
            counts[0] += int(row[14] == '>50K')
            counts[1] += 1
        test = list(records(archive, 'adult.test'))
    train_size = sum(n for _, n in train_counts.values())
    if (train_size, len(test)) != (32561, 16281):
        raise ValueError('Unexpected UCI Adult split sizes')
    education = [int(row[4]) for row in test]
    # Smoothed training-set positive rate by education; no test labels are used here.
    proxy = [(train_counts[e][0] + 1) / (train_counts[e][1] + 2) for e in education]
    table = pa.table({'grp': [row[1] for row in test], 'proxy': proxy,
                      'age': [int(row[0]) for row in test],
                      'hours': [int(row[12]) for row in test],
                      'oracle': [int(row[14].rstrip('.') == '>50K') for row in test]})
    path = ROOT / 'data/adult_test_replay.parquet'
    pq.write_table(table, path)
    engine = StratQEngine(path, group_col='grp', proxy_col='proxy',
                          covariate_cols=['age', 'hours'], oracle_col='oracle')
    with duckdb.connect() as con:
        con.register('adult_replay', table)
        truth_rows = con.sql('SELECT grp, count(*), sum(oracle) FROM adult_replay GROUP BY grp ORDER BY grp').fetchall()
    truth = {g: count for g, _, count in truth_rows}
    # Error budget is 4 percentage points on the positive rate of each group.
    eps = {g: 0.04 * n for g, n, _ in truth_rows}
    runs = []
    for seed in range(1000, 1000 + args.trials):
        for mode, predictor, covariates in [('no_proxy', False, False), ('proxy_only', True, False), ('proxy_covariates', True, True)]:
            result = engine.execute_grouped_agg(
                eps=eps, delta=0.1, agg_type='count', batch_size=32, pilot_per_group=16,
                seed=seed, use_control_variates=predictor, use_covariates=covariates,
            )
            if set(result.groups) != set(truth) or sum(result.oracle_calls_by_group.values()) != result.total_oracle_calls:
                raise AssertionError('Inconsistent groups or call accounting')
            groups = {g: {'estimate': result.estimates[g], 'lower': result.lower[g],
                          'upper': result.upper[g], 'calls': result.oracle_calls_by_group[g]}
                      for g in result.groups}
            runs.append({'mode': mode, 'seed': seed, 'calls': result.total_oracle_calls,
                         'covered': all(result.lower[g] - 1e-9 <= truth[g] <= result.upper[g] + 1e-9 for g in truth),
                         'target_met': all(result.half_widths[g] <= eps[g] + 1e-9 for g in truth),
                         'groups': groups})
    base = {r['seed']: r['calls'] for r in runs if r['mode'] == 'no_proxy'}
    summary = {}
    for mode in sorted({r['mode'] for r in runs}):
        subset = [r for r in runs if r['mode'] == mode]
        diffs = [r['calls'] - base[r['seed']] for r in subset]
        summary[mode] = {'mean_calls': float(np.mean([r['calls'] for r in subset])),
                         'covered_trials': sum(r['covered'] for r in subset),
                         'target_met_trials': sum(r['target_met'] for r in subset),
                         'mean_paired_call_delta': float(np.mean(diffs)),
                         'paired_delta_range': [min(diffs), max(diffs)]}
    report = {'source': URL, 'doi': '10.24432/C5XW20', 'license': 'CC-BY-4.0',
              'archive_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
              'mode': 'label_replay', 'train_rows': train_size, 'test_rows': len(test),
              'delta': 0.1, 'batch_size': 32, 'pilot_per_group': 16,
              'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                           'duckdb': duckdb.__version__, 'pyarrow': pa.__version__},
              'truth': {g: {'size': n, 'positives': count, 'epsilon': eps[g]} for g, n, count in truth_rows},
              'summary': summary, 'runs': runs}
    (ROOT / 'results/adult_count_replay.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

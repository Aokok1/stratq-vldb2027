"""Measure in-memory StratQ profiling and query costs on synthetic Parquet."""
import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from time import perf_counter

import duckdb
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from stratq.engine import StratQEngine

GROUPS = 20
DATA_SEED = 1729
QUERY_SEEDS = (2401, 2402, 2403)
SCHEMA = pa.schema([('grp', pa.string()), ('proxy', pa.float32()), ('label', pa.uint8())])


def generate(path, n):
    """Stream batches and accumulate ground truth during generation."""
    rng = np.random.default_rng(DATA_SEED)
    weights = np.arange(1, GROUPS + 1, dtype=float) ** -1.1
    weights /= weights.sum()
    prevalence = np.linspace(0.12, 0.62, GROUPS)
    names = pa.array([f'G{g:02d}' for g in range(GROUPS)])
    sizes = np.zeros(GROUPS, dtype=np.int64)
    positives = np.zeros_like(sizes)
    with pq.ParquetWriter(path, SCHEMA, compression='zstd') as writer:
        for offset in range(0, n, 250_000):
            size = min(250_000, n - offset)
            groups = rng.choice(GROUPS, size=size, p=weights)
            latent = rng.random(size)
            labels = (latent < prevalence[groups]).astype(np.uint8)
            score = np.clip(1 - latent + rng.normal(0, 0.25, size), 0, 1).astype(np.float32)
            writer.write_batch(pa.record_batch([names.take(pa.array(groups)), pa.array(score), pa.array(labels)], schema=SCHEMA))
            sizes += np.bincount(groups, minlength=GROUPS)
            positives += np.bincount(groups[labels == 1], minlength=GROUPS)
    return {g: (int(sizes[i]), int(positives[i])) for i, g in enumerate(names.to_pylist())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, choices=(100_000, 1_000_000, 10_000_000), required=True)
    n = parser.parse_args().rows
    path = ROOT / 'data' / f'scale_{n}.parquet'
    host = {'os': platform.platform(), 'cpu': os.environ.get('PROCESSOR_IDENTIFIER', platform.processor()),
            'logical_cpus': os.cpu_count(), 'ram_bytes': psutil.virtual_memory().total,
            'available_ram_bytes_at_start': psutil.virtual_memory().available,
            'disk_free_bytes_at_start': shutil.disk_usage(path.parent).free,
            'python': platform.python_version(), 'numpy': np.__version__,
            'duckdb': duckdb.__version__, 'pyarrow': pa.__version__}
    t0 = perf_counter()
    truth = generate(path, n)
    generation_ms = (perf_counter() - t0) * 1000
    with duckdb.connect() as con:
        t0 = perf_counter()
        rows = con.read_parquet(str(path)).aggregate('grp, count(*) AS n, sum(label) AS positives', 'grp').fetchall()
        full_sql_ms = (perf_counter() - t0) * 1000
    # Independent SQL check against the generator's integer counters.
    assert {group: (size, positives) for group, size, positives in rows} == truth
    t0 = perf_counter()
    engine = StratQEngine(path, group_col='grp', proxy_col='proxy', oracle_col='label')
    profile_ms = (perf_counter() - t0) * 1000
    eps = {g: 0.04 * size for g, (size, _) in truth.items()}
    runs = []
    for j, seed in enumerate(QUERY_SEEDS):
        for use_proxy in ((False, True) if j % 2 == 0 else (True, False)):
            result = engine.execute_grouped_agg(eps=eps, delta=0.1, agg_type='count', batch_size=128,
                                                pilot_per_group=32, seed=seed,
                                                use_control_variates=use_proxy, use_covariates=False)
            groups = {g: {'lower': result.lower[g], 'upper': result.upper[g], 'calls': result.oracle_calls_by_group[g]}
                      for g in result.groups}
            assert set(groups) == set(truth)
            assert sum(v['calls'] for v in groups.values()) == result.total_oracle_calls
            runs.append({'seed': seed, 'use_proxy': use_proxy, 'calls': result.total_oracle_calls,
                         'wall_ms': result.wall_ms, 'replay_ms': result.oracle_ms,
                         'covered': all(v['lower'] - 1e-8 <= truth[g][1] <= v['upper'] + 1e-8 for g, v in groups.items()),
                         'target_met': all((v['upper'] - v['lower']) / 2 <= eps[g] + 1e-8 for g, v in groups.items()),
                         'groups': groups})
    mem = psutil.Process().memory_info()
    report = {'rows': n, 'data_seed': DATA_SEED, 'delta': 0.1, 'rate_tolerance': 0.04, 'batch_size': 128,
              'pilot_per_group': 32, 'host': host, 'parquet_bytes': path.stat().st_size,
              'generation_ms': generation_ms, 'full_label_sql_ms': full_sql_ms, 'profile_ms': profile_ms,
              'peak_working_set_bytes': getattr(mem, 'peak_wset', None), 'truth': truth, 'runs': runs}
    (ROOT / 'results' / f'scale_{n}.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'{n:,} rows, profile {profile_ms:.1f} ms, peak working set {getattr(mem, "peak_wset", None)} bytes')
    for use_proxy in (False, True):
        subset = [r for r in runs if r['use_proxy'] == use_proxy]
        print(f'  {use_proxy=}: calls {[r["calls"] for r in subset]}, median query {np.median([r["wall_ms"] for r in subset]):.1f} ms, covered {sum(r["covered"] for r in subset)}/3')


if __name__ == '__main__':
    main()

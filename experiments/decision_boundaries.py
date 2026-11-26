"""Controlled threshold and rank boundaries on fixed synthetic snapshots."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from stratq.engine import StratQEngine

GAPS = (0, 1, 16, 64)
BUDGETS = (256, 768, 1024)
SEEDS = tuple(range(1000, 1020))
TAU = 128
K = 2


def run():
    trials = []
    populations = {}
    for gap in GAPS:
        totals = dict(A=192, B=TAU, C=TAU-gap, D=32)
        group = np.repeat(list(totals), 256)
        y = np.concatenate([np.arange(256) < total for total in totals.values()]).astype(int)
        rng = np.random.default_rng(813)
        proxy = np.clip(0.2 + 0.6*y + rng.normal(0, 0.25, len(y)), 0, 1)
        observed = {g: int(y[group == g].sum()) for g in totals}
        assert observed == totals
        populations[str(gap)] = observed
        calls = []

        def oracle(rows):
            calls.extend(rows.tolist())
            return y[rows]

        engine = StratQEngine(pa.table({'grp': group, 'proxy': proxy}),
                              group_col='grp', proxy_col='proxy', oracle_fn=oracle)
        for cap in BUDGETS:
            for seed in SEEDS:
                for use_proxy in (False, True):
                    options = dict(agg_type='count', seed=seed, delta=0.1, batch_size=16,
                                   pilot_per_group=8, max_oracle_budget=cap,
                                   use_control_variates=use_proxy, use_covariates=False,
                                   indifference_eps=0)
                    for query in ('having', 'topk'):
                        calls.clear()
                        if query == 'having':
                            out = engine.execute_grouped_having(tau=TAU, **options)
                            decision = dict(accepted=out.accepted_groups, rejected=out.rejected_groups,
                                            unresolved=out.uncertain_groups)
                        else:
                            out = engine.execute_grouped_topk(k=K, **options)
                            decision = dict(certified=out.certified, selected=out.topk_groups)
                        assert len(calls) == len(set(calls)) == out.total_oracle_calls <= cap
                        trials.append(dict(gap=gap, budget=cap, seed=seed, proxy=use_proxy, query=query,
                                           calls=len(calls), group_calls=out.oracle_calls_by_group,
                                           lower=out.lower, upper=out.upper, **decision))
    return dict(populations=populations, runs=trials, delta=0.1, group_size=256,
                threshold=TAU, k=K, pilot=8, batch_size=16, population_seed=813,
                gaps=GAPS, budgets=BUDGETS, seeds=SEEDS)


if __name__ == '__main__':
    report = run()
    (ROOT / 'results/decision_boundaries.json').write_text(
        json.dumps(report, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    print(f"Recorded {len(report['runs'])} trials.")

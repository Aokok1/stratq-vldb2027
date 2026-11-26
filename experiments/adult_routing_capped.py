"""Adult allocation ablation at three fixed reference-call caps."""
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from stratq.engine import StratQEngine


def metrics(groups, truth):
    widths = [(v['upper']-v['lower'])/(2*truth[g]['size']) for g, v in groups.items()]
    return (max(widths), sum(w <= .04+1e-12 for w in widths),
            all(v['lower']-1e-9 <= truth[g]['positives'] <= v['upper']+1e-9 for g, v in groups.items()))


def main():
    source = (ROOT/'results/adult_fixed_budget.json').read_bytes()
    original = json.loads(source)
    truth, delta = original['truth'], original['delta']
    engine = StratQEngine(ROOT/'data/adult_test_replay.parquet', group_col='grp', proxy_col='proxy', oracle_col='oracle')
    prof = engine.profile
    if set(prof.groups) != set(truth) or any((len(ids), int(prof.oracle_true[ids].sum())) !=
        (truth[g]['size'], truth[g]['positives']) for g, ids in zip(prof.groups, prof.group_row_indices)):
        raise ValueError('Adult replay does not match source truth')
    runs = []
    for base in original['runs']:
        if base['method'] != 'stratq':
            continue
        budget, seed = base['budget'], base['seed']
        for proxy in (False, True):
            for adaptive in (False, True):
                r = engine.execute_grouped_agg(agg_type='count', eps=0., delta=delta, seed=seed,
                                               batch_size=32, pilot_per_group=16, max_oracle_budget=budget,
                                               use_control_variates=proxy, use_covariates=False,
                                               use_adaptive_alloc=adaptive)
                if r.total_oracle_calls != budget:
                    raise ValueError('Run did not consume the declared cap')
                groups = {g: dict(calls=int(r.oracle_calls_by_group[g]), lower=r.lower[g], upper=r.upper[g]) for g in r.groups}
                runs.append(dict(budget=budget, seed=seed, proxy=proxy, adaptive=adaptive, groups=groups))
    report = dict(source_sha256=hashlib.sha256(source).hexdigest(), truth=truth, delta=delta,
                  batch=32, pilot=16, runs=runs)
    (ROOT/'results/adult_routing_capped.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for budget in sorted({r['budget'] for r in runs}):
        for proxy in (False, True):
            rr, ad = (sorted((r for r in runs if (r['budget'], r['proxy'], r['adaptive']) == (budget, proxy, a)),
                              key=lambda r: r['seed']) for a in (False, True))
            m0, m1 = ([metrics(r['groups'], truth) for r in arm] for arm in (rr, ad))
            print(budget, 'proxy', proxy, 'round-robin -> width-first',
                  [(round(mean(m[0] for m in ms)*100, 2), round(mean(m[1] for m in ms), 2), sum(m[2] for m in ms))
                   for ms in (m0, m1)], 'width-first lower/equal/higher',
                  tuple(sum(p(a[0], b[0]) for a, b in zip(m1, m0)) for p in
                        (lambda a, b: a < b-1e-12, lambda a, b: abs(a-b) <= 1e-12, lambda a, b: a > b+1e-12)))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()

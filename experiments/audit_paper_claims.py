"""Audit the draft's selected quantitative claims against raw result records."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from audit_decisions import audit as decision_audit
from audit_fashion import audit as fashion_audit
from audit_latency import audit as latency_audit
from audit_scale import audit as scale_audit
from audit_sms import audit as sms_audit

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / '4_LaTeX_Source' / 'vldb_template'


def main():
    inputs = {}
    checks = []

    def read(path):
        payload = path.read_bytes()
        key = str(path.relative_to(PAPER)) if path.is_relative_to(PAPER) else str(path)
        inputs[key] = 'sha256:' + hashlib.sha256(payload).hexdigest()
        return payload.decode('utf-8')

    def result(name):
        return json.loads(read(ROOT / 'results' / f'{name}.json'))

    def record(claim, passed):
        checks.append({'claim': claim, 'status': 'match' if passed else 'mismatch'})

    def text_claim(source, evidence, fragment):
        record({'file': source, 'evidence': evidence, 'expected_fragment': fragment}, fragment in tex[source])

    tex = {path.name: read(path) for path in sorted(PAPER.glob('stratq*.tex'))}
    abstract = 'stratq.tex'
    evaluation = 'stratq_evaluation.tex'
    adult = result('adult_count_replay')
    sms = result('sms_live')
    fashion = result('fashion_live')
    large = result('scale_10000000')
    hyper = result('adult_hypergeom')
    fixed = result('adult_fixed_budget')
    fs = fashion_audit(fashion)
    ss = sms_audit(sms)
    scale_points = []
    for size in (100_000, 1_000_000, 10_000_000):
        path = ROOT / 'results' / f'scale_{size}.json'
        read(path)
        scale_summary = scale_audit(path, size)
        scale_points.append({'rows': size, 'profile_seconds': scale_summary['profile_ms'] / 1000,
                             'peak_gib': scale_summary['peak_working_set_bytes'] / 2**30})
        record(f'{size} rows: all intervals cover and attain target',
               scale_summary['covered_trials'] == scale_summary['target_met_trials'] == scale_summary['trials'])

    manifest = json.loads(read(ROOT / 'figures' / 'figure_metrics.json'))
    record('Parquet scaling: plotted times and peak memory', scale_points == manifest['metrics']['parquet_scale'])
    record('All documented workloads: nominal 90 percent joint coverage',
           all(data['delta'] == 0.1 for data in (adult, sms, fashion, large, hyper, fixed)))
    record('Grouped COUNT: documented per-group tolerance',
           fashion['rate_tolerance'] == large['rate_tolerance'] == 0.04
           and all(v['epsilon'] == 0.04 * v['size'] for v in adult['truth'].values())
           and all(sms['epsilon'][g] == 0.06 * size for g, size in sms['sizes'].items()))
    for source, digest in manifest['source_sha256'].items():
        path = ROOT / 'results' / f'{source}.json'
        read(path)
        record(f'{source}: figure source hash', inputs[str(path)] == 'sha256:' + digest)

    adult_base = [r for r in adult['runs'] if r['mode'] == 'no_proxy']
    adult_proxy = [r for r in adult['runs'] if r['mode'] == 'proxy_only']
    rows = [
        ('Adult', adult_base, adult_proxy, 'adult_count_replay'),
        ('SMS', [r for r in sms['runs'] if r['query'] == 'count' and not r['proxy']],
         [r for r in sms['runs'] if r['query'] == 'count' and r['proxy']], 'sms_live'),
        ('Fashion-MNIST', [r for r in fashion['runs'] if not r['proxy']],
         [r for r in fashion['runs'] if r['proxy']], 'fashion_live'),
        ('Synthetic 10M', [r for r in large['runs'] if not r['use_proxy']],
         [r for r in large['runs'] if r['use_proxy']], 'scale_10000000'),
    ]
    for name, control, proxy, source in rows:
        c, p = ({r['seed']: r['calls'] for r in arm} for arm in (control, proxy))
        change = 100 * (mean(p.values()) / mean(c.values()) - 1)
        plotted = manifest['metrics']['proxy_calls'][name]
        record(f'{name}: matched seeds and plotted call counts', sorted(c) == sorted(p) == plotted['seeds']
               and plotted['no_proxy_calls'] == [c[seed] for seed in sorted(c)]
               and plotted['proxy_calls'] == [p[seed] for seed in sorted(p)]
               and abs(change - plotted['mean_calls_change_pct']) < 1e-9)
        label = 'a 10 million-row synthetic snapshot' if name == 'Synthetic 10M' else name
        text_claim(abstract, source, f'${change:+.2f}\\%$ on {label}')
    pair_counts = ', '.join(str(len(control)) for _, control, _, _ in rows)
    first, last = pair_counts.rsplit(', ', 1)
    text_claim(evaluation, 'four workload records', f'{first}, and {last} seed pairs')

    for source, fragment in [
        ('adult_count_replay', f"{adult['test_rows']:,}-row test split"),
        ('sms_live', f"{sms['test_rows']:,} held-out messages"),
        ('fashion_live', f"{fashion['test_rows']:,} official test images"),
        ('adult_hypergeom', f"{mean(r['calls'] for r in hyper['runs']):,.2f} mean calls"),
        ('adult_count_replay', f"{mean(r['calls'] for r in adult_base):,.2f} mean calls"),
        ('scale_10000000', f"{large['profile_ms'] / 1000:.2f} seconds"),
        ('scale_10000000', f"{large['peak_working_set_bytes'] / 2**30:.2f} GiB"),
        ('scale_10000000', f"{median(r['wall_ms'] for r in large['runs'] if r['use_proxy']) / 1000:.2f} seconds with the proxy"),
        ('sms_live', f"At threshold {sms['having_threshold']}"),
    ]:
        text_claim(evaluation, source, fragment)
    for query, summary in ss.items():
        text_claim(evaluation, 'sms_live', f"from {summary['False']['mean_calls']:.1f} to {summary['True']['mean_calls']:.1f}")
    record('SMS: full threshold resolution and certified top-k',
           all(r['decision']['correct'] and not r['decision']['uncertain'] for r in sms['runs'] if r['query'] == 'having')
           and all(r['decision']['correct'] and r['decision']['certified'] for r in sms['runs'] if r['query'] == 'topk'))
    record('Adaptive grouped COUNT: observed joint coverage and target attainment',
           all(set(r['groups']) == set(adult['truth']) and
               all(v['lower'] - 1e-9 <= adult['truth'][g]['positives'] <= v['upper'] + 1e-9
                   and (v['upper'] - v['lower']) / 2 <= adult['truth'][g]['epsilon'] + 1e-9
                   for g, v in r['groups'].items()) for r in adult['runs'])
           and all(r['covered'] and r['decision']['target_met'] for r in sms['runs'] if r['query'] == 'count')
           and all(r['covered_trials'] == r['width_met_trials'] == 5 for r in fs.values()))
    record('Adult hypergeometric: observed joint coverage and target attainment',
           len(hyper['runs']) == 30 and all(v['lower'] <= hyper['truth'][g]['positives'] <= v['upper']
               and (v['upper'] - v['lower']) / 2 <= hyper['truth'][g]['epsilon']
               for r in hyper['runs'] for g, v in r['groups'].items()))
    for method in ('proxy_stratified_fixed', 'stratq'):
        chart = manifest['metrics']['adult_fixed_budget'][method]
        for j, budget in enumerate((2048, 4096, 8192)):
            trials = [r for r in fixed['runs'] if r['method'] == method and r['budget'] == budget]
            widths = [100 * max((v['upper'] - v['lower']) / (2 * fixed['truth'][g]['size'])
                               for g, v in r['groups'].items()) for r in trials]
            record(f'Adult {method}, budget {budget}: plotted trials and mean', len(widths) == 30
                   and widths == chart['per_seed_pct'][str(budget)]
                   and abs(mean(widths) - chart['mean_max_rate_half_width_pct'][j]) < 1e-9)
            if budget == 4096:
                text_claim(evaluation, 'adult_fixed_budget', f'{mean(widths):.2f}')

    blazeit = result('adult_blazeit')
    fixed_path = ROOT / 'results' / 'adult_fixed_budget.json'
    record('BlazeIt adaptation uses the same fixed-budget population and seed schedule',
           blazeit['truth'] == fixed['truth'] and blazeit['delta'] == fixed['delta']
           and 'sha256:' + blazeit['source_sha256'] == inputs[str(fixed_path)]
           and sorted((r['budget'], r['seed']) for r in blazeit['runs']) ==
               sorted((r['budget'], r['seed']) for r in fixed['runs'] if r['method'] == 'stratq'))
    trials = [r for r in blazeit['runs'] if r['budget'] == 4096]
    widths = [100 * max((v['upper'] - v['lower']) / (2 * fixed['truth'][g]['size'])
                       for g, v in r['groups'].items()) for r in trials]
    record('BlazeIt adaptation: budget and observed coverage at 4096 calls',
           len(trials) == 30 and all(sum(v['calls'] for v in r['groups'].values()) == r['calls'] == 4096
               and set(r['groups']) == set(fixed['truth'])
               and all(v['lower'] <= fixed['truth'][g]['positives'] <= v['upper']
                       for g, v in r['groups'].items()) for r in trials))
    text_claim(evaluation, 'adult_blazeit',
               f'{mean(widths):.2f} percentage points, with coverage in all {len(trials)} trials')

    senate = result('adult_senate')
    schedule = [r for r in fixed['runs'] if r['method'] == 'stratq']
    record('Senate: same population, risk, and distinct budget-seed schedule',
           senate['truth'] == fixed['truth'] and senate['delta'] == fixed['delta']
           and 'sha256:' + senate['source_sha256'] == inputs[str(fixed_path)]
           and sorted((r['budget'], r['seed']) for r in senate['runs']) ==
               sorted((r['budget'], r['seed']) for r in schedule))
    sizes = {g: x['size'] for g, x in fixed['truth'].items()}
    record('Senate: fixed allocation, census endpoints, and valid intervals',
           all(r['groups'].keys() == sizes.keys() and sum(x['calls'] for x in r['groups'].values()) == r['budget']
               and all(0 <= x['calls'] <= sizes[g] and 0 <= x['lower'] <= x['upper'] <= sizes[g]
                       and (x['calls'] < sizes[g] or x['lower'] == x['upper'] == fixed['truth'][g]['positives'])
                       for g, x in r['groups'].items())
               and max(x['calls'] for x in r['groups'].values()) -
                   min(x['calls'] for g, x in r['groups'].items() if x['calls'] < sizes[g]) <= 1
               for r in senate['runs']))
    senate_rows = [r['groups'] for r in senate['runs'] if r['budget'] == 4096]
    senate_width = mean(max(100 * (x['upper'] - x['lower']) / (2 * sizes[g]) for g, x in row.items())
                       for row in senate_rows)
    senate_covered = sum(all(x['lower'] <= fixed['truth'][g]['positives'] <= x['upper'] for g, x in row.items())
                         for row in senate_rows)
    text_claim(evaluation, 'adult_senate', f'{senate_width:.2f} percentage points, '
               f'with coverage in {senate_covered} of {len(senate_rows)} trials')
    record('Senate: smaller mean maximum half-width than StratQ at 4096 calls',
           senate_width < 100*mean(r['max_rate_half_width'] for r in schedule if r['budget'] == 4096))

    online = result('adult_online_aggregation')
    record('Online Aggregation: source, risk and paired schedule',
           online['truth'] == fixed['truth'] and online['delta'] == fixed['delta']
           and 'sha256:' + online['source_sha256'] == inputs[str(fixed_path)]
           and sorted((r['budget'],r['seed']) for r in online['runs']) ==
               sorted((r['budget'],r['seed']) for r in schedule))
    record('Online Aggregation: fixed budget and feasible unsampled intervals',
           all(r['groups'].keys() == sizes.keys()
               and sum(v['calls'] for v in r['groups'].values()) == r['budget']
               and all(0 <= v['calls'] <= sizes[g] and 0 <= v['lower'] <= v['upper'] <= sizes[g]
                       and (v['calls'] > 0 or (v['lower'],v['upper']) == (0,sizes[g]))
                       for g,v in r['groups'].items()) for r in online['runs']))
    oa = [r['groups'] for r in online['runs'] if r['budget'] == 4096]
    oa_width = mean(max(100*(v['upper']-v['lower'])/(2*sizes[g]) for g,v in groups.items()) for groups in oa)
    oa_empty = sum(any(v['calls'] == 0 for v in groups.values()) for groups in oa)
    text_claim(evaluation, 'adult_online_aggregation', f'{oa_width:.2f} percentage points; '
               f'{oa_empty} of {len(oa)} prefixes leave at least one group unsampled')
    record('Online Aggregation: all 4096-call trials cover all groups',
           len(oa) == 30 and all(v['lower'] <= fixed['truth'][g]['positives'] <= v['upper']
                                 for groups in oa for g,v in groups.items()))
    text_claim(evaluation, 'adult_online_aggregation', f'All {len(oa)} trials cover all groups.')

    lws = result('adult_lws')
    lws_schedule = {(r['budget'],r['seed']):r['groups'] for r in lws['runs']}
    senate_schedule = {(r['budget'],r['seed']):r['groups'] for r in senate['runs']}
    record('LWS: source, score floor, nominal risk, and schedule',
           lws['truth'] == fixed['truth'] and lws['delta'] == fixed['delta'] and lws['score_floor'] == .05
           and 'sha256:' + lws['source_sha256'] == inputs[str(fixed_path)]
           and len(lws_schedule) == len(lws['runs']) and lws_schedule.keys() == senate_schedule.keys())
    record('LWS: Senate quotas and feasible interval endpoints',
           all(groups.keys() == sizes.keys() and all(0 <= v['lower'] <= v['upper'] <= sizes[g]
               and v['calls'] == senate_schedule[k][g]['calls'] for g,v in groups.items())
               for k, groups in lws_schedule.items()))
    ls = [groups for (b,_),groups in lws_schedule.items() if b == 4096]
    lw = mean(max(100*(v['upper']-v['lower'])/(2*sizes[g]) for g,v in groups.items()) for groups in ls)
    lc = sum(all(v['lower'] <= fixed['truth'][g]['positives'] <= v['upper'] for g,v in groups.items()) for groups in ls)
    text_claim(evaluation, 'adult_lws', f'{lw:.2f} percentage points, with joint coverage in {lc} of {len(ls)} trials')
    record('LWS: narrower mean maximum half-width than StratQ at 4096 calls',
           lw < 100*mean(r['max_rate_half_width'] for r in schedule if r['budget'] == 4096))

    verdict = result('adult_verdict')
    vd_schedule = {(r['budget'],r['seed']):r['groups'] for r in verdict['runs']}
    record('VerdictDB: Senate source, risk and matched budget schedule',
           verdict['truth'] == senate['truth'] and verdict['delta'] == senate['delta']
           and 'sha256:' + verdict['source_sha256'] == inputs[str(ROOT/'results/adult_senate.json')]
           and len(vd_schedule) == len(verdict['runs']) and vd_schedule.keys() == senate_schedule.keys())
    record('VerdictDB: feasible intervals and unchanged Senate call quotas',
           all(groups.keys() == sizes.keys() and all(0 <= v['lower'] <= v['upper'] <= sizes[g]
               and v['calls'] == senate_schedule[k][g]['calls'] for g,v in groups.items())
               for k, groups in vd_schedule.items()))
    vs = [groups for (b,_),groups in vd_schedule.items() if b == 4096]
    vw = mean(max(100*(v['upper']-v['lower'])/(2*sizes[g]) for g,v in groups.items()) for groups in vs)
    vc = sum(all(v['lower'] <= fixed['truth'][g]['positives'] <= v['upper'] for g,v in groups.items()) for groups in vs)
    text_claim(evaluation, 'adult_verdict', f'{vw:.2f} percentage points, with joint coverage in {vc} of {len(vs)} trials')

    for name, source, data, method in (
        ('StratQ', 'adult_fixed_budget', fixed, 'stratq'),
        ('ABae-inspired strata', 'adult_fixed_budget', fixed, 'proxy_stratified_fixed'),
        ('BlazeIt control variate', 'adult_blazeit', result('adult_blazeit'), None),
        ('Congressional Senate', 'adult_senate', senate, None),
        ('Online Aggregation scan', 'adult_online_aggregation', online, None),
        ('Learned weighted sampling', 'adult_lws', lws, None),
        ('VerdictDB subsampling', 'adult_verdict', verdict, None),
    ):
        rows = [r['groups'] for r in data['runs'] if r['budget'] == 4096 and r.get('method') == method]
        width = mean(max(50*(v['upper']-v['lower'])/sizes[g] for g,v in row.items()) for row in rows)
        hits = sum(all(v['lower'] <= fixed['truth'][g]['positives'] <= v['upper'] for g,v in row.items()) for row in rows)
        text_claim(evaluation, source, f'{name} & {width:.2f} & {hits}/{len(rows)}')

    routing = result('adult_routing')
    arm = [{r['seed']: r for r in routing['runs'] if r['adaptive'] == policy} for policy in (False, True)]
    record('Adult routing: identical paired calls and bounds', len(arm[0]) == len(arm[1]) == 30
           and arm[0].keys() == arm[1].keys()
           and all(all(a[key] == arm[1][seed][key] for key in ('calls', 'group_calls', 'bounds'))
                   for seed, a in arm[0].items())
           and routing['truth'] == adult['truth'] and routing['delta'] == adult['delta']
           and 'sha256:' + routing['source_sha256'] == inputs[str(ROOT / 'results/adult_count_replay.json')])
    text_claim(evaluation, 'adult_routing', f"each of {len(arm[0])} paired Adult seeds "
               f"({mean(r['calls'] for r in arm[0].values()):,.2f} calls on average for either policy)")

    capped = result('adult_routing_capped')
    pair = {(r['budget'], r['seed'], r['proxy'], r['adaptive']): r for r in capped['runs']}
    profile = {(r['budget'], r['seed']): r for r in schedule}
    record('Capped routing: source, parameter, and paired trial accounting',
           capped['truth'] == fixed['truth'] and capped['delta'] == fixed['delta']
           and 'sha256:' + capped['source_sha256'] == inputs[str(fixed_path)]
           and capped['batch'] == 32 and capped['pilot'] == 16 and len(pair) == len(capped['runs'])
           and set(pair) == {(b, s, p, a) for b, s in profile for p in (False, True) for a in (False, True)})
    record('Capped routing: budget, count coverage, and feasible bounds for each trial',
           all(r['groups'].keys() == sizes.keys() and sum(v['calls'] for v in r['groups'].values()) == r['budget']
               and all(0 <= v['calls'] <= sizes[g] and
                       0 <= v['lower'] <= capped['truth'][g]['positives'] <= v['upper'] <= sizes[g]
                       for g, v in r['groups'].items()) for r in capped['runs']))
    record('Capped routing: refitted width-first arm matches the fixed-budget experiment',
           all(all(x['calls'] == by_g[g]['calls'] and all(abs(x[k]-by_g[g][k]) <= 1e-9 for k in ('lower','upper'))
                   for g,x in pair[b,s,True,True]['groups'].items())
               for (b,s), base in profile.items() for by_g in (base['groups'],)))
    widths = {key: [(v['upper']-v['lower'])/(2*sizes[g]) for g,v in r['groups'].items()]
              for key,r in pair.items()}
    record('Capped routing: strictly narrower maximum interval in every paired comparison',
           all(max(widths[b,s,p,True]) + 1e-12 < max(widths[b,s,p,False])
               for b,s in profile for p in (False,True)))
    n_pairs = len(pair)//2
    seeds_per_cap = len([s for b,s in profile if b == 4096])
    text_claim(evaluation, 'adult_routing_capped', f'{n_pairs} matched comparisons ({seeds_per_cap} seeds per cap and proxy setting)')
    arms = [[w for (b,s,p,a),w in widths.items() if b == 4096 and p and a == mode] for mode in (False,True)]
    hi = [100*mean(max(w) for w in arm) for arm in arms]
    hit = [mean(sum(v <= .04+1e-12 for v in w) for w in arm) for arm in arms]
    text_claim(evaluation, 'adult_routing_capped', f'the mean maximum drops from {hi[0]:.2f} to {hi[1]:.2f} percentage points')
    text_claim(evaluation, 'adult_routing_capped', f'falls from {hit[0]:.2f} to {hit[1]:.2f} out of nine')

    refit = result('proxy_refit')
    record('Refitting: round-robin, pilot and rate tolerance',
           refit['routing'] == 'round_robin' and refit['pilot'] == 16 and refit['rate_tolerance'] == .04)
    for name, source_name, source_data, batch, label in (
        ('adult', 'adult_count_replay', adult, 32, 'Adult'),
        ('fashion', 'fashion_live', fashion, 64, 'Fashion-MNIST'),
    ):
        work = refit['workloads'][name]
        seeds = sorted({r['seed'] for r in source_data['runs']})
        record(f'Refitting: {name} source, population and paired seed schedule',
               'sha256:' + work['source_sha256'] == inputs[str(ROOT/'results'/f'{source_name}.json')]
               and work['truth'] == source_data['truth'] and work['delta'] == source_data['delta']
               and work['batch'] == batch and sorted((r['seed'], r['refit']) for r in work['runs']) ==
                   [(seed, mode) for seed in seeds for mode in (False, True)])
        record(f'Refitting: {name} query call accounting, coverage and width',
               all(r['groups'].keys() == work['truth'].keys()
                   and sum(v['calls'] for v in r['groups'].values()) == r['calls']
                   and all(0 <= v['calls'] <= work['truth'][g]['size']
                           and 0 <= v['lower'] <= work['truth'][g]['positives'] <= v['upper'] <= work['truth'][g]['size']
                           and (v['upper']-v['lower'])/2 <= .04 * work['truth'][g]['size']+1e-9
                           for g, v in r['groups'].items()) for r in work['runs']))
        calls = [mean(r['calls'] for r in work['runs'] if r['refit'] == mode) for mode in (False, True)]
        text_claim(evaluation, 'proxy_refit', f'{calls[0]:,.1f} to {calls[1]:,.1f} on {label} ({len(seeds)} paired seeds)')
    n_refit = sum(len(work['runs']) for work in refit['workloads'].values())
    text_claim(evaluation, 'proxy_refit', f'All {n_refit} runs meet the width target and cover every group')

    boundary_name = 'decision_boundaries'
    boundary = result(boundary_name)
    cells = decision_audit(boundary)
    caps = boundary['budgets']
    trials_per_cell = len(boundary['seeds'])
    settings = [(gap, proxy) for gap in boundary['gaps'] for proxy in (False, True)]
    text_claim(evaluation, boundary_name, f"four groups of {boundary['group_size']} rows each")
    text_claim(evaluation, boundary_name, f"threshold of {boundary['threshold']}")
    text_claim(evaluation, boundary_name, f"{trials_per_cell} seeds per setting")
    text_claim(evaluation, boundary_name, f"caps of {caps[0]:,}, {caps[1]:,}, and {caps[2]:,}")
    text_claim(evaluation, boundary_name, f"Across {len(boundary['runs'])} runs there were no incorrect certified decisions")
    record('Decision boundaries: no wrong certificates', not any(cell['wrong'] for cell in cells.values()))
    record('Decision boundaries: HAVING unresolved at low cap and fully resolved at mid cap',
           all(cells[g, caps[0], p, 'having']['unresolved'] == trials_per_cell
               and cells[g, caps[1], p, 'having']['correct'] == trials_per_cell for g, p in settings))
    record('Decision boundaries: top-2 gap sensitivity and exact census tie',
           all(cells[g, caps[1], p, 'topk']['correct'] == (trials_per_cell if g else 0) for g, p in settings)
           and all(cells[0, caps[2], p, 'topk']['correct'] == trials_per_cell
                   and cells[0, caps[2], p, 'topk']['mean_calls'] == caps[2] for p in (False, True)))
    disabled, enabled = (cells[64, 256, p, 'topk']['correct'] for p in (False, True))
    text_claim(evaluation, boundary_name, f"At gap 64 and a cap of 256, the proxy-disabled arm certified top-2 "
               f"in {disabled} of {trials_per_cell} trials and the proxy-enabled arm in {enabled} of {trials_per_cell}")

    latency_name = 'live_latency'
    latency = latency_audit(result(latency_name), sms, fashion)
    for label, mode in [('Exact', 'best_exact'), ('Fresh, no proxy', 'fresh_no_proxy'),
                        ('Fresh, proxy', 'fresh_proxy'), ('Reuse, proxy', 'reuse_proxy')]:
        values = [f"{latency[w][mode]['median_wall_ms']:,.0f}" for w in ('sms', 'fashion')]
        text_claim(evaluation, latency_name, f"{label} & {values[0]} & {values[1]}")
    text_claim(evaluation, latency_name, f"SMS: {latency['sms']['best_exact']['batch']:,}; "
               f"Fashion-MNIST: {latency['fashion']['best_exact']['batch']:,}")
    record('Live latency: SMS exact wins even with profile reuse; Fashion fresh proxy wins',
           latency['sms']['best_exact']['median_wall_ms'] < latency['sms']['reuse_proxy']['median_wall_ms']
           and latency['fashion']['fresh_proxy']['median_wall_ms'] < latency['fashion']['best_exact']['median_wall_ms'])

    mismatches = [check for check in checks if check['status'] == 'mismatch']
    report = {'audit_skill': 'experiment-claim-audit', 'verdict': 'FAIL' if mismatches else 'WARN',
              'reason_code': 'claim_mismatch' if mismatches else 'independent_review_pending',
              'summary': 'Deterministic checks only. Independent review and unregistered prose require manual inspection.',
              'generated_at': datetime.now(timezone.utc).isoformat(), 'audited_input_hashes': inputs,
              'details': {'total_claims': len(checks), 'mismatches': mismatches, 'checks': checks}}
    (PAPER / 'PAPER_CLAIM_AUDIT.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'{len(checks) - len(mismatches)}/{len(checks)} checks passed')
    for mismatch in mismatches:
        print(mismatch['claim'])
    return int(bool(mismatches))


if __name__ == '__main__':
    sys.exit(main())

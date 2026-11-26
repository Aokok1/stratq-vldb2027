"""Audit timing blocks, held-out populations, and published latency metrics."""
import json
from itertools import product
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def audit(data, sms, fashion):
    by_name = {work['dataset']: work for work in data['datasets']}
    assert by_name.keys() == {'sms', 'fashion'}
    named = {'sms': ('L', sms['sizes'], sms['positives'], sms['archive_sha256'], 32, .06),
             'fashion': ('B', {g: v['size'] for g, v in fashion['truth'].items()},
                         {g: v['positives'] for g, v in fashion['truth'].items()}, fashion['sha256'], 64, .04)}
    summaries = {}
    for name, (prefix, sizes, truth, digest, batch, tol) in named.items():
        work = by_name[name]
        n = work['rows']
        assert work['delta'] == .1 and work['tolerance'] == tol and work['source_hash'] == digest
        assert n == sum(sizes.values())
        assert work['sizes'] == [sizes[prefix+str(j)] for j in range(len(sizes))]
        assert work['truth'] == {g[1:]: v for g, v in truth.items()}
        modes = {('exact', b) for b in (batch, 128, 512, 1024, n)}
        modes |= {(mode, batch) for mode in ('fresh_no_proxy', 'fresh_proxy', 'reuse_proxy')}
        indexed = {(r['replicate'], r['mode'], r['batch']): r for r in work['runs']}
        assert set(indexed) == set((j, m, b) for j, (m, b) in product(range(11), modes))
        assert len(indexed) == len(work['runs'])
        for (_, mode, b), r in indexed.items():
            assert 0 < r['calls'] <= n and 0 < r['batches'] <= r['calls']
            assert abs(r['wall_ms'] - sum(r[col] for col in ('query_ms', 'profile_ms', 'proxy_ms'))) < 2
            assert r['query_ms'] > 0 and r['profile_ms'] >= 0 and r['proxy_ms'] >= 0
            assert set(r['bounds']) == set(work['truth'])
            for g, (lo, hi) in r['bounds'].items():
                count = truth[prefix+g]
                assert lo-1e-9 <= count <= hi+1e-9 and (hi-lo)/2 <= tol * sizes[prefix+g]+1e-9
                if mode == 'exact':
                    assert lo == hi == count and r['calls'] == n and r['batches'] == (n+b-1)//b
        stats = {}
        for m, b in modes:
            records = [indexed[j,m,b] for j in range(1, 11)]
            stats[m,b] = dict(mode=m, batch=b, n=len(records), **{
                'median_'+key: median(r[key] for r in records)
                for key in ('calls', 'wall_ms', 'proxy_ms', 'profile_ms', 'query_ms')})
        for j in range(11):
            a, b = (indexed[j, m, batch] for m in ('fresh_proxy', 'reuse_proxy'))
            assert a['calls'] == b['calls'] and a['bounds'] == b['bounds']
        assert stats == {(s['mode'], s['batch']): s for s in work['summary']}
        best = min((s for key,s in stats.items() if key[0] == 'exact'), key=lambda s:s['median_wall_ms'])
        summaries[name] = {'best_exact': best, **{m: stats[m,batch] for m in
                                               ('fresh_no_proxy', 'fresh_proxy', 'reuse_proxy')}}
    return summaries


if __name__ == '__main__':
    def result(key):
        return json.loads((ROOT/'results'/f'{key}.json').read_text(encoding='utf-8'))
    print(json.dumps(audit(result('live_latency'), result('sms_live'), result('fashion_live')), indent=2))

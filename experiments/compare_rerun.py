"""Compare deterministic evidence from an isolated experiment rerun to frozen JSON.

Timing, host/runtime fields, generation timestamps, and code/script provenance
hashes are excluded from numerical equality; source-result links are still
verified. Use the Sprint 4 code-change record to audit provenance transitions.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

DATASETS = ('adult_count_replay', 'adult_hypergeom', 'adult_fixed_budget', 'adult_blazeit',
            'adult_senate', 'adult_online_aggregation', 'adult_lws', 'adult_routing', 'adult_routing_capped',
            'adult_verdict', 'sms_live', 'fashion_live', 'proxy_refit',
            'live_latency', 'decision_boundaries', 'scale_100000', 'scale_1000000', 'scale_10000000')
SOURCE_LINKS = (
    ('adult_blazeit', (), 'adult_fixed_budget'), ('adult_senate', (), 'adult_fixed_budget'),
    ('adult_lws', (), 'adult_fixed_budget'), ('adult_verdict', (), 'adult_senate'),
    ('adult_online_aggregation', (), 'adult_fixed_budget'), ('adult_routing_capped', (), 'adult_fixed_budget'),
    ('adult_routing', (), 'adult_count_replay'), ('proxy_refit', ('workloads', 'adult'), 'adult_count_replay'),
    ('proxy_refit', ('workloads', 'fashion'), 'fashion_live'),
)
RUNTIME_ROOT = {'live_latency': {'threadpools'}, 'scale_100000': {'host', 'peak_working_set_bytes'},
                'scale_1000000': {'host', 'peak_working_set_bytes'},
                'scale_10000000': {'host', 'peak_working_set_bytes'}}
PROVENANCE_KEYS = {'code_sha256', 'script_sha256', 'generated_at_utc', 'current_code_sha256'}


def load(root):
    raw = {n: (root / (n + '.json')).read_bytes() for n in DATASETS}
    records = {n: json.loads(b) for n, b in raw.items()}
    for name, path, source in SOURCE_LINKS:
        item = records[name]
        for key in path:
            item = item[key]
        if item['source_sha256'] != hashlib.sha256(raw[source]).hexdigest():
            raise ValueError(f'{name} does not point to {source} from the same run')
        item['source_sha256'] = source
    return records


def differences(a, b, path):
    if type(a) in (int, float) and type(b) in (int, float):
        if isinstance(a, int) or isinstance(b, int):
            # Census/feasible endpoints may be serialized as ints in newer runs.
            match = a == b if type(a) is type(b) else math.isclose(a, b, rel_tol=0, abs_tol=1e-10)
        else:
            match = math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
        if not match:
            yield path
    elif type(a) is not type(b):
        yield path
    elif isinstance(a, dict):
        skipped = RUNTIME_ROOT.get(path, set())
        for key in sorted(a.keys() | b.keys()):
            if key.endswith('_ms') or key in skipped or key in PROVENANCE_KEYS:
                continue
            child = f'{path}.{key}'
            if key not in a or key not in b:
                yield child
            else:
                yield from differences(a[key], b[key], child)
    elif isinstance(a, list):
        if len(a) != len(b):
            yield path + '.length'
        for idx, (x, y) in enumerate(zip(a, b)):
            yield from differences(x, y, f'{path}[{idx}]')
    elif a != b:
        yield path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('frozen', type=Path, help='Directory of original result JSON')
    p.add_argument('rerun', type=Path, help='Directory of newly generated result JSON')
    args = p.parse_args()
    frozen, rerun = load(args.frozen), load(args.rerun)
    count = 0
    for name in DATASETS:
        changed = list(differences(frozen[name], rerun[name], name))
        count += len(changed)
        print(f'{name}: {len(changed)} non-timing differences')
        for path in changed[:10]:
            print(f'  {path}')
    return int(count > 0)


if __name__ == '__main__':
    raise SystemExit(main())

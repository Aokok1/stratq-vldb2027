"""Fashion-MNIST grouped footwear counts with on-demand CPU inference."""
import gzip
import hashlib
import json
import struct
import sys
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np
import pyarrow as pa
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from stratq.engine import StratQEngine

DATA = ROOT / 'data' / 'fashion'
SEEDS = tuple(range(2701, 2706))
MIRROR = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'


def load(split, expected):
    images = DATA / f'{split}-images-idx3-ubyte.gz'
    labels = DATA / f'{split}-labels-idx1-ubyte.gz'
    x, y = (gzip.decompress(p.read_bytes()) for p in (images, labels))
    if struct.unpack('>IIII', x[:16]) != (2051, expected, 28, 28) or len(x) != 16 + 784 * expected:
        raise ValueError(f'{split}: invalid IDX images')
    if struct.unpack('>II', y[:8]) != (2049, expected) or len(y) != 8 + expected:
        raise ValueError(f'{split}: invalid IDX labels')
    images_array = np.frombuffer(x, dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    classes = np.frombuffer(y, dtype=np.uint8, offset=8)
    if not np.array_equal(np.bincount(classes), np.full(10, expected // 10)):
        raise ValueError(f'{split}: class histogram mismatch')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (images, labels)}
    return images_array, np.isin(classes, (5, 7, 9)).astype(np.uint8), hashes


def pixels(images):
    return images.reshape(-1, 784).astype(np.float64) / 255


def pooled(images):
    return images.reshape(-1, 7, 4, 7, 4).mean(axis=(2, 4)).reshape(-1, 49) / 255


class Reference:
    def __init__(self, model, images):
        self.model = model
        self.images = images
        self.reset()

    def reset(self, expected=None):
        self.seen = np.zeros(len(self.images), dtype=bool)
        self.expected = expected
        self.calls = self.batches = 0
        self.inference_ms = 0.0

    def __call__(self, ids):
        start = perf_counter()
        labels = self.model.predict(pixels(self.images[ids]))
        self.inference_ms += 1000 * (perf_counter() - start)
        if self.seen[ids].any() or len(np.unique(ids)) != len(ids):
            raise ValueError('Repeated row in one query')
        if self.expected is not None and not np.array_equal(labels, self.expected[ids]):
            raise ValueError('Reference output changed with the batch')
        self.seen[ids] = True
        self.batches += 1
        self.calls += len(ids)
        return labels


def prepare():
    """Fit both models and return held-out input before scoring or profiling."""
    train, y_train, train_hash = load('train', 60000)
    test, y_test, test_hash = load('t10k', 10000)
    fit, _ = train_test_split(np.arange(len(train)), train_size=6000, stratify=y_train, random_state=1729)
    proxy = LogisticRegression(random_state=1729).fit(pooled(train[fit]), y_train[fit])
    ref_model = SVC(C=4, kernel='rbf').fit(pixels(train[fit]), y_train[fit])
    cuts = np.quantile(train.mean(axis=(1, 2)), [0.2, 0.4, 0.6, 0.8])
    return len(train), len(fit), test, y_test, train_hash | test_hash, proxy, Reference(ref_model, test), cuts


def main():
    n_train, n_fit, test, y_test, hashes, proxy, reference, cuts = prepare()
    start = perf_counter()
    scores = proxy.predict_proba(pooled(test))[:, 1]
    proxy_ms = 1000 * (perf_counter() - start)
    groups = np.array([f'B{i}' for i in np.searchsorted(cuts, test.mean(axis=(1, 2)), side='right')])
    n = len(test)
    table = pa.table({'grp': groups, 'proxy': scores, 'row_id': np.arange(n)})
    start = perf_counter()
    engine = StratQEngine(table, group_col='grp', proxy_col='proxy', covariate_cols=['row_id'], oracle_fn=reference)
    profile_ms = 1000 * (perf_counter() - start)
    if not np.array_equal(engine.profile.phi_all[:, 2], np.arange(n) / (n - 1)):
        raise ValueError('Callback row order does not match Arrow order')
    exhaustive = []
    exact = None
    for batch in (64, 512, n):
        reference.reset(exact)
        start = perf_counter()
        predicted = np.concatenate([reference(np.arange(i, min(i + batch, n))) for i in range(0, n, batch)])
        exhaustive.append({'batch_size': batch, 'calls': reference.calls, 'batches': reference.batches,
                           'wall_ms': 1000 * (perf_counter() - start), 'inference_ms': reference.inference_ms})
        exact = predicted
    truth = {g: {'size': int((groups == g).sum()), 'positives': int(exact[groups == g].sum())}
             for g in sorted(set(groups.tolist()))}
    eps = {g: 0.04 * stats['size'] for g, stats in truth.items()}
    runs = []
    for j, seed in enumerate(SEEDS):
        for use_proxy in ((False, True) if j % 2 == 0 else (True, False)):
            reference.reset(exact)
            result = engine.execute_grouped_agg(eps=eps, delta=0.1, agg_type='count', batch_size=64,
                                                pilot_per_group=16, use_control_variates=use_proxy,
                                                use_covariates=False, seed=seed)
            if set(result.groups) != set(truth) or result.total_oracle_calls != reference.calls:
                raise ValueError('Group or call accounting mismatch')
            runs.append({'seed': seed, 'proxy': use_proxy, 'calls': reference.calls, 'batches': reference.batches,
                         'inference_ms': reference.inference_ms, 'wall_ms': result.wall_ms,
                         'groups': {g: {'lower': result.lower[g], 'upper': result.upper[g],
                                        'estimate': result.estimates[g], 'calls': result.oracle_calls_by_group[g]}
                                    for g in result.groups}})
    report = {'source': 'https://github.com/zalandoresearch/fashion-mnist', 'mirror': MIRROR,
              'sha256': hashes, 'sklearn': sklearn.__version__, 'train_rows': n_train,
              'fit_rows': n_fit, 'test_rows': n, 'reference_accuracy': float(np.mean(exact == y_test)),
              'group_intensity_cuts': cuts.tolist(), 'proxy_ms': proxy_ms, 'profile_ms': profile_ms,
              'delta': 0.1, 'rate_tolerance': 0.04, 'truth': truth, 'exhaustive': exhaustive, 'runs': runs}
    from audit_fashion import audit
    print(json.dumps(audit(report), indent=2, allow_nan=False))
    print('Exhaustive:', exhaustive)
    (ROOT / 'results' / 'fashion_live.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()

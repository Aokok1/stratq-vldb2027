"""Run grouped queries with on-demand SMS reference-model predictions."""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from stratq.engine import StratQEngine

ARCHIVE = ROOT / 'data' / 'uci_sms.zip'
SEEDS = tuple(range(2601, 2611))
LENGTH_CUTS = (40, 80, 120, 160)


class Reference:
    def __init__(self, train_text, train_labels, test_text):
        self.model = make_pipeline(
            TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2),
            KNeighborsClassifier(n_neighbors=5, metric='cosine', algorithm='brute', n_jobs=1),
        ).fit(train_text, train_labels)
        self.text = test_text
        self.reset()

    def reset(self):
        self.calls = 0
        self.batches = 0
        self.oracle_ms = 0.0
        self.seen = np.zeros(len(self.text), dtype=bool)
        self.expected = None

    def __call__(self, ids):
        start = perf_counter()
        predictions = self.model.predict(self.text[ids])
        self.oracle_ms += (perf_counter() - start) * 1000
        self.calls += len(ids)
        self.batches += 1
        if np.unique(ids).size != len(ids) or self.seen[ids].any():
            raise AssertionError('A row was evaluated twice in one query')
        self.seen[ids] = True
        if self.expected is not None and not np.array_equal(predictions, self.expected[ids]):
            raise AssertionError('Reference predicate changed with batch composition')
        return predictions


def prepare():
    """Fit both models and return held-out input before scoring or profiling."""
    with ZipFile(ARCHIVE) as archive:
        lines = archive.read('SMSSpamCollection').decode('utf-8').splitlines()
    rows = [line.split('\t', maxsplit=1) for line in lines if line.strip()]
    if not all(len(row) == 2 and row[0] in ('ham', 'spam') for row in rows):
        raise ValueError('Invalid SMS corpus record')
    labels = np.array([row[0] == 'spam' for row in rows], dtype=np.uint8)
    texts = np.asarray([row[1] for row in rows], dtype=object)
    keys = np.asarray([re.sub(r'\s+', ' ', text.casefold()).strip() for text in texts])
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=1729)
                       .split(texts, labels, groups=keys))
    assert not set(keys[train]).intersection(keys[test])
    test_text = texts[test]
    proxy_model = make_pipeline(TfidfVectorizer(max_features=256), LogisticRegression(random_state=1729))
    proxy_model.fit(texts[train], labels[train])
    reference = Reference(texts[train], labels[train], test_text)
    return train, test, labels[test], test_text, proxy_model, reference


def main():
    train, test, holdout_labels, test_text, proxy_model, reference = prepare()
    start = perf_counter()
    scores = proxy_model.predict_proba(test_text)[:, 1]
    proxy_scan_ms = (perf_counter() - start) * 1000
    group_ids = np.searchsorted(LENGTH_CUTS, [len(text) for text in test_text], side='right')
    groups = np.asarray([f'L{group}' for group in group_ids])
    n = len(test)
    # The profiler's covariate projection also checks physical row identity.
    table = pa.table({'grp': groups, 'proxy': scores, 'row_id': np.arange(n)})
    start = perf_counter()
    engine = StratQEngine(table, group_col='grp', proxy_col='proxy', covariate_cols=['row_id'], oracle_fn=reference)
    profile_ms = (perf_counter() - start) * 1000
    assert np.array_equal(engine.profile.phi_all[:, 2], np.arange(n) / (n - 1))
    # Match every engine callback to the exhaustive reference on the same input row.
    exhaustive = []
    truth_labels = None
    for batch_size in (32, 128, n):
        reference.reset()
        start = perf_counter()
        predicted = np.concatenate([reference(np.arange(i, min(i + batch_size, n))) for i in range(0, n, batch_size)])
        wall_ms = (perf_counter() - start) * 1000
        if truth_labels is None:
            truth_labels = predicted
        elif not np.array_equal(truth_labels, predicted):
            raise AssertionError('Exhaustive reference outputs differ across batch sizes')
        exhaustive.append({'batch_size': batch_size, 'calls': reference.calls, 'batches': reference.batches,
                           'oracle_ms': reference.oracle_ms, 'wall_ms': wall_ms})
    sizes = Counter(groups.tolist())
    positives = {g: int(truth_labels[groups == g].sum()) for g in sorted(sizes)}
    expected_top = sorted(sizes, key=lambda g: (-positives[g], g))[:2]
    tau = 40
    eps = {g: 0.06 * size for g, size in sizes.items()}
    runs = []
    for query in ('count', 'having', 'topk'):
        for j, seed in enumerate(SEEDS):
            for proxy in ((False, True) if j % 2 == 0 else (True, False)):
                reference.reset()
                reference.expected = truth_labels
                options = dict(seed=seed, agg_type='count', delta=0.1, batch_size=32, pilot_per_group=8,
                               use_control_variates=proxy, use_covariates=False)
                if query == 'count':
                    result = engine.execute_grouped_agg(eps=eps, **options)
                    decision = {'target_met': all(result.half_widths[g] <= eps[g] + 1e-9 for g in sizes)}
                elif query == 'having':
                    result = engine.execute_grouped_having(tau=tau, indifference_eps=0, **options)
                    decision = {'accepted': result.accepted_groups, 'rejected': result.rejected_groups,
                                'uncertain': result.uncertain_groups,
                                'correct': all(positives[g] >= tau for g in result.accepted_groups)
                                and all(positives[g] < tau for g in result.rejected_groups)}
                else:
                    result = engine.execute_grouped_topk(k=2, **options)
                    decision = {'selected': result.topk_groups, 'certified': result.certified,
                                'correct': set(result.topk_groups) == set(expected_top)}
                if set(result.groups) != set(sizes) or result.total_oracle_calls != reference.calls:
                    raise AssertionError('Result group or reference-call accounting mismatch')
                runs.append({'query': query, 'seed': seed, 'proxy': proxy, 'calls': reference.calls,
                             'batches': reference.batches, 'oracle_ms': reference.oracle_ms,
                             'wall_ms': result.wall_ms, 'decision': decision,
                             'lower': result.lower, 'upper': result.upper,
                             'covered': all(result.lower[g] - 1e-9 <= positives[g] <= result.upper[g] + 1e-9 for g in sizes)})
    report = {'source': 'https://archive.ics.uci.edu/dataset/228/sms+spam+collection', 'doi': '10.24432/C5CC84',
              'archive_sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), 'sklearn': sklearn.__version__,
              'train_rows': len(train), 'test_rows': n, 'length_cuts': LENGTH_CUTS, 'delta': 0.1,
              'having_threshold': tau, 'topk': 2, 'proxy_scan_ms': proxy_scan_ms, 'profile_ms': profile_ms,
              'reference_accuracy_vs_annotations': float(np.mean(truth_labels == holdout_labels)),
              'sizes': dict(sorted(sizes.items())), 'positives': positives, 'epsilon': eps,
              'exhaustive': exhaustive, 'runs': runs}
    (ROOT / 'results' / 'sms_live.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'{len(train)} train, {n} test, group counts {positives}')
    for query in ('count', 'having', 'topk'):
        for proxy in (False, True):
            subset = [r for r in runs if r['query'] == query and r['proxy'] == proxy]
            print(query, proxy, 'mean calls', np.mean([r['calls'] for r in subset]),
                  'median ms', np.median([r['wall_ms'] for r in subset]),
                  'covered', sum(r['covered'] for r in subset))
    print('Exhaustive:', exhaustive)


if __name__ == '__main__':
    main()

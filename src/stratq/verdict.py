"""VerdictDB's variational subsampling adapted to known-size binary groups."""
from math import isqrt

import numpy as np

from .senate import senate_quotas


def variational_count(n, labels, sid, alpha):
    """Basic (reflected) quantiles of scaled nonempty subsample deviations."""
    m = len(labels)
    s = int(labels.sum())
    feasible = (s, s + n - m)
    center = n*s/m if m else n/2
    lo, hi = feasible
    if 0 < m < n:
        # ID 0 is reserved for full-sample rows not assigned to a subsample.
        sizes = np.bincount(sid)[1:]
        positives = np.bincount(sid[labels.astype(bool)], minlength=len(sizes)+1)[1:]
        ok = sizes > 0
        if ok.sum() >= 2 and np.unique(labels).size > 1:
            t = sizes[ok]
            dev = (positives[ok]/t - s/m) * np.sqrt(t/m) * n
            qlo, qhi = np.quantile(dev, (alpha/2, 1-alpha/2), method='inverted_cdf')
            a, b = max(lo, center-qhi), min(hi, center-qlo)
            if a <= b:
                lo, hi = a, b
    return dict(estimate=center, lower=lo, upper=hi, calls=m)


def verdict_fixed_counts(groups, oracle, *, budget, delta, seed):
    """One fixed look at Senate quotas; interval accuracy is asymptotic in source."""
    if not 0 < delta < 1:
        raise ValueError('Expected joint risk in (0,1)')
    quotas = senate_quotas([len(ids) for ids in groups.values()], budget)
    rng = np.random.default_rng(seed)
    bucket_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    out = {}
    for (g, ids), m in zip(groups.items(), quotas):
        draw = rng.permutation(ids)[:m]
        labels = np.asarray(oracle(draw)) if m else np.empty(0)
        if labels.shape != (m,) or not np.isin(labels, (0,1)).all():
            raise ValueError('Expected one binary label per row')
        sid = np.zeros(m, dtype=int)
        if 0 < m < len(ids):
            ns = isqrt(m)
            # floor(m/ns) buckets of expected size ns; leftover probability for ID 0.
            base = bucket_rng.integers(m, size=m)
            sid = np.where(base < (m//ns)*ns, 1+base//ns, 0)
        out[g] = variational_count(len(ids), labels, sid, delta/len(groups))
    return out

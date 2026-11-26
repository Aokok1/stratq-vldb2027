"""Grouped adaptation of the phase-two LWS estimator (Walenz et al., §4.1)."""
from statistics import NormalDist

import numpy as np

from .senate import senate_quotas


def des_raj_totals(weights, order, labels):
    """Eq. (3) multiplied by group size; weights need not sum to one."""
    selected = weights[order]
    # Sum unselected weights directly to avoid cancellation near census.
    remaining = np.ones(len(weights), dtype=bool)
    remaining[order] = False
    mass = np.cumsum(selected[::-1])[::-1] + weights[remaining].sum()
    past = np.cumsum(labels) - labels
    return past + labels * mass / selected


def lws_fixed_counts(groups, scores, oracle, *, budget, delta, seed, score_floor=.05):
    """Use fixed quotas and descriptive normal intervals, not certificates."""
    scores = np.asarray(scores, dtype=float)
    if not (0 < score_floor <= 1 and 0 < delta < 1) or scores.ndim != 1 or not (
        np.isfinite(scores) & (scores >= 0) & (scores <= 1)
    ).all():
        raise ValueError('Expected probabilities, positive floor, and 0 < delta < 1')
    quotas = senate_quotas([len(ids) for ids in groups.values()], budget)
    rng = np.random.default_rng(seed)
    z = NormalDist().inv_cdf(1 - delta/(2*len(groups)))
    out = {}
    for (g, ids), m in zip(groups.items(), quotas):
        n = len(ids)
        w = np.maximum(scores[ids], score_floor)
        # Independent exponential clocks sorted by arrival implement sequential PPS-WoR.
        order = np.argsort(rng.exponential(size=n)/w, kind='stable')[:m]
        labels = np.asarray(oracle(ids[order])) if m else np.empty(0)
        if labels.shape != (m,) or not np.isin(labels, (0,1)).all():
            raise ValueError('Expected one binary label per row')
        s = int(labels.sum())
        lo, hi = s, s + n - m
        center = (lo+hi)/2
        if 0 < m < n:
            totals = des_raj_totals(w, order, labels.astype(float))
            center = float(totals.mean())
            if m >= 2:
                radius = z * float(totals.std(ddof=1)) / np.sqrt(m)
                a, b = max(lo, center-radius), min(hi, center+radius)
                if a <= b:
                    lo, hi = a, b
        out[g] = dict(estimate=center, lower=lo, upper=hi, calls=m)
    return out

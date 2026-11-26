"""SENATE equal allocation and fixed-look intervals for binary group totals."""
import numpy as np
from scipy.stats import hypergeom


def senate_quotas(sizes, budget):
    """Equal quotas up to census caps; residual calls follow group order."""
    if not sizes or min(sizes) <= 0 or not 0 <= budget <= sum(sizes):
        raise ValueError('Invalid sizes or budget')
    lo, hi = 0, max(sizes)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(n, mid) for n in sizes) <= budget:
            lo = mid
        else:
            hi = mid - 1
    quotas = [min(n, lo) for n in sizes]
    left = budget - sum(quotas)
    for i, n in enumerate(sizes):
        if left == 0:
            break
        if quotas[i] < n:
            quotas[i] += 1
            left -= 1
    assert left == 0
    return quotas


def fixed_interval(n, m, s, alpha):
    """Invert two inclusive hypergeometric tails; not a confidence sequence."""
    if not (0 < n and 0 <= s <= m <= n and 0 < alpha < 1):
        raise ValueError('Invalid counts or risk')
    feasible = (s, s+n-m)
    if m == 0 or m == n:
        return feasible
    # Slightly lower the cutoff to avoid rejecting rounded equality.
    tail = alpha / 2 * (1 - 1e-10)
    lo, hi = feasible
    while lo < hi:
        x = (lo+hi)//2
        if hypergeom.sf(s-1, n, x, m) >= tail:
            hi = x
        else:
            lo = x+1
    lower = lo
    lo, hi = feasible
    while lo < hi:
        x = (lo+hi+1)//2
        if hypergeom.cdf(s, n, x, m) >= tail:
            lo = x
        else:
            hi = x-1
    return (lower, lo) if lower <= lo else feasible


def senate_fixed_counts(groups, oracle, *, budget, delta, seed):
    """One predeclared look, uniform without-replacement sampling by group."""
    if not 0 < delta < 1:
        raise ValueError('Invalid joint risk')
    quotas = senate_quotas([len(ids) for ids in groups.values()], budget)
    rng = np.random.default_rng(seed)
    output = {}
    for (g, ids), m in zip(groups.items(), quotas):
        draw = rng.permutation(ids)[:m]
        labels = np.asarray(oracle(draw)) if m else np.empty(0)
        if labels.shape != (m,) or not np.isin(labels, (0, 1)).all():
            raise ValueError('Expected one binary label per row')
        lo, hi = fixed_interval(len(ids), m, int(labels.sum()), delta / len(groups))
        output[g] = dict(estimate=(lo+hi)/2, lower=lo, upper=hi, calls=m)
    return output

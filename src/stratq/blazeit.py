"""Grouped fixed-budget adaptation of BlazeIt's control-variate operator."""
from collections.abc import Callable, Mapping
from math import log, sqrt

import numpy as np


def blazeit_fixed_counts(
    proxy: np.ndarray,
    groups: Mapping[str, np.ndarray],
    oracle: Callable[[np.ndarray], np.ndarray],
    *, budget: int, delta: float, seed: int, pilot: int = 16,
) -> dict[str, dict[str, float | int]]:
    """Budgeted grouped counts with a pilot-fitted fixed coefficient.

    Uses the control variate in BlazeIt §6 and the fixed-look radius in
    Appendix B.1. Independent second-stage draws are with replacement from
    non-pilot rows. Bonferroni allocates risk across groups. This adapts the
    aggregation operator and its bound to a fixed-budget grouped query.
    """
    if not groups or not 0 < delta < 1 or pilot < 2:
        raise ValueError('Required: nonempty groups, 0 < delta < 1, pilot >= 2')
    sizes = {g: len(ids) for g, ids in groups.items()}
    if any(n == 0 for n in sizes.values()):
        raise ValueError('Each group must contain at least one row')
    allocations = {g: min(n, pilot + 2) for g, n in sizes.items()}
    if not sum(allocations.values()) <= budget <= sum(sizes.values()):
        raise ValueError('Budget must support every noncensus group pilot and second stage')
    if not np.all(np.isfinite(proxy)):
        raise ValueError('Proxy scores must be finite')
    remaining = budget - sum(allocations.values())
    # Equalize total calls subject to the finite census caps.
    while remaining:
        for group in groups:
            if allocations[group] < sizes[group]:
                allocations[group] += 1
                remaining -= 1
                if remaining == 0:
                    break
    rng = np.random.default_rng(seed)
    log_risk = log(3 * len(groups) / delta)
    results = {}

    def binary_labels(rows):
        values = np.asarray(oracle(rows), dtype=float)
        if values.shape != rows.shape or not np.all((values == 0) | (values == 1)):
            raise ValueError('Expected one binary reference label per draw')
        return values

    for group, ids in groups.items():
        n = sizes[group]
        calls = allocations[group]
        if calls == n:
            lo = hi = int(binary_labels(ids).sum())
        else:
            permutation = rng.permutation(ids)
            initial, pool = permutation[:pilot], permutation[pilot:]
            y_initial = binary_labels(initial)
            x_initial = proxy[initial] - proxy[initial].mean()
            denominator = float(x_initial @ x_initial)
            slope = float(x_initial @ (y_initial - y_initial.mean()) / denominator) if denominator > 0 else 0.0
            draws = rng.choice(pool, size=calls - pilot, replace=True)
            f_pool = slope * proxy[pool]
            y_draws = binary_labels(draws)
            adjusted = y_draws + f_pool.mean() - slope * proxy[draws]
            span = 1 + float(np.ptp(f_pool))
            radius = (adjusted.std(ddof=0) * sqrt(2 * log_risk / len(draws))
                      + 3 * span * log_risk / len(draws))
            # A repeated draw incurs a call, but only unique labels reduce
            # the deterministic feasible range of the unknown count.
            _, first = np.unique(draws, return_index=True)
            pilot_sum = int(y_initial.sum())
            known = pilot_sum + int(y_draws[first].sum())
            feasible_hi = known + len(pool) - len(first)
            center = pilot_sum + len(pool) * adjusted.mean()
            lo = max(float(known), center - len(pool) * radius)
            hi = min(float(feasible_hi), center + len(pool) * radius)
            if lo > hi:
                lo, hi = known, feasible_hi
        results[group] = {'calls': calls, 'lower': lo, 'upper': hi, 'estimate': (lo + hi) / 2}
    return results

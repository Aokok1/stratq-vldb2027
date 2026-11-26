"""Online Aggregation Eq. (1), specialized to grouped binary means."""
import math

import numpy as np


def conservative_count(n: int, draws: int, positives: int, alpha: float) -> dict:
    """One fixed look; intersect Hoeffding with deterministic feasibility."""
    if not (0 <= positives <= draws <= n and n > 0 and 0 < alpha < 1):
        raise ValueError('Invalid group, draws, positives, or risk')
    lo, hi = positives, positives + n - draws
    center = (lo + hi) / 2
    if draws and draws < n:
        center = n * positives / draws
        radius = n * math.sqrt(math.log(2 / alpha) / (2 * draws))
        lo, hi = max(lo, center - radius), min(hi, center + radius)
    return dict(estimate=center, lower=lo, upper=hi, calls=draws)


def online_fixed_counts(groups, oracle, *, budget, delta, seed):
    """One prefix of a globally shuffled scan; no adaptive group index."""
    if not groups or any(len(ids) == 0 for ids in groups.values()) or not 0 < delta < 1:
        raise ValueError('Expected nonempty groups and 0 < delta < 1')
    sizes = np.array([len(ids) for ids in groups.values()])
    rows = np.concatenate(list(groups.values()))
    if not 0 <= budget <= rows.size:
        raise ValueError('Budget outside population')
    chosen = np.random.default_rng(seed).permutation(rows.size)[:budget]
    labels = np.asarray(oracle(rows[chosen])) if budget else np.empty(0, dtype=np.int64)
    if labels.shape != (budget,) or not np.isin(labels, (0, 1)).all():
        raise ValueError('Expected one binary label per chosen row')
    # Conditional on quotas, the fixed scan yields a uniform subset within each group.
    gi = np.searchsorted(sizes.cumsum(), chosen, side='right')
    draws = np.bincount(gi, minlength=len(groups))
    positive = np.bincount(gi[labels.astype(bool)], minlength=len(groups))
    return {g: conservative_count(int(n), int(m), int(s), delta/len(groups))
            for g, n, m, s in zip(groups, sizes, draws, positive)}

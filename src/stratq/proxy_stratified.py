"""Fixed-budget, proxy-stratified binary counts for known group keys."""
import math

import numpy as np


def allocate(total: int, weights: np.ndarray) -> np.ndarray:
    """Largest-remainder rounding with deterministic ties."""
    ideal = total * (weights / weights.sum())
    counts = np.floor(ideal).astype(np.int64)
    order = np.argsort(-(ideal - counts), kind='stable')
    counts[order[:total - int(counts.sum())]] += 1
    return counts


def fixed_budget_counts(scores, groups, evaluate, *, budget: int, delta: float, seed: int,
                        strata: int = 4, pilot: int = 8) -> dict:
    """Use ABae-inspired pilot allocation and independent stage-2 inference.

    Approximate the maximum rate variance across groups from pilot estimates.
    Stage 2 samples with replacement, making weighted Hoeffding valid
    conditional on the pilot and the resulting fixed sample sizes.
    """
    if not (strata > 0 and pilot > 0 and len(groups) > 0 and 0 < delta < 1):
        raise ValueError('Expected positive strata, pilot, groups and 0 < delta < 1')
    rng = np.random.default_rng(seed)
    splits = {}
    for group, ids in groups.items():
        if not len(ids):
            raise ValueError(f'Empty group {group}')
        ordered = ids[np.argsort(scores[ids], kind='stable')]
        splits[group] = np.array_split(ordered, min(strata, len(ids)))
    cells = sum(map(len, splits.values()))
    if budget < cells * (pilot + 1):
        raise ValueError(f'Budget {budget} cannot support {cells} strata')
    design = {}
    for group, partitions in splits.items():
        sizes = np.array([len(ids) for ids in partitions])
        n = int(sizes.sum())
        weights = sizes / n
        successes = np.array([evaluate(rng.choice(ids, size=pilot)).sum() for ids in partitions])
        rates = (successes + 0.5) / (pilot + 1)
        neyman = weights * np.sqrt(rates * (1 - rates))
        design[group] = (n, weights, neyman)
    remaining = budget - cells * (pilot + 1)
    # If V_g/n_g is the plug-in rate variance, allocating n_g ∝ V_g equalizes it.
    group_extra = allocate(remaining, np.array([neyman.sum() ** 2 for _, _, neyman in design.values()]))
    outputs = {}
    log_term = math.log(2 * len(groups) / delta)
    for (group, (n, weights, neyman)), extra in zip(design.items(), group_extra):
        samples = 1 + allocate(int(extra), neyman)
        rates = np.array([evaluate(rng.choice(ids, size=int(m))).mean()
                          for ids, m in zip(splits[group], samples)])
        point = n * float(weights @ rates)
        radius = n * math.sqrt(0.5 * log_term * float((weights ** 2) @ (1 / samples)))
        outputs[group] = {'estimate': point, 'lower': max(0.0, point - radius),
                          'upper': min(float(n), point + radius),
                          'calls': len(samples) * pilot + int(samples.sum()), 'stage2_by_stratum': samples.tolist()}
    return outputs

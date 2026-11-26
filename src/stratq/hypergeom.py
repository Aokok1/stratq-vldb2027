"""Uniform-prior hypergeometric confidence sequence for a binary total."""
import math

from scipy.stats import hypergeom


def count_interval(size: int, draws: int, positives: int, alpha: float) -> tuple[int, int]:
    """Invert the uniform-prior predictive/likelihood ratio at one look.

    The uniform prior on the population count induces a uniform distribution
    over the sampled count. The ordered-label combinatorial factor cancels.
    The likelihood is unimodal in the population count, so inversion uses two
    binary searches. Callers can intersect the intervals over past looks.
    """
    if not (0 < size and 0 <= positives <= draws <= size and 0 < alpha < 1):
        raise ValueError('Invalid population, sample count, or risk')
    lower, upper = positives, positives + size - draws
    if draws == 0 or lower == upper:
        return lower, upper
    mode = max(lower, min(upper, positives * (size + 1) // draws))
    # Widen slightly at the floating-point rejection boundary.
    cutoff = math.log(alpha) - math.log1p(draws) - 1e-8

    def accepted(total):
        return hypergeom.logpmf(positives, size, total, draws) >= cutoff

    if not accepted(mode):
        return lower, upper
    left, right = lower, mode
    while left < right:
        mid = (left + right) // 2
        if accepted(mid):
            right = mid
        else:
            left = mid + 1
    low = left
    left, right = mode, upper
    while left < right:
        mid = (left + right + 1) // 2
        if accepted(mid):
            left = mid
        else:
            right = mid - 1
    return low, left

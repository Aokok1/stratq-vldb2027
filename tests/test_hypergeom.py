"""Check binary search and anytime coverage against exact finite populations."""
import itertools
import math
import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from stratq.hypergeom import count_interval


def likelihood(size, draws, observed, total):
    # Hypergeometric PMF from counting subsets with the observed composition.
    return Fraction(math.comb(total, observed) * math.comb(size - total, draws - observed), math.comb(size, draws))


class HypergeomTests(unittest.TestCase):
    def test_matches_exact_inversion(self):
        for n in range(1, 13):
            for draws in range(n + 1):
                for observed in range(draws + 1):
                    for alpha in (Fraction(1, 10), Fraction(1, 3), Fraction(1, 2)):
                        included = [k for k in range(observed, observed + n - draws + 1)
                                    if likelihood(n, draws, observed, k) >= alpha / (draws + 1)]
                        self.assertEqual(count_interval(n, draws, observed, float(alpha)),
                                         (min(included), max(included)))

    def test_ever_excluded_mass(self):
        alpha = Fraction(1, 5)
        for n in range(1, 9):
            intervals = {(draws, observed): count_interval(n, draws, observed, float(alpha))
                         for draws in range(1, n + 1) for observed in range(draws + 1)}
            for k in range(n + 1):
                excluded = 0
                for positives in itertools.combinations(range(n), k):
                    observed = 0
                    for i in range(n):
                        observed += i in positives
                        lo, hi = intervals[i + 1, observed]
                        if not lo <= k <= hi:
                            excluded += 1
                            break
                # Every binary permutation has probability 1 / binom(n, k).
                self.assertLessEqual(Fraction(excluded, math.comb(n, k)), alpha)

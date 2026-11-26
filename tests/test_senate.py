import sys
import unittest
from fractions import Fraction
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from stratq.senate import fixed_interval, senate_quotas


class SenateTests(unittest.TestCase):
    def test_quota_rounding_and_census(self):
        # Census the size-2 group, evenly split the remaining nine calls.
        self.assertEqual(senate_quotas([2, 9, 10], 11), [2, 5, 4])
        self.assertEqual(senate_quotas([2, 9, 10], 21), [2, 9, 10])
        self.assertEqual(senate_quotas([2, 9, 10], 0), [0, 0, 0])

    def test_matches_rational_hypergeometric_inversion(self):
        cutoff = Fraction(1, 20)
        for n in range(1, 10):
            for m in range(n + 1):
                for s in range(m + 1):
                    accepted = []
                    for total in range(s, s+n-m+1):
                        # Count subsets with j positives and m-j negatives exactly.
                        mass = [Fraction(comb(total, j)*comb(n-total, m-j), comb(n, m))
                                for j in range(m+1)]
                        if min(sum(mass[:s+1]), sum(mass[s:])) >= cutoff:
                            accepted.append(total)
                    self.assertEqual(fixed_interval(n, m, s, .1), (min(accepted), max(accepted)))

"""Regression checks for the fixed-budget binary count comparator."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from stratq.proxy_stratified import allocate, fixed_budget_counts


class ProxyStratifiedTests(unittest.TestCase):
    def test_largest_remainder_preserves_budget(self):
        np.testing.assert_array_equal(allocate(7, np.array([1, 1, 2])), [2, 2, 3])

    def test_deterministic_cell_labels(self):
        # Two constant-label cells imply an exact weighted point of 2.
        scores = np.arange(4)
        labels = np.array([0, 0, 1, 1])
        group = fixed_budget_counts(scores, {'g': scores}, lambda ids: labels[ids],
                                    budget=84, delta=0.1, seed=8, strata=2, pilot=2)['g']
        self.assertEqual(group['calls'], 84)
        self.assertEqual(group['stage2_by_stratum'], [40, 40])
        self.assertEqual(group['estimate'], 2.0)
        # Weighted Hoeffding radius: sqrt(log(20) * sum(N_h^2 / m_h) / 2).
        radius = math.sqrt(math.log(20) * (4 / 40 + 4 / 40) / 2)
        self.assertAlmostEqual(group['lower'], 2.0 - radius)
        self.assertAlmostEqual(group['upper'], 2.0 + radius)

    def test_rejects_budget_without_second_stage_support(self):
        with self.assertRaisesRegex(ValueError, 'cannot support'):
            fixed_budget_counts(np.arange(4), {'g': np.arange(4)}, lambda ids: ids,
                                budget=5, delta=0.1, seed=8, strata=2, pilot=2)

import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_stratq import StratQEngine  # Initializes the local src import path.
from stratq.blazeit import blazeit_fixed_counts


class BlazeItFixedTests(unittest.TestCase):
    def test_pilot_and_repeat_accounting(self):
        actual_calls = []
        def oracle(rows):
            actual_calls.extend(rows.tolist())
            return np.ones(len(rows))

        rng = Mock()
        rng.permutation.side_effect = lambda ids: ids
        rng.choice.return_value = np.array([2, 2, 3, 3, 4, 4])
        with patch('stratq.blazeit.np.random.default_rng', return_value=rng):
            result = blazeit_fixed_counts(np.zeros(10), {'A': np.arange(10)}, oracle,
                                           budget=8, delta=0.1, seed=2, pilot=2)['A']
        # With all labels positive, distinct observed rows lower-bound the count.
        # With two pilot rows and six iid draws the EBS range term exceeds one.
        self.assertLess(len(set(actual_calls)), len(actual_calls))
        self.assertEqual(result['calls'], len(actual_calls))
        self.assertEqual(result['lower'], len(set(actual_calls)))
        self.assertEqual(result['upper'], 10)

    def test_group_census_and_budget_floor(self):
        labels = np.array([1, 0, 1, 1, 1, 0])
        groups = {'A': np.arange(2), 'B': np.arange(2, 6)}
        args = (np.zeros(6), groups, lambda rows: labels[rows])
        result = blazeit_fixed_counts(*args, budget=6, delta=0.1, seed=2, pilot=2)
        # All six labels are revealed. Group A has one positive and B has three.
        self.assertEqual({g: (r['lower'], r['upper']) for g, r in result.items()},
                         {'A': (1, 1), 'B': (3, 3)})
        with self.assertRaisesRegex(ValueError, 'Budget must support'):
            blazeit_fixed_counts(*args, budget=5, delta=0.1, seed=2, pilot=2)

import unittest

import numpy as np

from test_stratq import StratQEngine  # exposes local src
from stratq.verdict import variational_count, verdict_fixed_counts
from stratq.senate import senate_fixed_counts


class VerdictTests(unittest.TestCase):
    def test_reflection_and_scale_by_hand(self):
        # Subsample proportions 1 and 0, each with size one; two rows left out.
        # Full sample proportion 3/4. Scaled deviations in total units are 2.5 and -7.5.
        result = variational_count(20, np.array([1,0,1,1]), np.array([1,2,0,0]), .1)
        self.assertEqual((result['estimate'], result['lower'], result['upper']), (15.,12.5,19.))

    def test_same_draws_as_senate_and_exact_at_census(self):
        groups = {'a': np.arange(5), 'b': np.arange(5,15)}
        for budget in (0,8,15):
            calls = [[],[]]
            results = []
            for i,method in enumerate((senate_fixed_counts, verdict_fixed_counts)):
                def oracle(ids):
                    calls[i].extend(ids.tolist())
                    return ids % 2
                results.append(method(groups, oracle, budget=budget, delta=.1, seed=7))
            self.assertEqual(calls[0],calls[1])
            self.assertEqual(len(set(calls[0])),budget)
            if budget == 15:
                for g,ids in groups.items():
                    # Exhausted population: both bounds equal the count of odd row IDs.
                    self.assertEqual((results[1][g]['lower'], results[1][g]['upper']), (sum(ids%2),)*2)

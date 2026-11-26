import math
from itertools import combinations
import unittest

import numpy as np

from test_stratq import StratQEngine  # exposes the local src package
from stratq.online_aggregation import conservative_count, online_fixed_counts


class OnlineAggregationTests(unittest.TestCase):
    def test_known_hoeffding_radius(self):
        # Hellerstein et al. Eq. (1): choose risk 2e^-2, draws=16, size=64.
        # Radius is 64/4=16 around estimated total 32; feasibility is [8,56].
        out = conservative_count(64, 16, 8, 2*math.exp(-2))
        self.assertEqual((out['lower'], out['upper']), (16,48))

    def test_global_budget_census_and_empty_observation(self):
        groups = {'a': np.arange(2), 'b': np.arange(2,6)}
        labels = np.array([1,0,0,1,0,1])
        for budget in (0,1,6):
            seen = []
            def oracle(ids):
                seen.extend(ids)
                return labels[ids]
            out = online_fixed_counts(groups, oracle, budget=budget, delta=.1, seed=12)
            self.assertEqual(len(set(seen)), budget)
            self.assertEqual(sum(v['calls'] for v in out.values()), budget)
            if budget == 6:
                self.assertEqual({g:(v['lower'],v['upper']) for g,v in out.items()}, {'a': (1,1),'b': (2,2)})
            else:
                for g in groups:
                    if not out[g]['calls']:
                        self.assertEqual((out[g]['lower'], out[g]['upper']), (0,len(groups[g])))

    def test_small_population_fixed_look_coverage(self):
        # Enumerate all uniformly drawn subsets, so the miss probability is exact.
        n, alpha = 9, .1
        for total in range(n+1):
            for draws in range(n+1):
                outcomes = list(combinations(range(n), draws))
                miss = 0
                for rows in outcomes:
                    out = conservative_count(n, draws, sum(i < total for i in rows), alpha)
                    miss += not out['lower'] <= total <= out['upper']
                self.assertLessEqual(miss, alpha * len(outcomes))

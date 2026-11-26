from itertools import permutations
import unittest

import numpy as np

from test_stratq import StratQEngine  # exposes local src to unittest discovery
from stratq.lws import des_raj_totals, lws_fixed_counts


class LearnedWeightedSamplingTests(unittest.TestCase):
    def test_des_raj_every_look_under_exact_order_law(self):
        w, y = np.array([1., 2., 5., 3.]), np.array([0, 1, 0, 1])
        expected = np.zeros(len(w))
        for perm in permutations(range(len(w))):
            order = np.array(perm)
            ww = w[order]
            prob = np.prod(ww / np.cumsum(ww[::-1])[::-1])
            expected += prob * des_raj_totals(w, order, y[order])
        # Equation (3): each ordered term has expectation equal to the true total 2.
        np.testing.assert_allclose(expected, [2]*len(w), rtol=0, atol=1e-13)

    def test_census_and_sampling_without_replacement(self):
        ids = {'small': np.array([0,1]), 'large': np.arange(2,7)}
        y = np.array([0,1,1,0,0,1,0])
        for b in (3,7):
            visited = []
            def oracle(row):
                visited.extend(row)
                return y[row]
            out = lws_fixed_counts(ids, np.linspace(0,1,7), oracle, budget=b, delta=.1, seed=51)
            self.assertEqual(len(set(visited)), len(visited))
            self.assertEqual(len(visited), b)
            if b == 7:
                for g,rows in ids.items():
                    # When all labels are observed both endpoints are the literal sum.
                    self.assertEqual((out[g]['lower'], out[g]['upper']), (y[rows].sum(),)*2)

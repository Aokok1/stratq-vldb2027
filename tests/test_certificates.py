"""Deterministic regressions for grouped count and top-k certificates."""
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa

from test_stratq import StratQEngine
from stratq.bounds import WoRControlVariateCS


class CertificateTests(unittest.TestCase):
    def test_census_overrides_conflicting_history(self):
        cs = WoRControlVariateCS(4, 0.05, np.ones(1))
        cs.hi_norm = 0.1  # Inject a conflicting historical interval.
        cs.update_batch(np.ones(1), np.ones((1, 1)), np.zeros(1))
        # One positive with three unknown labels implies [1/4, 1].
        self.assertEqual(cs.bounds, (0.25, 1.0))
        self.assertEqual(cs.estimate, 0.625)
        cs.hi_norm = 0.3  # Inject a second conflict before census.
        cs.update_batch(np.array([1., 0., 0.]), np.ones((3, 1)), np.zeros(1))
        # The observed labels contain exactly two positives.
        self.assertEqual(cs.bounds, (0.5, 0.5))
        self.assertEqual(cs.estimate, 0.5)

    def test_topk_budget_and_indifference(self):
        engine = self.make_engine([1., 1., 0., 0.])
        options = dict(k=1, seed=9, pilot_per_group=0, batch_size=2)
        self.assertFalse(engine.execute_grouped_topk(max_oracle_budget=0, **options).certified)
        # At zero draws, each group's count can be anywhere in [0, 2].
        approx = engine.execute_grouped_topk(indifference_eps=2, **options)
        self.assertFalse(approx.certified)
        self.assertEqual(approx.total_oracle_calls, 0)
        # All four labels are needed by this scheduler. A has two positives, B zero.
        exact = engine.execute_grouped_topk(max_oracle_budget=4, **options)
        self.assertEqual(exact.total_oracle_calls, 4)
        self.assertTrue(exact.certified)
        self.assertEqual(exact.topk_groups, ['A'])

    def test_census_tie(self):
        engine = self.make_engine([1., 0., 1., 0.])
        exact = engine.execute_grouped_topk(k=1, seed=9, pilot_per_group=0, batch_size=2)
        # Both groups have one positive. The profile orders A before B.
        self.assertTrue(exact.certified)
        self.assertEqual(exact.topk_groups, ['A'])
        self.assertEqual(exact.lower, {'A': 1., 'B': 1.})
        self.assertEqual(exact.upper, exact.lower)

    def test_fixed_proxy_keeps_raw_score_in_all_batches(self):
        engine = StratQEngine(
            pa.table({'grp': ['A']*4, 'proxy': [.1, .2, .3, .4], 'oracle': [0, 1, 1, 0]}),
            group_col='grp', proxy_col='proxy', oracle_col='oracle',
        )
        seen = []
        update = WoRControlVariateCS.update_batch

        def record(cs, labels, features, theta, *, exact_f_bounds):
            seen.append((theta.tolist(), exact_f_bounds))
            return update(cs, labels, features, theta, exact_f_bounds=exact_f_bounds)

        with patch.object(WoRControlVariateCS, 'update_batch', record), patch.object(
            engine, '_fit_global_theta', side_effect=AssertionError('Unexpected fit')
        ):
            result = engine.execute_grouped_agg(agg_type='count', eps=0, batch_size=1,
                                                pilot_per_group=1, refit_proxy=False, seed=7)
        # The profile min-max normalizes the proxy to [0,1]; labels sum to 2.
        self.assertEqual(seen, [([0., 1.], (0., 1.))]*4)
        self.assertEqual(result.lower, {'A': 2})
        self.assertEqual(result.upper, result.lower)

    def test_count_rejects_weighted_configuration(self):
        engine = StratQEngine(
            pa.table({'grp': ['A'], 'proxy': [0.5], 'oracle': [1.], 'weight': [7.]}),
            group_col='grp', proxy_col='proxy', oracle_col='oracle', measure_col='weight',
        )
        # The positive-row count is one, whereas the weighted sum is seven.
        with self.assertRaisesRegex(ValueError, 'COUNT.*measure_col'):
            engine.execute_grouped_agg(agg_type='count')
        with self.assertRaisesRegex(ValueError, 'COUNT.*measure_col'):
            engine.execute_grouped_having(tau=1, agg_type='count')
        with self.assertRaisesRegex(ValueError, 'COUNT.*measure_col'):
            engine.execute_grouped_topk(agg_type='count')

    def test_grouped_agg_rejects_unsupported_sampling_model(self):
        engine = self.make_engine([1., 0., 1., 0.])
        with self.assertRaisesRegex(ValueError, 'without replacement'):
            engine.execute_grouped_agg(use_wor=False)

    def test_exact_count_endpoints_and_unequal_group_tie(self):
        # Binary64 maps (7 / 25) * 25 to 7.000000000000001.
        engine = StratQEngine(
            pa.table({'grp': ['A'] * 7 + ['B'] * 25, 'proxy': [0.5] * 32,
                      'oracle': [1] * 14 + [0] * 18}),
            group_col='grp', proxy_col='proxy', oracle_col='oracle',
        )
        options = dict(agg_type='count', pilot_per_group=32)
        count = engine.execute_grouped_agg(eps=0, **options)
        having = engine.execute_grouped_having(tau=7, **options)
        top = engine.execute_grouped_topk(k=1, **options)
        expected = {'A': 7, 'B': 7}
        for result in (count, having, top):
            self.assertEqual(result.lower, expected)
            self.assertEqual(result.upper, expected)
        self.assertEqual(having.accepted_groups, ['A', 'B'])
        self.assertTrue(top.certified)
        self.assertEqual(top.topk_groups, ['A'])

    def test_count_rejects_invalid_sampled_labels(self):
        for label in (float('nan'), 0.5, 2):
            with self.subTest(label=label):
                engine = self.make_engine([label, 1., 0., 0.])
                for run in (engine.execute_grouped_agg,
                            lambda **kw: engine.execute_grouped_having(tau=1, **kw),
                            lambda **kw: engine.execute_grouped_topk(k=1, **kw)):
                    with self.assertRaisesRegex(ValueError, 'binary reference label'):
                        run(agg_type='count', pilot_per_group=2)
        engine = StratQEngine(pa.table({'grp': ['A', 'A'], 'proxy': [0.5, 0.5]}),
                              group_col='grp', proxy_col='proxy', oracle_fn=lambda _: [1])
        with self.assertRaisesRegex(ValueError, 'per sampled row'):
            engine.execute_grouped_agg(agg_type='count', pilot_per_group=2)

    @staticmethod
    def make_engine(labels):
        return StratQEngine(
            pa.table({'grp': ['A', 'A', 'B', 'B'], 'proxy': [0.5] * 4, 'oracle': labels}),
            group_col='grp', proxy_col='proxy', oracle_col='oracle',
        )


if __name__ == '__main__':
    unittest.main()

"""Deterministic algebra/protocol tests, not native MARL performance evidence."""
import itertools
import unittest
import numpy as np
from current_packet_value_control import CurrentPacketController, ProbeOption


class ProtocolTests(unittest.TestCase):
    def controller(self):
        return CurrentPacketController((ProbeOption(2, .1), ProbeOption(4, .2)),
            gain_scale=1., value_weight=2., meta_step=.01, resource_targets=(1., 1.))

    def test_current_scores_change_current_action_and_all_work_is_paid(self):
        c = self.controller()
        opened = c.begin(candidate_hashes=("idle", "A", "B"),
            candidate_costs=((0, 0), (.4, .2), (.4, .2)), probe_costs=(.2, .4),
            sunk_cost=(.5, .3), rng=np.random.default_rng(1))
        samples = np.tile((0., -.2, .3), (opened["episodes"], 1))
        selected = c.select(samples, candidate_hashes=("idle", "A", "B"))
        self.assertEqual(selected["candidate_index"], 2)
        self.assertFalse(selected["selected_score_is_unbiased"])
        record = c.close(committed_hash="B", actual_total_charge=selected["expected_total_charge"])
        self.assertEqual(record["sunk_cost"], [.5, .3])
        self.assertEqual(c.round, 1)
        self.assertFalse(record["hard_budget_guarantee"])

    def test_frozen_candidates_charges_and_transaction_order(self):
        c = self.controller()
        with self.assertRaises(RuntimeError):
            c.select([[0, 1]], candidate_hashes=("idle", "A"))
        opened = c.begin(candidate_hashes=("idle", "A"), candidate_costs=((0, 0), (1, 0)),
            probe_costs=(.2, .4), sunk_cost=(2., 3.), rng=np.random.default_rng(2))
        with self.assertRaises(RuntimeError):
            c.begin(candidate_hashes=("idle",), candidate_costs=((0, 0),), probe_costs=(0, 0),
                sunk_cost=(0, 0), rng=np.random.default_rng(3))
        samples = np.tile((0., 1.), (opened["episodes"], 1))
        with self.assertRaises(ValueError):
            c.select(samples, candidate_hashes=("idle", "changed"))
        selected = c.select(samples, candidate_hashes=("idle", "A"))
        with self.assertRaises(ValueError):
            c.close(committed_hash="A", actual_total_charge=(0, 0))
        self.assertEqual(c.round, 0)
        self.assertIsNotNone(c._open)
        c.close(committed_hash="A", actual_total_charge=selected["expected_total_charge"])

    def test_selection_bias_is_real_and_expected_gap_bound_is_valid(self):
        # Exactly enumerate two independent Rademacher episodes. The action is
        # chosen after scoring, so E[selected estimate] is positive despite mu=0.
        selected_estimates = []
        for first, second in itertools.product((-1., 1.), repeat=2):
            selected_estimates.append(max(0., (first + second) / 2))
        self.assertAlmostEqual(np.mean(selected_estimates), .25)
        # All four outcomes for two candidate gains, with exact population trace.
        means = np.array([0., .2, .1])
        outcomes = [means + np.array([0., a * .3, b * .4])
                    for a, b in itertools.product((-1, 1), repeat=2)]
        variance_trace = .3 ** 2 + .4 ** 2
        gap = np.mean([max(means) - means[np.argmax((x + y) / 2)]
                       for x, y in itertools.product(outcomes, repeat=2)])
        self.assertLessEqual(gap, 2 * np.sqrt(variance_trace / 2))
        for lam in (.01, .1, 1.):
            self.assertLessEqual(2 * np.sqrt(variance_trace / 2), lam + variance_trace / (lam * 2))

    def test_unbiased_variance_feedback_with_data_dependent_action(self):
        outcomes = [np.array([0., a * .3, b * .4])
                    for a, b in itertools.product((-1, 1), repeat=2)]
        traces = [np.stack((x, y)).var(axis=0, ddof=1).sum()
                  for x, y in itertools.product(outcomes, repeat=2)]
        self.assertAlmostEqual(np.mean(traces), .25)
        # Choosing a current action does not change the full-vector sample trace.
        self.assertAlmostEqual(np.mean([s / (.1 * 2) for s in traces]), .25 / (.1 * 2))

    def test_probe_communication_is_priced_and_reconciled(self):
        c = self.controller()
        c.queues[:] = (3., 2.)
        opened = c.begin(candidate_hashes=("idle", "A"), candidate_costs=((0, 0), (.5, .2)),
            probe_costs=((.3, .2), (.8, .4)), sunk_cost=(.7, .5), rng=np.random.default_rng(8))
        selected = c.select(np.tile((0., .1), (opened["episodes"], 1)), candidate_hashes=("idle", "A"))
        self.assertEqual(selected["candidate_index"], 0)
        probe_cost = ((.3, .2), (.8, .4))[opened["probe_index"]]
        np.testing.assert_allclose(selected["expected_total_charge"], np.array([.7, .5])+probe_cost)
        record = c.close(committed_hash="idle", actual_total_charge=selected["expected_total_charge"])
        np.testing.assert_allclose(record["queues_after"], np.maximum(0., np.array([3., 2.])+np.array([.7,.5])+probe_cost-1))


if __name__ == "__main__":
    unittest.main()

"""Deterministic unit tests for the EXP-014A pilot controller."""

import math
import unittest

from run_state_risk_controller_smoke import (
    Action,
    PredictableState,
    choose_action,
    effective_participation,
    risk_score,
    stable,
)


class StateRiskControllerTests(unittest.TestCase):
    def test_effective_participation_matches_endpoints(self) -> None:
        self.assertEqual(effective_participation(32, 0.0), 32.0)
        self.assertEqual(effective_participation(32, 1.0), 1.0)
        self.assertAlmostEqual(
            effective_participation(4, 0.5), 4.0 / 2.5
        )

    def test_delay_screen_rejects_aggressive_action(self) -> None:
        self.assertTrue(stable(Action(4, 1, 0.02), 0))
        self.assertFalse(stable(Action(4, 1, 0.03), 8))
        self.assertTrue(stable(Action(4, 4, 0.03), 8))

    def test_risk_score_is_finite_only_for_safe_actions(self) -> None:
        state = PredictableState()
        self.assertTrue(math.isfinite(risk_score(Action(4, 1, 0.02), state, 0)))
        self.assertTrue(math.isinf(risk_score(Action(4, 1, 0.03), state, 8)))

    def test_gap_does_not_erase_cross_agent_correlation(self) -> None:
        state = PredictableState(rho_upper=0.9, grad_trace=3.0)
        short = risk_score(Action(4, 1, 0.02), state, 0)
        long = risk_score(Action(4, 4, 0.02), state, 0)
        self.assertAlmostEqual(long - short, 0.004 * 3)

    def test_state_risk_action_uses_candidate_set(self) -> None:
        state = PredictableState(
            loss=0.3,
            progress=-0.1,
            grad_trace=2.0,
            rho_upper=0.8,
            tail_gap=0.4,
        )
        action = choose_action("state_risk", state, 8, 0.5)
        self.assertIn(action.q, (1, 4, 16, 32))
        self.assertIn(action.b, (1, 2, 4))
        self.assertIn(action.eta, (0.01, 0.02, 0.03))
        self.assertTrue(stable(action, 8))

    def test_state_risk_cold_start_is_identifiable(self) -> None:
        action = choose_action(
            "state_risk", PredictableState(), delay=0, true_rho=0.0
        )
        self.assertEqual(action, Action(32, 1, 0.03))

    def test_oracle_does_not_use_uncertainty_inflation(self) -> None:
        state = PredictableState(rho_upper=1.0, tail_gap=10.0)
        action = choose_action(
            "charged_information_oracle", state, 0, true_rho=0.0
        )
        expected = min(
            (
                Action(q, b, eta)
                for q in (1, 4, 16, 32)
                for b in (1, 2, 4)
                for eta in (0.01, 0.02, 0.03)
            ),
            key=lambda candidate: risk_score(
                candidate,
                state,
                0,
                rho_override=0.0,
                omit_tail=True,
            ),
        )
        self.assertEqual(action, expected)


if __name__ == "__main__":
    unittest.main()

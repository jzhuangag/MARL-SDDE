from __future__ import annotations

from experiments.nonlinear_markov_td.t061_reward_free_fingerprint import (
    action_from_match_count,
    controller_updates,
    fingerprint_collision_upper_bound,
    phase_action,
    probe_match_count,
)


def test_action_is_nonincreasing_in_match_count() -> None:
    for overhead in (8, 32):
        actions = [
            action_from_match_count(matches=matches, blocks=96, overhead=overhead)
            for matches in range(97)
        ]
        assert all(first >= second for first, second in zip(actions, actions[1:]))
        assert actions[0] == 16
        assert actions[-1] == 1


def test_phase_endpoint_actions() -> None:
    assert phase_action(overhead=8, rho_estimate=0.0) == 16
    assert phase_action(overhead=8, rho_estimate=1.0) == 1
    assert fingerprint_collision_upper_bound(4) == 1 / 1296


def test_controller_charges_both_budgets_and_delay() -> None:
    for overhead in (8, 32):
        for delay in (0, 8):
            for q in (1, 4, 16):
                cost = controller_updates(
                    overhead=overhead,
                    q=q,
                    delay=delay,
                    target_qmax_updates=8192,
                    probe_blocks=96,
                    probe_q=2,
                    fingerprint_length=4,
                )
                assert cost["probe_message"] + cost["learning_message"] <= cost["message_budget"]
                assert cost["probe_environment"] + cost["learning_environment"] <= cost["environment_budget"]


def test_probe_endpoint_couplings_and_reproduction() -> None:
    common = probe_match_count(
        game="breakout", rho=1.0, blocks=8, length=4, master_seed=611
    )
    independent = probe_match_count(
        game="breakout", rho=0.0, blocks=8, length=4, master_seed=611
    )
    repeated = probe_match_count(
        game="breakout", rho=0.0, blocks=8, length=4, master_seed=611
    )
    assert common == 8
    assert independent == repeated
    assert 0 <= independent <= 8

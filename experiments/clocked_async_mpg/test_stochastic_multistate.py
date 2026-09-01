from __future__ import annotations

import numpy as np
import pytest

from .exact_multistate_confirmation import INITIAL_LOGITS, make_game
from .stochastic_multistate import (
    exact_truncated_gradient,
    sample_reinforce_packet,
    simulate_stochastic_asynchronous,
    simulate_stochastic_shadow_barrier,
)


def test_packet_mean_matches_exact_truncated_gradient() -> None:
    game = make_game(0.12)
    horizon = 6
    generator = np.random.default_rng(91901)
    packets = np.asarray(
        [
            sample_reinforce_packet(
                INITIAL_LOGITS,
                game,
                agent=0,
                horizon=horizon,
                batch_size=256,
                generator=generator,
            )
            for _ in range(500)
        ]
    )
    exact = exact_truncated_gradient(INITIAL_LOGITS, game, horizon)[0]
    standard_error = np.std(packets, axis=0, ddof=1)/np.sqrt(packets.shape[0])
    assert (
        np.abs(np.mean(packets, axis=0)-exact)
        <= 5.0*standard_error+2e-3
    ).all()


def test_service_and_trajectory_streams_produce_finite_development_runs() -> None:
    parameters = dict(
        coupling=0.08,
        service_ratio=3.0,
        seed_index=2,
        namespace="stochastic-unit-test",
        maximum_time=20.0,
        horizon=8,
        batch_size=8,
        step_fraction=0.08,
        target_normalized_gap=0.3,
    )
    asynchronous = simulate_stochastic_asynchronous(**parameters)
    shadow = simulate_stochastic_shadow_barrier(**parameters)
    for result in (asynchronous, shadow):
        assert np.isfinite(float(result["final_normalized_gap"]))
        assert int(result["completed_packets"]) > 0
        assert float(result["total_transition_work"]) >= float(
            result["completed_transition_work"]
        )
    assert int(asynchronous["max_realized_delay"]) <= int(
        asynchronous["registered_delay"]
    )
    assert float(shadow["cancelled_transition_work"]) >= 0.0


def test_all_registered_development_step_rules_run() -> None:
    for step_rule in (
        "single_flight_local",
        "single_flight_constant",
        "generic_rate_balanced",
        "common_global",
    ):
        result = simulate_stochastic_asynchronous(
            coupling=0.1,
            service_ratio=2.0,
            seed_index=1,
            namespace="stochastic-step-rule-test",
            maximum_time=8.0,
            horizon=5,
            batch_size=4,
            step_fraction=0.1,
            target_normalized_gap=0.5,
            step_rule=step_rule,
        )
        assert np.isfinite(float(result["final_normalized_gap"]))


def test_unknown_step_rule_is_rejected() -> None:
    with pytest.raises(ValueError):
        simulate_stochastic_asynchronous(
            coupling=0.1,
            service_ratio=2.0,
            seed_index=1,
            namespace="stochastic-step-rule-test",
            maximum_time=8.0,
            horizon=5,
            batch_size=4,
            step_fraction=0.1,
            target_normalized_gap=0.5,
            step_rule="invented",
        )


def test_packet_validation_is_strict() -> None:
    with pytest.raises(ValueError):
        sample_reinforce_packet(
            INITIAL_LOGITS,
            make_game(0.1),
            agent=3,
            horizon=5,
            batch_size=2,
            generator=np.random.default_rng(1),
        )


def test_shadow_charges_terminal_incomplete_round() -> None:
    result = simulate_stochastic_shadow_barrier(
        coupling=0.08,
        service_ratio=2.0,
        seed_index=3,
        namespace="stochastic-terminal-accounting-test",
        maximum_time=0.1,
        horizon=5,
        batch_size=4,
        step_fraction=0.1,
        target_normalized_gap=0.3,
    )
    assert int(result["applied_updates"]) == 0
    assert int(result["completed_packets"]) == 0
    assert float(result["completed_transition_work"]) == 0.0
    assert float(result["cancelled_transition_work"]) > 0.0
    assert float(result["total_transition_work"]) == pytest.approx(
        float(result["cancelled_transition_work"])
    )

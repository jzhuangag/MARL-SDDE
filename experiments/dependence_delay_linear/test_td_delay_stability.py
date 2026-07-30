"""Implementation tests for EXP-007B."""

import numpy as np

from linear_model import make_agent_delays
from linear_td_correlation import (
    LinearTDConfig,
    build_mrp,
    generate_base_paths,
    observed_transition_pairs,
)
from td_delay_stability import (
    build_mean_delay_transition,
    critical_step_size,
    simulate_stability_run,
    spectral_radius,
)


def test_companion_transition_without_delay_matches_mean_td() -> None:
    config = LinearTDConfig()
    mrp = build_mrp(config)
    eta = 0.01
    transition = build_mean_delay_transition(
        mrp["a_matrix"], np.zeros(8, dtype=int), eta
    )
    assert np.allclose(
        transition, np.eye(config.num_features) - eta * mrp["a_matrix"]
    )


def test_bisection_separates_stable_and_unstable_steps() -> None:
    config = LinearTDConfig()
    mrp = build_mrp(config)
    delays = make_agent_delays(
        max_agents=config.num_agents,
        max_delay=32,
        exponent=config.delay_exponent,
    )[:16]
    critical = critical_step_size(mrp["a_matrix"], delays)
    low = spectral_radius(
        build_mean_delay_transition(mrp["a_matrix"], delays, 0.95 * critical)
    )
    high = spectral_radius(
        build_mean_delay_transition(mrp["a_matrix"], delays, 1.05 * critical)
    )
    assert low < 1.0
    assert high > 1.0


def test_registered_simulator_records_stable_run() -> None:
    config = LinearTDConfig()
    mrp = build_mrp(config)
    paths = generate_base_paths(20261030, mrp, config)
    current, following = observed_transition_pairs(paths, 0.0)
    delays = make_agent_delays(
        max_agents=config.num_agents,
        max_delay=8,
        exponent=config.delay_exponent,
    )[:8]
    critical = critical_step_size(mrp["a_matrix"], delays)
    result = simulate_stability_run(
        current,
        following,
        mrp,
        max_delay=8,
        num_agents=8,
        eta=0.05 * critical,
        config=config,
    )
    assert result["finite"]
    assert not result["crossed_threshold"]
    assert result["crossing_time"] == -1

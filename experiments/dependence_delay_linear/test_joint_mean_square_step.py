"""Unit tests for EXP-007C joint mean-square step tools."""

import numpy as np

from joint_mean_square_step import (
    CHECKPOINTS,
    aggregate_second_moment,
    build_mean_boundaries,
    joint_step_size,
    multiplicative_curvature,
    registered_policy_steps,
    sharing_factor,
    simulate_checkpoint_run,
    single_jacobian_second_moment,
    strong_monotonicity,
)
from linear_td_correlation import (
    LinearTDConfig,
    build_mrp,
    generate_base_paths,
    observed_transition_pairs,
)


def test_second_moment_is_symmetric_positive() -> None:
    config = LinearTDConfig()
    mrp = build_mrp(config)
    second = single_jacobian_second_moment(mrp, config)
    assert np.allclose(second, second.T)
    assert np.linalg.eigvalsh(second).min() > 0.0
    assert strong_monotonicity(mrp["a_matrix"]) > 0.0


def test_exchangeable_aggregate_second_moment_endpoints() -> None:
    config = LinearTDConfig()
    mrp = build_mrp(config)
    a_matrix = mrp["a_matrix"]
    second = single_jacobian_second_moment(mrp, config)
    assert sharing_factor(16, 1.0) == 1.0
    assert np.allclose(
        aggregate_second_moment(a_matrix, second, 16, 1.0),
        second,
    )
    expected = (
        second / 16.0
        + 15.0 / 16.0 * a_matrix.T.dot(a_matrix)
    )
    assert np.allclose(
        aggregate_second_moment(a_matrix, second, 16, 0.0),
        expected,
    )


def test_correlation_saturates_multiplicative_curvature() -> None:
    config = LinearTDConfig()
    mrp = build_mrp(config)
    second = single_jacobian_second_moment(mrp, config)
    values = {}
    for q in (16, 32):
        for rho in (0.0, 0.9):
            values[(q, rho)] = multiplicative_curvature(
                mrp["a_matrix"], second, q, rho
            )
    assert values[(32, 0.0)] < 0.8 * values[(16, 0.0)]
    assert values[(32, 0.9)] > 0.98 * values[(16, 0.9)]
    assert values[(16, 0.9)] > 5.0 * values[(16, 0.0)]


def test_joint_step_is_below_both_component_thresholds() -> None:
    eta = joint_step_size(0.5, 2.0, 0.1)
    assert eta < 0.5
    assert eta < 2.0 * 0.1 / 2.0
    assert eta > 0.0


def test_registered_steps_are_correlation_and_delay_sensitive() -> None:
    config = LinearTDConfig()
    mrp = build_mrp(config)
    second = single_jacobian_second_moment(mrp, config)
    boundaries = build_mean_boundaries(mrp["a_matrix"], config)
    independent = registered_policy_steps(
        mrp["a_matrix"], second, boundaries, 32, 32, 0.0
    )
    correlated = registered_policy_steps(
        mrp["a_matrix"], second, boundaries, 32, 32, 0.9
    )
    assert correlated["joint_aware"] < independent["joint_aware"]
    assert np.isclose(
        correlated["correlation_blind"],
        independent["joint_aware"],
    )
    assert correlated["delay_blind"] > correlated["joint_aware"]
    assert correlated["joint_aware"] < correlated["mean_only"]


def test_checkpoint_simulator_returns_registered_outputs() -> None:
    config = LinearTDConfig()
    mrp = build_mrp(config)
    paths = generate_base_paths(711, mrp, config)
    current, following = observed_transition_pairs(paths, 0.0)
    result = simulate_checkpoint_run(
        current, following, mrp, 8, 16, 0.01, config
    )
    assert not result["crossed_threshold"]
    assert result["crossing_time"] == -1
    assert result["finite"]
    for checkpoint in CHECKPOINTS:
        assert np.isfinite(result["error_{0}".format(int(checkpoint))])


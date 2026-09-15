import itertools
import math

import numpy as np

from experiments.dependence_delay_linear.t051_fingerprint_probe import (
    catalogue_optimal_intervals,
    expected_plugin_coefficient_bound,
    fingerprint_correlation_certificate,
    minimum_fingerprint_length,
    pairwise_fingerprint_match_rate,
    plug_in_action,
    state_path_collision_probability,
    trajectory_switch_match_probability,
)


def test_state_path_collision_matches_complete_enumeration():
    transition = np.array([[0.75, 0.25], [0.5, 0.5]])
    stationary = np.array([2.0 / 3.0, 1.0 / 3.0])
    exact = state_path_collision_probability(
        transition=transition, stationary=stationary, transitions=2
    )
    paths = list(itertools.product(range(2), repeat=3))
    probabilities = []
    for path in paths:
        probabilities.append(
            stationary[path[0]]
            * transition[path[0], path[1]]
            * transition[path[1], path[2]]
        )
    assert abs(exact - sum(value**2 for value in probabilities)) < 1e-15


def test_minimum_fingerprint_length_is_shortest_valid_length():
    transition = np.array([[0.75, 0.25], [0.5, 0.5]])
    stationary = np.array([2.0 / 3.0, 1.0 / 3.0])
    result = minimum_fingerprint_length(
        transition=transition,
        stationary=stationary,
        maximum_collision=0.1,
    )
    assert result["collision_probability"] <= 0.1
    if result["transitions"] > 0:
        previous = state_path_collision_probability(
            transition=transition,
            stationary=stationary,
            transitions=result["transitions"] - 1,
        )
        assert previous > 0.1


def test_pairwise_match_rate_counts_clusters():
    fingerprints = np.array([[1, 2], [1, 2], [1, 2], [4, 5], [4, 5]])
    assert pairwise_fingerprint_match_rate(fingerprints) == 0.4


def test_match_identity_and_certificate_transform():
    collision = 0.02
    rho = 0.6
    match = trajectory_switch_match_probability(
        rho=rho, collision_probability=collision
    )
    assert match == collision + (1.0 - collision) * rho
    certificate = fingerprint_correlation_certificate(
        np.full(200, match), collision_probability=collision, alpha=0.05
    )
    assert abs(certificate.estimate - rho) < 1e-15
    assert certificate.lower <= rho <= certificate.upper
    expected_radius = math.sqrt(math.log(40.0) / 400.0) / 0.98
    assert abs(certificate.radius - expected_radius) < 1e-15


def test_catalogue_intervals_are_exact():
    intervals_8 = catalogue_optimal_intervals([1, 4, 16], overhead=8.0)
    assert np.allclose(intervals_8[16], (0.0, 1.0 / 9.0))
    assert np.allclose(intervals_8[4], (1.0 / 9.0, 2.0 / 3.0))
    assert np.allclose(intervals_8[1], (2.0 / 3.0, 1.0))
    intervals_32 = catalogue_optimal_intervals([1, 4, 16], overhead=32.0)
    assert np.allclose(intervals_32[16], (0.0, 1.0 / 3.0))
    assert np.allclose(intervals_32[4], (1.0 / 3.0, 8.0 / 9.0))
    assert np.allclose(intervals_32[1], (8.0 / 9.0, 1.0))


def test_plugin_action_moves_down_with_correlation():
    selected = [
        plug_in_action(rho, [1, 4, 16], overhead=8.0)
        for rho in np.linspace(0.0, 1.0, 101)
    ]
    assert all(left >= right for left, right in zip(selected, selected[1:]))
    assert selected[0] == 16
    assert selected[-1] == 1


def test_expected_bound_converges_to_oracle_away_from_boundaries():
    short = expected_plugin_coefficient_bound(
        rho=0.0,
        candidates=[1, 4, 16],
        overhead=8.0,
        blocks=16,
        collision_probability=0.01,
    )
    long = expected_plugin_coefficient_bound(
        rho=0.0,
        candidates=[1, 4, 16],
        overhead=8.0,
        blocks=1024,
        collision_probability=0.01,
    )
    assert long["expected_coefficient_upper_bound"] < short[
        "expected_coefficient_upper_bound"
    ]
    assert long["expected_coefficient_upper_bound"] >= long["oracle_coefficient"]

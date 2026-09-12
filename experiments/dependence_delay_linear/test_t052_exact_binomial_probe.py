import math

import numpy as np

from experiments.dependence_delay_linear.t051_fingerprint_probe import (
    expected_plugin_coefficient_bound,
)
from experiments.dependence_delay_linear.t052_exact_binomial_probe import (
    binomial_probabilities,
    exact_full_cost_plugin_ratio,
    exact_plugin_action_distribution,
    exact_plugin_expected_coefficient,
)


def test_binomial_mass_vector_has_exact_endpoints_and_moments():
    assert np.array_equal(binomial_probabilities(5, 0.0), [1, 0, 0, 0, 0, 0])
    assert np.array_equal(binomial_probabilities(5, 1.0), [0, 0, 0, 0, 0, 1])
    masses = binomial_probabilities(20, 0.3)
    counts = np.arange(21)
    assert abs(np.sum(masses) - 1.0) < 1e-15
    assert abs(float(counts @ masses) - 6.0) < 1e-13
    assert abs(float(((counts - 6.0) ** 2) @ masses) - 4.2) < 1e-12


def test_exact_action_distribution_is_normalized_and_directional():
    low = exact_plugin_action_distribution(
        rho=0.0,
        candidates=[1, 4, 16],
        overhead=8.0,
        blocks=96,
        collision_probability=0.01,
    )
    high = exact_plugin_action_distribution(
        rho=1.0,
        candidates=[1, 4, 16],
        overhead=8.0,
        blocks=96,
        collision_probability=0.01,
    )
    assert abs(sum(low.values()) - 1.0) < 1e-14
    assert abs(sum(high.values()) - 1.0) < 1e-14
    assert low[16] > 0.99
    assert high[1] == 1.0


def test_exact_expected_coefficient_is_below_valid_hoeffding_bound():
    for rho in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        exact = exact_plugin_expected_coefficient(
            rho=rho,
            candidates=[1, 4, 16],
            overhead=32.0,
            blocks=96,
            collision_probability=0.01,
        )
        bound = expected_plugin_coefficient_bound(
            rho=rho,
            candidates=[1, 4, 16],
            overhead=32.0,
            blocks=96,
            collision_probability=0.01,
        )
        assert exact["oracle_coefficient"] <= exact["expected_coefficient"]
        assert exact["expected_coefficient"] <= bound[
            "expected_coefficient_upper_bound"
        ] + 1e-12


def test_full_cost_ratio_charges_two_agent_probe_messages():
    result = exact_full_cost_plugin_ratio(
        rho=0.3,
        candidates=[1, 4, 16],
        overhead=32.0,
        baseline_q=16,
        learning_budget=5000.0 * 48.0,
        probe_blocks=96,
        collision_probability=0.01,
    )
    assert result["probe_message_cost"] == 96.0 * 34.0
    assert result["total_message_budget"] == 5000.0 * 48.0 + 96.0 * 34.0
    assert math.isfinite(result["expected_risk_ratio"])

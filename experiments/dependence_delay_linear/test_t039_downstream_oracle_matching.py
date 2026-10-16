import math

import pytest

from experiments.dependence_delay_linear.t039_downstream_oracle_matching import (
    logarithmic_probe_count,
    oracle_matching_bound,
)


def test_probe_count_certifies_polynomial_error() -> None:
    scale = 1000.0
    information = 0.4
    probes = logarithmic_probe_count(scale, information, error_power=3.0)
    assert 0.5 * math.exp(-information * probes) <= 0.5 * scale**-3


def test_matching_bound_has_log_over_budget_squared_excess() -> None:
    scales = [1e4, 1e5, 1e6]
    normalized = []
    relative = []
    for scale in scales:
        result = oracle_matching_bound(
            budget_scale=scale,
            information_exponent=0.5,
            probe_cost_per_sample=2.0,
            delay_cost=5.0,
            oracle_risk_lower_coefficient=0.2,
            oracle_budget_sensitivity=3.0,
            wrong_commit_coefficient=4.0,
        )
        assert result["feasible"]
        normalized.append(
            result["absolute_excess_bound"] * scale**2 / math.log(scale)
        )
        relative.append(result["relative_excess_bound"])
    assert max(normalized) / min(normalized) < 1.25
    assert relative[2] < relative[1] < relative[0]


def test_bound_refuses_budget_without_half_horizon_reserve() -> None:
    result = oracle_matching_bound(
        budget_scale=20.0,
        information_exponent=0.1,
        probe_cost_per_sample=4.0,
        delay_cost=5.0,
        oracle_risk_lower_coefficient=0.2,
        oracle_budget_sensitivity=3.0,
        wrong_commit_coefficient=4.0,
    )
    assert not result["feasible"]
    assert math.isinf(result["relative_excess_bound"])


def test_invalid_zero_oracle_coefficient_is_rejected() -> None:
    with pytest.raises(ValueError):
        oracle_matching_bound(
            budget_scale=100.0,
            information_exponent=0.5,
            probe_cost_per_sample=1.0,
            delay_cost=0.0,
            oracle_risk_lower_coefficient=0.0,
            oracle_budget_sensitivity=1.0,
            wrong_commit_coefficient=1.0,
        )

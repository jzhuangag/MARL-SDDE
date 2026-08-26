import math

import pytest

from experiments.dependence_delay_linear.t047_scheduled_participation import (
    AffineRisk,
)
from experiments.dependence_delay_linear.t052_exact_binomial_probe import (
    exact_full_cost_plugin_ratio,
)
from experiments.dependence_delay_linear.t056_end_to_end_finite_certificate import (
    exact_finite_plugin_certificate,
    feasible_fixed_horizon,
    finite_no_harm_certificate,
)


def test_dual_budget_horizon_charges_probe_and_delay():
    updates, use = feasible_fixed_horizon(
        message_budget=1000,
        environment_budget=100,
        overhead=8,
        participation=4,
        probe_message=100,
        probe_environment=10,
        delay=8,
    )
    completed_rounds = min((1000 - 100) // 12, (100 - 10) // 4)
    assert updates == completed_rounds - 8 == 14
    assert use.message == 100 + completed_rounds * 12 == 364
    assert use.environment == 10 + completed_rounds * 4 == 98
    assert use.delay_reserve == 8

    # A larger participation level consumes more actor transitions per round
    # and therefore has a shorter horizon under the same environment budget.
    larger_q_updates, larger_q_use = feasible_fixed_horizon(
        message_budget=1000,
        environment_budget=100,
        overhead=8,
        participation=10,
        probe_message=100,
        probe_environment=10,
        delay=8,
    )
    assert larger_q_updates == min(900 // 18, 90 // 10) - 8 == 1
    assert larger_q_updates < updates
    assert larger_q_use.environment == 100

    with pytest.raises(ValueError, match="exceed"):
        feasible_fixed_horizon(
            message_budget=50,
            environment_budget=10,
            overhead=8,
            participation=4,
            probe_message=51,
        )
    with pytest.raises(ValueError, match="delay reserve"):
        feasible_fixed_horizon(
            message_budget=100,
            environment_budget=10,
            overhead=8,
            participation=4,
            delay=3,
        )


def test_exact_finite_mixture_uses_candidate_specific_risks():
    result = exact_finite_plugin_certificate(
        rho=0.3,
        candidates=[1, 4, 16],
        overhead=8.0,
        probe_blocks=20,
        collision_probability=0.01,
        post_probe_risks={
            1: AffineRisk(1.2, 0.2),
            4: AffineRisk(0.8, 0.1),
            16: AffineRisk(0.7, 0.9),
        },
        full_budget_baseline_q=4,
        full_budget_baseline_risk=AffineRisk(0.75, 0.1),
    )
    distribution = result["action_distribution"]
    risks = result["post_probe_finite_risks"]
    direct = sum(distribution[q] * risks[q] for q in distribution)
    assert math.isclose(result["expected_controller_risk"], direct)
    assert math.isclose(sum(distribution.values()), 1.0)
    assert result["finite_oracle_q"] == 4


def test_perfect_identification_reduces_to_selected_finite_risk():
    result = exact_finite_plugin_certificate(
        rho=1.0,
        candidates=[1, 4, 16],
        overhead=32.0,
        probe_blocks=96,
        collision_probability=0.01,
        post_probe_risks={
            1: AffineRisk(0.4, 0.2),
            4: AffineRisk(0.5, 0.2),
            16: AffineRisk(0.6, 0.2),
        },
        full_budget_baseline_q=16,
        full_budget_baseline_risk=AffineRisk(0.7, 0.2),
    )
    assert result["action_distribution"][1] == 1.0
    assert result["expected_controller_risk"] == pytest.approx(0.6)


def test_finite_no_harm_certificate_has_explicit_margin():
    passed = finite_no_harm_certificate(
        expected_controller_risk=1.02, baseline_risk=1.0, tolerance=0.05
    )
    failed = finite_no_harm_certificate(
        expected_controller_risk=1.06, baseline_risk=1.0, tolerance=0.05
    )
    assert passed["certified"] and passed["margin"] == pytest.approx(0.03)
    assert not failed["certified"] and failed["margin"] == pytest.approx(-0.01)


def test_finite_identity_recovers_t052_leading_full_cost_ratio():
    overhead = 32.0
    candidates = [1, 4, 16]
    baseline_q = 16
    learning_budget = 240_000.0
    probe_blocks = 96
    total_budget = learning_budget + probe_blocks * (overhead + 2.0)

    def leading_risk(q: int, budget: float) -> AffineRisk:
        return AffineRisk(
            intercept=(overhead + q) / (q * budget),
            slope=(overhead + q) * (1.0 - 1.0 / q) / budget,
        )

    finite = exact_finite_plugin_certificate(
        rho=0.3,
        candidates=candidates,
        overhead=overhead,
        probe_blocks=probe_blocks,
        collision_probability=0.01,
        post_probe_risks={q: leading_risk(q, learning_budget) for q in candidates},
        full_budget_baseline_q=baseline_q,
        full_budget_baseline_risk=leading_risk(baseline_q, total_budget),
    )
    leading = exact_full_cost_plugin_ratio(
        rho=0.3,
        candidates=candidates,
        overhead=overhead,
        baseline_q=baseline_q,
        learning_budget=learning_budget,
        probe_blocks=probe_blocks,
        collision_probability=0.01,
    )
    assert finite["expected_controller_to_baseline_ratio"] == pytest.approx(
        leading["expected_risk_ratio"]
    )

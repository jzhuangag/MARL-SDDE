"""Tests for the predictable EXP-009A controller."""

import numpy as np

from predictable_mixing_controller import (
    PILOT_ALPHA,
    clopper_pearson_upper,
    exact_policy_metrics,
    select_action,
    select_finite_budget_action,
    select_joint_action,
)


def test_clopper_pearson_upper_is_one_sided_and_finite() -> None:
    upper = clopper_pearson_upper(1840, 2048, PILOT_ALPHA)
    assert 1840 / 2048 < upper < 1.0
    assert clopper_pearson_upper(2048, 2048, PILOT_ALPHA) == 1.0


def test_more_persistent_certificate_selects_larger_gap() -> None:
    low = select_action(0.55, rho=0.0, delay=0, pilot_cost=2048)
    high = select_action(0.985, rho=0.0, delay=0, pilot_cost=2048)
    assert high["gap"] > low["gap"]


def test_covered_action_is_exactly_stable() -> None:
    action = select_action(
        0.985, rho=0.9, delay=2, pilot_cost=2048
    )
    exact = exact_policy_metrics(
        action, true_persistence=0.98, rho=0.9, delay=2
    )
    assert exact["exact_radius"] < 1.0
    assert np.isfinite(exact["expected_final_error"])


def test_finite_budget_step_reduces_registered_risk() -> None:
    rate = select_action(0.9, rho=0.9, delay=0, pilot_cost=2048)
    finite = select_finite_budget_action(
        0.9, rho=0.9, delay=0, pilot_cost=2048
    )
    assert finite["risk_surrogate"] <= rate["risk_surrogate"]
    assert finite["eta"] > 0.0


def test_joint_gap_search_is_safe_and_no_worse_than_fixed_target() -> None:
    fixed = select_finite_budget_action(
        0.985, rho=0.9, delay=2, pilot_cost=2048
    )
    joint = select_joint_action(
        0.985, rho=0.9, delay=2, pilot_cost=2048
    )
    assert joint["risk_surrogate"] <= fixed["risk_surrogate"]
    exact = exact_policy_metrics(
        joint, true_persistence=0.98, rho=0.9, delay=2
    )
    assert exact["exact_radius"] < 1.0

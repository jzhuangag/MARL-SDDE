from __future__ import annotations

import pytest

from .dual_use_fingerprint import BinaryGeometryBelief
from .dual_use_value_of_information import (
    binary_gaussian_expectation_after_observation,
    binary_gaussian_posterior_hinge,
    choose_dual_use_lookahead,
)


def _decision(probability: float, **overrides: float | bool):
    values: dict[str, float | bool] = {
        "potential_to_rotation": 0.025,
        "rotation_to_potential": 0.075,
        "potential_log_gain": -0.2,
        "rotational_log_gain": 0.6,
        "observation_standard_deviation": 0.1,
        "resource_debt": 0.0,
        "average_optimism_budget": 0.25,
        "lyapunov_tradeoff": 32.0,
        "hard_feasible": True,
    }
    values.update(overrides)
    return choose_dual_use_lookahead(BinaryGeometryBelief(probability), **values)


def test_known_potential_does_not_buy_optimism_or_information() -> None:
    decision = _decision(0.0)
    assert not decision.use_optimism
    assert decision.same_debt_information_value == pytest.approx(0.0, abs=1e-12)


def test_known_rotation_buys_the_contracting_action() -> None:
    decision = _decision(1.0)
    assert decision.use_optimism
    assert decision.immediate_expected_log_gain == pytest.approx(0.6)
    assert decision.same_debt_information_value == pytest.approx(0.0, abs=1e-12)


def test_information_value_is_nonnegative_for_uncertain_belief() -> None:
    decision = _decision(0.35, resource_debt=8.0)
    assert decision.same_debt_information_value >= 0.0


def test_hard_infeasibility_overrides_a_favorable_call() -> None:
    assert not _decision(1.0, hard_feasible=False).use_optimism


def test_gaussian_posterior_preserves_the_prior_in_expectation() -> None:
    belief = BinaryGeometryBelief(0.37)
    expected = binary_gaussian_expectation_after_observation(
        belief,
        observation_standard_deviation=0.4,
        value=lambda posterior: posterior.rotation_probability,
        quadrature_order=101,
    )
    assert expected == pytest.approx(0.37, abs=2e-8)


@pytest.mark.parametrize(
    ("probability", "sigma", "intercept", "slope"),
    [
        (0.2, 0.15, 0.1, 0.8),
        (0.5, 0.4, 0.3, 0.9),
        (0.8, 0.7, -0.1, 0.4),
        (0.35, 0.3, -0.2, -0.8),
        (0.6, 0.5, 0.9, -1.2),
    ],
)
def test_closed_form_hinge_matches_high_order_quadrature(
    probability: float,
    sigma: float,
    intercept: float,
    slope: float,
) -> None:
    belief = BinaryGeometryBelief(probability)
    closed = binary_gaussian_posterior_hinge(
        belief,
        observation_standard_deviation=sigma,
        intercept=intercept,
        posterior_slope=slope,
    )
    numerical = binary_gaussian_expectation_after_observation(
        belief,
        observation_standard_deviation=sigma,
        value=lambda posterior: min(
            0.0, intercept - slope * posterior.rotation_probability
        ),
        quadrature_order=151,
    )
    assert closed == pytest.approx(numerical, abs=3e-5)


def test_call_debt_exceeds_no_call_debt_by_one_away_from_reflection() -> None:
    decision = _decision(0.5, resource_debt=2.0)
    assert (
        decision.resource_debt_if_call - decision.resource_debt_if_no_call
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("probability", (-0.1, 1.1))
def test_invalid_beliefs_are_rejected(probability: float) -> None:
    with pytest.raises(ValueError):
        _decision(probability)

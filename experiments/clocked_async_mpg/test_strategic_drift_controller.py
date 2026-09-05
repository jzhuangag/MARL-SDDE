from __future__ import annotations

import numpy as np
import pytest

from .strategic_drift_controller import (
    choose_strategic_drift_scale,
    no_harm_scale_cap,
    strategic_improvement_lower_bound,
    update_certificate_debt,
)


def test_mixed_drift_bound_is_exact_on_bilinear_quadratic() -> None:
    # J(x, y)=a*x-c*x^2+m*x*y.  Moving the teammate from y=0 to
    # y=-delta realizes exactly the claimed strategic-staleness penalty.
    gain, curvature, mixed, teammate_drift = 1.4, 0.7, 0.6, 0.25
    scale = 0.8
    true_change = (
        gain*scale-curvature*scale**2
        +mixed*scale*(-teammate_drift)
    )
    lower = strategic_improvement_lower_bound(
        scale,
        directional_gain=gain,
        curvature_penalty=curvature,
        stale_penalty=mixed*teammate_drift,
    )
    assert lower == pytest.approx(true_change)
    assert strategic_improvement_lower_bound(
        0.0, gain, curvature, mixed*teammate_drift
    ) == 0.0


def test_closed_form_decision_matches_dense_scalar_optimization() -> None:
    rng = np.random.default_rng(20270901)
    grid = np.linspace(0.0, 1.0, 200_001)
    for _ in range(40):
        gain = float(rng.uniform(-0.2, 2.0))
        curvature = float(rng.uniform(0.0, 1.5))
        stale = float(rng.uniform(0.0, 0.9))
        debt = float(rng.uniform(0.0, 4.0))
        tradeoff = float(rng.uniform(0.1, 3.0))
        cap = float(rng.uniform(0.1, 1.0))
        decision = choose_strategic_drift_scale(
            directional_gain=gain,
            curvature_penalty=curvature,
            stale_penalty=stale,
            debt=debt,
            risk_budget=0.1,
            tradeoff=tradeoff,
            maximum_scale=cap,
        )
        feasible = grid[grid <= cap]
        values = (
            tradeoff*gain*feasible
            -debt*(curvature*feasible**2+stale*feasible)
        )
        brute = float(feasible[int(np.argmax(values))])
        assert decision.scale == pytest.approx(brute, abs=1e-5)


def test_hard_shield_never_returns_negative_certified_improvement() -> None:
    for gain, curvature, stale in (
        (1.0, 0.4, 0.2),
        (0.1, 0.4, 0.2),
        (-0.1, 0.0, 0.0),
        (1.0, 0.0, 0.2),
    ):
        decision = choose_strategic_drift_scale(
            directional_gain=gain,
            curvature_penalty=curvature,
            stale_penalty=stale,
            debt=0.0,
            risk_budget=0.0,
            tradeoff=1.0,
            hard_no_harm=True,
        )
        assert decision.improvement_lower_bound >= -1e-12
    assert no_harm_scale_cap(0.1, 0.4, 0.2) == 0.0


def test_virtual_queue_gives_pathwise_cumulative_penalty_accounting() -> None:
    penalties = [0.6, 0.1, 0.5, 0.0, 0.8]
    budgets = [0.25]*len(penalties)
    debt = 0.0
    for penalty, budget in zip(penalties, budgets, strict=True):
        debt = update_certificate_debt(debt, penalty, budget)
    assert sum(penalties) <= sum(budgets)+debt+1e-12
    assert debt == pytest.approx(0.75)


def test_validation_selection_error_is_paid_explicitly() -> None:
    true_gain = 0.3
    noises = (-0.25, 0.25)
    selected = []
    for noise in noises:
        decision = choose_strategic_drift_scale(
            directional_gain=true_gain+noise,
            curvature_penalty=0.2,
            stale_penalty=0.1,
            debt=1.5,
            risk_budget=0.05,
            tradeoff=2.0,
        )
        selected.append((noise, decision.scale))
    true_selected_gain = np.mean(
        [true_gain*scale for _, scale in selected]
    )
    observed_selected_gain = np.mean(
        [(true_gain+noise)*scale for noise, scale in selected]
    )
    mean_absolute_noise = np.mean([abs(noise) for noise, _ in selected])
    assert true_selected_gain >= observed_selected_gain-mean_absolute_noise-1e-12


@pytest.mark.parametrize(
    "kwargs",
    [
        {"directional_gain": 1.0, "curvature_penalty": -1.0},
        {"directional_gain": 1.0, "stale_penalty": -1.0},
        {"directional_gain": 1.0, "debt": -1.0},
        {"directional_gain": 1.0, "risk_budget": -1.0},
        {"directional_gain": 1.0, "tradeoff": 0.0},
        {"directional_gain": 1.0, "maximum_scale": 2.0},
    ],
)
def test_invalid_controller_inputs_are_rejected(kwargs: dict[str, float]) -> None:
    values = {
        "directional_gain": 1.0,
        "curvature_penalty": 0.2,
        "stale_penalty": 0.1,
        "debt": 0.0,
        "risk_budget": 0.0,
        "tradeoff": 1.0,
        "maximum_scale": 1.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        choose_strategic_drift_scale(**values)

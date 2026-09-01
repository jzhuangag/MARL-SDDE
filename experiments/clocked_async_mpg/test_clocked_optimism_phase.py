from __future__ import annotations

import numpy as np
import pytest

from .clocked_optimism_phase import (
    choose_clocked_optimism,
    potential_coordinate_factors,
    randomized_factor,
    rotational_coordinate_factors,
    rotational_optimism_threshold,
)


@pytest.mark.parametrize("step", [0.1, 0.3, 0.6, 0.9])
def test_plain_coordinate_gradient_expands_a_rotational_game(step: float) -> None:
    factors = rotational_coordinate_factors(step)
    assert factors.plain > 1.0
    assert factors.optimistic < 1.0


@pytest.mark.parametrize("step", [0.1, 0.3, 0.6, 0.9])
def test_optimism_is_strictly_wasteful_in_a_potential_phase(step: float) -> None:
    factors = potential_coordinate_factors(step)
    assert factors.plain < factors.optimistic < 1.0


@pytest.mark.parametrize("step", [0.1, 0.3, 0.6, 0.9])
def test_exact_rotational_phase_boundary(step: float) -> None:
    threshold = rotational_optimism_threshold(step)
    assert threshold == pytest.approx(1.0 / (2.0 - step * step))
    factors = rotational_coordinate_factors(step)
    assert randomized_factor(factors, threshold - 1e-6) > 1.0
    assert randomized_factor(factors, threshold + 1e-6) < 1.0


def test_clocked_controller_uses_optimism_only_for_positive_drift_value() -> None:
    rotation = choose_clocked_optimism(
        energy=1.0,
        factors=rotational_coordinate_factors(0.5),
        resource_debt=0.0,
        average_optimism_budget=0.25,
        lyapunov_tradeoff=10.0,
    )
    potential = choose_clocked_optimism(
        energy=1.0,
        factors=potential_coordinate_factors(0.5),
        resource_debt=0.0,
        average_optimism_budget=0.25,
        lyapunov_tradeoff=10.0,
    )
    assert rotation.use_optimism
    assert not potential.use_optimism


def test_resource_debt_prevents_unbudgeted_always_optimistic_behavior() -> None:
    factors = rotational_coordinate_factors(0.4)
    debt = 0.0
    uses = []
    energy = 1.0
    for _ in range(200):
        decision = choose_clocked_optimism(
            energy=energy,
            factors=factors,
            resource_debt=debt,
            average_optimism_budget=0.25,
            lyapunov_tradeoff=5.0,
        )
        uses.append(decision.use_optimism)
        debt = decision.resource_debt_after
        # Hold energy fixed here to isolate virtual-queue accounting.
    assert np.mean(uses) <= 0.25 + 1.0 / len(uses)


def test_invalid_normalized_step_is_rejected() -> None:
    for step in (0.0, 1.0, np.inf):
        with pytest.raises(ValueError):
            rotational_coordinate_factors(step)

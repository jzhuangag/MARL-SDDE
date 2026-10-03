from __future__ import annotations

import numpy as np
import pytest

from experiments.policy_update_backpressure.transport_headroom import (
    HORIZON,
    baseline_specs,
    declared_cells,
    gradient,
    integrated_regret,
    matrix,
    potential,
    simulate_event_policy,
    validate_design,
)


def test_static_headroom_cardinality() -> None:
    design = validate_design()
    assert design["cells"] == 324
    assert design["phase_cells"] == {"low": 54, "transition": 216, "high": 54}
    assert design["comparators"] == 54
    assert design["trajectories"] == 17_820


def test_cell_ids_and_baseline_names_are_unique() -> None:
    assert len({cell.cell_id for cell in declared_cells()}) == 324
    names = [name for name, _ in baseline_specs()]
    assert len(names) == len(set(names)) == 54


def test_quadratic_gradient_matches_finite_difference() -> None:
    cell = declared_cells()[123]
    hessian = matrix(cell)
    theta = np.asarray(cell.initial, dtype=float)
    analytic = gradient(theta, hessian)
    eps = 1e-6
    for index in range(cell.n_agents):
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += eps
        minus[index] -= eps
        numeric = (potential(plus, hessian)-potential(minus, hessian))/(2*eps)
        assert analytic[index] == pytest.approx(numeric, abs=1e-8)


def test_wall_clock_integral() -> None:
    trajectory = [(0.0, -2.0), (3.0, -1.0), (5.0, 0.0)]
    assert integrated_regret(trajectory, horizon=7.0) == pytest.approx(8.0)


def test_exact_transport_has_no_harmful_steps_in_smoke_cells() -> None:
    selected = [declared_cells()[0], declared_cells()[-1], declared_cells()[201]]
    for cell in selected:
        result = simulate_event_policy(cell, mode="transport", eta=1.0)
        assert result["harmful"] == 0
        assert 0 <= result["normalized_regret"] <= 1
        assert np.isfinite(result["final_gradient_norm"])


def test_hvp_overhead_is_fully_charged() -> None:
    zero = next(
        cell for cell in declared_cells()
        if cell.phase == "high" and cell.hvp_overhead == 0.0
    )
    charged = next(
        cell for cell in declared_cells()
        if cell.n_agents == zero.n_agents
        and cell.coupling == zero.coupling
        and cell.latency_pattern == zero.latency_pattern
        and cell.slow_latency == zero.slow_latency
        and cell.initial_name == zero.initial_name
        and cell.hvp_overhead == 1.0
    )
    result_zero = simulate_event_policy(zero, mode="transport", eta=1.0)
    result_charged = simulate_event_policy(charged, mode="transport", eta=1.0)
    assert result_charged["accepted"] <= result_zero["accepted"]


def test_declared_horizon_is_positive() -> None:
    assert HORIZON == 96.0

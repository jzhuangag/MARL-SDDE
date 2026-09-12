from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.policy_update_backpressure.phase_pilot import (
    Cell,
    HORIZON,
    PILOT_SEEDS,
    baseline_specs,
    completion_schedule,
    declared_cells,
    gradient,
    hessian,
    integrate_regret,
    potential,
    simulate_event_baseline,
    simulate_pub,
    validate_design,
)


def test_frozen_design_cardinality() -> None:
    result = validate_design()
    assert result["cells"] == 36
    assert result["pilot_seeds"] == 48
    assert result["policies"] == 43
    assert result["trajectories"] == 74_304


def test_all_declared_hessians_are_positive_definite() -> None:
    for cell in declared_cells():
        assert np.linalg.eigvalsh(hessian(cell)).min() > 0


def test_gradient_matches_finite_difference() -> None:
    cell = declared_cells()[-1]
    matrix = hessian(cell)
    theta = np.asarray(cell.initial, dtype=float)
    analytic = gradient(theta, matrix)
    eps = 1e-6
    for i in range(cell.n_agents):
        plus = theta.copy()
        minus = theta.copy()
        plus[i] += eps
        minus[i] -= eps
        numeric = (potential(plus, matrix)-potential(minus, matrix))/(2*eps)
        assert numeric == pytest.approx(analytic[i], abs=1e-8)


def test_completion_schedule_is_deterministic_and_charged() -> None:
    cell = declared_cells()[7]
    first = completion_schedule(cell, 910_001)
    second = completion_schedule(cell, 910_001)
    assert first == second
    assert first
    assert all(0 < item.time <= HORIZON for item in first)


def test_regret_integral_uses_wall_clock() -> None:
    trajectory = [(0, -2.0), (3, -1.0), (5, 0.0)]
    assert integrate_regret(trajectory, horizon=7) == pytest.approx(8.0)


def test_pub_has_no_harmful_exact_quadratic_steps_on_smoke_cell() -> None:
    cell = next(c for c in declared_cells() if c.phase == "high" and c.n_agents == 3)
    result = simulate_pub(cell, completion_schedule(cell, 910_001))
    assert result["harmful"] == 0
    assert np.isfinite(result["normalized_regret"])
    assert 0 <= result["acceptance_rate"] <= 1


def test_accept_all_can_be_harmful_in_high_load_smoke_cell() -> None:
    cell = next(c for c in declared_cells() if c.phase == "high" and c.n_agents == 3)
    result = simulate_event_baseline(
        cell, completion_schedule(cell, 910_001), eta=1.0
    )
    assert result["harmful"] >= 0
    assert np.isfinite(result["normalized_regret"])


def test_baseline_names_are_unique() -> None:
    names = [name for name, _ in baseline_specs()]
    assert len(names) == len(set(names)) == 42


def test_preregistered_manifest_matches_executable_design() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root/"docs"/"perishable_update_phase_pilot_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    result = validate_design()
    assert manifest["population"]["cells"] == result["cells"]
    assert manifest["population"]["horizon"] == result["horizon"]
    assert manifest["population"]["potential_weight"] == result["potential_weight"]
    assert tuple(manifest["seeds"]["values"]) == PILOT_SEEDS
    assert manifest["policies"]["comparator_count"] == len(baseline_specs())
    assert manifest["cardinality"]["trajectories"] == result["trajectories"]

from __future__ import annotations

import itertools

import numpy as np
import pytest

from .parallel_commit_qp import (
    certified_joint_gain,
    choose_lyapunov_parallel_commit,
    solve_rank_one_box_qp,
    stale_directional_lower_bounds,
    update_commit_queues,
)


def _objective(
    x: np.ndarray,
    linear: np.ndarray,
    diagonal: np.ndarray,
    weights: np.ndarray,
    strength: float,
) -> float:
    return float(
        linear @ x
        - 0.5 * diagonal @ (x * x)
        - 0.5 * strength * (weights @ x) ** 2
    )


def test_scalar_root_matches_dense_grid_global_optimum() -> None:
    linear = np.asarray([1.3, 0.9])
    diagonal = np.asarray([0.7, 1.1])
    weights = np.asarray([1.0, 0.6])
    strength = 0.8
    decision = solve_rank_one_box_qp(
        linear=linear,
        curvature_diagonal=diagonal,
        interaction_weights=weights,
        interaction_strength=strength,
    )
    grid = np.linspace(0.0, 1.0, 1001)
    brute = max(
        _objective(np.asarray(point), linear, diagonal, weights, strength)
        for point in itertools.product(grid, repeat=2)
    )
    assert decision.queue_weighted_objective >= brute - 2e-6
    assert np.all((decision.scales >= 0.0) & (decision.scales <= 1.0))


def test_zero_interaction_reduces_to_clipped_coordinate_solution() -> None:
    decision = solve_rank_one_box_qp(
        linear=np.asarray([2.0, -1.0, 0.5]),
        curvature_diagonal=np.asarray([2.0, 1.0, 0.25]),
        interaction_weights=np.asarray([1.0, 1.0, 1.0]),
        interaction_strength=0.0,
        maximum_scales=np.asarray([0.8, 1.0, 0.6]),
    )
    np.testing.assert_allclose(decision.scales, [0.8, 0.0, 0.6])
    assert decision.active_count == 2
    assert decision.iterations == 0


def test_risk_debt_can_reject_only_expensive_agents() -> None:
    decision = choose_lyapunov_parallel_commit(
        gain_lower_bounds=np.asarray([1.0, 1.0, 1.0]),
        service_debts=np.zeros(3),
        risk_debt=2.0,
        risk_costs=np.asarray([0.1, 0.7, 0.1]),
        curvature_diagonal=np.ones(3),
        interaction_weights=np.asarray([0.3, 0.3, 0.3]),
        interaction_strength=0.2,
        tradeoff=1.0,
    )
    assert decision.scales[1] == 0.0
    assert decision.scales[0] > 0.0
    assert decision.scales[2] > 0.0


def test_service_debt_prevents_permanent_zero_scale() -> None:
    low = choose_lyapunov_parallel_commit(
        gain_lower_bounds=np.asarray([-0.1, 0.5]),
        service_debts=np.asarray([0.0, 0.0]),
        risk_debt=0.0,
        risk_costs=np.zeros(2),
        curvature_diagonal=np.ones(2),
        interaction_weights=np.ones(2),
        interaction_strength=0.1,
        tradeoff=1.0,
    )
    high = choose_lyapunov_parallel_commit(
        gain_lower_bounds=np.asarray([-0.1, 0.5]),
        service_debts=np.asarray([1.0, 0.0]),
        risk_debt=0.0,
        risk_costs=np.zeros(2),
        curvature_diagonal=np.ones(2),
        interaction_weights=np.ones(2),
        interaction_strength=0.1,
        tradeoff=1.0,
    )
    assert low.scales[0] == 0.0
    assert high.scales[0] > 0.0


def test_projected_queue_updates_are_pathwise() -> None:
    service, risk = update_commit_queues(
        service_debts=np.asarray([0.2, 1.1]),
        arrivals=np.asarray([1.0, 0.0]),
        scales=np.asarray([0.5, 1.0]),
        risk_debt=0.4,
        incurred_risk=0.3,
        risk_budget=0.5,
    )
    np.testing.assert_allclose(service, [0.7, 0.1])
    assert risk == pytest.approx(0.2)


def test_joint_gain_is_lower_bound_for_signed_rank_one_quadratics() -> None:
    rng = np.random.default_rng(20260903)
    for _ in range(200):
        agents = 6
        diagonal = rng.uniform(0.2, 1.5, size=agents)
        factor = rng.normal(size=agents)
        strength = float(rng.uniform(0.0, 1.2))
        hessian = np.diag(diagonal) + strength * np.outer(factor, factor)
        theta = rng.normal(size=agents)
        optimum = rng.normal(size=agents)
        gradient = hessian @ (optimum - theta)
        directions = rng.normal(size=agents)
        scales = rng.uniform(0.0, 1.0, size=agents)
        update = directions * scales
        exact = float(gradient @ update - 0.5 * update @ hessian @ update)
        bound = certified_joint_gain(
            scales=scales,
            gain_lower_bounds=gradient * directions,
            curvature_diagonal=diagonal * directions * directions,
            interaction_weights=np.abs(factor * directions),
            interaction_strength=strength,
        )
        assert exact >= bound - 1e-12


def test_stale_directional_bound_holds_for_exact_quadratic_path() -> None:
    rng = np.random.default_rng(1701)
    for _ in range(100):
        agents = 5
        matrix = rng.normal(size=(agents, agents))
        hessian = matrix.T @ matrix + 0.1 * np.eye(agents)
        birth = rng.normal(size=agents)
        displacement = rng.normal(scale=0.2, size=agents)
        current = birth + displacement
        optimum = rng.normal(size=agents)
        directions = rng.normal(size=agents)
        birth_gradient = hessian @ (optimum - birth)
        current_gradient = hessian @ (optimum - current)
        lower = stale_directional_lower_bounds(
            birth_directional_gains=birth_gradient * directions,
            proposal_directions=directions,
            interaction_absolute=np.abs(hessian),
            policy_displacement=displacement,
        )
        assert np.all(current_gradient * directions >= lower - 1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"curvature_diagonal": np.asarray([0.0, 1.0])},
        {"interaction_weights": np.asarray([-1.0, 1.0])},
        {"interaction_strength": -1.0},
        {"maximum_scales": np.asarray([1.1, 1.0])},
    ],
)
def test_invalid_qp_inputs_are_rejected(kwargs: dict) -> None:
    values = {
        "linear": np.asarray([1.0, 1.0]),
        "curvature_diagonal": np.ones(2),
        "interaction_weights": np.ones(2),
        "interaction_strength": 0.1,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        solve_rank_one_box_qp(**values)

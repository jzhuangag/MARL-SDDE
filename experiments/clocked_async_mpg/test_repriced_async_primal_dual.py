from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .repriced_async_primal_dual import (
    GradientPacket,
    birth_price_direction,
    cost,
    dual_price_delay_error,
    make_problem,
    reprice_packet,
    run,
    run_async,
    solve_constrained_optimum,
    validate_config,
)


CONFIG = Path(__file__).with_name("repriced_async_primal_dual_config.json")


def test_dual_repricing_identity_is_exact() -> None:
    rng = np.random.default_rng(19)
    for _ in range(200):
        packet = GradientPacket(
            owner=0,
            completion_time=1,
            reward_gradient=float(rng.normal()),
            cost_gradient=float(rng.normal()),
            birth_price=float(rng.uniform(0.0, 3.0)),
        )
        current = float(rng.uniform(0.0, 3.0))
        observed = reprice_packet(packet, current) - birth_price_direction(packet)
        assert abs(observed - dual_price_delay_error(packet, current)) <= 1e-14


def test_constrained_optimum_satisfies_kkt() -> None:
    problem = make_problem(
        seed=23,
        agents=4,
        interaction_strength=0.5,
        anisotropy=4.0,
        budget_fraction=0.35,
        service_profile="two_tier",
    )
    theta, price = solve_constrained_optimum(
        reward_hessian=problem.reward_hessian,
        cost_hessian=problem.cost_hessian,
        reward_linear=problem.reward_linear,
        cost_budget=problem.cost_budget,
    )
    stationarity = (
        problem.reward_linear
        - problem.reward_hessian @ theta
        - price * problem.cost_hessian @ theta
    )
    assert price > 0.0
    assert np.linalg.norm(stationarity) <= 1e-10
    assert abs(cost(theta, problem) - problem.cost_budget) <= 1e-10


def test_inactive_constraint_makes_pricing_paths_identical() -> None:
    problem = make_problem(
        seed=29,
        agents=4,
        interaction_strength=1.5,
        anisotropy=1.0,
        budget_fraction=1.2,
        service_profile="skewed",
    )
    kwargs = dict(
        horizon=40,
        primal_step=0.18,
        step_cap=0.3,
        lyapunov_tradeoff=12.0,
    )
    birth = run_async(problem, pricing="birth", **kwargs)
    arrival = run_async(problem, pricing="arrival", **kwargs)
    assert birth == arrival


def test_grid_is_complete_and_small_slice_is_reproducible() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert validate_config(config)["scenario_count"] == 432
    small = dict(config)
    small["development_seeds"] = [31013]
    small["agent_counts"] = [4]
    small["cost_interaction_strengths"] = [0.5]
    small["anisotropies"] = [1.0]
    small["budget_fractions"] = [0.35, 1.2]
    small["service_profiles"] = ["two_tier"]
    assert run(small) == run(small)

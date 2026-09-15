"""Exact CPU model for arrival-repriced asynchronous primal--dual learning.

The model is a local constrained potential-game abstraction.  It is used only
to test whether dual-price staleness has enough intrinsic dynamic value to
justify a full Markov-game theorem and sampled implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConstrainedQuadratic:
    scenario_id: str
    agents: int
    reward_hessian: np.ndarray
    cost_hessian: np.ndarray
    reward_linear: np.ndarray
    cost_budget: float
    initial: np.ndarray
    services: np.ndarray
    optimum: np.ndarray
    optimum_reward: float


@dataclass(frozen=True)
class GradientPacket:
    owner: int
    completion_time: int
    reward_gradient: float
    cost_gradient: float
    birth_price: float


def reward(theta: np.ndarray, problem: ConstrainedQuadratic) -> float:
    theta = np.asarray(theta, dtype=float)
    return float(
        problem.reward_linear @ theta
        - 0.5 * theta @ problem.reward_hessian @ theta
    )


def cost(theta: np.ndarray, problem: ConstrainedQuadratic) -> float:
    theta = np.asarray(theta, dtype=float)
    return 0.5 * float(theta @ problem.cost_hessian @ theta)


def component_gradients(
    theta: np.ndarray, problem: ConstrainedQuadratic
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    return (
        problem.reward_linear - problem.reward_hessian @ theta,
        problem.cost_hessian @ theta,
    )


def solve_constrained_optimum(
    *,
    reward_hessian: np.ndarray,
    cost_hessian: np.ndarray,
    reward_linear: np.ndarray,
    cost_budget: float,
    tolerance: float = 1e-13,
) -> tuple[np.ndarray, float]:
    """Solve the convex quadratic constraint problem through its dual root."""

    def policy(price: float) -> np.ndarray:
        return np.linalg.solve(
            reward_hessian + price * cost_hessian,
            reward_linear,
        )

    def constraint_value(price: float) -> float:
        theta = policy(price)
        return 0.5 * float(theta @ cost_hessian @ theta)

    if constraint_value(0.0) <= cost_budget:
        return policy(0.0), 0.0
    lower = 0.0
    upper = 1.0
    while constraint_value(upper) > cost_budget:
        upper *= 2.0
        if upper > 1e12:
            raise RuntimeError("failed to bracket constrained optimum")
    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        if constraint_value(midpoint) > cost_budget:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= tolerance * max(1.0, upper):
            break
    price = 0.5 * (lower + upper)
    return policy(price), price


def make_problem(
    *,
    seed: int,
    agents: int,
    interaction_strength: float,
    anisotropy: float,
    budget_fraction: float,
    service_profile: str,
) -> ConstrainedQuadratic:
    rng = np.random.default_rng(seed)
    reward_diagonal = np.geomspace(1.0, anisotropy, agents)
    reward_diagonal *= rng.uniform(0.9, 1.1, size=agents)
    reward_factor = rng.uniform(0.2, 0.8, size=agents)
    reward_hessian = np.diag(reward_diagonal) + 0.15 * np.outer(
        reward_factor, reward_factor
    )
    cost_diagonal = rng.uniform(0.4, 1.2, size=agents)
    cost_factor = rng.uniform(0.2, 1.0, size=agents)
    cost_hessian = np.diag(cost_diagonal) + interaction_strength * np.outer(
        cost_factor, cost_factor
    )
    desired_unconstrained = rng.uniform(0.5, 1.4, size=agents)
    reward_linear = reward_hessian @ desired_unconstrained
    unconstrained_cost = 0.5 * float(
        desired_unconstrained @ cost_hessian @ desired_unconstrained
    )
    cost_budget = budget_fraction * unconstrained_cost
    optimum, _ = solve_constrained_optimum(
        reward_hessian=reward_hessian,
        cost_hessian=cost_hessian,
        reward_linear=reward_linear,
        cost_budget=cost_budget,
    )
    if service_profile == "balanced":
        services = np.ones(agents, dtype=int)
    elif service_profile == "two_tier":
        services = np.resize(np.asarray([1, 2, 4, 8]), agents)
    elif service_profile == "skewed":
        services = 2 * np.arange(1, agents + 1, dtype=int)
    else:
        raise ValueError("unknown service profile")
    scenario_id = (
        f"n{agents}-c{interaction_strength:g}-a{anisotropy:g}-"
        f"b{budget_fraction:g}-{service_profile}-s{seed}"
    )
    initial = np.zeros(agents)
    return ConstrainedQuadratic(
        scenario_id=scenario_id,
        agents=agents,
        reward_hessian=reward_hessian,
        cost_hessian=cost_hessian,
        reward_linear=reward_linear,
        cost_budget=cost_budget,
        initial=initial,
        services=services,
        optimum=optimum,
        optimum_reward=float(
            reward_linear @ optimum
            - 0.5 * optimum @ reward_hessian @ optimum
        ),
    )


def reprice_packet(packet: GradientPacket, current_price: float) -> float:
    return packet.reward_gradient - current_price * packet.cost_gradient


def birth_price_direction(packet: GradientPacket) -> float:
    return packet.reward_gradient - packet.birth_price * packet.cost_gradient


def dual_price_delay_error(
    packet: GradientPacket, current_price: float
) -> float:
    """Arrival direction minus birth-price direction, exactly."""

    return -(current_price - packet.birth_price) * packet.cost_gradient


def _launch(
    problem: ConstrainedQuadratic,
    theta: np.ndarray,
    owner: int,
    time: int,
    price: float,
) -> GradientPacket:
    reward_gradient, cost_gradient = component_gradients(theta, problem)
    return GradientPacket(
        owner=owner,
        completion_time=time + int(problem.services[owner]),
        reward_gradient=float(reward_gradient[owner]),
        cost_gradient=float(cost_gradient[owner]),
        birth_price=float(price),
    )


def _risk(theta: np.ndarray, problem: ConstrainedQuadratic) -> float:
    error = theta - problem.optimum
    scale = max(problem.cost_budget, 1e-12)
    violation = max(0.0, cost(theta, problem) - problem.cost_budget) / scale
    return float(error @ problem.reward_hessian @ error + violation * violation)


def run_async(
    problem: ConstrainedQuadratic,
    *,
    pricing: str,
    horizon: int,
    primal_step: float,
    step_cap: float,
    lyapunov_tradeoff: float,
) -> dict[str, Any]:
    if pricing not in {"birth", "arrival"}:
        raise ValueError("pricing must be birth or arrival")
    theta = problem.initial.copy()
    queue = 0.0
    price = 0.0
    packets = [
        _launch(problem, theta, owner, 0, price)
        for owner in range(problem.agents)
    ]
    rewards = [reward(theta, problem)]
    costs = [cost(theta, problem)]
    risks = [_risk(theta, problem)]
    queues = [queue]
    completed = 0
    for time in range(1, horizon + 1):
        ready = [packet for packet in packets if packet.completion_time == time]
        delta = np.zeros(problem.agents)
        for packet in ready:
            direction = (
                reprice_packet(packet, price)
                if pricing == "arrival"
                else birth_price_direction(packet)
            )
            curvature = (
                problem.reward_hessian[packet.owner, packet.owner]
                + price * problem.cost_hessian[packet.owner, packet.owner]
            )
            delta[packet.owner] = np.clip(
                primal_step * direction / curvature,
                -step_cap,
                step_cap,
            )
        theta = theta + delta
        completed += len(ready)
        if ready:
            ready_owners = {packet.owner for packet in ready}
            packets = [
                packet for packet in packets if packet.owner not in ready_owners
            ]
        else:
            ready_owners = set()

        current_cost = cost(theta, problem)
        queue = max(0.0, queue + current_cost - problem.cost_budget)
        price = queue / lyapunov_tradeoff
        packets.extend(
            _launch(problem, theta, owner, time, price)
            for owner in sorted(ready_owners)
        )
        rewards.append(reward(theta, problem))
        costs.append(current_cost)
        risks.append(_risk(theta, problem))
        queues.append(queue)
    normalized_violation = [
        max(0.0, value - problem.cost_budget)
        / max(problem.cost_budget, 1e-12)
        for value in costs
    ]
    return {
        "risk_auc": float(np.mean(risks)),
        "terminal_risk": float(risks[-1]),
        "violation_auc": float(np.mean(normalized_violation)),
        "terminal_violation": float(normalized_violation[-1]),
        "reward_auc": float(np.mean(rewards)),
        "terminal_reward": float(rewards[-1]),
        "completed_proposals": completed,
        "mean_queue": float(np.mean(queues)),
        "terminal_queue": float(queues[-1]),
        "risk_curve": risks,
        "violation_curve": normalized_violation,
    }


def run_barrier(
    problem: ConstrainedQuadratic,
    *,
    horizon: int,
    primal_step: float,
    step_cap: float,
    lyapunov_tradeoff: float,
) -> dict[str, Any]:
    theta = problem.initial.copy()
    queue = 0.0
    price = 0.0
    completion_time = int(np.max(problem.services))
    packets = [
        _launch(problem, theta, owner, 0, price)
        for owner in range(problem.agents)
    ]
    rewards = [reward(theta, problem)]
    costs = [cost(theta, problem)]
    risks = [_risk(theta, problem)]
    queues = [queue]
    completed = 0
    for time in range(1, horizon + 1):
        if time == completion_time:
            delta = np.zeros(problem.agents)
            for packet in packets:
                direction = reprice_packet(packet, price)
                curvature = (
                    problem.reward_hessian[packet.owner, packet.owner]
                    + price * problem.cost_hessian[packet.owner, packet.owner]
                )
                delta[packet.owner] = np.clip(
                    primal_step * direction / curvature,
                    -step_cap,
                    step_cap,
                )
            theta = theta + delta
            completed += problem.agents
            next_completion = time + int(np.max(problem.services))
        else:
            next_completion = completion_time

        current_cost = cost(theta, problem)
        queue = max(0.0, queue + current_cost - problem.cost_budget)
        price = queue / lyapunov_tradeoff
        if time == completion_time:
            packets = [
                _launch(problem, theta, owner, time, price)
                for owner in range(problem.agents)
            ]
            completion_time = next_completion
        rewards.append(reward(theta, problem))
        costs.append(current_cost)
        risks.append(_risk(theta, problem))
        queues.append(queue)
    normalized_violation = [
        max(0.0, value - problem.cost_budget)
        / max(problem.cost_budget, 1e-12)
        for value in costs
    ]
    return {
        "risk_auc": float(np.mean(risks)),
        "terminal_risk": float(risks[-1]),
        "violation_auc": float(np.mean(normalized_violation)),
        "terminal_violation": float(normalized_violation[-1]),
        "reward_auc": float(np.mean(rewards)),
        "terminal_reward": float(rewards[-1]),
        "completed_proposals": completed,
        "mean_queue": float(np.mean(queues)),
        "terminal_queue": float(queues[-1]),
        "risk_curve": risks,
        "violation_curve": normalized_violation,
    }


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive inputs")
    return float(math.exp(np.mean(np.log(values))))


def evaluate(problem: ConstrainedQuadratic, config: dict[str, Any]) -> dict[str, Any]:
    kwargs = {
        "horizon": int(config["horizon"]),
        "primal_step": float(config["primal_step"]),
        "step_cap": float(config["step_cap"]),
        "lyapunov_tradeoff": float(config["lyapunov_tradeoff"]),
    }
    birth = run_async(problem, pricing="birth", **kwargs)
    arrival = run_async(problem, pricing="arrival", **kwargs)
    barrier = run_barrier(problem, **kwargs)
    active = bool(config["inactive_budget_fraction"] > 0) and (
        problem.cost_budget
        < cost(np.linalg.solve(problem.reward_hessian, problem.reward_linear), problem)
    )
    return {
        "scenario_id": problem.scenario_id,
        "agents": problem.agents,
        "services": problem.services.tolist(),
        "cost_budget": problem.cost_budget,
        "constraint_active": active,
        "birth": birth,
        "arrival": arrival,
        "barrier": barrier,
        "arrival_to_birth_risk": arrival["risk_auc"] / birth["risk_auc"],
        "arrival_to_birth_violation": (
            arrival["violation_auc"] / birth["violation_auc"]
            if birth["violation_auc"] > 0.0
            else 1.0
        ),
        "arrival_to_barrier_risk": arrival["risk_auc"] / barrier["risk_auc"],
        "async_to_barrier_proposals": (
            arrival["completed_proposals"] / barrier["completed_proposals"]
            if barrier["completed_proposals"] > 0
            else math.inf
        ),
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    problems = [
        make_problem(
            seed=int(seed),
            agents=int(agents),
            interaction_strength=float(interaction),
            anisotropy=float(anisotropy),
            budget_fraction=float(budget_fraction),
            service_profile=str(profile),
        )
        for seed in config["development_seeds"]
        for agents in config["agent_counts"]
        for interaction in config["cost_interaction_strengths"]
        for anisotropy in config["anisotropies"]
        for budget_fraction in config["budget_fractions"]
        for profile in config["service_profiles"]
    ]
    rows = [evaluate(problem, config) for problem in problems]
    active_heterogeneous = [
        row
        for row in rows
        if row["constraint_active"] and max(row["services"]) > 1
    ]
    inactive = [row for row in rows if not row["constraint_active"]]
    risk_ratios = [row["arrival_to_birth_risk"] for row in active_heterogeneous]
    violation_ratios = [
        row["arrival_to_birth_violation"] for row in active_heterogeneous
    ]
    barrier_ratios = [
        row["arrival_to_barrier_risk"] for row in active_heterogeneous
    ]
    throughput = [
        row["async_to_barrier_proposals"] for row in active_heterogeneous
    ]
    inactive_gap = max(
        (
            abs(row["arrival"]["risk_auc"] - row["birth"]["risk_auc"])
            for row in inactive
        ),
        default=0.0,
    )
    summary = {
        "scenario_count": len(rows),
        "active_heterogeneous_count": len(active_heterogeneous),
        "inactive_count": len(inactive),
        "arrival_birth_geometric_risk_ratio": geometric_mean(risk_ratios),
        "arrival_birth_geometric_violation_ratio": geometric_mean(
            [max(value, 1e-15) for value in violation_ratios]
        ),
        "arrival_risk_gain_fraction": float(
            np.mean([value <= 0.90 for value in risk_ratios])
        ),
        "arrival_violation_gain_fraction": float(
            np.mean([value <= 0.75 for value in violation_ratios])
        ),
        "arrival_barrier_geometric_risk_ratio": geometric_mean(barrier_ratios),
        "median_async_barrier_proposal_ratio": float(np.median(throughput)),
        "inactive_max_risk_auc_difference": inactive_gap,
    }
    gates = {
        "arrival_risk_improves_birth_by_ten_percent": summary[
            "arrival_birth_geometric_risk_ratio"
        ]
        <= 0.90,
        "arrival_violation_improves_birth_by_twenty_five_percent": summary[
            "arrival_birth_geometric_violation_ratio"
        ]
        <= 0.75,
        "broad_risk_improvement": summary["arrival_risk_gain_fraction"] >= 0.60,
        "broad_violation_improvement": summary[
            "arrival_violation_gain_fraction"
        ]
        >= 0.60,
        "within_ten_percent_of_barrier_risk": summary[
            "arrival_barrier_geometric_risk_ratio"
        ]
        <= 1.10,
        "twenty_percent_more_wall_clock_proposals": summary[
            "median_async_barrier_proposal_ratio"
        ]
        >= 1.20,
        "inactive_constraint_exact_control": inactive_gap <= 1e-14,
    }
    return {
        "artifact_id": config["artifact_id"],
        "scope": "exact deterministic development; not sampled evidence",
        "config": config,
        "summary": summary,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "rows": rows,
    }


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    scenario_count = math.prod(
        len(config[key])
        for key in (
            "development_seeds",
            "agent_counts",
            "cost_interaction_strengths",
            "anisotropies",
            "budget_fractions",
            "service_profiles",
        )
    )
    if scenario_count <= 0:
        raise ValueError("empty scenario grid")
    inactive_fraction = float(config["inactive_budget_fraction"])
    if inactive_fraction not in [float(value) for value in config["budget_fractions"]]:
        raise ValueError("inactive control fraction must be in budget grid")
    return {"scenario_count": scenario_count}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.mode == "validate":
        print(json.dumps(validate_config(config), indent=2, sort_keys=True))
        return
    if args.output is None:
        raise ValueError("run requires --output")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"output_sha256={_sha256(args.output)}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"all_gates_passed={result['all_gates_passed']}")


if __name__ == "__main__":
    main()

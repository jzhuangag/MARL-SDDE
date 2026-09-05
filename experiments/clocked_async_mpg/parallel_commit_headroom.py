"""Deterministic CPU headroom screen for Lyapunov parallel commits.

This is a development calculation, not sampled evidence.  It uses exact
concave cooperative quadratic potentials so that proposal birth gradients,
arrival-time gradients, staleness bounds, and potential gaps are all auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .parallel_commit_qp import (
    choose_lyapunov_parallel_commit,
    solve_rank_one_box_qp,
    update_commit_queues,
)


METHODS = (
    "bound_oracle",
    "causal_lyapunov",
    "best_fixed_async",
    "best_sequential",
)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    agents: int
    interaction_strength: float
    anisotropy: float
    service_profile: str
    base_diagonal: np.ndarray
    interaction_factor: np.ndarray
    optimum: np.ndarray
    initial: np.ndarray
    services: np.ndarray

    @property
    def hessian(self) -> np.ndarray:
        return np.diag(self.base_diagonal) + self.interaction_strength * np.outer(
            self.interaction_factor, self.interaction_factor
        )


@dataclass
class Proposal:
    owner: int
    birth_time: int
    completion_time: int
    birth_theta: np.ndarray
    birth_directional_gain: float
    direction: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def potential_gap(theta: np.ndarray, scenario: Scenario) -> float:
    error = np.asarray(theta, dtype=float) - scenario.optimum
    return 0.5 * float(error @ scenario.hessian @ error)


def gradient(theta: np.ndarray, scenario: Scenario) -> np.ndarray:
    return scenario.hessian @ (scenario.optimum - np.asarray(theta, dtype=float))


def make_scenario(
    *,
    seed: int,
    agents: int,
    interaction_strength: float,
    anisotropy: float,
    service_profile: str,
) -> Scenario:
    rng = np.random.default_rng(seed)
    base_diagonal = np.geomspace(1.0, anisotropy, agents)
    base_diagonal *= rng.uniform(0.85, 1.15, size=agents)
    interaction_factor = rng.uniform(0.2, 1.0, size=agents)
    interaction_factor *= rng.choice([-1.0, 1.0], size=agents)
    optimum = rng.normal(0.0, 0.6, size=agents)
    initial = optimum + rng.normal(0.0, 1.0, size=agents)
    if service_profile == "balanced":
        services = np.ones(agents, dtype=int)
    elif service_profile == "two_tier":
        template = np.asarray([1, 1, 2, 4], dtype=int)
        services = np.resize(template, agents)
    elif service_profile == "skewed":
        services = np.arange(1, agents + 1, dtype=int)
    else:
        raise ValueError("unknown service profile")
    identifier = (
        f"n{agents}-rho{interaction_strength:g}-a{anisotropy:g}-"
        f"{service_profile}-s{seed}"
    )
    return Scenario(
        scenario_id=identifier,
        seed=seed,
        agents=agents,
        interaction_strength=interaction_strength,
        anisotropy=anisotropy,
        service_profile=service_profile,
        base_diagonal=base_diagonal,
        interaction_factor=interaction_factor,
        optimum=optimum,
        initial=initial,
        services=services,
    )


def _launch(
    scenario: Scenario,
    theta: np.ndarray,
    owner: int,
    birth_time: int,
    step_cap: float,
) -> Proposal:
    current_gradient = gradient(theta, scenario)
    direction = float(
        np.clip(
            current_gradient[owner] / scenario.hessian[owner, owner],
            -step_cap,
            step_cap,
        )
    )
    return Proposal(
        owner=owner,
        birth_time=birth_time,
        completion_time=birth_time + int(scenario.services[owner]),
        birth_theta=theta.copy(),
        birth_directional_gain=float(current_gradient[owner] * direction),
        direction=direction,
    )


def _run_async(
    scenario: Scenario,
    *,
    method: str,
    horizon: int,
    step_cap: float,
    fixed_scale: float | None = None,
    tradeoff: float = 4.0,
    risk_budget: float = 0.02,
) -> dict[str, Any]:
    if method not in {"bound_oracle", "causal_lyapunov", "fixed_async"}:
        raise ValueError("unknown asynchronous method")
    if method == "fixed_async" and fixed_scale is None:
        raise ValueError("fixed_async requires a scale")
    theta = scenario.initial.copy()
    proposals = [
        _launch(scenario, theta, owner, 0, step_cap)
        for owner in range(scenario.agents)
    ]
    service_debts = np.zeros(scenario.agents)
    risk_debt = 0.0
    gaps = [potential_gap(theta, scenario)]
    completed = 0
    active = 0
    scales_used: list[float] = []
    risk_path: list[float] = [risk_debt]
    debt_path: list[float] = [float(np.sum(service_debts))]
    for time in range(1, horizon + 1):
        ready = [proposal for proposal in proposals if proposal.completion_time == time]
        caps = np.zeros(scenario.agents)
        directions = np.zeros(scenario.agents)
        birth_gains = np.zeros(scenario.agents)
        stale_costs = np.zeros(scenario.agents)
        gains = np.zeros(scenario.agents)
        for proposal in ready:
            owner = proposal.owner
            caps[owner] = 1.0
            directions[owner] = proposal.direction
            birth_gains[owner] = proposal.birth_directional_gain
            path_penalty = abs(proposal.direction) * float(
                np.dot(
                    np.abs(scenario.hessian[owner]),
                    np.abs(theta - proposal.birth_theta),
                )
            )
            lower = proposal.birth_directional_gain - path_penalty
            gains[owner] = lower
            stale_costs[owner] = max(0.0, proposal.birth_directional_gain - lower)

        if ready:
            current_gradient = gradient(theta, scenario)
            if method == "bound_oracle":
                gains = current_gradient * directions
                decision = solve_rank_one_box_qp(
                    linear=gains,
                    curvature_diagonal=(
                        scenario.base_diagonal * directions * directions + 1e-12
                    ),
                    interaction_weights=np.abs(
                        scenario.interaction_factor * directions
                    ),
                    interaction_strength=scenario.interaction_strength,
                    maximum_scales=caps,
                )
                scales = decision.scales
            elif method == "causal_lyapunov":
                decision = choose_lyapunov_parallel_commit(
                    gain_lower_bounds=gains,
                    service_debts=service_debts,
                    risk_debt=risk_debt,
                    risk_costs=stale_costs,
                    curvature_diagonal=(
                        scenario.base_diagonal * directions * directions + 1e-12
                    ),
                    interaction_weights=np.abs(
                        scenario.interaction_factor * directions
                    ),
                    interaction_strength=scenario.interaction_strength,
                    tradeoff=tradeoff,
                    maximum_scales=caps,
                )
                scales = decision.scales
            else:
                scales = caps * float(fixed_scale)

            theta += directions * scales
            completed += len(ready)
            active += int(np.count_nonzero(scales > 1e-12))
            scales_used.extend(float(scales[proposal.owner]) for proposal in ready)
            arrivals = caps
            incurred_risk = float(np.dot(stale_costs, scales))
            service_debts, risk_debt = update_commit_queues(
                service_debts=service_debts,
                arrivals=arrivals,
                scales=scales,
                risk_debt=risk_debt,
                incurred_risk=incurred_risk,
                risk_budget=risk_budget,
            )
            ready_owners = {proposal.owner for proposal in ready}
            proposals = [
                proposal for proposal in proposals if proposal.owner not in ready_owners
            ]
            proposals.extend(
                _launch(scenario, theta, owner, time, step_cap)
                for owner in sorted(ready_owners)
            )
        gaps.append(potential_gap(theta, scenario))
        risk_path.append(risk_debt)
        debt_path.append(float(np.sum(service_debts)))
    return {
        "gap_auc": float(np.mean(gaps)),
        "terminal_gap": float(gaps[-1]),
        "completed_proposals": completed,
        "active_proposals": active,
        "mean_scale": float(np.mean(scales_used)) if scales_used else 0.0,
        "mean_risk_debt": float(np.mean(risk_path)),
        "terminal_risk_debt": float(risk_path[-1]),
        "mean_service_debt": float(np.mean(debt_path)),
        "terminal_service_debt": float(debt_path[-1]),
        "gap_curve": gaps,
    }


def _candidate_orders(agents: int) -> list[tuple[int, ...]]:
    if agents <= 6:
        return list(itertools.permutations(range(agents)))
    base = tuple(range(agents))
    reversed_order = tuple(reversed(base))
    orders = [base, reversed_order]
    orders.extend(tuple(np.roll(np.asarray(base), shift)) for shift in range(1, agents))
    return list(dict.fromkeys(orders))


def _run_sequential(
    scenario: Scenario,
    *,
    order: tuple[int, ...],
    horizon: int,
    step_cap: float,
) -> dict[str, Any]:
    theta = scenario.initial.copy()
    gaps = [potential_gap(theta, scenario)]
    time = 0
    cursor = 0
    updates = 0
    while time < horizon:
        owner = order[cursor % len(order)]
        completion = time + int(scenario.services[owner])
        if completion > horizon:
            gaps.extend([gaps[-1]] * (horizon - time))
            break
        gaps.extend([gaps[-1]] * (completion - time - 1))
        current_gradient = gradient(theta, scenario)
        direction = float(
            np.clip(
                current_gradient[owner] / scenario.hessian[owner, owner],
                -step_cap,
                step_cap,
            )
        )
        theta[owner] += direction
        gaps.append(potential_gap(theta, scenario))
        time = completion
        cursor += 1
        updates += 1
    if len(gaps) < horizon + 1:
        gaps.extend([gaps[-1]] * (horizon + 1 - len(gaps)))
    return {
        "gap_auc": float(np.mean(gaps)),
        "terminal_gap": float(gaps[-1]),
        "completed_proposals": updates,
        "active_proposals": updates,
        "mean_scale": 1.0,
        "gap_curve": gaps,
    }


def evaluate_scenario(
    scenario: Scenario,
    *,
    horizon: int,
    step_cap: float,
    fixed_scale_grid: list[float],
    tradeoff: float,
    risk_budget: float,
) -> dict[str, Any]:
    oracle = _run_async(
        scenario,
        method="bound_oracle",
        horizon=horizon,
        step_cap=step_cap,
    )
    causal = _run_async(
        scenario,
        method="causal_lyapunov",
        horizon=horizon,
        step_cap=step_cap,
        tradeoff=tradeoff,
        risk_budget=risk_budget,
    )
    fixed_candidates = [
        (
            scale,
            _run_async(
                scenario,
                method="fixed_async",
                horizon=horizon,
                step_cap=step_cap,
                fixed_scale=scale,
            ),
        )
        for scale in fixed_scale_grid
    ]
    fixed_scale, fixed = min(fixed_candidates, key=lambda item: item[1]["gap_auc"])
    sequential_candidates = [
        (
            order,
            _run_sequential(
                scenario,
                order=order,
                horizon=horizon,
                step_cap=step_cap,
            ),
        )
        for order in _candidate_orders(scenario.agents)
    ]
    sequential_order, sequential = min(
        sequential_candidates, key=lambda item: item[1]["gap_auc"]
    )
    strong_cost = min(fixed["gap_auc"], sequential["gap_auc"])
    return {
        "scenario_id": scenario.scenario_id,
        "agents": scenario.agents,
        "interaction_strength": scenario.interaction_strength,
        "anisotropy": scenario.anisotropy,
        "service_profile": scenario.service_profile,
        "bound_oracle": oracle,
        "causal_lyapunov": causal,
        "best_fixed_async": {**fixed, "selected_scale": fixed_scale},
        "best_sequential": {
            **sequential,
            "selected_order": list(sequential_order),
        },
        "strong_static_gap_auc": strong_cost,
        "oracle_to_strong_ratio": oracle["gap_auc"] / strong_cost,
        "causal_to_strong_ratio": causal["gap_auc"] / strong_cost,
        "causal_headroom_retention": (
            (strong_cost - causal["gap_auc"])
            / max(strong_cost - oracle["gap_auc"], 1e-15)
            if oracle["gap_auc"] < strong_cost
            else 0.0
        ),
    }


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def run(config: dict[str, Any]) -> dict[str, Any]:
    scenarios = [
        make_scenario(
            seed=int(seed),
            agents=int(agents),
            interaction_strength=float(strength),
            anisotropy=float(anisotropy),
            service_profile=profile,
        )
        for seed in config["development_seeds"]
        for agents in config["agent_counts"]
        for strength in config["interaction_strengths"]
        for anisotropy in config["anisotropies"]
        for profile in config["service_profiles"]
    ]
    rows = [
        evaluate_scenario(
            scenario,
            horizon=int(config["horizon"]),
            step_cap=float(config["step_cap"]),
            fixed_scale_grid=[float(value) for value in config["fixed_scale_grid"]],
            tradeoff=float(config["tradeoff"]),
            risk_budget=float(config["risk_budget"]),
        )
        for scenario in scenarios
    ]
    oracle_ratios = [row["oracle_to_strong_ratio"] for row in rows]
    causal_ratios = [row["causal_to_strong_ratio"] for row in rows]
    oracle_better = [ratio <= 0.95 for ratio in oracle_ratios]
    causal_better = [ratio <= 0.95 for ratio in causal_ratios]
    eligible = [row for row in rows if row["oracle_to_strong_ratio"] < 1.0]
    retention = [row["causal_headroom_retention"] for row in eligible]
    summary = {
        "scenario_count": len(rows),
        "oracle_geometric_ratio": geometric_mean(oracle_ratios),
        "oracle_five_percent_fraction": sum(oracle_better) / len(rows),
        "causal_geometric_ratio": geometric_mean(causal_ratios),
        "causal_five_percent_fraction": sum(causal_better) / len(rows),
        "causal_median_headroom_retention": float(np.median(retention)) if retention else 0.0,
        "causal_mean_terminal_service_debt": float(
            np.mean([row["causal_lyapunov"]["terminal_service_debt"] for row in rows])
        ),
        "causal_mean_terminal_risk_debt": float(
            np.mean([row["causal_lyapunov"]["terminal_risk_debt"] for row in rows])
        ),
        "causal_mean_scale": float(
            np.mean([row["causal_lyapunov"]["mean_scale"] for row in rows])
        ),
    }
    gates = {
        "oracle_aggregate_gain_at_least_five_percent": summary[
            "oracle_geometric_ratio"
        ]
        <= 0.95,
        "oracle_gain_in_sixty_percent_cells": summary[
            "oracle_five_percent_fraction"
        ]
        >= 0.60,
        "causal_retains_eighty_percent_oracle_headroom": summary[
            "causal_median_headroom_retention"
        ]
        >= 0.80,
        "causal_gain_in_sixty_percent_cells": summary[
            "causal_five_percent_fraction"
        ]
        >= 0.60,
        "service_and_risk_debts_finite_nontrivial": (
            math.isfinite(summary["causal_mean_terminal_service_debt"])
            and math.isfinite(summary["causal_mean_terminal_risk_debt"])
            and summary["causal_mean_terminal_service_debt"] > 0.0
            and summary["causal_mean_terminal_risk_debt"] > 0.0
        ),
    }
    return {
        "artifact_id": config["artifact_id"],
        "scope": "deterministic CPU development headroom; not sampled evidence",
        "config": config,
        "summary": summary,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "rows": rows,
    }


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    scenarios = (
        len(config["development_seeds"])
        * len(config["agent_counts"])
        * len(config["interaction_strengths"])
        * len(config["anisotropies"])
        * len(config["service_profiles"])
    )
    if scenarios <= 0:
        raise ValueError("empty development grid")
    if sorted(config["fixed_scale_grid"]) != config["fixed_scale_grid"]:
        raise ValueError("fixed scale grid must be sorted")
    if not all(0.0 < value <= 1.0 for value in config["fixed_scale_grid"]):
        raise ValueError("fixed scale grid must lie in (0,1]")
    return {"scenario_count": scenarios, "methods": list(METHODS)}


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
        raise ValueError("run mode requires --output")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output_sha256={_sha256(args.output)}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"all_gates_passed={result['all_gates_passed']}")


if __name__ == "__main__":
    main()

"""Frozen exact-moment headroom scan for coupled asynchronous actor--critic.

This is a privileged oracle feasibility calculation, not an executable MARL
algorithm or a sampled efficacy experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .coupled_actor_critic_drift import solve_two_dimensional_box_qp


HORIZON = 64
ALPHA_CAP = 0.75
BETA_CAP = 1.0
CRITIC_CONTRACTION = 0.8
CRITIC_WEIGHT = 1.0
FIXED_GRID_POINTS = 15
FIXED_REFINEMENT_STARTS = 6
PSD_TOLERANCE = 1e-10


@dataclass(frozen=True)
class Scenario:
    population: str
    agents: int
    interaction: float
    critic_bias: float
    initial_critic_error: float
    target_sensitivity: float
    service: str
    noise: str

    @property
    def key(self) -> str:
        return (
            f"{self.population}-n{self.agents}-h{self.interaction:.2f}"
            f"-b{self.critic_bias:.2f}-e{self.initial_critic_error:.2f}"
            f"-k{self.target_sensitivity:.2f}"
            f"-{self.service}-{self.noise}"
        )


@dataclass
class MomentState:
    mean: np.ndarray
    covariance: np.ndarray
    next_times: np.ndarray


def frozen_scenarios() -> list[Scenario]:
    primary = [
        Scenario("primary", *values)
        for values in itertools.product(
            (2, 4),
            (0.15, 0.35),
            (0.3, 0.6),
            (0.35, 0.75),
            (0.4, 0.9),
            ("mild", "severe"),
            ("low", "high"),
        )
    ]
    zero_target = [
        Scenario("zero_target", agents, interaction, bias, 0.75, 0.0, service, "low")
        for agents, interaction, bias, service in itertools.product(
            (2, 4), (0.15, 0.35), (0.3, 0.6), ("mild", "severe")
        )
    ]
    zero_interaction = [
        Scenario("zero_interaction", agents, 0.0, bias, 0.75, target, service, "low")
        for agents, bias, target, service in itertools.product(
            (2, 4), (0.3, 0.6), (0.4, 0.9), ("mild", "severe")
        )
    ]
    result = primary + zero_target + zero_interaction
    if len(primary) != 128 or len(result) != 160:
        raise AssertionError("frozen scenario cardinality changed")
    if len({scenario.key for scenario in result}) != len(result):
        raise AssertionError("scenario keys are not unique")
    return result


def _service_intervals(scenario: Scenario) -> np.ndarray:
    if scenario.agents == 2:
        values = (1.0, 2.0) if scenario.service == "mild" else (1.0, 5.0)
    else:
        values = (
            (1.0, 1.7, 2.3, 3.0)
            if scenario.service == "mild"
            else (1.0, 2.0, 5.0, 8.0)
        )
    return np.asarray(values, dtype=float)


def _game_hessian(scenario: Scenario) -> np.ndarray:
    agents = scenario.agents
    hessian = np.eye(agents, dtype=float) * 1.2
    if agents == 2:
        hessian[0, 1] = hessian[1, 0] = scenario.interaction
    else:
        for index in range(agents):
            neighbor = (index + 1) % agents
            hessian[index, neighbor] = scenario.interaction
            hessian[neighbor, index] = scenario.interaction
    if np.linalg.eigvalsh(hessian)[0] <= 0.0:
        raise AssertionError("frozen Hessian is not positive definite")
    return hessian


def _noise_variances(scenario: Scenario) -> tuple[float, float]:
    return (0.01, 0.008) if scenario.noise == "low" else (0.10, 0.08)


def _target_vector(scenario: Scenario) -> np.ndarray:
    taper = np.linspace(1.0, 0.7, scenario.agents)
    return scenario.target_sensitivity * taper


def _layout(agents: int) -> tuple[int, int, Callable[[int], slice]]:
    critic = agents

    def birth(owner: int) -> slice:
        start = agents + 1 + owner * agents
        return slice(start, start + agents)

    return agents + 1 + agents * agents, critic, birth


def _initial_state(scenario: Scenario) -> MomentState:
    dimension, critic, birth_slice = _layout(scenario.agents)
    actor = np.asarray([(-1.0) ** index * (1.0 - 0.12 * index) for index in range(scenario.agents)])
    mean = np.zeros(dimension, dtype=float)
    mean[: scenario.agents] = actor
    mean[critic] = scenario.initial_critic_error
    for owner in range(scenario.agents):
        mean[birth_slice(owner)] = actor
    return MomentState(
        mean=mean,
        covariance=np.zeros((dimension, dimension), dtype=float),
        next_times=_service_intervals(scenario).copy(),
    )


def _metric(scenario: Scenario) -> np.ndarray:
    dimension, critic, _ = _layout(scenario.agents)
    metric = np.zeros((dimension, dimension), dtype=float)
    metric[: scenario.agents, : scenario.agents] = _game_hessian(scenario)
    metric[critic, critic] = CRITIC_WEIGHT
    return metric


def _event_matrices(
    scenario: Scenario, owner: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    agents = scenario.agents
    dimension, critic, birth_slice = _layout(agents)
    hessian = _game_hessian(scenario)
    gradient = np.zeros(dimension, dtype=float)
    gradient[owner] = hessian[owner, owner]
    gradient[critic] = scenario.critic_bias
    for teammate in range(agents):
        if teammate != owner:
            gradient[birth_slice(owner).start + teammate] = hessian[owner, teammate]

    target = _target_vector(scenario)[owner]
    actor_response = np.zeros(dimension, dtype=float)
    actor_response[owner] = -1.0
    actor_response[critic] = target
    actor_matrix = np.outer(actor_response, gradient)
    critic_matrix = np.zeros((dimension, dimension), dtype=float)
    critic_matrix[critic, critic] = -CRITIC_CONTRACTION

    reset = np.eye(dimension, dtype=float)
    reset[birth_slice(owner), :] = 0.0
    reset[birth_slice(owner), :agents] = np.eye(agents)
    actor_noise = actor_response
    critic_noise = np.zeros(dimension, dtype=float)
    critic_noise[critic] = 1.0
    return actor_matrix, critic_matrix, reset, actor_noise, critic_noise


def _drift_qp(
    scenario: Scenario, state: MomentState, owner: int
) -> tuple[np.ndarray, np.ndarray, float]:
    metric = _metric(scenario)
    actor_matrix, critic_matrix, _, actor_noise, critic_noise = _event_matrices(
        scenario, owner
    )
    second = state.covariance + np.outer(state.mean, state.mean)
    actor_variance, critic_variance = _noise_variances(scenario)
    linear = 2.0 * np.asarray(
        [
            np.trace(actor_matrix.T @ metric @ second),
            np.trace(critic_matrix.T @ metric @ second),
        ]
    )
    quadratic = 2.0 * np.asarray(
        [
            [
                np.trace(actor_matrix.T @ metric @ actor_matrix @ second)
                + actor_variance * float(actor_noise @ metric @ actor_noise),
                np.trace(actor_matrix.T @ metric @ critic_matrix @ second),
            ],
            [
                np.trace(actor_matrix.T @ metric @ critic_matrix @ second),
                np.trace(critic_matrix.T @ metric @ critic_matrix @ second)
                + critic_variance * float(critic_noise @ metric @ critic_noise),
            ],
        ]
    )
    return linear, quadratic, float(quadratic[0, 1])


def _advance(
    scenario: Scenario,
    state: MomentState,
    owner: int,
    alpha: float,
    beta: float,
) -> MomentState:
    actor_matrix, critic_matrix, reset, actor_noise, critic_noise = _event_matrices(
        scenario, owner
    )
    transition = np.eye(state.mean.size) + alpha * actor_matrix + beta * critic_matrix
    actor_variance, critic_variance = _noise_variances(scenario)
    innovation = (
        alpha**2 * actor_variance * np.outer(actor_noise, actor_noise)
        + beta**2 * critic_variance * np.outer(critic_noise, critic_noise)
    )
    next_mean = reset @ transition @ state.mean
    next_covariance = reset @ (
        transition @ state.covariance @ transition.T + innovation
    ) @ reset.T
    next_covariance = 0.5 * (next_covariance + next_covariance.T)
    next_times = state.next_times.copy()
    next_times[owner] += _service_intervals(scenario)[owner]
    return MomentState(next_mean, next_covariance, next_times)


def _risk(scenario: Scenario, state: MomentState) -> float:
    metric = _metric(scenario)
    second = state.covariance + np.outer(state.mean, state.mean)
    return float(np.trace(metric @ second))


def _next_owner(state: MomentState) -> tuple[int, float]:
    owner = int(np.argmin(state.next_times))
    return owner, float(state.next_times[owner])


def simulate(
    scenario: Scenario,
    method: str,
    fixed_action: tuple[float, float] | None = None,
) -> dict[str, object]:
    if method not in {"coupled", "diagonal_online", "fixed"}:
        raise ValueError("unknown method")
    if method == "fixed" and fixed_action is None:
        raise ValueError("fixed method requires an action")
    state = _initial_state(scenario)
    initial_risk = _risk(scenario, state)
    risks: list[float] = []
    alphas: list[float] = []
    betas: list[float] = []
    cross_terms: list[float] = []
    minimum_eigenvalue = math.inf
    interior = 0
    event_time = 0.0

    for _ in range(HORIZON):
        owner, event_time = _next_owner(state)
        linear, quadratic, cross = _drift_qp(scenario, state, owner)
        minimum_eigenvalue = min(minimum_eigenvalue, float(np.linalg.eigvalsh(quadratic)[0]))
        if method == "coupled":
            action = solve_two_dimensional_box_qp(
                linear=linear,
                quadratic=quadratic,
                upper=np.asarray([ALPHA_CAP, BETA_CAP]),
            ).action
        elif method == "diagonal_online":
            action = solve_two_dimensional_box_qp(
                linear=linear,
                quadratic=np.diag(np.diag(quadratic)),
                upper=np.asarray([ALPHA_CAP, BETA_CAP]),
            ).action
        else:
            action = np.asarray(fixed_action, dtype=float)
        alpha, beta = map(float, action)
        if not (0.0 <= alpha <= ALPHA_CAP and 0.0 <= beta <= BETA_CAP):
            raise AssertionError("selected action left the frozen box")
        interior += int(0.0 < alpha < ALPHA_CAP and 0.0 < beta < BETA_CAP)
        state = _advance(scenario, state, owner, alpha, beta)
        risk = _risk(scenario, state)
        if not math.isfinite(risk) or risk < 0.0:
            raise FloatingPointError("non-finite or negative Lyapunov risk")
        risks.append(risk)
        alphas.append(alpha)
        betas.append(beta)
        cross_terms.append(cross)

    return {
        "normalized_auc": float(np.mean(risks) / initial_risk),
        "normalized_terminal": float(risks[-1] / initial_risk),
        "mean_alpha": float(np.mean(alphas)),
        "mean_beta": float(np.mean(betas)),
        "joint_interior_fraction": float(interior / HORIZON),
        "mean_abs_cross_curvature": float(np.mean(np.abs(cross_terms))),
        "max_abs_cross_curvature": float(np.max(np.abs(cross_terms))),
        "minimum_qp_eigenvalue": float(minimum_eigenvalue),
        "last_event_time": event_time,
    }


def best_fixed_action(scenario: Scenario) -> tuple[tuple[float, float], dict[str, object]]:
    alphas = np.linspace(0.0, ALPHA_CAP, FIXED_GRID_POINTS)
    betas = np.linspace(0.0, BETA_CAP, FIXED_GRID_POINTS)
    grid: list[tuple[float, float, float]] = []
    for alpha, beta in itertools.product(alphas, betas):
        result = simulate(scenario, "fixed", (float(alpha), float(beta)))
        grid.append((float(result["normalized_auc"]), float(alpha), float(beta)))
    grid.sort(key=lambda row: row[0])
    best_value, best_alpha, best_beta = grid[0]
    refinement_worse = False
    refinement_successes = 0

    def objective(action: np.ndarray) -> float:
        return float(
            simulate(scenario, "fixed", (float(action[0]), float(action[1])))[
                "normalized_auc"
            ]
        )

    for _, alpha, beta in grid[:FIXED_REFINEMENT_STARTS]:
        optimized = minimize(
            objective,
            x0=np.asarray([alpha, beta]),
            method="L-BFGS-B",
            bounds=((0.0, ALPHA_CAP), (0.0, BETA_CAP)),
            options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 180},
        )
        value = objective(optimized.x)
        refinement_successes += int(bool(optimized.success))
        refinement_worse = refinement_worse or value > objective(np.asarray([alpha, beta])) + 1e-10
        if value < best_value:
            best_value = value
            best_alpha, best_beta = map(float, optimized.x)
    result = simulate(scenario, "fixed", (best_alpha, best_beta))
    result.update(
        {
            "alpha": best_alpha,
            "beta": best_beta,
            "grid_best_auc": float(grid[0][0]),
            "refinement_not_worse": not refinement_worse,
            "refinement_successes": refinement_successes,
        }
    )
    return (best_alpha, best_beta), result


def _geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if np.any(array <= 0.0) or np.any(~np.isfinite(array)):
        raise ValueError("geometric mean requires positive finite inputs")
    return float(np.exp(np.mean(np.log(array))))


def execute() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for scenario in frozen_scenarios():
        coupled = simulate(scenario, "coupled")
        diagonal = simulate(scenario, "diagonal_online")
        fixed_action, fixed = best_fixed_action(scenario)
        rows.append(
            {
                **asdict(scenario),
                "key": scenario.key,
                "coupled": coupled,
                "diagonal_online": diagonal,
                "best_fixed": fixed,
                "best_fixed_action": list(fixed_action),
            }
        )

    primary = [row for row in rows if row["population"] == "primary"]
    zero_target = [row for row in rows if row["population"] == "zero_target"]
    coupled_fixed_ratios = [
        float(row["coupled"]["normalized_auc"] / row["best_fixed"]["normalized_auc"])
        for row in primary
    ]
    coupled_diagonal_ratios = [
        float(row["coupled"]["normalized_auc"] / row["diagonal_online"]["normalized_auc"])
        for row in primary
    ]
    low_improvements = [
        1.0 - ratio
        for ratio, row in zip(coupled_diagonal_ratios, primary)
        if row["target_sensitivity"] == 0.4
    ]
    high_improvements = [
        1.0 - ratio
        for ratio, row in zip(coupled_diagonal_ratios, primary)
        if row["target_sensitivity"] == 0.9
    ]
    zero_target_error = max(
        max(
            abs(float(row["coupled"][field]) - float(row["diagonal_online"][field]))
            for field in ("normalized_auc", "normalized_terminal", "mean_alpha", "mean_beta")
        )
        for row in zero_target
    )
    gates = {
        "H1": len(primary) == 128 and len(rows) == 160,
        "H2": all(
            float(row["coupled"]["minimum_qp_eigenvalue"]) >= -PSD_TOLERANCE
            for row in rows
        ),
        "H3": _geometric_mean(coupled_fixed_ratios) <= 0.90,
        "H4": _geometric_mean(coupled_diagonal_ratios) <= 0.95,
        "H5": (
            float(np.mean(np.asarray(coupled_fixed_ratios) < 1.0)) >= 0.70
            and float(np.mean(np.asarray(coupled_diagonal_ratios) < 1.0)) >= 0.60
        ),
        "H6": (
            all(float(row["coupled"]["mean_alpha"]) > 0.0 for row in primary)
            and all(float(row["coupled"]["mean_beta"]) > 0.0 for row in primary)
            and float(np.mean([row["coupled"]["joint_interior_fraction"] for row in primary]))
            >= 0.05
        ),
        "H7": (
            zero_target_error <= 1e-10
            and all(float(row["coupled"]["max_abs_cross_curvature"]) <= 1e-10 for row in zero_target)
        ),
        "H8": float(np.median(high_improvements)) > float(np.median(low_improvements)),
        "H9": all(
            bool(row["best_fixed"]["refinement_not_worse"])
            and 0.0 <= float(row["best_fixed"]["alpha"]) <= ALPHA_CAP
            and 0.0 <= float(row["best_fixed"]["beta"]) <= BETA_CAP
            for row in rows
        ),
    }
    summary = {
        "scope": "privileged exact-moment oracle headroom; not efficacy",
        "frozen_constants": {
            "horizon": HORIZON,
            "alpha_cap": ALPHA_CAP,
            "beta_cap": BETA_CAP,
            "critic_contraction": CRITIC_CONTRACTION,
            "critic_weight": CRITIC_WEIGHT,
            "fixed_grid_points": FIXED_GRID_POINTS,
            "fixed_refinement_starts": FIXED_REFINEMENT_STARTS,
        },
        "counts": {
            "all": len(rows),
            "primary": len(primary),
            "zero_target": len(zero_target),
            "zero_interaction": sum(row["population"] == "zero_interaction" for row in rows),
        },
        "metrics": {
            "coupled_over_best_fixed_geometric_auc_ratio": _geometric_mean(coupled_fixed_ratios),
            "coupled_over_diagonal_geometric_auc_ratio": _geometric_mean(coupled_diagonal_ratios),
            "coupled_better_than_fixed_fraction": float(
                np.mean(np.asarray(coupled_fixed_ratios) < 1.0)
            ),
            "coupled_better_than_diagonal_fraction": float(
                np.mean(np.asarray(coupled_diagonal_ratios) < 1.0)
            ),
            "joint_interior_action_fraction": float(
                np.mean([row["coupled"]["joint_interior_fraction"] for row in primary])
            ),
            "low_target_median_diagonal_improvement": float(np.median(low_improvements)),
            "high_target_median_diagonal_improvement": float(np.median(high_improvements)),
            "zero_target_max_reduction_error": zero_target_error,
        },
        "gates": gates,
        "formal_or_sampled_authorized": False,
        "rows": rows,
    }
    summary["decision"] = "PASS" if all(gates.values()) else "STOP"
    return summary


def validate() -> dict[str, object]:
    scenarios = frozen_scenarios()
    counts = {
        population: sum(scenario.population == population for scenario in scenarios)
        for population in ("primary", "zero_target", "zero_interaction")
    }
    payload = {
        "status": "valid",
        "outcome_free": True,
        "counts": counts,
        "constants": {
            "horizon": HORIZON,
            "fixed_grid_points": FIXED_GRID_POINTS,
            "fixed_refinement_starts": FIXED_REFINEMENT_STARTS,
        },
        "scenario_hash": hashlib.sha256(
            json.dumps([asdict(scenario) for scenario in scenarios], sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = validate() if args.validate else execute()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

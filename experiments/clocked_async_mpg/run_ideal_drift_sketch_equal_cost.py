"""Frozen equal-cost upper-bound audit for an ideal Lyapunov drift sketch."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .coupled_actor_critic_drift import solve_two_dimensional_box_qp
from .run_coupled_actor_critic_headroom import (
    ALPHA_CAP,
    BETA_CAP,
    HORIZON,
    MomentState,
    Scenario,
    _event_matrices,
    _geometric_mean,
    _initial_state,
    _metric,
    _next_owner,
    _noise_variances,
    _risk,
    best_fixed_action,
    frozen_scenarios,
    simulate,
    validate as validate_source,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "docs" / "ideal_drift_sketch_equal_cost_gates.json"


def load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _draw_gaussian_state(
    state: MomentState, sample_count: int, rng: np.random.Generator
) -> np.ndarray:
    covariance = 0.5 * (state.covariance + state.covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[0] < -1e-9:
        raise FloatingPointError("moment covariance is not positive semidefinite")
    factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    standard = rng.normal(size=(sample_count, state.mean.size))
    return state.mean[None, :] + standard @ factor.T


def _true_split_coefficients(
    scenario: Scenario, state: MomentState, owner: int, variance_multiplier: float
) -> tuple[np.ndarray, np.ndarray]:
    metric = _metric(scenario)
    actor_matrix, critic_matrix, _, actor_noise, critic_noise = _event_matrices(
        scenario, owner
    )
    second = state.covariance + np.outer(state.mean, state.mean)
    matrices = (actor_matrix, critic_matrix)
    linear = 2.0 * np.asarray(
        [np.trace(matrix.T @ metric @ second) for matrix in matrices]
    )
    quadratic = np.empty((2, 2), dtype=float)
    for row, left in enumerate(matrices):
        for column, right in enumerate(matrices):
            quadratic[row, column] = 2.0 * np.trace(
                left.T @ metric @ right @ second
            )
    actor_variance, critic_variance = _noise_variances(scenario)
    quadratic[0, 0] += (
        2.0
        * variance_multiplier
        * actor_variance
        * float(actor_noise @ metric @ actor_noise)
    )
    quadratic[1, 1] += (
        2.0
        * variance_multiplier
        * critic_variance
        * float(critic_noise @ metric @ critic_noise)
    )
    return linear, 0.5 * (quadratic + quadratic.T)


def _ideal_sketch_coefficients(
    scenario: Scenario,
    state: MomentState,
    owner: int,
    *,
    s1: int,
    s2: int,
    variance_multiplier: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    metric = _metric(scenario)
    actor_matrix, critic_matrix, _, actor_noise, critic_noise = _event_matrices(
        scenario, owner
    )
    linear_samples = _draw_gaussian_state(state, s1, rng)
    actor_linear_response = linear_samples @ actor_matrix.T
    critic_linear_response = linear_samples @ critic_matrix.T
    linear = 2.0 * np.asarray(
        [
            np.mean(np.einsum("bi,ij,bj->b", actor_linear_response, metric, linear_samples)),
            np.mean(np.einsum("bi,ij,bj->b", critic_linear_response, metric, linear_samples)),
        ]
    )

    quadratic_samples = _draw_gaussian_state(state, s2, rng)
    responses = np.stack(
        [quadratic_samples @ actor_matrix.T, quadratic_samples @ critic_matrix.T],
        axis=1,
    )
    quadratic = 2.0 * np.mean(
        np.einsum("bai,ij,bcj->bac", responses, metric, responses), axis=0
    )
    actor_variance, critic_variance = _noise_variances(scenario)
    quadratic[0, 0] += (
        2.0
        * variance_multiplier
        * actor_variance
        * float(actor_noise @ metric @ actor_noise)
    )
    quadratic[1, 1] += (
        2.0
        * variance_multiplier
        * critic_variance
        * float(critic_noise @ metric @ critic_noise)
    )
    return linear, 0.5 * (quadratic + quadratic.T)


def _advance_split(
    scenario: Scenario,
    state: MomentState,
    owner: int,
    alpha: float,
    beta: float,
    variance_multiplier: float,
) -> MomentState:
    actor_matrix, critic_matrix, reset, actor_noise, critic_noise = _event_matrices(
        scenario, owner
    )
    transition = np.eye(state.mean.size) + alpha * actor_matrix + beta * critic_matrix
    actor_variance, critic_variance = _noise_variances(scenario)
    innovation = variance_multiplier * (
        alpha**2 * actor_variance * np.outer(actor_noise, actor_noise)
        + beta**2 * critic_variance * np.outer(critic_noise, critic_noise)
    )
    next_mean = reset @ transition @ state.mean
    next_covariance = reset @ (
        transition @ state.covariance @ transition.T + innovation
    ) @ reset.T
    next_covariance = 0.5 * (next_covariance + next_covariance.T)
    next_times = state.next_times.copy()
    from .run_coupled_actor_critic_headroom import _service_intervals

    next_times[owner] += _service_intervals(scenario)[owner]
    return MomentState(next_mean, next_covariance, next_times)


def simulate_ideal_sketch(
    scenario: Scenario, seed: int, *, diagonalize: bool = False
) -> dict[str, float]:
    config = load_config()
    full_batch = int(config["full_batch_trajectories"])
    s1 = int(config["sensor_trajectories_s1"])
    s2 = int(config["sensor_trajectories_s2"])
    update = int(config["update_trajectories_u"])
    if s1 + s2 + update != full_batch:
        raise AssertionError("the frozen trajectory split is not exhaustive")
    variance_multiplier = full_batch / update
    rng = np.random.default_rng(seed)
    state = _initial_state(scenario)
    initial_risk = _risk(scenario, state)
    risks: list[float] = []
    regret = 0.0
    oracle_descent = 0.0
    minimum_sampled_eigenvalue = math.inf

    for _ in range(HORIZON):
        owner, _ = _next_owner(state)
        true_linear, true_quadratic = _true_split_coefficients(
            scenario, state, owner, variance_multiplier
        )
        estimated_linear, estimated_quadratic = _ideal_sketch_coefficients(
            scenario,
            state,
            owner,
            s1=s1,
            s2=s2,
            variance_multiplier=variance_multiplier,
            rng=rng,
        )
        minimum_sampled_eigenvalue = min(
            minimum_sampled_eigenvalue,
            float(np.linalg.eigvalsh(estimated_quadratic)[0]),
        )
        decision_quadratic = (
            np.diag(np.diag(estimated_quadratic))
            if diagonalize
            else estimated_quadratic
        )
        action = solve_two_dimensional_box_qp(
            linear=estimated_linear,
            quadratic=decision_quadratic,
            upper=np.asarray([ALPHA_CAP, BETA_CAP]),
        ).action
        oracle = solve_two_dimensional_box_qp(
            linear=true_linear,
            quadratic=true_quadratic,
            upper=np.asarray([ALPHA_CAP, BETA_CAP]),
        )
        true_selected = float(
            true_linear @ action + 0.5 * action @ true_quadratic @ action
        )
        regret += max(0.0, true_selected - float(oracle.objective))
        oracle_descent += max(0.0, -float(oracle.objective))
        alpha, beta = map(float, action)
        state = _advance_split(
            scenario, state, owner, alpha, beta, variance_multiplier
        )
        risks.append(_risk(scenario, state))

    return {
        "normalized_auc": float(np.mean(risks) / initial_risk),
        "normalized_terminal": float(risks[-1] / initial_risk),
        "decision_regret_fraction": float(regret / max(oracle_descent, 1e-15)),
        "accumulated_decision_regret": float(regret),
        "accumulated_oracle_descent": float(oracle_descent),
        "minimum_sampled_qp_eigenvalue": float(minimum_sampled_eigenvalue),
        "variance_multiplier": float(variance_multiplier),
    }


def validate() -> dict[str, object]:
    config = load_config()
    source = validate_source()
    seeds = list(config["primary_sensor_seeds"])
    split_total = sum(
        int(config[field])
        for field in (
            "sensor_trajectories_s1",
            "sensor_trajectories_s2",
            "update_trajectories_u",
        )
    )
    payload = {
        "status": "valid",
        "outcome_free": True,
        "source_scenario_hash_matches": (
            source["scenario_hash"] == config["source_scenario_hash"]
        ),
        "seed_count": len(seeds),
        "seeds_unique": len(seeds) == len(set(seeds)),
        "split_total": split_total,
        "split_exhaustive": split_total == int(config["full_batch_trajectories"]),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
    }
    if not all(
        (
            payload["source_scenario_hash_matches"],
            payload["seeds_unique"],
            payload["split_exhaustive"],
            payload["seed_count"] == 16,
        )
    ):
        raise AssertionError("invalid frozen ideal-sketch configuration")
    return payload


def execute() -> dict[str, object]:
    config = load_config()
    seeds = [int(seed) for seed in config["primary_sensor_seeds"]]
    rows: list[dict[str, object]] = []
    for scenario in frozen_scenarios():
        exact_coupled = simulate(scenario, "coupled")
        exact_diagonal = simulate(scenario, "diagonal_online")
        fixed_action, fixed = best_fixed_action(scenario)
        for seed in seeds:
            sketch = simulate_ideal_sketch(scenario, seed)
            row: dict[str, object] = {
                **asdict(scenario),
                "key": scenario.key,
                "seed": seed,
                "ideal_sketch": sketch,
                "exact_coupled": exact_coupled,
                "exact_diagonal": exact_diagonal,
                "best_fixed": fixed,
                "best_fixed_action": list(fixed_action),
            }
            if scenario.population == "zero_target":
                row["ideal_diagonal_sketch"] = simulate_ideal_sketch(
                    scenario, seed, diagonalize=True
                )
            rows.append(row)

    primary = [row for row in rows if row["population"] == "primary"]
    zero_target = [row for row in rows if row["population"] == "zero_target"]
    fixed_ratios = [
        float(row["ideal_sketch"]["normalized_auc"])
        / float(row["best_fixed"]["normalized_auc"])
        for row in primary
    ]
    diagonal_ratios = [
        float(row["ideal_sketch"]["normalized_auc"])
        / float(row["exact_diagonal"]["normalized_auc"])
        for row in primary
    ]
    exact_ratios = [
        float(row["exact_coupled"]["normalized_auc"])
        / float(row["exact_diagonal"]["normalized_auc"])
        for row in primary
    ]
    sketch_gain = 1.0 - _geometric_mean(diagonal_ratios)
    exact_gain = 1.0 - _geometric_mean(exact_ratios)
    recovered = sketch_gain / exact_gain if exact_gain > 0.0 else -math.inf
    regret_good = [
        float(row["ideal_sketch"]["decision_regret_fraction"])
        <= float(config["gates"]["I7"]["per_cell_regret_fraction_maximum"])
        for row in primary
    ]
    zero_target_difference = max(
        abs(
            float(row["ideal_sketch"][metric])
            - float(row["ideal_diagonal_sketch"][metric])
        )
        for row in zero_target
        for metric in ("normalized_auc", "normalized_terminal")
    )
    metrics = {
        "ideal_over_best_fixed_geometric_auc_ratio": _geometric_mean(fixed_ratios),
        "ideal_over_exact_diagonal_geometric_auc_ratio": _geometric_mean(
            diagonal_ratios
        ),
        "ideal_better_than_exact_diagonal_fraction": float(
            np.mean(np.asarray(diagonal_ratios) < 1.0)
        ),
        "exact_coupled_over_diagonal_geometric_auc_ratio": _geometric_mean(
            exact_ratios
        ),
        "exact_coupled_headroom_recovered_fraction": float(recovered),
        "decision_regret_gate_fraction": float(np.mean(regret_good)),
        "median_decision_regret_fraction": float(
            np.median(
                [row["ideal_sketch"]["decision_regret_fraction"] for row in primary]
            )
        ),
        "zero_target_max_full_vs_diagonal_sketch_difference": zero_target_difference,
        "minimum_sampled_qp_eigenvalue": min(
            float(row["ideal_sketch"]["minimum_sampled_qp_eigenvalue"])
            for row in rows
        ),
    }
    thresholds = config["gates"]
    gates = {
        "I1": (
            len(primary) == 128 * len(seeds)
            and len(zero_target) == 16 * len(seeds)
            and all(math.isfinite(value) for value in metrics.values())
            and metrics["minimum_sampled_qp_eigenvalue"] >= -1e-9
        ),
        "I2": (
            int(config["sensor_trajectories_s1"])
            + int(config["sensor_trajectories_s2"])
            + int(config["update_trajectories_u"])
            == int(config["full_batch_trajectories"])
        ),
        "I3": metrics["ideal_over_best_fixed_geometric_auc_ratio"]
        <= float(thresholds["I3"]["maximum"]),
        "I4": metrics["ideal_over_exact_diagonal_geometric_auc_ratio"]
        <= float(thresholds["I4"]["maximum"]),
        "I5": metrics["ideal_better_than_exact_diagonal_fraction"]
        >= float(thresholds["I5"]["minimum"]),
        "I6": metrics["exact_coupled_headroom_recovered_fraction"]
        >= float(thresholds["I6"]["minimum"]),
        "I7": metrics["decision_regret_gate_fraction"]
        >= float(thresholds["I7"]["minimum"]),
        "I8": zero_target_difference <= float(thresholds["I8"]["maximum"]),
        "I9": True,
    }
    return {
        "scope": "privileged latent-state five-scalar upper bound; not efficacy",
        "configuration": config,
        "counts": {
            "rows": len(rows),
            "primary_seed_cells": len(primary),
            "zero_target_seed_cells": len(zero_target),
        },
        "metrics": metrics,
        "gates": gates,
        "decision_before_reproduction": "PASS" if all(gates.values()) else "STOP",
        "authorization": {
            "observable_markov_estimator": all(gates.values()),
            "formal": False,
            "gpu": False,
            "hpc4": False,
        },
        "rows": rows,
    }


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

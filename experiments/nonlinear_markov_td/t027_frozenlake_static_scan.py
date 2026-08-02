"""Outcome-free exact FrozenLake participation-value scan for T-027."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np


CONFIG = {
    "task": "FrozenLake-v1",
    "map_name": "8x8",
    "is_slippery": True,
    "policy": "uniform_random_over_four_actions",
    "terminal_convention": "reset_to_standard_start_on_next_transition",
    "gymnasium_version": "1.0.0",
    "parameter_count": 3169,
    "server_overhead_bytes": 65536,
    "bytes_per_parameter": 4,
    "q_values": [1, 4, 16, 32],
    "rho_values": [0.0, 0.1, 0.5, 0.9],
    "target_horizons": [512, 2048],
    "budget_rays": ["message", "environment"],
    "delay_fractions": [0.0, 0.05, 0.2],
    "mixing_tv_target": 0.05,
    "mixing_search_max": 2048,
    "mixing_gate_max": 512,
    "aggregate_oracle_improvement_gate": 0.05,
    "directional_cell_fraction_gate": 0.60,
    "required_distinct_oracle_q": 3,
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def config_sha256() -> str:
    return hashlib.sha256(canonical_json(CONFIG).encode("utf-8")).hexdigest()


def continuing_transition_matrix() -> tuple[np.ndarray, int, list[int]]:
    env = gym.make(
        CONFIG["task"],
        map_name=CONFIG["map_name"],
        is_slippery=CONFIG["is_slippery"],
    )
    if gym.__version__ != CONFIG["gymnasium_version"]:
        raise RuntimeError(
            f"gymnasium version drift: {gym.__version__} != "
            f"{CONFIG['gymnasium_version']}"
        )
    raw = env.unwrapped.P
    desc = env.unwrapped.desc.reshape(-1)
    n_states = int(env.observation_space.n)
    start = int(np.flatnonzero(desc == b"S")[0])
    terminals = [
        int(i) for i, value in enumerate(desc) if bytes(value) in (b"H", b"G")
    ]
    transition = np.zeros((n_states, n_states), dtype=np.float64)
    for state in range(n_states):
        if state in terminals:
            transition[state, start] = 1.0
            continue
        for action in range(int(env.action_space.n)):
            for probability, next_state, _reward, _done in raw[state][action]:
                transition[state, int(next_state)] += float(probability) / 4.0
    env.close()
    return transition, start, terminals


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    n = transition.shape[0]
    system = np.vstack([transition.T - np.eye(n), np.ones((1, n))])
    target = np.concatenate([np.zeros(n), np.ones(1)])
    stationary, *_ = np.linalg.lstsq(system, target, rcond=None)
    stationary = np.maximum(stationary, 0.0)
    stationary /= stationary.sum()
    return stationary


def mixing_stride(
    transition: np.ndarray, stationary: np.ndarray
) -> tuple[int | None, float]:
    power = np.eye(transition.shape[0])
    last_tv = math.inf
    for stride in range(1, int(CONFIG["mixing_search_max"]) + 1):
        power = power @ transition
        last_tv = float(0.5 * np.abs(power - stationary).sum(axis=1).max())
        if last_tv <= float(CONFIG["mixing_tv_target"]):
            return stride, last_tv
    return None, last_tv


def message_cost(q: int) -> int:
    return int(CONFIG["server_overhead_bytes"]) + int(
        CONFIG["bytes_per_parameter"]
    ) * int(CONFIG["parameter_count"]) * q


def variance_factor(q: int, rho: float) -> float:
    return rho + (1.0 - rho) / q


def usable_horizon(
    q: int, target_horizon: int, budget_ray: str, delay_fraction: float, stride: int
) -> tuple[int, int, int, int]:
    if budget_ray == "message":
        message_budget = target_horizon * message_cost(4)
        environment_budget = 2 * target_horizon * stride
    elif budget_ray == "environment":
        message_budget = target_horizon * message_cost(32)
        environment_budget = target_horizon * stride
    else:
        raise ValueError(budget_ray)
    delay_steps = int(round(delay_fraction * target_horizon * stride))
    updates = max(
        1,
        min(
            message_budget // message_cost(q),
            environment_budget // stride,
        )
        - math.ceil(delay_steps / stride),
    )
    return updates, message_budget, environment_budget, delay_steps


def geometric_mean(values: list[float]) -> float:
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def run_scan() -> dict[str, object]:
    transition, start, terminals = continuing_transition_matrix()
    stationary = stationary_distribution(transition)
    stride, terminal_tv = mixing_stride(transition, stationary)
    effective_stride = stride or int(CONFIG["mixing_search_max"])

    arms: list[dict[str, object]] = []
    for target_horizon in CONFIG["target_horizons"]:
        for budget_ray in CONFIG["budget_rays"]:
            for delay_fraction in CONFIG["delay_fractions"]:
                for rho in CONFIG["rho_values"]:
                    for q in CONFIG["q_values"]:
                        horizon, msg_budget, env_budget, delay_steps = usable_horizon(
                            q,
                            int(target_horizon),
                            str(budget_ray),
                            float(delay_fraction),
                            effective_stride,
                        )
                        arms.append(
                            {
                                "target_horizon": int(target_horizon),
                                "budget_ray": str(budget_ray),
                                "delay_fraction": float(delay_fraction),
                                "rho": float(rho),
                                "q": int(q),
                                "horizon": int(horizon),
                                "message_budget": int(msg_budget),
                                "environment_budget": int(env_budget),
                                "delay_steps": int(delay_steps),
                                "risk_proxy": variance_factor(int(q), float(rho))
                                / horizon,
                            }
                        )

    fallbacks: dict[tuple[int, str], int] = {}
    for target_horizon in CONFIG["target_horizons"]:
        for budget_ray in CONFIG["budget_rays"]:
            candidates: dict[int, list[float]] = {q: [] for q in CONFIG["q_values"]}
            for arm in arms:
                if (
                    arm["target_horizon"] == target_horizon
                    and arm["budget_ray"] == budget_ray
                ):
                    candidates[int(arm["q"])].append(float(arm["risk_proxy"]))
            fallbacks[(int(target_horizon), str(budget_ray))] = min(
                candidates, key=lambda q: (geometric_mean(candidates[q]), q)
            )

    cells: list[dict[str, object]] = []
    for target_horizon in CONFIG["target_horizons"]:
        for budget_ray in CONFIG["budget_rays"]:
            fallback_q = fallbacks[(int(target_horizon), str(budget_ray))]
            for delay_fraction in CONFIG["delay_fractions"]:
                for rho in CONFIG["rho_values"]:
                    group = [
                        arm
                        for arm in arms
                        if arm["target_horizon"] == target_horizon
                        and arm["budget_ray"] == budget_ray
                        and arm["delay_fraction"] == delay_fraction
                        and arm["rho"] == rho
                    ]
                    oracle = min(group, key=lambda arm: (arm["risk_proxy"], arm["q"]))
                    fallback = next(arm for arm in group if arm["q"] == fallback_q)
                    cells.append(
                        {
                            "target_horizon": int(target_horizon),
                            "budget_ray": str(budget_ray),
                            "delay_fraction": float(delay_fraction),
                            "rho": float(rho),
                            "fallback_q": int(fallback_q),
                            "oracle_q": int(oracle["q"]),
                            "fallback_risk": float(fallback["risk_proxy"]),
                            "oracle_risk": float(oracle["risk_proxy"]),
                        }
                    )

    fallback_geo = geometric_mean([float(cell["fallback_risk"]) for cell in cells])
    oracle_geo = geometric_mean([float(cell["oracle_risk"]) for cell in cells])
    improvement = 1.0 - oracle_geo / fallback_geo
    strict = sum(
        float(cell["oracle_risk"]) < float(cell["fallback_risk"]) - 1e-15
        for cell in cells
    )
    distinct_q = sorted({int(cell["oracle_q"]) for cell in cells})
    message_internal = any(
        cell["budget_ray"] == "message" and cell["oracle_q"] in (4, 16)
        for cell in cells
    )
    direction_pairs = 0
    direction_pass = 0
    lookup = {
        (
            cell["target_horizon"],
            cell["delay_fraction"],
            cell["rho"],
            cell["budget_ray"],
        ): int(cell["oracle_q"])
        for cell in cells
    }
    for target_horizon in CONFIG["target_horizons"]:
        for delay_fraction in CONFIG["delay_fractions"]:
            for rho in CONFIG["rho_values"]:
                direction_pairs += 1
                if lookup[(target_horizon, delay_fraction, rho, "environment")] >= lookup[
                    (target_horizon, delay_fraction, rho, "message")
                ]:
                    direction_pass += 1

    gates = {
        "S1_exact_markov_validity": bool(
            np.all(np.isfinite(transition))
            and np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 1e-12
            and np.max(np.abs(stationary @ transition - stationary)) <= 1e-10
        ),
        "S2_finite_mixing_certificate": stride is not None
        and stride <= int(CONFIG["mixing_gate_max"]),
        "S3_aggregate_oracle_value": improvement
        >= float(CONFIG["aggregate_oracle_improvement_gate"]),
        "S4_directional_cells": strict / len(cells)
        >= float(CONFIG["directional_cell_fraction_gate"]),
        "S5_distinct_oracle_participation": len(distinct_q)
        >= int(CONFIG["required_distinct_oracle_q"]),
        "S6_internal_message_optimum": message_internal,
        "S7_budget_direction": direction_pass == direction_pairs,
        "S8_no_trajectory_taint": True,
    }
    return {
        "task": "T-027",
        "config_sha256": config_sha256(),
        "scientific_trajectories": 0,
        "transition": {
            "states": int(transition.shape[0]),
            "start_state": start,
            "terminal_states": terminals,
            "row_sum_max_error": float(
                np.max(np.abs(transition.sum(axis=1) - 1.0))
            ),
            "stationarity_max_error": float(
                np.max(np.abs(stationary @ transition - stationary))
            ),
            "stationary_min": float(stationary.min()),
        },
        "mixing": {
            "target_tv": float(CONFIG["mixing_tv_target"]),
            "stride": stride,
            "terminal_tv": terminal_tv,
        },
        "design": {
            "arm_rows": len(arms),
            "cells": len(cells),
            "fallbacks": {
                f"H{h}_{ray}": q for (h, ray), q in sorted(fallbacks.items())
            },
        },
        "value": {
            "fallback_geometric_risk": fallback_geo,
            "oracle_geometric_risk": oracle_geo,
            "aggregate_improvement": improvement,
            "strict_cells": strict,
            "strict_fraction": strict / len(cells),
            "distinct_oracle_q": distinct_q,
            "budget_direction_passed": direction_pass,
            "budget_direction_total": direction_pairs,
        },
        "gates": gates,
        "passed_gates": sum(gates.values()),
        "total_gates": len(gates),
        "cpu_nonlinear_pilot_authorized": all(gates.values()),
        "gpu_authorized": False,
        "arms": arms,
        "cells_detail": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "run"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "validate":
        print(
            json.dumps(
                {
                    "config_sha256": config_sha256(),
                    "expected_arm_rows": 192,
                    "expected_cells": 48,
                    "scientific_trajectories": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.output_dir is None:
        parser.error("--output-dir is required in run mode")
    result = run_scan()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("gates", "value", "mixing")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


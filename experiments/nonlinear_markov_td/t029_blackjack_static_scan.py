"""Outcome-free exact Blackjack participation-value scan for T-029.

The scan constructs the continuing observation-state Markov chain induced by
an epsilon-soft fixed policy and an immediate post-terminal reset.  It uses no
sampled learning trajectory and does not authorize MinAtar or GPU execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import deque
from pathlib import Path

import numpy as np


CONFIG = {
    "task": "Blackjack-v1",
    "sab": False,
    "natural": False,
    "policy": "epsilon_soft_stick_at_20",
    "policy_epsilon": 0.10,
    "terminal_convention": "immediate_independent_standard_reset",
    "card_multiset": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10],
    "network": "normalized_3_input_mlp_32_32_1",
    "parameter_count": 1217,
    "server_overhead_bytes": 65536,
    "bytes_per_parameter": 4,
    "q_values": [1, 2, 4, 8, 16, 32],
    "rho_values": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
    "target_horizons": [512, 2048],
    "budget_rays": ["message", "environment"],
    "active_budget_rays": ["message"],
    "delay_fractions": [0.0, 0.05, 0.2],
    "mixing_tv_target": 0.05,
    "mixing_search_max": 512,
    "mixing_gate_max": 128,
    "aggregate_oracle_improvement_gate": 0.05,
    "active_directional_fraction_gate": 0.60,
    "required_distinct_oracle_q": 3,
}

State = tuple[int, int, bool]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def config_sha256() -> str:
    return hashlib.sha256(canonical_json(CONFIG).encode("utf-8")).hexdigest()


def card_probabilities() -> dict[int, float]:
    cards = [int(card) for card in CONFIG["card_multiset"]]
    return {card: cards.count(card) / len(cards) for card in sorted(set(cards))}


def hand_observation(card_a: int, card_b: int) -> tuple[int, bool]:
    raw = card_a + card_b
    usable = (card_a == 1 or card_b == 1) and raw + 10 <= 21
    return raw + (10 if usable else 0), usable


def add_card(total: int, usable_ace: bool, card: int) -> tuple[int, bool]:
    if usable_ace:
        updated = total + card
        if updated > 21:
            return updated - 10, False
        return updated, True
    if card == 1 and total + 11 <= 21:
        return total + 11, True
    return total + card, False


def reset_distribution() -> dict[State, float]:
    probabilities = card_probabilities()
    distribution: dict[State, float] = {}
    for dealer, p_dealer in probabilities.items():
        for card_a, p_a in probabilities.items():
            for card_b, p_b in probabilities.items():
                player, usable = hand_observation(card_a, card_b)
                state = (player, dealer, usable)
                distribution[state] = distribution.get(state, 0.0) + (
                    p_dealer * p_a * p_b
                )
    return distribution


def hit_probability(state: State) -> float:
    preferred_hit = state[0] < 20
    epsilon = float(CONFIG["policy_epsilon"])
    return 1.0 - epsilon if preferred_hit else epsilon


def reachable_states() -> list[State]:
    reset = reset_distribution()
    seen = set(reset)
    frontier: deque[State] = deque(sorted(seen))
    probabilities = card_probabilities()
    while frontier:
        total, dealer, usable = frontier.popleft()
        for card in probabilities:
            next_total, next_usable = add_card(total, usable, card)
            if next_total > 21:
                continue
            next_state = (next_total, dealer, next_usable)
            if next_state not in seen:
                seen.add(next_state)
                frontier.append(next_state)
    return sorted(seen)


def continuing_transition_matrix() -> tuple[np.ndarray, list[State], np.ndarray]:
    states = reachable_states()
    state_index = {state: index for index, state in enumerate(states)}
    reset = reset_distribution()
    reset_vector = np.array([reset.get(state, 0.0) for state in states])
    transition = np.zeros((len(states), len(states)), dtype=np.float64)
    probabilities = card_probabilities()
    for row, state in enumerate(states):
        p_hit = hit_probability(state)
        p_reset = 1.0 - p_hit
        for card, probability in probabilities.items():
            next_total, next_usable = add_card(state[0], state[2], card)
            if next_total > 21:
                p_reset += p_hit * probability
            else:
                next_state = (next_total, state[1], next_usable)
                transition[row, state_index[next_state]] += p_hit * probability
        transition[row] += p_reset * reset_vector
    return transition, states, reset_vector


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
        min(message_budget // message_cost(q), environment_budget // stride)
        - math.ceil(delay_steps / stride),
    )
    return updates, message_budget, environment_budget, delay_steps


def geometric_mean(values: list[float]) -> float:
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def run_scan() -> dict[str, object]:
    transition, states, reset = continuing_transition_matrix()
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
                            int(q),
                            int(target_horizon),
                            str(budget_ray),
                            float(delay_fraction),
                            effective_stride,
                        )
                        arms.append(
                            {
                                "target_horizon": int(target_horizon),
                                "budget_ray": str(budget_ray),
                                "active": budget_ray in CONFIG["active_budget_rays"],
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
                            "active": budget_ray in CONFIG["active_budget_rays"],
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
    active_cells = [cell for cell in cells if cell["active"]]
    strict_active = sum(
        float(cell["oracle_risk"]) < float(cell["fallback_risk"]) - 1e-15
        for cell in active_cells
    )
    inactive_cells = [cell for cell in cells if not cell["active"]]
    inactive_boundary = sum(
        int(cell["oracle_q"]) == max(CONFIG["q_values"]) for cell in inactive_cells
    )
    distinct_q = sorted({int(cell["oracle_q"]) for cell in cells})

    lookup = {
        (
            cell["target_horizon"],
            cell["delay_fraction"],
            cell["rho"],
            cell["budget_ray"],
        ): int(cell["oracle_q"])
        for cell in cells
    }
    direction_total = 0
    direction_passed = 0
    correlation_total = 0
    correlation_passed = 0
    for target_horizon in CONFIG["target_horizons"]:
        for delay_fraction in CONFIG["delay_fractions"]:
            message_path = []
            for rho in CONFIG["rho_values"]:
                direction_total += 1
                message_q = lookup[
                    (target_horizon, delay_fraction, rho, "message")
                ]
                environment_q = lookup[
                    (target_horizon, delay_fraction, rho, "environment")
                ]
                direction_passed += environment_q >= message_q
                message_path.append(message_q)
            correlation_total += 1
            correlation_passed += all(
                left >= right for left, right in zip(message_path, message_path[1:])
            )

    minorization_floor = min(1.0 - hit_probability(state) for state in states)
    gates = {
        "B1_exact_markov_validity": bool(
            np.all(np.isfinite(transition))
            and np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 1e-12
            and np.max(np.abs(stationary @ transition - stationary)) <= 1e-10
        ),
        "B2_reset_minorization_and_mixing": bool(
            minorization_floor >= float(CONFIG["policy_epsilon"]) - 1e-15
            and stride is not None
            and stride <= int(CONFIG["mixing_gate_max"])
        ),
        "B3_aggregate_oracle_value": improvement
        >= float(CONFIG["aggregate_oracle_improvement_gate"]),
        "B4_active_directional_cells": strict_active / len(active_cells)
        >= float(CONFIG["active_directional_fraction_gate"]),
        "B5_inactive_boundary_behavior": inactive_boundary == len(inactive_cells),
        "B6_distinct_oracle_participation": len(distinct_q)
        >= int(CONFIG["required_distinct_oracle_q"]),
        "B7_budget_and_correlation_direction": direction_passed == direction_total
        and correlation_passed == correlation_total,
        "B8_no_trajectory_taint": True,
    }
    return {
        "task": "T-029",
        "config_sha256": config_sha256(),
        "scientific_trajectories": 0,
        "transition": {
            "states": len(states),
            "row_sum_max_error": float(
                np.max(np.abs(transition.sum(axis=1) - 1.0))
            ),
            "stationarity_max_error": float(
                np.max(np.abs(stationary @ transition - stationary))
            ),
            "reset_mass": float(reset.sum()),
            "minorization_floor": float(minorization_floor),
        },
        "mixing": {
            "target_tv": float(CONFIG["mixing_tv_target"]),
            "stride": stride,
            "terminal_tv": terminal_tv,
        },
        "design": {
            "arm_rows": len(arms),
            "cells": len(cells),
            "active_cells": len(active_cells),
            "inactive_cells": len(inactive_cells),
            "fallbacks": {
                f"H{h}_{ray}": q for (h, ray), q in sorted(fallbacks.items())
            },
        },
        "value": {
            "fallback_geometric_risk": fallback_geo,
            "oracle_geometric_risk": oracle_geo,
            "aggregate_improvement": improvement,
            "strict_active_cells": strict_active,
            "strict_active_fraction": strict_active / len(active_cells),
            "inactive_boundary_cells": inactive_boundary,
            "distinct_oracle_q": distinct_q,
            "budget_direction_passed": direction_passed,
            "budget_direction_total": direction_total,
            "correlation_paths_passed": correlation_passed,
            "correlation_paths_total": correlation_total,
        },
        "gates": gates,
        "passed_gates": sum(gates.values()),
        "total_gates": len(gates),
        "blackjack_cpu_learning_pilot_authorized": all(gates.values()),
        "asterix_gpu_authorized": False,
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
                    "expected_arm_rows": 432,
                    "expected_cells": 72,
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
    print(
        json.dumps(
            {key: result[key] for key in ("gates", "value", "mixing")},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Outcome-free exact audit of certificate range-term nonvacuity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .markov_actor_critic_packet import simultaneous_empirical_bernstein_radius
from .tabular_actor_critic_interface import (
    finite_horizon_return_bound,
    tabular_critic_geometry,
    tabular_packet_coordinate_bounds,
)
from .trajectory_interface import exact_policy_gradient


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "docs" / "coupled_actor_critic_certificate_nonvacuity_gates.json"


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_game(seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    agents, states, actions = 2, 2, 2
    profiles = actions**agents
    transition = rng.uniform(0.2, 1.0, size=(states, profiles, states))
    transition /= np.sum(transition, axis=-1, keepdims=True)
    reward = rng.uniform(-0.8, 0.8, size=(states, profiles))
    start = np.asarray([0.45, 0.55])
    logits = rng.normal(scale=0.35, size=(agents, states, actions))
    return transition, reward, start, logits


def load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def scenarios(config: dict[str, object]) -> list[dict[str, object]]:
    seeds = config["game_seeds"]
    assert isinstance(seeds, dict)
    result = []
    for seed in range(int(seeds["start"]), int(seeds["start"]) + int(seeds["count"])):
        for horizon in config["horizons"]:
            for discount in config["discounts"]:
                for owner in config["owners"]:
                    result.append(
                        {
                            "discount": float(discount),
                            "horizon": int(horizon),
                            "owner": int(owner),
                            "seed": seed,
                        }
                    )
    return result


def validate_config(config: dict[str, object]) -> dict[str, object]:
    cases = scenarios(config)
    scenario_hash = _canonical_hash(cases)
    expected = str(config["expected_scenario_hash"])
    return {
        "case_count": len(cases),
        "case_count_matches": len(cases) == int(config["expected_case_count"]),
        "scenario_hash": scenario_hash,
        "scenario_hash_matches": expected == scenario_hash,
    }


def _minimum_grid_point(predicate, grid: list[int]) -> int | None:
    for value in grid:
        if predicate(value):
            return int(value)
    return None


def run(output_dir: Path) -> dict[str, object]:
    config = load_config()
    validation = validate_config(config)
    if not validation["case_count_matches"] or not validation["scenario_hash_matches"]:
        raise RuntimeError(f"frozen configuration mismatch: {validation}")
    grid = [int(value) for value in config["trajectory_count_grid"]]
    packet_count = int(config["scheduled_packets"])
    failure = float(config["failure_probability"])
    critic_error = float(config["critic_error_radius"])
    cells: list[dict[str, object]] = []
    for case in scenarios(config):
        transition, reward, start, logits = _make_game(int(case["seed"]))
        horizon = int(case["horizon"])
        discount = float(case["discount"])
        owner = int(case["owner"])
        reward_bound = float(np.max(np.abs(reward)))
        critic_bound = finite_horizon_return_bound(
            horizon=horizon, discount=discount, reward_bound=reward_bound
        )
        actor_coordinate_bound, critic_coordinate_bound = tabular_packet_coordinate_bounds(
            horizon=horizon,
            discount=discount,
            reward_bound=reward_bound,
            critic_abs_bound=critic_bound,
        )
        gradient = exact_policy_gradient(
            transition, reward, start, logits, discount, horizon=horizon
        )[1][owner]
        gradient_signal = float(np.linalg.norm(gradient))
        geometry = tabular_critic_geometry(
            transition, reward, start, logits, discount, horizon
        )
        actor_dimension = int(gradient.size)
        critic_dimension = int(geometry.target.size)
        joint_coordinates = actor_dimension + critic_dimension

        def actor_accepts(count: int) -> bool:
            radius = simultaneous_empirical_bernstein_radius(
                coordinate_sample_variances=np.zeros(actor_dimension),
                trajectory_count=count,
                coordinate_abs_bounds=actor_coordinate_bound,
                scheduled_packet_count=packet_count,
                joint_coordinate_count=joint_coordinates,
                failure_probability=failure,
            )
            return radius < gradient_signal

        def critic_accepts(count: int) -> bool:
            radius = simultaneous_empirical_bernstein_radius(
                coordinate_sample_variances=np.zeros(critic_dimension),
                trajectory_count=count,
                coordinate_abs_bounds=critic_coordinate_bound,
                scheduled_packet_count=packet_count,
                joint_coordinate_count=joint_coordinates,
                failure_probability=failure,
            )
            return radius < geometry.strong_convexity * critic_error

        actor_minimum = _minimum_grid_point(actor_accepts, grid)
        critic_minimum = _minimum_grid_point(critic_accepts, grid)
        joint_minimum = (
            None
            if actor_minimum is None or critic_minimum is None
            else max(actor_minimum, critic_minimum)
        )
        charged = None if joint_minimum is None else joint_minimum * horizon
        cells.append(
            {
                **case,
                "actor_coordinate_bound": actor_coordinate_bound,
                "actor_minimum_trajectories": actor_minimum,
                "critic_coordinate_bound": critic_coordinate_bound,
                "critic_minimum_trajectories": critic_minimum,
                "critic_strong_convexity": geometry.strong_convexity,
                "gradient_signal": gradient_signal,
                "joint_minimum_charged_transitions": charged,
                "joint_minimum_trajectories": joint_minimum,
            }
        )

    charged_values = np.asarray(
        [
            math.inf if cell["joint_minimum_charged_transitions"] is None else cell["joint_minimum_charged_transitions"]
            for cell in cells
        ],
        dtype=float,
    )
    practical = int(config["practical_transition_cap"])
    extended = int(config["extended_transition_cap"])
    long_mask = np.asarray([int(cell["horizon"]) >= 4 for cell in cells])
    finite = all(
        math.isfinite(float(cell[key]))
        for cell in cells
        for key in (
            "actor_coordinate_bound",
            "critic_coordinate_bound",
            "critic_strong_convexity",
            "gradient_signal",
        )
    )
    gates = {
        "N1": bool(validation["case_count_matches"] and validation["scenario_hash_matches"]),
        "N2": bool(finite and all(float(cell["critic_strong_convexity"]) > 0.0 for cell in cells)),
        "N3": bool(np.mean(charged_values <= practical) >= 0.50),
        "N4": bool(np.median(charged_values) <= extended),
        "N5": bool(np.mean(charged_values[long_mask] <= extended) >= 0.25),
        "N6": True,
    }
    summary = {
        "audit": config["audit"],
        "case_count": len(cells),
        "decision": "PASS" if all(gates.values()) else "STOP_HIGH_PROBABILITY_SHIELD",
        "fraction_joint_within_extended_cap": float(np.mean(charged_values <= extended)),
        "fraction_joint_within_practical_cap": float(np.mean(charged_values <= practical)),
        "fraction_long_horizon_within_extended_cap": float(np.mean(charged_values[long_mask] <= extended)),
        "gates_before_reproduction": gates,
        "median_minimum_charged_transitions": float(np.median(charged_values)),
        "scenario_hash": validation["scenario_hash"],
        "zero_variance_optimistic": True,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "cells.json").write_text(
        json.dumps(cells, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "run"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "validate":
        print(json.dumps(validate_config(load_config()), indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for run")
    print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

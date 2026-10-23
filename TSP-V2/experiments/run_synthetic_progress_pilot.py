"""CPU mechanism pilot for the Lyapunov learning-progress controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from learning_progress_controller import (
    ActionCost,
    ResourceBudget,
    affine_terminal_risk,
    fit_affine_drift_certificate,
    select_learning_progress_action,
)


CATALOGUE = (1, 2, 4, 8)
COSTS = {
    1: ActionCost(1000, 200),
    2: ActionCost(1200, 200),
    4: ActionCost(1600, 200),
    8: ActionCost(2400, 200),
}
SCENARIOS = {
    "broad": {
        1: (0.92, 0.050),
        2: (0.88, 0.035),
        4: (0.82, 0.020),
        8: (0.72, 0.010),
    },
    "interior": {
        1: (0.90, 0.050),
        2: (0.72, 0.010),
        4: (0.82, 0.040),
        8: (0.88, 0.060),
    },
    "local": {
        1: (0.72, 0.010),
        2: (0.82, 0.030),
        4: (0.88, 0.050),
        8: (0.92, 0.070),
    },
}


def sample_pairs(
    rng: np.random.Generator,
    a: float,
    c: float,
    count: int,
    noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(2.0, 10.0, count)
    y = np.maximum(a * x + c + rng.normal(0.0, noise, count), 0.0)
    return x, y


def oracle_action(
    scenario: dict[int, tuple[float, float]],
    budget: ResourceBudget,
    initial: float,
) -> int:
    risks = {}
    for q, (a, c) in scenario.items():
        horizon = min(
            budget.message_remaining // COSTS[q].messages_per_update,
            budget.environment_remaining // COSTS[q].environment_per_update,
        )
        certificate = type("Exact", (), {"a_upper": a, "c_upper": c})()
        risks[q] = affine_terminal_risk(initial, horizon, certificate)
    return min(risks, key=lambda q: (risks[q], q))


def run(seeds: int, samples: int, noise: float) -> dict:
    budget = ResourceBudget(5_000_000, 1_000_000)
    initial = 10.0
    rows = []
    for scenario_name, scenario in SCENARIOS.items():
        oracle = oracle_action(scenario, budget, initial)
        for seed in range(seeds):
            rng = np.random.default_rng(44000 + seed)
            certificates = {}
            covered = True
            try:
                for q, (a, c) in scenario.items():
                    x, y = sample_pairs(rng, a, c, samples, noise)
                    certificates[q] = fit_affine_drift_certificate(
                        x,
                        y,
                        confidence=1.0 - 0.05 / len(CATALOGUE),
                        noise_scale=noise,
                        parameter_radius=1.5,
                        ridge=0.01,
                    )
                    covered = covered and certificates[q].a_upper >= a
                    covered = covered and certificates[q].c_upper >= c
                selected = select_learning_progress_action(
                    initial_lyapunov=initial,
                    budget=budget,
                    costs=COSTS,
                    certificates=certificates,
                ).selected_q
            except ValueError:
                selected = None
                covered = False
            rows.append(
                {
                    "scenario": scenario_name,
                    "seed": seed,
                    "oracle_q": oracle,
                    "selected_q": selected,
                    "covered": bool(covered),
                    "correct": selected == oracle,
                }
            )
    summary = {}
    for name in SCENARIOS:
        cell = [row for row in rows if row["scenario"] == name]
        summary[name] = {
            "oracle_q": cell[0]["oracle_q"],
            "selection_rate": sum(row["correct"] for row in cell) / len(cell),
            "rectangle_coverage": sum(row["covered"] for row in cell) / len(cell),
        }
    return {"config": {"seeds": seeds, "samples": samples, "noise": noise}, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=256)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seeds, args.samples, args.noise)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()

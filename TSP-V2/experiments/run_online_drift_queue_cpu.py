"""CPU feasibility study for the online Lyapunov-bandit controller.

This synthetic study contains no manuscript benchmark result.  It checks the
algorithmic interfaces that must hold before a fresh-seed MARL development
experiment: online adaptation, exact dual-budget safety, and nontrivial value
relative to every budget-feasible fixed catalogue action.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from online_drift_queue_controller import (
    OnlineActionCost,
    SlidingWindowDriftQueueController,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    first: dict[int, float]
    second: dict[int, float]


CATALOGUE = (1, 2, 4, 8)
COSTS = {
    1: OnlineActionCost(messages=10, environment_ticks=10),
    2: OnlineActionCost(messages=20, environment_ticks=10),
    4: OnlineActionCost(messages=45, environment_ticks=10),
    8: OnlineActionCost(messages=90, environment_ticks=10),
}
SCENARIOS = (
    Scenario(
        name="low-dependence",
        first={1: 0.0006, 2: 0.0012, 4: 0.0028, 8: 0.0030},
        second={1: 0.0008, 2: 0.0014, 4: 0.0024, 8: 0.0022},
    ),
    Scenario(
        name="high-dependence",
        first={1: 0.0020, 2: 0.0016, 4: 0.0004, 8: -0.0004},
        second={1: 0.0022, 2: 0.0014, 4: 0.0002, 8: -0.0006},
    ),
    Scenario(
        name="phase-switch",
        first={1: 0.0004, 2: 0.0012, 4: 0.0032, 8: 0.0028},
        second={1: 0.0028, 2: 0.0016, 4: 0.0002, 8: -0.0006},
    ),
)


def means(scenario: Scenario, step: int, horizon: int) -> dict[int, float]:
    return scenario.first if step < horizon // 2 else scenario.second


def apply_progress(risk: float, progress: float) -> tuple[float, float]:
    after = float(np.clip(risk - progress, 0.0, 1.0))
    return after, risk - after


def fixed_feasible(action: int, *, horizon: int, messages: int, environment: int) -> bool:
    cost = COSTS[action]
    return cost.messages * horizon <= messages and cost.environment_ticks * horizon <= environment


def run_one(
    *,
    scenario: Scenario,
    seed: int,
    horizon: int,
    messages: int,
    environment: int,
    bonus_scale: float,
) -> dict:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.0008, size=(horizon, len(CATALOGUE)))
    controller = SlidingWindowDriftQueueController(
        costs=COSTS,
        total_messages=messages,
        total_environment_ticks=environment,
        decisions=horizon,
        window=8,
        bonus_scale=bonus_scale,
        probe_interval=40,
        progress_scale=0.004,
        queue_weight=80.0,
    )
    risk = 0.65
    actions = []
    realized_progress = []
    for step in range(horizon):
        decision = controller.decide()
        action = decision.action
        action_index = CATALOGUE.index(action)
        raw_progress = means(scenario, step, horizon)[action] + noise[step, action_index]
        after, actual_progress = apply_progress(risk, raw_progress)
        controller.observe(risk_before=risk, risk_after=after)
        risk = after
        actions.append(action)
        realized_progress.append(actual_progress)

    fixed = {}
    for action in CATALOGUE:
        if not fixed_feasible(
            action, horizon=horizon, messages=messages, environment=environment
        ):
            continue
        fixed_risk = 0.65
        action_index = CATALOGUE.index(action)
        for step in range(horizon):
            raw_progress = means(scenario, step, horizon)[action] + noise[step, action_index]
            fixed_risk, _ = apply_progress(fixed_risk, raw_progress)
        fixed[str(action)] = fixed_risk

    return {
        "scenario": scenario.name,
        "seed": seed,
        "controller_final_risk": risk,
        "best_fixed_final_risk": min(fixed.values()),
        "fixed_final_risks": fixed,
        "action_counts": {str(q): actions.count(q) for q in CATALOGUE},
        "message_remaining": controller.message_remaining,
        "environment_remaining": controller.environment_remaining,
        "progress_sum": float(np.sum(realized_progress)),
    }


def summarize(rows: list[dict]) -> dict:
    result = {}
    for scenario in SCENARIOS:
        subset = [row for row in rows if row["scenario"] == scenario.name]
        controller = np.asarray([row["controller_final_risk"] for row in subset])
        baseline = np.asarray([row["best_fixed_final_risk"] for row in subset])
        improvement = (baseline - controller) / np.maximum(baseline, 1e-12)
        result[scenario.name] = {
            "seeds": len(subset),
            "median_controller_final_risk": float(np.median(controller)),
            "median_best_fixed_final_risk": float(np.median(baseline)),
            "median_relative_improvement": float(np.median(improvement)),
            "win_rate": float(np.mean(controller < baseline)),
            "min_message_remaining": int(min(row["message_remaining"] for row in subset)),
            "min_environment_remaining": int(
                min(row["environment_remaining"] for row in subset)
            ),
            "mean_action_counts": {
                str(q): float(
                    np.mean([row["action_counts"][str(q)] for row in subset])
                )
                for q in CATALOGUE
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--messages", type=int, default=3000)
    parser.add_argument("--environment", type=int, default=1200)
    parser.add_argument("--bonus-scale", type=float, default=0.08)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows = [
        run_one(
            scenario=scenario,
            seed=150000 + seed,
            horizon=args.horizon,
            messages=args.messages,
            environment=args.environment,
            bonus_scale=args.bonus_scale,
        )
        for scenario in SCENARIOS
        for seed in range(args.seeds)
    ]
    payload = {
        "role": "CPU algorithmic feasibility; not manuscript evidence",
        "seeds": args.seeds,
        "horizon": args.horizon,
        "message_budget": args.messages,
        "environment_budget": args.environment,
        "bonus_scale": args.bonus_scale,
        "catalogue": list(CATALOGUE),
        "costs": {str(q): asdict(COSTS[q]) for q in CATALOGUE},
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

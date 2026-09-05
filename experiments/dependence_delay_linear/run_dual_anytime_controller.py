"""Pilot and formal runner for dual anytime p/rho adaptation."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dual_anytime_controller import (
    INITIAL_PAIR_TRIALS,
    PAIR_PROBE_COST,
    block_observation_counts,
    dual_confidence_bounds,
    select_dual_action,
)
from predictable_mixing_controller import RESOURCE_BUDGET
from progressive_mixing_controller import (
    BLOCK_BUDGET,
    advance_action,
    advance_observations,
    block_execution_counts,
    final_expected_error,
    initial_covariance_state,
)


PERSISTENCES = (0.5, 0.9, 0.98)
CORRELATIONS = (0.0, 0.5, 0.9)
DELAYS = (0, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "dual_anytime_controller_smoke",
    )
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=20261211)
    return parser.parse_args()


def oracle_error(
    persistence: float,
    rho: float,
    delay: int,
    cache: Dict[Tuple[object, ...], Dict[str, float]],
) -> float:
    state = initial_covariance_state(delay)
    remaining = RESOURCE_BUDGET
    while remaining > 0:
        block = min(BLOCK_BUDGET, remaining)
        key = ("oracle", persistence, rho, delay, remaining)
        if key not in cache:
            cache[key] = select_dual_action(
                persistence, rho, delay, remaining
            )
        action = cache[key]
        counts = block_execution_counts(action, block)
        result = advance_action(
            state,
            action,
            persistence,
            rho,
            delay,
            counts["updates"],
        )
        state = advance_observations(
            result["state"],
            persistence,
            counts["leftover_observations"],
            delay,
        )
        remaining -= block
    return final_expected_error(state, delay)


def run_policy(
    persistence: float,
    rho: float,
    delay: int,
    seed: int,
    policy: str,
    cache: Dict[Tuple[object, ...], Dict[str, float]],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    rng = np.random.RandomState(seed)
    stays = int(rng.binomial(INITIAL_PAIR_TRIALS, persistence))
    shared = int(rng.binomial(INITIAL_PAIR_TRIALS, rho))
    transition_trials = INITIAL_PAIR_TRIALS
    sharing_trials = INITIAL_PAIR_TRIALS
    initial_cost = INITIAL_PAIR_TRIALS * PAIR_PROBE_COST
    remaining = RESOURCE_BUDGET - initial_cost
    state = initial_covariance_state(delay)
    decision = 0
    covered = True
    largest_radius = 0.0
    selected_q: List[int] = []
    rows: List[Dict[str, object]] = []
    initial_p_upper = float("nan")
    initial_rho_upper = float("nan")
    final_p_upper = float("nan")
    final_rho_upper = float("nan")
    while remaining > 0:
        decision += 1
        bounds = dual_confidence_bounds(
            stays,
            transition_trials,
            shared,
            sharing_trials,
            decision,
        )
        if decision == 1:
            initial_p_upper = bounds["persistence_upper"]
            initial_rho_upper = bounds["rho_upper"]
        final_p_upper = bounds["persistence_upper"]
        final_rho_upper = bounds["rho_upper"]
        covered = bool(
            covered
            and persistence <= bounds["persistence_upper"]
            and rho <= bounds["rho_upper"]
        )
        used_rho = (
            bounds["certified_rho"] if policy == "dual_ucb" else 0.0
        )
        key = (
            policy,
            bounds["certified_persistence"],
            used_rho,
            delay,
            remaining,
        )
        if key not in cache:
            cache[key] = select_dual_action(
                bounds["certified_persistence"],
                used_rho,
                delay,
                remaining,
            )
        action = cache[key]
        block = min(BLOCK_BUDGET, remaining)
        execution = block_execution_counts(action, block)
        result = advance_action(
            state,
            action,
            persistence,
            rho,
            delay,
            execution["updates"],
        )
        if execution["updates"] > 0:
            largest_radius = max(
                largest_radius, float(result["radius"])
            )
        state = result["state"]
        observed = block_observation_counts(
            action,
            execution["updates"],
            execution["leftover_observations"],
        )
        state = advance_observations(
            state,
            persistence,
            observed["pair_probes"],
            delay,
        )
        if observed["transition_trials"] > 0:
            stays += int(
                rng.binomial(
                    observed["transition_trials"], persistence
                )
            )
            transition_trials += observed["transition_trials"]
        if observed["sharing_trials"] > 0:
            shared += int(
                rng.binomial(observed["sharing_trials"], rho)
            )
            sharing_trials += observed["sharing_trials"]
        selected_q.append(int(action["num_agents"]))
        rows.append(
            {
                "policy": policy,
                "persistence": persistence,
                "rho": rho,
                "delay": delay,
                "seed": seed,
                "decision": decision,
                "covered_so_far": covered,
                "persistence_upper": bounds["persistence_upper"],
                "rho_upper": bounds["rho_upper"],
                "certified_persistence": bounds[
                    "certified_persistence"
                ],
                "certified_rho": (
                    bounds["certified_rho"]
                    if policy == "dual_ucb"
                    else 0.0
                ),
                "transition_trials": transition_trials,
                "sharing_trials": sharing_trials,
                "q": action["num_agents"],
                "gap": action["gap"],
                "eta": action["eta"],
                "updates": execution["updates"],
                "exact_radius": result["radius"],
            }
        )
        remaining -= block
    return (
        {
            "policy": policy,
            "persistence": persistence,
            "rho": rho,
            "delay": delay,
            "seed": seed,
            "simultaneous_coverage": covered,
            "largest_action_radius": largest_radius,
            "final_error": final_expected_error(state, delay),
            "median_q": float(np.median(selected_q)),
            "initial_p_upper": initial_p_upper,
            "final_p_upper": final_p_upper,
            "initial_rho_upper": initial_rho_upper,
            "final_rho_upper": final_rho_upper,
        },
        rows,
    )


def run_experiment(
    num_seeds: int, base_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cache: Dict[Tuple[object, ...], Dict[str, float]] = {}
    metrics: List[Dict[str, object]] = []
    actions: List[Dict[str, object]] = []
    scenario = 0
    for persistence in PERSISTENCES:
        for rho in CORRELATIONS:
            for delay in DELAYS:
                oracle = oracle_error(
                    persistence, rho, delay, cache
                )
                for index in range(num_seeds):
                    seed = base_seed + scenario * 10000 + index
                    for policy_index, policy in enumerate(
                        ("dual_ucb", "correlation_blind")
                    ):
                        row, block_rows = run_policy(
                            persistence,
                            rho,
                            delay,
                            seed + policy_index * 1000000,
                            policy,
                            cache,
                        )
                        row["random_seed"] = row["seed"]
                        row["seed"] = seed
                        row["oracle_error"] = oracle
                        row["to_oracle"] = row["final_error"] / oracle
                        metrics.append(row)
                        actions.extend(block_rows)
                scenario += 1
    return pd.DataFrame(metrics), pd.DataFrame(actions)


def evaluate(metrics: pd.DataFrame) -> Dict[str, object]:
    dual = metrics[metrics["policy"] == "dual_ucb"].copy()
    blind = metrics[metrics["policy"] == "correlation_blind"][
        ["persistence", "rho", "delay", "seed", "final_error"]
    ].rename(columns={"final_error": "blind_error"})
    paired = dual.merge(
        blind, on=["persistence", "rho", "delay", "seed"], how="left"
    )
    scenario = (
        paired.groupby(["persistence", "rho", "delay"], as_index=False)
        .agg(
            dual_error=("final_error", "median"),
            blind_error=("blind_error", "median"),
            oracle_ratio=("to_oracle", "median"),
            median_q=("median_q", "median"),
            initial_p_upper=("initial_p_upper", "median"),
            final_p_upper=("final_p_upper", "median"),
            initial_rho_upper=("initial_rho_upper", "median"),
            final_rho_upper=("final_rho_upper", "median"),
        )
    )
    covered = dual[dual["simultaneous_coverage"]]
    q_by_rho = scenario.groupby("rho")["median_q"].median()
    high = scenario[scenario["rho"] == 0.9]
    independent = scenario[scenario["rho"] == 0.0]
    persistent = scenario[scenario["persistence"] == 0.98]
    gates = {
        "joint_anytime_coverage": {
            "value": float(dual["simultaneous_coverage"].mean()),
            "pass": bool(
                dual["simultaneous_coverage"].mean() >= 0.975
            ),
        },
        "conditional_exact_safety": {
            "value": float(covered["largest_action_radius"].max()),
            "pass": bool(
                (covered["largest_action_radius"] < 1.0).all()
            ),
        },
        "participation_response": {
            "q_rho0": float(q_by_rho.loc[0.0]),
            "q_rho0p9": float(q_by_rho.loc[0.9]),
            "pass": bool(q_by_rho.loc[0.9] < q_by_rho.loc[0.0]),
        },
        "high_correlation_blind_advantage": {
            "improved_scenarios": int(
                (high["dual_error"] < high["blind_error"]).sum()
            ),
            "pass": bool(
                (high["dual_error"] < high["blind_error"]).sum() >= 5
            ),
        },
        "independent_data_efficiency": {
            "maximum_dual_blind_ratio": float(
                (independent["dual_error"] / independent["blind_error"]).max()
            ),
            "pass": bool(
                (
                    independent["dual_error"]
                    / independent["blind_error"]
                ).max()
                <= 1.5
            ),
        },
        "mixing_refinement": {
            "cells": int(
                (
                    persistent["final_p_upper"]
                    < persistent["initial_p_upper"]
                ).sum()
            ),
            "pass": bool(
                (
                    persistent["final_p_upper"]
                    < persistent["initial_p_upper"]
                ).sum()
                >= 5
            ),
        },
        "correlation_refinement": {
            "cells": int(
                (
                    scenario["final_rho_upper"]
                    < scenario["initial_rho_upper"]
                ).sum()
            ),
            "pass": bool(
                (
                    scenario["final_rho_upper"]
                    < scenario["initial_rho_upper"]
                ).sum()
                >= 12
            ),
        },
    }
    return {
        "gates": gates,
        "passes": int(sum(item["pass"] for item in gates.values())),
        "validity_pass": bool(
            gates["joint_anytime_coverage"]["pass"]
            and gates["conditional_exact_safety"]["pass"]
        ),
        "scientific_passes": int(
            sum(
                gates[name]["pass"]
                for name in (
                    "participation_response",
                    "high_correlation_blind_advantage",
                    "independent_data_efficiency",
                    "mixing_refinement",
                    "correlation_refinement",
                )
            )
        ),
        "overall_pass": bool(
            gates["joint_anytime_coverage"]["pass"]
            and gates["conditional_exact_safety"]["pass"]
            and sum(
                gates[name]["pass"]
                for name in (
                    "participation_response",
                    "high_correlation_blind_advantage",
                    "independent_data_efficiency",
                    "mixing_refinement",
                    "correlation_refinement",
                )
            )
            >= 4
        ),
        "scenario_summary": scenario.to_dict(orient="records"),
    }


def save_figures(
    metrics: pd.DataFrame, output_dir: Path
) -> None:
    scenario = (
        metrics.groupby(
            ["policy", "persistence", "rho", "delay"], as_index=False
        )
        .agg(error=("final_error", "median"), q=("median_q", "median"))
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    dual = scenario[scenario["policy"] == "dual_ucb"]
    for delay, marker in ((0, "o"), (2, "s")):
        line = (
            dual[dual["delay"] == delay]
            .groupby("rho", as_index=False)
            .agg(q=("q", "median"))
        )
        axes[0].plot(
            line["rho"], line["q"], marker=marker, label=f"$D={delay}$"
        )
    high = scenario[scenario["rho"] == 0.9]
    for policy, marker in (
        ("dual_ucb", "o"),
        ("correlation_blind", "s"),
    ):
        line = (
            high[high["policy"] == policy]
            .groupby("persistence", as_index=False)
            .agg(error=("error", "median"))
        )
        axes[1].plot(
            line["persistence"],
            line["error"],
            marker=marker,
            label=policy,
        )
    axes[0].set_xlabel("true correlation $\\rho$")
    axes[0].set_ylabel("median selected $q$")
    axes[1].set_xlabel("persistence")
    axes[1].set_ylabel("median exact error")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "dual_anytime_summary.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, actions = run_experiment(args.num_seeds, args.base_seed)
    evaluation = evaluate(metrics)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    actions.to_csv(args.output_dir / "actions.csv", index=False)
    save_figures(metrics, args.output_dir)
    summary = {
        "experiment": "EXP-011B-pilot"
        if args.num_seeds < 32
        else "EXP-011B",
        "num_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        **evaluation,
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

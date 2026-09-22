"""Run preregistered EXP-009D progressive anytime controller."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predictable_mixing_controller import (
    PILOT_ALPHA,
    RESOURCE_BUDGET,
    clopper_pearson_upper,
    exact_policy_metrics,
    select_joint_action,
)
from progressive_mixing_controller import (
    ANYTIME_ALPHA,
    BLOCK_BUDGET,
    INITIAL_PILOT,
    advance_action,
    advance_observations,
    block_execution_counts,
    final_expected_error,
    initial_covariance_state,
)
from run_predictable_mixing_controller import (
    CORRELATIONS,
    DELAYS,
    PERSISTENCES,
    SEEDS,
)


EXP009C_MAXIMUM_RATIO = 10.464074324341452


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "progressive_anytime_controller",
    )
    return parser.parse_args()


def rounded_upper(upper: float) -> float:
    return float(min(1.0, np.ceil(1000.0 * upper) / 1000.0))


def oracle_progressive(
    persistence: float,
    rho: float,
    delay: int,
    action_cache: Dict[Tuple[object, ...], Dict[str, float]],
) -> float:
    state = initial_covariance_state(delay)
    remaining = RESOURCE_BUDGET
    while remaining > 0:
        block = min(BLOCK_BUDGET, remaining)
        key = ("oracle", persistence, rho, delay, remaining)
        if key not in action_cache:
            action_cache[key] = select_joint_action(
                persistence,
                rho,
                delay,
                pilot_cost=0,
                resource_budget=remaining,
            )
        action = action_cache[key]
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


def run_experiment() -> tuple:
    metrics_rows: List[Dict[str, object]] = []
    action_rows: List[Dict[str, object]] = []
    action_cache: Dict[Tuple[object, ...], Dict[str, float]] = {}
    static_path = (
        Path(__file__).resolve().parent
        / "results"
        / "joint_qbe_controller"
        / "joint_qbe_runs.csv"
    )
    static = pd.read_csv(static_path)
    static = static[static["policy"] == "online_ucb"].set_index(
        ["persistence", "rho", "delay", "seed"]
    )
    total = (
        len(PERSISTENCES)
        * len(CORRELATIONS)
        * len(DELAYS)
        * len(SEEDS)
    )
    completed = 0
    scenario_index = 0
    for persistence in PERSISTENCES:
        for rho in CORRELATIONS:
            for delay in DELAYS:
                oracle_error = oracle_progressive(
                    persistence, rho, delay, action_cache
                )
                worst_action = select_joint_action(
                    0.98,
                    rho,
                    delay,
                    pilot_cost=0,
                    resource_budget=RESOURCE_BUDGET,
                )
                worst_error = exact_policy_metrics(
                    worst_action, persistence, rho, delay
                )["expected_final_error"]
                for seed in SEEDS:
                    rng = np.random.RandomState(
                        3_000_000 + scenario_index * 10_000 + seed
                    )
                    transitions = INITIAL_PILOT
                    stays = int(rng.binomial(INITIAL_PILOT, persistence))
                    state = initial_covariance_state(delay)
                    remaining = RESOURCE_BUDGET - INITIAL_PILOT
                    decision = 0
                    simultaneous_coverage = True
                    largest_safe_radius = 0.0
                    first_updating_gap = -1
                    final_gap = -1
                    selected_q = []
                    while remaining > 0:
                        decision += 1
                        alpha_m = ANYTIME_ALPHA / (
                            decision * (decision + 1)
                        )
                        upper = clopper_pearson_upper(
                            stays, transitions, alpha_m
                        )
                        simultaneous_coverage = bool(
                            simultaneous_coverage
                            and persistence <= upper
                        )
                        certified = rounded_upper(upper)
                        cache_key = (
                            "online",
                            certified,
                            rho,
                            delay,
                            remaining,
                        )
                        if cache_key not in action_cache:
                            action_cache[cache_key] = select_joint_action(
                                certified,
                                rho,
                                delay,
                                pilot_cost=0,
                                resource_budget=remaining,
                            )
                        action = action_cache[cache_key]
                        block = min(BLOCK_BUDGET, remaining)
                        counts = block_execution_counts(action, block)
                        result = advance_action(
                            state,
                            action,
                            persistence,
                            rho,
                            delay,
                            counts["updates"],
                        )
                        if counts["updates"] > 0:
                            largest_safe_radius = max(
                                largest_safe_radius,
                                float(result["radius"]),
                            )
                            if first_updating_gap < 0:
                                first_updating_gap = int(action["gap"])
                            final_gap = int(action["gap"])
                            selected_q.append(int(action["num_agents"]))
                        state = advance_observations(
                            result["state"],
                            persistence,
                            counts["leftover_observations"],
                            delay,
                        )
                        observations = counts[
                            "observation_transitions"
                        ]
                        if observations > 0:
                            stays += int(
                                rng.binomial(observations, persistence)
                            )
                            transitions += observations
                        action_rows.append(
                            {
                                "persistence": persistence,
                                "rho": rho,
                                "delay": delay,
                                "seed": seed,
                                "decision": decision,
                                "alpha_m": alpha_m,
                                "transitions_before": (
                                    transitions - observations
                                ),
                                "persistence_upper": upper,
                                "certified_upper": certified,
                                "covered_so_far": simultaneous_coverage,
                                "remaining_before": remaining,
                                "block_budget": block,
                                "gap": action["gap"],
                                "num_agents": action["num_agents"],
                                "eta": action["eta"],
                                "updates": counts["updates"],
                                "observations": observations,
                                "exact_radius": result["radius"],
                            }
                        )
                        remaining -= block
                    online_error = final_expected_error(state, delay)
                    static_error = float(
                        static.loc[
                            (persistence, rho, delay, seed),
                            "expected_final_error",
                        ]
                    )
                    metrics_rows.append(
                        {
                            "persistence": persistence,
                            "rho": rho,
                            "delay": delay,
                            "seed": seed,
                            "simultaneous_coverage": (
                                simultaneous_coverage
                            ),
                            "largest_action_radius": largest_safe_radius,
                            "online_error": online_error,
                            "oracle_error": oracle_error,
                            "static_error": static_error,
                            "worst_mixing_error": worst_error,
                            "online_to_oracle": (
                                online_error / oracle_error
                            ),
                            "first_updating_gap": first_updating_gap,
                            "final_gap": final_gap,
                            "median_selected_q": (
                                float(np.median(selected_q))
                                if selected_q
                                else 0.0
                            ),
                            "final_transition_count": transitions,
                        }
                    )
                    completed += 1
                    if completed % 64 == 0:
                        print(
                            "completed {0}/{1} scenario-seeds".format(
                                completed, total
                            ),
                            flush=True,
                        )
                scenario_index += 1
    return pd.DataFrame(metrics_rows), pd.DataFrame(action_rows)


def evaluate_gates(metrics: pd.DataFrame) -> Dict[str, object]:
    coverage = float(metrics["simultaneous_coverage"].mean())
    covered = metrics[metrics["simultaneous_coverage"]]
    scenario = (
        metrics.groupby(["persistence", "rho", "delay"], as_index=False)
        .agg(
            ratio=("online_to_oracle", "median"),
            online=("online_error", "median"),
            worst=("worst_mixing_error", "median"),
            first_gap=("first_updating_gap", "median"),
            final_gap=("final_gap", "median"),
            median_q=("median_selected_q", "median"),
        )
    )
    maximum_ratio = float(scenario["ratio"].max())
    improved_worst = int((scenario["online"] < scenario["worst"]).sum())
    persistent = scenario[np.isclose(scenario["persistence"], 0.98)]
    refined = bool((persistent["final_gap"] < persistent["first_gap"]).all())
    q_by_rho = metrics.groupby("rho")["median_selected_q"].median().to_dict()
    gates = {
        "anytime_coverage": {
            "pass": bool(coverage >= 0.985),
            "observed_simultaneous_coverage": coverage,
        },
        "conditional_exact_safety": {
            "pass": bool(
                (covered["largest_action_radius"] < 1.0).all()
            ),
            "covered_seeds": int(len(covered)),
            "largest_covered_action_radius": float(
                covered["largest_action_radius"].max()
            ),
        },
        "near_oracle_expected_risk": {
            "pass": bool(maximum_ratio <= 5.0),
            "largest_scenario_median_ratio": maximum_ratio,
        },
        "static_pilot_improvement": {
            "pass": bool(maximum_ratio < EXP009C_MAXIMUM_RATIO),
            "progressive_ratio": maximum_ratio,
            "static_exp009c_ratio": EXP009C_MAXIMUM_RATIO,
        },
        "worst_baseline_improvement": {
            "pass": bool(improved_worst >= 10),
            "improved_scenarios": improved_worst,
        },
        "progressive_refinement": {
            "pass": refined,
            "persistent_first_gaps": persistent[
                "first_gap"
            ].tolist(),
            "persistent_final_gaps": persistent["final_gap"].tolist(),
        },
        "participation_response": {
            "pass": bool(q_by_rho[0.9] <= q_by_rho[0.0]),
            "median_q_rho0": float(q_by_rho[0.0]),
            "median_q_rho0p9": float(q_by_rho[0.9]),
        },
    }
    return {
        "gates": gates,
        "overall_pass": bool(
            all(value["pass"] for value in gates.values())
        ),
        "scenario_summary": scenario.to_dict(orient="records"),
    }


def save_figures(
    metrics: pd.DataFrame, actions: pd.DataFrame, output_dir: Path
) -> None:
    scenario = (
        metrics.groupby(["persistence", "rho", "delay"], as_index=False)
        .agg(
            ratio=("online_to_oracle", "median"),
            online=("online_error", "median"),
            oracle=("oracle_error", "median"),
            static=("static_error", "median"),
        )
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for rho, marker in ((0.0, "o"), (0.9, "s")):
        for delay, linestyle in ((0, "-"), (2, "--")):
            line = scenario[
                np.isclose(scenario["rho"], rho)
                & (scenario["delay"] == delay)
            ].sort_values("persistence")
            label = rf"$\rho={rho:g},D={delay}$"
            axes[0].plot(
                line["persistence"],
                line["ratio"],
                marker=marker,
                linestyle=linestyle,
                label=label,
            )
            axes[1].plot(
                line["persistence"],
                line["online"] / line["static"],
                marker=marker,
                linestyle=linestyle,
            )
    axes[0].axhline(5.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("progressive / oracle expected error")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("progressive / static expected error")
    for axis in axes:
        axis.set_xlabel("persistence")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "fig1_progressive_ratios.png", dpi=220)
    plt.close(figure)

    high = actions[np.isclose(actions["persistence"], 0.98)]
    medians = (
        high.groupby(["rho", "delay", "decision"], as_index=False)[
            "gap"
        ]
        .median()
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for rho, marker in ((0.0, "o"), (0.9, "s")):
        for delay, linestyle in ((0, "-"), (2, "--")):
            line = medians[
                np.isclose(medians["rho"], rho)
                & (medians["delay"] == delay)
            ]
            axis.plot(
                line["decision"],
                line["gap"],
                marker=marker,
                linestyle=linestyle,
                label=rf"$\rho={rho:g},D={delay}$",
            )
    axis.set_xlabel("decision block")
    axis.set_ylabel("median certified gap")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "fig2_gap_refinement.png", dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, actions = run_experiment()
    evaluation = evaluate_gates(metrics)
    metrics.to_csv(args.output_dir / "progressive_metrics.csv", index=False)
    actions.to_csv(args.output_dir / "progressive_actions.csv", index=False)
    summary = {
        "experiment": "EXP-009D",
        "status": "PASS" if evaluation["overall_pass"] else "FAIL",
        "registered_seeds": int(len(metrics)),
        "registered_action_rows": int(len(actions)),
        **evaluation,
        "artifacts": [
            "progressive_metrics.csv",
            "progressive_actions.csv",
            "summary.json",
            "fig1_progressive_ratios.png",
            "fig2_gap_refinement.png",
        ],
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    save_figures(metrics, actions, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

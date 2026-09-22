"""Run preregistered EXP-009C joint q-gap-step controller."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predictable_mixing_controller import (
    PILOT_ALPHA,
    PILOT_TRANSITIONS,
    clopper_pearson_upper,
    exact_policy_metrics,
    select_joint_action,
)
from run_predictable_mixing_controller import (
    CORRELATIONS,
    DELAYS,
    PERSISTENCES,
    POLICIES,
    SEEDS,
)


EXP009B_MAXIMUM_RATIO = 11.634653512134712


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "joint_qbe_controller",
    )
    return parser.parse_args()


def fixed_actions(
    persistence: float, rho: float, delay: int
) -> Dict[str, Dict[str, float]]:
    selector = select_joint_action
    return {
        "oracle": selector(persistence, rho, delay, pilot_cost=0),
        "iid_naive": selector(0.5, rho, delay, pilot_cost=0),
        "worst_mixing": selector(0.98, rho, delay, pilot_cost=0),
        "oracle_q1": selector(
            persistence, rho, delay, pilot_cost=0, fixed_q=1
        ),
        "oracle_q32": selector(
            persistence, rho, delay, pilot_cost=0, fixed_q=32
        ),
    }


def run_experiment() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
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
                fixed = fixed_actions(persistence, rho, delay)
                fixed_metrics = {
                    policy: exact_policy_metrics(
                        action, persistence, rho, delay
                    )
                    for policy, action in fixed.items()
                }
                online_cache = {}
                for seed in SEEDS:
                    pilot_rng = np.random.RandomState(
                        1_000_000 + scenario_index * 10_000 + seed
                    )
                    stays = int(
                        pilot_rng.binomial(
                            PILOT_TRANSITIONS, persistence
                        )
                    )
                    if stays not in online_cache:
                        upper = clopper_pearson_upper(
                            stays, PILOT_TRANSITIONS, PILOT_ALPHA
                        )
                        action = select_joint_action(
                            upper,
                            rho,
                            delay,
                            pilot_cost=PILOT_TRANSITIONS,
                        )
                        metrics = exact_policy_metrics(
                            action, persistence, rho, delay
                        )
                        online_cache[stays] = (upper, action, metrics)
                    upper, online, online_metrics = online_cache[stays]
                    coverage = bool(persistence <= upper)
                    actions = {"online_ucb": online, **fixed}
                    metrics = {
                        "online_ucb": online_metrics,
                        **fixed_metrics,
                    }
                    for policy in POLICIES:
                        action = actions[policy]
                        rows.append(
                            {
                                "persistence": persistence,
                                "rho": rho,
                                "delay": delay,
                                "seed": seed,
                                "policy": policy,
                                "pilot_stays": (
                                    stays
                                    if policy == "online_ucb"
                                    else -1
                                ),
                                "persistence_upper": action[
                                    "persistence_upper"
                                ],
                                "certificate_covered": (
                                    coverage
                                    if policy == "online_ucb"
                                    else True
                                ),
                                "gap": action["gap"],
                                "delta_upper": action["delta_upper"],
                                "num_agents": action["num_agents"],
                                "eta": action["eta"],
                                "theorem_contraction": action[
                                    "contraction"
                                ],
                                "updates": action["updates"],
                                "risk_surrogate": action[
                                    "risk_surrogate"
                                ],
                                **metrics[policy],
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
    return pd.DataFrame(rows)


def evaluate_gates(frame: pd.DataFrame) -> Dict[str, object]:
    online = frame[frame["policy"] == "online_ucb"]
    coverage = float(online["certificate_covered"].mean())
    covered = online[online["certificate_covered"]]
    pivot = frame.pivot_table(
        index=["persistence", "rho", "delay", "seed"],
        columns="policy",
        values="expected_final_error",
    ).reset_index()
    pivot["online_to_oracle"] = (
        pivot["online_ucb"] / pivot["oracle"]
    )
    scenario_ratios = (
        pivot.groupby(["persistence", "rho", "delay"])[
            "online_to_oracle"
        ]
        .median()
        .reset_index()
    )
    maximum_ratio = float(scenario_ratios["online_to_oracle"].max())
    low = pivot[pivot["persistence"] <= 0.9]
    low_medians = (
        low.groupby(["persistence", "rho", "delay"])[
            ["online_ucb", "worst_mixing"]
        ]
        .median()
        .reset_index()
    )
    improved = int(
        (
            low_medians["online_ucb"]
            < low_medians["worst_mixing"]
        ).sum()
    )
    q_medians = online.groupby("rho")["num_agents"].median().to_dict()
    scenario_gaps = (
        online.groupby(["persistence", "rho", "delay"])["gap"]
        .median()
        .to_numpy()
    )
    distinct_gaps = int(len(np.unique(scenario_gaps)))
    gates = {
        "certificate_coverage": {
            "pass": bool(coverage >= 0.985),
            "observed_coverage": coverage,
        },
        "conditional_exact_safety": {
            "pass": bool((covered["exact_radius"] < 1.0).all()),
            "covered_runs": int(len(covered)),
            "largest_covered_radius": float(
                covered["exact_radius"].max()
            ),
        },
        "expected_oracle_competitiveness": {
            "pass": bool(maximum_ratio <= 5.0),
            "largest_scenario_median_ratio": maximum_ratio,
        },
        "robust_baseline_improvement": {
            "pass": bool(improved >= 6),
            "improved_scenarios": improved,
            "total_scenarios": int(len(low_medians)),
        },
        "participation_response": {
            "pass": bool(q_medians[0.9] <= q_medians[0.0]),
            "median_q_rho0": float(q_medians[0.0]),
            "median_q_rho0p9": float(q_medians[0.9]),
        },
        "exp009b_improvement": {
            "pass": bool(maximum_ratio < EXP009B_MAXIMUM_RATIO),
            "exp009c_largest_ratio": maximum_ratio,
            "exp009b_frozen_ratio": EXP009B_MAXIMUM_RATIO,
        },
        "gap_is_active": {
            "pass": bool(distinct_gaps >= 2),
            "distinct_scenario_median_gaps": distinct_gaps,
            "scenario_median_gaps": [
                float(value) for value in scenario_gaps
            ],
        },
    }
    return {
        "gates": gates,
        "overall_pass": bool(
            all(value["pass"] for value in gates.values())
        ),
        "scenario_ratios": scenario_ratios.to_dict(orient="records"),
        "low_persistence_medians": low_medians.to_dict(orient="records"),
    }


def save_figures(frame: pd.DataFrame, output_dir: Path) -> None:
    medians = (
        frame.groupby(
            ["persistence", "rho", "delay", "policy"], as_index=False
        )["expected_final_error"]
        .median()
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True)
    for axis, (rho, delay) in zip(
        axes.ravel(),
        [(0.0, 0), (0.9, 0), (0.0, 2), (0.9, 2)],
    ):
        subset = medians[
            np.isclose(medians["rho"], rho)
            & (medians["delay"] == delay)
        ]
        for policy in POLICIES:
            line = subset[subset["policy"] == policy].sort_values(
                "persistence"
            )
            axis.plot(
                line["persistence"],
                line["expected_final_error"],
                marker="o",
                label=policy if rho == 0.0 and delay == 0 else None,
            )
        axis.set_yscale("log")
        axis.set_title(rf"$\rho={rho:g},D={delay}$")
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("persistence")
    axes[1, 1].set_xlabel("persistence")
    axes[0, 0].set_ylabel("median exact expected error")
    axes[1, 0].set_ylabel("median exact expected error")
    axes[0, 0].legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(
        output_dir / "fig1_joint_policy_error.png", dpi=220
    )
    plt.close(figure)

    online = frame[frame["policy"] == "online_ucb"]
    actions = (
        online.groupby(["persistence", "rho", "delay"], as_index=False)
        .agg(
            median_gap=("gap", "median"),
            median_q=("num_agents", "median"),
            median_eta=("eta", "median"),
        )
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for rho, marker in ((0.0, "o"), (0.9, "s")):
        for delay, linestyle in ((0, "-"), (2, "--")):
            line = actions[
                np.isclose(actions["rho"], rho)
                & (actions["delay"] == delay)
            ].sort_values("persistence")
            label = rf"$\rho={rho:g},D={delay}$"
            for axis, column in zip(
                axes, ("median_gap", "median_q", "median_eta")
            ):
                axis.plot(
                    line["persistence"],
                    line[column],
                    marker=marker,
                    linestyle=linestyle,
                    label=label if axis is axes[0] else None,
                )
    for axis, label in zip(
        axes, ("selected gap", "selected q", "selected eta")
    ):
        axis.set_xlabel("persistence")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "fig2_joint_actions.png", dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = run_experiment()
    evaluation = evaluate_gates(frame)
    frame.to_csv(args.output_dir / "joint_qbe_runs.csv", index=False)
    summary = {
        "experiment": "EXP-009C",
        "status": "PASS" if evaluation["overall_pass"] else "FAIL",
        "registered_runs": int(len(frame)),
        **evaluation,
        "artifacts": [
            "joint_qbe_runs.csv",
            "summary.json",
            "fig1_joint_policy_error.png",
            "fig2_joint_actions.png",
        ],
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    save_figures(frame, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

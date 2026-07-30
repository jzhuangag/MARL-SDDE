"""Run preregistered EXP-009A predictable mixing controller."""

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
    PILOT_TRANSITIONS,
    clopper_pearson_upper,
    exact_policy_metrics,
    select_action,
    simulate_policy,
)


PERSISTENCES: Tuple[float, ...] = (0.5, 0.9, 0.98)
CORRELATIONS: Tuple[float, ...] = (0.0, 0.9)
DELAYS: Tuple[int, ...] = (0, 2)
SEEDS: Tuple[int, ...] = tuple(range(128))
POLICIES: Tuple[str, ...] = (
    "online_ucb",
    "oracle",
    "iid_naive",
    "worst_mixing",
    "oracle_q1",
    "oracle_q32",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "predictable_mixing_controller",
    )
    return parser.parse_args()


def frozen_actions(
    persistence: float, rho: float, delay: int
) -> Dict[str, Dict[str, float]]:
    return {
        "oracle": select_action(
            persistence, rho, delay, pilot_cost=0
        ),
        "iid_naive": select_action(
            0.5, rho, delay, pilot_cost=0
        ),
        "worst_mixing": select_action(
            0.98, rho, delay, pilot_cost=0
        ),
        "oracle_q1": select_action(
            persistence, rho, delay, pilot_cost=0, fixed_q=1
        ),
        "oracle_q32": select_action(
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
                fixed = frozen_actions(persistence, rho, delay)
                fixed_exact = {
                    policy: exact_policy_metrics(
                        action, persistence, rho, delay
                    )
                    for policy, action in fixed.items()
                }
                for seed in SEEDS:
                    pilot_rng = np.random.RandomState(
                        1_000_000 + scenario_index * 10_000 + seed
                    )
                    stays = int(
                        pilot_rng.binomial(
                            PILOT_TRANSITIONS, persistence
                        )
                    )
                    upper = clopper_pearson_upper(
                        stays, PILOT_TRANSITIONS, PILOT_ALPHA
                    )
                    coverage = bool(persistence <= upper)
                    online = select_action(
                        upper,
                        rho,
                        delay,
                        pilot_cost=PILOT_TRANSITIONS,
                    )
                    actions = {"online_ucb": online, **fixed}
                    exacts = {
                        "online_ucb": exact_policy_metrics(
                            online, persistence, rho, delay
                        ),
                        **fixed_exact,
                    }
                    simulation_seed = (
                        2_000_000 + scenario_index * 10_000 + seed
                    )
                    for policy in POLICIES:
                        action = actions[policy]
                        exact = exacts[policy]
                        simulation = simulate_policy(
                            simulation_seed,
                            action,
                            persistence,
                            rho,
                            delay,
                        )
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
                                **exact,
                                **simulation,
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
    coverage_rate = float(online["certificate_covered"].mean())
    covered = online[online["certificate_covered"]]
    conditional_safe = bool((covered["exact_radius"] < 1.0).all())
    online_trajectory_safe = bool((~online["diverged"]).all())
    high_naive = frame[
        (frame["policy"] == "iid_naive")
        & np.isclose(frame["persistence"], 0.98)
    ]
    naive_unstable_rate = float(
        (high_naive["exact_radius"] >= 1.0).mean()
    )

    paired = frame.pivot_table(
        index=["persistence", "rho", "delay", "seed"],
        columns="policy",
        values="final_error",
    ).reset_index()
    paired["online_to_oracle"] = (
        paired["online_ucb"] / paired["oracle"]
    )
    scenario_ratios = (
        paired.groupby(["persistence", "rho", "delay"])[
            "online_to_oracle"
        ]
        .median()
        .reset_index()
    )
    oracle_competitive = bool(
        (scenario_ratios["online_to_oracle"] <= 5.0).all()
    )

    low = paired[paired["persistence"] <= 0.9].copy()
    low["online_better_worst"] = (
        low["online_ucb"] < low["worst_mixing"]
    )
    low_scenarios = (
        low.groupby(["persistence", "rho", "delay"])[
            ["online_ucb", "worst_mixing"]
        ]
        .median()
        .reset_index()
    )
    improved_scenarios = int(
        (
            low_scenarios["online_ucb"]
            < low_scenarios["worst_mixing"]
        ).sum()
    )
    robust_improvement = bool(improved_scenarios >= 6)

    selected = (
        online.groupby(["rho"])["num_agents"].median().to_dict()
    )
    participation_response = bool(selected[0.9] <= selected[0.0])
    gates = {
        "certificate_coverage": {
            "pass": bool(coverage_rate >= 0.985),
            "observed_coverage": coverage_rate,
            "nominal_coverage": 1.0 - PILOT_ALPHA,
            "online_pilots": int(len(online)),
        },
        "conditional_exact_safety": {
            "pass": conditional_safe,
            "covered_runs": int(len(covered)),
            "largest_covered_exact_radius": float(
                covered["exact_radius"].max()
            ),
        },
        "online_trajectory_safety": {
            "pass": online_trajectory_safe,
            "divergences": int(online["diverged"].sum()),
        },
        "naive_failure_detectable": {
            "pass": bool(naive_unstable_rate >= 0.9),
            "unstable_rate_at_p0p98": naive_unstable_rate,
        },
        "oracle_competitiveness": {
            "pass": oracle_competitive,
            "largest_scenario_median_online_to_oracle": float(
                scenario_ratios["online_to_oracle"].max()
            ),
        },
        "robust_baseline_improvement": {
            "pass": robust_improvement,
            "improved_scenarios": improved_scenarios,
            "total_scenarios": int(len(low_scenarios)),
        },
        "participation_responds_to_correlation": {
            "pass": participation_response,
            "median_q_rho0": float(selected[0.0]),
            "median_q_rho0p9": float(selected[0.9]),
        },
    }
    return {
        "gates": gates,
        "overall_pass": bool(
            all(value["pass"] for value in gates.values())
        ),
        "scenario_ratios": scenario_ratios.to_dict(orient="records"),
        "low_persistence_medians": low_scenarios.to_dict(
            orient="records"
        ),
    }


def save_figures(frame: pd.DataFrame, output_dir: Path) -> None:
    medians = (
        frame.groupby(
            ["persistence", "rho", "delay", "policy"], as_index=False
        )["final_error"]
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
                line["final_error"],
                marker="o",
                label=policy if rho == 0.0 and delay == 0 else None,
            )
        axis.set_yscale("log")
        axis.set_title(rf"$\rho={rho:g},D={delay}$")
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("persistence")
    axes[1, 1].set_xlabel("persistence")
    axes[0, 0].set_ylabel("median final squared error")
    axes[1, 0].set_ylabel("median final squared error")
    axes[0, 0].legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "fig1_policy_error.png", dpi=220)
    plt.close(figure)

    online = frame[frame["policy"] == "online_ucb"]
    selected = (
        online.groupby(["persistence", "rho", "delay"], as_index=False)
        .agg(
            median_gap=("gap", "median"),
            median_q=("num_agents", "median"),
        )
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for rho, marker in ((0.0, "o"), (0.9, "s")):
        subset = selected[np.isclose(selected["rho"], rho)]
        for delay, linestyle in ((0, "-"), (2, "--")):
            line = subset[subset["delay"] == delay].sort_values(
                "persistence"
            )
            axes[0].plot(
                line["persistence"],
                line["median_gap"],
                marker=marker,
                linestyle=linestyle,
                label=rf"$\rho={rho:g},D={delay}$",
            )
            axes[1].plot(
                line["persistence"],
                line["median_q"],
                marker=marker,
                linestyle=linestyle,
            )
    axes[0].set_ylabel("median selected gap")
    axes[1].set_ylabel("median selected q")
    for axis in axes:
        axis.set_xlabel("persistence")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "fig2_online_actions.png", dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = run_experiment()
    evaluation = evaluate_gates(frame)
    frame.to_csv(args.output_dir / "controller_runs.csv", index=False)
    summary = {
        "experiment": "EXP-009A",
        "status": "PASS" if evaluation["overall_pass"] else "FAIL",
        "registered_runs": int(len(frame)),
        **evaluation,
        "artifacts": [
            "controller_runs.csv",
            "summary.json",
            "fig1_policy_error.png",
            "fig2_online_actions.png",
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

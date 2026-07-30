"""Run preregistered EXP-009B finite-budget safe controller."""

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
    select_finite_budget_action,
    simulate_policy,
)
from run_predictable_mixing_controller import (
    CORRELATIONS,
    DELAYS,
    PERSISTENCES,
    POLICIES,
    SEEDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "finite_budget_controller",
    )
    return parser.parse_args()


def frozen_actions(
    persistence: float, rho: float, delay: int
) -> Dict[str, Dict[str, float]]:
    selector = select_finite_budget_action
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
                    online = select_finite_budget_action(
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


def maximum_expected_oracle_ratio(frame: pd.DataFrame) -> tuple:
    pivot = frame.pivot_table(
        index=["persistence", "rho", "delay", "seed"],
        columns="policy",
        values="expected_final_error",
    ).reset_index()
    pivot["online_to_oracle"] = (
        pivot["online_ucb"] / pivot["oracle"]
    )
    scenarios = (
        pivot.groupby(["persistence", "rho", "delay"])[
            "online_to_oracle"
        ]
        .median()
        .reset_index()
    )
    return float(scenarios["online_to_oracle"].max()), pivot, scenarios


def exp009a_reference_ratio() -> float:
    path = (
        Path(__file__).resolve().parent
        / "results"
        / "predictable_mixing_controller"
        / "controller_runs.csv"
    )
    reference = pd.read_csv(path)
    maximum, _, _ = maximum_expected_oracle_ratio(reference)
    return maximum


def evaluate_gates(frame: pd.DataFrame) -> Dict[str, object]:
    online = frame[frame["policy"] == "online_ucb"]
    coverage_rate = float(online["certificate_covered"].mean())
    covered = online[online["certificate_covered"]]
    high_naive = frame[
        (frame["policy"] == "iid_naive")
        & np.isclose(frame["persistence"], 0.98)
    ]
    naive_unstable = float(
        (high_naive["exact_radius"] >= 1.0).mean()
    )
    maximum_ratio, pivot, scenario_ratios = (
        maximum_expected_oracle_ratio(frame)
    )
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
    reference_ratio = exp009a_reference_ratio()
    gates = {
        "certificate_coverage": {
            "pass": bool(coverage_rate >= 0.985),
            "observed_coverage": coverage_rate,
        },
        "conditional_exact_safety": {
            "pass": bool((covered["exact_radius"] < 1.0).all()),
            "covered_runs": int(len(covered)),
            "largest_covered_radius": float(
                covered["exact_radius"].max()
            ),
        },
        "online_trajectory_safety": {
            "pass": bool((~online["diverged"]).all()),
            "divergences": int(online["diverged"].sum()),
        },
        "naive_failure_detectable": {
            "pass": bool(naive_unstable >= 0.9),
            "unstable_rate_at_p0p98": naive_unstable,
        },
        "expected_oracle_competitiveness": {
            "pass": bool(maximum_ratio <= 5.0),
            "largest_scenario_median_ratio": maximum_ratio,
        },
        "expected_robust_baseline_improvement": {
            "pass": bool(improved >= 6),
            "improved_scenarios": improved,
            "total_scenarios": int(len(low_medians)),
        },
        "participation_response": {
            "pass": bool(q_medians[0.9] <= q_medians[0.0]),
            "median_q_rho0": float(q_medians[0.0]),
            "median_q_rho0p9": float(q_medians[0.9]),
        },
        "correction_efficacy": {
            "pass": bool(maximum_ratio < reference_ratio),
            "exp009b_largest_ratio": maximum_ratio,
            "exp009a_reference_largest_ratio": reference_ratio,
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
        output_dir / "fig1_expected_policy_error.png", dpi=220
    )
    plt.close(figure)

    online = frame[frame["policy"] == "online_ucb"]
    actions = (
        online.groupby(["persistence", "rho", "delay"], as_index=False)
        .agg(
            median_eta=("eta", "median"),
            median_q=("num_agents", "median"),
            median_gap=("gap", "median"),
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
            axes[0].plot(
                line["persistence"],
                line["median_gap"],
                marker=marker,
                linestyle=linestyle,
                label=label,
            )
            axes[1].plot(
                line["persistence"],
                line["median_q"],
                marker=marker,
                linestyle=linestyle,
            )
            axes[2].plot(
                line["persistence"],
                line["median_eta"],
                marker=marker,
                linestyle=linestyle,
            )
    for axis, label in zip(
        axes, ("selected gap", "selected q", "selected eta")
    ):
        axis.set_xlabel("persistence")
        axis.set_ylabel(label)
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
    frame.to_csv(args.output_dir / "finite_budget_runs.csv", index=False)
    summary = {
        "experiment": "EXP-009B",
        "status": "PASS" if evaluation["overall_pass"] else "FAIL",
        "registered_runs": int(len(frame)),
        **evaluation,
        "artifacts": [
            "finite_budget_runs.csv",
            "summary.json",
            "fig1_expected_policy_error.png",
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

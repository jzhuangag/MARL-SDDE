"""Run preregistered EXP-010B affine finite-time certificate audit."""

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affine_markov_certificate import (
    affine_candidate_actions,
    select_affine_action,
)
from multistate_certificate import (
    CORRELATIONS_TRANSFER,
    DELAYS_TRANSFER,
    PERSISTENCES,
    RESOURCE_BUDGET,
    build_transfer_mrp,
    generate_unit_paths,
    simulate_certified_action,
)
from run_state_correlation import bootstrap_ratio


POLICIES = ("joint", "q1", "q32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "affine_finite_time_certificate",
    )
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=20261130)
    parser.add_argument("--bootstrap-replications", type=int, default=2000)
    return parser.parse_args()


def build_actions() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[float, dict]]:
    candidate_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    models = {}
    for persistence in PERSISTENCES:
        model = build_transfer_mrp(persistence)
        models[persistence] = model
        for maximum_delay in DELAYS_TRANSFER:
            for rho in CORRELATIONS_TRANSFER:
                candidates = affine_candidate_actions(
                    model, rho, maximum_delay
                )
                candidate_rows.extend(
                    {
                        "persistence": persistence,
                        **row,
                    }
                    for row in candidates
                )
                choices = {
                    "joint": select_affine_action(candidates),
                    "q1": select_affine_action(
                        candidates, restricted_q=1
                    ),
                    "q32": select_affine_action(
                        candidates, restricted_q=32
                    ),
                }
                for policy, action in choices.items():
                    selected_rows.append(
                        {
                            "persistence": persistence,
                            "policy": policy,
                            **action,
                        }
                    )
    return (
        pd.DataFrame(candidate_rows),
        pd.DataFrame(selected_rows),
        models,
    )


def run_simulations(
    selected: pd.DataFrame,
    models: Dict[float, dict],
    num_seeds: int,
    base_seed: int,
) -> pd.DataFrame:
    lookup = {
        (
            float(row["persistence"]),
            float(row["rho"]),
            int(row["maximum_delay"]),
            str(row["policy"]),
        ): row
        for row in selected.to_dict("records")
    }
    rows = []
    for persistence in PERSISTENCES:
        model = models[persistence]
        for seed_index in range(num_seeds):
            seed = base_seed + seed_index
            streams = generate_unit_paths(seed, model)
            for maximum_delay in DELAYS_TRANSFER:
                for rho in CORRELATIONS_TRANSFER:
                    for policy in POLICIES:
                        action = lookup[
                            (persistence, rho, maximum_delay, policy)
                        ]
                        result = simulate_certified_action(
                            streams, model, action
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "seed_index": seed_index,
                                "persistence": persistence,
                                "rho": rho,
                                "maximum_delay": maximum_delay,
                                "policy": policy,
                                "num_agents": int(action["num_agents"]),
                                "gap": int(action["gap"]),
                                "eta": float(action["eta"]),
                                "updates": int(action["updates"]),
                                "finite_time_bound": float(
                                    action["finite_time_bound"]
                                ),
                                **result,
                            }
                        )
            if (seed_index + 1) % 4 == 0:
                print(
                    "persistence {0:g}: completed {1}/{2} paired seeds".format(
                        persistence, seed_index + 1, num_seeds
                    ),
                    flush=True,
                )
    return pd.DataFrame(rows)


def one_sided_bootstrap_upper(
    values: np.ndarray, replications: int, seed: int
) -> float:
    rng = np.random.RandomState(seed)
    count = len(values)
    indices = rng.randint(0, count, size=(replications, count))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 99.0))


def build_calibration(
    metrics: pd.DataFrame, replications: int, base_seed: int
) -> pd.DataFrame:
    rows = []
    keys = ["persistence", "rho", "maximum_delay", "policy"]
    for index, (key, group) in enumerate(metrics.groupby(keys, sort=True)):
        values = group.sort_values("seed_index")[
            "squared_parameter_error"
        ].to_numpy(dtype=float)
        bound = float(group["finite_time_bound"].iloc[0])
        mean = float(values.mean())
        rows.append(
            {
                "persistence": key[0],
                "rho": key[1],
                "maximum_delay": key[2],
                "policy": key[3],
                "mean_error": mean,
                "bootstrap_99_upper_mean": one_sided_bootstrap_upper(
                    values, replications, base_seed + 20_000 + index
                ),
                "finite_time_bound": bound,
                "bound_to_mean_ratio": bound
                / max(mean, np.finfo(float).eps),
            }
        )
    return pd.DataFrame(rows)


def build_comparisons(
    metrics: pd.DataFrame, replications: int, base_seed: int
) -> pd.DataFrame:
    rows = []
    keys = ["persistence", "rho", "maximum_delay"]
    for scenario_index, (key, group) in enumerate(
        metrics.groupby(keys, sort=True)
    ):
        values = {
            policy: group[group["policy"] == policy]
            .sort_values("seed_index")["squared_parameter_error"]
            .to_numpy(dtype=float)
            for policy in POLICIES
        }
        for baseline_index, baseline in enumerate(("q1", "q32")):
            rows.append(
                {
                    "persistence": key[0],
                    "rho": key[1],
                    "maximum_delay": key[2],
                    "numerator": "joint",
                    "denominator": baseline,
                    **bootstrap_ratio(
                        values["joint"],
                        values[baseline],
                        replications,
                        base_seed
                        + 30_000
                        + 10 * scenario_index
                        + baseline_index,
                    ),
                }
            )
    return pd.DataFrame(rows)


def evaluate_gates(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    num_seeds: int,
) -> Dict[str, object]:
    delayed = candidates[candidates["rms_delay"] > 0.0]
    delayed_direct = (
        np.sqrt(delayed["a_delta"].to_numpy())
        + np.sqrt(delayed["h_delay"].to_numpy())
    ) ** 2
    delayed_difference = float(
        np.max(
            np.abs(
                delayed_direct
                - delayed["contraction"].to_numpy()
            )
        )
    )
    zero = candidates[np.isclose(candidates["rms_delay"], 0.0)]
    zero_difference = float(
        np.max(
            np.abs(
                zero["a_delta"].to_numpy()
                - zero["contraction"].to_numpy()
            )
        )
    )
    numerical = {
        "certificate_validity": {
            "pass": bool(
                (selected["a_delta"] > 0.0).all()
                and (selected["contraction"] > 0.0).all()
                and (selected["contraction"] < 1.0).all()
                and np.isfinite(
                    selected[
                        [
                            "forcing",
                            "residual",
                            "finite_time_bound",
                        ]
                    ].to_numpy()
                ).all()
                and (selected["effective_monotonicity"] > 0.0).all()
            ),
            "maximum_contraction": float(
                selected["contraction"].max()
            ),
            "minimum_effective_monotonicity": float(
                selected["effective_monotonicity"].min()
            ),
        },
        "execution_validity": {
            "pass": bool(
                len(metrics) == num_seeds * 12 * len(POLICIES)
                and metrics["finite"].all()
                and metrics["within_budget"].all()
                and not metrics["diverged"].any()
            ),
            "rows": int(len(metrics)),
            "expected_rows": int(num_seeds * 12 * len(POLICIES)),
            "divergences": int(metrics["diverged"].sum()),
        },
        "algebraic_reproduction": {
            "pass": bool(
                delayed_difference <= 1e-12
                and zero_difference <= 1e-12
            ),
            "maximum_delayed_difference": delayed_difference,
            "maximum_zero_delay_difference": zero_difference,
        },
    }
    joint = selected[selected["policy"] == "joint"].copy()
    nonvacuous = int(
        (joint["finite_time_bound"] < joint["initial_error"]).sum()
    )
    pivot_q = joint.pivot(
        index=["persistence", "maximum_delay"],
        columns="rho",
        values="num_agents",
    )
    weak_q = int((pivot_q[0.9] <= pivot_q[0.0]).sum())
    strict_q = int((pivot_q[0.9] < pivot_q[0.0]).sum())
    gaps = joint.groupby("persistence")["gap"].median()
    pivot_eta = joint.pivot(
        index=["persistence", "rho"],
        columns="maximum_delay",
        values="eta",
    )
    delay_nonincrease = int((pivot_eta[8] <= pivot_eta[0] + 1e-15).sum())
    joint_calibration = calibration[calibration["policy"] == "joint"]
    covered = int(
        (
            joint_calibration["bootstrap_99_upper_mean"]
            <= joint_calibration["finite_time_bound"]
        ).sum()
    )
    informative = int(
        (joint_calibration["bound_to_mean_ratio"] <= 1e3).sum()
    )
    scientific = {
        "finite_time_nonvacuity": {
            "pass": bool(
                nonvacuous >= 8 and (joint["updates"] >= 50).all()
            ),
            "useful_scenarios": nonvacuous,
            "minimum_updates": int(joint["updates"].min()),
        },
        "correlation_response": {
            "pass": bool(weak_q == 6 and strict_q >= 4),
            "weak_decreases": weak_q,
            "strict_decreases": strict_q,
        },
        "mixing_response": {
            "pass": bool(gaps.loc[0.98] >= 4.0 * gaps.loc[0.0]),
            "median_gap_kappa_0": float(gaps.loc[0.0]),
            "median_gap_kappa_098": float(gaps.loc[0.98]),
            "ratio": float(gaps.loc[0.98] / gaps.loc[0.0]),
        },
        "delay_response": {
            "pass": bool(delay_nonincrease >= 5),
            "nonincreasing_cells": delay_nonincrease,
            "total_cells": 6,
        },
        "empirical_upper_calibration": {
            "pass": bool(covered == 12),
            "covered_scenarios": covered,
            "total_scenarios": 12,
            "maximum_upper_to_bound": float(
                (
                    joint_calibration["bootstrap_99_upper_mean"]
                    / joint_calibration["finite_time_bound"]
                ).max()
            ),
        },
        "bound_informativeness": {
            "pass": bool(informative >= 9),
            "informative_scenarios": informative,
            "total_scenarios": 12,
            "maximum_bound_to_mean_ratio": float(
                joint_calibration["bound_to_mean_ratio"].max()
            ),
            "median_bound_to_mean_ratio": float(
                joint_calibration["bound_to_mean_ratio"].median()
            ),
        },
    }
    numerical_passes = sum(int(row["pass"]) for row in numerical.values())
    scientific_passes = sum(int(row["pass"]) for row in scientific.values())
    return {
        "numerical": numerical,
        "scientific": scientific,
        "overall": {
            "pass": bool(numerical_passes == 3 and scientific_passes >= 5),
            "numerical_passes": numerical_passes,
            "scientific_passes": scientific_passes,
            "criterion": "3/3 numerical and at least 5/6 scientific gates",
        },
    }


def plot_outputs(
    selected: pd.DataFrame,
    calibration: pd.DataFrame,
    output_dir: Path,
) -> None:
    joint = selected[selected["policy"] == "joint"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for maximum_delay in DELAYS_TRANSFER:
        for rho in CORRELATIONS_TRANSFER:
            curve = joint[
                (joint["maximum_delay"] == maximum_delay)
                & np.isclose(joint["rho"], rho)
            ]
            label = "D={0}, rho={1:g}".format(maximum_delay, rho)
            axes[0].plot(
                curve["persistence"],
                curve["num_agents"],
                marker="o",
                label=label,
            )
            axes[1].plot(
                curve["persistence"],
                curve["gap"],
                marker="o",
                label=label,
            )
    axes[0].set_yscale("log", base=2)
    axes[0].set_yticks((1, 2, 4, 8, 16, 32))
    axes[0].get_yaxis().set_major_formatter(
        matplotlib.ticker.ScalarFormatter()
    )
    axes[0].set_xlabel("Laziness kappa")
    axes[0].set_ylabel("Affine-certified q")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Laziness kappa")
    axes[1].set_ylabel("Certified separation b")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_affine_actions.png", dpi=220)
    plt.close(fig)

    joint_calibration = calibration[calibration["policy"] == "joint"]
    x = np.arange(len(joint_calibration))
    fig, ax = plt.subplots(figsize=(11.0, 4.5))
    ax.bar(
        x - 0.2,
        joint_calibration["mean_error"],
        0.4,
        label="empirical mean",
    )
    ax.bar(
        x + 0.2,
        joint_calibration["finite_time_bound"],
        0.4,
        label="proved bound",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "{0:g}/{1:g}/{2}".format(
                row.persistence, row.rho, int(row.maximum_delay)
            )
            for row in joint_calibration.itertuples()
        ],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("kappa / rho / D")
    ax.set_ylabel("Squared parameter error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_bound_calibration.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Building affine finite-time certificates", flush=True)
    candidates, selected, models = build_actions()
    print(
        "Starting EXP-010B with {0} paired seeds".format(args.num_seeds),
        flush=True,
    )
    metrics = run_simulations(
        selected, models, args.num_seeds, args.base_seed
    )
    calibration = build_calibration(
        metrics, args.bootstrap_replications, args.base_seed
    )
    comparisons = build_comparisons(
        metrics, args.bootstrap_replications, args.base_seed
    )
    gates = evaluate_gates(
        candidates, selected, metrics, calibration, args.num_seeds
    )
    candidates.to_csv(args.output_dir / "candidate_actions.csv", index=False)
    selected.to_csv(args.output_dir / "selected_actions.csv", index=False)
    metrics.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    calibration.to_csv(args.output_dir / "bound_calibration.csv", index=False)
    comparisons.to_csv(
        args.output_dir / "paired_bootstrap_comparisons.csv", index=False
    )
    plot_outputs(selected, calibration, args.output_dir)
    summary = {
        "experiment_id": "EXP-010B-affine-finite-time-certificate",
        "status": "COMPLETED_PENDING_REPRODUCTION",
        "num_paired_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        "bootstrap_replications": args.bootstrap_replications,
        "resource_budget": RESOURCE_BUDGET,
        "gates": gates,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "artifacts": [
            "candidate_actions.csv",
            "selected_actions.csv",
            "per_seed_metrics.csv",
            "bound_calibration.csv",
            "paired_bootstrap_comparisons.csv",
            "fig1_affine_actions.png",
            "fig2_bound_calibration.png",
            "summary.json",
        ],
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(gates, indent=2), flush=True)
    print("Outputs written to {0}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()

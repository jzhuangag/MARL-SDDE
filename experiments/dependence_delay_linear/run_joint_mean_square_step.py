"""Run EXP-007C joint correlation--delay mean-square step experiment."""

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

from joint_mean_square_step import (
    AGENT_COUNTS_JOINT,
    CHECKPOINTS,
    CORRELATIONS_JOINT,
    GRID_STEPS,
    MAX_DELAYS_JOINT,
    POLICIES,
    build_mean_boundaries,
    joint_step_size,
    multiplicative_curvature,
    registered_policy_steps,
    sharing_factor,
    simulate_checkpoint_run,
    single_jacobian_second_moment,
    strong_monotonicity,
)
from linear_model import make_agent_delays
from linear_td_correlation import (
    LinearTDConfig,
    build_mrp,
    generate_base_paths,
    observed_transition_pairs,
)
from td_delay_stability import DIVERGENCE_THRESHOLD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "joint_mean_square_step",
    )
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=20261230)
    return parser.parse_args()


def moment_table(
    mrp: Dict[str, np.ndarray],
    second_moment: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    a_matrix = mrp["a_matrix"]
    mu = strong_monotonicity(a_matrix)
    for num_agents in AGENT_COUNTS_JOINT:
        for rho in (0.0, 0.9, 1.0):
            curvature = multiplicative_curvature(
                a_matrix, second_moment, num_agents, rho
            )
            rows.append(
                {
                    "num_agents": num_agents,
                    "rho": rho,
                    "sharing_factor": sharing_factor(num_agents, rho),
                    "multiplicative_curvature": curvature,
                    "iid_second_moment_threshold": 2.0 * mu / curvature,
                }
            )
    return pd.DataFrame(rows)


def policy_step_table(
    mrp: Dict[str, np.ndarray],
    second_moment: np.ndarray,
    boundaries: Dict[Tuple[int, int], float],
    config: LinearTDConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    a_matrix = mrp["a_matrix"]
    mu = strong_monotonicity(a_matrix)
    for num_agents in AGENT_COUNTS_JOINT:
        for max_delay in MAX_DELAYS_JOINT:
            full_delays = make_agent_delays(
                max_agents=config.num_agents,
                max_delay=max_delay,
                exponent=config.delay_exponent,
            )
            selected_max = int(np.max(full_delays[:num_agents]))
            for rho in CORRELATIONS_JOINT:
                curvature = multiplicative_curvature(
                    a_matrix, second_moment, num_agents, rho
                )
                steps = registered_policy_steps(
                    a_matrix,
                    second_moment,
                    boundaries,
                    num_agents,
                    max_delay,
                    rho,
                )
                for policy, eta in steps.items():
                    rows.append(
                        {
                            "rho": rho,
                            "num_agents": num_agents,
                            "max_delay": max_delay,
                            "selected_max_delay": selected_max,
                            "policy": policy,
                            "eta": eta,
                            "mean_boundary": boundaries[
                                (num_agents, max_delay)
                            ],
                            "multiplicative_curvature": curvature,
                            "strong_monotonicity": mu,
                        }
                    )
    return pd.DataFrame(rows)


def run_experiment(
    config: LinearTDConfig,
    num_seeds: int,
    base_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mrp = build_mrp(config)
    second_moment = single_jacobian_second_moment(mrp, config)
    boundaries = build_mean_boundaries(mrp["a_matrix"], config)
    moments = moment_table(mrp, second_moment)
    policy_steps = policy_step_table(
        mrp, second_moment, boundaries, config
    )
    step_lookup = {
        (
            float(row.rho),
            int(row.num_agents),
            int(row.max_delay),
            str(row.policy),
        ): float(row.eta)
        for row in policy_steps.itertuples()
    }
    rows: List[Dict[str, object]] = []
    for seed_index in range(num_seeds):
        seed = base_seed + seed_index
        paths = generate_base_paths(seed, mrp, config)
        for rho in CORRELATIONS_JOINT:
            current, following = observed_transition_pairs(paths, rho)
            for num_agents in AGENT_COUNTS_JOINT:
                for max_delay in MAX_DELAYS_JOINT:
                    registered = [
                        (
                            policy,
                            step_lookup[
                                (rho, num_agents, max_delay, policy)
                            ],
                            float("nan"),
                        )
                        for policy in POLICIES
                    ]
                    grid = [
                        ("grid", float(eta), int(index))
                        for index, eta in enumerate(GRID_STEPS)
                    ]
                    for policy, eta, grid_index in registered + grid:
                        result = simulate_checkpoint_run(
                            current,
                            following,
                            mrp,
                            max_delay,
                            num_agents,
                            eta,
                            config,
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "seed_index": seed_index,
                                "rho": rho,
                                "num_agents": num_agents,
                                "max_delay": max_delay,
                                "policy": policy,
                                "grid_index": grid_index,
                                "eta": eta,
                                **result,
                            }
                        )
        if (seed_index + 1) % 4 == 0:
            print(
                "completed {0}/{1} paired seeds".format(
                    seed_index + 1, num_seeds
                ),
                flush=True,
            )
    return pd.DataFrame(rows), moments, policy_steps


def oracle_tightness_table(runs: pd.DataFrame) -> pd.DataFrame:
    cell_columns = ["rho", "num_agents", "max_delay"]
    grid = runs[runs["policy"] == "grid"]
    rates = (
        grid.groupby(cell_columns + ["eta"], as_index=False)[
            "crossed_threshold"
        ]
        .mean()
        .rename(columns={"crossed_threshold": "crossing_rate"})
    )
    rows: List[Dict[str, object]] = []
    aware = runs[runs["policy"] == "joint_aware"]
    for key, group in rates.groupby(cell_columns):
        safe = group[group["crossing_rate"] <= 0.05]
        largest_safe = (
            float(safe["eta"].max()) if len(safe) else float("nan")
        )
        mask = np.ones(len(aware), dtype=bool)
        for column, value in zip(cell_columns, key):
            mask &= np.isclose(aware[column].to_numpy(dtype=float), value)
        eta = float(aware.loc[mask, "eta"].iloc[0])
        rows.append(
            {
                "rho": float(key[0]),
                "num_agents": int(key[1]),
                "max_delay": int(key[2]),
                "joint_eta": eta,
                "largest_safe_grid_eta": largest_safe,
                "joint_to_safe_grid_ratio": eta / largest_safe,
            }
        )
    return pd.DataFrame(rows)


def evaluate_gates(
    runs: pd.DataFrame,
    moments: pd.DataFrame,
    tightness: pd.DataFrame,
    num_seeds: int,
) -> Dict[str, object]:
    curvature = moments.pivot(
        index="num_agents",
        columns="rho",
        values="multiplicative_curvature",
    )
    independent_drop = float(
        1.0 - curvature.loc[32, 0.0] / curvature.loc[16, 0.0]
    )
    correlated_drop = float(
        1.0 - curvature.loc[32, 0.9] / curvature.loc[16, 0.9]
    )
    inflation = {
        int(q): float(curvature.loc[q, 0.9] / curvature.loc[q, 0.0])
        for q in AGENT_COUNTS_JOINT
    }
    saturation = bool(
        independent_drop >= 0.20
        and correlated_drop <= 0.02
        and all(value >= 5.0 for value in inflation.values())
    )
    aware = runs[runs["policy"] == "joint_aware"]
    aware_medians = aware.groupby(
        ["rho", "num_agents", "max_delay"]
    )["final_error"].median()
    aware_safe = bool(
        (~aware["crossed_threshold"]).all()
        and (aware_medians < 1.0).all()
    )
    correlation_blind = runs[
        (runs["policy"] == "correlation_blind")
        & np.isclose(runs["rho"], 0.9)
    ]
    correlation_blind_crossing = float(
        correlation_blind["crossed_threshold"].mean()
    )
    correlation_value = bool(
        correlation_blind_crossing >= 0.25
        and (~aware[np.isclose(aware["rho"], 0.9)][
            "crossed_threshold"
        ]).all()
    )
    delay_blind = runs[
        (runs["policy"] == "delay_blind")
        & (runs["num_agents"] == 32)
        & (runs["max_delay"] == 32)
    ]
    delay_blind_crossing = float(
        delay_blind["crossed_threshold"].mean()
    )
    delay_value = bool(
        delay_blind_crossing >= 0.25
        and (~aware[
            (aware["num_agents"] == 32)
            & (aware["max_delay"] == 32)
        ]["crossed_threshold"]).all()
    )
    tight_cells = int(
        (tightness["joint_to_safe_grid_ratio"] >= 0.25).sum()
    )
    independent_aware = aware[np.isclose(aware["rho"], 0.0)].copy()
    worst = runs[
        (runs["policy"] == "worstcase_correlation")
        & np.isclose(runs["rho"], 0.0)
    ].copy()
    independent_aware["effective_half_time"] = independent_aware[
        "half_error_time"
    ].where(independent_aware["half_error_time"] > 0, 4001)
    worst["effective_half_time"] = worst["half_error_time"].where(
        worst["half_error_time"] > 0, 4001
    )
    aware_half = independent_aware.groupby(
        ["num_agents", "max_delay"]
    )["effective_half_time"].median()
    worst_half = worst.groupby(
        ["num_agents", "max_delay"]
    )["effective_half_time"].median()
    faster_cells = int((aware_half <= worst_half).sum())
    nonvacuous = bool(tight_cells >= 6 and faster_cells >= 3)
    expected = (
        num_seeds
        * len(CORRELATIONS_JOINT)
        * len(AGENT_COUNTS_JOINT)
        * len(MAX_DELAYS_JOINT)
        * (len(POLICIES) + len(GRID_STEPS))
    )
    checkpoint_columns = [
        "error_{0}".format(int(checkpoint)) for checkpoint in CHECKPOINTS
    ]
    validity = bool(
        len(runs) == expected
        and runs["finite"].all()
        and np.isfinite(runs[checkpoint_columns].to_numpy()).all()
        and (
            (runs["crossed_threshold"] & (runs["crossing_time"] > 0))
            | (~runs["crossed_threshold"] & (runs["crossing_time"] == -1))
        ).all()
        and len(tightness) == 8
    )
    gates: Dict[str, object] = {
        "analytic_correlation_saturation": {
            "pass": saturation,
            "independent_curvature_drop_q16_to_q32": independent_drop,
            "rho0p9_curvature_drop_q16_to_q32": correlated_drop,
            "rho0p9_to_rho0_curvature_inflation": inflation,
        },
        "joint_safety_and_contraction": {
            "pass": aware_safe,
            "crossing_fraction": float(
                aware["crossed_threshold"].mean()
            ),
            "largest_cell_median_final_error": float(
                aware_medians.max()
            ),
        },
        "correlation_awareness_value": {
            "pass": correlation_value,
            "correlation_blind_rho0p9_crossing_fraction": (
                correlation_blind_crossing
            ),
            "joint_rho0p9_crossing_fraction": float(
                aware[np.isclose(aware["rho"], 0.9)][
                    "crossed_threshold"
                ].mean()
            ),
        },
        "delay_awareness_value": {
            "pass": delay_value,
            "delay_blind_q32_d32_crossing_fraction": (
                delay_blind_crossing
            ),
            "joint_q32_d32_crossing_fraction": float(
                aware[
                    (aware["num_agents"] == 32)
                    & (aware["max_delay"] == 32)
                ]["crossed_threshold"].mean()
            ),
        },
        "nonvacuous_tightness_and_speed": {
            "pass": nonvacuous,
            "tight_cells_at_least_quarter_oracle": tight_cells,
            "independent_cells_no_slower_than_worstcase": faster_cells,
        },
        "accounting_determinism_numerical_validity": {
            "pass": validity,
            "observed_runs": int(len(runs)),
            "expected_runs": int(expected),
            "divergence_threshold": DIVERGENCE_THRESHOLD,
        },
    }
    gates["overall"] = {
        "pass": bool(all(value["pass"] for value in gates.values())),
        "criterion": "all six joint mean-square gates pass",
    }
    return gates


def plot_outputs(
    runs: pd.DataFrame,
    moments: pd.DataFrame,
    policy_steps: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for rho in (0.0, 0.9):
        selected = moments[np.isclose(moments["rho"], rho)]
        ax.plot(
            selected["num_agents"],
            selected["multiplicative_curvature"],
            marker="o",
            label="rho={0}".format(rho),
        )
    ax.set_xlabel("Participating agents q")
    ax.set_ylabel("Multiplicative curvature K(q,rho)")
    ax.set_title("Correlation saturates the Jacobian second moment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_multiplicative_curvature.png")
    plt.close(fig)

    policy_subset = runs[runs["policy"].isin(POLICIES)]
    crossing = (
        policy_subset.groupby(["rho", "policy"], as_index=False)[
            "crossed_threshold"
        ]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    policies = list(POLICIES)
    x = np.arange(len(policies))
    width = 0.34
    for index, rho in enumerate(CORRELATIONS_JOINT):
        selected = crossing[np.isclose(crossing["rho"], rho)].set_index(
            "policy"
        )
        values = [float(selected.loc[p, "crossed_threshold"]) for p in policies]
        ax.bar(
            x + (index - 0.5) * width,
            values,
            width=width,
            label="rho={0}".format(rho),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=24, ha="right")
    ax.set_ylabel("Threshold-crossing fraction")
    ax.set_title("Joint, blind, and mean-only stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_policy_crossings.png")
    plt.close(fig)

    selected = policy_subset[
        (policy_subset["num_agents"] == 32)
        & (policy_subset["max_delay"] == 32)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, rho in zip(axes, CORRELATIONS_JOINT):
        cell = selected[np.isclose(selected["rho"], rho)]
        for policy in POLICIES:
            policy_rows = cell[cell["policy"] == policy]
            medians = [
                float(policy_rows["error_{0}".format(int(point))].median())
                for point in CHECKPOINTS
            ]
            ax.plot(CHECKPOINTS, medians, marker="o", label=policy)
        ax.set_yscale("log")
        ax.set_title("rho={0}".format(rho))
        ax.set_xlabel("Update")
    axes[0].set_ylabel("Median squared parameter error")
    axes[1].legend(fontsize=7)
    fig.suptitle("q=32, D=32 checkpoint trajectories")
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_checkpoint_trajectories.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = LinearTDConfig()
    runs, moments, policy_steps = run_experiment(
        config, args.num_seeds, args.base_seed
    )
    tightness = oracle_tightness_table(runs)
    gates = evaluate_gates(
        runs, moments, tightness, args.num_seeds
    )
    runs.to_csv(args.output_dir / "joint_step_runs.csv", index=False)
    moments.to_csv(args.output_dir / "second_moments.csv", index=False)
    policy_steps.to_csv(args.output_dir / "policy_steps.csv", index=False)
    tightness.to_csv(args.output_dir / "oracle_tightness.csv", index=False)
    plot_outputs(runs, moments, policy_steps, args.output_dir)
    summary = {
        "experiment_id": "EXP-007C-joint-mean-square-step",
        "status": "PASS" if gates["overall"]["pass"] else "FAIL",
        "config": {
            "num_seeds": args.num_seeds,
            "base_seed": args.base_seed,
            "agent_counts": list(AGENT_COUNTS_JOINT),
            "max_delays": list(MAX_DELAYS_JOINT),
            "correlations": list(CORRELATIONS_JOINT),
            "grid_steps": [float(value) for value in GRID_STEPS],
            "checkpoints": [int(value) for value in CHECKPOINTS],
        },
        "gates": gates,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


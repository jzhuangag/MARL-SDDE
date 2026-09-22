"""Run EXP-007D fresh-seed mean-square confirmation."""

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from joint_mean_square_step import (
    AGENT_COUNTS_JOINT,
    CHECKPOINTS,
    CORRELATIONS_JOINT,
    GRID_STEPS,
    MAX_DELAYS_JOINT,
    POLICIES,
)
from run_joint_mean_square_step import (
    plot_outputs,
    run_experiment,
)
from linear_td_correlation import LinearTDConfig
from td_delay_stability import DIVERGENCE_THRESHOLD


BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 20270730


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "joint_ms_confirmation",
    )
    parser.add_argument("--num-seeds", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=20270130)
    return parser.parse_args()


def bootstrap_statistic_limit(
    values: np.ndarray,
    rng: np.random.RandomState,
    statistic: str,
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=float)
    sample_indices = rng.randint(
        0,
        len(values),
        size=(BOOTSTRAP_DRAWS, len(values)),
    )
    samples = values[sample_indices]
    if statistic == "mean":
        statistics = samples.mean(axis=1)
    elif statistic == "median":
        statistics = np.median(samples, axis=1)
    else:
        raise ValueError("unsupported bootstrap statistic")
    return float(np.quantile(statistics, quantile))


def joint_confidence_table(
    runs: pd.DataFrame,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    aware = runs[runs["policy"] == "joint_aware"]
    for key, group in aware.groupby(
        ["rho", "num_agents", "max_delay"], sort=True
    ):
        values = group["final_error"].to_numpy(dtype=float)
        rows.append(
            {
                "rho": float(key[0]),
                "num_agents": int(key[1]),
                "max_delay": int(key[2]),
                "eta": float(group["eta"].iloc[0]),
                "mean_final_error": float(np.mean(values)),
                "median_final_error": float(np.median(values)),
                "upper_99_bootstrap_mean": bootstrap_statistic_limit(
                    values, rng, "mean", 0.99
                ),
                "crossing_rate": float(
                    group["crossed_threshold"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_effect_table(
    runs: pd.DataFrame,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    targets = [
        ("correlation_blind", 0.9, q, d)
        for q in AGENT_COUNTS_JOINT
        for d in MAX_DELAYS_JOINT
    ] + [
        ("delay_blind", rho, 32, 32)
        for rho in CORRELATIONS_JOINT
    ]
    for policy, rho, num_agents, max_delay in targets:
        cell = runs[
            np.isclose(runs["rho"], rho)
            & (runs["num_agents"] == num_agents)
            & (runs["max_delay"] == max_delay)
            & runs["policy"].isin(["joint_aware", policy])
        ]
        pivot = cell.pivot(
            index="seed", columns="policy", values="final_error"
        ).sort_index()
        ratios = (
            pivot[policy].to_numpy(dtype=float)
            / pivot["joint_aware"].to_numpy(dtype=float)
        )
        rows.append(
            {
                "comparison_policy": policy,
                "rho": float(rho),
                "num_agents": int(num_agents),
                "max_delay": int(max_delay),
                "paired_median_error_ratio": float(
                    np.median(ratios)
                ),
                "lower_99_bootstrap_median_ratio": (
                    bootstrap_statistic_limit(
                        ratios, rng, "median", 0.01
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def useful_grid_table(
    runs: pd.DataFrame,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grid = runs[runs["policy"] == "grid"]
    aware_eta = (
        runs[runs["policy"] == "joint_aware"]
        .groupby(["rho", "num_agents", "max_delay"])["eta"]
        .first()
    )
    for key, cell in grid.groupby(
        ["rho", "num_agents", "max_delay"], sort=True
    ):
        step_rows: List[Dict[str, object]] = []
        for eta, group in cell.groupby("eta", sort=True):
            values = group["final_error"].to_numpy(dtype=float)
            upper = bootstrap_statistic_limit(
                values, rng, "mean", 0.99
            )
            crossing = float(group["crossed_threshold"].mean())
            step_rows.append(
                {
                    "eta": float(eta),
                    "upper": upper,
                    "crossing": crossing,
                    "useful": bool(crossing <= 0.05 and upper < 1.0),
                }
            )
        useful = [row for row in step_rows if row["useful"]]
        largest = (
            max(row["eta"] for row in useful)
            if useful
            else float("nan")
        )
        eta_joint = float(aware_eta.loc[key])
        rows.append(
            {
                "rho": float(key[0]),
                "num_agents": int(key[1]),
                "max_delay": int(key[2]),
                "joint_eta": eta_joint,
                "largest_useful_grid_eta": largest,
                "joint_to_useful_grid_ratio": eta_joint / largest,
                "useful_grid_count": int(len(useful)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_confirmation_gates(
    runs: pd.DataFrame,
    moments: pd.DataFrame,
    confidence: pd.DataFrame,
    paired: pd.DataFrame,
    useful: pd.DataFrame,
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
    analytic = bool(
        independent_drop >= 0.20
        and correlated_drop <= 0.02
        and all(value >= 5.0 for value in inflation.values())
    )
    contraction = bool(
        (confidence["crossing_rate"] == 0.0).all()
        and (confidence["upper_99_bootstrap_mean"] < 1.0).all()
    )
    correlation_effects = paired[
        paired["comparison_policy"] == "correlation_blind"
    ]
    correlation_value = bool(
        (
            correlation_effects[
                "lower_99_bootstrap_median_ratio"
            ]
            > 2.0
        ).all()
        and len(correlation_effects) == 4
    )
    delay_effects = paired[
        paired["comparison_policy"] == "delay_blind"
    ]
    delay_value = bool(
        (
            delay_effects["lower_99_bootstrap_median_ratio"]
            > 1.05
        ).all()
        and len(delay_effects) == 2
    )
    tight = bool(
        np.isfinite(useful["largest_useful_grid_eta"]).all()
        and (useful["joint_to_useful_grid_ratio"] >= 0.40).all()
    )
    aware = runs[
        (runs["policy"] == "joint_aware")
        & np.isclose(runs["rho"], 0.0)
    ].copy()
    worst = runs[
        (runs["policy"] == "worstcase_correlation")
        & np.isclose(runs["rho"], 0.0)
    ].copy()
    aware["effective_half_time"] = aware["half_error_time"].where(
        aware["half_error_time"] > 0, 4001
    )
    worst["effective_half_time"] = worst["half_error_time"].where(
        worst["half_error_time"] > 0, 4001
    )
    group_columns = ["num_agents", "max_delay"]
    aware_time = aware.groupby(group_columns)[
        "effective_half_time"
    ].median()
    worst_time = worst.groupby(group_columns)[
        "effective_half_time"
    ].median()
    aware_eta = aware.groupby(group_columns)["eta"].first()
    worst_eta = worst.groupby(group_columns)["eta"].first()
    faster_cells = int((aware_time <= worst_time).sum())
    fourfold_cells = int((aware_eta >= 4.0 * worst_eta).sum())
    speed = bool(faster_cells == 4 and fourfold_cells == 4)
    expected = (
        num_seeds
        * len(CORRELATIONS_JOINT)
        * len(AGENT_COUNTS_JOINT)
        * len(MAX_DELAYS_JOINT)
        * (len(POLICIES) + len(GRID_STEPS))
    )
    checkpoint_columns = [
        "error_{0}".format(int(point)) for point in CHECKPOINTS
    ]
    validity = bool(
        len(runs) == expected
        and runs["finite"].all()
        and np.isfinite(runs[checkpoint_columns].to_numpy()).all()
        and (
            (runs["crossed_threshold"] & (runs["crossing_time"] > 0))
            | (~runs["crossed_threshold"] & (runs["crossing_time"] == -1))
        ).all()
        and len(confidence) == 8
        and len(paired) == 6
        and len(useful) == 8
    )
    gates: Dict[str, object] = {
        "analytic_participation_saturation": {
            "pass": analytic,
            "independent_curvature_drop_q16_to_q32": independent_drop,
            "rho0p9_curvature_drop_q16_to_q32": correlated_drop,
            "rho0p9_to_rho0_curvature_inflation": inflation,
        },
        "joint_mean_square_contraction": {
            "pass": contraction,
            "largest_upper_99_bootstrap_mean": float(
                confidence["upper_99_bootstrap_mean"].max()
            ),
            "crossing_fraction": float(
                runs[runs["policy"] == "joint_aware"][
                    "crossed_threshold"
                ].mean()
            ),
        },
        "correlation_awareness_value": {
            "pass": correlation_value,
            "smallest_lower_99_paired_ratio": float(
                correlation_effects[
                    "lower_99_bootstrap_median_ratio"
                ].min()
            ),
        },
        "delay_awareness_value": {
            "pass": delay_value,
            "smallest_lower_99_paired_ratio": float(
                delay_effects[
                    "lower_99_bootstrap_median_ratio"
                ].min()
            ),
        },
        "nonvacuous_tightness": {
            "pass": tight,
            "smallest_joint_to_useful_grid_ratio": float(
                useful["joint_to_useful_grid_ratio"].min()
            ),
        },
        "correlation_adaptation_retains_speed": {
            "pass": speed,
            "cells_no_slower_than_worstcase": faster_cells,
            "cells_at_least_fourfold_step": fourfold_cells,
        },
        "accounting_determinism_numerical_validity": {
            "pass": validity,
            "observed_runs": int(len(runs)),
            "expected_runs": int(expected),
            "divergence_threshold": DIVERGENCE_THRESHOLD,
            "reproducibility": "PENDING_EXTERNAL_RERUN",
        },
    }
    gates["statistical_overall"] = {
        "pass": bool(all(value["pass"] for value in gates.values())),
        "criterion": (
            "all seven statistical/accounting gates pass; final verdict "
            "also requires external exact reproduction"
        ),
    }
    return gates


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = LinearTDConfig()
    runs, moments, policy_steps = run_experiment(
        config, args.num_seeds, args.base_seed
    )
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    confidence = joint_confidence_table(runs, rng)
    paired = paired_effect_table(runs, rng)
    useful = useful_grid_table(runs, rng)
    gates = evaluate_confirmation_gates(
        runs,
        moments,
        confidence,
        paired,
        useful,
        args.num_seeds,
    )
    runs.to_csv(args.output_dir / "confirmation_runs.csv", index=False)
    moments.to_csv(args.output_dir / "second_moments.csv", index=False)
    policy_steps.to_csv(args.output_dir / "policy_steps.csv", index=False)
    confidence.to_csv(
        args.output_dir / "joint_confidence.csv", index=False
    )
    paired.to_csv(args.output_dir / "paired_effects.csv", index=False)
    useful.to_csv(args.output_dir / "useful_grid.csv", index=False)
    plot_outputs(runs, moments, policy_steps, args.output_dir)
    summary = {
        "experiment_id": "EXP-007D-joint-ms-confirmation",
        "status": (
            "PENDING_REPRODUCTION"
            if gates["statistical_overall"]["pass"]
            else "FAIL"
        ),
        "config": {
            "num_seeds": args.num_seeds,
            "base_seed": args.base_seed,
            "agent_counts": list(AGENT_COUNTS_JOINT),
            "max_delays": list(MAX_DELAYS_JOINT),
            "correlations": list(CORRELATIONS_JOINT),
            "grid_steps": [float(value) for value in GRID_STEPS],
            "checkpoints": [int(value) for value in CHECKPOINTS],
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
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
    print(json.dumps(gates, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


"""Run EXP-008A exact lifted mean-square boundary audit."""

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exact_lifted_ms import (
    delay_counts,
    dense_lifted_matrix,
    first_stability_boundary,
    lifted_spectral_radius,
    td_jacobian_distribution,
)
from joint_mean_square_step import (
    AGENT_COUNTS_JOINT,
    CORRELATIONS_JOINT,
    build_mean_boundaries,
    registered_policy_steps,
    single_jacobian_second_moment,
)
from linear_model import make_agent_delays
from linear_td_correlation import LinearTDConfig, build_mrp


DELAYS_EXACT = (0, 8, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "exact_lifted_boundary",
    )
    return parser.parse_args()


def dense_validation(
    mrp: Dict[str, np.ndarray],
    jacobians: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, float]:
    counts = np.asarray([2], dtype=np.int64)
    eta = 0.04
    rho = 0.7
    dense = dense_lifted_matrix(
        eta,
        mrp["a_matrix"],
        jacobians,
        weights,
        counts,
        rho,
    )
    dense_radius = float(np.max(np.abs(np.linalg.eigvals(dense))))
    matrix_free, residual, dimension = lifted_spectral_radius(
        eta,
        mrp["a_matrix"],
        jacobians,
        weights,
        counts,
        rho,
    )
    return {
        "eta": eta,
        "rho": rho,
        "num_agents": 2,
        "dense_radius": dense_radius,
        "matrix_free_radius": matrix_free,
        "absolute_difference": abs(dense_radius - matrix_free),
        "matrix_free_residual": residual,
        "operator_dimension": dimension,
    }


def run_boundaries(
    config: LinearTDConfig,
) -> tuple:
    mrp = build_mrp(config)
    jacobians, weights = td_jacobian_distribution(mrp, config)
    second = single_jacobian_second_moment(mrp, config)
    mean_boundaries = build_mean_boundaries(mrp["a_matrix"], config)
    validation = dense_validation(mrp, jacobians, weights)
    rows: List[Dict[str, object]] = []
    total = (
        len(AGENT_COUNTS_JOINT)
        * len(DELAYS_EXACT)
        * len(CORRELATIONS_JOINT)
    )
    completed = 0
    for num_agents in AGENT_COUNTS_JOINT:
        for max_delay in DELAYS_EXACT:
            delays = make_agent_delays(
                max_agents=config.num_agents,
                max_delay=max_delay,
                exponent=config.delay_exponent,
            )[:num_agents]
            counts = delay_counts(delays)
            mean_boundary = mean_boundaries[(num_agents, max_delay)]
            for rho in CORRELATIONS_JOINT:
                steps = registered_policy_steps(
                    mrp["a_matrix"],
                    second,
                    mean_boundaries,
                    num_agents,
                    max_delay,
                    rho,
                )
                joint = steps["joint_aware"]
                exact = first_stability_boundary(
                    mean_boundary,
                    joint,
                    mrp["a_matrix"],
                    jacobians,
                    weights,
                    counts,
                    rho,
                )
                rows.append(
                    {
                        "rho": rho,
                        "num_agents": num_agents,
                        "max_delay": max_delay,
                        "selected_max_delay": int(np.max(delays)),
                        "active_delay_groups": int(np.count_nonzero(counts)),
                        "mean_boundary": mean_boundary,
                        "joint_eta": joint,
                        **exact,
                        "joint_to_exact_ratio": (
                            joint / exact["exact_boundary"]
                        ),
                        "exact_to_mean_ratio": (
                            exact["exact_boundary"] / mean_boundary
                        ),
                    }
                )
                completed += 1
                print(
                    "completed {0}/{1}: q={2}, D={3}, rho={4}".format(
                        completed,
                        total,
                        num_agents,
                        max_delay,
                        rho,
                    ),
                    flush=True,
                )
    return pd.DataFrame(rows), validation


def evaluate_gates(
    boundaries: pd.DataFrame,
    validation: Dict[str, float],
) -> Dict[str, object]:
    dense_pass = bool(
        validation["absolute_difference"] <= 1e-8
        and validation["matrix_free_residual"] <= 1e-7
    )
    max_residual = float(
        boundaries[
            ["below_residual", "above_residual", "joint_residual"]
        ]
        .to_numpy()
        .max()
    )
    valid = bool(
        np.isfinite(boundaries["exact_boundary"]).all()
        and (boundaries["exact_boundary"] > 0.0).all()
        and (boundaries["below_radius"] < 1.0).all()
        and (boundaries["above_radius"] > 1.0).all()
        and max_residual <= 1e-7
        and len(boundaries) == 12
    )
    joint_safe = bool((boundaries["joint_radius"] < 1.0).all())
    tight_cells = int(
        (boundaries["joint_to_exact_ratio"] >= 0.25).sum()
    )
    tight = bool(
        tight_cells >= 10
        and (boundaries["joint_to_exact_ratio"] <= 1.0).all()
    )
    pivot = boundaries.pivot_table(
        index=["num_agents", "max_delay"],
        columns="rho",
        values="exact_boundary",
    )
    correlation_ratios = pivot[0.9] / pivot[0.0]
    correlation = bool((correlation_ratios <= 0.5).all())
    d0 = boundaries[boundaries["max_delay"] == 0].pivot(
        index="num_agents", columns="rho", values="exact_boundary"
    )
    independent_gain = float(d0.loc[32, 0.0] / d0.loc[16, 0.0])
    correlated_gain = float(d0.loc[32, 0.9] / d0.loc[16, 0.9])
    saturation = bool(
        independent_gain >= 1.15 and correlated_gain <= 1.05
    )
    mean_insufficient_cells = int(
        (boundaries["exact_to_mean_ratio"] <= 0.5).sum()
    )
    mean_insufficient = bool(mean_insufficient_cells >= 10)
    gates: Dict[str, object] = {
        "independent_numerical_implementation": {
            "pass": dense_pass,
            **validation,
        },
        "boundary_validity": {
            "pass": valid,
            "observed_cells": int(len(boundaries)),
            "max_eigensolver_residual": max_residual,
            "max_operator_dimension": int(
                boundaries["operator_dimension"].max()
            ),
        },
        "scalar_rule_safety": {
            "pass": joint_safe,
            "largest_joint_radius": float(
                boundaries["joint_radius"].max()
            ),
        },
        "nonvacuous_scalar_tightness": {
            "pass": tight,
            "cells_ratio_at_least_quarter": tight_cells,
            "smallest_ratio": float(
                boundaries["joint_to_exact_ratio"].min()
            ),
            "largest_ratio": float(
                boundaries["joint_to_exact_ratio"].max()
            ),
        },
        "correlation_shrinks_exact_region": {
            "pass": correlation,
            "largest_rho0p9_to_rho0_boundary_ratio": float(
                correlation_ratios.max()
            ),
        },
        "agent_count_saturation_d0": {
            "pass": saturation,
            "independent_q32_to_q16_boundary_gain": independent_gain,
            "rho0p9_q32_to_q16_boundary_gain": correlated_gain,
        },
        "mean_stability_insufficient": {
            "pass": mean_insufficient,
            "cells_exact_at_most_half_mean": mean_insufficient_cells,
            "largest_exact_to_mean_ratio": float(
                boundaries["exact_to_mean_ratio"].max()
            ),
        },
    }
    gates["overall"] = {
        "pass": bool(all(value["pass"] for value in gates.values())),
        "criterion": "all seven exact lifted-boundary gates pass",
    }
    return gates


def plot_outputs(boundaries: pd.DataFrame, output_dir: Path) -> None:
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
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, rho in zip(axes, CORRELATIONS_JOINT):
        selected = boundaries[np.isclose(boundaries["rho"], rho)]
        for num_agents in AGENT_COUNTS_JOINT:
            cell = selected[selected["num_agents"] == num_agents]
            ax.plot(
                cell["max_delay"],
                cell["exact_boundary"],
                marker="o",
                label="exact q={0}".format(num_agents),
            )
            ax.plot(
                cell["max_delay"],
                cell["joint_eta"],
                marker="x",
                linestyle="--",
                label="joint q={0}".format(num_agents),
            )
        ax.set_yscale("log")
        ax.set_xlabel("Registered maximum delay D")
        ax.set_title("rho={0}".format(rho))
    axes[0].set_ylabel("Step size")
    axes[1].legend(fontsize=7)
    fig.suptitle("Exact lifted boundary and scalar joint step")
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_exact_vs_joint.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for rho in CORRELATIONS_JOINT:
        selected = boundaries[np.isclose(boundaries["rho"], rho)]
        labels = [
            "q{0}-D{1}".format(int(row.num_agents), int(row.max_delay))
            for row in selected.itertuples()
        ]
        ax.plot(
            labels,
            selected["joint_to_exact_ratio"],
            marker="o",
            label="rho={0}".format(rho),
        )
    ax.axhline(0.25, color="black", linestyle="--")
    ax.axhline(1.0, color="black", linestyle=":")
    ax.set_ylabel("joint step / exact boundary")
    ax.set_title("Conservatism of the low-complexity scalar rule")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_scalar_tightness.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    boundaries, validation = run_boundaries(LinearTDConfig())
    gates = evaluate_gates(boundaries, validation)
    boundaries.to_csv(
        args.output_dir / "exact_lifted_boundaries.csv", index=False
    )
    with (args.output_dir / "dense_validation.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(validation, handle, indent=2, sort_keys=True)
    plot_outputs(boundaries, args.output_dir)
    summary = {
        "experiment_id": "EXP-008A-exact-lifted-boundary",
        "status": "PASS" if gates["overall"]["pass"] else "FAIL",
        "config": {
            "agent_counts": list(AGENT_COUNTS_JOINT),
            "max_delays": list(DELAYS_EXACT),
            "correlations": list(CORRELATIONS_JOINT),
            "temporal_sampling": "independent stationary transition pairs",
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


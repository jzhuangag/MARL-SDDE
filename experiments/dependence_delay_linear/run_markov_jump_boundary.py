"""Run the preregistered EXP-008B exact Markov-jump boundary audit."""

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

from exact_lifted_ms import delay_counts, dense_lifted_matrix
from markov_jump_ms import (
    AGENT_COUNTS_MARKOV,
    CORRELATIONS_MARKOV,
    MAX_DELAYS_MARKOV,
    REGIME_PERSISTENCES,
    aggregate_same_time_curvature,
    covariance_operator_coefficients,
    direct_conditional_operator,
    first_stability_boundary,
    mean_operator_coefficients,
    registered_delays,
    registered_td_model,
    spectral_radius_with_residual,
)
from td_delay_stability import critical_step_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "markov_jump_boundary",
    )
    return parser.parse_args()


def direct_validation(model: Dict[str, np.ndarray]) -> Dict[str, float]:
    delays = registered_delays(2, 2)
    eta = 0.23
    rho = 0.37
    mode = 0
    moment = dense_lifted_matrix(
        eta,
        model["conditional_means"][mode],
        model["jacobians"],
        model["weights"][mode],
        delay_counts(delays),
        rho,
    )
    enumerated = direct_conditional_operator(
        model, delays, rho, mode, eta
    )
    return {
        "eta": eta,
        "rho": rho,
        "mode": mode,
        "num_agents": 2,
        "maximum_delay": 2,
        "maximum_absolute_difference": float(
            np.max(np.abs(moment - enumerated))
        ),
        "frobenius_difference": float(
            np.linalg.norm(moment - enumerated)
        ),
    }


def run_boundaries(
    model: Dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    iid_cache: Dict[Tuple[int, int, float], Dict[str, float]] = {}
    total = (
        len(AGENT_COUNTS_MARKOV)
        * len(MAX_DELAYS_MARKOV)
        * len(REGIME_PERSISTENCES)
        * len(CORRELATIONS_MARKOV)
    )
    completed = 0
    for num_agents in AGENT_COUNTS_MARKOV:
        for maximum_delay in MAX_DELAYS_MARKOV:
            delays = registered_delays(num_agents, maximum_delay)
            stationary_mean_boundary = critical_step_size(
                model["stationary_mean"], delays
            )
            for rho in CORRELATIONS_MARKOV:
                _, curvature, monotonicity = (
                    aggregate_same_time_curvature(
                        model, num_agents, rho
                    )
                )
                scalar_eta = 1.0 / (
                    1.0 / stationary_mean_boundary
                    + curvature / (2.0 * monotonicity)
                )
                cache_key = (num_agents, maximum_delay, rho)
                for persistence in REGIME_PERSISTENCES:
                    covariance = covariance_operator_coefficients(
                        model, delays, rho, persistence
                    )
                    if cache_key not in iid_cache:
                        iid_cache[cache_key] = first_stability_boundary(
                            covariance["iid"], scalar_eta
                        )
                    iid = iid_cache[cache_key]
                    markov = first_stability_boundary(
                        covariance["markov"], scalar_eta
                    )
                    mean_coefficients = mean_operator_coefficients(
                        model, delays, persistence
                    )
                    markov_mean = first_stability_boundary(
                        mean_coefficients["markov"],
                        stationary_mean_boundary,
                    )
                    scalar_radius, scalar_residual = (
                        spectral_radius_with_residual(
                            covariance["markov"], scalar_eta
                        )
                    )
                    rows.append(
                        {
                            "persistence": persistence,
                            "rho": rho,
                            "num_agents": num_agents,
                            "maximum_delay": maximum_delay,
                            "delay_profile": ",".join(
                                str(int(value)) for value in delays
                            ),
                            "curvature": curvature,
                            "monotonicity": monotonicity,
                            "stationary_mean_boundary": (
                                stationary_mean_boundary
                            ),
                            "scalar_eta": scalar_eta,
                            "scalar_radius": scalar_radius,
                            "scalar_residual": scalar_residual,
                            "markov_boundary": markov["boundary"],
                            "markov_below_radius": markov[
                                "below_radius"
                            ],
                            "markov_below_residual": markov[
                                "below_residual"
                            ],
                            "markov_above_radius": markov[
                                "above_radius"
                            ],
                            "markov_above_residual": markov[
                                "above_residual"
                            ],
                            "markov_operator_dimension": markov[
                                "operator_dimension"
                            ],
                            "iid_boundary": iid["boundary"],
                            "iid_below_radius": iid["below_radius"],
                            "iid_below_residual": iid["below_residual"],
                            "iid_above_radius": iid["above_radius"],
                            "iid_above_residual": iid["above_residual"],
                            "markov_mean_boundary": markov_mean[
                                "boundary"
                            ],
                            "markov_mean_below_radius": markov_mean[
                                "below_radius"
                            ],
                            "markov_mean_below_residual": markov_mean[
                                "below_residual"
                            ],
                            "markov_mean_above_radius": markov_mean[
                                "above_radius"
                            ],
                            "markov_mean_above_residual": markov_mean[
                                "above_residual"
                            ],
                            "markov_to_iid_ratio": (
                                markov["boundary"] / iid["boundary"]
                            ),
                            "scalar_to_markov_ratio": (
                                scalar_eta / markov["boundary"]
                            ),
                            "ms_to_markov_mean_ratio": (
                                markov["boundary"]
                                / markov_mean["boundary"]
                            ),
                        }
                    )
                    completed += 1
                    print(
                        "completed {0}/{1}: q={2}, D={3}, p={4}, "
                        "rho={5}".format(
                            completed,
                            total,
                            num_agents,
                            maximum_delay,
                            persistence,
                            rho,
                        ),
                        flush=True,
                    )
    return pd.DataFrame(rows)


def evaluate_gates(
    frame: pd.DataFrame,
    validation: Dict[str, float],
) -> Dict[str, object]:
    independent = bool(
        validation["maximum_absolute_difference"] <= 1e-11
    )
    residual_columns = [
        "markov_below_residual",
        "markov_above_residual",
        "iid_below_residual",
        "iid_above_residual",
        "markov_mean_below_residual",
        "markov_mean_above_residual",
        "scalar_residual",
    ]
    max_residual = float(frame[residual_columns].to_numpy().max())
    boundary_valid = bool(
        len(frame) == 36
        and np.isfinite(
            frame[
                [
                    "markov_boundary",
                    "iid_boundary",
                    "markov_mean_boundary",
                ]
            ].to_numpy()
        ).all()
        and (
            frame[
                [
                    "markov_boundary",
                    "iid_boundary",
                    "markov_mean_boundary",
                ]
            ]
            > 0.0
        )
        .all()
        .all()
        and (frame["markov_below_radius"] < 1.0).all()
        and (frame["markov_above_radius"] > 1.0).all()
        and (frame["iid_below_radius"] < 1.0).all()
        and (frame["iid_above_radius"] > 1.0).all()
        and (frame["markov_mean_below_radius"] < 1.0).all()
        and (frame["markov_mean_above_radius"] > 1.0).all()
        and max_residual <= 1e-7
    )
    iid_rows = frame[np.isclose(frame["persistence"], 0.5)]
    iid_relative = np.abs(
        iid_rows["markov_boundary"] - iid_rows["iid_boundary"]
    ) / iid_rows["iid_boundary"]
    iid_reduction = bool((iid_relative <= 1e-7).all())

    q1 = frame[frame["num_agents"] == 1]
    q1_boundary = q1.pivot_table(
        index=["persistence", "maximum_delay"],
        columns="rho",
        values="markov_boundary",
    )
    q1_scalar = q1.pivot_table(
        index=["persistence", "maximum_delay"],
        columns="rho",
        values="scalar_eta",
    )
    q1_boundary_relative = np.abs(
        q1_boundary[0.9] - q1_boundary[0.0]
    ) / q1_boundary[0.0]
    q1_scalar_relative = np.abs(
        q1_scalar[0.9] - q1_scalar[0.0]
    ) / q1_scalar[0.0]
    q1_invariant = bool(
        (q1_boundary_relative <= 1e-10).all()
        and (q1_scalar_relative <= 1e-10).all()
    )

    persistent = frame[np.isclose(frame["persistence"], 0.98)]
    active_cells = int(
        (persistent["markov_to_iid_ratio"] <= 0.8).sum()
    )
    temporal_active = bool(active_cells >= 6)
    scalar_safe = bool((frame["scalar_radius"] < 1.0).all())

    gains = frame.pivot_table(
        index=["persistence", "maximum_delay", "rho"],
        columns="num_agents",
        values="markov_boundary",
    )
    gains["q3_to_q1"] = gains[3] / gains[1]
    low_gain = gains.xs(0.0, level="rho")["q3_to_q1"]
    high_gain = gains.xs(0.9, level="rho")["q3_to_q1"]
    agent_mechanistic = bool(
        (low_gain + 1e-10 >= high_gain).all()
    )
    saturated_slices = int((high_gain <= 1.10).sum())
    saturation = bool(saturated_slices >= 4)

    delay_table = frame.pivot_table(
        index=["num_agents", "rho", "persistence"],
        columns="maximum_delay",
        values="markov_boundary",
    )
    delay_table["d2_to_d0"] = delay_table[2] / delay_table[0]
    iid_delay = delay_table.xs(0.5, level="persistence")[
        "d2_to_d0"
    ]
    persistent_delay = delay_table.xs(0.98, level="persistence")[
        "d2_to_d0"
    ]
    interacting_slices = int(
        (persistent_delay <= iid_delay + 1e-10).sum()
    )
    delay_interaction = bool(interacting_slices >= 4)

    mean_insufficient_cells = int(
        (frame["ms_to_markov_mean_ratio"] <= 0.8).sum()
    )
    mean_insufficient = bool(mean_insufficient_cells >= 24)

    numerical = {
        "independent_construction": {
            "pass": independent,
            **validation,
        },
        "boundary_validity": {
            "pass": boundary_valid,
            "observed_cells": int(len(frame)),
            "maximum_eigen_residual": max_residual,
            "maximum_covariance_operator_dimension": int(
                frame["markov_operator_dimension"].max()
            ),
        },
        "iid_reduction": {
            "pass": iid_reduction,
            "maximum_relative_boundary_error": float(
                iid_relative.max()
            ),
        },
        "one_agent_correlation_invariance": {
            "pass": q1_invariant,
            "maximum_boundary_relative_change": float(
                q1_boundary_relative.max()
            ),
            "maximum_scalar_relative_change": float(
                q1_scalar_relative.max()
            ),
        },
    }
    scientific = {
        "temporal_persistence_active": {
            "pass": temporal_active,
            "cells_ratio_at_most_0p8": active_cells,
            "minimum_markov_to_iid_ratio_at_p0p98": float(
                persistent["markov_to_iid_ratio"].min()
            ),
            "maximum_markov_to_iid_ratio_at_p0p98": float(
                persistent["markov_to_iid_ratio"].max()
            ),
        },
        "iid_scalar_rule_markov_safe": {
            "pass": scalar_safe,
            "largest_scalar_radius": float(
                frame["scalar_radius"].max()
            ),
            "smallest_scalar_to_markov_ratio": float(
                frame["scalar_to_markov_ratio"].min()
            ),
            "largest_scalar_to_markov_ratio": float(
                frame["scalar_to_markov_ratio"].max()
            ),
        },
        "agent_count_mechanistic": {
            "pass": agent_mechanistic,
            "minimum_independent_q3_to_q1_gain": float(
                low_gain.min()
            ),
            "maximum_correlated_q3_to_q1_gain": float(
                high_gain.max()
            ),
        },
        "correlation_limited_saturation": {
            "pass": saturation,
            "saturated_slices": saturated_slices,
            "total_slices": int(len(high_gain)),
        },
        "delay_persistence_interaction": {
            "pass": delay_interaction,
            "interacting_slices": interacting_slices,
            "total_slices": int(len(iid_delay)),
        },
        "mean_stability_insufficient": {
            "pass": mean_insufficient,
            "cells_ms_at_most_0p8_mean": mean_insufficient_cells,
            "minimum_ms_to_mean_ratio": float(
                frame["ms_to_markov_mean_ratio"].min()
            ),
            "maximum_ms_to_mean_ratio": float(
                frame["ms_to_markov_mean_ratio"].max()
            ),
        },
    }
    return {
        "numerical_checks": numerical,
        "scientific_gates": scientific,
        "numerical_overall_pass": bool(
            all(value["pass"] for value in numerical.values())
        ),
        "scientific_pass_count": int(
            sum(value["pass"] for value in scientific.values())
        ),
        "scientific_total": int(len(scientific)),
    }


def save_figures(frame: pd.DataFrame, output_dir: Path) -> None:
    colors = {0.0: "#2474b5", 0.9: "#d95f02"}
    markers = {1: "o", 2: "s", 3: "^"}
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
    for axis, maximum_delay in zip(axes, MAX_DELAYS_MARKOV):
        subset = frame[frame["maximum_delay"] == maximum_delay]
        for rho in CORRELATIONS_MARKOV:
            for num_agents in AGENT_COUNTS_MARKOV:
                line = subset[
                    np.isclose(subset["rho"], rho)
                    & (subset["num_agents"] == num_agents)
                ].sort_values("persistence")
                axis.plot(
                    line["persistence"],
                    line["markov_to_iid_ratio"],
                    color=colors[rho],
                    marker=markers[num_agents],
                    label=(
                        rf"$\rho={rho:g},q={num_agents}$"
                        if maximum_delay == 0
                        else None
                    ),
                )
        axis.axhline(0.8, color="black", linestyle="--", linewidth=1)
        axis.set_title(rf"$D={maximum_delay}$")
        axis.set_xlabel("regime persistence")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("exact Markov / i.i.d. boundary")
    axes[0].legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(
        output_dir / "fig1_temporal_boundary_ratio.png", dpi=220
    )
    plt.close(figure)

    figure, axes = plt.subplots(
        len(REGIME_PERSISTENCES),
        len(MAX_DELAYS_MARKOV),
        figsize=(9.0, 9.0),
        sharex=True,
    )
    for row, persistence in enumerate(REGIME_PERSISTENCES):
        for column, maximum_delay in enumerate(MAX_DELAYS_MARKOV):
            axis = axes[row, column]
            subset = frame[
                np.isclose(frame["persistence"], persistence)
                & (frame["maximum_delay"] == maximum_delay)
            ]
            for rho in CORRELATIONS_MARKOV:
                line = subset[np.isclose(subset["rho"], rho)].sort_values(
                    "num_agents"
                )
                axis.plot(
                    line["num_agents"],
                    line["markov_boundary"],
                    color=colors[rho],
                    marker="o",
                    label=rf"$\rho={rho:g}$" if row == 0 else None,
                )
            axis.set_title(rf"$p={persistence:g},D={maximum_delay}$")
            axis.grid(alpha=0.25)
            if row == len(REGIME_PERSISTENCES) - 1:
                axis.set_xlabel("agent count")
            if column == 0:
                axis.set_ylabel("exact boundary")
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "fig2_agent_correlation_boundary.png", dpi=220
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for persistence in REGIME_PERSISTENCES:
        subset = frame[np.isclose(frame["persistence"], persistence)]
        axis.scatter(
            subset["markov_boundary"],
            subset["scalar_eta"],
            label=rf"$p={persistence:g}$",
            alpha=0.8,
        )
    limit = float(
        max(frame["markov_boundary"].max(), frame["scalar_eta"].max())
    )
    axis.plot((0.0, limit), (0.0, limit), "k--", linewidth=1)
    axis.set_xlabel("exact Markov boundary")
    axis.set_ylabel("same-time scalar step")
    axis.set_title("Safety and conservatism of the uninflated scalar rule")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "fig3_scalar_safety.png", dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = registered_td_model()
    validation = direct_validation(model)
    frame = run_boundaries(model)
    gates = evaluate_gates(frame, validation)
    frame.to_csv(args.output_dir / "markov_jump_boundaries.csv", index=False)
    with (args.output_dir / "direct_validation.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(validation, handle, indent=2)
    summary = {
        "experiment": "EXP-008B",
        "status": (
            "VALID"
            if gates["numerical_overall_pass"]
            else "INVALID_NUMERICS"
        ),
        "registered_cells": 36,
        "gates": gates,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "artifacts": [
            "markov_jump_boundaries.csv",
            "direct_validation.json",
            "fig1_temporal_boundary_ratio.png",
            "fig2_agent_correlation_boundary.png",
            "fig3_scalar_safety.png",
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

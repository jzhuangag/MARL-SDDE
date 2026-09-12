"""Run preregistered EXP-008C locally expanding Markov TD audit."""

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
    AGENT_COUNTS_EXPANDING,
    CORRELATIONS_MARKOV,
    MAX_DELAYS_MARKOV,
    REGIME_PERSISTENCES,
    aggregate_same_time_curvature,
    covariance_operator_coefficients,
    direct_conditional_operator,
    first_stability_boundary,
    homogeneous_delays,
    mean_operator_coefficients,
    registered_expanding_td_model,
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
        / "expanding_markov_td",
    )
    return parser.parse_args()


def direct_validation(model: Dict[str, np.ndarray]) -> Dict[str, float]:
    delays = homogeneous_delays(2, 2)
    eta = 0.17
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
        "homogeneous_delay": 2,
        "maximum_absolute_difference": float(
            np.max(np.abs(moment - enumerated))
        ),
        "frobenius_difference": float(
            np.linalg.norm(moment - enumerated)
        ),
    }


def run_boundaries(model: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    iid_cache: Dict[Tuple[int, int, float], Dict[str, float]] = {}
    total = (
        len(AGENT_COUNTS_EXPANDING)
        * len(MAX_DELAYS_MARKOV)
        * len(REGIME_PERSISTENCES)
        * len(CORRELATIONS_MARKOV)
    )
    completed = 0
    for num_agents in AGENT_COUNTS_EXPANDING:
        for delay in MAX_DELAYS_MARKOV:
            delays = homogeneous_delays(num_agents, delay)
            stationary_mean_boundary = critical_step_size(
                model["stationary_mean"], delays
            )
            for rho in CORRELATIONS_MARKOV:
                _, curvature, monotonicity = (
                    aggregate_same_time_curvature(
                        model, num_agents, rho
                    )
                )
                iid_eta = 1.0 / (
                    1.0 / stationary_mean_boundary
                    + curvature / (2.0 * monotonicity)
                )
                cache_key = (num_agents, delay, rho)
                for persistence in REGIME_PERSISTENCES:
                    mixing_inflation = persistence / (1.0 - persistence)
                    gap_eta = 1.0 / (
                        1.0 / stationary_mean_boundary
                        + mixing_inflation
                        * curvature
                        / (2.0 * monotonicity)
                    )
                    covariance = covariance_operator_coefficients(
                        model, delays, rho, persistence
                    )
                    if cache_key not in iid_cache:
                        iid_cache[cache_key] = first_stability_boundary(
                            covariance["iid"], iid_eta
                        )
                    iid = iid_cache[cache_key]
                    markov = first_stability_boundary(
                        covariance["markov"], gap_eta
                    )
                    mean_coefficients = mean_operator_coefficients(
                        model, delays, persistence
                    )
                    markov_mean = first_stability_boundary(
                        mean_coefficients["markov"], gap_eta
                    )
                    iid_radius, iid_residual = (
                        spectral_radius_with_residual(
                            covariance["markov"], iid_eta
                        )
                    )
                    gap_radius, gap_residual = (
                        spectral_radius_with_residual(
                            covariance["markov"], gap_eta
                        )
                    )
                    required_inflation = (
                        2.0
                        * monotonicity
                        / curvature
                        * (
                            1.0 / markov["boundary"]
                            - 1.0 / stationary_mean_boundary
                        )
                    )
                    rows.append(
                        {
                            "persistence": persistence,
                            "rho": rho,
                            "num_agents": num_agents,
                            "homogeneous_delay": delay,
                            "curvature": curvature,
                            "monotonicity": monotonicity,
                            "stationary_mean_boundary": (
                                stationary_mean_boundary
                            ),
                            "mixing_inflation": mixing_inflation,
                            "iid_eta": iid_eta,
                            "gap_eta": gap_eta,
                            "iid_rule_radius": iid_radius,
                            "iid_rule_residual": iid_residual,
                            "gap_rule_radius": gap_radius,
                            "gap_rule_residual": gap_residual,
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
                            "iid_to_markov_ratio": (
                                iid_eta / markov["boundary"]
                            ),
                            "gap_to_markov_ratio": (
                                gap_eta / markov["boundary"]
                            ),
                            "ms_to_markov_mean_ratio": (
                                markov["boundary"]
                                / markov_mean["boundary"]
                            ),
                            "required_inflation": required_inflation,
                            "required_to_gap_inflation": (
                                required_inflation / mixing_inflation
                            ),
                        }
                    )
                    completed += 1
                    print(
                        "completed {0}/{1}: q={2}, delay={3}, p={4}, "
                        "rho={5}".format(
                            completed,
                            total,
                            num_agents,
                            delay,
                            persistence,
                            rho,
                        ),
                        flush=True,
                    )
    return pd.DataFrame(rows)


def evaluate_gates(
    frame: pd.DataFrame, validation: Dict[str, float]
) -> Dict[str, object]:
    residual_columns = [
        "markov_below_residual",
        "markov_above_residual",
        "iid_below_residual",
        "iid_above_residual",
        "markov_mean_below_residual",
        "markov_mean_above_residual",
        "iid_rule_residual",
        "gap_rule_residual",
    ]
    maximum_residual = float(frame[residual_columns].to_numpy().max())
    numerical = {
        "independent_construction": {
            "pass": bool(
                validation["maximum_absolute_difference"] <= 1e-11
            ),
            **validation,
        },
        "boundary_validity": {
            "pass": bool(
                len(frame) == 72
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
                and maximum_residual <= 1e-7
            ),
            "observed_cells": int(len(frame)),
            "maximum_eigen_residual": maximum_residual,
            "maximum_covariance_operator_dimension": int(
                frame["markov_operator_dimension"].max()
            ),
        },
    }
    iid_rows = frame[np.isclose(frame["persistence"], 0.5)]
    iid_relative = np.abs(
        iid_rows["markov_boundary"] - iid_rows["iid_boundary"]
    ) / iid_rows["iid_boundary"]
    numerical["iid_reduction"] = {
        "pass": bool((iid_relative <= 1e-7).all()),
        "maximum_relative_boundary_error": float(iid_relative.max()),
    }
    q1 = frame[frame["num_agents"] == 1]
    q1_boundary = q1.pivot_table(
        index=["persistence", "homogeneous_delay"],
        columns="rho",
        values="markov_boundary",
    )
    q1_gap = q1.pivot_table(
        index=["persistence", "homogeneous_delay"],
        columns="rho",
        values="gap_eta",
    )
    q1_boundary_change = np.abs(
        q1_boundary[0.9] - q1_boundary[0.0]
    ) / q1_boundary[0.0]
    q1_gap_change = np.abs(q1_gap[0.9] - q1_gap[0.0]) / q1_gap[0.0]
    numerical["one_agent_correlation_invariance"] = {
        "pass": bool(
            (q1_boundary_change <= 1e-10).all()
            and (q1_gap_change <= 1e-10).all()
        ),
        "maximum_boundary_relative_change": float(
            q1_boundary_change.max()
        ),
        "maximum_gap_rule_relative_change": float(
            q1_gap_change.max()
        ),
    }

    persistent = frame[np.isclose(frame["persistence"], 0.98)]
    active_cells = int(
        (persistent["markov_to_iid_ratio"] <= 0.5).sum()
    )
    iid_unsafe_cells = int((persistent["iid_rule_radius"] >= 1.0).sum())
    gap_safe = bool((frame["gap_rule_radius"] < 1.0).all())
    nonvacuous_cells = int(
        (frame["gap_to_markov_ratio"] >= 0.20).sum()
    )
    gap_nonvacuous = bool(
        nonvacuous_cells >= 60
        and (frame["gap_to_markov_ratio"] <= 1.0).all()
    )

    q_ge_two = frame[frame["num_agents"] >= 2]
    correlation_table = q_ge_two.pivot_table(
        index=["num_agents", "homogeneous_delay", "persistence"],
        columns="rho",
        values="markov_boundary",
    )
    correlation_nonimproving = bool(
        (correlation_table[0.9] <= correlation_table[0.0] + 1e-10).all()
    )

    gains = frame.pivot_table(
        index=["homogeneous_delay", "persistence", "rho"],
        columns="num_agents",
        values="markov_boundary",
    )
    gains["q32_to_q16"] = gains[32] / gains[16]
    independent_gain = gains.xs(0.0, level="rho")["q32_to_q16"]
    correlated_gain = gains.xs(0.9, level="rho")["q32_to_q16"]
    correlation_limited = bool(
        (correlated_gain <= 1.05).all()
        and (correlated_gain <= independent_gain + 1e-10).all()
    )

    mean_insufficient_cells = int(
        (frame["ms_to_markov_mean_ratio"] <= 0.8).sum()
    )
    scientific = {
        "persistence_activates_instability": {
            "pass": bool(active_cells >= 18),
            "active_cells": active_cells,
            "total_p0p98_cells": int(len(persistent)),
            "minimum_markov_to_iid_ratio": float(
                persistent["markov_to_iid_ratio"].min()
            ),
            "maximum_markov_to_iid_ratio": float(
                persistent["markov_to_iid_ratio"].max()
            ),
        },
        "iid_rule_insufficient": {
            "pass": bool(iid_unsafe_cells >= 12),
            "unstable_p0p98_cells": iid_unsafe_cells,
            "largest_iid_rule_radius_at_p0p98": float(
                persistent["iid_rule_radius"].max()
            ),
        },
        "gap_inflated_safety": {
            "pass": gap_safe,
            "largest_gap_rule_radius": float(
                frame["gap_rule_radius"].max()
            ),
        },
        "gap_inflated_nonvacuity": {
            "pass": gap_nonvacuous,
            "cells_ratio_at_least_0p20": nonvacuous_cells,
            "smallest_gap_to_exact_ratio": float(
                frame["gap_to_markov_ratio"].min()
            ),
            "largest_gap_to_exact_ratio": float(
                frame["gap_to_markov_ratio"].max()
            ),
        },
        "correlation_does_not_improve_stability": {
            "pass": correlation_nonimproving,
            "largest_correlated_to_independent_boundary_ratio": float(
                (correlation_table[0.9] / correlation_table[0.0]).max()
            ),
        },
        "correlation_limited_participation": {
            "pass": correlation_limited,
            "largest_correlated_q32_to_q16_gain": float(
                correlated_gain.max()
            ),
            "smallest_independent_minus_correlated_gain": float(
                (independent_gain - correlated_gain).min()
            ),
        },
        "mean_stability_insufficient": {
            "pass": bool(mean_insufficient_cells >= 48),
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
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for axis, delay in zip(axes, MAX_DELAYS_MARKOV):
        subset = frame[frame["homogeneous_delay"] == delay]
        for rho in CORRELATIONS_MARKOV:
            for persistence in REGIME_PERSISTENCES:
                line = subset[
                    np.isclose(subset["rho"], rho)
                    & np.isclose(subset["persistence"], persistence)
                ].sort_values("num_agents")
                axis.plot(
                    line["num_agents"],
                    line["markov_boundary"],
                    marker="o",
                    color=colors[rho],
                    linestyle=(
                        "-" if persistence == 0.5
                        else "--" if persistence == 0.9
                        else ":"
                    ),
                    label=(
                        rf"$\rho={rho:g},p={persistence:g}$"
                        if delay == 0
                        else None
                    ),
                )
        axis.set_xscale("log", base=2)
        axis.set_xlabel("agent count")
        axis.set_title(rf"homogeneous delay {delay}")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("exact Markov boundary")
    axes[0].legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(
        output_dir / "fig1_exact_boundary_scaling.png", dpi=220
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for axis, persistence in zip(axes, (0.9, 0.98)):
        subset = frame[np.isclose(frame["persistence"], persistence)]
        for rule, marker, color in (
            ("iid_to_markov_ratio", "x", "#c44e52"),
            ("gap_to_markov_ratio", "o", "#4c9f70"),
        ):
            axis.scatter(
                subset["num_agents"]
                + 0.05 * subset["rho"]
                + 0.02 * subset["homogeneous_delay"],
                subset[rule],
                marker=marker,
                color=color,
                alpha=0.75,
                label=rule.replace("_to_markov_ratio", "")
                if persistence == 0.9
                else None,
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.axhline(0.2, color="gray", linestyle=":", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("agent count (jittered)")
        axis.set_title(rf"$p={persistence:g}$")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("selected step / exact boundary")
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "fig2_rule_tightness.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for delay, marker in ((0, "o"), (2, "s")):
        subset = frame[frame["homogeneous_delay"] == delay]
        axis.scatter(
            subset["mixing_inflation"],
            subset["required_inflation"],
            marker=marker,
            alpha=0.7,
            label=rf"delay {delay}",
        )
    limit = float(
        max(
            frame["mixing_inflation"].max(),
            frame["required_inflation"].max(),
        )
    )
    axis.plot((0.0, limit), (0.0, limit), "k--", linewidth=1)
    axis.set_xscale("log")
    axis.set_yscale("symlog", linthresh=0.1)
    axis.set_xlabel(r"frozen gap inflation $\chi_{\rm gap}$")
    axis.set_ylabel(r"exact required inflation $\chi_{\rm req}$")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "fig3_required_inflation.png", dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = registered_expanding_td_model()
    validation = direct_validation(model)
    frame = run_boundaries(model)
    gates = evaluate_gates(frame, validation)
    frame.to_csv(
        args.output_dir / "expanding_markov_boundaries.csv", index=False
    )
    with (args.output_dir / "direct_validation.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(validation, handle, indent=2)
    summary = {
        "experiment": "EXP-008C",
        "status": (
            "VALID"
            if gates["numerical_overall_pass"]
            else "INVALID_NUMERICS"
        ),
        "registered_cells": 72,
        "conditional_means": (
            model["conditional_means"].reshape(-1).tolist()
        ),
        "stationary_mean": float(model["stationary_mean"][0, 0]),
        "gates": gates,
        "exploratory": {
            "minimum_required_to_gap_inflation": float(
                frame["required_to_gap_inflation"].min()
            ),
            "maximum_required_to_gap_inflation": float(
                frame["required_to_gap_inflation"].max()
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "artifacts": [
            "expanding_markov_boundaries.csv",
            "direct_validation.json",
            "fig1_exact_boundary_scaling.png",
            "fig2_rule_tightness.png",
            "fig3_required_inflation.png",
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

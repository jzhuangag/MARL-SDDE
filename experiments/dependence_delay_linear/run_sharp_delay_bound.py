"""Run preregistered EXP-008E sharp delayed theorem audit."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from markov_jump_ms import (
    AGENT_COUNTS_EXPANDING,
    CORRELATIONS_MARKOV,
    MAX_DELAYS_MARKOV,
    REGIME_PERSISTENCES,
    covariance_operator_coefficients,
    first_stability_boundary,
    homogeneous_delays,
    minimum_decorrelation_gap,
    mixing_tv_after_gap,
    registered_expanding_td_model,
    sharp_theorem_steps,
    spectral_radius_with_residual,
    thinned_persistence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "sharp_delay_bound",
    )
    return parser.parse_args()


def run_cells(model: Dict[str, np.ndarray]) -> pd.DataFrame:
    lipschitz = float(
        max(
            np.linalg.norm(matrix, ord=2)
            for matrix in model["jacobians"]
        )
    )
    monotonicity = float(model["stationary_mean"][0, 0])
    target_delta = monotonicity / (4.0 * lipschitz)
    rows: List[Dict[str, object]] = []
    total = 72
    completed = 0
    for persistence in REGIME_PERSISTENCES:
        gap = minimum_decorrelation_gap(persistence, target_delta)
        delta = mixing_tv_after_gap(persistence, gap)
        used_persistence = thinned_persistence(persistence, gap)
        for delay in MAX_DELAYS_MARKOV:
            for rho in CORRELATIONS_MARKOV:
                for num_agents in AGENT_COUNTS_EXPANDING:
                    delays = homogeneous_delays(num_agents, delay)
                    theorem = sharp_theorem_steps(
                        model,
                        num_agents,
                        rho,
                        delays,
                        delta,
                    )
                    covariance = covariance_operator_coefficients(
                        model, delays, rho, used_persistence
                    )
                    exact = first_stability_boundary(
                        covariance["markov"],
                        theorem["sharp_safe_eta"],
                    )
                    sharp_radius, sharp_residual = (
                        spectral_radius_with_residual(
                            covariance["markov"],
                            theorem["sharp_safe_eta"],
                        )
                    )
                    rate_radius, rate_residual = (
                        spectral_radius_with_residual(
                            covariance["markov"],
                            theorem["rate_eta"],
                        )
                    )
                    window = 2 * delay + 1
                    rows.append(
                        {
                            "original_persistence": persistence,
                            "decorrelation_gap": gap,
                            "target_delta": target_delta,
                            "actual_delta": delta,
                            "used_persistence": used_persistence,
                            "homogeneous_delay": delay,
                            "rho": rho,
                            "num_agents": num_agents,
                            **theorem,
                            "exact_boundary": exact["boundary"],
                            "exact_below_radius": exact["below_radius"],
                            "exact_below_residual": exact[
                                "below_residual"
                            ],
                            "exact_above_radius": exact["above_radius"],
                            "exact_above_residual": exact[
                                "above_residual"
                            ],
                            "sharp_radius": sharp_radius,
                            "sharp_residual": sharp_residual,
                            "rate_radius": rate_radius,
                            "rate_residual": rate_residual,
                            "sharp_to_exact_ratio": (
                                theorem["sharp_safe_eta"]
                                / exact["boundary"]
                            ),
                            "rate_to_exact_ratio": (
                                theorem["rate_eta"]
                                / exact["boundary"]
                            ),
                            "exact_window_contraction": (
                                rate_radius ** window
                            ),
                            "theorem_envelope_slack": (
                                theorem["rate_contraction_coefficient"]
                                - rate_radius ** window
                            ),
                        }
                    )
                    completed += 1
                    print(
                        "completed {0}/{1}: p={2}, D={3}, rho={4}, "
                        "q={5}".format(
                            completed,
                            total,
                            persistence,
                            delay,
                            rho,
                            num_agents,
                        ),
                        flush=True,
                    )
    return pd.DataFrame(rows)


def evaluate_gates(frame: pd.DataFrame) -> Dict[str, object]:
    maximum_residual = float(
        frame[
            [
                "exact_below_residual",
                "exact_above_residual",
                "sharp_residual",
                "rate_residual",
            ]
        ]
        .to_numpy()
        .max()
    )
    numerical = {
        "sharp_root_and_rate_solve": {
            "pass": bool(
                (
                    np.abs(frame["sharp_root_factor"] - 1.0) <= 1e-9
                ).all()
                and (frame["rate_eta"] > 0.0).all()
                and (frame["rate_eta"] < frame["sharp_root"]).all()
                and (
                    frame["rate_contraction_coefficient"]
                    <= frame["sharp_safe_factor"] ** 2
                ).all()
            ),
            "maximum_root_factor_error": float(
                np.abs(frame["sharp_root_factor"] - 1.0).max()
            ),
        },
        "exact_boundary_validity": {
            "pass": bool(
                len(frame) == 72
                and (frame["exact_boundary"] > 0.0).all()
                and (frame["exact_below_radius"] < 1.0).all()
                and (frame["exact_above_radius"] > 1.0).all()
                and maximum_residual <= 1e-7
            ),
            "observed_cells": int(len(frame)),
            "maximum_eigen_residual": maximum_residual,
        },
    }
    sharp_safe = bool((frame["sharp_radius"] < 1.0).all())
    nonvacuous_cells = int(
        (frame["sharp_to_exact_ratio"] >= 0.08).sum()
    )
    nonvacuous = bool(
        nonvacuous_cells >= 66
        and (frame["sharp_to_exact_ratio"] <= 1.0).all()
    )
    rate_safe = bool((frame["rate_radius"] < 1.0).all())
    envelope = bool((frame["theorem_envelope_slack"] >= -1e-8).all())
    delayed = frame[frame["homogeneous_delay"] == 2]
    improvement = bool(
        (delayed["sharp_safe_eta"] > delayed["coarse_eta"]).all()
    )
    scientific = {
        "sharp_boundary_exact_safety": {
            "pass": sharp_safe,
            "largest_sharp_radius": float(
                frame["sharp_radius"].max()
            ),
        },
        "sharp_boundary_nonvacuity": {
            "pass": nonvacuous,
            "cells_ratio_at_least_0p08": nonvacuous_cells,
            "smallest_sharp_to_exact_ratio": float(
                frame["sharp_to_exact_ratio"].min()
            ),
            "largest_sharp_to_exact_ratio": float(
                frame["sharp_to_exact_ratio"].max()
            ),
        },
        "rate_step_exact_safety": {
            "pass": rate_safe,
            "largest_rate_radius": float(
                frame["rate_radius"].max()
            ),
        },
        "theorem_envelope_validity": {
            "pass": envelope,
            "minimum_envelope_slack": float(
                frame["theorem_envelope_slack"].min()
            ),
        },
        "delay_improvement": {
            "pass": improvement,
            "smallest_sharp_to_coarse_step_ratio_delayed": float(
                (
                    delayed["sharp_safe_eta"] / delayed["coarse_eta"]
                ).min()
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
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for axis, delay in zip(axes, MAX_DELAYS_MARKOV):
        subset = frame[frame["homogeneous_delay"] == delay]
        for persistence in REGIME_PERSISTENCES:
            line = subset[
                np.isclose(subset["original_persistence"], persistence)
                & np.isclose(subset["rho"], 0.9)
            ].sort_values("num_agents")
            axis.plot(
                line["num_agents"],
                line["sharp_to_exact_ratio"],
                marker="o",
                label=rf"$p={persistence:g}$",
            )
        axis.axhline(0.08, color="black", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("agent count")
        axis.set_title(rf"delay {delay}, $\rho=0.9$")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("sharp theorem boundary / exact boundary")
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "fig1_sharp_boundary_tightness.png", dpi=220
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    delayed = frame[frame["homogeneous_delay"] == 2]
    axis.scatter(
        delayed["coarse_eta"],
        delayed["sharp_safe_eta"],
        c=delayed["original_persistence"],
        cmap="viridis",
        alpha=0.8,
    )
    limit = float(
        max(delayed["coarse_eta"].max(), delayed["sharp_safe_eta"].max())
    )
    axis.plot((0.0, limit), (0.0, limit), "k--", linewidth=1)
    axis.set_xlabel("coarse theorem step")
    axis.set_ylabel("sharp theorem step")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_dir / "fig2_coarse_vs_sharp.png", dpi=220
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = run_cells(registered_expanding_td_model())
    gates = evaluate_gates(frame)
    frame.to_csv(args.output_dir / "sharp_boundaries.csv", index=False)
    summary = {
        "experiment": "EXP-008E",
        "status": (
            "VALID"
            if gates["numerical_overall_pass"]
            else "INVALID_NUMERICS"
        ),
        "registered_cells": 72,
        "gates": gates,
        "artifacts": [
            "sharp_boundaries.csv",
            "summary.json",
            "fig1_sharp_boundary_tightness.png",
            "fig2_coarse_vs_sharp.png",
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

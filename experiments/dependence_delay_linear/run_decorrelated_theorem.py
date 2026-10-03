"""Run preregistered EXP-008D proof-derived decorrelation audit."""

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
    spectral_radius_with_residual,
    theorem_safe_step,
    thinned_persistence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "decorrelated_theorem",
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
    total = (
        len(AGENT_COUNTS_EXPANDING)
        * len(MAX_DELAYS_MARKOV)
        * len(REGIME_PERSISTENCES)
        * len(CORRELATIONS_MARKOV)
    )
    completed = 0
    for persistence in REGIME_PERSISTENCES:
        gap = minimum_decorrelation_gap(persistence, target_delta)
        delta = mixing_tv_after_gap(persistence, gap)
        used_persistence = thinned_persistence(persistence, gap)
        for delay in MAX_DELAYS_MARKOV:
            for rho in CORRELATIONS_MARKOV:
                for num_agents in AGENT_COUNTS_EXPANDING:
                    delays = homogeneous_delays(num_agents, delay)
                    theorem = theorem_safe_step(
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
                        covariance["markov"], theorem["eta"]
                    )
                    radius, residual = spectral_radius_with_residual(
                        covariance["markov"], theorem["eta"]
                    )
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
                            "exact_operator_dimension": exact[
                                "operator_dimension"
                            ],
                            "theorem_radius": radius,
                            "theorem_residual": residual,
                            "theorem_to_exact_ratio": (
                                theorem["eta"] / exact["boundary"]
                            ),
                            "progress_per_raw_transition": (
                                -np.log(radius) / gap
                            ),
                        }
                    )
                    completed += 1
                    print(
                        "completed {0}/{1}: p={2}, b={3}, D={4}, "
                        "rho={5}, q={6}".format(
                            completed,
                            total,
                            persistence,
                            gap,
                            delay,
                            rho,
                            num_agents,
                        ),
                        flush=True,
                    )
    return pd.DataFrame(rows)


def evaluate_gates(frame: pd.DataFrame) -> Dict[str, object]:
    gap_table = (
        frame[
            [
                "original_persistence",
                "decorrelation_gap",
                "target_delta",
                "actual_delta",
            ]
        ]
        .drop_duplicates()
        .sort_values("original_persistence")
    )
    minimal = True
    for row in gap_table.itertuples(index=False):
        if row.actual_delta > row.target_delta:
            minimal = False
        if row.decorrelation_gap > 1:
            previous = mixing_tv_after_gap(
                row.original_persistence, row.decorrelation_gap - 1
            )
            if previous <= row.target_delta:
                minimal = False
    maximum_residual = float(
        frame[
            [
                "exact_below_residual",
                "exact_above_residual",
                "theorem_residual",
            ]
        ]
        .to_numpy()
        .max()
    )
    numerical = {
        "boundary_validity": {
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
        "minimal_gap_and_polynomial": {
            "pass": bool(
                minimal
                and (
                    frame["polynomial_at_eta"]
                    <= frame["right_hand_side"]
                ).all()
            ),
            "gap_table": gap_table.to_dict(orient="records"),
            "largest_polynomial_to_rhs_ratio": float(
                (
                    frame["polynomial_at_eta"]
                    / frame["right_hand_side"]
                ).max()
            ),
        },
    }
    safe = bool((frame["theorem_radius"] < 1.0).all())
    strict = bool((frame["contraction_coefficient"] < 1.0).all())
    nonvacuous_cells = int(
        (frame["theorem_to_exact_ratio"] >= 0.05).sum()
    )
    nonvacuous = bool(
        nonvacuous_cells >= 66
        and (frame["theorem_to_exact_ratio"] <= 1.0).all()
    )
    gaps = gap_table["decorrelation_gap"].to_numpy()
    adaptation = bool(np.all(np.diff(gaps) > 0))
    gains = frame.pivot_table(
        index=[
            "original_persistence",
            "homogeneous_delay",
            "rho",
        ],
        columns="num_agents",
        values="exact_boundary",
    )
    high = gains.xs(0.9, level="rho")
    high_gain = high[32] / high[16]
    saturation = bool((high_gain <= 1.05).all())
    scientific = {
        "exact_safety": {
            "pass": safe,
            "largest_theorem_radius": float(
                frame["theorem_radius"].max()
            ),
        },
        "strict_theorem_slack": {
            "pass": strict,
            "largest_contraction_coefficient": float(
                frame["contraction_coefficient"].max()
            ),
        },
        "nonvacuity": {
            "pass": nonvacuous,
            "cells_ratio_at_least_0p05": nonvacuous_cells,
            "smallest_theorem_to_exact_ratio": float(
                frame["theorem_to_exact_ratio"].min()
            ),
            "largest_theorem_to_exact_ratio": float(
                frame["theorem_to_exact_ratio"].max()
            ),
        },
        "mixing_adaptation": {
            "pass": adaptation,
            "selected_gaps": [
                int(value) for value in gaps.tolist()
            ],
        },
        "correlation_limited_participation": {
            "pass": saturation,
            "largest_rho0p9_q32_to_q16_gain": float(
                high_gain.max()
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
                line["theorem_to_exact_ratio"],
                marker="o",
                label=rf"$p={persistence:g}$",
            )
        axis.axhline(0.05, color="black", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("agent count")
        axis.set_title(rf"delay {delay}, $\rho=0.9$")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("theorem step / exact boundary")
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "fig1_theorem_tightness.png", dpi=220
    )
    plt.close(figure)

    best = (
        frame.groupby(
            ["original_persistence", "homogeneous_delay", "rho"],
            as_index=False,
        )
        .apply(
            lambda group: group.loc[
                group["progress_per_raw_transition"].idxmax()
            ]
        )
        .reset_index(drop=True)
    )
    figure, axis = plt.subplots(figsize=(7.3, 4.5))
    for rho, marker in ((0.0, "o"), (0.9, "s")):
        line = best[np.isclose(best["rho"], rho)]
        axis.scatter(
            line["original_persistence"]
            + 0.003 * line["homogeneous_delay"],
            line["num_agents"],
            marker=marker,
            label=rf"$\rho={rho:g}$",
            s=55,
        )
    axis.set_yscale("log", base=2)
    axis.set_xlabel("original persistence (delay jittered)")
    axis.set_ylabel("best q by theorem progress / raw transition")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "fig2_best_participation.png", dpi=220
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = registered_expanding_td_model()
    frame = run_cells(model)
    gates = evaluate_gates(frame)
    frame.to_csv(
        args.output_dir / "decorrelated_boundaries.csv", index=False
    )
    summary = {
        "experiment": "EXP-008D",
        "status": (
            "VALID"
            if gates["numerical_overall_pass"]
            else "INVALID_NUMERICS"
        ),
        "registered_cells": 72,
        "gates": gates,
        "artifacts": [
            "decorrelated_boundaries.csv",
            "summary.json",
            "fig1_theorem_tightness.png",
            "fig2_best_participation.png",
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

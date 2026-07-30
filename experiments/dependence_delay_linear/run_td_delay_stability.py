"""Run EXP-007B active delayed-TD stability experiment."""

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

from linear_model import make_agent_delays
from linear_td_correlation import (
    LinearTDConfig,
    build_mrp,
    generate_base_paths,
    observed_transition_pairs,
)
from td_delay_stability import (
    AGENT_COUNTS_STABILITY,
    CORRELATIONS_STABILITY,
    DIVERGENCE_THRESHOLD,
    HORIZON,
    MAX_DELAYS_STABILITY,
    MULTIPLIERS,
    build_boundary_table,
    build_mean_delay_transition,
    simulate_stability_run,
    spectral_radius,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "td_delay_stability",
    )
    parser.add_argument("--num-seeds", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=20261030)
    return parser.parse_args()


def boundary_lookup(boundaries: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    return {
        (int(row.num_agents), int(row.max_delay)): float(row.critical_eta)
        for row in boundaries.itertuples()
    }


def build_spectral_table(
    boundaries: pd.DataFrame,
    mrp: Dict[str, np.ndarray],
    config: LinearTDConfig,
) -> pd.DataFrame:
    lookup = boundary_lookup(boundaries)
    rows: List[Dict[str, object]] = []
    for q in AGENT_COUNTS_STABILITY:
        for max_delay in MAX_DELAYS_STABILITY:
            full_delays = make_agent_delays(
                max_agents=config.num_agents,
                max_delay=max_delay,
                exponent=config.delay_exponent,
            )
            selected_delays = full_delays[:q]
            critical = lookup[(q, max_delay)]
            for multiplier in MULTIPLIERS:
                eta = multiplier * critical
                radius = spectral_radius(
                    build_mean_delay_transition(
                        mrp["a_matrix"], selected_delays, eta
                    )
                )
                rows.append(
                    {
                        "num_agents": q,
                        "max_delay": max_delay,
                        "multiplier": multiplier,
                        "eta": eta,
                        "spectral_radius": radius,
                        "predicted_stable": bool(radius < 1.0),
                    }
                )
    return pd.DataFrame(rows)


def run_experiment(
    config: LinearTDConfig,
    num_seeds: int,
    base_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mrp = build_mrp(config)
    boundaries = pd.DataFrame(
        build_boundary_table(mrp["a_matrix"], config)
    )
    spectral = build_spectral_table(boundaries, mrp, config)
    lookup = boundary_lookup(boundaries)
    rows: List[Dict[str, object]] = []
    for seed_index in range(num_seeds):
        seed = base_seed + seed_index
        paths = generate_base_paths(seed, mrp, config)
        for rho in CORRELATIONS_STABILITY:
            current, following = observed_transition_pairs(paths, rho)
            for q in AGENT_COUNTS_STABILITY:
                for max_delay in MAX_DELAYS_STABILITY:
                    critical = lookup[(q, max_delay)]
                    for multiplier in MULTIPLIERS:
                        eta = multiplier * critical
                        result = simulate_stability_run(
                            current,
                            following,
                            mrp,
                            max_delay,
                            q,
                            eta,
                            config,
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "seed_index": seed_index,
                                "rho": rho,
                                "num_agents": q,
                                "max_delay": max_delay,
                                "policy": "relative_boundary",
                                "multiplier": multiplier,
                                "eta": eta,
                                "local_critical_eta": critical,
                                "eta_to_local_boundary": multiplier,
                                **result,
                            }
                        )
                    if max_delay > 0:
                        blind_eta = 0.8 * lookup[(q, 0)]
                        result = simulate_stability_run(
                            current,
                            following,
                            mrp,
                            max_delay,
                            q,
                            blind_eta,
                            config,
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "seed_index": seed_index,
                                "rho": rho,
                                "num_agents": q,
                                "max_delay": max_delay,
                                "policy": "delay_blind",
                                "multiplier": float("nan"),
                                "eta": blind_eta,
                                "local_critical_eta": critical,
                                "eta_to_local_boundary": (
                                    blind_eta / critical
                                ),
                                **result,
                            }
                        )
        if (seed_index + 1) % max(1, min(4, num_seeds)) == 0:
            print(
                "completed {0}/{1} paired seeds".format(
                    seed_index + 1, num_seeds
                ),
                flush=True,
            )
    return pd.DataFrame(rows), boundaries, spectral


def evaluate_gates(
    runs: pd.DataFrame,
    boundaries: pd.DataFrame,
    spectral: pd.DataFrame,
    num_seeds: int,
) -> Dict[str, object]:
    pivot = boundaries.pivot(
        index="num_agents", columns="max_delay", values="critical_eta"
    )
    ratios = {
        q: float(pivot.loc[q, 32] / pivot.loc[q, 0])
        for q in (16, 32)
    }
    active = bool(all(value <= 0.35 for value in ratios.values()))
    low = spectral[spectral["multiplier"] <= 0.95]
    high = spectral[spectral["multiplier"] >= 1.05]
    separation = bool(
        (low["spectral_radius"] < 1.0).all()
        and (high["spectral_radius"] > 1.0).all()
    )
    relative = runs[runs["policy"] == "relative_boundary"]
    stable = relative[np.isclose(relative["multiplier"], 0.8)]
    stable_medians = stable.groupby(
        ["rho", "num_agents", "max_delay"]
    )["final_error"].median()
    stable_valid = bool(
        (~stable["crossed_threshold"]).all()
        and (stable_medians < 1.0).all()
    )
    unstable = relative[
        np.isclose(relative["multiplier"], 1.2)
        & (relative["max_delay"] > 0)
    ]
    unstable_fraction = float(unstable["crossed_threshold"].mean())
    monte_carlo = bool(stable_valid and unstable_fraction >= 0.90)
    adaptive = stable[
        (stable["max_delay"] == 32)
        & (stable["num_agents"].isin([16, 32]))
    ]
    blind = runs[
        (runs["policy"] == "delay_blind")
        & (runs["max_delay"] == 32)
        & (runs["num_agents"].isin([16, 32]))
    ]
    blind_fraction = float(blind["crossed_threshold"].mean())
    adaptive_value = bool(
        (~adaptive["crossed_threshold"]).all()
        and blind_fraction >= 0.90
    )
    medians = (
        stable.groupby(["rho", "num_agents", "max_delay"], as_index=False)[
            "final_error"
        ]
        .median()
        .pivot(
            index=["num_agents", "max_delay"],
            columns="rho",
            values="final_error",
        )
    )
    correlation_fraction = float(
        (medians[0.9] >= medians[0.0]).mean()
    )
    correlation_separation = bool(correlation_fraction >= 0.80)
    expected_relative = (
        num_seeds
        * len(CORRELATIONS_STABILITY)
        * len(AGENT_COUNTS_STABILITY)
        * len(MAX_DELAYS_STABILITY)
        * len(MULTIPLIERS)
    )
    expected_blind = (
        num_seeds
        * len(CORRELATIONS_STABILITY)
        * len(AGENT_COUNTS_STABILITY)
        * 2
    )
    expected_runs = expected_relative + expected_blind
    validity = bool(
        len(runs) == expected_runs
        and runs["finite"].all()
        and np.isfinite(runs["final_error"]).all()
        and (
            (runs["crossed_threshold"] & (runs["crossing_time"] > 0))
            | (~runs["crossed_threshold"] & (runs["crossing_time"] == -1))
        ).all()
        and len(boundaries)
        == len(AGENT_COUNTS_STABILITY) * len(MAX_DELAYS_STABILITY)
    )
    gates: Dict[str, object] = {
        "active_boundary": {
            "pass": active,
            "critical_eta_ratio_d32_to_d0": ratios,
        },
        "exact_spectral_separation": {
            "pass": separation,
            "stable_cells": int(len(low)),
            "unstable_cells": int(len(high)),
        },
        "monte_carlo_boundary_agreement": {
            "pass": monte_carlo,
            "stable_runs_no_crossing": bool(
                (~stable["crossed_threshold"]).all()
            ),
            "all_stable_cell_medians_below_initial": bool(
                (stable_medians < 1.0).all()
            ),
            "delayed_m1p2_crossing_fraction": unstable_fraction,
        },
        "delay_adaptive_value": {
            "pass": adaptive_value,
            "adaptive_runs_no_crossing": bool(
                (~adaptive["crossed_threshold"]).all()
            ),
            "delay_blind_crossing_fraction": blind_fraction,
        },
        "correlation_stability_separation": {
            "pass": correlation_separation,
            "high_correlation_error_not_smaller_fraction": (
                correlation_fraction
            ),
            "comparison_count": int(len(medians)),
        },
        "accounting_determinism_numerical_validity": {
            "pass": validity,
            "observed_runs": int(len(runs)),
            "expected_runs": int(expected_runs),
            "divergence_threshold": DIVERGENCE_THRESHOLD,
            "horizon": HORIZON,
        },
    }
    gates["overall"] = {
        "pass": bool(all(value["pass"] for value in gates.values())),
        "criterion": "all six active delay-stability gates pass",
    }
    return gates


def plot_outputs(
    runs: pd.DataFrame,
    boundaries: pd.DataFrame,
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
    for q in AGENT_COUNTS_STABILITY:
        selected = boundaries[boundaries["num_agents"] == q]
        ax.plot(
            selected["max_delay"],
            selected["critical_eta"],
            marker="o",
            label="q={0}".format(q),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Registered maximum delay D")
    ax.set_ylabel("Exact critical step size")
    ax.set_title("Delayed TD stability boundary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_critical_eta.png")
    plt.close(fig)

    relative = runs[runs["policy"] == "relative_boundary"]
    crossing = (
        relative.groupby(
            ["max_delay", "multiplier"], as_index=False
        )["crossed_threshold"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for max_delay in MAX_DELAYS_STABILITY:
        selected = crossing[crossing["max_delay"] == max_delay]
        ax.plot(
            selected["multiplier"],
            selected["crossed_threshold"],
            marker="o",
            label="D={0}".format(max_delay),
        )
    ax.axvline(1.0, color="black", linestyle="--")
    ax.set_xlabel("Step size / exact critical step size")
    ax.set_ylabel("Threshold-crossing fraction")
    ax.set_title("Monte Carlo agreement with exact boundary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_boundary_crossings.png")
    plt.close(fig)

    selected = runs[
        (
            (runs["policy"] == "relative_boundary")
            & np.isclose(runs["multiplier"], 0.8)
        )
        | (runs["policy"] == "delay_blind")
    ]
    summary = (
        selected.groupby(
            ["policy", "rho", "num_agents", "max_delay"], as_index=False
        )["final_error"]
        .median()
    )
    summary["label"] = (
        summary["policy"]
        + ", rho="
        + summary["rho"].map(lambda value: "{0:g}".format(value))
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for ax, q in zip(axes, (16, 32)):
        for label, group in summary[summary["num_agents"] == q].groupby(
            "label"
        ):
            ax.plot(
                group["max_delay"],
                group["final_error"],
                marker="o",
                label=label,
            )
        ax.set_yscale("log")
        ax.set_title("q={0}".format(q))
        ax.set_xlabel("Maximum delay D")
    axes[0].set_ylabel("Median final squared error")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_delay_adaptive.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = LinearTDConfig()
    print(
        "Starting EXP-007B with {0} paired seeds".format(args.num_seeds),
        flush=True,
    )
    runs, boundaries, spectral = run_experiment(
        config, args.num_seeds, args.base_seed
    )
    gates = evaluate_gates(
        runs, boundaries, spectral, args.num_seeds
    )
    runs.to_csv(args.output_dir / "stability_runs.csv", index=False)
    boundaries.to_csv(
        args.output_dir / "critical_boundaries.csv", index=False
    )
    spectral.to_csv(
        args.output_dir / "spectral_classification.csv", index=False
    )
    plot_outputs(runs, boundaries, args.output_dir)
    summary = {
        "experiment_id": "EXP-007B-td-delay-stability",
        "status": "COMPLETED_PENDING_REPRODUCTION",
        "num_paired_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        "horizon": HORIZON,
        "divergence_threshold": DIVERGENCE_THRESHOLD,
        "multipliers": list(MULTIPLIERS),
        "agent_counts": list(AGENT_COUNTS_STABILITY),
        "max_delays": list(MAX_DELAYS_STABILITY),
        "correlations": list(CORRELATIONS_STABILITY),
        "gates": gates,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(gates, indent=2), flush=True)
    print("Outputs written to {0}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()

"""Run EXP-005A, the budget-matched participation falsification surface."""

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

from budget_participation import (
    AGENT_COUNTS,
    ALIGNMENTS,
    MAX_DELAYS,
    RHO_VALUES,
    SELECTION_RULES,
    BudgetConfig,
    budget_horizon,
    per_update_cost,
    resource_specs,
    selected_delays,
)
from linear_model import ModelConfig, exact_risk, monte_carlo_risk


PRIMARY_RESOURCE = "message_overhead_4"
EXTREME_CELLS: Tuple[Tuple[float, int], ...] = (
    (0.0, 4),
    (0.0, 16),
    (0.9, 4),
    (0.9, 16),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "budget_participation",
    )
    parser.add_argument("--mc-replications", type=int, default=10000)
    parser.add_argument("--base-seed", type=int, default=20260729)
    return parser.parse_args()


def build_surface(config: BudgetConfig) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    specs = resource_specs(config)
    for alignment in ALIGNMENTS:
        for selection_rule in SELECTION_RULES:
            for rho in RHO_VALUES:
                for max_delay in MAX_DELAYS:
                    for resource_name, spec in specs.items():
                        for num_agents in AGENT_COUNTS:
                            delays = selected_delays(
                                max_delay=max_delay,
                                selected_count=num_agents,
                                rule=selection_rule,
                                config=config,
                            )
                            horizon = budget_horizon(
                                selected_count=num_agents,
                                delays=delays,
                                spec=spec,
                            )
                            update_cost = per_update_cost(
                                selected_count=num_agents,
                                delays=delays,
                                spec=spec,
                            )
                            model_config = ModelConfig(
                                horizon=horizon,
                                common_noise_alignment=alignment,
                            )
                            for eta in config.eta_grid:
                                metrics = exact_risk(
                                    eta=float(eta),
                                    rho=rho,
                                    num_agents=num_agents,
                                    delays=delays,
                                    config=model_config,
                                )
                                rows.append(
                                    {
                                        "alignment": alignment,
                                        "selection_rule": selection_rule,
                                        "rho": rho,
                                        "configured_max_delay": max_delay,
                                        "resource": resource_name,
                                        "resource_kind": spec["kind"],
                                        "resource_budget": spec["budget"],
                                        "num_agents": num_agents,
                                        "eta": float(eta),
                                        "horizon": horizon,
                                        "per_update_cost": update_cost,
                                        "selected_mean_delay": float(
                                            np.mean(delays)
                                        ),
                                        "selected_max_delay": int(
                                            np.max(delays)
                                        ),
                                        **metrics,
                                    }
                                )
    return pd.DataFrame(rows)


def select_optimal_actions(surface: pd.DataFrame) -> pd.DataFrame:
    stable = surface[
        surface["stable"] & np.isfinite(surface["finite_mse"])
    ].copy()
    group_columns = [
        "alignment",
        "selection_rule",
        "rho",
        "configured_max_delay",
        "resource",
    ]
    rows: List[Dict[str, object]] = []
    for keys, subset in stable.groupby(group_columns, sort=True):
        best = subset.loc[subset["finite_mse"].idxmin()]
        all_agents = subset[subset["num_agents"] == max(AGENT_COUNTS)]
        best_all_agents = all_agents.loc[all_agents["finite_mse"].idxmin()]
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update(
            {
                "optimal_num_agents": int(best["num_agents"]),
                "optimal_eta": float(best["eta"]),
                "optimal_horizon": int(best["horizon"]),
                "optimal_mse": float(best["finite_mse"]),
                "all_agents_eta": float(best_all_agents["eta"]),
                "all_agents_horizon": int(best_all_agents["horizon"]),
                "all_agents_mse": float(best_all_agents["finite_mse"]),
                "optimal_to_all_agents_ratio": float(
                    best["finite_mse"] / best_all_agents["finite_mse"]
                ),
                "selected_stable": bool(best["stable"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def monte_carlo_validation(
    optimal: pd.DataFrame,
    config: BudgetConfig,
    replications: int,
    base_seed: int,
) -> pd.DataFrame:
    primary = optimal[
        (optimal["alignment"] == "server_time")
        & (optimal["selection_rule"] == "fastest")
        & (optimal["resource"] == PRIMARY_RESOURCE)
    ]
    rows: List[Dict[str, object]] = []
    action_index = 0
    for rho, max_delay in EXTREME_CELLS:
        cell = primary[
            (primary["rho"] == rho)
            & (primary["configured_max_delay"] == max_delay)
        ].iloc[0]
        actions = [
            (
                "optimal",
                int(cell["optimal_num_agents"]),
                float(cell["optimal_eta"]),
                int(cell["optimal_horizon"]),
                float(cell["optimal_mse"]),
            ),
            (
                "all_agents",
                max(AGENT_COUNTS),
                float(cell["all_agents_eta"]),
                int(cell["all_agents_horizon"]),
                float(cell["all_agents_mse"]),
            ),
        ]
        for role, num_agents, eta, horizon, exact_mse in actions:
            delays = selected_delays(
                max_delay=max_delay,
                selected_count=num_agents,
                rule="fastest",
                config=config,
            )
            metrics = monte_carlo_risk(
                eta=eta,
                rho=rho,
                num_agents=num_agents,
                delays=delays,
                config=ModelConfig(
                    horizon=horizon,
                    common_noise_alignment="server_time",
                ),
                num_replications=replications,
                seed=base_seed + action_index,
            )
            relative_difference = abs(metrics["mc_mse"] - exact_mse) / exact_mse
            tolerance = max(
                0.05,
                3.0 * metrics["mc_standard_error"] / exact_mse,
            )
            rows.append(
                {
                    "rho": rho,
                    "configured_max_delay": max_delay,
                    "role": role,
                    "num_agents": num_agents,
                    "eta": eta,
                    "horizon": horizon,
                    "exact_mse": exact_mse,
                    **metrics,
                    "relative_difference": relative_difference,
                    "allowed_relative_difference": tolerance,
                    "pass": bool(relative_difference <= tolerance),
                }
            )
            action_index += 1
    return pd.DataFrame(rows)


def evaluate_gates(
    optimal: pd.DataFrame,
    monte_carlo: pd.DataFrame,
) -> Dict[str, object]:
    primary = optimal[
        (optimal["alignment"] == "server_time")
        & (optimal["selection_rule"] == "fastest")
        & (optimal["resource"] == PRIMARY_RESOURCE)
    ]
    independent = primary[primary["rho"] == 0.0]
    high = primary[primary["rho"] == 0.9]
    wallclock_hard = optimal[
        (optimal["alignment"] == "server_time")
        & (optimal["selection_rule"] == "fastest")
        & (optimal["resource"] == "wallclock")
        & (optimal["rho"].isin([0.6, 0.9]))
    ]
    uniform_high = optimal[
        (optimal["alignment"] == "server_time")
        & (optimal["selection_rule"] == "uniform_rank")
        & (optimal["resource"] == PRIMARY_RESOURCE)
        & (optimal["rho"] == 0.9)
    ]

    regime_pass = bool(
        (independent["optimal_num_agents"] >= 16).all()
        and (high["optimal_num_agents"] <= 8).all()
    )
    high_gain_pass = bool(
        (high["optimal_to_all_agents_ratio"] <= 0.90).all()
    )
    wallclock_cell_passes = (
        (wallclock_hard["optimal_num_agents"] <= 8)
        & (wallclock_hard["optimal_to_all_agents_ratio"] <= 0.90)
    )
    wallclock_pass = bool(wallclock_cell_passes.sum() >= 3)
    uniform_cell_passes = (
        (uniform_high["optimal_num_agents"] <= 8)
        & (uniform_high["optimal_to_all_agents_ratio"] <= 0.95)
    )
    uniform_pass = bool(uniform_cell_passes.any())
    numerical_pass = bool(
        optimal["selected_stable"].all()
        and np.isfinite(optimal["optimal_mse"]).all()
        and monte_carlo["pass"].all()
    )
    gates = {
        "participation_regime_change": {
            "pass": regime_pass,
            "criterion": (
                "primary message q>=16 at rho=0 and q<=8 at rho=0.9"
            ),
            "independent_q": independent[
                ["configured_max_delay", "optimal_num_agents"]
            ].to_dict(orient="records"),
            "high_correlation_q": high[
                ["configured_max_delay", "optimal_num_agents"]
            ].to_dict(orient="records"),
        },
        "material_high_correlation_gain": {
            "pass": high_gain_pass,
            "criterion": "both rho=0.9 ratios <= 0.90",
            "cells": high[
                [
                    "configured_max_delay",
                    "optimal_num_agents",
                    "optimal_to_all_agents_ratio",
                ]
            ].to_dict(orient="records"),
        },
        "wallclock_relevance": {
            "pass": wallclock_pass,
            "criterion": "at least 3/4 hard cells have q<=8 and ratio<=0.90",
            "passing_cells": int(wallclock_cell_passes.sum()),
            "total_cells": int(len(wallclock_cell_passes)),
        },
        "not_only_straggler_deletion": {
            "pass": uniform_pass,
            "criterion": "at least one uniform-rank rho=0.9 cell q<=8, ratio<=0.95",
            "passing_cells": int(uniform_cell_passes.sum()),
            "total_cells": int(len(uniform_cell_passes)),
        },
        "numerical_validity": {
            "pass": numerical_pass,
            "criterion": "stable finite optima and all MC checks pass",
            "mc_passes": int(monte_carlo["pass"].sum()),
            "mc_total": int(len(monte_carlo)),
        },
    }
    gates["overall"] = {
        "pass": bool(all(value["pass"] for value in gates.values())),
        "criterion": "all five pre-registered gates pass",
    }
    return gates


def plot_outputs(
    optimal: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )
    primary = optimal[
        (optimal["alignment"] == "server_time")
        & (optimal["selection_rule"] == "fastest")
        & (optimal["resource"].isin([PRIMARY_RESOURCE, "wallclock"]))
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharey=True)
    for axis, resource in zip(
        axes, [PRIMARY_RESOURCE, "wallclock"]
    ):
        subset_resource = primary[primary["resource"] == resource]
        for max_delay in MAX_DELAYS:
            subset = subset_resource[
                subset_resource["configured_max_delay"] == max_delay
            ]
            axis.plot(
                subset["rho"],
                subset["optimal_num_agents"],
                marker="o",
                label="max delay={0}".format(max_delay),
            )
        axis.set_yscale("log", base=2)
        axis.set_yticks(AGENT_COUNTS)
        axis.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_xlabel("Cross-agent correlation")
        axis.set_title(resource.replace("_", " "))
    axes[0].set_ylabel("Jointly optimal participating agents")
    axes[1].legend()
    fig.suptitle("Resource-matched optimal participation")
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_optimal_participation.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharey=True)
    for axis, selection_rule in zip(axes, SELECTION_RULES):
        subset_rule = optimal[
            (optimal["alignment"] == "server_time")
            & (optimal["selection_rule"] == selection_rule)
            & (optimal["resource"] == PRIMARY_RESOURCE)
        ]
        for max_delay in MAX_DELAYS:
            subset = subset_rule[
                subset_rule["configured_max_delay"] == max_delay
            ]
            axis.plot(
                subset["rho"],
                subset["optimal_to_all_agents_ratio"],
                marker="o",
                label="max delay={0}".format(max_delay),
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.axhline(0.9, color="tab:red", linestyle=":", linewidth=1)
        axis.set_xlabel("Cross-agent correlation")
        axis.set_title(selection_rule.replace("_", " "))
    axes[0].set_ylabel("Optimal MSE / best all-agent MSE")
    axes[1].legend()
    fig.suptitle("Benefit of participation optimization at matched message budget")
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_risk_ratio.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = BudgetConfig()
    print("Starting EXP-005A exact surface", flush=True)
    surface = build_surface(config)
    print("Exact surface complete: {0} rows".format(len(surface)), flush=True)
    optimal = select_optimal_actions(surface)
    monte_carlo = monte_carlo_validation(
        optimal=optimal,
        config=config,
        replications=args.mc_replications,
        base_seed=args.base_seed,
    )
    gates = evaluate_gates(optimal, monte_carlo)

    surface.to_csv(args.output_dir / "surface.csv", index=False)
    optimal.to_csv(args.output_dir / "optimal_actions.csv", index=False)
    monte_carlo.to_csv(
        args.output_dir / "monte_carlo_validation.csv", index=False
    )
    plot_outputs(optimal, args.output_dir)

    summary = {
        "experiment_id": "EXP-005A-budget-participation-surface",
        "status": "COMPLETED_PENDING_REPRODUCTION",
        "mc_replications": args.mc_replications,
        "base_seed": args.base_seed,
        "config": {
            "agent_counts": list(AGENT_COUNTS),
            "rho_values": list(RHO_VALUES),
            "max_delays": list(MAX_DELAYS),
            "alignments": list(ALIGNMENTS),
            "selection_rules": list(SELECTION_RULES),
            "eta_grid": config.eta_grid.tolist(),
            "resource_specs": resource_specs(config),
        },
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
    print("Outputs written to {0}".format(args.output_dir.resolve()), flush=True)


if __name__ == "__main__":
    main()


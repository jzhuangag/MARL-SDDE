"""Run EXP-005B, the online probe-charging participation controller."""

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from online_participation import (
    POLICIES,
    SCENARIOS,
    FiniteBudgetProxyCache,
    OnlineConfig,
    build_noise_table,
    generate_factor_paths,
    simulate_policy,
)


MAX_DELAYS: Tuple[int, ...] = (4, 16)
CORRELATED_SCENARIOS: Tuple[str, ...] = ("clustered", "global", "mixed")
FIXED_POLICIES: Tuple[str, ...] = (
    "all_agents_adaptive_eta",
    "fixed_q8_adaptive_eta",
    "fixed_q1_adaptive_eta",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "online_participation",
    )
    parser.add_argument("--num-seeds", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=20260729)
    parser.add_argument("--bootstrap-replications", type=int, default=2000)
    return parser.parse_args()


def run_experiment(
    config: OnlineConfig,
    num_seeds: int,
    base_seed: int,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    cell_names = [
        (scenario, max_delay)
        for scenario in SCENARIOS
        for max_delay in MAX_DELAYS
    ]
    errors = np.empty(
        (
            len(POLICIES),
            num_seeds,
            len(cell_names),
            config.checkpoint_count,
        ),
        dtype=float,
    )
    metric_rows: List[Dict[str, object]] = []
    action_rows: List[Dict[str, object]] = []
    proxy_cache = FiniteBudgetProxyCache(config)

    for seed_index in range(num_seeds):
        seed = base_seed + seed_index
        paths = generate_factor_paths(
            seed=seed,
            maximum_delay=max(MAX_DELAYS),
            config=config,
        )
        for cell_index, (scenario, max_delay) in enumerate(cell_names):
            noise_table = build_noise_table(
                scenario=scenario,
                max_delay=max_delay,
                paths=paths,
                config=config,
            )
            for policy_index, policy in enumerate(POLICIES):
                result = simulate_policy(
                    policy=policy,
                    scenario=scenario,
                    max_delay=max_delay,
                    paths=paths,
                    config=config,
                    proxy_cache=proxy_cache,
                    noise_table=noise_table,
                )
                checkpoint_errors = np.asarray(
                    result["checkpoint_errors"], dtype=float
                )
                errors[policy_index, seed_index, cell_index] = (
                    checkpoint_errors
                )
                metric_rows.append(
                    {
                        "seed": seed,
                        "seed_index": seed_index,
                        "scenario": scenario,
                        "max_delay": max_delay,
                        "policy": policy,
                        "final_window_mse": float(
                            np.mean(checkpoint_errors[81:101])
                        ),
                        "budget_auc_mse": float(
                            np.mean(checkpoint_errors[1:])
                        ),
                        "finite": bool(result["finite"]),
                        "within_budget": bool(result["within_budget"]),
                        "budget_used": int(result["budget_used"]),
                        "observed_messages": int(
                            result["observed_messages"]
                        ),
                        "total_updates": int(result["total_updates"]),
                    }
                )
                action_rows.append(
                    {
                        "seed": seed,
                        "seed_index": seed_index,
                        "scenario": scenario,
                        "max_delay": max_delay,
                        "policy": policy,
                        **result["action"],
                    }
                )
        if (seed_index + 1) % max(1, min(8, num_seeds)) == 0:
            print(
                "completed {0}/{1} paired seeds".format(
                    seed_index + 1, num_seeds
                ),
                flush=True,
            )
    return errors, pd.DataFrame(metric_rows), pd.DataFrame(action_rows)


def build_trajectory_summary(
    errors: np.ndarray,
    config: OnlineConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    cells = [
        (scenario, max_delay)
        for scenario in SCENARIOS
        for max_delay in MAX_DELAYS
    ]
    checkpoints = np.linspace(
        0, config.total_budget, config.checkpoint_count
    )
    num_seeds = errors.shape[1]
    for policy_index, policy in enumerate(POLICIES):
        for cell_index, (scenario, max_delay) in enumerate(cells):
            means = np.mean(errors[policy_index, :, cell_index, :], axis=0)
            standard_errors = np.std(
                errors[policy_index, :, cell_index, :], axis=0, ddof=1
            ) / np.sqrt(num_seeds)
            for checkpoint_index, budget in enumerate(checkpoints):
                rows.append(
                    {
                        "policy": policy,
                        "scenario": scenario,
                        "max_delay": max_delay,
                        "budget": float(budget),
                        "mean_mse": float(means[checkpoint_index]),
                        "standard_error": float(
                            standard_errors[checkpoint_index]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    replications: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    count = len(numerator)
    values = np.empty(replications, dtype=float)
    for index in range(replications):
        sample = rng.randint(0, count, size=count)
        values[index] = float(
            np.mean(numerator[sample]) / np.mean(denominator[sample])
        )
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {
        "ratio": float(np.mean(numerator) / np.mean(denominator)),
        "bootstrap_95_lower": float(lower),
        "bootstrap_95_upper": float(upper),
        "bootstrap_replications": int(replications),
    }


def seed_level_cell_average(
    metrics: pd.DataFrame,
    policy: str,
    scenarios: Sequence[str],
) -> np.ndarray:
    subset = metrics[
        (metrics["policy"] == policy)
        & (metrics["scenario"].isin(scenarios))
    ]
    return (
        subset.groupby("seed_index", sort=True)["final_window_mse"]
        .mean()
        .values
    )


def build_comparisons(
    metrics: pd.DataFrame,
    bootstrap_replications: int,
    base_seed: int,
) -> Tuple[pd.DataFrame, str]:
    rows: List[Dict[str, object]] = []
    specifications = [
        (
            "correlated",
            CORRELATED_SCENARIOS,
            "all_agents_adaptive_eta",
        ),
        (
            "all_cells",
            tuple(SCENARIOS.keys()),
            "probe_oracle",
        ),
    ]
    for index, (scope, scenarios, denominator) in enumerate(specifications):
        numerator_values = seed_level_cell_average(
            metrics, "adaptive_probe", scenarios
        )
        denominator_values = seed_level_cell_average(
            metrics, denominator, scenarios
        )
        rows.append(
            {
                "comparison": (
                    "adaptive_vs_all_agents"
                    if denominator == "all_agents_adaptive_eta"
                    else "adaptive_vs_probe_oracle"
                ),
                "scope": scope,
                "numerator": "adaptive_probe",
                "denominator": denominator,
                **bootstrap_ratio(
                    numerator_values,
                    denominator_values,
                    bootstrap_replications,
                    base_seed + 10000 + index,
                ),
            }
        )

    all_scenarios = tuple(SCENARIOS.keys())
    fixed_means = {
        policy: float(
            metrics[metrics["policy"] == policy]["final_window_mse"].mean()
        )
        for policy in FIXED_POLICIES
    }
    best_fixed = min(fixed_means, key=fixed_means.get)
    numerator_values = seed_level_cell_average(
        metrics, "adaptive_probe", all_scenarios
    )
    denominator_values = seed_level_cell_average(
        metrics, best_fixed, all_scenarios
    )
    rows.append(
        {
            "comparison": "adaptive_vs_best_fixed",
            "scope": "all_cells",
            "numerator": "adaptive_probe",
            "denominator": best_fixed,
            **bootstrap_ratio(
                numerator_values,
                denominator_values,
                bootstrap_replications,
                base_seed + 10002,
            ),
        }
    )
    return pd.DataFrame(rows), best_fixed


def evaluate_gates(
    metrics: pd.DataFrame,
    actions: pd.DataFrame,
    comparisons: pd.DataFrame,
    config: OnlineConfig,
) -> Dict[str, object]:
    adaptive = actions[actions["policy"] == "adaptive_probe"]
    action_summary = (
        adaptive.groupby(["scenario", "max_delay"], as_index=False)[
            "selected_num_agents"
        ]
        .median()
        .rename(columns={"selected_num_agents": "median_q"})
    )
    independent = action_summary[
        action_summary["scenario"] == "independent"
    ]
    correlated = action_summary[
        action_summary["scenario"].isin(CORRELATED_SCENARIOS)
    ]
    independent_pass = bool((independent["median_q"] >= 16).all())
    correlated_cell_passes = correlated["median_q"] <= 8

    comparison_lookup = comparisons.set_index("comparison")
    all_agents = comparison_lookup.loc["adaptive_vs_all_agents"]
    oracle = comparison_lookup.loc["adaptive_vs_probe_oracle"]
    fixed = comparison_lookup.loc["adaptive_vs_best_fixed"]

    full_probe_policies = (
        "adaptive_probe",
        "probe_oracle",
        "all_agents_adaptive_eta",
    )
    full_probe_actions = actions[
        actions["policy"].isin(full_probe_policies)
    ]
    accounting_pass = bool(
        metrics["finite"].all()
        and metrics["within_budget"].all()
        and (metrics["budget_used"] <= config.total_budget).all()
        and (
            full_probe_actions["probe_cost"] == config.full_probe_cost
        ).all()
    )
    gates: Dict[str, object] = {
        "independent_participation": {
            "pass": independent_pass,
            "criterion": "median q >= 16 in both independent cells",
            "cells": independent.to_dict(orient="records"),
        },
        "correlated_response": {
            "pass": bool(correlated_cell_passes.sum() >= 5),
            "criterion": "median q <= 8 in at least 5/6 correlated cells",
            "passing_cells": int(correlated_cell_passes.sum()),
            "total_cells": int(len(correlated_cell_passes)),
            "cells": correlated.to_dict(orient="records"),
        },
        "all_agent_resource_gain": {
            "pass": bool(
                all_agents["ratio"] <= 0.80
                and all_agents["bootstrap_95_upper"] < 0.95
            ),
            "criterion": "ratio <= 0.80 and 95% upper < 0.95",
            "ratio": float(all_agents["ratio"]),
            "bootstrap_95_upper": float(
                all_agents["bootstrap_95_upper"]
            ),
        },
        "oracle_proximity": {
            "pass": bool(
                oracle["ratio"] <= 1.25
                and oracle["bootstrap_95_upper"] < 1.40
            ),
            "criterion": "ratio <= 1.25 and 95% upper < 1.40",
            "ratio": float(oracle["ratio"]),
            "bootstrap_95_upper": float(oracle["bootstrap_95_upper"]),
        },
        "fixed_policy_adaptivity_gain": {
            "pass": bool(fixed["ratio"] <= 0.95),
            "criterion": "ratio to best registered fixed-q policy <= 0.95",
            "best_fixed_policy": str(fixed["denominator"]),
            "ratio": float(fixed["ratio"]),
            "bootstrap_95_upper": float(fixed["bootstrap_95_upper"]),
        },
        "accounting_and_numerical_validity": {
            "pass": accounting_pass,
            "criterion": (
                "finite, within budget, and full probe cost exactly 2880"
            ),
            "full_probe_cost": config.full_probe_cost,
        },
    }
    gates["overall"] = {
        "pass": bool(all(value["pass"] for value in gates.values())),
        "criterion": "all six registered gates pass",
    }
    return gates


def plot_outputs(
    trajectory: pd.DataFrame,
    actions: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
    scopes = [
        ("independent", ("independent",)),
        ("correlated average", CORRELATED_SCENARIOS),
    ]
    for axis, (title, scenarios) in zip(axes, scopes):
        subset_scope = trajectory[
            trajectory["scenario"].isin(scenarios)
        ]
        grouped = (
            subset_scope.groupby(["policy", "budget"], as_index=False)[
                "mean_mse"
            ]
            .mean()
        )
        for policy in POLICIES:
            subset = grouped[grouped["policy"] == policy]
            axis.plot(
                subset["budget"],
                subset["mean_mse"],
                label=policy,
            )
        axis.set_yscale("log")
        axis.set_xlabel("Message-equivalent budget")
        axis.set_title(title)
    axes[0].set_ylabel("Mean squared error")
    axes[1].legend(fontsize=7)
    fig.suptitle("Online participation at matched resource checkpoints")
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_budget_mse.png")
    plt.close(fig)

    adaptive = actions[actions["policy"] == "adaptive_probe"]
    summary = (
        adaptive.groupby(["scenario", "max_delay"], as_index=False)[
            "selected_num_agents"
        ]
        .median()
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(SCENARIOS))
    width = 0.35
    for offset, max_delay in zip((-0.5, 0.5), MAX_DELAYS):
        subset = summary[summary["max_delay"] == max_delay]
        ax.bar(
            x + offset * width,
            subset["selected_num_agents"],
            width=width,
            label="max delay={0}".format(max_delay),
        )
    ax.set_yscale("log", base=2)
    ax.set_yticks([1, 2, 4, 8, 16, 32])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels(list(SCENARIOS.keys()))
    ax.set_ylabel("Median selected agents")
    ax.set_title("Participation selected after charged probe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_selected_participation.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = OnlineConfig()
    print(
        "Starting EXP-005B with {0} paired seeds".format(args.num_seeds),
        flush=True,
    )
    errors, metrics, actions = run_experiment(
        config=config,
        num_seeds=args.num_seeds,
        base_seed=args.base_seed,
    )
    trajectory = build_trajectory_summary(errors, config)
    comparisons, best_fixed = build_comparisons(
        metrics=metrics,
        bootstrap_replications=args.bootstrap_replications,
        base_seed=args.base_seed,
    )
    gates = evaluate_gates(metrics, actions, comparisons, config)

    metrics.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    actions.to_csv(args.output_dir / "actions.csv", index=False)
    trajectory.to_csv(
        args.output_dir / "budget_trajectories.csv", index=False
    )
    comparisons.to_csv(
        args.output_dir / "paired_bootstrap_ratios.csv", index=False
    )
    plot_outputs(trajectory, actions, args.output_dir)
    summary = {
        "experiment_id": "EXP-005B-online-probe-controller",
        "status": "COMPLETED_PENDING_REPRODUCTION",
        "num_paired_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        "bootstrap_replications": args.bootstrap_replications,
        "best_registered_fixed_policy": best_fixed,
        "config": {
            "num_agents": config.num_agents,
            "agent_counts": [1, 2, 4, 8, 16, 32],
            "total_budget": config.total_budget,
            "update_overhead": config.update_overhead,
            "probe_updates": config.probe_updates,
            "full_probe_cost": config.full_probe_cost,
            "batch_size": config.batch_size,
            "eta_grid": config.eta_grid.tolist(),
            "scenarios": SCENARIOS,
            "max_delays": list(MAX_DELAYS),
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

"""Run EXP-005C, sparse participation control under regime shifts."""

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

from online_participation import (
    FiniteBudgetProxyCache,
    build_noise_table,
    generate_factor_paths,
)
from sparse_dynamic import (
    POLICIES,
    REGIME_SEQUENCE,
    DynamicConfig,
    simulate_dynamic_policy,
)


MAX_DELAYS: Tuple[int, ...] = (4, 16)
FIXED_POLICIES: Tuple[str, ...] = (
    "fixed_q1_oracle_eta",
    "fixed_q4_oracle_eta",
    "fixed_q8_oracle_eta",
    "fixed_q32_oracle_eta",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "sparse_dynamic",
    )
    parser.add_argument("--num-seeds", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=20260729)
    parser.add_argument("--bootstrap-replications", type=int, default=2000)
    return parser.parse_args()


def run_experiment(
    config: DynamicConfig,
    num_seeds: int,
    base_seed: int,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    errors = np.empty(
        (
            len(POLICIES),
            num_seeds,
            len(MAX_DELAYS),
            config.checkpoint_count,
        ),
        dtype=float,
    )
    action_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []
    cache = FiniteBudgetProxyCache(config)
    for seed_index in range(num_seeds):
        seed = base_seed + seed_index
        paths = generate_factor_paths(
            seed=seed,
            maximum_delay=max(MAX_DELAYS),
            config=config,
        )
        for delay_index, max_delay in enumerate(MAX_DELAYS):
            noise_tables = {
                scenario: build_noise_table(
                    scenario=scenario,
                    max_delay=max_delay,
                    paths=paths,
                    config=config,
                )
                for scenario in REGIME_SEQUENCE
            }
            for policy_index, policy in enumerate(POLICIES):
                result = simulate_dynamic_policy(
                    policy=policy,
                    max_delay=max_delay,
                    noise_tables=noise_tables,
                    config=config,
                    proxy_cache=cache,
                )
                errors[policy_index, seed_index, delay_index] = np.asarray(
                    result["checkpoint_errors"], dtype=float
                )
                run_rows.append(
                    {
                        "seed": seed,
                        "seed_index": seed_index,
                        "max_delay": max_delay,
                        "policy": policy,
                        "charged_budget": int(result["charged_budget"]),
                        "observed_messages": int(
                            result["observed_messages"]
                        ),
                        "total_probe_cost": int(
                            result["total_probe_cost"]
                        ),
                        "total_updates": int(result["total_updates"]),
                        "finite": bool(result["finite"]),
                        "within_budget": bool(result["within_budget"]),
                    }
                )
                for action in result["actions"]:
                    action_rows.append(
                        {
                            "seed": seed,
                            "seed_index": seed_index,
                            "max_delay": max_delay,
                            "policy": policy,
                            **action,
                        }
                    )
        if (seed_index + 1) % max(1, min(8, num_seeds)) == 0:
            print(
                "completed {0}/{1} paired seeds".format(
                    seed_index + 1, num_seeds
                ),
                flush=True,
            )
    return errors, pd.DataFrame(action_rows), pd.DataFrame(run_rows)


def build_regime_metrics(
    errors: np.ndarray,
    config: DynamicConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    checkpoints_per_regime = (
        config.checkpoint_count - 1
    ) // len(REGIME_SEQUENCE)
    for policy_index, policy in enumerate(POLICIES):
        for seed_index in range(errors.shape[1]):
            for delay_index, max_delay in enumerate(MAX_DELAYS):
                trajectory = errors[policy_index, seed_index, delay_index]
                for regime_index, regime in enumerate(REGIME_SEQUENCE):
                    start = regime_index * checkpoints_per_regime
                    end = (regime_index + 1) * checkpoints_per_regime
                    latter_half = trajectory[
                        start + checkpoints_per_regime // 2 + 1 : end + 1
                    ]
                    rows.append(
                        {
                            "seed_index": seed_index,
                            "max_delay": max_delay,
                            "regime": regime,
                            "policy": policy,
                            "regime_mse": float(np.mean(latter_half)),
                        }
                    )
    frame = pd.DataFrame(rows)
    oracle = frame[frame["policy"] == "piecewise_oracle"][
        ["seed_index", "max_delay", "regime", "regime_mse"]
    ].rename(columns={"regime_mse": "oracle_regime_mse"})
    frame = frame.merge(
        oracle,
        on=["seed_index", "max_delay", "regime"],
        how="left",
        validate="many_to_one",
    )
    frame["normalized_regime_score"] = (
        frame["regime_mse"] / frame["oracle_regime_mse"]
    )
    return frame


def build_trajectory_summary(
    errors: np.ndarray,
    config: DynamicConfig,
) -> pd.DataFrame:
    checkpoints = np.linspace(
        0, config.total_budget, config.checkpoint_count
    )
    rows: List[Dict[str, object]] = []
    count = errors.shape[1] * errors.shape[2]
    for policy_index, policy in enumerate(POLICIES):
        flattened = errors[policy_index].reshape(
            count, config.checkpoint_count
        )
        mean = np.mean(flattened, axis=0)
        standard_error = np.std(flattened, axis=0, ddof=1) / np.sqrt(
            count
        )
        for index, budget in enumerate(checkpoints):
            rows.append(
                {
                    "policy": policy,
                    "budget": float(budget),
                    "mean_mse": float(mean[index]),
                    "standard_error": float(standard_error[index]),
                }
            )
    return pd.DataFrame(rows)


def seed_dynamic_scores(metrics: pd.DataFrame, policy: str) -> np.ndarray:
    return (
        metrics[metrics["policy"] == policy]
        .groupby("seed_index", sort=True)["normalized_regime_score"]
        .mean()
        .values
    )


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


def build_comparisons(
    metrics: pd.DataFrame,
    replications: int,
    base_seed: int,
) -> Tuple[pd.DataFrame, str]:
    fixed_scores = {
        policy: float(np.mean(seed_dynamic_scores(metrics, policy)))
        for policy in FIXED_POLICIES
    }
    best_fixed = min(fixed_scores, key=fixed_scores.get)
    adaptive = seed_dynamic_scores(metrics, "sparse_adaptive")
    best_fixed_values = seed_dynamic_scores(metrics, best_fixed)
    oracle = seed_dynamic_scores(metrics, "piecewise_oracle")
    rows = [
        {
            "comparison": "adaptive_vs_best_fixed",
            "numerator": "sparse_adaptive",
            "denominator": best_fixed,
            **bootstrap_ratio(
                adaptive,
                best_fixed_values,
                replications,
                base_seed + 10000,
            ),
        },
        {
            "comparison": "adaptive_vs_piecewise_oracle",
            "numerator": "sparse_adaptive",
            "denominator": "piecewise_oracle",
            **bootstrap_ratio(
                adaptive,
                oracle,
                replications,
                base_seed + 10001,
            ),
        },
    ]
    return pd.DataFrame(rows), best_fixed


def evaluate_gates(
    comparisons: pd.DataFrame,
    actions: pd.DataFrame,
    runs: pd.DataFrame,
    config: DynamicConfig,
) -> Dict[str, object]:
    lookup = comparisons.set_index("comparison")
    fixed = lookup.loc["adaptive_vs_best_fixed"]
    oracle = lookup.loc["adaptive_vs_piecewise_oracle"]
    target_blocks = {
        "independent": 1,
        "clustered": 5,
        "global": 9,
        "mixed": 13,
    }
    adaptive_actions = actions[actions["policy"] == "sparse_adaptive"]
    switch_cells: List[Dict[str, object]] = []
    passing = 0
    for regime, block in target_blocks.items():
        values = adaptive_actions[adaptive_actions["block"] == float(block)][
            "selected_num_agents"
        ]
        median_q = float(values.median())
        if regime == "independent":
            passed = median_q >= 16
            criterion = "q >= 16"
        elif regime == "global":
            passed = median_q <= 4
            criterion = "q <= 4"
        else:
            passed = median_q <= 8
            criterion = "q <= 8"
        passing += int(passed)
        switch_cells.append(
            {
                "regime": regime,
                "block": block,
                "median_q": median_q,
                "criterion": criterion,
                "pass": bool(passed),
            }
        )
    adaptive_runs = runs[runs["policy"] == "sparse_adaptive"]
    exploration_pass = bool(
        (adaptive_runs["total_probe_cost"] == config.sparse_probe_cost).all()
        and config.sparse_probe_cost / config.total_budget <= 0.05
    )
    validity_pass = bool(
        runs["finite"].all()
        and runs["within_budget"].all()
        and (runs["charged_budget"] <= config.total_budget).all()
    )
    gates: Dict[str, object] = {
        "best_fixed_improvement": {
            "pass": bool(
                fixed["ratio"] <= 0.90
                and fixed["bootstrap_95_upper"] < 1.0
            ),
            "criterion": "ratio <= 0.90 and 95% upper < 1.0",
            "best_fixed_policy": str(fixed["denominator"]),
            "ratio": float(fixed["ratio"]),
            "bootstrap_95_upper": float(fixed["bootstrap_95_upper"]),
        },
        "oracle_proximity": {
            "pass": bool(
                oracle["ratio"] <= 1.25
                and oracle["bootstrap_95_upper"] < 1.40
            ),
            "criterion": "score <= 1.25 and 95% upper < 1.40",
            "normalized_score": float(oracle["ratio"]),
            "bootstrap_95_upper": float(oracle["bootstrap_95_upper"]),
        },
        "switch_response": {
            "pass": bool(passing >= 3),
            "criterion": "correct direction in at least 3/4 regimes",
            "passing_regimes": passing,
            "cells": switch_cells,
        },
        "exploration_budget": {
            "pass": exploration_pass,
            "criterion": "exactly 768 units and <= 5% total budget",
            "probe_cost": config.sparse_probe_cost,
            "probe_fraction": (
                config.sparse_probe_cost / config.total_budget
            ),
        },
        "accounting_and_numerical_validity": {
            "pass": validity_pass,
            "criterion": "finite, charged, within block and total budgets",
        },
    }
    gates["overall"] = {
        "pass": bool(all(value["pass"] for value in gates.values())),
        "criterion": "all five final go/no-go gates pass",
    }
    return gates


def plot_outputs(
    trajectory: pd.DataFrame,
    actions: pd.DataFrame,
    output_dir: Path,
    config: DynamicConfig,
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
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for policy in POLICIES:
        subset = trajectory[trajectory["policy"] == policy]
        ax.plot(subset["budget"], subset["mean_mse"], label=policy)
    for boundary in [8000, 16000, 24000]:
        ax.axvline(boundary, color="black", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Message-equivalent budget")
    ax.set_ylabel("Mean squared error")
    ax.set_title("Sparse participation control under regime shifts")
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_dynamic_mse.png")
    plt.close(fig)

    adaptive = actions[actions["policy"] == "sparse_adaptive"]
    summary = adaptive.groupby("block", as_index=False).agg(
        median_q=("selected_num_agents", "median"),
        q25=("selected_num_agents", lambda x: np.percentile(x, 25)),
        q75=("selected_num_agents", lambda x: np.percentile(x, 75)),
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.step(
        summary["block"],
        summary["median_q"],
        where="mid",
        marker="o",
    )
    ax.fill_between(
        summary["block"],
        summary["q25"],
        summary["q75"],
        step="mid",
        alpha=0.2,
    )
    for boundary in [4, 8, 12]:
        ax.axvline(boundary - 0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_yscale("log", base=2)
    ax.set_yticks([1, 2, 4, 8, 16, 32])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Block")
    ax.set_ylabel("Selected agents")
    ax.set_title("Sparse controller participation response")
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_block_participation.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = DynamicConfig()
    print(
        "Starting EXP-005C with {0} paired seeds".format(args.num_seeds),
        flush=True,
    )
    errors, actions, runs = run_experiment(
        config=config,
        num_seeds=args.num_seeds,
        base_seed=args.base_seed,
    )
    metrics = build_regime_metrics(errors, config)
    trajectory = build_trajectory_summary(errors, config)
    comparisons, best_fixed = build_comparisons(
        metrics,
        args.bootstrap_replications,
        args.base_seed,
    )
    gates = evaluate_gates(comparisons, actions, runs, config)

    metrics.to_csv(
        args.output_dir / "per_seed_regime_metrics.csv", index=False
    )
    actions.to_csv(args.output_dir / "block_actions.csv", index=False)
    runs.to_csv(args.output_dir / "run_accounting.csv", index=False)
    trajectory.to_csv(
        args.output_dir / "budget_trajectories.csv", index=False
    )
    comparisons.to_csv(
        args.output_dir / "paired_bootstrap_ratios.csv", index=False
    )
    plot_outputs(trajectory, actions, args.output_dir, config)
    summary = {
        "experiment_id": "EXP-005C-sparse-dynamic-controller",
        "status": "COMPLETED_PENDING_REPRODUCTION",
        "num_paired_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        "bootstrap_replications": args.bootstrap_replications,
        "best_registered_fixed_policy": best_fixed,
        "config": {
            "total_budget": config.total_budget,
            "block_budget": config.block_budget,
            "num_blocks": config.num_blocks,
            "regime_sequence": list(REGIME_SEQUENCE),
            "sparse_probe_q": config.sparse_probe_q,
            "sparse_probe_updates": config.sparse_probe_updates,
            "sparse_probe_cost": config.sparse_probe_cost,
            "rolling_probe_snapshots": config.rolling_probe_snapshots,
            "max_delays": list(MAX_DELAYS),
            "eta_grid": config.eta_grid.tolist(),
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

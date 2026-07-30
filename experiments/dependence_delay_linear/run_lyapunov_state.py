"""Run EXP-006C scalar Lyapunov-surrogate participation control."""

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

from lyapunov_state import (
    ADAPTIVE_POLICIES,
    POLICIES,
    LyapunovStateConfig,
    simulate_lyapunov_state_policy,
)
from online_participation import FiniteBudgetProxyCache, generate_factor_paths
from run_state_correlation import bootstrap_ratio
from state_correlation import SCENARIOS, build_noise_table_components


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
        / "lyapunov_state",
    )
    parser.add_argument("--num-seeds", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=20260830)
    parser.add_argument("--bootstrap-replications", type=int, default=2000)
    return parser.parse_args()


def run_experiment(
    config: LyapunovStateConfig,
    num_seeds: int,
    base_seed: int,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    scenario_names = tuple(SCENARIOS)
    errors = np.empty(
        (
            len(POLICIES),
            num_seeds,
            len(scenario_names),
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
                scenario: build_noise_table_components(
                    rho_global=SCENARIOS[scenario][0],
                    rho_cluster=SCENARIOS[scenario][1],
                    max_delay=max_delay,
                    paths=paths,
                    config=config,
                )
                for scenario in scenario_names
            }
            for scenario_index, scenario in enumerate(scenario_names):
                for policy_index, policy in enumerate(POLICIES):
                    result = simulate_lyapunov_state_policy(
                        policy=policy,
                        scenario=scenario,
                        max_delay=max_delay,
                        noise_table=noise_tables[scenario],
                        config=config,
                        proxy_cache=cache,
                    )
                    errors[
                        policy_index,
                        seed_index,
                        scenario_index,
                        delay_index,
                    ] = np.asarray(
                        result["checkpoint_errors"], dtype=float
                    )
                    run_rows.append(
                        {
                            "seed": seed,
                            "seed_index": seed_index,
                            "scenario": scenario,
                            "max_delay": max_delay,
                            "policy": policy,
                            "charged_budget": int(
                                result["charged_budget"]
                            ),
                            "observed_messages": int(
                                result["observed_messages"]
                            ),
                            "total_probe_cost": int(
                                result["total_probe_cost"]
                            ),
                            "total_updates": int(result["total_updates"]),
                            "finite": bool(result["finite"]),
                            "within_budget": bool(
                                result["within_budget"]
                            ),
                        }
                    )
                    for action in result["actions"]:
                        action_rows.append(
                            {
                                "seed": seed,
                                "seed_index": seed_index,
                                "scenario": scenario,
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


def build_cell_metrics(errors: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    scenario_names = tuple(SCENARIOS)
    for policy_index, policy in enumerate(POLICIES):
        for seed_index in range(errors.shape[1]):
            for scenario_index, scenario in enumerate(scenario_names):
                for delay_index, max_delay in enumerate(MAX_DELAYS):
                    mse = float(
                        np.mean(
                            errors[
                                policy_index,
                                seed_index,
                                scenario_index,
                                delay_index,
                                1:,
                            ]
                        )
                    )
                    rows.append(
                        {
                            "seed_index": seed_index,
                            "scenario": scenario,
                            "max_delay": max_delay,
                            "policy": policy,
                            "cell_mse": mse,
                        }
                    )
    frame = pd.DataFrame(rows)
    oracle = frame[frame["policy"] == "state_oracle"][
        ["seed_index", "scenario", "max_delay", "cell_mse"]
    ].rename(columns={"cell_mse": "oracle_cell_mse"})
    frame = frame.merge(
        oracle,
        on=["seed_index", "scenario", "max_delay"],
        how="left",
        validate="many_to_one",
    )
    frame["normalized_cell_score"] = (
        frame["cell_mse"] / frame["oracle_cell_mse"]
    )
    return frame


def seed_scores(metrics: pd.DataFrame, policy: str) -> np.ndarray:
    return (
        metrics[metrics["policy"] == policy]
        .groupby("seed_index", sort=True)["normalized_cell_score"]
        .mean()
        .to_numpy(dtype=float)
    )


def build_comparisons(
    metrics: pd.DataFrame,
    replications: int,
    base_seed: int,
) -> Tuple[pd.DataFrame, str]:
    fixed_scores = {
        policy: float(np.mean(seed_scores(metrics, policy)))
        for policy in FIXED_POLICIES
    }
    best_fixed = min(fixed_scores, key=fixed_scores.get)
    adaptive = seed_scores(metrics, "lyapunov_state_adaptive")
    comparisons = (
        ("lyapunov_vs_raw", "raw_state_adaptive", base_seed + 10000),
        (
            "lyapunov_vs_correlation_only",
            "correlation_only_adaptive",
            base_seed + 10001,
        ),
        ("lyapunov_vs_best_fixed", best_fixed, base_seed + 10002),
        ("lyapunov_vs_oracle", "state_oracle", base_seed + 10003),
    )
    rows = []
    for name, denominator, seed in comparisons:
        rows.append(
            {
                "comparison": name,
                "numerator": "lyapunov_state_adaptive",
                "denominator": denominator,
                **bootstrap_ratio(
                    adaptive,
                    seed_scores(metrics, denominator),
                    replications,
                    seed,
                ),
            }
        )
    return pd.DataFrame(rows), best_fixed


def action_agreement(actions: pd.DataFrame) -> Dict[str, float]:
    keys = ["seed_index", "scenario", "max_delay", "block"]
    adaptive = actions[
        actions["policy"] == "lyapunov_state_adaptive"
    ][keys + ["selected_num_agents"]].rename(
        columns={"selected_num_agents": "adaptive_q"}
    )
    oracle = actions[actions["policy"] == "state_oracle"][
        keys + ["selected_num_agents"]
    ].rename(columns={"selected_num_agents": "oracle_q"})
    paired = adaptive.merge(oracle, on=keys, validate="one_to_one")
    paired = paired[paired["block"] >= 1]
    ratio = np.maximum(paired["adaptive_q"], paired["oracle_q"]) / np.minimum(
        paired["adaptive_q"], paired["oracle_q"]
    )
    return {
        "agreement_fraction": float((ratio <= 2.0).mean()),
        "comparison_count": int(len(paired)),
    }


def surrogate_calibration(actions: pd.DataFrame) -> Dict[str, float]:
    selected = actions[
        actions["policy"] == "lyapunov_state_adaptive"
    ].copy()
    predicted = np.sqrt(
        selected["lyapunov_surrogate_after_probe"].to_numpy(dtype=float)
    )
    truth = selected["true_error_magnitude"].to_numpy(dtype=float)
    floor = 1e-12
    correlation = float(
        np.corrcoef(
            np.log(np.maximum(predicted, floor)),
            np.log(np.maximum(truth, floor)),
        )[0, 1]
    )
    return {
        "log_correlation": correlation,
        "median_predicted_to_true_ratio": float(
            np.median(predicted / np.maximum(truth, floor))
        ),
        "mean_predicted_state": float(np.mean(predicted)),
        "mean_true_error": float(np.mean(truth)),
    }


def evaluate_gates(
    comparisons: pd.DataFrame,
    actions: pd.DataFrame,
    runs: pd.DataFrame,
    metrics: pd.DataFrame,
    config: LyapunovStateConfig,
    expected_num_seeds: int,
) -> Dict[str, object]:
    lookup = comparisons.set_index("comparison")
    replacement = lookup.loc["lyapunov_vs_raw"]
    state_value = lookup.loc["lyapunov_vs_correlation_only"]
    fixed = lookup.loc["lyapunov_vs_best_fixed"]
    oracle = lookup.loc["lyapunov_vs_oracle"]
    agreement = action_agreement(actions)
    adaptive_runs = runs[runs["policy"].isin(ADAPTIVE_POLICIES)]
    probe_valid = bool(
        (adaptive_runs["total_probe_cost"] == config.probe_cost).all()
        and config.probe_cost / config.total_budget <= 0.05
    )
    expected_runs = (
        expected_num_seeds
        * len(SCENARIOS)
        * len(MAX_DELAYS)
        * len(POLICIES)
    )
    validity = bool(
        len(runs) == expected_runs
        and runs["finite"].all()
        and runs["within_budget"].all()
        and (runs["charged_budget"] <= config.total_budget).all()
        and np.isfinite(metrics["normalized_cell_score"]).all()
    )
    gates: Dict[str, object] = {
        "replacement_value": {
            "pass": bool(
                replacement["ratio"] <= 0.90
                and replacement["bootstrap_95_upper"] < 1.0
            ),
            "ratio": float(replacement["ratio"]),
            "bootstrap_95_upper": float(
                replacement["bootstrap_95_upper"]
            ),
        },
        "state_value": {
            "pass": bool(
                state_value["ratio"] <= 0.90
                and state_value["bootstrap_95_upper"] < 1.0
            ),
            "ratio": float(state_value["ratio"]),
            "bootstrap_95_upper": float(
                state_value["bootstrap_95_upper"]
            ),
        },
        "best_fixed_improvement": {
            "pass": bool(
                fixed["ratio"] <= 0.90
                and fixed["bootstrap_95_upper"] < 1.0
            ),
            "best_fixed_policy": str(fixed["denominator"]),
            "ratio": float(fixed["ratio"]),
            "bootstrap_95_upper": float(fixed["bootstrap_95_upper"]),
        },
        "oracle_proximity": {
            "pass": bool(
                oracle["ratio"] <= 1.50
                and oracle["bootstrap_95_upper"] < 1.75
            ),
            "normalized_score": float(oracle["ratio"]),
            "bootstrap_95_upper": float(oracle["bootstrap_95_upper"]),
        },
        "action_agreement": {
            "pass": bool(agreement["agreement_fraction"] >= 0.50),
            **agreement,
        },
        "probe_budget": {
            "pass": probe_valid,
            "probe_cost": config.probe_cost,
            "probe_fraction": config.probe_cost / config.total_budget,
        },
        "accounting_and_numerical_validity": {
            "pass": validity,
            "observed_runs": int(len(runs)),
            "expected_runs": int(expected_runs),
        },
    }
    gates["overall"] = {
        "pass": bool(all(value["pass"] for value in gates.values())),
        "criterion": "all seven Lyapunov-controller gates pass",
    }
    return gates


def build_trajectory_summary(
    errors: np.ndarray,
    config: LyapunovStateConfig,
) -> pd.DataFrame:
    checkpoints = np.linspace(
        0, config.total_budget, config.checkpoint_count
    )
    rows: List[Dict[str, object]] = []
    scenario_names = tuple(SCENARIOS)
    for policy_index, policy in enumerate(POLICIES):
        for scenario_index, scenario in enumerate(scenario_names):
            flattened = errors[
                policy_index, :, scenario_index
            ].reshape(-1, config.checkpoint_count)
            mean = np.mean(flattened, axis=0)
            standard_error = np.std(
                flattened, axis=0, ddof=1
            ) / np.sqrt(len(flattened))
            for checkpoint, budget in enumerate(checkpoints):
                rows.append(
                    {
                        "policy": policy,
                        "scenario": scenario,
                        "budget": float(budget),
                        "mean_mse": float(mean[checkpoint]),
                        "standard_error": float(
                            standard_error[checkpoint]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot_outputs(
    trajectory: pd.DataFrame,
    actions: pd.DataFrame,
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
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True)
    for ax, scenario in zip(axes.ravel(), SCENARIOS):
        for policy in POLICIES:
            subset = trajectory[
                (trajectory["scenario"] == scenario)
                & (trajectory["policy"] == policy)
            ]
            ax.plot(
                subset["budget"],
                subset["mean_mse"],
                label=policy,
            )
        ax.set_yscale("log")
        ax.set_title(scenario)
        ax.set_xlabel("Message-equivalent budget")
        ax.set_ylabel("Mean squared error")
    axes[0, 0].legend(fontsize=5.5)
    fig.suptitle("Lyapunov-surrogate participation controller")
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_mse_by_scenario.png")
    plt.close(fig)

    selected = actions[
        actions["policy"].isin(
            [
                "lyapunov_state_adaptive",
                "raw_state_adaptive",
                "state_oracle",
            ]
        )
    ]
    summary = (
        selected.groupby(["policy", "scenario", "block"], as_index=False)[
            "selected_num_agents"
        ]
        .median()
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), sharex=True)
    plotted = (
        "lyapunov_state_adaptive",
        "raw_state_adaptive",
        "state_oracle",
    )
    for ax, scenario in zip(axes.ravel(), SCENARIOS):
        for policy in plotted:
            subset = summary[
                (summary["scenario"] == scenario)
                & (summary["policy"] == policy)
            ]
            ax.step(
                subset["block"],
                subset["selected_num_agents"],
                where="mid",
                marker="o",
                label=policy,
            )
        ax.set_yscale("log", base=2)
        ax.set_yticks([1, 2, 4, 8, 16, 32])
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter()
        )
        ax.set_title(scenario)
        ax.set_xlabel("Block")
        ax.set_ylabel("Median selected agents")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_participation_by_block.png")
    plt.close(fig)

    adaptive = actions[
        actions["policy"] == "lyapunov_state_adaptive"
    ]
    predicted = np.sqrt(adaptive["lyapunov_surrogate_after_probe"])
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(
        adaptive["true_error_magnitude"],
        predicted,
        s=8,
        alpha=0.25,
    )
    limits = [1e-3, 1.0]
    ax.plot(limits, limits, color="black", linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("True error magnitude (audit only)")
    ax.set_ylabel("Predicted Lyapunov state")
    ax.set_title("Lyapunov-surrogate calibration")
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_surrogate_calibration.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = LyapunovStateConfig()
    print(
        "Starting EXP-006C with {0} paired seeds".format(args.num_seeds),
        flush=True,
    )
    errors, actions, runs = run_experiment(
        config, args.num_seeds, args.base_seed
    )
    metrics = build_cell_metrics(errors)
    trajectory = build_trajectory_summary(errors, config)
    comparisons, best_fixed = build_comparisons(
        metrics, args.bootstrap_replications, args.base_seed
    )
    gates = evaluate_gates(
        comparisons,
        actions,
        runs,
        metrics,
        config,
        args.num_seeds,
    )
    calibration = surrogate_calibration(actions)
    metrics.to_csv(
        args.output_dir / "per_seed_cell_metrics.csv", index=False
    )
    actions.to_csv(args.output_dir / "block_actions.csv", index=False)
    runs.to_csv(args.output_dir / "run_accounting.csv", index=False)
    trajectory.to_csv(
        args.output_dir / "budget_trajectories.csv", index=False
    )
    comparisons.to_csv(
        args.output_dir / "paired_bootstrap_ratios.csv", index=False
    )
    plot_outputs(trajectory, actions, args.output_dir)
    summary = {
        "experiment_id": "EXP-006C-lyapunov-state-controller",
        "status": "COMPLETED_PENDING_REPRODUCTION",
        "num_paired_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        "bootstrap_replications": args.bootstrap_replications,
        "best_registered_fixed_policy": best_fixed,
        "surrogate_calibration": calibration,
        "config": {
            "scenarios": SCENARIOS,
            "max_delays": list(MAX_DELAYS),
            "total_budget": config.total_budget,
            "block_budget": config.block_budget,
            "num_blocks": config.num_blocks,
            "initial_error": config.initial_error,
            "probe_q": config.probe_q,
            "probe_updates_per_block": config.probe_updates_per_block,
            "probe_cost": config.probe_cost,
            "rolling_probe_vectors": config.rolling_probe_vectors,
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
    print(json.dumps(calibration, indent=2), flush=True)
    print("Outputs written to {0}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()

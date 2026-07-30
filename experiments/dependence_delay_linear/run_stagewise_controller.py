"""Run EXP-004, the predictable stagewise step-participation experiment."""

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

from stagewise_controller import (
    POLICIES,
    StagewiseConfig,
    generate_markov_paths,
    simulate_policy,
)


ALIGNMENTS = ("server_time", "sample_time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "stagewise",
    )
    parser.add_argument("--num-seeds", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=20260729)
    parser.add_argument("--bootstrap-replications", type=int, default=2000)
    return parser.parse_args()


def phase_slices(config: StagewiseConfig) -> Dict[str, slice]:
    return {
        "independent": slice(1, 4 * config.stage_length + 1),
        "high_all": slice(
            4 * config.stage_length + 1, 8 * config.stage_length + 1
        ),
        "high_adapted": slice(
            5 * config.stage_length + 1, 8 * config.stage_length + 1
        ),
        "partial": slice(
            8 * config.stage_length + 1, config.total_steps + 1
        ),
        "final_window": slice(
            config.total_steps - config.stage_length + 1,
            config.total_steps + 1,
        ),
        "full": slice(1, config.total_steps + 1),
    }


def bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    replications: int,
    seed: int,
) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    num_seeds = len(numerator)
    ratios = np.empty(replications, dtype=float)
    for index in range(replications):
        sample = rng.randint(0, num_seeds, size=num_seeds)
        ratios[index] = float(
            np.mean(numerator[sample]) / np.mean(denominator[sample])
        )
    point = float(np.mean(numerator) / np.mean(denominator))
    lower, upper = np.percentile(ratios, [2.5, 97.5])
    return point, float(lower), float(upper)


def run_experiment(
    config: StagewiseConfig,
    num_seeds: int,
    base_seed: int,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    errors = {
        alignment: np.empty(
            (len(POLICIES), num_seeds, config.total_steps + 1), dtype=float
        )
        for alignment in ALIGNMENTS
    }
    action_rows: List[Dict[str, object]] = []

    for alignment in ALIGNMENTS:
        for seed_index in range(num_seeds):
            seed = base_seed + seed_index
            common, idiosyncratic, maximum_delay = generate_markov_paths(
                seed=seed,
                config=config,
            )
            for policy_index, policy in enumerate(POLICIES):
                result = simulate_policy(
                    policy=policy,
                    common=common,
                    idiosyncratic=idiosyncratic,
                    maximum_delay=maximum_delay,
                    alignment=alignment,
                    config=config,
                )
                errors[alignment][policy_index, seed_index, :] = result[
                    "squared_errors"
                ]
                for action in result["actions"]:
                    action_rows.append(
                        {
                            "alignment": alignment,
                            "seed": seed,
                            "policy": policy,
                            **action,
                        }
                    )
            if (seed_index + 1) % max(1, min(8, num_seeds)) == 0:
                print(
                    "alignment={0}: completed {1}/{2} seeds".format(
                        alignment, seed_index + 1, num_seeds
                    ),
                    flush=True,
                )
    return errors, pd.DataFrame(action_rows)


def build_metric_tables(
    errors: Dict[str, np.ndarray],
    actions: pd.DataFrame,
    config: StagewiseConfig,
    base_seed: int,
    bootstrap_replications: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slices = phase_slices(config)
    per_seed_rows: List[Dict[str, object]] = []
    trajectory_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []

    for alignment_index, alignment in enumerate(ALIGNMENTS):
        values = errors[alignment]
        num_seeds = values.shape[1]
        for policy_index, policy in enumerate(POLICIES):
            for seed_index in range(num_seeds):
                row: Dict[str, object] = {
                    "alignment": alignment,
                    "seed_index": seed_index,
                    "policy": policy,
                    "finite": bool(
                        np.all(np.isfinite(values[policy_index, seed_index]))
                    ),
                }
                for phase, phase_slice in slices.items():
                    row[phase + "_mse"] = float(
                        np.mean(values[policy_index, seed_index, phase_slice])
                    )
                per_seed_rows.append(row)

            mean = np.mean(values[policy_index], axis=0)
            standard_error = np.std(
                values[policy_index], axis=0, ddof=1
            ) / np.sqrt(num_seeds)
            recorded_steps = sorted(
                set(range(0, config.total_steps + 1, 5))
                | {
                    stage * config.stage_length
                    for stage in range(config.num_stages + 1)
                }
            )
            for step in recorded_steps:
                trajectory_rows.append(
                    {
                        "alignment": alignment,
                        "policy": policy,
                        "step": step,
                        "mean_mse": float(mean[step]),
                        "standard_error": float(standard_error[step]),
                    }
                )

        adaptive = values[POLICIES.index("adaptive_joint"), :, slices["high_adapted"]].mean(axis=1)
        for comparison_index, baseline in enumerate(
            ["delay_only", "all_agents_adaptive_eta", "proxy_oracle"]
        ):
            baseline_values = values[
                POLICIES.index(baseline), :, slices["high_adapted"]
            ].mean(axis=1)
            point, lower, upper = bootstrap_ratio(
                numerator=adaptive,
                denominator=baseline_values,
                replications=bootstrap_replications,
                seed=base_seed
                + 10000
                + 100 * alignment_index
                + comparison_index,
            )
            comparison_rows.append(
                {
                    "alignment": alignment,
                    "numerator": "adaptive_joint",
                    "denominator": baseline,
                    "phase": "high_adapted",
                    "mse_ratio": point,
                    "bootstrap_95_lower": lower,
                    "bootstrap_95_upper": upper,
                    "bootstrap_replications": bootstrap_replications,
                }
            )

    action_summary = (
        actions.groupby(["alignment", "policy", "stage"], as_index=False)
        .agg(
            median_num_agents=("num_agents", "median"),
            q25_num_agents=("num_agents", lambda values: np.percentile(values, 25)),
            q75_num_agents=("num_agents", lambda values: np.percentile(values, 75)),
            mean_eta=("eta", "mean"),
            median_eta=("eta", "median"),
            mean_lrv_used=("lrv_used", "mean"),
        )
        .sort_values(["alignment", "policy", "stage"])
    )
    return (
        pd.DataFrame(per_seed_rows),
        pd.DataFrame(trajectory_rows),
        action_summary,
        pd.DataFrame(comparison_rows),
    )


def evaluate_criteria(
    per_seed: pd.DataFrame,
    actions: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> Dict[str, object]:
    criteria: Dict[str, object] = {}
    for alignment in ALIGNMENTS:
        comparison = comparisons[comparisons["alignment"] == alignment].set_index(
            "denominator"
        )
        adaptive_actions = actions[
            (actions["alignment"] == alignment)
            & (actions["policy"] == "adaptive_joint")
        ]
        independent_q = float(
            adaptive_actions[adaptive_actions["stage"].isin([1.0, 2.0, 3.0])][
                "num_agents"
            ].median()
        )
        high_q = float(
            adaptive_actions[adaptive_actions["stage"].isin([5.0, 6.0, 7.0])][
                "num_agents"
            ].median()
        )
        all_finite = bool(
            per_seed[per_seed["alignment"] == alignment]["finite"].all()
        )
        delay_ratio = float(comparison.loc["delay_only", "mse_ratio"])
        all_agents_ratio = float(
            comparison.loc["all_agents_adaptive_eta", "mse_ratio"]
        )
        oracle_ratio = float(comparison.loc["proxy_oracle", "mse_ratio"])
        criteria[alignment] = {
            "adaptive_vs_delay_only": {
                "pass": bool(delay_ratio <= 0.90),
                "mse_ratio": delay_ratio,
                "criterion": "ratio <= 0.90",
            },
            "adaptive_vs_all_agents_adaptive_eta": {
                "pass": bool(all_agents_ratio <= 0.95),
                "mse_ratio": all_agents_ratio,
                "criterion": "ratio <= 0.95",
            },
            "adaptive_vs_proxy_oracle": {
                "pass": bool(oracle_ratio <= 1.25),
                "mse_ratio": oracle_ratio,
                "criterion": "ratio <= 1.25",
            },
            "participation_response": {
                "pass": bool(independent_q >= 16.0 and high_q <= 8.0),
                "independent_stage_median_q": independent_q,
                "high_correlation_stage_median_q": high_q,
                "criterion": "independent q >= 16 and high-correlation q <= 8",
            },
            "finite_trajectories": {
                "pass": all_finite,
                "criterion": "all simulated squared errors are finite",
            },
        }
    primary = criteria["server_time"]
    criteria["primary_gate_without_reproduction"] = {
        "pass": bool(
            all(item["pass"] for item in primary.values())
        ),
        "criterion": "all server-time criteria pass before reproduction gate",
    }
    criteria["sample_time_direction"] = {
        "pass": bool(
            criteria["sample_time"]["adaptive_vs_delay_only"]["mse_ratio"] < 1.0
            and criteria["sample_time"]["participation_response"]["pass"]
        ),
        "criterion": "delay-only direction and participation response retained",
    }
    return criteria


def configure_plot_style() -> None:
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


def plot_outputs(
    trajectory: pd.DataFrame,
    action_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: Path,
    config: StagewiseConfig,
) -> None:
    configure_plot_style()
    primary = trajectory[trajectory["alignment"] == "server_time"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for policy in POLICIES:
        subset = primary[primary["policy"] == policy]
        ax.plot(subset["step"], subset["mean_mse"], label=policy)
    for boundary in [4 * config.stage_length, 8 * config.stage_length]:
        ax.axvline(boundary, color="black", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Server updates")
    ax.set_ylabel("Mean squared error")
    ax.set_title("Predictable controller under dependence and delay shifts")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_stagewise_mse.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for policy in ["adaptive_joint", "delay_only", "proxy_oracle"]:
        subset = action_summary[
            (action_summary["alignment"] == "server_time")
            & (action_summary["policy"] == policy)
        ]
        ax.step(
            subset["stage"],
            subset["median_num_agents"],
            where="mid",
            marker="o",
            label=policy,
        )
    ax.axvspan(4, 7, color="tab:red", alpha=0.08, label=r"$\rho=0.9$")
    ax.set_yscale("log", base=2)
    ax.set_yticks(list(config.agent_counts))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Stage")
    ax.set_ylabel("Median accepted agents")
    ax.set_title("Participation response")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_participation_schedule.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for policy in [
        "adaptive_joint",
        "delay_only",
        "all_agents_adaptive_eta",
        "proxy_oracle",
    ]:
        subset = action_summary[
            (action_summary["alignment"] == "server_time")
            & (action_summary["policy"] == policy)
        ]
        ax.step(
            subset["stage"],
            subset["median_eta"],
            where="mid",
            marker="o",
            label=policy,
        )
    ax.axvspan(4, 7, color="tab:red", alpha=0.08)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Median step size")
    ax.set_title("Predictable step-size response")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_stepsize_schedule.png")
    plt.close(fig)

    primary_comparison = comparisons[
        comparisons["alignment"] == "server_time"
    ].copy()
    labels = {
        "delay_only": "delay-only",
        "all_agents_adaptive_eta": "all-agent adaptive step",
        "proxy_oracle": "proxy oracle",
    }
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    positions = np.arange(len(primary_comparison))
    values = primary_comparison["mse_ratio"].values
    lower = values - primary_comparison["bootstrap_95_lower"].values
    upper = primary_comparison["bootstrap_95_upper"].values - values
    ax.bar(positions, values, color=["tab:blue", "tab:orange", "tab:green"])
    ax.errorbar(
        positions,
        values,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [labels[value] for value in primary_comparison["denominator"]],
        rotation=12,
        ha="right",
    )
    ax.set_ylabel("Adaptive-joint MSE / baseline MSE")
    ax.set_title("Adapted high-correlation stages 5–7")
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_high_correlation_ratios.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = StagewiseConfig()
    print(
        "Starting EXP-004 with {0} paired seeds".format(args.num_seeds),
        flush=True,
    )
    errors, actions = run_experiment(
        config=config,
        num_seeds=args.num_seeds,
        base_seed=args.base_seed,
    )
    per_seed, trajectory, action_summary, comparisons = build_metric_tables(
        errors=errors,
        actions=actions,
        config=config,
        base_seed=args.base_seed,
        bootstrap_replications=args.bootstrap_replications,
    )
    criteria = evaluate_criteria(per_seed, actions, comparisons)

    actions.to_csv(args.output_dir / "actions.csv", index=False)
    action_summary.to_csv(args.output_dir / "action_summary.csv", index=False)
    per_seed.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    trajectory.to_csv(args.output_dir / "trajectory_summary.csv", index=False)
    comparisons.to_csv(args.output_dir / "paired_bootstrap_ratios.csv", index=False)
    plot_outputs(
        trajectory=trajectory,
        action_summary=action_summary,
        comparisons=comparisons,
        output_dir=args.output_dir,
        config=config,
    )

    summary = {
        "experiment_id": "EXP-004-stagewise-controller",
        "status": "COMPLETED_PENDING_REPRODUCTION",
        "num_paired_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        "bootstrap_replications": args.bootstrap_replications,
        "config": {
            "num_agents": config.num_agents,
            "agent_counts": list(config.agent_counts),
            "num_stages": config.num_stages,
            "stage_length": config.stage_length,
            "rho_schedule": list(config.rho_schedule),
            "max_delay_schedule": list(config.max_delay_schedule),
            "eta_grid": config.eta_grid.tolist(),
            "batch_size": config.batch_size,
        },
        "criteria": criteria,
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

    print(json.dumps(criteria, indent=2), flush=True)
    print("Outputs written to {0}".format(args.output_dir.resolve()), flush=True)


if __name__ == "__main__":
    main()

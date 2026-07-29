"""Run the first dependence-delay go/no-go experiment.

Outputs:
  sweep.csv
  policy_comparison.csv
  monte_carlo_validation.csv
  summary.json
  fig1_correlation_saturation.png
  fig2_agent_count_risk.png
  fig3_stability_region.png
  fig4_policy_gap.png
"""

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

from linear_model import ModelConfig, exact_risk, make_agent_delays, monte_carlo_risk


DEFAULT_AGENT_COUNTS = [1, 2, 4, 8, 16, 32]
DEFAULT_CORRELATIONS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "baseline",
    )
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--max-delay", type=int, default=16)
    parser.add_argument("--delay-exponent", type=float, default=1.25)
    parser.add_argument("--eta-min", type=float, default=0.0025)
    parser.add_argument("--eta-max", type=float, default=0.35)
    parser.add_argument("--eta-count", type=int, default=55)
    parser.add_argument("--mc-replications", type=int, default=4000)
    parser.add_argument("--mc-seed", type=int, default=20260729)
    parser.add_argument(
        "--common-noise-alignment",
        choices=["sample_time", "server_time"],
        default="sample_time",
        help=(
            "Whether the common Markov factor is evaluated at each agent's "
            "sample time or at the shared server/environment time."
        ),
    )
    return parser.parse_args()


def delay_profiles(
    agent_counts: Sequence[int],
    max_delay: int,
    exponent: float,
) -> Dict[str, np.ndarray]:
    max_agents = int(max(agent_counts))
    return {
        "synchronous": np.zeros(max_agents, dtype=int),
        "heterogeneous": make_agent_delays(
            max_agents=max_agents,
            max_delay=max_delay,
            exponent=exponent,
        ),
    }


def run_sweep(
    eta_grid: np.ndarray,
    correlations: Sequence[float],
    agent_counts: Sequence[int],
    profiles: Dict[str, np.ndarray],
    config: ModelConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    total = (
        len(eta_grid)
        * len(correlations)
        * len(agent_counts)
        * len(profiles)
    )
    completed = 0

    for scenario, all_delays in profiles.items():
        for rho in correlations:
            for num_agents in agent_counts:
                selected_delays = all_delays[:num_agents]
                for eta in eta_grid:
                    metrics = exact_risk(
                        eta=float(eta),
                        rho=float(rho),
                        num_agents=int(num_agents),
                        delays=selected_delays,
                        config=config,
                    )
                    rows.append(
                        {
                            "scenario": scenario,
                            "rho": float(rho),
                            "num_agents": int(num_agents),
                            "eta": float(eta),
                            "mean_delay": float(np.mean(selected_delays)),
                            "max_delay": int(np.max(selected_delays)),
                            **metrics,
                        }
                    )
                    completed += 1
            print(
                "completed scenario={0}, rho={1:g}: {2}/{3} exact evaluations".format(
                    scenario, rho, completed, total
                ),
                flush=True,
            )
    return pd.DataFrame(rows)


def best_rows(frame: pd.DataFrame) -> pd.DataFrame:
    stable = frame[np.isfinite(frame["finite_mse"])].copy()
    indices = stable.groupby(
        ["scenario", "rho", "num_agents"], sort=True
    )["finite_mse"].idxmin()
    return stable.loc[indices].sort_values(
        ["scenario", "rho", "num_agents"]
    )


def evaluate_policy(
    sweep: pd.DataFrame,
    profiles: Dict[str, np.ndarray],
    correlations: Sequence[float],
    config: ModelConfig,
) -> pd.DataFrame:
    """Compare joint oracle, delay-only selection, and fixed-q oracles."""

    rows: List[Dict[str, object]] = []
    hetero = sweep[sweep["scenario"] == "heterogeneous"].copy()
    finite = hetero[np.isfinite(hetero["finite_mse"])].copy()

    independent = finite[np.isclose(finite["rho"], 0.0)]
    delay_only_choice = independent.loc[independent["finite_mse"].idxmin()]

    for rho in correlations:
        subset = finite[np.isclose(finite["rho"], rho)]
        joint_choice = subset.loc[subset["finite_mse"].idxmin()]

        for policy, choice in [
            ("joint_oracle", joint_choice),
            ("delay_only", delay_only_choice),
        ]:
            chosen_q = int(choice["num_agents"])
            chosen_eta = float(choice["eta"])
            evaluated = exact_risk(
                eta=chosen_eta,
                rho=float(rho),
                num_agents=chosen_q,
                delays=profiles["heterogeneous"][:chosen_q],
                config=config,
            )
            rows.append(
                {
                    "rho": float(rho),
                    "policy": policy,
                    "num_agents": chosen_q,
                    "eta": chosen_eta,
                    "finite_mse": evaluated["finite_mse"],
                    "spectral_radius": evaluated["spectral_radius"],
                }
            )

        for policy, q in [
            ("single_agent_oracle_eta", 1),
            ("all_agents_oracle_eta", int(max(DEFAULT_AGENT_COUNTS))),
        ]:
            q_subset = subset[subset["num_agents"] == q]
            choice = q_subset.loc[q_subset["finite_mse"].idxmin()]
            rows.append(
                {
                    "rho": float(rho),
                    "policy": policy,
                    "num_agents": int(q),
                    "eta": float(choice["eta"]),
                    "finite_mse": float(choice["finite_mse"]),
                    "spectral_radius": float(choice["spectral_radius"]),
                }
            )
    result = pd.DataFrame(rows)
    oracle_risk = (
        result[result["policy"] == "joint_oracle"]
        .set_index("rho")["finite_mse"]
        .to_dict()
    )
    result["risk_ratio_to_joint_oracle"] = result.apply(
        lambda row: row["finite_mse"] / oracle_risk[row["rho"]], axis=1
    )
    return result


def run_mc_validation(
    sweep: pd.DataFrame,
    profiles: Dict[str, np.ndarray],
    config: ModelConfig,
    replications: int,
    seed: int,
) -> pd.DataFrame:
    best = best_rows(sweep)
    requested: List[Tuple[str, float, int]] = [
        ("synchronous", 0.0, 32),
        ("synchronous", 0.9, 32),
    ]

    high_corr_hetero = best[
        (best["scenario"] == "heterogeneous") & np.isclose(best["rho"], 0.9)
    ]
    selected = high_corr_hetero.loc[high_corr_hetero["finite_mse"].idxmin()]
    requested.append(("heterogeneous", 0.9, int(selected["num_agents"])))

    rows: List[Dict[str, object]] = []
    for index, (scenario, rho, q) in enumerate(requested):
        candidate = best[
            (best["scenario"] == scenario)
            & np.isclose(best["rho"], rho)
            & (best["num_agents"] == q)
        ].iloc[0]
        eta = float(candidate["eta"])
        exact = float(candidate["finite_mse"])
        mc = monte_carlo_risk(
            eta=eta,
            rho=rho,
            num_agents=q,
            delays=profiles[scenario][:q],
            config=config,
            num_replications=replications,
            seed=seed + index,
        )
        rows.append(
            {
                "scenario": scenario,
                "rho": rho,
                "num_agents": q,
                "eta": eta,
                "exact_mse": exact,
                **mc,
                "relative_error": abs(mc["mc_mse"] - exact) / exact,
                "exact_within_95pct_mc_interval": bool(
                    abs(mc["mc_mse"] - exact)
                    <= 1.96 * mc["mc_standard_error"]
                ),
            }
        )
        print(
            "Monte Carlo validation {0}/{1} complete".format(
                index + 1, len(requested)
            ),
            flush=True,
        )
    return pd.DataFrame(rows)


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


def plot_correlation_saturation(
    best: pd.DataFrame, output_dir: Path
) -> None:
    frame = best[best["scenario"] == "synchronous"]
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    for rho in [0.0, 0.3, 0.7, 0.9, 0.99]:
        subset = frame[np.isclose(frame["rho"], rho)]
        ax.plot(
            subset["num_agents"],
            subset["finite_mse"],
            marker="o",
            label=r"$\rho={0:g}$".format(rho),
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(DEFAULT_AGENT_COUNTS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Oracle finite-horizon MSE")
    ax.set_title("Correlation limits parallel variance reduction")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_correlation_saturation.png")
    plt.close(fig)


def plot_agent_count_risk(best: pd.DataFrame, output_dir: Path) -> None:
    frame = best[best["scenario"] == "heterogeneous"]
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    for rho in [0.0, 0.3, 0.7, 0.9, 0.99]:
        subset = frame[np.isclose(frame["rho"], rho)]
        ax.plot(
            subset["num_agents"],
            subset["finite_mse"],
            marker="o",
            label=r"$\rho={0:g}$".format(rho),
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(DEFAULT_AGENT_COUNTS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Accepted agents (fastest first)")
    ax.set_ylabel("Oracle finite-horizon MSE")
    ax.set_title("Benefit of more agents under heterogeneous staleness")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_agent_count_risk.png")
    plt.close(fig)


def plot_stability_region(sweep: pd.DataFrame, output_dir: Path) -> None:
    frame = sweep[
        (sweep["scenario"] == "heterogeneous") & np.isclose(sweep["rho"], 0.0)
    ]
    pivot = frame.pivot(
        index="eta", columns="num_agents", values="spectral_radius"
    ).sort_index()
    grid_default = plt.rcParams["axes.grid"]
    plt.rcParams["axes.grid"] = False
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.grid(False)
    image = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            0.5,
            len(pivot.columns) + 0.5,
            float(pivot.index.min()),
            float(pivot.index.max()),
        ],
        cmap="viridis",
        vmin=min(0.95, float(np.nanmin(pivot.values))),
        vmax=min(1.08, float(np.nanmax(pivot.values))),
    )
    ax.set_xticks(np.arange(1, len(pivot.columns) + 1))
    ax.set_xticklabels([str(value) for value in pivot.columns])
    ax.set_xlabel("Accepted agents (fastest first)")
    ax.set_ylabel("Step size")
    ax.set_title("Spectral radius under heterogeneous delays")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Spectral radius")
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_stability_region.png")
    plt.close(fig)
    plt.rcParams["axes.grid"] = grid_default


def plot_policy_gap(policy: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    labels = {
        "delay_only": "Delay-only selection",
        "all_agents_oracle_eta": "All agents, tuned step",
        "single_agent_oracle_eta": "Single agent, tuned step",
    }
    for name, label in labels.items():
        subset = policy[policy["policy"] == name]
        ax.plot(
            subset["rho"],
            subset["risk_ratio_to_joint_oracle"],
            marker="o",
            label=label,
        )
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("Cross-agent common-noise fraction")
    ax.set_ylabel("MSE / joint-oracle MSE")
    ax.set_title("Cost of ignoring dependence in joint tuning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_policy_gap.png")
    plt.close(fig)


def summarize(
    best: pd.DataFrame,
    policy: pd.DataFrame,
    mc: pd.DataFrame,
    profiles: Dict[str, np.ndarray],
    args: argparse.Namespace,
    config: ModelConfig,
) -> Dict[str, object]:
    sync_independent = best[
        (best["scenario"] == "synchronous") & np.isclose(best["rho"], 0.0)
    ].set_index("num_agents")
    sync_high_corr = best[
        (best["scenario"] == "synchronous") & np.isclose(best["rho"], 0.9)
    ].set_index("num_agents")
    hetero_high_corr = best[
        (best["scenario"] == "heterogeneous") & np.isclose(best["rho"], 0.9)
    ]
    joint_high_corr = policy[
        (policy["policy"] == "joint_oracle") & np.isclose(policy["rho"], 0.9)
    ].iloc[0]
    delay_only_high_corr = policy[
        (policy["policy"] == "delay_only") & np.isclose(policy["rho"], 0.9)
    ].iloc[0]

    independent_gain = float(
        sync_independent.loc[1, "finite_mse"]
        / sync_independent.loc[32, "finite_mse"]
    )
    correlated_gain = float(
        sync_high_corr.loc[1, "finite_mse"]
        / sync_high_corr.loc[32, "finite_mse"]
    )
    high_corr_optimal_q = int(
        hetero_high_corr.loc[hetero_high_corr["finite_mse"].idxmin()][
            "num_agents"
        ]
    )
    delay_only_gap = float(
        delay_only_high_corr["finite_mse"] / joint_high_corr["finite_mse"]
    )
    mc_max_relative_error = float(mc["relative_error"].max())
    all_mc_intervals_cover = bool(mc["exact_within_95pct_mc_interval"].all())

    go_no_go = {
        "correlation_saturation": {
            "pass": bool(correlated_gain < 0.5 * independent_gain),
            "independent_q1_to_q32_gain": independent_gain,
            "rho_0p9_q1_to_q32_gain": correlated_gain,
            "criterion": "high-correlation gain is less than half the independent gain",
        },
        "interior_parallelism_optimum": {
            "pass": bool(high_corr_optimal_q < 32),
            "rho_0p9_optimal_q": high_corr_optimal_q,
            "criterion": "joint oracle chooses fewer than all 32 agents",
        },
        "delay_only_suboptimality": {
            "pass": bool(delay_only_gap >= 1.2),
            "rho_0p9_risk_ratio": delay_only_gap,
            "criterion": "delay-only risk is at least 20% above joint oracle",
        },
        "exact_mc_agreement": {
            "pass": bool(mc_max_relative_error <= 0.05),
            "max_relative_error": mc_max_relative_error,
            "all_exact_values_in_95pct_mc_intervals": all_mc_intervals_cover,
            "criterion": "Monte Carlo relative error is at most 5%",
        },
    }

    return {
        "experiment_id": "EXP-001-dependence-delay-linear",
        "status": "COMPLETED",
        "model": {
            "curvature": config.curvature,
            "common_ar": config.common_ar,
            "idiosyncratic_ar": config.idiosyncratic_ar,
            "initial_error": config.initial_error,
            "horizon": config.horizon,
            "common_noise_alignment": config.common_noise_alignment,
        },
        "grid": {
            "agent_counts": DEFAULT_AGENT_COUNTS,
            "correlations": DEFAULT_CORRELATIONS,
            "eta_min": args.eta_min,
            "eta_max": args.eta_max,
            "eta_count": args.eta_count,
            "heterogeneous_delays": profiles["heterogeneous"].tolist(),
        },
        "go_no_go": go_no_go,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ModelConfig(
        horizon=args.horizon,
        common_noise_alignment=args.common_noise_alignment,
    )
    eta_grid = np.geomspace(args.eta_min, args.eta_max, args.eta_count)
    profiles = delay_profiles(
        agent_counts=DEFAULT_AGENT_COUNTS,
        max_delay=args.max_delay,
        exponent=args.delay_exponent,
    )

    print("Starting exact parameter sweep", flush=True)
    sweep = run_sweep(
        eta_grid=eta_grid,
        correlations=DEFAULT_CORRELATIONS,
        agent_counts=DEFAULT_AGENT_COUNTS,
        profiles=profiles,
        config=config,
    )
    sweep.to_csv(args.output_dir / "sweep.csv", index=False)

    best = best_rows(sweep)
    best.to_csv(args.output_dir / "best_by_setting.csv", index=False)

    policy = evaluate_policy(
        sweep=sweep,
        profiles=profiles,
        correlations=DEFAULT_CORRELATIONS,
        config=config,
    )
    policy.to_csv(args.output_dir / "policy_comparison.csv", index=False)

    mc = run_mc_validation(
        sweep=sweep,
        profiles=profiles,
        config=config,
        replications=args.mc_replications,
        seed=args.mc_seed,
    )
    mc.to_csv(args.output_dir / "monte_carlo_validation.csv", index=False)

    configure_plot_style()
    plot_correlation_saturation(best, args.output_dir)
    plot_agent_count_risk(best, args.output_dir)
    plot_stability_region(sweep, args.output_dir)
    plot_policy_gap(policy, args.output_dir)

    summary = summarize(
        best=best,
        policy=policy,
        mc=mc,
        profiles=profiles,
        args=args,
        config=config,
    )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary["go_no_go"], indent=2), flush=True)
    print("Outputs written to {0}".format(args.output_dir.resolve()), flush=True)


if __name__ == "__main__":
    main()

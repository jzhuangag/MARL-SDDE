"""Run EXP-007A correlation-limited delayed linear TD."""

import argparse
import concurrent.futures
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from budget_participation import AGENT_COUNTS
from linear_td_correlation import (
    BUDGETS,
    CORRELATIONS,
    MAX_DELAYS,
    LinearTDConfig,
    build_mrp,
    effective_participation_rows,
    generate_base_paths,
    observed_transition_pairs,
    simulate_td_eta_grid,
    td_noise_gradients,
)
from run_state_correlation import bootstrap_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "linear_td_correlation",
    )
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=20260930)
    parser.add_argument("--bootstrap-replications", type=int, default=2000)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    return parser.parse_args()


def run_experiment(
    config: LinearTDConfig,
    num_seeds: int,
    base_seed: int,
    workers: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    mrp = build_mrp(config)
    metric_rows: List[Dict[str, object]] = []
    lrv_rows: List[Dict[str, object]] = []

    def run_seed(
        seed_index: int,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        local_metrics: List[Dict[str, object]] = []
        local_lrv: List[Dict[str, object]] = []
        seed = base_seed + seed_index
        paths = generate_base_paths(seed, mrp, config)
        for rho in CORRELATIONS:
            current, following = observed_transition_pairs(paths, rho)
            gradients = td_noise_gradients(
                current, following, mrp, config
            )
            local_lrv.extend(
                effective_participation_rows(
                    gradients, rho, seed, config
                )
            )
            for max_delay in MAX_DELAYS:
                for q in AGENT_COUNTS:
                    update_cost = config.update_overhead + q
                    result = simulate_td_eta_grid(
                        current_states=current,
                        next_states=following,
                        mrp=mrp,
                        max_delay=max_delay,
                        num_agents=q,
                        config=config,
                    )
                    for eta_index, eta in enumerate(config.eta_grid):
                        for budget_index, budget in enumerate(BUDGETS):
                            local_metrics.append(
                                {
                                    "seed": int(seed),
                                    "seed_index": int(seed_index),
                                    "rho": float(rho),
                                    "max_delay": int(max_delay),
                                    "num_agents": int(q),
                                    "eta": float(eta),
                                    "budget": int(budget),
                                    "squared_parameter_error": float(
                                        result["errors"][
                                            eta_index, budget_index
                                        ]
                                    ),
                                    "updates": int(
                                        result["updates"][
                                            eta_index, budget_index
                                        ]
                                    ),
                                    "update_cost": int(update_cost),
                                    "charged_budget": int(
                                        result["charged_budgets"][
                                            eta_index, budget_index
                                        ]
                                    ),
                                    "finite": bool(
                                        result["finite"][
                                            eta_index, budget_index
                                        ]
                                    ),
                                }
                            )
        return local_metrics, local_lrv

    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        iterator = map(run_seed, range(num_seeds))
        executor = None
    else:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers
        )
        iterator = executor.map(run_seed, range(num_seeds))
    try:
        for seed_index, (seed_metrics, seed_lrv) in enumerate(iterator):
            metric_rows.extend(seed_metrics)
            lrv_rows.extend(seed_lrv)
            if (seed_index + 1) % max(1, min(4, num_seeds)) == 0:
                print(
                    "completed {0}/{1} paired seeds".format(
                        seed_index + 1, num_seeds
                    ),
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return pd.DataFrame(metric_rows), pd.DataFrame(lrv_rows), mrp


def select_best_policies(
    metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = (
        metrics.groupby(
            ["rho", "max_delay", "budget", "num_agents", "eta"],
            as_index=False,
        )["squared_parameter_error"]
        .mean()
        .rename(columns={"squared_parameter_error": "mean_error"})
    )
    best_eta = (
        aggregate.sort_values(
            ["mean_error", "eta", "num_agents"],
            ascending=[True, True, False],
        )
        .groupby(
            ["rho", "max_delay", "budget", "num_agents"],
            as_index=False,
            sort=True,
        )
        .first()
    )
    oracle = (
        best_eta.sort_values(
            ["mean_error", "num_agents", "eta"],
            ascending=[True, False, True],
        )
        .groupby(["rho", "max_delay", "budget"], as_index=False, sort=True)
        .first()
        .rename(
            columns={
                "num_agents": "oracle_num_agents",
                "eta": "oracle_eta",
                "mean_error": "oracle_mean_error",
            }
        )
    )
    return best_eta, oracle


def _cell_row(
    frame: pd.DataFrame,
    rho: float,
    max_delay: int,
    budget: int,
    num_agents: int = None,
) -> pd.Series:
    selected = frame[
        np.isclose(frame["rho"], rho)
        & (frame["max_delay"] == max_delay)
        & (frame["budget"] == budget)
    ]
    if num_agents is not None:
        selected = selected[selected["num_agents"] == num_agents]
    if len(selected) != 1:
        raise RuntimeError("registered cell lookup is not unique")
    return selected.iloc[0]


def _seed_errors_at_action(
    metrics: pd.DataFrame,
    rho: float,
    max_delay: int,
    budget: int,
    num_agents: int,
    eta: float,
) -> np.ndarray:
    selected = metrics[
        np.isclose(metrics["rho"], rho)
        & (metrics["max_delay"] == max_delay)
        & (metrics["budget"] == budget)
        & (metrics["num_agents"] == num_agents)
        & np.isclose(metrics["eta"], eta, rtol=0.0, atol=1e-15)
    ].sort_values("seed_index")
    return selected["squared_parameter_error"].to_numpy(dtype=float)


def build_material_comparisons(
    metrics: pd.DataFrame,
    best_eta: pd.DataFrame,
    oracle: pd.DataFrame,
    replications: int,
    base_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    independent = _cell_row(oracle, 0.0, 8, max(BUDGETS))
    correlated = _cell_row(oracle, 0.9, 8, max(BUDGETS))
    q_independent = int(independent["oracle_num_agents"])
    q_correlated = int(correlated["oracle_num_agents"])
    rows = []
    comparisons = (
        (
            "independent_oracle_q_vs_correlated_q",
            0.0,
            q_independent,
            q_correlated,
            base_seed + 10000,
        ),
        (
            "correlated_oracle_q_vs_independent_q",
            0.9,
            q_correlated,
            q_independent,
            base_seed + 10001,
        ),
    )
    for name, rho, numerator_q, denominator_q, seed in comparisons:
        numerator_action = _cell_row(
            best_eta, rho, 8, max(BUDGETS), numerator_q
        )
        denominator_action = _cell_row(
            best_eta, rho, 8, max(BUDGETS), denominator_q
        )
        numerator = _seed_errors_at_action(
            metrics,
            rho,
            8,
            max(BUDGETS),
            numerator_q,
            float(numerator_action["eta"]),
        )
        denominator = _seed_errors_at_action(
            metrics,
            rho,
            8,
            max(BUDGETS),
            denominator_q,
            float(denominator_action["eta"]),
        )
        rows.append(
            {
                "comparison": name,
                "rho": rho,
                "numerator_q": numerator_q,
                "denominator_q": denominator_q,
                "numerator_eta": float(numerator_action["eta"]),
                "denominator_eta": float(denominator_action["eta"]),
                **bootstrap_ratio(
                    numerator,
                    denominator,
                    replications,
                    seed,
                ),
            }
        )
    return pd.DataFrame(rows), {
        "independent_oracle_q": q_independent,
        "correlated_oracle_q": q_correlated,
    }


def evaluate_gates(
    metrics: pd.DataFrame,
    lrv: pd.DataFrame,
    best_eta: pd.DataFrame,
    oracle: pd.DataFrame,
    comparisons: pd.DataFrame,
    num_seeds: int,
    config: LinearTDConfig,
) -> Dict[str, object]:
    independent_neff = float(
        lrv[
            np.isclose(lrv["rho"], 0.0)
            & (lrv["num_agents"] == 32)
        ]["effective_participation"].median()
    )
    correlated_neff = float(
        lrv[
            np.isclose(lrv["rho"], 0.9)
            & (lrv["num_agents"] == 32)
        ]["effective_participation"].median()
    )
    independent_cell = _cell_row(oracle, 0.0, 8, max(BUDGETS))
    correlated_cell = _cell_row(oracle, 0.9, 8, max(BUDGETS))
    q_independent = int(independent_cell["oracle_num_agents"])
    q_correlated = int(correlated_cell["oracle_num_agents"])
    comparison_lookup = comparisons.set_index("comparison")
    independent_ratio = float(
        comparison_lookup.loc[
            "independent_oracle_q_vs_correlated_q", "ratio"
        ]
    )
    correlated_ratio = float(
        comparison_lookup.loc[
            "correlated_oracle_q_vs_independent_q", "ratio"
        ]
    )
    eta_zero = best_eta[best_eta["max_delay"] == 0][
        ["rho", "budget", "num_agents", "eta"]
    ].rename(columns={"eta": "eta_d0"})
    eta_large = best_eta[best_eta["max_delay"] == 32][
        ["rho", "budget", "num_agents", "eta"]
    ].rename(columns={"eta": "eta_d32"})
    eta_pairs = eta_zero.merge(
        eta_large,
        on=["rho", "budget", "num_agents"],
        validate="one_to_one",
    )
    eta_fraction = float(
        (eta_pairs["eta_d32"] <= eta_pairs["eta_d0"]).mean()
    )
    expected_rows = (
        num_seeds
        * len(CORRELATIONS)
        * len(MAX_DELAYS)
        * len(AGENT_COUNTS)
        * config.eta_count
        * len(BUDGETS)
    )
    validity = bool(
        len(metrics) == expected_rows
        and metrics["finite"].all()
        and np.isfinite(metrics["squared_parameter_error"]).all()
        and (
            metrics["charged_budget"]
            == metrics["updates"] * metrics["update_cost"]
        ).all()
        and (metrics["charged_budget"] <= metrics["budget"]).all()
        and np.isfinite(lrv["trace_lrv"]).all()
        and (lrv["trace_lrv"] > 0).all()
    )
    gates: Dict[str, object] = {
        "independent_speedup": {
            "pass": bool(independent_neff >= 16.0),
            "median_neff_q32": independent_neff,
        },
        "correlation_saturation": {
            "pass": bool(correlated_neff <= 4.0),
            "median_neff_q32": correlated_neff,
        },
        "participation_transition": {
            "pass": bool(q_independent >= 16 and q_correlated <= 4),
            "independent_oracle_q": q_independent,
            "correlated_oracle_q": q_correlated,
            "max_delay": 8,
            "budget": max(BUDGETS),
        },
        "material_resource_value": {
            "pass": bool(
                independent_ratio <= 0.90
                and correlated_ratio <= 0.90
            ),
            "independent_ratio": independent_ratio,
            "correlated_ratio": correlated_ratio,
        },
        "delay_step_size_consistency": {
            "pass": bool(eta_fraction >= 0.80),
            "fraction": eta_fraction,
            "comparison_count": int(len(eta_pairs)),
        },
        "accounting_and_numerical_validity": {
            "pass": validity,
            "observed_rows": int(len(metrics)),
            "expected_rows": int(expected_rows),
        },
    }
    gates["overall"] = {
        "pass": bool(all(value["pass"] for value in gates.values())),
        "criterion": "all six linear-TD mechanism gates pass",
    }
    return gates


def save_mrp_artifacts(
    mrp: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    pd.DataFrame(mrp["transition"]).to_csv(
        output_dir / "mrp_transition.csv", index=False
    )
    pd.DataFrame(mrp["features"]).to_csv(
        output_dir / "mrp_features.csv", index=False
    )
    pd.DataFrame(mrp["a_matrix"]).to_csv(
        output_dir / "mrp_a_matrix.csv", index=False
    )
    pd.DataFrame(
        {
            "state": np.arange(len(mrp["stationary"])),
            "stationary": mrp["stationary"],
            "reward": mrp["reward"],
            "projected_value": mrp["projected_value"],
            "b_vector_padded": np.pad(
                mrp["b_vector"],
                (0, len(mrp["stationary"]) - len(mrp["b_vector"])),
                constant_values=np.nan,
            ),
            "theta_star_padded": np.pad(
                mrp["theta_star"],
                (0, len(mrp["stationary"]) - len(mrp["theta_star"])),
                constant_values=np.nan,
            ),
        }
    ).to_csv(output_dir / "mrp_vectors.csv", index=False)


def plot_outputs(
    lrv: pd.DataFrame,
    best_eta: pd.DataFrame,
    oracle: pd.DataFrame,
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
    neff = (
        lrv.groupby(["rho", "num_agents"], as_index=False)[
            "effective_participation"
        ]
        .median()
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for rho in CORRELATIONS:
        selected = neff[np.isclose(neff["rho"], rho)]
        ax.plot(
            selected["num_agents"],
            selected["effective_participation"],
            marker="o",
            label="rho={0:g}".format(rho),
        )
    ax.plot(AGENT_COUNTS, AGENT_COUNTS, color="black", linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(AGENT_COUNTS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Participating agents q")
    ax.set_ylabel("Median effective participation")
    ax.set_title("Cross-agent correlation limits TD speedup")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_effective_participation.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    for ax, budget in zip(axes, BUDGETS):
        selected = oracle[oracle["budget"] == budget]
        for max_delay in MAX_DELAYS:
            curve = selected[selected["max_delay"] == max_delay]
            ax.step(
                curve["rho"],
                curve["oracle_num_agents"],
                where="mid",
                marker="o",
                label="D={0}".format(max_delay),
            )
        ax.set_yscale("log", base=2)
        ax.set_yticks(AGENT_COUNTS)
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter()
        )
        ax.set_xlabel("Pairwise sharing probability rho")
        ax.set_title("Budget {0}".format(budget))
    axes[0].set_ylabel("Oracle participating agents")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_oracle_participation.png")
    plt.close(fig)

    selected = best_eta[
        (best_eta["max_delay"] == 8)
        & (best_eta["budget"] == max(BUDGETS))
        & (
            np.isclose(best_eta["rho"], 0.0)
            | np.isclose(best_eta["rho"], 0.9)
        )
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for rho in (0.0, 0.9):
        curve = selected[np.isclose(selected["rho"], rho)]
        ax.plot(
            curve["num_agents"],
            curve["mean_error"],
            marker="o",
            label="rho={0:g}".format(rho),
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(AGENT_COUNTS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Participating agents q")
    ax.set_ylabel("Best-step-size mean squared error")
    ax.set_title("Finite-budget participation transition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_budget_phase_transition.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = LinearTDConfig()
    print(
        "Starting EXP-007A with {0} paired seeds".format(args.num_seeds),
        flush=True,
    )
    metrics, lrv, mrp = run_experiment(
        config, args.num_seeds, args.base_seed, args.workers
    )
    best_eta, oracle = select_best_policies(metrics)
    comparisons, endpoint_actions = build_material_comparisons(
        metrics,
        best_eta,
        oracle,
        args.bootstrap_replications,
        args.base_seed,
    )
    gates = evaluate_gates(
        metrics,
        lrv,
        best_eta,
        oracle,
        comparisons,
        args.num_seeds,
        config,
    )
    metrics.to_csv(
        args.output_dir / "per_seed_td_metrics.csv", index=False
    )
    lrv.to_csv(
        args.output_dir / "effective_participation.csv", index=False
    )
    best_eta.to_csv(args.output_dir / "best_eta_by_q.csv", index=False)
    oracle.to_csv(
        args.output_dir / "oracle_participation_surface.csv", index=False
    )
    comparisons.to_csv(
        args.output_dir / "paired_endpoint_comparisons.csv", index=False
    )
    save_mrp_artifacts(mrp, args.output_dir)
    plot_outputs(lrv, best_eta, oracle, args.output_dir)
    summary = {
        "experiment_id": "EXP-007A-linear-td-correlation",
        "status": "COMPLETED_PENDING_REPRODUCTION",
        "num_paired_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        "bootstrap_replications": args.bootstrap_replications,
        "workers": args.workers,
        "endpoint_actions": endpoint_actions,
        "config": {
            "num_states": config.num_states,
            "num_features": config.num_features,
            "num_agents": config.num_agents,
            "gamma": config.gamma,
            "update_overhead": config.update_overhead,
            "correlations": list(CORRELATIONS),
            "max_delays": list(MAX_DELAYS),
            "budgets": list(BUDGETS),
            "agent_counts": list(AGENT_COUNTS),
            "eta_grid": config.eta_grid.tolist(),
            "lrv_batch_size": config.lrv_batch_size,
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
    print("Outputs written to {0}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()

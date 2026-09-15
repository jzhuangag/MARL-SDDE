"""EXP-013B preregistered realizable nonlinear TD confirmation."""

import argparse
import json
import platform
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from run_nonlinear_td_smoke import (
    AGENT_COUNTS,
    LEARNING_RATE,
    MESSAGE_BUDGET,
    SERVER_OVERHEAD,
)
from run_realizable_td_smoke import (
    REWARD_NOISE_STANDARD_DEVIATION,
    TEACHER_SEED,
    run_configuration,
)


CORRELATIONS = (0.0, 0.25, 0.5, 0.9)
DELAYS = (0, 8)
BOOTSTRAP_REPLICATIONS = 20000
BOOTSTRAP_SEED = 20270702


def paired_cluster_ratio(
    metrics: pd.DataFrame,
    rho: float,
    numerator_q: int,
    denominator_q: int,
    bootstrap_replications: int = BOOTSTRAP_REPLICATIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """Compute a paired geometric ratio and one-sided 99% cluster UCL."""

    selected = metrics[metrics["rho"] == rho]
    pivot = selected.pivot(
        index=["seed", "delay"],
        columns="num_agents",
        values="teacher_mse",
    )
    log_ratios = np.log(
        pivot[numerator_q].to_numpy()
        / pivot[denominator_q].to_numpy()
    )
    seed_values = (
        pd.Series(log_ratios, index=pivot.index)
        .groupby(level="seed")
        .mean()
        .to_numpy()
    )
    rng = np.random.RandomState(bootstrap_seed)
    draws = rng.choice(
        seed_values,
        size=(bootstrap_replications, len(seed_values)),
        replace=True,
    )
    bootstrap_ratios = np.exp(draws.mean(axis=1))
    return {
        "ratio": float(np.exp(seed_values.mean())),
        "upper_99": float(np.quantile(bootstrap_ratios, 0.99)),
        "num_seed_clusters": int(len(seed_values)),
    }


def delay_ratios(
    metrics: pd.DataFrame,
    rho: float,
    numerator_q: int,
    denominator_q: int,
) -> Dict[str, float]:
    """Return paired geometric ratios separately for each delay."""

    result = {}
    for delay in DELAYS:
        selected = metrics[
            (metrics["rho"] == rho)
            & (metrics["delay"] == delay)
        ]
        pivot = selected.pivot(
            index="seed",
            columns="num_agents",
            values="teacher_mse",
        )
        ratio = np.exp(
            np.mean(
                np.log(
                    pivot[numerator_q].to_numpy()
                    / pivot[denominator_q].to_numpy()
                )
            )
        )
        result[str(delay)] = float(ratio)
    return result


def oracle_choices(
    metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Choose q by mean log error across delays for each seed and rho."""

    grouped = (
        metrics.assign(log_mse=np.log(metrics["teacher_mse"]))
        .groupby(["seed", "rho", "num_agents"], as_index=False)
        .agg(mean_log_mse=("log_mse", "mean"))
        .sort_values(
            ["seed", "rho", "mean_log_mse", "num_agents"],
            kind="mergesort",
        )
    )
    choices = (
        grouped.groupby(["seed", "rho"], as_index=False)
        .first()
        .rename(columns={"num_agents": "oracle_q"})
    )
    medians = {
        str(rho): float(
            choices.loc[choices["rho"] == rho, "oracle_q"].median()
        )
        for rho in CORRELATIONS
    }
    return choices, medians


def analyze(metrics: pd.DataFrame) -> Dict[str, object]:
    """Apply the five preregistered gates."""

    low_ratio = paired_cluster_ratio(metrics, 0.0, 32, 1)
    high_ratio = paired_cluster_ratio(metrics, 0.9, 4, 32)
    low_by_delay = delay_ratios(metrics, 0.0, 32, 1)
    high_by_delay = delay_ratios(metrics, 0.9, 4, 32)
    choices, medians = oracle_choices(metrics)
    gates = {
        "all_finite": bool(
            metrics["finite"].all()
            and np.isfinite(metrics["teacher_mse"]).all()
        ),
        "low_rho_large_q_upper_99_below_0_70": bool(
            low_ratio["upper_99"] < 0.70
        ),
        "high_rho_small_q_upper_99_below_0_85": bool(
            high_ratio["upper_99"] < 0.85
        ),
        "oracle_choice_shift": bool(
            medians["0.0"] >= 16 and medians["0.9"] <= 4
        ),
        "both_delays_preserve_direction_and_magnitude": bool(
            max(low_by_delay.values()) < 0.80
            and max(high_by_delay.values()) < 0.90
        ),
    }
    return {
        "experiment": "EXP-013B",
        "evidence_status": "preregistered_confirmation",
        "all_gates_pass": bool(all(gates.values())),
        "gates": gates,
        "low_rho_q32_over_q1": low_ratio,
        "high_rho_q4_over_q32": high_ratio,
        "low_rho_ratios_by_delay": low_by_delay,
        "high_rho_ratios_by_delay": high_by_delay,
        "median_oracle_q_by_rho": medians,
        "oracle_choice_counts": (
            choices.groupby(["rho", "oracle_q"])
            .size()
            .rename("count")
            .reset_index()
            .to_dict(orient="records")
        ),
    }


def save_figure(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Plot geometric mean teacher MSE by correlation and delay."""

    summary = (
        metrics.assign(log_mse=np.log(metrics["teacher_mse"]))
        .groupby(["rho", "delay", "num_agents"], as_index=False)
        .agg(log_mse=("log_mse", "mean"))
    )
    summary["geometric_mse"] = np.exp(summary["log_mse"])
    figure, axes = plt.subplots(
        2, 2, figsize=(9.0, 7.2), sharex=True, sharey=True
    )
    for axis, rho in zip(axes.ravel(), CORRELATIONS):
        for delay, marker in zip(DELAYS, ("o", "s")):
            group = summary[
                (summary["rho"] == rho)
                & (summary["delay"] == delay)
            ]
            axis.plot(
                group["num_agents"],
                group["geometric_mse"],
                marker=marker,
                label=f"$D={delay}$",
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(f"$\\rho={rho:g}$")
        axis.grid(alpha=0.3)
        axis.legend()
    for axis in axes[-1]:
        axis.set_xlabel("participating agents $q$")
    for axis in axes[:, 0]:
        axis.set_ylabel("geometric mean teacher MSE")
    figure.tight_layout()
    figure.savefig(
        output_dir / "realizable_td_confirmation.png", dpi=180
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=20270701)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "realizable_td_confirmation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for seed_index in range(args.num_seeds):
        seed = args.base_seed + seed_index
        for rho in CORRELATIONS:
            for delay in DELAYS:
                for num_agents in AGENT_COUNTS:
                    rows.append(
                        run_configuration(
                            seed, rho, delay, num_agents
                        )
                    )
    metrics = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    summary = analyze(metrics)
    summary["configuration"] = {
        "num_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        "correlations": list(CORRELATIONS),
        "delays": list(DELAYS),
        "agent_counts": list(AGENT_COUNTS),
        "message_budget": MESSAGE_BUDGET,
        "server_overhead": SERVER_OVERHEAD,
        "learning_rate": LEARNING_RATE,
        "reward_noise_standard_deviation": (
            REWARD_NOISE_STANDARD_DEVIATION
        ),
        "teacher_seed": TEACHER_SEED,
        "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    summary["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    choices, _ = oracle_choices(metrics)
    choices.to_csv(
        args.output_dir / "oracle_choices.csv", index=False
    )
    save_figure(metrics, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

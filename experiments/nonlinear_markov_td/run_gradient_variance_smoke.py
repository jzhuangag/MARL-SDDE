"""Mechanism smoke test for correlation-limited nonlinear TD gradients."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from run_nonlinear_td_smoke import (
    GAMMA,
    MIXING,
    STATE_DIMENSION,
    ValueNetwork,
    teacher_value,
)


AGENT_COUNTS = (1, 2, 4, 8, 16, 32)
CORRELATIONS = (0.0, 0.25, 0.9)


def flattened_gradient(model: ValueNetwork) -> np.ndarray:
    """Return a copied flattened parameter gradient."""

    return np.concatenate(
        [
            parameter.grad.detach().cpu().numpy().ravel()
            for parameter in model.parameters()
        ]
    ).astype(np.float64, copy=False)


def sample_gradient(
    model: ValueNetwork,
    rng: np.random.RandomState,
    rho: float,
    num_agents: int,
) -> np.ndarray:
    """Sample one averaged semi-gradient with equicorrelated transitions."""

    innovations_scale = np.sqrt(1.0 - MIXING ** 2)
    current_paths = rng.standard_normal(
        (num_agents + 1, STATE_DIMENSION)
    ).astype(np.float32)
    following_paths = (
        MIXING * current_paths
        + innovations_scale
        * rng.standard_normal(
            (num_agents + 1, STATE_DIMENSION)
        ).astype(np.float32)
    )
    sources = np.arange(1, num_agents + 1)
    shared = (
        rng.random_sample(num_agents) < np.sqrt(rho)
    )
    sources = np.where(shared, 0, sources)
    current = torch.from_numpy(current_paths[sources])
    following = torch.from_numpy(following_paths[sources])
    rewards = (
        teacher_value(current)
        - GAMMA * teacher_value(following)
    )
    prediction = model(current)
    with torch.no_grad():
        target = rewards + GAMMA * model(following)
    loss = 0.5 * ((prediction - target) ** 2).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    return flattened_gradient(model)


def estimate_trace_variance(
    model: ValueNetwork,
    seed: int,
    rho: float,
    num_agents: int,
    replicates: int,
) -> Dict[str, float]:
    """Estimate the covariance trace with vector Welford updates."""

    rng = np.random.RandomState(seed)
    mean = None
    sum_squared_deviation = None
    for replicate in range(1, replicates + 1):
        gradient = sample_gradient(model, rng, rho, num_agents)
        if mean is None:
            mean = np.zeros_like(gradient)
            sum_squared_deviation = np.zeros_like(gradient)
        delta = gradient - mean
        mean += delta / replicate
        sum_squared_deviation += delta * (gradient - mean)
    trace_variance = float(
        np.sum(sum_squared_deviation) / (replicates - 1)
    )
    return {
        "rho": rho,
        "num_agents": num_agents,
        "replicates": replicates,
        "trace_variance": trace_variance,
        "mean_gradient_squared_norm": float(np.dot(mean, mean)),
    }


def add_theory_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    """Normalize each curve and attach the equicorrelation prediction."""

    frames = []
    for rho, group in metrics.groupby("rho", sort=True):
        group = group.copy()
        variance_one = float(
            group.loc[
                group["num_agents"] == 1, "trace_variance"
            ].iloc[0]
        )
        group["normalized_trace_variance"] = (
            group["trace_variance"] / variance_one
        )
        group["theory_ratio"] = rho + (
            1.0 - rho
        ) / group["num_agents"]
        group["absolute_ratio_error"] = np.abs(
            group["normalized_trace_variance"]
            - group["theory_ratio"]
        )
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def save_figure(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Plot empirical and theoretical variance ratios."""

    figure, axis = plt.subplots(figsize=(6.4, 4.3))
    colors = plt.cm.viridis(
        np.linspace(0.1, 0.9, len(CORRELATIONS))
    )
    for rho, color in zip(CORRELATIONS, colors):
        group = metrics[metrics["rho"] == rho]
        axis.plot(
            group["num_agents"],
            group["normalized_trace_variance"],
            marker="o",
            color=color,
            label=f"empirical $\\rho={rho:g}$",
        )
        axis.plot(
            group["num_agents"],
            group["theory_ratio"],
            linestyle="--",
            color=color,
            alpha=0.8,
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("participating agents $q$")
    axis.set_ylabel("normalized gradient covariance trace")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(
        output_dir / "gradient_variance_smoke.png", dpi=180
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20270517)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "gradient_variance_smoke",
    )
    return parser.parse_args()


def ordered_configurations() -> Iterable[tuple]:
    for rho in CORRELATIONS:
        for num_agents in AGENT_COUNTS:
            yield rho, num_agents


def main() -> None:
    args = parse_args()
    if args.replicates < 2:
        raise ValueError("--replicates must be at least 2")
    torch.manual_seed(20270516)
    torch.set_num_threads(1)
    model = ValueNetwork()
    rows = []
    for index, (rho, num_agents) in enumerate(
        ordered_configurations()
    ):
        rows.append(
            estimate_trace_variance(
                model=model,
                seed=args.seed + 1009 * index,
                rho=rho,
                num_agents=num_agents,
                replicates=args.replicates,
            )
        )
    metrics = add_theory_columns(pd.DataFrame(rows))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    save_figure(metrics, args.output_dir)
    summary = {
        "experiment": "EXP-013A-gradient-variance-smoke",
        "evidence_status": "implementation_only",
        "replicates_per_configuration": args.replicates,
        "configurations": int(len(metrics)),
        "maximum_absolute_ratio_error": float(
            metrics["absolute_ratio_error"].max()
        ),
        "mean_absolute_ratio_error": float(
            metrics["absolute_ratio_error"].mean()
        ),
        "rows": metrics.to_dict(orient="records"),
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

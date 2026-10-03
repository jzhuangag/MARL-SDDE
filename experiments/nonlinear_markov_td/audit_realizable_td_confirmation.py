"""Post-hoc descriptive audit for the failed EXP-013B confirmation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


AUDIT_BOOTSTRAP_REPLICATIONS = 20000
AUDIT_BOOTSTRAP_SEED = 20270800


def clustered_log_values(
    metrics: pd.DataFrame,
    rho: float,
    numerator_q: int,
    denominator_q: int,
) -> np.ndarray:
    selected = metrics[metrics["rho"] == rho]
    pivot = selected.pivot(
        index=["seed", "delay"],
        columns="num_agents",
        values="teacher_mse",
    )
    values = np.log(
        pivot[numerator_q].to_numpy()
        / pivot[denominator_q].to_numpy()
    )
    return (
        pd.Series(values, index=pivot.index)
        .groupby(level="seed")
        .mean()
        .to_numpy()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.metrics)
    geometric = (
        metrics.assign(log_mse=np.log(metrics["teacher_mse"]))
        .groupby(["rho", "delay", "num_agents"], as_index=False)
        .agg(log_mse=("log_mse", "mean"))
    )
    geometric["geometric_mse"] = np.exp(geometric["log_mse"])
    high_seed_ratios = {}
    for delay in (0, 8):
        pivot = metrics[
            (metrics["rho"] == 0.9)
            & (metrics["delay"] == delay)
        ].pivot(
            index="seed",
            columns="num_agents",
            values="teacher_mse",
        )
        ratios = pivot[4] / pivot[32]
        high_seed_ratios[str(delay)] = {
            "q4_wins": int((ratios < 1).sum()),
            "total_seeds": int(len(ratios)),
            "quantiles": {
                str(level): float(ratios.quantile(level))
                for level in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            },
        }
    low = clustered_log_values(metrics, 0.0, 32, 1)
    high = clustered_log_values(metrics, 0.9, 32, 1)
    interaction = high - low
    rng = np.random.RandomState(AUDIT_BOOTSTRAP_SEED)
    draws = rng.choice(
        interaction,
        size=(
            AUDIT_BOOTSTRAP_REPLICATIONS,
            len(interaction),
        ),
        replace=True,
    )
    interaction_bootstrap = np.exp(draws.mean(axis=1))
    audit = {
        "experiment": "EXP-013B-posthoc-audit",
        "evidence_status": "descriptive_not_preregistered",
        "rows": int(len(metrics)),
        "duplicate_keys": int(
            metrics.duplicated(
                ["seed", "rho", "delay", "num_agents"]
            ).sum()
        ),
        "finite_runs": int(metrics["finite"].sum()),
        "geometric_mse": geometric.drop(
            columns="log_mse"
        ).to_dict(orient="records"),
        "high_rho_q4_over_q32_seed_distribution": high_seed_ratios,
        "q32_over_q1_interaction": {
            "low_rho_ratio": float(np.exp(low.mean())),
            "high_rho_ratio": float(np.exp(high.mean())),
            "high_over_low": float(np.exp(interaction.mean())),
            "descriptive_lower_99": float(
                np.quantile(interaction_bootstrap, 0.01)
            ),
            "bootstrap_replications": (
                AUDIT_BOOTSTRAP_REPLICATIONS
            ),
            "bootstrap_seed": AUDIT_BOOTSTRAP_SEED,
        },
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

"""Merge deterministic EXP-013B scheduling chunks and run frozen analysis."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_nonlinear_td_smoke import (
    AGENT_COUNTS,
    LEARNING_RATE,
    MESSAGE_BUDGET,
    SERVER_OVERHEAD,
)
from run_realizable_td_confirmation import (
    BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_SEED,
    CORRELATIONS,
    DELAYS,
    analyze,
    oracle_choices,
    save_figure,
)
from run_realizable_td_smoke import (
    REWARD_NOISE_STANDARD_DEVIATION,
    TEACHER_SEED,
)


EXPECTED_SEEDS = tuple(range(20270701, 20270733))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [
        pd.read_csv(chunk_dir / "metrics.csv")
        for chunk_dir in args.chunk_dirs
    ]
    metrics = pd.concat(frames, ignore_index=True)
    key = ["seed", "rho", "delay", "num_agents"]
    if metrics.duplicated(key).any():
        raise RuntimeError("duplicate configuration keys across chunks")
    observed_seeds = tuple(
        sorted(int(seed) for seed in metrics["seed"].unique())
    )
    if observed_seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            f"unexpected seed set: {observed_seeds}"
        )
    expected_rows = (
        len(EXPECTED_SEEDS)
        * len(CORRELATIONS)
        * len(DELAYS)
        * len(AGENT_COUNTS)
    )
    if len(metrics) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} rows, observed {len(metrics)}"
        )
    metrics = metrics.sort_values(key, kind="mergesort").reset_index(
        drop=True
    )
    summary = analyze(metrics)
    summary["configuration"] = {
        "num_seeds": len(EXPECTED_SEEDS),
        "base_seed": EXPECTED_SEEDS[0],
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
    summary["execution"] = {
        "scheduler": "four_disjoint_eight_seed_cpu_chunks",
        "chunk_directories": [
            str(path) for path in args.chunk_dirs
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    choices, _ = oracle_choices(metrics)
    choices.to_csv(
        args.output_dir / "oracle_choices.csv", index=False
    )
    save_figure(metrics, args.output_dir)
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

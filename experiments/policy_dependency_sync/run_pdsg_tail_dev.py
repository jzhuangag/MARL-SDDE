"""Outcome-development smoke for the HalfCheetah policy-dependency tail.

This is explicitly not a preregistered confirmation experiment and its reset
seeds must never be reused by a later registered audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .halfcheetah_tail import (
    ACTION_DIM,
    finite_horizon_return,
    fixed_policy_parameters,
    influence_matrix,
    summarize_influence,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--horizons", type=int, nargs="+", default=[40, 80])
    parser.add_argument("--owner-steps", type=float, nargs="+", default=[0.01, 0.02])
    parser.add_argument("--donor-step", type=float, default=0.08)
    parser.add_argument("--policy-seed", type=int, default=731)
    parser.add_argument("--first-reset-seed", type=int, default=81001)
    args = parser.parse_args()

    import gymnasium as gym

    args.output.mkdir(parents=True, exist_ok=False)
    theta = fixed_policy_parameters(args.policy_seed)
    rows: list[dict[str, float | int]] = []
    matrices: list[dict[str, object]] = []
    env = gym.make("HalfCheetah-v5")
    try:
        for seed_offset in range(args.seeds):
            reset_seed = args.first_reset_seed + seed_offset
            for horizon in args.horizons:
                for owner_step in args.owner_steps:
                    objective = lambda candidate, rs=reset_seed, hz=horizon: finite_horizon_return(
                        env, candidate, rs, hz
                    )
                    influence = influence_matrix(
                        objective,
                        theta,
                        owner_step=owner_step,
                        donor_step=args.donor_step,
                    )
                    summary = summarize_influence(
                        influence,
                        random_graphs=512,
                        graph_seed=reset_seed + 17 * horizon,
                    )
                    rows.append(
                        {
                            "reset_seed": reset_seed,
                            "horizon": horizon,
                            "owner_step": owner_step,
                            "donor_step": args.donor_step,
                            **summary.__dict__,
                        }
                    )
                    matrices.append(
                        {
                            "reset_seed": reset_seed,
                            "horizon": horizon,
                            "owner_step": owner_step,
                            "donor_step": args.donor_step,
                            "influence": influence.tolist(),
                        }
                    )
    finally:
        env.close()

    csv_path = args.output / "summary_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    matrix_path = args.output / "influence_matrices.json"
    matrix_path.write_text(json.dumps(matrices, indent=2), encoding="utf-8")

    finite = [row for row in rows if np.isfinite(row["physical_fraction"])]
    summary = {
        "status": "development_only_not_confirmation",
        "environment": "Gymnasium HalfCheetah-v5",
        "partition": "Multi-Agent MuJoCo 6x1",
        "seeds": args.seeds,
        "rows": len(rows),
        "rollouts_per_row": ACTION_DIM * (ACTION_DIM - 1) * 4 * 4 * 4,
        "physical_fraction_median": float(np.median([row["physical_fraction"] for row in finite])),
        "physical_minus_random_median": float(
            np.median([row["physical_minus_random_median"] for row in finite])
        ),
        "neighbor_top_rate_median": float(np.median([row["neighbor_top_rate"] for row in finite])),
        "owner_step_stability": {},
    }
    for horizon in args.horizons:
        subset = [row for row in finite if row["horizon"] == horizon]
        summary["owner_step_stability"][str(horizon)] = {
            "physical_fraction_range": float(
                max(row["physical_fraction"] for row in subset)
                - min(row["physical_fraction"] for row in subset)
            )
        }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    hashes = {path.name: sha256(path) for path in (csv_path, matrix_path, summary_path)}
    (args.output / "SHA256SUMS.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

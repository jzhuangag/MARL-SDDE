"""Outcome-development CPU smoke for state-dependent Pistonball supports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .pistonball_tail import (
    directional_influence_matrix,
    finite_horizon_return,
    initial_ball_index,
    predictive_tube_support,
    summarize_support,
    summarize_piston_influence,
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
    parser.add_argument("--n-pistons", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--horizons", type=int, nargs="+", default=[12, 24])
    parser.add_argument("--steps", type=float, nargs="+", default=[0.04, 0.08])
    parser.add_argument("--burn-ins", type=int, nargs="+", default=[0])
    parser.add_argument("--active-radii", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--action-noise-std", type=float, default=0.20)
    parser.add_argument("--policy-seed", type=int, default=913)
    parser.add_argument("--first-reset-seed", type=int, default=82001)
    args = parser.parse_args()

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from pettingzoo.butterfly.pistonball.pistonball import raw_env

    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.policy_seed)
    theta = rng.normal(scale=0.18, size=args.n_pistons)
    rows: list[dict[str, float | int]] = []
    matrices: list[dict[str, object]] = []
    env = raw_env(
        n_pistons=args.n_pistons,
        continuous=True,
        random_drop=True,
        random_rotate=False,
        max_cycles=max(args.burn_ins) + max(args.horizons),
        render_mode=None,
    )
    try:
        for seed_offset in range(args.seeds):
            reset_seed = args.first_reset_seed + seed_offset
            for burn_in in args.burn_ins:
                _, launch_x, launch_velocity_x = finite_horizon_return(
                    env, theta, reset_seed, 1, burn_in=burn_in
                )
                ball_index = initial_ball_index(launch_x, args.n_pistons)
                for horizon in args.horizons:
                    noise_rng = np.random.default_rng(reset_seed + 1009 * burn_in + 17 * horizon)
                    action_noise = noise_rng.normal(
                        scale=args.action_noise_std, size=(horizon, args.n_pistons)
                    )
                    for step in args.steps:
                        objective = lambda candidate, rs=reset_seed, hz=horizon, bi=burn_in, noise=action_noise: finite_horizon_return(
                            env, candidate, rs, hz, burn_in=bi, action_noise=noise
                        )[0]
                        influence = directional_influence_matrix(objective, theta, step)
                        matrices.append(
                            {
                                "reset_seed": reset_seed,
                                "launch_ball_x": launch_x,
                                "launch_ball_velocity_x": launch_velocity_x,
                                "ball_index": ball_index,
                                "burn_in": burn_in,
                                "horizon": horizon,
                                "step": step,
                                "influence": influence.tolist(),
                            }
                        )
                        for radius in args.active_radii:
                            result = summarize_piston_influence(
                                influence,
                                ball_index=ball_index,
                                active_radius=radius,
                                graph_seed=reset_seed + 31 * radius + horizon,
                            )
                            tube = predictive_tube_support(
                                args.n_pistons,
                                launch_x=launch_x,
                                launch_velocity_x=launch_velocity_x,
                                horizon=horizon,
                                radius=radius,
                            )
                            (
                                tube_fraction,
                                tube_cost_fraction,
                                tube_random_median,
                                tube_minus_random,
                            ) = summarize_support(
                                influence,
                                tube,
                                graph_seed=reset_seed + 97 * radius + 3 * horizon,
                            )
                            rows.append(
                                {
                                    "reset_seed": reset_seed,
                                    "launch_ball_x": launch_x,
                                    "launch_ball_velocity_x": launch_velocity_x,
                                    "burn_in": burn_in,
                                    "horizon": horizon,
                                    "step": step,
                                    **result.__dict__,
                                    "tube_edge_fraction": tube_fraction,
                                    "tube_edge_cost_fraction": tube_cost_fraction,
                                    "tube_random_median_fraction": tube_random_median,
                                    "tube_minus_random_median": tube_minus_random,
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
    best_by_matrix = []
    for reset_seed in sorted({int(row["reset_seed"]) for row in rows}):
        for burn_in in args.burn_ins:
            for horizon in args.horizons:
                for step in args.steps:
                    candidates = [
                        row
                        for row in rows
                        if row["reset_seed"] == reset_seed
                        and row["burn_in"] == burn_in
                        and row["horizon"] == horizon
                        and row["step"] == step
                    ]
                    best_by_matrix.append(max(candidates, key=lambda row: row["active_edge_fraction"]))
    summary = {
        "status": "development_only_not_confirmation",
        "environment": "PettingZoo pistonball_v6",
        "n_pistons": args.n_pistons,
        "seeds": args.seeds,
        "matrix_count": len(matrices),
        "rows": len(rows),
        "physical_chain_fraction_median": float(np.nanmedian([row["chain_edge_fraction"] for row in rows])),
        "best_active_fraction_median": float(
            np.nanmedian([row["active_edge_fraction"] for row in best_by_matrix])
        ),
        "best_active_cost_fraction_median": float(
            np.nanmedian([row["active_edge_cost_fraction"] for row in best_by_matrix])
        ),
        "best_active_minus_random_median": float(
            np.nanmedian([row["active_minus_random_median"] for row in best_by_matrix])
        ),
        "best_tube_fraction_median": float(
            np.nanmedian(
                [
                    max(
                        row["tube_edge_fraction"]
                        for row in rows
                        if row["reset_seed"] == matrix["reset_seed"]
                        and row["burn_in"] == matrix["burn_in"]
                        and row["horizon"] == matrix["horizon"]
                        and row["step"] == matrix["step"]
                    )
                    for matrix in matrices
                ]
            )
        ),
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

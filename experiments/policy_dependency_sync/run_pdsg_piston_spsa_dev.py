"""Development-only smoothed-Hessian dependency audit on Pistonball."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .pistonball_tail import (
    finite_horizon_return,
    initial_ball_index,
    predictive_tube_support,
    spsa_hessian_from_objectives,
    summarize_support,
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
    parser.add_argument("--burn-ins", type=int, nargs="+", default=[5, 15, 25])
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--steps", type=float, nargs="+", default=[0.03, 0.06])
    parser.add_argument("--policy-noise-std", type=float, default=0.30)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--first-sample-seed", type=int, default=85001)
    parser.add_argument("--policy-seed", type=int, default=913)
    parser.add_argument("--reset-seed", type=int, default=17)
    args = parser.parse_args()

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from pettingzoo.butterfly.pistonball.pistonball import raw_env

    args.output.mkdir(parents=True, exist_ok=False)
    theta = np.random.default_rng(args.policy_seed).normal(scale=0.18, size=args.n_pistons)
    sample_seeds = list(range(args.first_sample_seed, args.first_sample_seed + args.samples))
    env = raw_env(
        n_pistons=args.n_pistons,
        continuous=True,
        random_drop=False,
        random_rotate=False,
        max_cycles=max(args.burn_ins) + args.horizon,
        render_mode=None,
    )
    rows: list[dict[str, float | int]] = []
    matrices: list[dict[str, object]] = []
    try:
        for burn_in in args.burn_ins:
            _, launch_x, launch_velocity_x = finite_horizon_return(
                env, theta, args.reset_seed, 1, burn_in=burn_in
            )
            ball_index = initial_ball_index(launch_x, args.n_pistons)
            for step in args.steps:
                def builder(sample_seed: int, bi=burn_in):
                    rng = np.random.default_rng(sample_seed + 7919)
                    action_noise = rng.normal(
                        scale=args.policy_noise_std,
                        size=(args.horizon, args.n_pistons),
                    )
                    return lambda candidate: finite_horizon_return(
                        env,
                        candidate,
                        args.reset_seed,
                        args.horizon,
                        burn_in=bi,
                        action_noise=action_noise,
                    )[0]

                signed, standard_error = spsa_hessian_from_objectives(
                    theta, step=step, sample_seeds=sample_seeds, objective_builder=builder
                )
                influence = np.abs(signed)
                matrices.append(
                    {
                        "burn_in": burn_in,
                        "launch_ball_x": launch_x,
                        "launch_ball_velocity_x": launch_velocity_x,
                        "ball_index": ball_index,
                        "horizon": args.horizon,
                        "step": step,
                        "signed_hessian": signed.tolist(),
                        "standard_error": standard_error.tolist(),
                    }
                )
                for radius in (0, 1, 2):
                    support = predictive_tube_support(
                        args.n_pistons,
                        launch_x,
                        launch_velocity_x,
                        args.horizon,
                        radius,
                    )
                    retained, cost, random_median, gap = summarize_support(
                        influence,
                        support,
                        graph_seed=args.reset_seed + 103 * burn_in + radius,
                    )
                    mask = np.zeros_like(influence, dtype=bool)
                    for owner, donors in support.items():
                        for donor in donors:
                            mask[owner, donor] = True
                    rows.append(
                        {
                            "burn_in": burn_in,
                            "launch_ball_x": launch_x,
                            "launch_ball_velocity_x": launch_velocity_x,
                            "ball_index": ball_index,
                            "horizon": args.horizon,
                            "step": step,
                            "radius": radius,
                            "retained_fraction": retained,
                            "edge_cost_fraction": cost,
                            "random_median_fraction": random_median,
                            "retained_minus_random": gap,
                            "selected_median_signal_to_se": float(
                                np.median(
                                    np.abs(signed[mask])
                                    / np.maximum(standard_error[mask], 1e-12)
                                )
                            )
                            if np.any(mask)
                            else float("nan"),
                        }
                    )
    finally:
        env.close()

    csv_path = args.output / "summary_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    matrix_path = args.output / "spsa_matrices.json"
    matrix_path.write_text(json.dumps(matrices, indent=2), encoding="utf-8")
    radius_one = [row for row in rows if row["radius"] == 1]
    summary = {
        "status": "development_only_not_confirmation",
        "environment": "PettingZoo pistonball_v6",
        "estimator": "paired bilinear Rademacher smoothed Hessian",
        "n_pistons": args.n_pistons,
        "matrix_count": len(matrices),
        "samples_per_matrix": args.samples,
        "radius_one_retained_median": float(
            np.median([row["retained_fraction"] for row in radius_one])
        ),
        "radius_one_cost_median": float(
            np.median([row["edge_cost_fraction"] for row in radius_one])
        ),
        "radius_one_random_gap_median": float(
            np.median([row["retained_minus_random"] for row in radius_one])
        ),
        "radius_one_signal_to_se_median": float(
            np.median([row["selected_median_signal_to_se"] for row in radius_one])
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

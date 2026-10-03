"""Small development-only calibration smoke for Pistonball trajectory tubes."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from .conformal_cone import split_conformal_radius
from .pistonball_tail import predictive_tube_support, trajectory_tube_residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--first-seed", type=int, default=86001)
    args = parser.parse_args()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from pettingzoo.butterfly.pistonball.pistonball import raw_env

    args.output.mkdir(parents=True, exist_ok=False)
    n_agents = 20
    policy_noise_std = 0.30
    policy_seeds = (913, 1201, 1777)
    burn_ins = (5, 15, 25)
    horizons = (4, 6, 8)
    rows = []
    env = raw_env(
        n_pistons=n_agents,
        continuous=True,
        random_drop=False,
        random_rotate=False,
        max_cycles=max(burn_ins) + max(horizons),
        render_mode=None,
    )
    try:
        for policy_seed in policy_seeds:
            theta = np.random.default_rng(policy_seed).normal(scale=0.18, size=n_agents)
            for burn_in in burn_ins:
                for horizon in horizons:
                    residuals = []
                    launch_x = launch_velocity_x = 0.0
                    for offset in range(args.samples):
                        trajectory_seed = args.first_seed + offset
                        noise = np.random.default_rng(trajectory_seed).normal(
                            scale=policy_noise_std, size=(horizon, n_agents)
                        )
                        residual, launch_x, launch_velocity_x, _ = trajectory_tube_residual(
                            env,
                            theta,
                            reset_seed=17,
                            burn_in=burn_in,
                            horizon=horizon,
                            action_noise=noise,
                        )
                        residuals.append(residual)
                    # Development uses a descriptive 0.8 quantile because 16
                    # points cannot certify the later 0.1 miscoverage target.
                    radius = float(np.quantile(residuals, 0.8, method="higher"))
                    shell = int(np.ceil(radius + 2.0))
                    support = predictive_tube_support(
                        n_agents,
                        launch_x,
                        launch_velocity_x,
                        horizon,
                        shell,
                    )
                    edges = sum(len(donors) for donors in support.values())
                    rows.append(
                        {
                            "policy_seed": policy_seed,
                            "burn_in": burn_in,
                            "horizon": horizon,
                            "launch_x": launch_x,
                            "launch_velocity_x": launch_velocity_x,
                            "residual_q80": radius,
                            "contact_expanded_shell": shell,
                            "edge_cost_fraction": edges / (n_agents * (n_agents - 1)),
                            "conformal_alpha_0_1_finite_with_16": np.isfinite(
                                split_conformal_radius(residuals, 0.1)
                            ),
                        }
                    )
    finally:
        env.close()
    csv_path = args.output / "development_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "development_only_not_confirmation",
        "rows": len(rows),
        "samples_per_row": args.samples,
        "residual_q80_median": float(np.median([row["residual_q80"] for row in rows])),
        "residual_q80_p90": float(np.quantile([row["residual_q80"] for row in rows], 0.9)),
        "edge_cost_fraction_median": float(
            np.median([row["edge_cost_fraction"] for row in rows])
        ),
        "edge_cost_fraction_p90": float(
            np.quantile([row["edge_cost_fraction"] for row in rows], 0.9)
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

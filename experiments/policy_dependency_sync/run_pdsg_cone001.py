"""Frozen PDSG-CONE-001 conformal trajectory-cone confirmation runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from .conformal_cone import (
    gaussian_trajectory_kl_bound,
    shifted_escape_probability,
    split_conformal_radius,
    total_variation_from_kl,
)
from .pistonball_tail import predictive_tube_support, trajectory_tube_residual


EXPECTED_MANIFEST_SHA256 = "B46BBCE028B7416DAF065F8F8362271CB7CD5EC8673E98E5EA998435736F54BA"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def edge_set(support: dict[int, tuple[int, ...]]) -> frozenset[tuple[int, int]]:
    return frozenset((owner, donor) for owner, donors in support.items() for donor in donors)


def jaccard(left: frozenset[tuple[int, int]], right: frozenset[tuple[int, int]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def evaluate_gates(
    cells: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, bool]:
    grid = manifest["grid"]
    seeds = manifest["seeds"]
    expected_cells = (
        len(manifest["reference_policy"]["policy_seeds"])
        * len(grid["burn_in_cycles"])
        * len(grid["rollout_horizons"])
    )
    expected_trajectories = expected_cells * (
        seeds["calibration_count"] + seeds["holdout_count"]
    )
    calibration = set(
        range(seeds["calibration_start"], seeds["calibration_start"] + seeds["calibration_count"])
    )
    holdout = set(range(seeds["holdout_start"], seeds["holdout_start"] + seeds["holdout_count"]))
    if len(cells) != expected_cells or len(trajectory_rows) != expected_trajectories:
        return {f"C{index}": False for index in range(1, 12)}

    all_finite = all(
        math.isfinite(float(row["residual"])) for row in trajectory_rows
    ) and all(
        math.isfinite(float(cell[key]))
        for cell in cells
        for key in ("conformal_radius", "holdout_coverage", "edge_cost_fraction")
    )
    holdout_rows = [row for row in trajectory_rows if row["population"] == "holdout"]
    pooled_coverage = float(np.mean([bool(row["covered"]) for row in holdout_rows]))
    minimum_coverage = min(float(cell["holdout_coverage"]) for cell in cells)
    horizon_four = [cell for cell in cells if int(cell["horizon"]) == 4]
    horizon_four_costs = [float(cell["edge_cost_fraction"]) for cell in horizon_four]

    by_path: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    for cell in cells:
        by_path.setdefault((int(cell["policy_seed"]), int(cell["burn_in"])), {})[
            int(cell["horizon"])
        ] = cell
    nondecreasing = sum(
        float(path[4]["edge_cost_fraction"])
        <= float(path[6]["edge_cost_fraction"])
        <= float(path[8]["edge_cost_fraction"])
        for path in by_path.values()
    )
    horizon_differences = [
        float(path[8]["edge_cost_fraction"]) - float(path[4]["edge_cost_fraction"])
        for path in by_path.values()
    ]

    by_state: dict[tuple[int, int], dict[int, frozenset[tuple[int, int]]]] = {}
    for cell in cells:
        support = frozenset(tuple(edge) for edge in cell["support_edges"])
        by_state.setdefault((int(cell["policy_seed"]), int(cell["horizon"])), {})[
            int(cell["burn_in"])
        ] = support
    horizon_four_jaccards = [
        jaccard(states[5], states[25])
        for (policy_seed, horizon), states in by_state.items()
        if horizon == 4
    ]

    development_seeds = {81001, 82001, 84001, 85001, 86001}
    source = inspect.getsource(trajectory_tube_residual)
    return {
        "C1": len(cells) == expected_cells
        and len(trajectory_rows) == expected_trajectories
        and not (calibration & holdout)
        and all_finite,
        "C2": ".rewards" not in source and "['reward']" not in source and '["reward"]' not in source,
        "C3": all(math.isfinite(float(cell["conformal_radius"])) for cell in cells),
        "C4": pooled_coverage >= 0.88,
        "C5": minimum_coverage >= 0.80,
        "C6": float(np.median(horizon_four_costs)) <= 0.45
        and quantile(horizon_four_costs, 0.9) <= 0.65,
        "C7": float(np.median(horizon_differences)) >= 0.10 and nondecreasing >= 7,
        "C8": float(np.median(horizon_four_jaccards)) <= 0.40,
        "C9": all(float(cell["shifted_escape_upper"]) <= 0.15 for cell in cells),
        "C10": all(
            int(cell["edge_count"]) == len(cell["support_edges"])
            and int(cell["edge_count"]) <= 380
            and np.isclose(
                float(cell["edge_cost_fraction"]), int(cell["edge_count"]) / 380.0
            )
            for cell in cells
        ),
        "C11": not ((calibration | holdout) & development_seeds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("manifest hash does not match the frozen preregistration")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import gymnasium
    import pettingzoo
    import pygame
    import pymunk
    from pettingzoo.butterfly.pistonball.pistonball import raw_env

    environment = manifest["environment"]
    policy = manifest["reference_policy"]
    grid = manifest["grid"]
    seeds = manifest["seeds"]
    n_agents = int(environment["n_pistons"])
    calibration_seeds = list(
        range(seeds["calibration_start"], seeds["calibration_start"] + seeds["calibration_count"])
    )
    holdout_seeds = list(
        range(seeds["holdout_start"], seeds["holdout_start"] + seeds["holdout_count"])
    )

    env = raw_env(
        n_pistons=n_agents,
        continuous=environment["continuous"],
        random_drop=environment["random_drop"],
        random_rotate=environment["random_rotate"],
        max_cycles=max(grid["burn_in_cycles"]) + max(grid["rollout_horizons"]),
        render_mode=None,
    )
    trajectory_rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    try:
        for policy_seed in policy["policy_seeds"]:
            theta = np.random.default_rng(policy_seed).normal(
                scale=policy["parameter_scale"], size=n_agents
            )
            for burn_in in grid["burn_in_cycles"]:
                for horizon in grid["rollout_horizons"]:
                    population_residuals: dict[str, list[float]] = {
                        "calibration": [],
                        "holdout": [],
                    }
                    launch_x = launch_velocity_x = 0.0
                    for population, selected_seeds in (
                        ("calibration", calibration_seeds),
                        ("holdout", holdout_seeds),
                    ):
                        for trajectory_seed in selected_seeds:
                            action_noise = np.random.default_rng(trajectory_seed).normal(
                                scale=policy["policy_noise_std"], size=(horizon, n_agents)
                            )
                            residual, launch_x, launch_velocity_x, _ = trajectory_tube_residual(
                                env,
                                theta,
                                reset_seed=environment["reset_seed"],
                                burn_in=burn_in,
                                horizon=horizon,
                                action_noise=action_noise,
                            )
                            population_residuals[population].append(residual)
                            trajectory_rows.append(
                                {
                                    "policy_seed": policy_seed,
                                    "burn_in": burn_in,
                                    "horizon": horizon,
                                    "population": population,
                                    "trajectory_seed": trajectory_seed,
                                    "residual": residual,
                                    "covered": "",
                                }
                            )
                    radius = split_conformal_radius(
                        population_residuals["calibration"], grid["calibration_miscoverage"]
                    )
                    for row in trajectory_rows:
                        if (
                            row["population"] == "holdout"
                            and row["policy_seed"] == policy_seed
                            and row["burn_in"] == burn_in
                            and row["horizon"] == horizon
                        ):
                            row["covered"] = bool(float(row["residual"]) <= radius)
                    shell = int(math.ceil(radius + grid["contact_padding_piston_widths"]))
                    support = predictive_tube_support(
                        n_agents,
                        launch_x,
                        launch_velocity_x,
                        horizon,
                        shell,
                    )
                    edges = sorted(edge_set(support))
                    holdout_coverage = float(
                        np.mean(np.asarray(population_residuals["holdout"]) <= radius)
                    )
                    kl = gaussian_trajectory_kl_bound(
                        horizon,
                        grid["declared_joint_mean_l2_policy_shift"],
                        policy["policy_noise_std"],
                    )
                    tv = total_variation_from_kl(kl)
                    cells.append(
                        {
                            "policy_seed": policy_seed,
                            "burn_in": burn_in,
                            "horizon": horizon,
                            "launch_x": launch_x,
                            "launch_velocity_x": launch_velocity_x,
                            "conformal_radius": radius,
                            "contact_expanded_shell": shell,
                            "holdout_coverage": holdout_coverage,
                            "trajectory_kl_upper": kl,
                            "total_variation_upper": tv,
                            "shifted_escape_upper": shifted_escape_probability(
                                grid["calibration_miscoverage"], tv
                            ),
                            "edge_count": len(edges),
                            "edge_cost_fraction": len(edges) / (n_agents * (n_agents - 1)),
                            "support_edges": [list(edge) for edge in edges],
                        }
                    )
    finally:
        env.close()

    gates = evaluate_gates(cells, trajectory_rows, manifest)
    holdout_rows = [row for row in trajectory_rows if row["population"] == "holdout"]
    horizon_costs = {
        str(horizon): [
            float(cell["edge_cost_fraction"])
            for cell in cells
            if int(cell["horizon"]) == int(horizon)
        ]
        for horizon in grid["rollout_horizons"]
    }
    summary = {
        "experiment_id": manifest["experiment_id"],
        "status": "pass" if all(gates.values()) else "fail",
        "controller_preregistration_design_authorized": all(gates.values()),
        "gpu_authorized": False,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "cells": len(cells),
        "trajectories": len(trajectory_rows),
        "pooled_holdout_coverage": float(np.mean([bool(row["covered"]) for row in holdout_rows])),
        "minimum_cell_coverage": min(float(cell["holdout_coverage"]) for cell in cells),
        "edge_cost_by_horizon": {
            horizon: {
                "median": float(np.median(values)),
                "p90": quantile(values, 0.9),
                "maximum": max(values),
            }
            for horizon, values in horizon_costs.items()
        },
        "gates": gates,
        "runtime_recorded_separately": True,
        "environment_versions": {
            "python": platform.python_version(),
            "gymnasium": gymnasium.__version__,
            "pettingzoo": pettingzoo.__version__,
            "pygame_ce": pygame.version.ver,
            "pymunk": pymunk.version,
            "numpy": np.__version__,
        },
    }

    trajectory_path = args.output / "trajectories.csv"
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)
    cell_path = args.output / "cells.json"
    cell_path.write_text(json.dumps(cells, indent=2, sort_keys=True), encoding="utf-8")
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    execution_path = args.output / "execution.json"
    execution_path.write_text(
        json.dumps({"runtime_seconds": time.time() - started}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    hashes = {path.name: sha256(path) for path in (trajectory_path, cell_path, summary_path)}
    (args.output / "SHA256SUMS.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

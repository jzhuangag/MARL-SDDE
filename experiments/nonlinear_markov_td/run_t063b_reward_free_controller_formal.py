"""Run the prospective T-063B collision-gate correction experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import time
from typing import Any

from scipy.stats import beta
import torch

from experiments.nonlinear_markov_td.run_t060a_minatar_fixed_q_pilot import (
    canonical_config_hash,
)
from experiments.nonlinear_markov_td.run_t061a_reward_free_controller_pilot import (
    analyze as legacy_analyze,
    file_sha256,
    load_config as load_pilot_config,
    load_references,
    write_csv,
)
from experiments.nonlinear_markov_td.run_t063a_reward_free_controller_formal import (
    run_game_seed,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "docs" / "t063b_reward_free_controller_formal_preregistration.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "nonlinear_markov_td" / "results" / "t063b_reward_free_controller_formal"


def load_config(path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if canonical_config_hash(spec) != spec["configuration_sha256"]:
        raise RuntimeError("T-063B configuration hash mismatch")
    base_path = ROOT / "docs" / "t063a_reward_free_controller_formal_preregistration.json"
    if file_sha256(base_path) != spec["base_preregistration"]["sha256"]:
        raise RuntimeError("T-063A base preregistration hash mismatch")
    with base_path.open("r", encoding="utf-8") as handle:
        base = json.load(handle)
    if canonical_config_hash(base) != base["configuration_sha256"]:
        raise RuntimeError("T-063A base configuration hash mismatch")
    source_path = ROOT / base["source_pilot_config"]["path"]
    if file_sha256(source_path) != base["source_pilot_config"]["sha256"]:
        raise RuntimeError("source pilot configuration hash mismatch")
    config = load_pilot_config(source_path)
    registry = spec["formal_seed_registry"]
    start = int(registry["inclusive_start"])
    stop = int(registry["inclusive_end"])
    seeds = list(range(start, stop + 1))
    if len(seeds) != int(registry["count"]):
        raise RuntimeError("T-063B seed registry count mismatch")
    config.update(
        {
            "experiment_id": spec["experiment_id"],
            "configuration_sha256": spec["configuration_sha256"],
            "pilot_seeds": seeds,
            "gates": dict(base["point_gates"]),
            "formal_analysis": dict(base["formal_analysis"]),
            "formal_spec": {"base": base, "t063b": spec},
            "collision_gate": dict(spec["collision_gate"]),
        }
    )
    return config


def aggregate_collision_gate(config: dict[str, Any], endpoints: list[dict[str, Any]]) -> dict[str, Any]:
    gate = config["collision_gate"]
    blocks = int(gate["probe_blocks_per_seed_task"])
    unique: dict[tuple[int, str], int] = {}
    for row in endpoints:
        if float(row["rho"]) == 0.0:
            unique[(int(row["master_seed"]), str(row["game"]))] = int(row["match_count"])
    total_matches = int(sum(unique.values()))
    total_trials = int(len(unique) * blocks)
    alpha = float(gate["alpha"])
    c_max = float(gate["independent_path_probability_upper_bound"])
    if total_trials == 0:
        raise ValueError("no rho=0 probe blocks")
    upper = 1.0 if total_matches == total_trials else float(
        beta.ppf(1.0 - alpha, total_matches + 1, total_trials - total_matches)
    )
    return {
        "kind": gate["kind"],
        "alpha": alpha,
        "independent_path_probability_upper_bound": c_max,
        "seed_task_blocks": len(unique),
        "total_matches": total_matches,
        "total_trials": total_trials,
        "aggregate_rate": total_matches / total_trials,
        "one_sided_upper_probability": upper,
        "pass": upper <= c_max,
        "blockwise_maximum_rate": max(unique.values(), default=0) / blocks,
    }


def point_analyze(config: dict[str, Any], endpoints: list[dict[str, Any]]) -> dict[str, Any]:
    summary = legacy_analyze(config, endpoints)
    collision = aggregate_collision_gate(config, endpoints)
    summary["t063b_collision_gate"] = collision
    summary["gates"]["P10_independent_collision"] = bool(collision["pass"])
    summary["pre_reproduction_pilot_gate_pass"] = all(summary["gates"].values())
    return summary


def validate(config: dict[str, Any]) -> dict[str, Any]:
    spec = config["formal_spec"]["t063b"]
    seeds = config["pilot_seeds"]
    return {
        "configuration_sha256": spec["configuration_sha256"],
        "seed_count": len(seeds),
        "seed_start": min(seeds),
        "seed_end": max(seeds),
        "collision_gate": config["collision_gate"],
        "authorized": bool(spec["authorization"]["formal_run_authorized_after_static_audit"]),
    }


def run(config: dict[str, Any], output: Path, *, workers: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if workers < 1:
        raise ValueError("workers must be positive")
    torch.use_deterministic_algorithms(True)
    jobs = [
        (config, game, int(master_seed))
        for game in config["tasks"]
        for master_seed in config["pilot_seeds"]
    ]
    started = time.perf_counter()
    if workers == 1:
        chunks = [run_game_seed(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            chunks = list(executor.map(run_game_seed, jobs, chunksize=1))
    endpoints = [row for chunk in chunks for row in chunk]
    summary = point_analyze(config, endpoints)
    cells = summary.pop("cell_rows")
    runtime = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "endpoints.csv", endpoints)
    write_csv(output / "cells.csv", cells)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {**summary, "runtime_seconds": runtime, "workers": workers}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    result = validate(config) if arguments.mode == "validate" else run(config, arguments.output, workers=arguments.workers)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

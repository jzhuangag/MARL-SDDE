"""Run the preregistered T-063A formal reward-free MinAtar experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
from typing import Any

import torch

from experiments.nonlinear_markov_td.run_t060a_minatar_fixed_q_pilot import (
    canonical_config_hash,
    learning_bank,
)
from experiments.nonlinear_markov_td.run_t061a_reward_free_controller_pilot import (
    analyze,
    estimate as pilot_estimate,
    file_sha256,
    load_config as load_pilot_config,
    load_references,
    run_endpoint,
    write_csv,
)
from experiments.nonlinear_markov_td.t061_reward_free_fingerprint import (
    probe_match_count,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "docs" / "t063a_reward_free_controller_formal_preregistration.json"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "nonlinear_markov_td"
    / "results"
    / "t063a_reward_free_controller_formal"
)


def load_config(path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if canonical_config_hash(spec) != spec["configuration_sha256"]:
        raise RuntimeError("formal configuration hash mismatch")
    source_path = ROOT / spec["source_pilot_config"]["path"]
    if file_sha256(source_path) != spec["source_pilot_config"]["sha256"]:
        raise RuntimeError("source pilot configuration hash mismatch")
    config = load_pilot_config(source_path)
    registry = spec["formal_seed_registry"]
    start = int(registry["inclusive_start"])
    stop = int(registry["inclusive_end"])
    seeds = list(range(start, stop + 1))
    if len(seeds) != int(registry["count"]):
        raise RuntimeError("formal seed registry count mismatch")
    config.update(
        {
            "experiment_id": spec["experiment_id"],
            "configuration_sha256": spec["configuration_sha256"],
            "pilot_seeds": seeds,
            "gates": spec["point_gates"],
            "formal_analysis": spec["formal_analysis"],
            "formal_spec": spec,
        }
    )
    return config


def validate(config: dict[str, Any]) -> dict[str, Any]:
    spec = config["formal_spec"]
    seeds = config["pilot_seeds"]
    excluded = {
        seed
        for interval in spec["seed_isolation"]["excluded_intervals"]
        for seed in range(int(interval[0]), int(interval[1]) + 1)
    }
    source_hashes = {
        label: file_sha256(ROOT / relative)
        for label, relative in spec["software_paths"].items()
    }
    return {
        "configuration_hash_matches": canonical_config_hash(spec)
        == spec["configuration_sha256"],
        "seed_count_matches": len(seeds) == spec["formal_seed_registry"]["count"],
        "seeds_are_contiguous": seeds == list(range(seeds[0], seeds[-1] + 1)),
        "seed_isolation": excluded.isdisjoint(seeds),
        "source_hashes_match": source_hashes == spec["software_sha256"],
        "local_cpu_only": not spec["authorization"]["gpu"]
        and not spec["authorization"]["hpc4"],
    }


def estimate(config: dict[str, Any], *, workers: int) -> dict[str, Any]:
    result = pilot_estimate(config)
    pilot_seconds = float(config["formal_spec"]["runtime_model"]["T061A_wall_seconds"])
    pilot_seeds = int(config["formal_spec"]["runtime_model"]["T061A_seed_clusters"])
    ideal = pilot_seconds * len(config["pilot_seeds"]) / pilot_seeds / workers
    result.update(
        {
            "workers": workers,
            "estimated_primary_wall_hours_at_ideal_scaling": ideal / 3600.0,
            "estimated_primary_plus_reproduction_wall_hours_at_ideal_scaling": 2.0
            * ideal
            / 3600.0,
            "recommended_device": "local CPU",
        }
    )
    return result


def run_game_seed(payload: tuple[dict[str, Any], str, int]) -> list[dict[str, Any]]:
    config, game, master_seed = payload
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    references, diagnostics = load_references(config)
    bank, encoder_hash = learning_bank(config, game=game, master_seed=master_seed)
    if encoder_hash != diagnostics[game]["encoder_sha256"]:
        raise RuntimeError("encoder fingerprint changed")
    rows: list[dict[str, Any]] = []
    for rho in config["grid"]["correlations"]:
        matches = probe_match_count(
            game=game,
            rho=float(rho),
            blocks=int(config["probe"]["blocks"]),
            length=int(config["probe"]["length"]),
            master_seed=master_seed,
            sticky_action_probability=float(
                config["environment"]["sticky_action_probability"]
            ),
            difficulty_ramping=bool(config["environment"]["difficulty_ramping"]),
        )
        for overhead in config["grid"]["overheads"]:
            for delay in config["grid"]["delays"]:
                rows.append(
                    run_endpoint(
                        config=config,
                        reference=references[game],
                        diagnostics=diagnostics[game],
                        bank=bank,
                        game=game,
                        master_seed=master_seed,
                        rho=float(rho),
                        match_count=int(matches),
                        overhead=int(overhead),
                        delay=int(delay),
                    )
                )
    return rows


def run(config: dict[str, Any], output: Path, *, workers: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if workers < 1:
        raise ValueError("workers must be positive")
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
    summary = analyze(config, endpoints)
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
    parser.add_argument("mode", choices=("validate", "estimate", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.mode == "validate":
        result = validate(config)
    elif arguments.mode == "estimate":
        result = estimate(config, workers=arguments.workers)
    else:
        result = run(config, arguments.output, workers=arguments.workers)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

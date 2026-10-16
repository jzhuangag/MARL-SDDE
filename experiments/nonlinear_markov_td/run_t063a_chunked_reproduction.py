"""Chunked, byte-reproducible T-063A clean reproduction runner.

The frozen T-063A worker and analyzer are unchanged.  This orchestration layer
restarts the process pool after a fixed number of game-seed jobs to avoid
long-lived Windows worker failures; jobs and rows retain the frozen order.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

import torch

from experiments.nonlinear_markov_td.run_t063a_reward_free_controller_formal import (
    load_config,
    run_game_seed,
)
from experiments.nonlinear_markov_td.run_t061a_reward_free_controller_pilot import (
    analyze,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "experiments" / "nonlinear_markov_td" / "results" / "t063a_reward_free_controller_formal_reproduction"


def run(config: dict[str, Any], output: Path, *, workers: int, jobs_per_chunk: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if workers < 1 or jobs_per_chunk < 1:
        raise ValueError("workers and jobs_per_chunk must be positive")
    torch.use_deterministic_algorithms(True)
    jobs = [
        (config, game, int(master_seed))
        for game in config["tasks"]
        for master_seed in config["pilot_seeds"]
    ]
    endpoints: list[dict[str, Any]] = []
    for start in range(0, len(jobs), jobs_per_chunk):
        batch = jobs[start : start + jobs_per_chunk]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            chunks = list(executor.map(run_game_seed, batch, chunksize=1))
        endpoints.extend(row for chunk in chunks for row in chunk)
    summary = analyze(config, endpoints)
    cells = summary.pop("cell_rows")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "endpoints.csv", endpoints)
    write_csv(output / "cells.csv", cells)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--jobs-per-chunk", type=int, default=24)
    arguments = parser.parse_args()
    result = run(
        load_config(),
        arguments.output,
        workers=arguments.workers,
        jobs_per_chunk=arguments.jobs_per_chunk,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

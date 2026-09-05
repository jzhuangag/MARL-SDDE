"""Run the frozen T-083A formal CPU confirmation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from experiments.dependence_delay_linear.run_t071a_sampled_observable_graph_pilot import (
    load_config as load_t071_config,
    scenarios,
)
from experiments.dependence_delay_linear.run_t081_end_block_controller_calibration import (
    T080_CELLS,
    analyze as calibration_analyze,
    continuous_static_weights,
    run_endpoint,
    write_csv,
)
from experiments.dependence_delay_linear.run_t082a_independent_end_block_pilot import (
    load_config as load_pilot_config,
)
from experiments.dependence_delay_linear.t081_end_block_primal_dual_controller import (
    __file__ as controller_file,
)
from experiments.reproducible_artifacts import (
    write_execution_metadata,
    write_scientific_summary,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs/t083a_formal_end_block_confirmation.json"
DEFAULT_OUTPUT = ROOT / "experiments/dependence_delay_linear/results/t083a_formal_end_block_confirmation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def formal_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    cells = scenarios(load_t071_config())
    return [{**cell, "seed": int(seed)} for cell in cells for seed in config["formal_seeds"]]


def validate(config: dict[str, Any]) -> dict[str, Any]:
    if sha256(T080_CELLS) != config["frozen_sources"]["t080_cells_sha256"]:
        raise ValueError("T-080 source hash mismatch")
    if sha256(Path(controller_file)) != config["frozen_sources"]["controller_sha256"]:
        raise ValueError("controller hash mismatch")
    seeds = config["formal_seeds"]
    if len(seeds) != 128 or len(set(seeds)) != 128:
        raise ValueError("formal seed registry mismatch")
    used = set(load_t071_config()["pilot_seeds"]) | set(load_pilot_config()["pilot_seeds"])
    if used.intersection(seeds):
        raise ValueError("formal seed overlap")
    if any(config["frozen_sources"][key] is not False for key in (
        "controller_changed_after_pilot", "primary_population_changed_after_pilot",
        "gates_changed_after_pilot",
    )):
        raise ValueError("post-pilot change declaration mismatch")
    rows = formal_rows(config)
    primary = {row["cell_id"] for row in rows if (
        row["schedule_family"] in {"single_switch", "alternating"}
        and row["target_scale"] in {0.3, 0.6}
        and row["temporal_correlation"] == 0.0
    )}
    observed = {
        "cells": len({row["cell_id"] for row in rows}),
        "primary_cells": len(primary),
        "formal_seeds": len(seeds),
        "endpoints": len(rows),
    }
    for key, value in observed.items():
        if value != config["workload"][key]:
            raise ValueError(f"workload mismatch for {key}")
    return {"experiment_id": config["experiment_id"], **observed,
            "prior_seed_overlap": 0, "scientific_outcome_created": False}


def _payload(payload):
    return run_endpoint(payload)


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    validate(config)
    static = continuous_static_weights()
    rows = formal_rows(config)
    started = time.perf_counter()
    payloads = ((config, row, static[row["cell_id"]]) for row in rows)
    with ProcessPoolExecutor(max_workers=config["workload"]["workers"]) as pool:
        endpoints = list(pool.map(_payload, payloads, chunksize=config["workload"]["chunksize"]))
    runtime = time.perf_counter() - started
    calibration_summary, cells = calibration_analyze(config, endpoints)
    gates = {f"F{i}": calibration_summary["gates"][f"C{i}"] for i in range(1, 12)}
    gates["F12"] = runtime <= 60.0 * config["workload"]["hard_timeout_minutes"]
    gates["F13"] = False
    gates["F14"] = False
    summary = {
        "experiment_id": config["experiment_id"],
        "classification": config["classification"],
        "metrics": calibration_summary["metrics"],
        "gates": gates,
        "all_pre_reproduction_gates_pass": all(gates[f"F{i}"] for i in range(1, 13)),
        "authorization": config["authorization"],
    }
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "endpoints.csv", endpoints)
    write_csv(output / "cells.csv", cells)
    write_scientific_summary(output / "summary.json", summary)
    write_execution_metadata(
        output / "execution.json", runtime_seconds=runtime,
        workers=config["workload"]["workers"], chunksize=config["workload"]["chunksize"],
    )
    return {**summary, "runtime_seconds": runtime}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_config(args.config)
    result = validate(config) if args.command == "validate" else run(config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

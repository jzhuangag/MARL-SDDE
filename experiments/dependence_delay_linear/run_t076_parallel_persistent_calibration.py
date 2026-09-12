"""Deterministically parallelize the scientifically frozen T-074 calibration."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from experiments.dependence_delay_linear.run_t072_dual_use_architecture_calibration import (
    source_rows, write_csv,
)
from experiments.dependence_delay_linear.run_t074_persistent_certificate_architecture_calibration import (
    analyze as analyze_t074,
    load_config as load_t074_config,
    run_endpoint as run_t074_endpoint,
    validate as validate_t074,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs/t076_parallel_persistent_calibration.json"
DEFAULT_OUTPUT = ROOT / "experiments/dependence_delay_linear/results/t076_parallel_persistent_calibration"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / config["scientific_config"]["path"]
    if sha256(path) != config["scientific_config"]["sha256"]:
        raise ValueError("frozen T-074 scientific config hash mismatch")
    return load_t074_config(path)


def validate(config: dict[str, Any]) -> dict[str, Any]:
    execution = config["execution"]
    if execution["workers"] != 4 or execution["ordered_map"] is not True:
        raise ValueError("parallel execution contract mismatch")
    if execution["output_is_created_only_after_all_endpoints_complete"] is not True:
        raise ValueError("partial output is forbidden")
    if (os.cpu_count() or 0) < execution["minimum_logical_processors"]:
        raise ValueError("insufficient local logical processors")
    frozen = validate_t074(scientific_config(config))
    return {"experiment_id": config["experiment_id"], "workers": 4,
            "ordered_endpoints": frozen["endpoints"], "cells": frozen["cells"],
            "scientific_config_unchanged": True}


def _run_payload(payload: tuple[dict[str, Any], dict[str, str]]) -> dict[str, Any]:
    config, row = payload
    return run_t074_endpoint(config, row)


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    validate(config)
    frozen = scientific_config(config)
    rows = source_rows()
    execution = config["execution"]
    started = time.perf_counter()
    payloads = ((frozen, row) for row in rows)
    with ProcessPoolExecutor(max_workers=execution["workers"]) as pool:
        endpoints = list(pool.map(_run_payload, payloads, chunksize=execution["chunksize"]))
    runtime = time.perf_counter() - started
    summary, cells = analyze_t074(frozen, endpoints)
    compute_gates = {
        "E1": len(summary["descriptive_criteria"]) == 11,
        "E2": len(endpoints) == 13_824 and len(cells) == 432,
        "E3": runtime <= 60.0 * execution["hard_timeout_minutes"],
        "E4": False,
        "E5": False,
    }
    result = {**summary, "execution_experiment_id": config["experiment_id"],
              "runtime_seconds": runtime, "compute_gates": compute_gates,
              "all_pre_reproduction_compute_gates_pass": all(
                  compute_gates[key] for key in ("E1", "E2", "E3"))}
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "endpoints.csv", endpoints)
    write_csv(output / "cells.csv", cells)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


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

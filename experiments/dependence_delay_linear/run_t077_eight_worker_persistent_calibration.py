"""Eight-worker ordered execution of the frozen T-074 calibration."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time
from typing import Any

from experiments.dependence_delay_linear.run_t072_dual_use_architecture_calibration import (
    source_rows, write_csv,
)
from experiments.dependence_delay_linear.run_t074_persistent_certificate_architecture_calibration import (
    analyze as analyze_t074, run_endpoint as run_t074_endpoint,
    validate as validate_t074,
)
from experiments.dependence_delay_linear.run_t076_parallel_persistent_calibration import (
    ROOT, scientific_config,
)


DEFAULT_CONFIG = ROOT / "docs/t077_eight_worker_persistent_calibration.json"
DEFAULT_OUTPUT = ROOT / "experiments/dependence_delay_linear/results/t077_eight_worker_persistent_calibration"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(config: dict[str, Any]) -> dict[str, Any]:
    execution = config["execution"]
    if execution["workers"] != 8 or execution["ordered_map"] is not True:
        raise ValueError("eight-worker execution contract mismatch")
    if (os.cpu_count() or 0) < execution["required_logical_processors"]:
        raise ValueError("eight logical processors are unavailable")
    frozen = validate_t074(scientific_config(config))
    return {"experiment_id": config["experiment_id"], "workers": 8,
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
    with ProcessPoolExecutor(max_workers=execution["workers"]) as pool:
        endpoints = list(pool.map(
            _run_payload, ((frozen, row) for row in rows),
            chunksize=execution["chunksize"]))
    runtime = time.perf_counter() - started
    summary, cells = analyze_t074(frozen, endpoints)
    compute_gates = {
        "F1": len(summary["descriptive_criteria"]) == 11,
        "F2": len(endpoints) == 13_824 and len(cells) == 432,
        "F3": runtime <= 60.0 * execution["hard_timeout_minutes"],
        "F4": False, "F5": False,
    }
    result = {**summary, "execution_experiment_id": config["experiment_id"],
              "runtime_seconds": runtime, "compute_gates": compute_gates,
              "all_pre_reproduction_compute_gates_pass": all(
                  compute_gates[key] for key in ("F1", "F2", "F3"))}
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

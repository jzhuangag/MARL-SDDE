"""Strict post-run validation for the frozen T-060A pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.nonlinear_markov_td.run_t060a_minatar_fixed_q_pilot import (
    DEFAULT_CONFIG,
    analyze,
    file_sha256,
    load_config,
)


ARTIFACTS = ("endpoints.csv", "cells.csv", "reference_moments.json", "summary.json")
INTEGER_FIELDS = {
    "master_seed",
    "overhead",
    "delay",
    "q",
    "updates",
    "message_budget",
    "message_used",
    "environment_budget",
    "environment_used",
    "selected_common_actors",
}
FLOAT_FIELDS = {
    "rho",
    "prediction_risk",
    "bellman_residual",
    "average_weight_norm",
}


def read_endpoints(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in INTEGER_FIELDS:
            row[field] = int(row[field])
        for field in FLOAT_FIELDS:
            row[field] = float(row[field])
    return rows


def maximum_numeric_difference(first: Any, second: Any) -> float:
    if isinstance(first, dict) and isinstance(second, dict):
        if set(first) != set(second):
            return float("inf")
        return max((maximum_numeric_difference(first[key], second[key]) for key in first), default=0.0)
    if isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            return float("inf")
        return max((maximum_numeric_difference(a, b) for a, b in zip(first, second)), default=0.0)
    if isinstance(first, (int, float)) and isinstance(second, (int, float)) and not isinstance(first, bool) and not isinstance(second, bool):
        return abs(float(first) - float(second))
    return 0.0 if first == second else float("inf")


def validate(
    *, config_path: Path, primary: Path, reproduction: Path
) -> dict[str, Any]:
    config = load_config(config_path)
    hashes = {}
    exact = True
    for name in ARTIFACTS:
        first = file_sha256(primary / name)
        second = file_sha256(reproduction / name)
        hashes[name] = {"primary": first, "reproduction": second, "exact": first == second}
        exact &= first == second
    with (primary / "summary.json").open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    endpoints = read_endpoints(primary / "endpoints.csv")
    replay = analyze(config, endpoints, stored["reference_diagnostics"])
    replay.pop("cell_rows")
    difference = maximum_numeric_difference(stored, replay)
    replay_pass = difference <= 1e-12
    pre_gate = bool(stored["pre_reproduction_value_gate_pass"] and all(stored["gates"].values()))
    return {
        "experiment_id": config["experiment_id"],
        "configuration_sha256": config["configuration_sha256"],
        "artifact_hashes": hashes,
        "byte_exact_reproduction": exact,
        "strict_summary_replay": replay_pass,
        "maximum_summary_numeric_difference": difference,
        "pre_reproduction_value_gate_pass": pre_gate,
        "controller_pilot_authorized": bool(exact and replay_pass and pre_gate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = validate(
        config_path=arguments.config,
        primary=arguments.primary,
        reproduction=arguments.reproduction,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        if arguments.output.exists():
            raise FileExistsError(f"refusing to overwrite {arguments.output}")
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

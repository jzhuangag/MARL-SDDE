"""Strict reproduction and replay validator for T-061A."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.nonlinear_markov_td.run_t061a_reward_free_controller_pilot import (
    DEFAULT_CONFIG,
    analyze,
    file_sha256,
    load_config,
)
from experiments.nonlinear_markov_td.validate_t060a_minatar_fixed_q_pilot import (
    maximum_numeric_difference,
)


ARTIFACTS = ("endpoints.csv", "cells.csv", "summary.json")
INTEGER_FIELDS = {
    "master_seed", "overhead", "delay", "match_count", "selected_q",
    "strong_q", "true_rho_q", "controller_updates", "message_budget",
    "environment_budget", "probe_message", "probe_environment",
    "learning_message", "learning_environment",
}
FLOAT_FIELDS = {
    "rho", "controller_risk", "strong_risk", "true_rho_full_budget_risk",
    "controller_bellman_residual", "strong_bellman_residual",
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


def validate(*, config_path: Path, primary: Path, reproduction: Path) -> dict[str, Any]:
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
    replay = analyze(config, read_endpoints(primary / "endpoints.csv"))
    replay.pop("cell_rows")
    difference = maximum_numeric_difference(stored, replay)
    replay_pass = difference <= 1e-12
    pre_gate = bool(stored["pre_reproduction_pilot_gate_pass"] and all(stored["gates"].values()))
    return {
        "experiment_id": config["experiment_id"],
        "configuration_sha256": config["configuration_sha256"],
        "artifact_hashes": hashes,
        "byte_exact_reproduction": exact,
        "strict_summary_replay": replay_pass,
        "maximum_summary_numeric_difference": difference,
        "pre_reproduction_pilot_gate_pass": pre_gate,
        "formal_preregistration_authorized": bool(exact and replay_pass and pre_gate),
        "gpu_authorized": False,
        "hpc4_authorized": False,
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

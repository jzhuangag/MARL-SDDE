"""Strict post-result replay audit for the under-enforced frozen F11 code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.dependence_delay_linear.run_t055a_confirmation_cpu_pilot import (
    analyze as base_analyze,
)
from experiments.dependence_delay_linear.run_t055a_validation import (
    maximum_numeric_difference,
)
from experiments.dependence_delay_linear.run_t057a_formal_cpu import load_config


ROOT = Path(__file__).resolve().parents[2]
RESULTS = (
    ROOT
    / "experiments"
    / "dependence_delay_linear"
    / "results"
    / "t057a_formal_cpu"
)
DEFAULT_OUTPUT = ROOT / "docs" / "t057a_summary_replay_audit.json"


def execute(results: Path = RESULTS) -> dict[str, Any]:
    config = load_config()
    endpoints = pd.read_csv(results / "endpoints.csv")
    with (results / "summary.json").open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    replayed = base_analyze(config, endpoints.to_dict("records"))
    replayed.pop("cell_rows")
    maximum_error = maximum_numeric_difference(replayed, stored)
    duplicates = int(endpoints.duplicated(["cell_id", "seed"]).sum())
    per_cell_counts = endpoints.groupby("cell_id")["seed"].nunique()
    coverage = (
        len(endpoints) == config["expected_workload"]["endpoints"]
        and endpoints["cell_id"].nunique() == config["expected_workload"]["cells"]
        and endpoints["seed"].nunique() == config["expected_workload"]["seeds"]
        and duplicates == 0
        and bool((per_cell_counts == config["expected_workload"]["seeds"]).all())
    )
    return {
        "audit_id": "T-057A-strict-summary-replay",
        "classification": "post-result integrity audit; no scientific gate or threshold changed",
        "frozen_analyzer_F11_implementation": "gate dictionaries equal",
        "preregistered_F11_text": "stored summary replays from endpoints",
        "strict_numeric_tolerance": 1e-12,
        "maximum_absolute_numeric_difference": maximum_error,
        "strict_full_summary_replay_pass": maximum_error <= 1e-12,
        "endpoint_rows": int(len(endpoints)),
        "unique_cells": int(endpoints["cell_id"].nunique()),
        "unique_seeds": int(endpoints["seed"].nunique()),
        "duplicate_cell_seed_rows": duplicates,
        "coverage_pass": coverage,
        "F11_textual_intent_satisfied": coverage and maximum_error <= 1e-12,
        "formal_decision_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = execute()
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    arguments.output.write_text(data, encoding="utf-8")
    print(data, end="")


if __name__ == "__main__":
    main()

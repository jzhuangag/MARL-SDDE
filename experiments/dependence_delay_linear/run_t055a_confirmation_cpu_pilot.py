"""Run the independently preregistered T-055A CPU confirmation pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from experiments.dependence_delay_linear import run_t053a_sampled_cpu_pilot as base


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs" / "t055a_confirmation_cpu_pilot_preregistration.json"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "dependence_delay_linear"
    / "results"
    / "t055a_confirmation_cpu_pilot"
)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return base.load_config(path)


def analyze(config: dict[str, Any], endpoints: list[dict[str, Any]]) -> dict[str, Any]:
    """Reuse the frozen T-053A analysis, changing only registered coverage."""

    summary = base.analyze(config, endpoints)
    expected = config["expected_workload"]
    coverage = (
        len(endpoints) == expected["endpoints"]
        and summary["cells"] == expected["cells"]
        and len(config["pilot_seeds"]) == expected["seeds"]
    )
    summary["gates"]["P11"] = coverage
    summary["pre_reproduction_gates_pass"] = all(summary["gates"].values())
    return summary


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    prepared = base.prepare(config)
    started = time.perf_counter()
    endpoints = []
    for scenario in base.scenario_rows(config):
        for seed in config["pilot_seeds"]:
            endpoints.append(
                base.run_endpoint(
                    config=config,
                    prepared=prepared,
                    scenario=scenario,
                    master_seed=int(seed),
                )
            )
    summary = analyze(config, endpoints)
    runtime_seconds = time.perf_counter() - started
    cell_rows = summary.pop("cell_rows")
    output.mkdir(parents=True, exist_ok=False)
    base.write_csv(output / "endpoints.csv", endpoints)
    base.write_csv(output / "cells.csv", cell_rows)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {**summary, "runtime_seconds": runtime_seconds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "estimate", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.mode == "validate":
        result = base.validate(config)
    elif arguments.mode == "estimate":
        result = base.estimate(config)
    else:
        result = run(config, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

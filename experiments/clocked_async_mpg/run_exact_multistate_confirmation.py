"""Frozen runner for the exact multi-state asynchronous MPG confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .exact_multistate_confirmation import (
    simulate_asynchronous,
    simulate_shadow_barrier,
    summarize_trajectory,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR/"exact_confirmation_config.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "config_version",
        "couplings",
        "development_namespaces_excluded",
        "maximum_time",
        "policies",
        "primary_service_ratios",
        "seed_count",
        "seed_namespace",
        "service_ratios",
        "target_normalized_gap",
    }
    if set(config) != required:
        raise ValueError("configuration keys do not match the frozen schema")
    if config["config_version"] != 1:
        raise ValueError("unsupported config version")
    if config["seed_namespace"] in config["development_namespaces_excluded"]:
        raise ValueError("confirmation namespace overlaps development")
    if config["policies"] != [
        "single_flight_async",
        "fully_utilized_shadow_barrier",
    ]:
        raise ValueError("policy registry changed")
    if int(config["seed_count"]) <= 0:
        raise ValueError("seed_count must be positive")
    return config


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values)/len(values))


def run(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    namespace = str(config["seed_namespace"])
    target = float(config["target_normalized_gap"])
    maximum_time = float(config["maximum_time"])
    for coupling in config["couplings"]:
        for ratio in config["service_ratios"]:
            for seed_index in range(int(config["seed_count"])):
                asynchronous = simulate_asynchronous(
                    float(coupling),
                    float(ratio),
                    seed_index,
                    namespace,
                    maximum_time,
                )
                shadow = simulate_shadow_barrier(
                    float(coupling),
                    float(ratio),
                    seed_index,
                    namespace,
                    maximum_time,
                )
                for policy, result in (
                    ("single_flight_async", asynchronous),
                    ("fully_utilized_shadow_barrier", shadow),
                ):
                    endpoint = summarize_trajectory(result["trajectory"], target)
                    row = {
                        "coupling": float(coupling),
                        "final_gradient_norm": endpoint["final_gradient_norm"],
                        "final_normalized_gap": endpoint["final_normalized_gap"],
                        "packets": endpoint["packets"],
                        "policy": policy,
                        "seed_index": seed_index,
                        "service_ratio": float(ratio),
                        "time_to_target": endpoint["time_to_target"],
                        "updates": endpoint["updates"],
                    }
                    if policy == "single_flight_async":
                        row["max_realized_delay"] = int(result["max_realized_delay"])
                        row["registered_delay"] = int(result["registered_delay"])
                    rows.append(row)
    return rows, analyze(rows, config)


def analyze(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    indexed: dict[tuple[float, float, int, str], dict[str, Any]] = {}
    finite = True
    delay_valid = True
    target_count = 0
    for row in rows:
        key = (
            float(row["coupling"]),
            float(row["service_ratio"]),
            int(row["seed_index"]),
            str(row["policy"]),
        )
        if key in indexed:
            raise ValueError("duplicate endpoint row")
        indexed[key] = row
        numeric = [
            row["final_gradient_norm"],
            row["final_normalized_gap"],
            row["packets"],
            row["updates"],
        ]
        finite &= all(math.isfinite(float(value)) for value in numeric)
        target_count += row["time_to_target"] is not None
        if row["policy"] == "single_flight_async":
            delay_valid &= int(row["max_realized_delay"]) <= int(
                row["registered_delay"]
            )

    cells: list[dict[str, Any]] = []
    seed_count = int(config["seed_count"])
    for coupling in config["couplings"]:
        for ratio in config["service_ratios"]:
            time_ratios: list[float] = []
            final_gap_ratios: list[float] = []
            for seed_index in range(seed_count):
                async_row = indexed[(
                    float(coupling), float(ratio), seed_index, "single_flight_async"
                )]
                shadow_row = indexed[(
                    float(coupling),
                    float(ratio),
                    seed_index,
                    "fully_utilized_shadow_barrier",
                )]
                if (
                    async_row["time_to_target"] is not None
                    and shadow_row["time_to_target"] is not None
                ):
                    time_ratios.append(
                        float(async_row["time_to_target"])
                        /float(shadow_row["time_to_target"])
                    )
                final_gap_ratios.append(
                    max(1e-15, float(async_row["final_normalized_gap"]))
                    /max(1e-15, float(shadow_row["final_normalized_gap"]))
                )
            cells.append(
                {
                    "coupling": float(coupling),
                    "final_gap_geometric_ratio": _geometric_mean(final_gap_ratios),
                    "median_time_ratio": (
                        statistics.median(time_ratios) if time_ratios else None
                    ),
                    "service_ratio": float(ratio),
                    "target_pair_coverage": len(time_ratios)/seed_count,
                }
            )

    primary = [
        cell
        for cell in cells
        if cell["service_ratio"] in config["primary_service_ratios"]
    ]
    primary_time = [
        float(cell["median_time_ratio"])
        for cell in primary
        if cell["median_time_ratio"] is not None
    ]
    primary_gap = [float(cell["final_gap_geometric_ratio"]) for cell in primary]
    directional_heterogeneity = all(
        next(
            float(cell["median_time_ratio"])
            for cell in cells
            if cell["coupling"] == float(coupling) and cell["service_ratio"] == 8.0
        )
        < next(
            float(cell["median_time_ratio"])
            for cell in cells
            if cell["coupling"] == float(coupling) and cell["service_ratio"] == 2.0
        )
        for coupling in config["couplings"]
    )
    directional_coupling = all(
        next(
            float(cell["median_time_ratio"])
            for cell in cells
            if cell["coupling"] == 0.24 and cell["service_ratio"] == float(ratio)
        )
        >= next(
            float(cell["median_time_ratio"])
            for cell in cells
            if cell["coupling"] == 0.0 and cell["service_ratio"] == float(ratio)
        )
        for ratio in config["primary_service_ratios"]
    )
    gates = {
        "C1_finite_and_delay_valid": finite and delay_valid,
        "C2_target_pair_coverage_at_least_95pct": (
            target_count/len(rows) >= 0.95
        ),
        "C3_primary_geometric_time_ratio_at_most_0_80": (
            len(primary_time) == len(primary) and _geometric_mean(primary_time) <= 0.80
        ),
        "C4_at_least_10_of_12_primary_cells_faster": (
            sum(value < 1.0 for value in primary_time) >= 10
        ),
        "C5_heterogeneity_direction_in_all_couplings": directional_heterogeneity,
        "C6_coupling_cost_direction_in_all_primary_ratios": directional_coupling,
        "C7_primary_final_gap_geometric_ratio_at_most_1": (
            _geometric_mean(primary_gap) <= 1.0
        ),
        "C8_all_packets_fully_counted": all(float(row["packets"]) >= float(row["updates"]) for row in rows),
        "C9_confirmation_namespace_disjoint": (
            config["seed_namespace"] not in config["development_namespaces_excluded"]
        ),
    }
    return {
        "all_pre_reproduction_gates_pass": all(gates.values()),
        "cells": cells,
        "gates": gates,
        "primary_cell_count": len(primary),
        "primary_final_gap_geometric_ratio": _geometric_mean(primary_gap),
        "primary_time_geometric_ratio": (
            _geometric_mean(primary_time) if primary_time else None
        ),
        "row_count": len(rows),
        "target_endpoint_coverage": target_count/len(rows),
    }


def write_results(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    config_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir/"endpoints.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"))+"\n")
    payload = {
        "config_sha256": sha256(config_path),
        "endpoints_sha256": sha256(rows_path),
        "summary": summary,
    }
    with (output_dir/"summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "validate":
        print(json.dumps({"config_sha256": sha256(args.config), "valid": True}))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for run")
    rows, summary = run(config)
    write_results(args.output_dir, rows, summary, args.config)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

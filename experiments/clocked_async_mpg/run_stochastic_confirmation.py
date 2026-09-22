"""Frozen CPU runner for the stochastic clocked-MPG confirmation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .stochastic_multistate import (
    simulate_stochastic_asynchronous,
    simulate_stochastic_shadow_barrier,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR/"stochastic_confirmation_config.json"
POLICIES = [
    "single_flight_constant",
    "common_global",
    "single_flight_local",
    "generic_rate_balanced",
    "fully_utilized_shadow_barrier",
]
POLICIES_V2 = [
    "single_flight_pathwise_constant",
    "single_flight_constant",
    "common_global",
    "single_flight_local",
    "generic_rate_balanced",
    "fully_utilized_shadow_barrier",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values)/len(values))


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "batch_size",
        "config_version",
        "couplings",
        "development_namespaces_excluded",
        "horizon",
        "maximum_time",
        "policies",
        "primary_policy",
        "primary_service_ratios",
        "seed_count",
        "seed_namespace",
        "service_ratios",
        "step_fraction",
        "strong_async_policy",
        "synchronous_policy",
        "target_normalized_gap",
    }
    if int(config.get("config_version", -1)) == 2:
        required = required|{"history_inflation"}
    if set(config) != required:
        raise ValueError("configuration keys do not match the frozen schema")
    expected_policies = POLICIES if config["config_version"] == 1 else POLICIES_V2
    if config["config_version"] not in (1, 2) or config["policies"] != expected_policies:
        raise ValueError("unsupported configuration version or policy registry")
    expected_primary = (
        "single_flight_constant"
        if config["config_version"] == 1
        else "single_flight_pathwise_constant"
    )
    if config["primary_policy"] != expected_primary:
        raise ValueError("primary policy changed")
    if config["strong_async_policy"] != "common_global":
        raise ValueError("strong asynchronous policy changed")
    if config["synchronous_policy"] != "fully_utilized_shadow_barrier":
        raise ValueError("synchronous policy changed")
    if config["seed_namespace"] in config["development_namespaces_excluded"]:
        raise ValueError("confirmation namespace overlaps development")
    if int(config["seed_count"]) <= 0:
        raise ValueError("seed_count must be positive")
    if float(config["step_fraction"]) != 1.0:
        raise ValueError("registered maximal step fraction changed")
    if config["config_version"] == 2 and float(config["history_inflation"]) != 2.0:
        raise ValueError("registered pathwise history inflation changed")
    return config


def _one(job: tuple[Any, ...]) -> dict[str, Any]:
    (
        coupling,
        service_ratio,
        seed_index,
        namespace,
        maximum_time,
        horizon,
        batch_size,
        step_fraction,
        target_normalized_gap,
        policy,
        history_inflation,
    ) = job
    parameters = {
        "coupling": float(coupling),
        "service_ratio": float(service_ratio),
        "seed_index": int(seed_index),
        "namespace": str(namespace),
        "maximum_time": float(maximum_time),
        "horizon": int(horizon),
        "batch_size": int(batch_size),
        "step_fraction": float(step_fraction),
        "target_normalized_gap": float(target_normalized_gap),
    }
    if policy == "fully_utilized_shadow_barrier":
        result = simulate_stochastic_shadow_barrier(**parameters)
    else:
        result = simulate_stochastic_asynchronous(
            **parameters,
            step_rule=str(policy),
            history_inflation=float(history_inflation),
        )
    return {
        "coupling": float(coupling),
        "policy": str(policy),
        "seed_index": int(seed_index),
        "service_ratio": float(service_ratio),
        **result,
    }


def run(config: dict[str, Any], workers: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    jobs = [
        (
            coupling,
            service_ratio,
            seed_index,
            config["seed_namespace"],
            config["maximum_time"],
            config["horizon"],
            config["batch_size"],
            config["step_fraction"],
            config["target_normalized_gap"],
            policy,
            float(config.get("history_inflation", 1.0)),
        )
        for coupling in config["couplings"]
        for service_ratio in config["service_ratios"]
        for seed_index in range(int(config["seed_count"]))
        for policy in config["policies"]
    ]
    if workers == 1:
        rows = [_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_one, jobs, chunksize=5))
    return rows, analyze(rows, config)


def _cell_comparison(
    indexed: dict[tuple[float, float, int, str], dict[str, Any]],
    coupling: float,
    service_ratio: float,
    seed_count: int,
    candidate_policy: str,
    comparator_policy: str,
) -> dict[str, float | int | None]:
    time_ratios: list[float] = []
    work_ratios: list[float] = []
    gap_ratios: list[float] = []
    for seed_index in range(seed_count):
        candidate = indexed[(coupling, service_ratio, seed_index, candidate_policy)]
        comparator = indexed[(coupling, service_ratio, seed_index, comparator_policy)]
        if (
            candidate["time_to_target"] is not None
            and comparator["time_to_target"] is not None
        ):
            time_ratios.append(
                float(candidate["time_to_target"])
                /float(comparator["time_to_target"])
            )
            work_ratios.append(
                float(candidate["transition_work_at_target"])
                /float(comparator["transition_work_at_target"])
            )
        gap_ratios.append(
            max(1e-15, float(candidate["final_normalized_gap"]))
            /max(1e-15, float(comparator["final_normalized_gap"]))
        )
    return {
        "final_gap_geometric_ratio": _geometric_mean(gap_ratios),
        "median_time_ratio": statistics.median(time_ratios) if time_ratios else None,
        "median_work_ratio": statistics.median(work_ratios) if work_ratios else None,
        "paired_target_count": len(time_ratios),
        "paired_target_coverage": len(time_ratios)/seed_count,
    }


def analyze(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    expected_rows = (
        len(config["couplings"])
        *len(config["service_ratios"])
        *int(config["seed_count"])
        *len(config["policies"])
    )
    indexed: dict[tuple[float, float, int, str], dict[str, Any]] = {}
    finite = True
    accounting_valid = True
    delay_valid = True
    policy_target_count = {policy: 0 for policy in config["policies"]}
    packet_cost = int(config["batch_size"])*int(config["horizon"])
    for row in rows:
        key = (
            float(row["coupling"]),
            float(row["service_ratio"]),
            int(row["seed_index"]),
            str(row["policy"]),
        )
        if key in indexed:
            raise ValueError("duplicate endpoint key")
        indexed[key] = row
        numeric = [
            row["applied_updates"],
            row["cancelled_transition_work"],
            row["completed_packets"],
            row["completed_transition_work"],
            row["final_gradient_norm"],
            row["final_normalized_gap"],
            row["total_transition_work"],
        ]
        finite &= all(math.isfinite(float(value)) for value in numeric)
        completed = float(row["completed_transition_work"])
        total = float(row["total_transition_work"])
        cancelled = float(row["cancelled_transition_work"])
        accounting_valid &= (
            completed == int(row["completed_packets"])*packet_cost
            and total+1e-10 >= completed
            and completed >= 0.0
            and cancelled >= 0.0
        )
        target_time = row["time_to_target"]
        target_work = row["transition_work_at_target"]
        accounting_valid &= (target_time is None) == (target_work is None)
        if target_work is not None:
            accounting_valid &= (
                math.isfinite(float(target_work))
                and 0.0 <= float(target_work) <= total+1e-8
            )
            policy_target_count[str(row["policy"])] += 1
        if row["policy"] != config["synchronous_policy"]:
            delay_valid &= int(row["max_realized_delay"]) <= int(
                row["registered_delay"]
            )

    expected_keys = expected_rows == len(rows) == len(indexed)
    cells: list[dict[str, Any]] = []
    seed_count = int(config["seed_count"])
    primary_policy = str(config["primary_policy"])
    barrier_policy = str(config["synchronous_policy"])
    raw_policy = str(config["strong_async_policy"])
    for coupling_value in config["couplings"]:
        coupling = float(coupling_value)
        for ratio_value in config["service_ratios"]:
            ratio = float(ratio_value)
            primary_barrier = _cell_comparison(
                indexed,
                coupling,
                ratio,
                seed_count,
                primary_policy,
                barrier_policy,
            )
            primary_raw = _cell_comparison(
                indexed,
                coupling,
                ratio,
                seed_count,
                primary_policy,
                raw_policy,
            )
            coverage = {
                policy: sum(
                    indexed[(coupling, ratio, seed_index, policy)]["time_to_target"]
                    is not None
                    for seed_index in range(seed_count)
                )/seed_count
                for policy in config["policies"]
            }
            cells.append(
                {
                    "coupling": coupling,
                    "policy_target_coverage": coverage,
                    "primary_vs_barrier": primary_barrier,
                    "primary_vs_raw_async": primary_raw,
                    "service_ratio": ratio,
                }
            )

    primary_ratios = {float(value) for value in config["primary_service_ratios"]}
    primary_cells = [
        cell for cell in cells if float(cell["service_ratio"]) in primary_ratios
    ]
    barrier_times = [
        float(cell["primary_vs_barrier"]["median_time_ratio"])
        for cell in primary_cells
        if cell["primary_vs_barrier"]["median_time_ratio"] is not None
    ]
    barrier_work = [
        float(cell["primary_vs_barrier"]["median_work_ratio"])
        for cell in primary_cells
        if cell["primary_vs_barrier"]["median_work_ratio"] is not None
    ]
    barrier_gap = [
        float(cell["primary_vs_barrier"]["final_gap_geometric_ratio"])
        for cell in primary_cells
    ]
    raw_times = [
        float(cell["primary_vs_raw_async"]["median_time_ratio"])
        for cell in primary_cells
        if cell["primary_vs_raw_async"]["median_time_ratio"] is not None
    ]
    raw_work = [
        float(cell["primary_vs_raw_async"]["median_work_ratio"])
        for cell in primary_cells
        if cell["primary_vs_raw_async"]["median_work_ratio"] is not None
    ]
    policy_coverage = {
        policy: policy_target_count[policy]
        /(
            len(config["couplings"])
            *len(config["service_ratios"])
            *seed_count
        )
        for policy in config["policies"]
    }
    primary_count = len(primary_cells)
    primary_barrier_time = (
        _geometric_mean(barrier_times) if barrier_times else None
    )
    primary_barrier_work = (
        _geometric_mean(barrier_work) if barrier_work else None
    )
    primary_barrier_gap = _geometric_mean(barrier_gap)
    primary_raw_time = _geometric_mean(raw_times) if raw_times else None
    primary_raw_work = _geometric_mean(raw_work) if raw_work else None
    certificate_cost_limit = 1.35 if config["config_version"] == 1 else 2.0
    gates = {
        "S1_schema_unique_finite": expected_keys and finite,
        "S2_primary_cell_paired_coverage_at_least_0_95": all(
            float(cell["primary_vs_barrier"]["paired_target_coverage"]) >= 0.95
            for cell in primary_cells
        ),
        "S3_primary_time_ratio_at_most_0_75": (
            len(barrier_times) == primary_count
            and primary_barrier_time is not None
            and primary_barrier_time <= 0.75
        ),
        "S4_primary_work_ratio_at_most_0_75": (
            len(barrier_work) == primary_count
            and primary_barrier_work is not None
            and primary_barrier_work <= 0.75
        ),
        "S5_at_least_10_of_12_primary_cells_faster": sum(
            value < 1.0 for value in barrier_times
        ) >= 10,
        "S6_primary_final_gap_ratio_at_most_1_05": primary_barrier_gap <= 1.05,
        "S7_certificate_cost_vs_raw_within_registered_limit": (
            len(raw_times) == primary_count
            and len(raw_work) == primary_count
            and primary_raw_time is not None
            and primary_raw_work is not None
            and primary_raw_time <= certificate_cost_limit
            and primary_raw_work <= certificate_cost_limit
        ),
        "S8_transition_accounting_valid": accounting_valid,
        "S9_registered_delay_valid": delay_valid,
        "S10_confirmation_namespace_disjoint": (
            config["seed_namespace"] not in config["development_namespaces_excluded"]
        ),
        "S11_all_policy_target_coverage_at_least_0_95": all(
            value >= 0.95 for value in policy_coverage.values()
        ),
    }
    return {
        "all_prereproduction_gates_pass": all(gates.values()),
        "certificate_cost_limit": certificate_cost_limit,
        "cells": cells,
        "gates": gates,
        "policy_target_coverage": policy_coverage,
        "primary_cell_count": primary_count,
        "primary_vs_barrier_final_gap_geometric_ratio": primary_barrier_gap,
        "primary_vs_barrier_time_geometric_ratio": primary_barrier_time,
        "primary_vs_barrier_work_geometric_ratio": primary_barrier_work,
        "primary_vs_raw_time_geometric_ratio": primary_raw_time,
        "primary_vs_raw_work_geometric_ratio": primary_raw_work,
        "row_count": len(rows),
    }


def write_results(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    config_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    endpoints_path = output_dir/"endpoints.jsonl"
    with endpoints_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"))+"\n")
    source_paths = {
        "runner": Path(__file__).resolve(),
        "simulator": PACKAGE_DIR/"stochastic_multistate.py",
        "drift": PACKAGE_DIR/"finite_time_drift.py",
    }
    payload = {
        "config_sha256": sha256(config_path),
        "endpoints_sha256": sha256(endpoints_path),
        "source_sha256": {
            name: sha256(path) for name, path in source_paths.items()
        },
        "summary": summary,
    }
    with (output_dir/"summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")


def check_reproduction(original: Path, reproduction: Path) -> dict[str, Any]:
    comparisons = {}
    for filename in ("endpoints.jsonl", "summary.json"):
        first = original/filename
        second = reproduction/filename
        comparisons[filename] = {
            "byte_identical": first.read_bytes() == second.read_bytes(),
            "original_sha256": sha256(first),
            "reproduction_sha256": sha256(second),
        }
    return {
        "S12_byte_exact_reproduction": all(
            entry["byte_identical"] for entry in comparisons.values()
        ),
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "run", "reproduce-check"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--reproduction", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "validate":
        print(json.dumps({"config_sha256": sha256(args.config), "valid": True}))
        return
    if args.mode == "reproduce-check":
        if args.original is None or args.reproduction is None:
            parser.error("--original and --reproduction are required")
        print(json.dumps(check_reproduction(args.original, args.reproduction), sort_keys=True))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for run")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    rows, summary = run(config, args.workers)
    write_results(args.output_dir, rows, summary, args.config)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

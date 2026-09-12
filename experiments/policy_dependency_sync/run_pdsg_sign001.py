"""Frozen PDSG-SIGN-001 exact-model CPU confirmation runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .async_packet_game import geometric_mean, make_budget_cells, simulate


EXPECTED_MANIFEST_SHA256 = (
    "1AC08E5407DCF653670E6C316CD46B8A6AF1D7868F661A4A20D97C5B157D0B6B"
)
PROPOSED_POLICY = "signed_oracle_graph_h4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_manifest(path: Path) -> dict[str, Any]:
    if sha256(path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("manifest hash does not match frozen preregistration")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "PDSG-SIGN-001":
        raise RuntimeError("wrong experiment identifier")
    return manifest


def _run_cell(arguments: tuple[Any, tuple[str, ...], tuple[int, ...], int]):
    cell, policies, seeds, launches = arguments
    rows = [
        simulate(cell, policy, seed, launches)
        for policy in policies
        for seed in seeds
    ]
    return cell, rows


def _aggregate(rows: list[dict[str, Any]], policies: tuple[str, ...]) -> dict:
    by_policy = {
        policy: [row for row in rows if row["policy"] == policy]
        for policy in policies
    }
    return {
        policy: {
            "risk": geometric_mean([float(row["risk"]) for row in policy_rows]),
            "terminal_risk": geometric_mean(
                [max(float(row["terminal_risk"]), 1e-15) for row in policy_rows]
            ),
            "messages_per_launch": float(
                np.mean([float(row["messages_per_launch"]) for row in policy_rows])
            ),
            "environment_per_launch": float(
                np.mean([float(row["environment_per_launch"]) for row in policy_rows])
            ),
            "accepted_step_fraction": float(
                np.mean([float(row["accepted_step_fraction"]) for row in policy_rows])
            ),
            "distinct_graph_supports": float(
                np.mean([float(row["distinct_graph_supports"]) for row in policy_rows])
            ),
            "finite": all(bool(row["finite"]) for row in policy_rows),
        }
        for policy, policy_rows in by_policy.items()
    }


def _median_by_group(
    cells: list[dict[str, Any]], field: str, values: tuple[float | int, ...]
) -> dict[str, float]:
    return {
        str(value): float(
            np.median([row["terminal_gain"] for row in cells if row[field] == value])
        )
        for value in values
    }


def evaluate(
    manifest: dict[str, Any], workers: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    policies = tuple(manifest["policies"])
    baselines = policies[1:]
    launches = int(manifest["model"]["launches"])
    seed_spec = manifest["seeds"]
    seeds = tuple(range(seed_spec["start"], seed_spec["start"] + seed_spec["count"]))
    cells = make_budget_cells()
    tasks = [(cell, policies, seeds, launches) for cell in cells]
    if workers == 1:
        cell_outputs = [_run_cell(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            cell_outputs = list(pool.map(_run_cell, tasks))

    endpoints: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    tolerance = manifest["baseline_selection"]
    for cell, rows in cell_outputs:
        endpoints.extend(rows)
        aggregates = _aggregate(rows, policies)
        feasible = [
            policy
            for policy in baselines
            if aggregates[policy]["messages_per_launch"]
            <= cell.message_budget + float(tolerance["message_tolerance"])
            and aggregates[policy]["environment_per_launch"]
            <= cell.environment_budget + float(tolerance["environment_tolerance"])
        ]
        if not feasible:
            raise RuntimeError(f"no feasible baseline for {cell.key}")
        strongest_auc = min(feasible, key=lambda name: aggregates[name]["risk"])
        strongest_terminal = min(
            feasible, key=lambda name: aggregates[name]["terminal_risk"]
        )
        proposed = aggregates[PROPOSED_POLICY]
        auc_baseline = aggregates[strongest_auc]["risk"]
        terminal_baseline = aggregates[strongest_terminal]["terminal_risk"]
        cell_rows.append(
            {
                "cell": cell.key,
                "coupling": cell.coupling,
                "maximum_extra_delay": cell.maximum_extra_delay,
                "message_budget": cell.message_budget,
                "environment_budget": cell.environment_budget,
                "mode_switch_probability": cell.mode_switch_probability,
                "strongest_auc_baseline": strongest_auc,
                "strongest_terminal_baseline": strongest_terminal,
                "auc_gain": float(
                    (auc_baseline - proposed["risk"]) / auc_baseline
                ),
                "terminal_gain": float(
                    (terminal_baseline - proposed["terminal_risk"])
                    / terminal_baseline
                ),
                "proposed_messages_per_launch": proposed["messages_per_launch"],
                "proposed_environment_per_launch": proposed[
                    "environment_per_launch"
                ],
                "proposed_distinct_graph_supports": proposed[
                    "distinct_graph_supports"
                ],
                "feasible_baseline_count": len(feasible),
                "aggregates": aggregates,
            }
        )

    endpoints.sort(key=lambda row: (row["cell"], row["policy"], row["seed"]))
    cell_rows.sort(key=lambda row: row["cell"])
    active = [row for row in cell_rows if float(row["coupling"]) > 0.0]
    controls = [row for row in cell_rows if float(row["coupling"]) == 0.0]
    active_terminal = np.asarray([row["terminal_gain"] for row in active])
    active_auc = np.asarray([row["auc_gain"] for row in active])
    group_medians = {
        "coupling": _median_by_group(active, "coupling", (0.6, 0.9)),
        "maximum_extra_delay": _median_by_group(
            active, "maximum_extra_delay", (2, 5)
        ),
        "message_budget": _median_by_group(active, "message_budget", (0.5, 1.0)),
        "mode_switch_probability": _median_by_group(
            active, "mode_switch_probability", (0.03, 0.12)
        ),
    }
    expected_endpoints = len(cells) * len(policies) * len(seeds)
    development_seeds = set(seed_spec["development_seeds_excluded"])
    endpoint_seeds = {int(row["seed"]) for row in endpoints}
    all_finite = all(
        bool(row["finite"])
        and math.isfinite(float(row["risk"]))
        and math.isfinite(float(row["terminal_risk"]))
        for row in endpoints
    )
    maximum_message_excess = max(
        row["proposed_messages_per_launch"] - row["message_budget"]
        for row in cell_rows
    )
    maximum_environment_error = max(
        abs(row["proposed_environment_per_launch"] - 4.0) for row in cell_rows
    )
    maximum_control_auc = max(abs(row["auc_gain"]) for row in controls)
    maximum_control_terminal = max(abs(row["terminal_gain"]) for row in controls)
    metrics = {
        "endpoint_count": len(endpoints),
        "cell_count": len(cell_rows),
        "active_cell_count": len(active),
        "control_cell_count": len(controls),
        "active_median_terminal_gain": float(np.median(active_terminal)),
        "active_strict_terminal_gain_fraction": float(
            np.mean(active_terminal > 0.0)
        ),
        "active_minimum_terminal_gain": float(np.min(active_terminal)),
        "active_median_auc_gain": float(np.median(active_auc)),
        "active_strict_auc_gain_fraction": float(np.mean(active_auc > 0.0)),
        "terminal_gain_group_medians": group_medians,
        "maximum_uncoupled_absolute_auc_gain": float(maximum_control_auc),
        "maximum_uncoupled_absolute_terminal_gain": float(
            maximum_control_terminal
        ),
        "minimum_active_mean_distinct_graph_supports": float(
            min(row["proposed_distinct_graph_supports"] for row in active)
        ),
        "maximum_message_excess": float(maximum_message_excess),
        "maximum_environment_charge_error": float(maximum_environment_error),
        "strongest_auc_baseline_counts": dict(
            sorted(Counter(row["strongest_auc_baseline"] for row in active).items())
        ),
        "strongest_terminal_baseline_counts": dict(
            sorted(
                Counter(
                    row["strongest_terminal_baseline"] for row in active
                ).items()
            )
        ),
    }
    gates = {
        "S1": len(endpoints) == expected_endpoints
        and len(cell_rows) == manifest["grid"]["cell_count"]
        and all_finite
        and not (development_seeds & endpoint_seeds),
        "S2": all(
            abs(float(row["environment_per_launch"]) - 4.0) <= 1e-12
            for row in endpoints
        ),
        "S3": maximum_message_excess <= 1.0 / launches + 1e-12,
        "S4": metrics["active_median_terminal_gain"] >= 0.10,
        "S5": metrics["active_strict_terminal_gain_fraction"] >= 0.80
        and metrics["active_minimum_terminal_gain"] >= 0.0,
        "S6": metrics["active_median_auc_gain"] > 0.0
        and metrics["active_strict_auc_gain_fraction"] >= 0.80,
        "S7": all(
            value > 0.0
            for by_value in group_medians.values()
            for value in by_value.values()
        ),
        "S8": maximum_control_auc <= 1e-12
        and maximum_control_terminal <= 1e-12,
        "S9": metrics["minimum_active_mean_distinct_graph_supports"] >= 5.0,
        "S10": len(baselines) == 24
        and all(row["feasible_baseline_count"] > 0 for row in cell_rows),
        "S11": manifest["status"] == "frozen_preregistration"
        and manifest["formal_or_gpu_authorized"] is False,
    }
    summary = {
        "experiment_id": "PDSG-SIGN-001",
        "status": "confirmation",
        "scientific_scope": manifest["scientific_scope"],
        "seeds": list(seeds),
        "launches": launches,
        "policies": list(policies),
        "metrics": metrics,
        "gates": gates,
        "primary_pass": all(gates.values()),
        "S12_reproduction": "not_evaluated_in_single_run",
        "formal_or_gpu_authorized": False,
    }
    return endpoints, cell_rows, summary


def write_outputs(
    output: Path,
    endpoints: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    elapsed_seconds: float,
    workers: int,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    endpoint_path = output / "endpoints.csv"
    fieldnames = list(endpoints[0])
    with endpoint_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in endpoints:
            serial = dict(row)
            serial["horizon_counts"] = json.dumps(
                serial["horizon_counts"], sort_keys=True, separators=(",", ":")
            )
            writer.writerow(serial)
    (output / "cells.json").write_text(
        json.dumps(cell_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "elapsed_seconds": elapsed_seconds,
        "workers": workers,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "hashes": {
            name: sha256(output / name)
            for name in ("endpoints.csv", "cells.json", "summary.json")
        },
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "run"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.mode == "validate":
        cells = make_budget_cells()
        print(
            json.dumps(
                {
                    "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                    "cells": len(cells),
                    "policies": len(manifest["policies"]),
                    "seeds": manifest["seeds"]["count"],
                    "expected_endpoints": len(cells)
                    * len(manifest["policies"])
                    * manifest["seeds"]["count"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.output is None:
        parser.error("--output is required in run mode")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    started = time.time()
    endpoints, cells, summary = evaluate(manifest, args.workers)
    elapsed = time.time() - started
    write_outputs(args.output, endpoints, cells, summary, elapsed, args.workers)
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    print(json.dumps(summary["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

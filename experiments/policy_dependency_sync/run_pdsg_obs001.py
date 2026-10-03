"""Frozen PDSG-OBS-001 observable signed-controller CPU runner."""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from . import run_pdsg_sign001 as sign
from .async_packet_game import make_budget_cells


EXPECTED_MANIFEST_SHA256 = (
    "32C7148C74CF0EA086A94FD85C3676107B32422AAF9C0819AC0487E187FA95A4"
)
PROPOSED_POLICY = "signed_online_model_graph_h4"
ORACLE_POLICY = "signed_oracle_graph_h4"


def load_manifest(path: Path) -> dict[str, Any]:
    if sign.sha256(path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("manifest hash does not match frozen preregistration")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "PDSG-OBS-001":
        raise RuntimeError("wrong experiment identifier")
    return manifest


def _oracle_outputs(
    cells: list[Any], seeds: tuple[int, ...], launches: int, workers: int
) -> list[tuple[Any, list[dict[str, Any]]]]:
    tasks = [(cell, (ORACLE_POLICY,), seeds, launches) for cell in cells]
    if workers == 1:
        return [sign._run_cell(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(sign._run_cell, tasks))


def evaluate(
    manifest: dict[str, Any], workers: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sign.PROPOSED_POLICY = PROPOSED_POLICY
    endpoints, cell_rows, base_summary = sign.evaluate(manifest, workers)
    cells = make_budget_cells()
    launches = int(manifest["model"]["launches"])
    seed_spec = manifest["seeds"]
    seeds = tuple(range(seed_spec["start"], seed_spec["start"] + seed_spec["count"]))
    oracle_by_cell: dict[str, list[dict[str, Any]]] = {}
    for cell, rows in _oracle_outputs(cells, seeds, launches, workers):
        oracle_by_cell[cell.key] = rows
        endpoints.extend(rows)

    online_by_key = {
        (str(row["cell"]), int(row["seed"])): row
        for row in endpoints
        if row["policy"] == PROPOSED_POLICY
    }
    oracle_recovery = []
    trace_agreement = []
    target_errors = []
    oracle_denominators = []
    for row in cell_rows:
        oracle_rows = oracle_by_cell[row["cell"]]
        oracle_aggregate = sign._aggregate(oracle_rows, (ORACLE_POLICY,))[ORACLE_POLICY]
        row["oracle_diagnostic"] = oracle_aggregate
        baseline = float(
            row["aggregates"][row["strongest_terminal_baseline"]]["terminal_risk"]
        )
        observed = float(row["aggregates"][PROPOSED_POLICY]["terminal_risk"])
        oracle = float(oracle_aggregate["terminal_risk"])
        denominator = baseline - oracle
        row["oracle_terminal_denominator"] = denominator
        row["oracle_terminal_headroom_recovery"] = (
            (baseline - observed) / denominator if denominator > 0.0 else None
        )
        if float(row["coupling"]) > 0.0:
            oracle_denominators.append(denominator)
            if denominator > 0.0:
                oracle_recovery.append(float(row["oracle_terminal_headroom_recovery"]))
            for oracle_row in oracle_rows:
                online_row = online_by_key[(row["cell"], int(oracle_row["seed"]))]
                trace_agreement.append(
                    online_row["graph_trace_sha256"]
                    == oracle_row["graph_trace_sha256"]
                )
                target_errors.append(float(online_row["target_estimation_error"]))

    endpoints.sort(key=lambda row: (row["cell"], row["policy"], row["seed"]))
    cell_rows.sort(key=lambda row: row["cell"])
    metrics = dict(base_summary["metrics"])
    metrics.update(
        {
            "endpoint_count_with_oracle_diagnostic": len(endpoints),
            "active_oracle_positive_denominator_count": int(
                sum(value > 0.0 for value in oracle_denominators)
            ),
            "active_median_oracle_terminal_headroom_recovery": float(
                np.median(oracle_recovery) if oracle_recovery else float("nan")
            ),
            "active_minimum_oracle_terminal_headroom_recovery": float(
                min(oracle_recovery) if oracle_recovery else float("nan")
            ),
            "active_online_oracle_trace_agreement_fraction": float(
                np.mean(trace_agreement)
            ),
            "active_median_target_estimation_error": float(np.median(target_errors)),
            "active_maximum_target_estimation_error": float(max(target_errors)),
        }
    )

    excluded = set(seed_spec["excluded"])
    observed_seeds = {int(row["seed"]) for row in endpoints}
    active_online = [
        row
        for row in endpoints
        if row["policy"] == PROPOSED_POLICY
        and row["cell"].split("|")[0] != "c=0"
    ]
    gates = {
        "O1": len(endpoints) == 24 * 26 * 16
        and len(cell_rows) == 24
        and not (excluded & observed_seeds)
        and all(
            bool(row["finite"])
            and math.isfinite(float(row["risk"]))
            and math.isfinite(float(row["terminal_risk"]))
            for row in endpoints
        ),
        "O2": all(
            abs(float(row["environment_per_launch"]) - 4.0) <= 1e-12
            for row in endpoints
        ),
        "O3": metrics["maximum_message_excess"] <= 1.0 / launches + 1e-12,
        "O4": metrics["active_median_terminal_gain"] >= 0.10,
        "O5": metrics["active_strict_terminal_gain_fraction"] >= 0.80
        and metrics["active_minimum_terminal_gain"] >= 0.0,
        "O6": metrics["active_median_auc_gain"] > 0.0
        and metrics["active_strict_auc_gain_fraction"] >= 0.80,
        "O7": all(
            value > 0.0
            for by_value in metrics["terminal_gain_group_medians"].values()
            for value in by_value.values()
        ),
        "O8": metrics["maximum_uncoupled_absolute_auc_gain"] <= 1e-12
        and metrics["maximum_uncoupled_absolute_terminal_gain"] <= 1e-12,
        "O9": metrics["minimum_active_mean_distinct_graph_supports"] >= 5.0,
        "O10": len(manifest["policies"][1:]) == 24
        and all(row["feasible_baseline_count"] > 0 for row in cell_rows),
        "O11": manifest["status"] == "frozen_preregistration"
        and manifest["formal_or_gpu_authorized"] is False
        and sign.sha256(
            Path("experiments/policy_dependency_sync/async_packet_game.py")
        )
        == manifest["implementation"]["game_source_sha256"]
        and manifest["implementation"]["extra_probe_trajectories"] == 0
        and manifest["implementation"][
            "online_scorer_has_environment_target_argument"
        ]
        is False,
        "O12": len(oracle_recovery) == 16
        and metrics["active_median_oracle_terminal_headroom_recovery"] >= 0.75,
        "O13": all(
            math.isfinite(float(row["target_estimation_error"]))
            and len(str(row["graph_trace_sha256"])) == 64
            for row in active_online
        ),
    }
    summary = {
        "experiment_id": "PDSG-OBS-001",
        "status": "confirmation",
        "scientific_scope": manifest["scientific_scope"],
        "seeds": list(seeds),
        "launches": launches,
        "policies": list(manifest["policies"]),
        "oracle_diagnostic_policy": ORACLE_POLICY,
        "metrics": metrics,
        "gates": gates,
        "primary_pass": all(gates.values()),
        "O14_reproduction": "not_evaluated_in_single_run",
        "formal_or_gpu_authorized": False,
    }
    return endpoints, cell_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "run"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.mode == "validate":
        print(
            json.dumps(
                {
                    "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                    "cells": 24,
                    "core_policies": len(manifest["policies"]),
                    "oracle_diagnostics": 1,
                    "seeds": manifest["seeds"]["count"],
                    "expected_endpoints": 24 * 26 * manifest["seeds"]["count"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.output is None:
        raise ValueError("--output is required in run mode")
    started = time.perf_counter()
    endpoints, cells, summary = evaluate(manifest, args.workers)
    sign.write_outputs(
        args.output,
        endpoints,
        cells,
        summary,
        time.perf_counter() - started,
        args.workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

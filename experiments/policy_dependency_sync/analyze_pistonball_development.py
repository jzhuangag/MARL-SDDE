"""Analyze the explicitly non-confirmatory Pistonball GPU development matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EXPECTED_SCHEDULERS = {
    "signed_lyapunov",
    "signed_only",
    "mismatch",
    "complete_burst",
    "no_refresh",
}


def normalized_return_auc(rows: list[dict[str, Any]]) -> float:
    x = np.asarray([float(row["actor_transitions"]) for row in rows])
    y = np.asarray([float(row["return"]) for row in rows])
    if len(rows) < 2 or x[-1] <= x[0] or np.any(np.diff(x) <= 0.0):
        raise ValueError("evaluation transitions must be strictly increasing")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def analyze(paths: Iterable[Path]) -> dict[str, Any]:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_scheduler = {record["scheduler"]: record for record in records}
    if set(by_scheduler) != EXPECTED_SCHEDULERS:
        raise ValueError("development matrix scheduler set is incomplete")
    reference = by_scheduler["signed_lyapunov"]
    invariant_keys = ("seed", "allowed_refresh_units", "launched_actor_transitions")
    if any(
        record[key] != reference[key]
        for record in records
        for key in invariant_keys
    ):
        raise ValueError("matched-resource development invariants differ")
    if any(
        not record["finite"]
        or not record["budget_feasible"]
        or record["remaining_packets_after_drain"] != 0
        for record in records
    ):
        raise ValueError("a development run is invalid")

    rows: dict[str, dict[str, float | int]] = {}
    for scheduler, record in by_scheduler.items():
        evaluations = record["evaluations"]
        rows[scheduler] = {
            "initial_return": float(evaluations[0]["return"]),
            "terminal_return": float(evaluations[-1]["return"]),
            "return_auc": normalized_return_auc(evaluations),
            "spent_refresh_units": int(record["spent_refresh_units"]),
            "allowed_refresh_units": int(record["allowed_refresh_units"]),
            "optional_policy_bytes": int(record["optional_policy_bytes"]),
            "runtime_seconds": float(record["runtime_seconds"]),
        }
    main = rows["signed_lyapunov"]
    comparators = {
        scheduler: row
        for scheduler, row in rows.items()
        if scheduler != "signed_lyapunov"
    }
    best_terminal_name = max(
        comparators, key=lambda scheduler: comparators[scheduler]["terminal_return"]
    )
    best_auc_name = max(
        comparators, key=lambda scheduler: comparators[scheduler]["return_auc"]
    )
    all_finite = all(
        math.isfinite(float(value))
        for row in rows.values()
        for value in row.values()
    )
    return {
        "experiment_id": "PISTONBALL-POLICY-FRESHNESS-GPU-DEVELOPMENT",
        "confirmatory": False,
        "pilot_or_formal_evidence": False,
        "all_finite": all_finite,
        "seed": reference["seed"],
        "rows": rows,
        "best_nonmain_terminal_scheduler": best_terminal_name,
        "best_nonmain_auc_scheduler": best_auc_name,
        "main_minus_best_terminal": float(
            main["terminal_return"] - comparators[best_terminal_name]["terminal_return"]
        ),
        "main_minus_best_auc": float(
            main["return_auc"] - comparators[best_auc_name]["return_auc"]
        ),
        "main_learning_delta": float(
            main["terminal_return"] - main["initial_return"]
        ),
        "decision_rule": (
            "use only to debug learning and choose an outcome-free pilot design; "
            "new seeds and frozen gates are mandatory"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

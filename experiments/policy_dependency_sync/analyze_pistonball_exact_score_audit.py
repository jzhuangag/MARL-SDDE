"""Outcome-blind scale and overhead audit for exact sparse cache scoring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(float(value) for value in values), dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("diagnostics must be finite and nonempty")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
    }


def _reciprocal_median(values: Iterable[float]) -> float | None:
    nonzero = np.asarray(
        [abs(float(value)) for value in values if abs(float(value)) > 1e-12],
        dtype=float,
    )
    if nonzero.size == 0:
        return None
    return float(np.clip(1.0 / float(np.median(nonzero)), 1e-3, 1e8))


def analyze(exact_path: Path, no_refresh_path: Path) -> dict[str, Any]:
    exact = json.loads(exact_path.read_text(encoding="utf-8"))
    no_refresh = json.loads(no_refresh_path.read_text(encoding="utf-8"))
    records = {"exact": exact, "no_refresh": no_refresh}
    for code, record in records.items():
        config = record["config"]
        expected_scheduler = "exact_lyapunov" if code == "exact" else "no_refresh"
        valid = (
            record["scheduler"] == expected_scheduler
            and record["seed"] == 79008
            and record["development_only"]
            and record["finite"]
            and record["budget_feasible"]
            and record["remaining_packets_after_drain"] == 0
            and record["received_packets"] == record["launched_gradient_packets"]
            and record["launched_actor_transitions"] == 1024 * 4 * 20
            and config["network_activation"] == "silu"
            and config["launches"] == 1024
            and config["n_agents"] == 20
            and config["rollout_horizon"] == 4
            and config["max_cycles"] == 125
            and config["batch_size"] == 32
            and config["evaluations"] == 2
            and config["evaluation_episodes"] == 1
            and config["cone_shell"] == 6
            and config["maximum_delay"] == 8
            and config["actor_receipt_step"] == 0.01
            and config["critic_step"] == 0.0003
            and config["budget_rate"] == 0.5
            and config["budget_enforcement"] == "terminal"
            and config["queue_step"] == 1.0
            and config["lyapunov_weight"] == 1.0
            and config["cache_debt_weight"] == 1.0
            and config["random_drop"]
            and config["random_rotate"]
        )
        if not valid:
            raise ValueError(f"{code} run violates the frozen interface")
    exact_grid = tuple(row["actor_transitions"] for row in exact["evaluations"])
    no_grid = tuple(row["actor_transitions"] for row in no_refresh["evaluations"])
    if exact_grid != no_grid or exact["evaluations"][0]["return"] != no_refresh["evaluations"][0]["return"]:
        raise ValueError("runtime controls do not share initialization and grid")

    diagnostics = [
        row["score_diagnostic"]
        for row in exact["launch_trace"]
        if row["score_diagnostic"] is not None
        and row["score_diagnostic"]["best_edge_index"] is not None
    ]
    if not diagnostics:
        raise ValueError("exact scorer produced no diagnostics")
    learning = [float(row["best_edge_learning_index_delta"]) for row in diagnostics]
    cache = [float(row["best_edge_cache_reset_benefit"]) for row in diagnostics]
    queue = [float(row["best_edge_queue_price"]) for row in diagnostics]
    reverse_calls = [float(row["vjp_calls"]) for row in diagnostics]
    for row in diagnostics:
        expected_delta = (
            float(row["best_edge_learning_index_delta"])
            - float(row["best_edge_cache_reset_benefit"])
            + float(row["best_edge_queue_price"])
        )
        observed_delta = float(row["best_edge_index"]) - float(row["null_index"])
        if not math.isclose(expected_delta, observed_delta, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("exact index component identity failed")
        if int(row["vjp_calls"]) != int(row["candidate_count"]) + 1:
            raise ValueError("reverse-call count does not match sparse candidate count")

    recommended_v = _reciprocal_median(learning)
    recommended_beta = _reciprocal_median(cache)
    signed_favorable_fraction = sum(value < 0.0 for value in learning) / len(learning)
    queue_active_fraction = sum(value > 0.0 for value in queue) / len(queue)
    runtime_ratio = float(exact["runtime_seconds"]) / float(no_refresh["runtime_seconds"])
    gates = {
        "E1_nonzero_signed_component": recommended_v is not None,
        "E2_nonzero_cache_component": recommended_beta is not None,
        "E3_signed_favorable": signed_favorable_fraction >= 0.01,
        "E4_queue_active": queue_active_fraction > 0.0,
        "E5_sparse_reverse_bound": max(reverse_calls) <= 8.0,
        "E6_runtime": runtime_ratio <= 3.0,
    }
    passed = all(gates.values())
    return {
        "experiment_id": "PISTONBALL-EXACT-SPARSE-SCORE-AUDIT",
        "confirmatory": False,
        "outcome_blind_calibration": True,
        "pilot_or_formal_evidence": False,
        "passed": passed,
        "decision": (
            "authorize a frozen exact-score controller-headroom design"
            if passed
            else "stop the Pistonball policy-cache mainline"
        ),
        "gates": gates,
        "scored_launches": len(diagnostics),
        "signed_favorable_fraction": signed_favorable_fraction,
        "queue_active_fraction": queue_active_fraction,
        "selected_fraction": sum(
            row["selected_donor"] is not None for row in diagnostics
        )
        / len(diagnostics),
        "learning_drift_delta": _summary(learning),
        "cache_reset_energy": _summary(cache),
        "queue_price": _summary(queue),
        "reverse_calls": _summary(reverse_calls),
        "recommended_learning_weight": recommended_v,
        "recommended_cache_debt_weight": recommended_beta,
        "runtime_ratio_to_no_refresh": runtime_ratio,
        "exact_runtime_seconds": float(exact["runtime_seconds"]),
        "no_refresh_runtime_seconds": float(no_refresh["runtime_seconds"]),
        "exact_refresh_units": int(exact["spent_refresh_units"]),
        "allowed_refresh_units": int(exact["allowed_refresh_units"]),
        "descriptive_returns_not_used_by_gates": {
            code: record["evaluations"] for code, record in records.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--no-refresh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.exact, args.no_refresh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

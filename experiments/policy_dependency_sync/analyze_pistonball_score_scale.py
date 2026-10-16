"""Outcome-blind scale audit for the Pistonball Lyapunov action index."""

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
        raise ValueError("diagnostic sample must be nonempty and finite")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def _reciprocal_positive_median(values: Iterable[float]) -> float | None:
    positive = np.asarray(
        [abs(float(value)) for value in values if abs(float(value)) > 1e-12],
        dtype=float,
    )
    if positive.size == 0:
        return None
    return float(np.clip(1.0 / float(np.median(positive)), 1e-3, 1e8))


def analyze(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    config = record["config"]
    required = {
        "scheduler": "signed_lyapunov",
        "seed": 79004,
        "development_only": True,
        "finite": True,
        "budget_feasible": True,
        "remaining_packets_after_drain": 0,
    }
    for key, expected in required.items():
        if record[key] != expected:
            raise ValueError(f"unexpected {key}: {record[key]!r}")
    expected_config = {
        "launches": 2048,
        "n_agents": 20,
        "rollout_horizon": 4,
        "actor_receipt_step": 0.01,
        "critic_step": 0.0003,
        "budget_rate": 0.5,
        "budget_enforcement": "terminal",
        "queue_step": 1.0,
        "maximum_delay": 8,
        "lyapunov_weight": 100000.0,
        "cache_debt_weight": 100000.0,
    }
    for key, expected in expected_config.items():
        if config[key] != expected:
            raise ValueError(f"unexpected config {key}: {config[key]!r}")

    diagnostics = [
        row["score_diagnostic"]
        for row in record["launch_trace"]
        if row["score_diagnostic"] is not None
        and row["score_diagnostic"]["best_edge_index"] is not None
    ]
    if not diagnostics:
        raise ValueError("no signed edge diagnostics were recorded")
    learning_weight = float(config["lyapunov_weight"])
    cache_weight = float(config["cache_debt_weight"])
    raw_learning = [
        float(row["best_edge_learning_index_delta"]) / learning_weight
        for row in diagnostics
    ]
    raw_cache = [
        float(row["best_edge_cache_reset_benefit"]) / cache_weight
        for row in diagnostics
    ]
    queue_prices = [float(row["best_edge_queue_price"]) for row in diagnostics]
    net_deltas = [
        float(row["best_edge_index"]) - float(row["null_index"])
        for row in diagnostics
    ]
    for diagnostic, net_delta in zip(diagnostics, net_deltas):
        reconstructed = (
            float(diagnostic["best_edge_learning_index_delta"])
            - float(diagnostic["best_edge_cache_reset_benefit"])
            + float(diagnostic["best_edge_queue_price"])
        )
        if not math.isclose(net_delta, reconstructed, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError("action-index component identity failed")

    recommended_learning_weight = _reciprocal_positive_median(raw_learning)
    recommended_cache_weight = _reciprocal_positive_median(raw_cache)
    selected = sum(row["selected_donor"] is not None for row in diagnostics)
    signed_favorable = sum(value < 0.0 for value in raw_learning)
    queue_active = sum(value > 0.0 for value in queue_prices)
    learning_component_nonzero = recommended_learning_weight is not None
    cache_component_nonzero = recommended_cache_weight is not None
    queue_component_active = queue_active > 0
    scale_audit_passed = (
        learning_component_nonzero
        and cache_component_nonzero
        and queue_component_active
    )
    return {
        "experiment_id": "PISTONBALL-LYAPUNOV-SCORE-SCALE-AUDIT",
        "confirmatory": False,
        "outcome_blind_calibration": True,
        "pilot_or_formal_evidence": False,
        "seed": int(record["seed"]),
        "scored_launches": len(diagnostics),
        "selected_fraction_under_legacy_scale": selected / len(diagnostics),
        "signed_favorable_fraction": signed_favorable / len(diagnostics),
        "queue_active_fraction": queue_active / len(diagnostics),
        "learning_component_nonzero": learning_component_nonzero,
        "cache_component_nonzero": cache_component_nonzero,
        "queue_component_active": queue_component_active,
        "scale_audit_passed": scale_audit_passed,
        "decision": (
            "authorize an equal-resource controller-headroom matrix"
            if scale_audit_passed
            else "stop before controller headroom and repair the differentiable score interface"
        ),
        "raw_learning_drift_delta": _summary(raw_learning),
        "raw_cache_reset_energy": _summary(raw_cache),
        "queue_price": _summary(queue_prices),
        "legacy_net_edge_minus_null": _summary(net_deltas),
        "recommended_learning_weight": recommended_learning_weight,
        "recommended_cache_debt_weight": recommended_cache_weight,
        "calibration_rule": (
            "reciprocal median nonzero absolute raw component, clipped to [1e-3,1e8]"
        ),
        "descriptive_return_path": record["evaluations"],
        "spent_refresh_units": int(record["spent_refresh_units"]),
        "allowed_refresh_units": int(record["allowed_refresh_units"]),
        "optional_policy_bytes": int(record["optional_policy_bytes"]),
        "runtime_seconds": float(record["runtime_seconds"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

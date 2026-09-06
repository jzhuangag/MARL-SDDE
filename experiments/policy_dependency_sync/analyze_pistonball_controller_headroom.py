"""Validate the outcome-separated smooth Pistonball controller headroom matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SEEDS = (79006, 79007)
EQUAL_CODES = (
    "signed",
    "signed_only",
    "cache",
    "mismatch_terminal",
    "complete_terminal",
    "age_prefix",
    "mismatch_prefix",
    "round_prefix",
    "static_prefix",
    "no_refresh",
)
ALL_CODES = EQUAL_CODES + ("full_refresh",)
STRONG_CODES = (
    "cache",
    "mismatch_terminal",
    "complete_terminal",
    "age_prefix",
    "mismatch_prefix",
    "round_prefix",
    "static_prefix",
    "no_refresh",
)
SCHEDULER_BY_CODE = {
    "signed": "signed_lyapunov",
    "signed_only": "signed_only",
    "cache": "cache_lyapunov",
    "mismatch_terminal": "mismatch",
    "complete_terminal": "complete_burst",
    "age_prefix": "age",
    "mismatch_prefix": "mismatch",
    "round_prefix": "round_robin",
    "static_prefix": "static_chain",
    "no_refresh": "no_refresh",
    "full_refresh": "complete_burst",
}


def _auc(record: dict[str, Any]) -> float:
    evaluations = record["evaluations"]
    x = np.asarray([float(row["actor_transitions"]) for row in evaluations])
    y = np.asarray([float(row["return"]) for row in evaluations])
    if x.size < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid evaluation curve")
    if not np.all(np.diff(x) > 0.0) or x[-1] <= 0.0:
        raise ValueError("evaluation transition grid must be strictly increasing")
    return float(np.trapezoid(y, x / x[-1]))


def _mean(values: Iterable[float]) -> float:
    values = tuple(float(value) for value in values)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("aggregate values must be finite and nonempty")
    return float(np.mean(values))


def analyze(paths: Iterable[Path]) -> dict[str, Any]:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_key = {
        (int(record["seed"]), str(record["headroom_code"])): record
        for record in records
    }
    expected = {(seed, code) for seed in SEEDS for code in ALL_CODES}
    if set(by_key) != expected:
        raise ValueError("headroom seed/configuration matrix is incomplete")

    per_seed: dict[str, Any] = {}
    invariant_passed = True
    for seed in SEEDS:
        seed_records = {code: by_key[(seed, code)] for code in ALL_CODES}
        grids = {
            tuple(int(row["actor_transitions"]) for row in record["evaluations"])
            for record in seed_records.values()
        }
        initials = {
            float(record["evaluations"][0]["return"])
            for record in seed_records.values()
        }
        if len(grids) != 1 or len(initials) != 1:
            raise ValueError("matched transition grid or initialization failed")
        rows: dict[str, Any] = {}
        for code, record in seed_records.items():
            config = record["config"]
            equal_resource = code != "full_refresh"
            expected_rate = 0.5 if equal_resource else 19.0
            expected_enforcement = (
                "prefix"
                if code.endswith("_prefix")
                or code in {"no_refresh", "full_refresh"}
                else "terminal"
            )
            valid = (
                record["scheduler"] == SCHEDULER_BY_CODE[code]
                and record["development_only"]
                and record["finite"]
                and record["budget_feasible"]
                and record["remaining_packets_after_drain"] == 0
                and record["received_packets"] == record["launched_gradient_packets"]
                and record["launched_actor_transitions"] == 4096 * 4 * 20
                and config["launches"] == 4096
                and config["n_agents"] == 20
                and config["rollout_horizon"] == 4
                and config["network_activation"] == "silu"
                and config["actor_receipt_step"] == 0.01
                and config["critic_step"] == 0.0003
                and config["maximum_delay"] == 8
                and config["cone_shell"] == (20 if code == "full_refresh" else 6)
                and config["budget_rate"] == expected_rate
                and config["budget_enforcement"] == expected_enforcement
                and config["queue_step"] == 1.0
                and config["lyapunov_weight"] == 100000000.0
                and math.isclose(
                    float(config["cache_debt_weight"]),
                    83650.1230871728,
                    rel_tol=1e-12,
                )
            )
            expected_allowance = 2048 if equal_resource else 77824
            valid = valid and record["allowed_refresh_units"] == expected_allowance
            if code == "no_refresh":
                valid = valid and record["spent_refresh_units"] == 0
            invariant_passed = invariant_passed and valid
            rows[code] = {
                "terminal_return": float(record["evaluations"][-1]["return"]),
                "return_auc": _auc(record),
                "refresh_units": int(record["spent_refresh_units"]),
                "optional_policy_bytes": int(record["optional_policy_bytes"]),
                "runtime_seconds": float(record["runtime_seconds"]),
                "budget_enforcement": str(config["budget_enforcement"]),
                "valid": valid,
            }
        per_seed[str(seed)] = rows

    aggregates: dict[str, Any] = {}
    for code in ALL_CODES:
        aggregates[code] = {
            "mean_terminal_return": _mean(
                per_seed[str(seed)][code]["terminal_return"] for seed in SEEDS
            ),
            "mean_return_auc": _mean(
                per_seed[str(seed)][code]["return_auc"] for seed in SEEDS
            ),
            "mean_refresh_units": _mean(
                per_seed[str(seed)][code]["refresh_units"] for seed in SEEDS
            ),
            "mean_optional_policy_bytes": _mean(
                per_seed[str(seed)][code]["optional_policy_bytes"] for seed in SEEDS
            ),
            "mean_runtime_seconds": _mean(
                per_seed[str(seed)][code]["runtime_seconds"] for seed in SEEDS
            ),
        }

    best_strong_terminal = max(
        STRONG_CODES, key=lambda code: aggregates[code]["mean_terminal_return"]
    )
    best_strong_auc = max(
        STRONG_CODES, key=lambda code: aggregates[code]["mean_return_auc"]
    )
    freshness_deltas = [
        per_seed[str(seed)]["full_refresh"]["terminal_return"]
        - per_seed[str(seed)]["no_refresh"]["terminal_return"]
        for seed in SEEDS
    ]
    strong_deltas = [
        per_seed[str(seed)]["signed"]["terminal_return"]
        - per_seed[str(seed)][best_strong_terminal]["terminal_return"]
        for seed in SEEDS
    ]
    freshness_headroom = _mean(freshness_deltas)
    strong_gain = _mean(strong_deltas)
    signed_vs_cache = (
        aggregates["signed"]["mean_terminal_return"]
        - aggregates["cache"]["mean_terminal_return"]
    )
    signed_vs_learning_only = (
        aggregates["signed"]["mean_terminal_return"]
        - aggregates["signed_only"]["mean_terminal_return"]
    )
    auc_gain = (
        aggregates["signed"]["mean_return_auc"]
        - aggregates[best_strong_auc]["mean_return_auc"]
    )
    normalized_strong_gain = (
        strong_gain / freshness_headroom if freshness_headroom > 0.0 else -math.inf
    )
    signed_spend_nontrivial = all(
        0.05 * 2048
        <= per_seed[str(seed)]["signed"]["refresh_units"]
        <= 2048
        for seed in SEEDS
    )
    runtime_ratio = (
        aggregates["signed"]["mean_runtime_seconds"]
        / aggregates["no_refresh"]["mean_runtime_seconds"]
    )
    gates = {
        "H1_invariants": invariant_passed,
        "H2_freshness_headroom": (
            freshness_headroom >= 1.0 and min(freshness_deltas) > 0.0
        ),
        "H3_nontrivial_signed_spend": signed_spend_nontrivial,
        "H4_strong_terminal_gain": strong_gain >= 0.5 and min(strong_deltas) > 0.0,
        "H5_normalized_headroom": normalized_strong_gain >= 0.10,
        "H6_component_value": (
            signed_vs_cache >= 0.25 and signed_vs_learning_only >= 0.25
        ),
        "H7_auc_no_harm": auc_gain >= 0.0,
        "H8_runtime": runtime_ratio <= 2.0,
    }
    passed = all(gates.values())
    return {
        "experiment_id": "PISTONBALL-CONTROLLER-HEADROOM-DEVELOPMENT",
        "confirmatory": False,
        "pilot_or_formal_evidence": False,
        "passed": passed,
        "decision": (
            "authorize a separately preregistered multi-seed efficacy pilot"
            if passed
            else "stop this controller configuration before an efficacy pilot"
        ),
        "gates": gates,
        "best_strong_terminal_code": best_strong_terminal,
        "best_strong_auc_code": best_strong_auc,
        "freshness_deltas": freshness_deltas,
        "mean_freshness_headroom": freshness_headroom,
        "signed_minus_best_strong_by_seed": strong_deltas,
        "mean_signed_minus_best_strong": strong_gain,
        "normalized_strong_gain": normalized_strong_gain,
        "signed_minus_cache": signed_vs_cache,
        "signed_minus_signed_only": signed_vs_learning_only,
        "signed_minus_best_strong_auc": auc_gain,
        "signed_runtime_ratio_to_no_refresh": runtime_ratio,
        "aggregates": aggregates,
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

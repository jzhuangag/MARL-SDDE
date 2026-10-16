"""Run the frozen fresh-seed causal LSFF conditional-risk confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .freshness_headroom import (
    best_periodic_refresh_value,
    causal_resource_schedule_fast,
    markov_regime_path,
    oracle_refresh_value,
    refresh_value,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if (array <= 0.0).any() or not np.isfinite(array).all():
        raise ValueError("geometric mean inputs must be finite and positive")
    return float(np.exp(np.mean(np.log(array))))


def _validate(config: dict[str, Any]) -> None:
    required = {
        "experiment_id",
        "horizon",
        "seeds",
        "high_prevalences",
        "persistence_values",
        "high_risk_multipliers",
        "fresh_variance_ratios",
        "refresh_fractions",
        "risk_tradeoff",
        "gates",
    }
    if set(config) != required:
        raise ValueError("confirmation config keys do not match the frozen schema")
    if len(config["seeds"]) != 64 or len(config["seeds"]) != len(set(config["seeds"])):
        raise ValueError("confirmation requires 64 unique seeds")
    if any(91001 <= int(seed) <= 91064 for seed in config["seeds"]):
        raise ValueError("confirmation seeds overlap development")
    if config["risk_tradeoff"] != 4.0:
        raise ValueError("the development-selected tradeoff must remain frozen")


def _rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    horizon = int(config["horizon"])
    rows: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        for prevalence in config["high_prevalences"]:
            for persistence in config["persistence_values"]:
                regimes = markov_regime_path(
                    horizon=horizon,
                    high_prevalence=float(prevalence),
                    persistence=float(persistence),
                    seed=int(seed),
                )
                for multiplier in config["high_risk_multipliers"]:
                    stale_risk = np.where(regimes, float(multiplier), 1.0)
                    never_risk = float(np.sum(stale_risk))
                    for variance in config["fresh_variance_ratios"]:
                        values = refresh_value(stale_risk, float(variance))
                        for fraction in config["refresh_fractions"]:
                            maximum_count = round(horizon * float(fraction))
                            causal = causal_resource_schedule_fast(
                                stale_risk,
                                fresh_variance=float(variance),
                                maximum_refresh_count=maximum_count,
                                average_refresh_budget=float(fraction),
                                risk_tradeoff=float(config["risk_tradeoff"]),
                            )
                            periodic_risk = never_risk - best_periodic_refresh_value(
                                values, causal.refresh_count
                            )
                            oracle_risk = never_risk - oracle_refresh_value(
                                values, causal.refresh_count
                            )
                            available = periodic_risk - oracle_risk
                            capture = (
                                (periodic_risk - causal.incurred_risk) / available
                                if available > 1e-12
                                else 0.0
                            )
                            rows.append(
                                {
                                    "seed": int(seed),
                                    "high_prevalence": float(prevalence),
                                    "persistence": float(persistence),
                                    "high_risk_multiplier": float(multiplier),
                                    "fresh_variance_ratio": float(variance),
                                    "refresh_fraction_budget": float(fraction),
                                    "maximum_refresh_count": maximum_count,
                                    "actual_refresh_count": causal.refresh_count,
                                    "budget_utilization": causal.refresh_count / maximum_count,
                                    "causal_risk": causal.incurred_risk,
                                    "periodic_same_count_risk": periodic_risk,
                                    "oracle_same_count_risk": oracle_risk,
                                    "causal_over_periodic_ratio": causal.incurred_risk
                                    / periodic_risk,
                                    "oracle_headroom_captured": capture,
                                    "final_resource_debt": causal.final_resource_debt,
                                    "maximum_resource_debt": causal.maximum_resource_debt,
                                }
                            )
    return rows


def _summary(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    stationary = [row for row in rows if row["high_risk_multiplier"] == 1.0]
    dynamic = [row for row in rows if row["high_risk_multiplier"] > 1.0]
    dynamic_ratio = _geometric_mean(
        [row["causal_over_periodic_ratio"] for row in dynamic]
    )
    better_fraction = float(
        np.mean([row["causal_over_periodic_ratio"] < 1.0 for row in dynamic])
    )
    median_capture = float(
        np.median([row["oracle_headroom_captured"] for row in dynamic])
    )
    utilization = float(np.mean([row["budget_utilization"] for row in rows]))
    by_persistence = {
        str(value): _geometric_mean(
            [
                row["causal_over_periodic_ratio"]
                for row in dynamic
                if row["persistence"] == value
            ]
        )
        for value in config["persistence_values"]
    }
    gates = {
        "C1_finite_and_ordered": all(
            math.isfinite(value)
            for row in rows
            for value in (
                row["causal_risk"],
                row["periodic_same_count_risk"],
                row["oracle_same_count_risk"],
            )
        )
        and all(
            row["oracle_same_count_risk"]
            <= row["periodic_same_count_risk"] + 1e-9
            for row in rows
        ),
        "C2_stationary_alignment": max(
            abs(row["causal_over_periodic_ratio"] - 1.0) for row in stationary
        )
        <= config["gates"]["stationary_max_absolute_ratio_error"],
        "C3_dynamic_aggregate": dynamic_ratio
        <= config["gates"]["dynamic_max_geometric_ratio"],
        "C4_dynamic_direction": better_fraction
        >= config["gates"]["minimum_better_fraction"],
        "C5_oracle_capture": median_capture
        >= config["gates"]["minimum_median_oracle_capture"],
        "C6_persistence_robustness": all(
            ratio <= config["gates"]["persistence_max_geometric_ratio"]
            for ratio in by_persistence.values()
        ),
        "C7_budget_utilization": utilization
        >= config["gates"]["minimum_mean_budget_utilization"]
        and all(
            row["actual_refresh_count"] <= row["maximum_refresh_count"]
            for row in rows
        ),
    }
    return {
        "experiment_id": config["experiment_id"],
        "scope": "fresh-seed conditional-risk confirmation; not RL evidence",
        "row_count": len(rows),
        "dynamic_geometric_causal_over_periodic_ratio": dynamic_ratio,
        "dynamic_better_fraction": better_fraction,
        "dynamic_median_oracle_headroom_captured": median_capture,
        "mean_budget_utilization": utilization,
        "stationary_max_absolute_ratio_error": max(
            abs(row["causal_over_periodic_ratio"] - 1.0) for row in stationary
        ),
        "by_persistence": by_persistence,
        "gates": gates,
        "all_scientific_gates_pass": all(gates.values()),
        "gpu_authorized": False,
    }


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate(config)
    rows = _rows(config)
    rows.sort(
        key=lambda row: tuple(
            row[key]
            for key in (
                "seed",
                "high_prevalence",
                "persistence",
                "high_risk_multiplier",
                "fresh_variance_ratio",
                "refresh_fraction_budget",
            )
        )
    )
    summary = _summary(config, rows)
    summary["config_sha256"] = _sha256(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_path = output_dir / "rows.csv"
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["rows_sha256"] = _sha256(row_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _validate(config)
    if args.validate:
        print(json.dumps({"valid": True, "config_sha256": _sha256(args.config)}))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --validate is used")
    print(json.dumps(run(args.config, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

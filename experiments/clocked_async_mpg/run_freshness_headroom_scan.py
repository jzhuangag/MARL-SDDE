"""Run the frozen equal-cost freshness oracle-headroom feasibility scan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .freshness_headroom import equal_cost_headroom, markov_regime_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if (array <= 0.0).any() or not np.isfinite(array).all():
        raise ValueError("geometric mean inputs must be finite and positive")
    return float(np.exp(np.mean(np.log(array))))


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "experiment_id",
        "horizon",
        "seeds",
        "high_prevalences",
        "persistence_values",
        "high_risk_multipliers",
        "fresh_variance_ratios",
        "refresh_fractions",
        "gates",
    }
    if set(config) != required:
        raise ValueError("config keys do not match the frozen schema")
    if config["horizon"] <= 0 or len(config["seeds"]) != len(set(config["seeds"])):
        raise ValueError("invalid horizon or duplicate seeds")
    if 1.0 not in config["high_risk_multipliers"]:
        raise ValueError("stationary multiplier-one controls are mandatory")
    for fraction in config["refresh_fractions"]:
        count = round(config["horizon"] * fraction)
        if count <= 0 or count >= config["horizon"]:
            raise ValueError("refresh fractions must yield interior exact budgets")


def _rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    horizon = int(config["horizon"])
    output: list[dict[str, Any]] = []
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
                    for variance_ratio in config["fresh_variance_ratios"]:
                        for fraction in config["refresh_fractions"]:
                            refresh_count = round(horizon * float(fraction))
                            result = equal_cost_headroom(
                                stale_risk,
                                fresh_variance=float(variance_ratio),
                                refresh_count=refresh_count,
                            )
                            output.append(
                                {
                                    "seed": int(seed),
                                    "high_prevalence": float(prevalence),
                                    "persistence": float(persistence),
                                    "high_risk_multiplier": float(multiplier),
                                    "fresh_variance_ratio": float(variance_ratio),
                                    "refresh_fraction": float(fraction),
                                    "refresh_count": refresh_count,
                                    "realized_high_fraction": float(np.mean(regimes)),
                                    "never_refresh_risk": result.never_refresh_risk,
                                    "always_refresh_risk": result.always_refresh_risk,
                                    "periodic_risk": result.periodic_risk,
                                    "oracle_risk": result.oracle_risk,
                                    "oracle_over_periodic_ratio": result.oracle_over_periodic_ratio,
                                    "relative_oracle_improvement": result.relative_oracle_improvement,
                                }
                            )
    return output


def _summarize(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    stationary = [row for row in rows if row["high_risk_multiplier"] == 1.0]
    dynamic = [row for row in rows if row["high_risk_multiplier"] > 1.0]
    high_contrast = [row for row in rows if row["high_risk_multiplier"] >= 4.0]
    by_persistence: dict[str, Any] = {}
    for persistence in config["persistence_values"]:
        selected = [row for row in dynamic if row["persistence"] == persistence]
        by_persistence[str(persistence)] = {
            "geometric_oracle_over_periodic_ratio": _geometric_mean(
                [row["oracle_over_periodic_ratio"] for row in selected]
            ),
            "mean_relative_oracle_improvement": float(
                np.mean([row["relative_oracle_improvement"] for row in selected])
            ),
        }
    gates = {
        "G1_finite_and_ordered": all(
            math.isfinite(value)
            for row in rows
            for value in (
                row["periodic_risk"],
                row["oracle_risk"],
                row["oracle_over_periodic_ratio"],
            )
        )
        and all(row["oracle_risk"] <= row["periodic_risk"] + 1e-9 for row in rows),
        "G2_stationary_zero_headroom": max(
            row["relative_oracle_improvement"] for row in stationary
        )
        <= config["gates"]["stationary_max_improvement"],
        "G3_dynamic_aggregate_headroom": _geometric_mean(
            [row["oracle_over_periodic_ratio"] for row in dynamic]
        )
        <= config["gates"]["dynamic_max_geometric_ratio"],
        "G4_dynamic_directional_coverage": float(
            np.mean(
                [
                    row["relative_oracle_improvement"]
                    >= config["gates"]["cell_improvement_threshold"]
                    for row in dynamic
                ]
            )
        )
        >= config["gates"]["minimum_directional_fraction"],
        "G5_high_contrast_headroom": _geometric_mean(
            [row["oracle_over_periodic_ratio"] for row in high_contrast]
        )
        <= config["gates"]["high_contrast_max_geometric_ratio"],
        "G6_each_persistence_stratum": all(
            cell["geometric_oracle_over_periodic_ratio"]
            <= config["gates"]["persistence_max_geometric_ratio"]
            for cell in by_persistence.values()
        ),
        "G7_exact_equal_refresh_charge": all(
            row["refresh_count"]
            == round(config["horizon"] * row["refresh_fraction"])
            for row in rows
        ),
    }
    directional_fraction = float(
        np.mean(
            [
                row["relative_oracle_improvement"]
                >= config["gates"]["cell_improvement_threshold"]
                for row in dynamic
            ]
        )
    )
    return {
        "experiment_id": config["experiment_id"],
        "scope": "outcome-free conditional-risk feasibility; not RL evidence",
        "row_count": len(rows),
        "stationary_row_count": len(stationary),
        "dynamic_row_count": len(dynamic),
        "dynamic_geometric_oracle_over_periodic_ratio": _geometric_mean(
            [row["oracle_over_periodic_ratio"] for row in dynamic]
        ),
        "dynamic_mean_relative_oracle_improvement": float(
            np.mean([row["relative_oracle_improvement"] for row in dynamic])
        ),
        "dynamic_directional_fraction": directional_fraction,
        "high_contrast_geometric_oracle_over_periodic_ratio": _geometric_mean(
            [row["oracle_over_periodic_ratio"] for row in high_contrast]
        ),
        "stationary_max_relative_oracle_improvement": max(
            row["relative_oracle_improvement"] for row in stationary
        ),
        "by_persistence": by_persistence,
        "gates": gates,
        "all_mandatory_gates_pass": all(gates.values()),
        "next_step_authorized": "causal_lsff_cpu_development"
        if all(gates.values())
        else "stop_freshness_mainline",
        "gpu_authorized": False,
    }


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    rows = _rows(config)
    rows.sort(
        key=lambda row: (
            row["seed"],
            row["high_prevalence"],
            row["persistence"],
            row["high_risk_multiplier"],
            row["fresh_variance_ratio"],
            row["refresh_fraction"],
        )
    )
    summary = _summarize(config, rows)
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
    _validate_config(config)
    if args.validate:
        print(json.dumps({"valid": True, "config_sha256": _sha256(args.config)}))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --validate is used")
    summary = run(args.config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

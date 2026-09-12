"""Development-only causal LSFF scan on the frozen Markov risk paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


TRADEOFFS = (0.25, 0.5, 1.0, 2.0, 4.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(array))))


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
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
                    for variance in config["fresh_variance_ratios"]:
                        values = refresh_value(stale_risk, float(variance))
                        never_risk = float(np.sum(stale_risk))
                        for fraction in config["refresh_fractions"]:
                            maximum_count = round(horizon * float(fraction))
                            for tradeoff in TRADEOFFS:
                                causal = causal_resource_schedule_fast(
                                    stale_risk,
                                    fresh_variance=float(variance),
                                    maximum_refresh_count=maximum_count,
                                    average_refresh_budget=float(fraction),
                                    risk_tradeoff=tradeoff,
                                )
                                periodic_value = best_periodic_refresh_value(
                                    values, causal.refresh_count
                                )
                                oracle_value = oracle_refresh_value(
                                    values, causal.refresh_count
                                )
                                periodic_risk = never_risk - periodic_value
                                oracle_risk = never_risk - oracle_value
                                denominator = periodic_risk - oracle_risk
                                captured = (
                                    (periodic_risk - causal.incurred_risk) / denominator
                                    if denominator > 1e-12
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
                                        "risk_tradeoff": tradeoff,
                                        "maximum_refresh_count": maximum_count,
                                        "actual_refresh_count": causal.refresh_count,
                                        "budget_utilization": causal.refresh_count / maximum_count,
                                        "causal_risk": causal.incurred_risk,
                                        "periodic_same_count_risk": periodic_risk,
                                        "oracle_same_count_risk": oracle_risk,
                                        "causal_over_periodic_ratio": causal.incurred_risk
                                        / periodic_risk,
                                        "oracle_headroom_captured": captured,
                                        "final_resource_debt": causal.final_resource_debt,
                                        "maximum_resource_debt": causal.maximum_resource_debt,
                                    }
                                )
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
                "risk_tradeoff",
            )
        )
    )
    dynamic = [row for row in rows if row["high_risk_multiplier"] > 1.0]
    by_tradeoff: dict[str, Any] = {}
    for tradeoff in TRADEOFFS:
        selected = [row for row in dynamic if row["risk_tradeoff"] == tradeoff]
        by_tradeoff[str(tradeoff)] = {
            "geometric_causal_over_periodic_ratio": _geometric_mean(
                [row["causal_over_periodic_ratio"] for row in selected]
            ),
            "mean_budget_utilization": float(
                np.mean([row["budget_utilization"] for row in selected])
            ),
            "fraction_better_than_periodic": float(
                np.mean([row["causal_over_periodic_ratio"] < 1.0 for row in selected])
            ),
            "median_oracle_headroom_captured": float(
                np.median([row["oracle_headroom_captured"] for row in selected])
            ),
        }
    eligible = [
        (float(key), value)
        for key, value in by_tradeoff.items()
        if value["mean_budget_utilization"] >= 0.9
    ]
    selected_tradeoff = min(
        eligible,
        key=lambda item: (item[1]["geometric_causal_over_periodic_ratio"], item[0]),
    )[0]
    summary = {
        "scope": "controller development on prior feasibility paths; not formal evidence",
        "config_sha256": _sha256(config_path),
        "tradeoffs": list(TRADEOFFS),
        "row_count": len(rows),
        "by_tradeoff": by_tradeoff,
        "development_selected_tradeoff": selected_tradeoff,
        "selection_rule": "minimum dynamic geometric risk ratio among mean utilization >= 0.9; ties choose smaller",
        "gpu_authorized": False,
    }
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Outcome-free upper-bound scan for action-dependent geometry sensing."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .clocked_optimism_phase import rotational_optimism_threshold
from .controlled_sensing_upper_bound import (
    best_periodic_fixed_cost,
    exact_phase_cost,
    phase_log_multiplier_table,
    solve_perfect_observation_bound,
)


EXPECTED_CONFIG_SHA256 = (
    "ca199339a419ef66e556769fdf4eea7e0dc3d0771fa19770b417f7d7813f523a"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    if _sha256(path).lower() != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("LCO-U0 configuration hash mismatch")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment"] != "LCO-U0-ANALYTIC-UPPER-BOUND":
        raise RuntimeError("unexpected analytic experiment")
    if config["formal_evidence"]:
        raise RuntimeError("upper-bound scan cannot be formal evidence")
    ages = [int(value) for value in config["maximum_ages"]]
    if ages != sorted(ages) or len(ages) < 2:
        raise RuntimeError("maximum ages must be increasing")
    return config


def _specifications(config: dict[str, Any]) -> list[dict[str, float]]:
    return [
        {
            "step": float(step),
            "arrival": float(arrival),
            "persistence": float(persistence),
            "rotation_fraction": float(rotation_fraction),
            "budget": float(budget),
        }
        for step in config["normalized_steps"]
        for arrival in config["first_agent_arrival_probabilities"]
        for persistence in config["phase_persistence"]
        for rotation_fraction in config["rotation_stationary_fractions"]
        for budget in config["optimism_budgets"]
    ]


def _run_one(spec: dict[str, float], config: dict[str, Any]) -> dict[str, Any]:
    table = phase_log_multiplier_table(spec["step"], spec["arrival"])
    fixed_cost, fixed_rate = best_periodic_fixed_cost(
        table,
        rotation_fraction=spec["rotation_fraction"],
        optimism_budget=spec["budget"],
        period=int(config["fixed_schedule_period"]),
    )
    oracle_cost, oracle_rate = exact_phase_cost(
        table,
        rotation_fraction=spec["rotation_fraction"],
        optimism_budget=spec["budget"],
    )
    age_results = {
        str(age): solve_perfect_observation_bound(
            table,
            persistence=spec["persistence"],
            rotation_fraction=spec["rotation_fraction"],
            optimism_budget=spec["budget"],
            maximum_age=int(age),
        )
        for age in config["maximum_ages"]
    }
    maximum_age = str(max(int(age) for age in config["maximum_ages"]))
    selected = age_results[maximum_age]
    sorted_ages = sorted(int(age) for age in config["maximum_ages"])
    previous = age_results[str(sorted_ages[-2])]
    upper_gain = fixed_cost - selected.optimal_log_cost
    exact_headroom = fixed_cost - oracle_cost
    capture = upper_gain / exact_headroom if exact_headroom > 1e-15 else 1.0
    threshold = rotational_optimism_threshold(spec["step"])
    margin = float(config["dynamic_separation_margin"])
    separated_dynamic = bool(
        0.0 < spec["rotation_fraction"] < 1.0
        and spec["budget"] / spec["rotation_fraction"] >= threshold + margin
        and spec["budget"] <= threshold - margin
    )
    return {
        **spec,
        "fixed_log_cost": fixed_cost,
        "fixed_call_rate": fixed_rate,
        "exact_phase_log_cost": oracle_cost,
        "exact_phase_call_rate": oracle_rate,
        "perfect_sensing_log_cost": selected.optimal_log_cost,
        "perfect_sensing_call_rate": selected.call_rate,
        "perfect_sensing_upper_gain": upper_gain,
        "exact_phase_headroom": exact_headroom,
        "exact_headroom_capture": capture,
        "age_convergence_gap": abs(
            selected.optimal_log_cost - previous.optimal_log_cost
        ),
        "belief_tail_at_maximum_age": spec["persistence"] ** int(maximum_age),
        "flow_residual": selected.flow_residual,
        "normalization_residual": selected.normalization_residual,
        "calibration_residual": selected.calibration_residual,
        "solver_status": selected.solver_status,
        "state_count": selected.state_count,
        "separated_dynamic": separated_dynamic,
        "costs_by_maximum_age": {
            age: result.optimal_log_cost for age, result in age_results.items()
        },
    }


def _run_payload(payload: tuple[dict[str, float], dict[str, Any]]) -> dict[str, Any]:
    return _run_one(*payload)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def _summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    dynamic = [row for row in rows if row["separated_dynamic"]]
    gains = np.asarray([row["perfect_sensing_upper_gain"] for row in dynamic])
    low_persistence = min(config["phase_persistence"])
    low_budget = min(config["optimism_budgets"])
    low_persistence_rows = [
        row for row in dynamic if row["persistence"] == low_persistence
    ]
    low_budget_rows = [row for row in dynamic if row["budget"] == low_budget]
    maxima = {
        "flow_residual": max(row["flow_residual"] for row in rows),
        "normalization_residual": max(row["normalization_residual"] for row in rows),
        "calibration_residual": max(row["calibration_residual"] for row in rows),
        "age_convergence_gap": max(row["age_convergence_gap"] for row in rows),
        "budget_overshoot": max(
            row["perfect_sensing_call_rate"] - row["budget"] for row in rows
        ),
        "stationary_potential_call_rate": max(
            row["perfect_sensing_call_rate"]
            for row in rows
            if row["rotation_fraction"] == 0.0
        ),
    }
    ordering_violations = sum(
        not (
            row["exact_phase_log_cost"]
            <= row["perfect_sensing_log_cost"] + 1e-9
            and row["perfect_sensing_log_cost"] <= row["fixed_log_cost"] + 1e-9
        )
        for row in rows
    )
    thresholds = config["survival_gates"]
    metrics = {
        "mean_dynamic_upper_gain": float(np.mean(gains)),
        "dynamic_fraction_above_0_02": float(np.mean(gains >= 0.02)),
        "low_persistence_mean_upper_gain": _mean(
            low_persistence_rows, "perfect_sensing_upper_gain"
        ),
        "low_budget_mean_upper_gain": _mean(
            low_budget_rows, "perfect_sensing_upper_gain"
        ),
        "median_exact_headroom_capture": float(
            np.median([row["exact_headroom_capture"] for row in dynamic])
        ),
    }
    gates = {
        "U1_solver_and_finite": all(
            row["solver_status"] == 0
            and all(
                math.isfinite(row[key])
                for key in (
                    "fixed_log_cost",
                    "exact_phase_log_cost",
                    "perfect_sensing_log_cost",
                )
            )
            for row in rows
        ),
        "U2_flow": maxima["flow_residual"] <= thresholds["maximum_flow_residual"],
        "U3_normalization": maxima["normalization_residual"]
        <= thresholds["maximum_normalization_residual"],
        "U4_calibration": maxima["calibration_residual"]
        <= thresholds["maximum_calibration_residual"],
        "U5_age_convergence": maxima["age_convergence_gap"]
        <= thresholds["maximum_age_convergence_gap"],
        "U6_ordering": ordering_violations == 0,
        "U7_mean_headroom": metrics["mean_dynamic_upper_gain"]
        >= thresholds["minimum_mean_dynamic_upper_gain"],
        "U8_broad_headroom": metrics["dynamic_fraction_above_0_02"]
        >= thresholds["minimum_dynamic_fraction_above_0_02"],
        "U9_low_persistence": metrics["low_persistence_mean_upper_gain"]
        >= thresholds["minimum_low_persistence_mean_upper_gain"],
        "U10_low_budget": metrics["low_budget_mean_upper_gain"]
        >= thresholds["minimum_low_budget_mean_upper_gain"],
        "U11_capture": metrics["median_exact_headroom_capture"]
        >= thresholds["minimum_median_exact_headroom_capture"],
        "U12_budget": maxima["budget_overshoot"]
        <= thresholds["maximum_budget_overshoot"],
        "U13_stationary_potential": maxima["stationary_potential_call_rate"]
        <= thresholds["maximum_stationary_potential_call_rate"],
        "U14_analytic_only": not bool(config["formal_evidence"]),
    }
    return {
        "row_count": len(rows),
        "dynamic_cell_count": len(dynamic),
        "metrics": metrics,
        "maxima": maxima,
        "ordering_violations": ordering_violations,
        "gates": gates,
        "active_sensing_survives": all(gates.values()),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "estimate", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = _load_config(args.config)
    specifications = _specifications(config)
    if args.command == "validate":
        print(f"config_sha256={_sha256(args.config)}")
        print("validation=pass")
        return
    if args.command == "estimate":
        print(f"analytic_cells={len(specifications)}")
        print(f"linear_programs={len(specifications) * len(config['maximum_ages'])}")
        return
    if args.output_dir is None:
        raise ValueError("run requires --output-dir")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    payloads = ((specification, config) for specification in specifications)
    if args.workers == 1:
        rows = [_run_payload(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_run_payload, payloads, chunksize=1))
    summary = _summarize(rows, config)
    payload = {
        "experiment": config["experiment"],
        "config_sha256": _sha256(args.config),
        "summary": summary,
        "formal_evidence": False,
        "gpu_authorized": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "summary.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "rows"},
            indent=2,
            sort_keys=True,
        )
    )
    print(f"summary_sha256={_sha256(output)}")


if __name__ == "__main__":
    main()

"""Prospective exact phase-map runner for T-041A."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from hashlib import sha256
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dependence_delay_linear.t035_scalar_phase_theorem import (
    exact_scalar_risk,
)
from experiments.dependence_delay_linear.t037_vector_markov_phase import (
    dual_budget_updates,
    equicorrelated_ar_lag_covariances,
    exact_vector_risk,
)
from experiments.dependence_delay_linear.t038_gaussian_markov_lower_bound import (
    ProbeAction,
    fixed_sequence_minimax_risk,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs" / "t041a_exact_phase_preregistration.json"


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def scenario_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    shared = config["shared_grid"]
    rows: list[dict[str, Any]] = []
    for family, family_config in config["families"].items():
        for rho, (matrix_name, matrix), delay, mixing, step_size in product(
            family_config["rho"],
            shared["matrices"].items(),
            shared["delay"],
            shared["markov_lambda"],
            shared["step_size"],
        ):
            rows.append(
                {
                    "scenario_id": (
                        f"{family}-{rho:g}-{matrix_name}-D{delay}"
                        f"-L{mixing:g}-E{step_size:g}"
                    ),
                    "family": family,
                    "rho": float(rho),
                    "matrix_name": matrix_name,
                    "matrix": matrix,
                    "delay": int(delay),
                    "markov_lambda": float(mixing),
                    "step_size": float(step_size),
                    "initial_scale": float(family_config["initial_scale"]),
                    "message_budget": int(family_config["message_budget"]),
                    "environment_budget": int(family_config["environment_budget"]),
                }
            )
    return rows


def static_validate(config: dict[str, Any]) -> dict[str, Any]:
    scenarios = scenario_rows(config)
    actions = config["action_catalogue"]
    expected = config["expected_workload"]
    if len(scenarios) != expected["scenarios"]:
        raise ValueError("scenario count differs from frozen workload")
    action_count = len(actions["q"]) * len(actions["stride"])
    if action_count != expected["actions_per_scenario"]:
        raise ValueError("action count differs from frozen workload")
    if len(scenarios) * action_count != expected["rows"]:
        raise ValueError("row count differs from frozen workload")
    if config["analysis"]["no_seeds_or_confidence_intervals"] is not True:
        raise ValueError("T-041A must remain analytic")
    if config["formal_or_gpu_authorized"] is not False:
        raise ValueError("GPU/formal execution must remain unauthorized")
    for scenario in scenarios:
        matrix = np.asarray(scenario["matrix"], dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("every matrix must be square")
        if np.min(np.linalg.eigvalsh((matrix + matrix.T) / 2.0)) <= 0.0:
            raise ValueError("every registered drift must be positive definite")
    return {
        "experiment_id": config["experiment_id"],
        "scenarios": len(scenarios),
        "actions": action_count,
        "rows": len(scenarios) * action_count,
        "gpu": False,
    }


def estimate(config: dict[str, Any]) -> dict[str, Any]:
    validated = static_validate(config)
    largest_horizon = 0
    overhead = config["action_catalogue"]["message_overhead"]
    for scenario in scenario_rows(config):
        for q, stride in product(
            config["action_catalogue"]["q"],
            config["action_catalogue"]["stride"],
        ):
            largest_horizon = max(
                largest_horizon,
                dual_budget_updates(
                    message_budget=scenario["message_budget"],
                    environment_budget=scenario["environment_budget"],
                    message_cost=overhead + q,
                    stride=stride,
                    delay=scenario["delay"],
                ),
            )
    return {
        **validated,
        "largest_horizon": largest_horizon,
        "dominant_work": "finite impulse covariance contractions",
        "recommended_hardware": "local CPU",
    }


@lru_cache(maxsize=None)
def cached_fixed_sequence_minimax_risk(
    q: int,
    stride: int,
    updates: int,
    common_variance: float,
    private_variance: float,
    markov_lambda: float,
) -> float:
    return fixed_sequence_minimax_risk(
        actions=[ProbeAction(q=q, stride=stride)] * updates,
        common_variance=common_variance,
        private_variance=private_variance,
        markov_lambda=markov_lambda,
    )


def evaluate_action(
    scenario: dict[str, Any],
    *,
    q: int,
    stride: int,
    overhead: int,
    private_variance_floor: float,
) -> dict[str, Any]:
    drift = np.asarray(scenario["matrix"], dtype=float)
    dimension = drift.shape[0]
    updates = dual_budget_updates(
        message_budget=scenario["message_budget"],
        environment_budget=scenario["environment_budget"],
        message_cost=overhead + q,
        stride=stride,
        delay=scenario["delay"],
    )
    if updates < 1:
        raise ValueError("every registered action must have at least one update")
    lags = equicorrelated_ar_lag_covariances(
        horizon=updates,
        single_agent_covariance=np.eye(dimension),
        q=q,
        rho=scenario["rho"],
        markov_lambda=scenario["markov_lambda"] ** stride,
    )
    risk = exact_vector_risk(
        initial_history=np.full(
            (scenario["delay"] + 1, dimension), scenario["initial_scale"]
        ),
        drift=drift,
        step_size=scenario["step_size"],
        delay=scenario["delay"],
        updates=updates,
        lag_covariances=lags,
    )
    common_variance = scenario["rho"]
    private_variance = max(1.0 - scenario["rho"], private_variance_floor)
    minimax = cached_fixed_sequence_minimax_risk(
        q,
        stride,
        updates,
        common_variance,
        private_variance,
        scenario["markov_lambda"],
    )
    binding = (
        "message"
        if scenario["message_budget"] // (overhead + q)
        <= max(scenario["environment_budget"] - scenario["delay"], 0) // stride
        else "environment"
    )
    scalar_reference = None
    if dimension == 1:
        scalar_reference = exact_scalar_risk(
            initial_error=scenario["initial_scale"],
            mu=float(drift[0, 0]),
            step_size=scenario["step_size"],
            delay=scenario["delay"],
            updates=updates,
            single_variance=1.0,
            q=q,
            rho=scenario["rho"],
            markov_lambda=scenario["markov_lambda"] ** stride,
        )["risk"]
    return {
        "scenario_id": scenario["scenario_id"],
        "family": scenario["family"],
        "matrix_name": scenario["matrix_name"],
        "rho": scenario["rho"],
        "delay": scenario["delay"],
        "markov_lambda": scenario["markov_lambda"],
        "step_size": scenario["step_size"],
        "q": q,
        "stride": stride,
        "updates": updates,
        "binding": binding,
        "risk": float(risk["risk"]),
        "bias_risk": float(risk["bias_risk"]),
        "noise_risk": float(risk["noise_risk"]),
        "spectral_radius": float(risk["spectral_radius"]),
        "fixed_sequence_minimax_risk": minimax,
        "scalar_reference_risk": scalar_reference,
    }


def analyze(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    primary_stride = config["analysis"]["primary_stride"]
    small_q = config["analysis"]["primary_small_q"]
    large_q = config["analysis"]["primary_large_q"]
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(row["scenario_id"], []).append(row)
    comparisons: list[dict[str, Any]] = []
    best_q_support: set[int] = set()
    scalar_errors: list[float] = []
    for scenario_rows_ in by_scenario.values():
        best = min(
            scenario_rows_,
            key=lambda row: (row["risk"], row["q"], row["stride"]),
        )
        best_q_support.add(int(best["q"]))
        primary = {
            int(row["q"]): row
            for row in scenario_rows_
            if row["stride"] == primary_stride
        }
        small = primary[small_q]
        large = primary[large_q]
        comparisons.append(
            {
                "family": small["family"],
                "ratio": large["risk"] / small["risk"],
                "small_binding": small["binding"],
                "large_binding": large["binding"],
            }
        )
        for row in scenario_rows_:
            reference = row["scalar_reference_risk"]
            if reference is not None:
                scalar_errors.append(
                    abs(row["risk"] - reference) / max(abs(reference), 1e-300)
                )
    family = {
        name: [item for item in comparisons if item["family"] == name]
        for name in config["families"]
    }
    speed_fraction = float(
        np.mean([item["ratio"] <= 0.95 for item in family["speedup"]])
    )
    saturation_max_error = float(
        max(abs(item["ratio"] - 1.0) for item in family["saturation"])
    )
    reversal_fraction = float(
        np.mean([item["ratio"] >= 1.05 for item in family["reversal"]])
    )
    positive_cells = sum(item["ratio"] <= 0.95 for item in comparisons)
    no_value_cells = sum(abs(item["ratio"] - 1.0) <= 0.01 for item in comparisons)
    gates = {
        "P1": all(
            np.isfinite(row["risk"])
            and np.isfinite(row["fixed_sequence_minimax_risk"])
            and row["spectral_radius"] < 1.0
            and row["updates"] >= 1
            for row in rows
        ),
        "P2": speed_fraction >= 0.95,
        "P3": saturation_max_error <= 1e-10,
        "P4": reversal_fraction >= 0.90,
        "P5": len(best_q_support) >= 2,
        "P6": all(
            item["small_binding"] == "environment"
            and item["large_binding"] == "environment"
            for name in ("speedup", "saturation")
            for item in family[name]
        )
        and all(
            item["small_binding"] == "message"
            and item["large_binding"] == "message"
            for item in family["reversal"]
        ),
        "P7": max(scalar_errors, default=0.0) <= 1e-10,
        "P8": positive_cells > 0 and no_value_cells > 0,
    }
    return {
        "experiment_id": config["experiment_id"],
        "rows": len(rows),
        "scenarios": len(by_scenario),
        "speedup_directional_fraction": speed_fraction,
        "saturation_max_relative_error": saturation_max_error,
        "reversal_directional_fraction": reversal_fraction,
        "best_q_support": sorted(best_q_support),
        "positive_5pct_cells": int(positive_cells),
        "no_value_1pct_cells": int(no_value_cells),
        "max_scalar_reference_relative_error": max(scalar_errors, default=0.0),
        "gates": gates,
        "all_scientific_gates_pass": all(gates.values()),
    }


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    static_validate(config)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    rows = [
        evaluate_action(
            scenario,
            q=q,
            stride=stride,
            overhead=config["action_catalogue"]["message_overhead"],
            private_variance_floor=config["analysis"][
                "minimax_private_variance_floor"
            ],
        )
        for scenario in scenario_rows(config)
        for q, stride in product(
            config["action_catalogue"]["q"],
            config["action_catalogue"]["stride"],
        )
    ]
    row_path = output / "rows.csv"
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = analyze(rows, config)
    summary["configuration_sha256"] = sha256_file(DEFAULT_CONFIG)
    summary["runner_sha256"] = sha256_file(Path(__file__))
    summary["rows_sha256"] = sha256_file(row_path)
    summary_path = output / "summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "estimate", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "validate":
        result = static_validate(config)
    elif args.mode == "estimate":
        result = estimate(config)
    else:
        if args.output is None:
            raise ValueError("--output is required in run mode")
        result = run(config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

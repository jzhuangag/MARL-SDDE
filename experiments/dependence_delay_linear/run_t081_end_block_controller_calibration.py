"""Run the frozen old-seed T-081 end-block architecture calibration."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.dependence_delay_linear.run_t070a_exact_nonstationary_graph_scan import unit_schedule
from experiments.dependence_delay_linear.run_t071a_sampled_observable_graph_pilot import (
    load_t070_config,
    load_config as load_t071_config,
)
from experiments.dependence_delay_linear.run_t072_dual_use_architecture_calibration import (
    geometric_mean,
    source_rows,
    write_csv,
)
from experiments.dependence_delay_linear.t071_sampled_graph_controller import (
    sample_markov_observations,
    stable_seed,
)
from experiments.dependence_delay_linear.t081_end_block_primal_dual_controller import (
    simulate_end_block_controller,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs/t081_end_block_controller_calibration.json"
T071_ENDPOINTS = ROOT / "experiments/dependence_delay_linear/results/t071a_sampled_observable_graph_pilot/endpoints.csv"
T080_CELLS = ROOT / "experiments/dependence_delay_linear/results/t080_chunked_continuous_static_execution/cells.csv"
DEFAULT_OUTPUT = ROOT / "experiments/dependence_delay_linear/results/t081_end_block_controller_calibration"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def continuous_static_weights() -> dict[str, np.ndarray]:
    with T080_CELLS.open(newline="", encoding="utf-8") as handle:
        return {
            row["cell_id"]: np.asarray(json.loads(row["continuous_static_weights"]), dtype=float)
            for row in csv.DictReader(handle)
        }


def is_primary(row: dict[str, Any]) -> bool:
    return (
        row["schedule_family"] in {"single_switch", "alternating"}
        and float(row["target_scale"]) in {0.3, 0.6}
        and float(row["temporal_correlation"]) == 0.0
    )


def validate(config: dict[str, Any]) -> dict[str, Any]:
    if sha256(T071_ENDPOINTS) != config["source"]["t071_endpoints_sha256"]:
        raise ValueError("T-071A endpoint source hash mismatch")
    if sha256(T080_CELLS) != config["source"]["t080_cells_sha256"]:
        raise ValueError("T-080 continuous-static source hash mismatch")
    if config["source"]["new_scientific_seeds"] is not False:
        raise ValueError("T-081 calibration cannot use new seeds")
    rows = source_rows()
    cells = {row["cell_id"] for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    primary = {row["cell_id"] for row in rows if is_primary(row)}
    static = continuous_static_weights()
    observed = {
        "cells": len(cells),
        "primary_cells": len(primary),
        "old_design_seeds": len(seeds),
        "endpoints": len(rows),
    }
    for key, value in observed.items():
        if value != config["workload"][key]:
            raise ValueError(f"workload mismatch for {key}")
    if set(static) != cells:
        raise ValueError("continuous-static cell coverage mismatch")
    return {"experiment_id": config["experiment_id"], **observed, "scientific_outcome_created": False}


def run_endpoint(payload: tuple[dict[str, Any], dict[str, str], np.ndarray]) -> dict[str, Any]:
    config, source, fixed = payload
    parent = load_t070_config()
    t071 = load_t071_config()
    controller = config["controller"]
    targets = float(source["target_scale"]) * unit_schedule(parent, source["schedule_family"])
    observations, _ = sample_markov_observations(
        targets=targets,
        steps_per_block=t071["model"]["learning_steps_per_block"],
        noise_scale=float(source["noise_scale"]),
        spatial_correlation=float(source["spatial_correlation"]),
        temporal_correlation=float(source["temporal_correlation"]),
        seed=stable_seed(int(source["seed"]), source["cell_id"], "observations"),
    )
    common = dict(
        observations=observations,
        targets=targets,
        initial_parameter=float(source["initial_common_parameter"]),
        gain=t071["model"]["gain"],
        delay=int(source["delay"]),
        decision_blocks=t071["model"]["decision_blocks"],
        drift_weight=controller["drift_weight"],
        variance_weight=controller["variance_weight"],
        safety_slack=controller["safety_slack"],
        certificate_delta=controller["certificate_delta"],
        rho_cap=controller["rho_cap"],
        fingerprint_message_units=controller["fingerprint_message_units_per_decision"],
        mixing_message_units=controller["mixing_message_units_per_decision"],
    )
    observable = simulate_end_block_controller(**common)
    static = simulate_end_block_controller(**common, fixed_weights=fixed)
    local = simulate_end_block_controller(**common, fixed_weights=np.eye(4))
    identity = np.eye(4)
    nonlocal_rate = float(np.mean(
        np.max(np.abs(observable.accepted_weights - identity), axis=2) > 1e-8
    ))
    return {
        **{key: source[key] for key in (
            "cell_id", "schedule_family", "target_scale", "initial_common_parameter",
            "noise_scale", "spatial_correlation", "temporal_correlation", "delay", "seed",
        )},
        "primary_identifiable": is_primary(source),
        "observable_auc": observable.auc_risk,
        "continuous_static_auc": static.auc_risk,
        "local_auc": local.auc_risk,
        "learning_transitions": observable.learning_transitions,
        "extra_probe_transitions": observable.extra_probe_transitions,
        "message_units": observable.message_units,
        "mean_debt": float(np.mean(observable.debt_path)),
        "max_debt": float(np.max(observable.debt_path)),
        "mean_rho_upper": float(np.mean(observable.rho_upper_path)),
        "mean_effective_samples": float(np.mean(observable.effective_samples_path)),
        "mean_iterations_per_recipient_decision": observable.qp_iterations / 24.0,
        "accepted_nonlocal_rate": nonlocal_rate,
    }


def analyze(config: dict[str, Any], endpoints: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in endpoints:
        grouped.setdefault(row["cell_id"], []).append(row)
    cells = []
    for cell_id, rows in sorted(grouped.items()):
        first = rows[0]
        observable = float(np.mean([row["observable_auc"] for row in rows]))
        static = float(np.mean([row["continuous_static_auc"] for row in rows]))
        local = float(np.mean([row["local_auc"] for row in rows]))
        cells.append({
            "cell_id": cell_id,
            **{key: first[key] for key in (
                "schedule_family", "target_scale", "initial_common_parameter", "noise_scale",
                "spatial_correlation", "temporal_correlation", "delay", "primary_identifiable",
            )},
            "mean_observable_auc": observable,
            "mean_continuous_static_auc": static,
            "mean_local_auc": local,
            "observable_to_continuous_static_ratio": observable / static,
            "observable_to_local_ratio": observable / local,
        })

    primary = [row for row in cells if row["primary_identifiable"]]
    stationary = [row for row in cells if row["schedule_family"] == "stationary"]
    low_signal = [row for row in cells if row["schedule_family"] != "stationary"
                  and float(row["target_scale"]) == 0.1
                  and float(row["temporal_correlation"]) == 0.0]
    high_temporal = [row for row in cells if row["schedule_family"] != "stationary"
                     and float(row["target_scale"]) in {0.3, 0.6}
                     and float(row["temporal_correlation"]) == 0.9]

    def ratio(rows: list[dict[str, Any]], key: str) -> float:
        return geometric_mean([float(row[key]) for row in rows])

    primary_ratio = ratio(primary, "observable_to_continuous_static_ratio")
    schedule = {name: ratio([row for row in primary if row["schedule_family"] == name],
                            "observable_to_continuous_static_ratio")
                for name in ("single_switch", "alternating")}
    delay = {str(value): ratio([row for row in primary if int(row["delay"]) == value],
                               "observable_to_continuous_static_ratio")
             for value in (0, 1, 3)}
    low_t_endpoints = [row for row in endpoints if float(row["temporal_correlation"]) == 0.0]
    high_t_endpoints = [row for row in endpoints if float(row["temporal_correlation"]) == 0.9]
    metrics = {
        "primary_ratio": primary_ratio,
        "primary_improvement": 1.0 - primary_ratio,
        "primary_strict_cell_fraction": float(np.mean([
            row["observable_to_continuous_static_ratio"] < 1.0 for row in primary
        ])),
        "primary_to_local_ratio": ratio(primary, "observable_to_local_ratio"),
        "schedule_ratios": schedule,
        "delay_ratios": delay,
        "stationary_to_local_ratio": ratio(stationary, "observable_to_local_ratio"),
        "low_signal_ratio": ratio(low_signal, "observable_to_continuous_static_ratio"),
        "high_temporal_ratio": ratio(high_temporal, "observable_to_continuous_static_ratio"),
        "low_temporal_mean_rho_upper": float(np.mean([row["mean_rho_upper"] for row in low_t_endpoints])),
        "high_temporal_mean_rho_upper": float(np.mean([row["mean_rho_upper"] for row in high_t_endpoints])),
        "low_temporal_mean_effective_samples": float(np.mean([row["mean_effective_samples"] for row in low_t_endpoints])),
        "high_temporal_mean_effective_samples": float(np.mean([row["mean_effective_samples"] for row in high_t_endpoints])),
        "accepted_nonlocal_rate": float(np.mean([row["accepted_nonlocal_rate"] for row in endpoints])),
        "mean_debt": float(np.mean([row["mean_debt"] for row in endpoints])),
        "max_debt": float(np.max([row["max_debt"] for row in endpoints])),
        "mean_iterations_per_recipient_decision": float(np.mean([
            row["mean_iterations_per_recipient_decision"] for row in endpoints
        ])),
    }
    complete = len(endpoints) == config["workload"]["endpoints"] and len(cells) == 432 and all(
        math.isfinite(float(row["observable_auc"])) for row in endpoints
    )
    budget = all(row["learning_transitions"] == 240 and row["extra_probe_transitions"] == 0
                 and row["message_units"] <= 18 for row in endpoints)
    gates = {
        "C1": complete,
        "C2": metrics["primary_improvement"] >= 0.10,
        "C3": metrics["primary_strict_cell_fraction"] >= 0.80,
        "C4": all(1.0 - value >= 0.08 for value in schedule.values()),
        "C5": all(1.0 - value >= 0.05 for value in delay.values()),
        "C6": 1.0 - metrics["primary_to_local_ratio"] >= 0.10,
        "C7": metrics["stationary_to_local_ratio"] <= 1.06,
        "C8": primary_ratio < metrics["low_signal_ratio"]
              and primary_ratio < metrics["high_temporal_ratio"],
        "C9": budget,
        "C10": metrics["high_temporal_mean_rho_upper"] > metrics["low_temporal_mean_rho_upper"]
               and metrics["high_temporal_mean_effective_samples"] < metrics["low_temporal_mean_effective_samples"],
        "C11": metrics["accepted_nonlocal_rate"] > 0.0 and math.isfinite(metrics["max_debt"])
               and metrics["mean_iterations_per_recipient_decision"] <= 35.0,
        "C12": False,
        "C13": False,
        "C14": False,
    }
    return ({
        "experiment_id": config["experiment_id"],
        "classification": config["classification"],
        "metrics": metrics,
        "gates": gates,
        "authorization": config["authorization"],
    }, cells)


def _payload(payload):
    return run_endpoint(payload)


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    validate(config)
    static = continuous_static_weights()
    rows = source_rows()
    started = time.perf_counter()
    payloads = ((config, row, static[row["cell_id"]]) for row in rows)
    with ProcessPoolExecutor(max_workers=config["workload"]["workers"]) as pool:
        endpoints = list(pool.map(_payload, payloads, chunksize=config["workload"]["chunksize"]))
    runtime = time.perf_counter() - started
    summary, cells = analyze(config, endpoints)
    summary["runtime_seconds"] = runtime
    summary["gates"]["C12"] = runtime <= 60.0 * config["workload"]["hard_timeout_minutes"]
    summary["all_pre_reproduction_gates_pass"] = all(
        summary["gates"][f"C{i}"] for i in range(1, 13)
    )
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "endpoints.csv", endpoints)
    write_csv(output / "cells.csv", cells)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_config(args.config)
    result = validate(config) if args.command == "validate" else run(config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

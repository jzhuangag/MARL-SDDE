"""Validate the frozen TSP-V2 MARL learning-progress sensor G0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(root: Path, config_path: Path, replay_identical: bool) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records = []
    for path in sorted(root.glob("*/progress_sensor_g0.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    expected = {
        (task["environment"], task["task"], coupling, seed)
        for task in config["tasks"]
        for coupling in task["couplings"]
        for seed in task["seeds"]
    }
    observed = {
        (row["environment"], row["task"], row["coupling"], row["seed"])
        for row in records
    }
    complete = len(records) == len(expected) and observed == expected
    valid = complete
    selections = {}
    for row in records:
        cell = (row["environment"], row["task"], row["coupling"], row["seed"])
        selections["|".join(map(str, cell))] = row["selected_q_diagnostic"]
        hashes = {entry["initial_parameter_sha256"] for entry in row["results"].values()}
        valid = valid and len(hashes) == 1
        valid = valid and row["evaluation_return_computed"] is False
        valid = valid and sorted(map(int, row["results"])) == config["candidate_q"]
        for q_text, entry in row["results"].items():
            trace = np.asarray(entry["reward_trace"], dtype=float)
            valid = valid and trace.size == config["micro_updates"]
            valid = valid and bool(np.isfinite(trace).all())
            valid = valid and entry["charged_messages"] > 0
            valid = valid and entry["charged_environment_ticks"] > 0
    mam_ind = [
        row["selected_q_diagnostic"]
        for row in records
        if row["environment"] == "mamujoco" and row["coupling"] == "independent"
    ]
    mam_shared = [
        row["selected_q_diagnostic"]
        for row in records
        if row["environment"] == "mamujoco" and row["coupling"] == "shared"
    ]
    pet_traces = [
        np.asarray(entry["reward_trace"], dtype=float)
        for row in records
        if row["environment"] == "pettingzoo_mpe"
        for entry in row["results"].values()
    ]
    gates = {
        "complete_unique_cells": complete,
        "identical_initialization_finite_exact_interface": bool(valid),
        "mamujoco_independent_broad_signal": bool(
            mam_ind and 8 in mam_ind and not all(q == 1 for q in mam_ind)
        ),
        "mamujoco_shared_interior_signal": bool(
            mam_shared and any(q in {2, 4} for q in mam_shared)
        ),
        "pettingzoo_nonconstant_signal": bool(
            pet_traces and all(float(np.ptp(trace)) > 0.0 for trace in pet_traces)
        ),
        "analysis_replay_byte_identical": bool(replay_identical),
    }
    passed = all(gates.values())
    return {
        "experiment_id": config["experiment_id"],
        "records": len(records),
        "expected_records": len(expected),
        "selections": selections,
        "gates": gates,
        "pass": passed,
        "decision": "authorize-certificate-design" if passed else "stop",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-identical", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.root, args.config, args.replay_identical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "records": result["records"]}))


if __name__ == "__main__":
    main()


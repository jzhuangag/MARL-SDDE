"""Validate the non-confirmatory Pistonball actor--critic backbone scan."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


EXPECTED_CONFIGS = {
    "fresh_s1",
    "fresh_s3",
    "fresh_s10",
    "fresh_c1_s3",
    "no_s3",
}


def analyze(paths: Iterable[Path]) -> dict[str, Any]:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_code = {str(record["qualification_code"]): record for record in records}
    if set(by_code) != EXPECTED_CONFIGS:
        raise ValueError("backbone qualification configuration set is incomplete")
    reference = by_code["fresh_s1"]
    transition_grids = {
        tuple(int(row["actor_transitions"]) for row in record["evaluations"])
        for record in records
    }
    initial_returns = {
        float(record["evaluations"][0]["return"]) for record in records
    }
    if len(transition_grids) != 1 or len(initial_returns) != 1:
        raise ValueError("qualification transition grids or initialization differ")
    if any(
        record["seed"] != reference["seed"]
        or record["launched_actor_transitions"]
        != reference["launched_actor_transitions"]
        or not record["finite"]
        or not record["budget_feasible"]
        or not record["deterministic_algorithms"]
        or record["remaining_packets_after_drain"] != 0
        or record["received_packets"] != record["launched_gradient_packets"]
        for record in records
    ):
        raise ValueError("a backbone qualification invariant failed")

    initial_return = next(iter(initial_returns))
    rows: dict[str, dict[str, float | int | str]] = {}
    for code, record in by_code.items():
        terminal = float(record["evaluations"][-1]["return"])
        drift = [float(value) for value in record["actor_parameter_drift_l2"]]
        values = (
            terminal,
            float(record["mean_td_loss"]),
            float(record["mean_packet_gradient_norm"]),
            *drift,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite backbone diagnostic")
        rows[code] = {
            "scheduler": str(record["scheduler"]),
            "terminal_return": terminal,
            "learning_delta": terminal - initial_return,
            "critic_step": float(record["config"]["critic_step"]),
            "actor_receipt_step": float(record["config"]["actor_receipt_step"]),
            "mean_td_loss": float(record["mean_td_loss"]),
            "mean_packet_gradient_norm": float(record["mean_packet_gradient_norm"]),
            "mean_actor_parameter_drift_l2": sum(drift) / len(drift),
            "maximum_actor_parameter_drift_l2": max(drift),
            "refresh_units": int(record["spent_refresh_units"]),
            "optional_policy_bytes": int(record["optional_policy_bytes"]),
            "runtime_seconds": float(record["runtime_seconds"]),
        }
    fresh_codes = sorted(code for code in rows if code.startswith("fresh_"))
    best_fresh = max(fresh_codes, key=lambda code: float(rows[code]["learning_delta"]))
    threshold = 2.0
    qualified = (
        float(rows[best_fresh]["learning_delta"]) >= threshold
        and float(rows[best_fresh]["mean_actor_parameter_drift_l2"]) > 0.0
        and float(rows[best_fresh]["mean_packet_gradient_norm"]) > 0.0
    )
    return {
        "experiment_id": "PISTONBALL-BACKBONE-LEARNING-QUALIFICATION",
        "confirmatory": False,
        "pilot_or_formal_evidence": False,
        "seed": reference["seed"],
        "initial_return": initial_return,
        "minimum_learning_delta": threshold,
        "best_fresh_code": best_fresh,
        "qualified": qualified,
        "decision": (
            "retain the deterministic actor--critic backbone for controller development"
            if qualified
            else "stop this deterministic actor--critic backbone and implement a strong PPO/MAPPO backbone"
        ),
        "rows": rows,
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

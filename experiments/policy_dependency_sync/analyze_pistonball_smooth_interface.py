"""Joint learner and signed-score qualification for the smooth Pistonball model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .analyze_pistonball_score_scale import analyze as analyze_scale


def analyze(fresh_path: Path, scale_path: Path) -> dict[str, Any]:
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    expected_fresh = {
        "scheduler": "complete_burst",
        "seed": 79005,
        "development_only": True,
        "finite": True,
        "budget_feasible": True,
        "remaining_packets_after_drain": 0,
    }
    for key, expected in expected_fresh.items():
        if fresh[key] != expected:
            raise ValueError(f"unexpected fresh {key}: {fresh[key]!r}")
    fresh_config = fresh["config"]
    required_config = {
        "network_activation": "silu",
        "launches": 4096,
        "n_agents": 20,
        "rollout_horizon": 4,
        "actor_receipt_step": 0.01,
        "critic_step": 0.0003,
        "budget_rate": 19.0,
        "budget_enforcement": "prefix",
        "maximum_delay": 0,
    }
    for key, expected in required_config.items():
        if fresh_config[key] != expected:
            raise ValueError(f"unexpected fresh config {key}: {fresh_config[key]!r}")
    if fresh["received_packets"] != fresh["launched_gradient_packets"]:
        raise ValueError("fresh packet accounting failed")
    if fresh["launched_actor_transitions"] != 4096 * 4 * 20:
        raise ValueError("fresh actor-transition accounting failed")

    initial_return = float(fresh["evaluations"][0]["return"])
    terminal_return = float(fresh["evaluations"][-1]["return"])
    learning_delta = terminal_return - initial_return
    gradient_norm = float(fresh["mean_packet_gradient_norm"])
    actor_drift = [float(value) for value in fresh["actor_parameter_drift_l2"]]
    fresh_values = (initial_return, terminal_return, gradient_norm, *actor_drift)
    if not all(math.isfinite(value) for value in fresh_values):
        raise ValueError("non-finite smooth learner diagnostic")
    learner_passed = (
        learning_delta >= 2.0
        and gradient_norm > 0.0
        and sum(actor_drift) / len(actor_drift) > 0.0
    )

    scale = analyze_scale(scale_path)
    scale_raw = json.loads(scale_path.read_text(encoding="utf-8"))
    if scale_raw["config"].get("network_activation") != "silu":
        raise ValueError("scale run did not use the smooth activation")
    signed_active = float(scale["signed_favorable_fraction"]) >= 0.01
    score_passed = bool(scale["scale_audit_passed"]) and signed_active
    qualified = learner_passed and score_passed
    return {
        "experiment_id": "PISTONBALL-SMOOTH-INTERFACE-QUALIFICATION",
        "confirmatory": False,
        "pilot_or_formal_evidence": False,
        "qualified": qualified,
        "decision": (
            "authorize outcome-free design of the matched controller-headroom matrix"
            if qualified
            else "stop before controller headroom and redesign the smooth learner or score"
        ),
        "learner_gate": {
            "passed": learner_passed,
            "minimum_learning_delta": 2.0,
            "initial_return": initial_return,
            "terminal_return": terminal_return,
            "learning_delta": learning_delta,
            "mean_packet_gradient_norm": gradient_norm,
            "mean_actor_parameter_drift_l2": sum(actor_drift) / len(actor_drift),
            "evaluations": fresh["evaluations"],
            "refresh_units": int(fresh["spent_refresh_units"]),
            "optional_policy_bytes": int(fresh["optional_policy_bytes"]),
            "runtime_seconds": float(fresh["runtime_seconds"]),
        },
        "score_gate": {
            "passed": score_passed,
            "minimum_signed_favorable_fraction": 0.01,
            "scale_summary": scale,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--scale", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.fresh, args.scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

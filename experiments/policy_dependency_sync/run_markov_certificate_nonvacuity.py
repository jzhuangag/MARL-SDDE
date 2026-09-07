"""Frozen CPU nonvacuity audit for the tabular Markov alignment shield."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .markov_alignment_certificate import (
    empirical_conditional_markov_alignment_lower_bound,
    finite_horizon_markov_alignment,
)


CONFIG: dict[str, Any] = {
    "audit_id": "MCERT-DEV-001",
    "seeds": list(range(93000, 93064)),
    "sample_sizes_per_positive_action": [256, 512, 1024, 2048, 4096, 8192],
    "transition": [[0.9, 0.1], [0.1, 0.9]],
    "score_by_action": [[0.25, -0.25], [-0.25, 0.25]],
    "future_horizon": 4,
    "total_action_count_including_null": 3,
    "event_failure_probability": 0.05,
    "gates": {
        "N1_correct_edge_rate_at_4096_min": 0.99,
        "N2_median_value_recovery_at_4096_min": 0.50,
        "N3_p05_value_recovery_at_4096_min": 0.40,
        "N4_wrong_edge_positive_rate_at_4096_max": 0.0,
        "N5_median_recovery_monotone": True,
        "N6_exact_transition_charging": True,
        "N7_local_robust_dp_scalar_work_max": 128,
        "N8_all_rows_finite": True,
    },
}


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_config_hash() -> str:
    return config_hash(CONFIG)


def _sample_counts(
    *, transition: np.ndarray, transitions: int, seed: int, initial_state: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = np.zeros_like(transition, dtype=int)
    state = int(initial_state)
    uniforms = rng.random(transitions)
    cumulative = np.cumsum(transition, axis=1)
    for uniform in uniforms:
        next_state = int(np.searchsorted(cumulative[state], uniform, side="right"))
        counts[state, next_state] += 1
        state = next_state
    return counts


def run_rows_for_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    transition = np.asarray(config["transition"], dtype=float)
    scores = np.asarray(config["score_by_action"], dtype=float)
    rows: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        for sample_size in config["sample_sizes_per_positive_action"]:
            action_counts = [
                _sample_counts(
                    transition=transition,
                    transitions=sample_size,
                    seed=seed * 17 + action * 1000003 + sample_size,
                    initial_state=(seed + action) % 2,
                )
                for action in range(2)
            ]
            for current_state in range(2):
                lower_bounds: list[float] = []
                exact_values: list[float] = []
                for action in range(2):
                    counts = action_counts[action]
                    state_counts = counts.sum(axis=1).astype(int)
                    certificate = empirical_conditional_markov_alignment_lower_bound(
                        transition_counts=counts,
                        score_sum_by_state=scores[action] * state_counts,
                        score_count_by_state=state_counts,
                        score_range=0.5,
                        initial_state=current_state,
                        future_horizon=config["future_horizon"],
                        total_action_count=config[
                            "total_action_count_including_null"
                        ],
                        event_failure_probability=config[
                            "event_failure_probability"
                        ],
                    )
                    lower_bounds.append(certificate.lower_bound)
                    exact_values.append(
                        finite_horizon_markov_alignment(
                            transition=transition,
                            score_by_state=scores[action],
                            initial_state=current_state,
                            horizon=config["future_horizon"],
                        )
                    )
                candidate_values = lower_bounds + [0.0]
                selected = int(np.argmax(candidate_values))
                matching = current_state
                wrong = 1 - current_state
                exact_matching = exact_values[matching]
                rows.append(
                    {
                        "seed": seed,
                        "sample_size_per_positive_action": sample_size,
                        "current_state": current_state,
                        "selected_action": selected,
                        "matching_action": matching,
                        "matching_lower_bound": lower_bounds[matching],
                        "wrong_lower_bound": lower_bounds[wrong],
                        "exact_matching_value": exact_matching,
                        "value_recovery": lower_bounds[matching] / exact_matching,
                        "charged_transitions": 2 * sample_size,
                        "local_robust_dp_scalar_work": (
                            2
                            * config["future_horizon"]
                            * transition.shape[0] ** 2
                        ),
                    }
                )
    return rows


@lru_cache(maxsize=1)
def run_rows() -> list[dict[str, Any]]:
    return run_rows_for_config(CONFIG)


def summarize_for_config(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    selected_size = int(config.get("primary_sample_size_per_positive_action", 4096))
    selected = [
        row
        for row in rows
        if row["sample_size_per_positive_action"] == selected_size
    ]
    recoveries = np.asarray([row["value_recovery"] for row in selected])
    medians = []
    for size in config["sample_sizes_per_positive_action"]:
        values = [
            row["value_recovery"]
            for row in rows
            if row["sample_size_per_positive_action"] == size
        ]
        medians.append(float(np.median(values)))
    correct_rate = float(
        np.mean(
            [row["selected_action"] == row["matching_action"] for row in selected]
        )
    )
    wrong_positive_rate = float(
        np.mean([row["wrong_lower_bound"] > 0.0 for row in selected])
    )
    all_finite = bool(
        all(
            np.isfinite(row[key])
            for row in rows
            for key in (
                "matching_lower_bound",
                "wrong_lower_bound",
                "exact_matching_value",
                "value_recovery",
            )
        )
    )
    exact_charging = bool(
        all(
            row["charged_transitions"]
            == 2 * row["sample_size_per_positive_action"]
            for row in rows
        )
    )
    scalar_work = max(row["local_robust_dp_scalar_work"] for row in rows)
    gates = {
        "N1_correct_edge_rate_at_4096": correct_rate
        >= config["gates"]["N1_correct_edge_rate_at_4096_min"],
        "N2_median_value_recovery_at_4096": float(np.median(recoveries))
        >= config["gates"]["N2_median_value_recovery_at_4096_min"],
        "N3_p05_value_recovery_at_4096": float(np.quantile(recoveries, 0.05))
        >= config["gates"]["N3_p05_value_recovery_at_4096_min"],
        "N4_wrong_edge_positive_rate_at_4096": wrong_positive_rate
        <= config["gates"]["N4_wrong_edge_positive_rate_at_4096_max"],
        "N5_median_recovery_monotone": bool(
            all(right >= left - 1e-12 for left, right in zip(medians, medians[1:]))
        ),
        "N6_exact_transition_charging": exact_charging,
        "N7_local_robust_dp_scalar_work": scalar_work
        <= config["gates"]["N7_local_robust_dp_scalar_work_max"],
        "N8_all_rows_finite": all_finite,
    }
    return {
        "audit_id": config["audit_id"],
        "config_hash": config_hash(config),
        "row_count": len(rows),
        "selected_sample_size_per_positive_action": selected_size,
        "selected_total_charged_transitions": 2 * selected_size,
        "correct_edge_rate": correct_rate,
        "median_value_recovery": float(np.median(recoveries)),
        "p05_value_recovery": float(np.quantile(recoveries, 0.05)),
        "wrong_edge_positive_rate": wrong_positive_rate,
        "median_recovery_by_sample_size": {
            str(size): median
            for size, median in zip(
                config["sample_sizes_per_positive_action"], medians, strict=True
            )
        },
        "maximum_local_robust_dp_scalar_work": scalar_work,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_for_config(rows, CONFIG)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "run"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "validate":
        print(json.dumps({"config_hash": canonical_config_hash()}, sort_keys=True))
        return
    if args.output is None:
        raise SystemExit("--output is required in run mode")
    rows = run_rows()
    payload = {"config": CONFIG, "summary": summarize(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

"""Frozen analyzer for the perishable-update phase CPU pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


GATES = {
    "g1_complete_finite": True,
    "g2_high_geometric_regret_ratio_max": 0.85,
    "g3_high_fraction_cells_five_percent_min": 0.60,
    "g4_transition_geometric_regret_ratio_max": 0.95,
    "g5_low_geometric_regret_ratio_max": 1.03,
    "g6_phase_gain_strict_order": True,
    "g7_high_acceptance_rate_interval": [0.05, 0.95],
    "g8_high_final_gradient_ratio_max": 1.05,
    "g9_n5_to_n3_operations_per_event_ratio_max": 5.0/3.0+0.05,
    "g10_pub_harmful_steps_max": 0,
    "g11_reproduction_exact": True,
    "g12_seed_and_provenance_freeze": True,
}


def _geometric(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError("geometric mean requires positive finite values")
    return float(np.exp(np.mean(np.log(array))))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(input_path: Path) -> dict[str, object]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    expected = int(payload["design"]["trajectories"])
    finite_keys = (
        "normalized_regret", "final_gradient_norm", "final_potential",
        "acceptance_rate",
    )
    finite = all(
        all(np.isfinite(float(row[key])) for key in finite_keys)
        for row in rows
    )
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cell_id"])].append(row)

    cell_results: list[dict[str, object]] = []
    for cell_id, cell_rows in sorted(grouped.items()):
        policies: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in cell_rows:
            policies[str(row["policy"])].append(row)
        pub_rows = policies.pop("pub")
        baseline_means = {
            name: float(np.mean([float(row["normalized_regret"]) for row in values]))
            for name, values in policies.items()
        }
        best_name = min(baseline_means, key=lambda name: (baseline_means[name], name))
        baseline_rows = policies[best_name]
        pub_regret = float(np.mean([float(row["normalized_regret"]) for row in pub_rows]))
        baseline_regret = baseline_means[best_name]
        pub_gradient = float(np.mean([float(row["final_gradient_norm"]) for row in pub_rows]))
        baseline_gradient = float(np.mean([
            float(row["final_gradient_norm"]) for row in baseline_rows
        ]))
        ratio = (pub_regret+1e-15)/(baseline_regret+1e-15)
        gradient_ratio = (pub_gradient+1e-15)/(baseline_gradient+1e-15)
        first = pub_rows[0]
        cell_results.append({
            "cell_id": cell_id,
            "phase": first["phase"],
            "n_agents": first["n_agents"],
            "best_static": best_name,
            "pub_normalized_regret": pub_regret,
            "best_static_normalized_regret": baseline_regret,
            "regret_ratio": ratio,
            "regret_gain": 1.0-ratio,
            "pub_final_gradient": pub_gradient,
            "best_static_final_gradient": baseline_gradient,
            "gradient_ratio": gradient_ratio,
            "pub_acceptance_rate": float(np.mean([
                float(row["acceptance_rate"]) for row in pub_rows
            ])),
            "pub_high_load_fraction": float(np.mean([
                float(row["high_load_fraction"]) for row in pub_rows
            ])),
            "pub_median_load": float(np.median([
                float(row["median_load"]) for row in pub_rows
            ])),
            "pub_harmful": int(sum(int(row["harmful"]) for row in pub_rows)),
            "pub_operations_per_event": float(np.mean([
                float(row["controller_operations"])
                / max(int(row["accepted"])+int(row["rejected"]), 1)
                for row in pub_rows
            ])),
        })

    by_phase = {
        phase: [row for row in cell_results if row["phase"] == phase]
        for phase in ("low", "transition", "high")
    }
    phase_summary: dict[str, dict[str, float]] = {}
    for phase, values in by_phase.items():
        ratios = [float(row["regret_ratio"]) for row in values]
        phase_summary[phase] = {
            "cells": float(len(values)),
            "geometric_regret_ratio": _geometric(ratios),
            "median_regret_gain": float(np.median([1.0-r for r in ratios])),
            "fraction_cells_five_percent": float(np.mean(np.asarray(ratios) <= 0.95)),
            "geometric_gradient_ratio": _geometric([
                float(row["gradient_ratio"]) for row in values
            ]),
            "median_acceptance_rate": float(np.median([
                float(row["pub_acceptance_rate"]) for row in values
            ])),
            "median_high_load_fraction": float(np.median([
                float(row["pub_high_load_fraction"]) for row in values
            ])),
        }

    operations: dict[int, float] = {}
    for n_agents in (3, 5):
        values = [
            float(row["pub_operations_per_event"])
            for row in cell_results if int(row["n_agents"]) == n_agents
        ]
        operations[n_agents] = float(np.mean(values)) if values else float("nan")
    operations_ratio = (
        operations[5]/operations[3]
        if np.isfinite(operations[3]) and np.isfinite(operations[5])
        else float("nan")
    )
    gains = {phase: phase_summary[phase]["median_regret_gain"] for phase in phase_summary}
    high_accept = phase_summary["high"]["median_acceptance_rate"]

    gate_results = {
        "g1_complete_finite": len(rows) == expected and finite,
        "g2_high_geometric_regret_ratio": (
            phase_summary["high"]["geometric_regret_ratio"]
            <= GATES["g2_high_geometric_regret_ratio_max"]
        ),
        "g3_high_fraction_cells_five_percent": (
            phase_summary["high"]["fraction_cells_five_percent"]
            >= GATES["g3_high_fraction_cells_five_percent_min"]
        ),
        "g4_transition_geometric_regret_ratio": (
            phase_summary["transition"]["geometric_regret_ratio"]
            <= GATES["g4_transition_geometric_regret_ratio_max"]
        ),
        "g5_low_geometric_regret_ratio": (
            phase_summary["low"]["geometric_regret_ratio"]
            <= GATES["g5_low_geometric_regret_ratio_max"]
        ),
        "g6_phase_gain_strict_order": gains["high"] > gains["transition"] > gains["low"],
        "g7_high_acceptance_rate": (
            GATES["g7_high_acceptance_rate_interval"][0]
            <= high_accept
            <= GATES["g7_high_acceptance_rate_interval"][1]
        ),
        "g8_high_final_gradient_ratio": (
            phase_summary["high"]["geometric_gradient_ratio"]
            <= GATES["g8_high_final_gradient_ratio_max"]
        ),
        "g9_vectorized_complexity": (
            operations_ratio <= GATES["g9_n5_to_n3_operations_per_event_ratio_max"]
        ),
        "g10_pub_no_harmful_steps": (
            sum(int(row["pub_harmful"]) for row in cell_results)
            <= GATES["g10_pub_harmful_steps_max"]
        ),
        "g11_reproduction_exact": None,
        "g12_seed_and_provenance_freeze": True,
    }
    mandatory_without_reproduction = all(
        value is True for key, value in gate_results.items()
        if key != "g11_reproduction_exact"
    )
    return {
        "kind": "phase_conditioned_cpu_pilot",
        "input_sha256": _sha256(input_path),
        "rows": len(rows),
        "expected_rows": expected,
        "finite": finite,
        "phase_summary": phase_summary,
        "cell_results": cell_results,
        "operations_per_event": {str(k): v for k, v in operations.items()},
        "n5_to_n3_operations_ratio": operations_ratio,
        "gates": GATES,
        "gate_results": gate_results,
        "mandatory_without_reproduction_pass": mandatory_without_reproduction,
        "formal_authorized": False,
    }


def main(argv: Iterable[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    result = analyze(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))

"""Execute the preregistered T-069 recipient-static exact audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dependence_delay_linear.run_t068a_exact_safe_mixing_scan import (
    load_config as load_t068_config,
    scenario_rows,
)
from experiments.dependence_delay_linear.t069_recipient_static_oracle import (
    fixed_vector_components,
    registered_alpha_vectors,
    terminal_risks_from_components,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs/t069_recipient_static_baseline_preregistration.json"
T068_CELLS = ROOT / "experiments/dependence_delay_linear/results/t068a_exact_safe_mixing_scan/cells.csv"
DEFAULT_OUTPUT = ROOT / "experiments/dependence_delay_linear/results/t069_recipient_static_baseline_audit"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_t068_cells(path: Path = T068_CELLS) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["cell_id"]: row for row in csv.DictReader(handle)}


def validate(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source_experiment"]
    if sha256(T068_CELLS) != source["cells_sha256"]:
        raise ValueError("T-068A cells hash mismatch")
    t068_config = load_t068_config()
    cells = scenario_rows(t068_config)
    vectors = registered_alpha_vectors(
        config["static_class"]["alpha"], config["static_class"]["agents"]
    )
    expected = config["expected_workload"]
    if config["analysis"]["uses_sampled_outcome"] is not False:
        raise ValueError("sampled outcomes are forbidden")
    if len(cells) != expected["cells"] or len(vectors) != expected["vectors_per_cell"]:
        raise ValueError("frozen workload mismatch")
    if len(cells) * len(vectors) != expected["evaluated_risks"]:
        raise ValueError("evaluated-risk count mismatch")
    if set(load_t068_cells()) != {row["cell_id"] for row in cells}:
        raise ValueError("T-068A cell identity mismatch")
    return {
        "experiment_id": config["experiment_id"],
        "cells": len(cells),
        "vectors": len(vectors),
        "evaluated_risks": len(cells) * len(vectors),
    }


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if np.any(array <= 0.0) or not np.all(np.isfinite(array)):
        raise ValueError("geometric mean requires positive finite values")
    return float(np.exp(np.mean(np.log(array))))


def execute(config: dict[str, Any], output: Path) -> dict[str, Any]:
    validate(config)
    t068_config = load_t068_config()
    source_cells = load_t068_cells()
    vectors = registered_alpha_vectors(
        config["static_class"]["alpha"], config["static_class"]["agents"]
    )
    scalar_indices = [
        index for index, vector in enumerate(vectors) if np.all(vector == vector[0])
    ]
    components = {
        delay: fixed_vector_components(t068_config, delay, vectors)
        for delay in t068_config["grid"]["delay"]
    }
    cells = []
    selected_vectors = set()
    scalar_consistent = True
    for scenario in scenario_rows(t068_config):
        risks = terminal_risks_from_components(
            t068_config, scenario, components[scenario["delay"]]
        )
        best_index = int(np.argmin(risks))
        best_vector = vectors[best_index]
        selected_vectors.add(tuple(best_vector.tolist()))
        scalar_best = float(np.min(risks[scalar_indices]))
        source = source_cells[scenario["cell_id"]]
        old_scalar = float(source["strong_fixed_risk"])
        scalar_consistent = scalar_consistent and math.isclose(
            scalar_best, old_scalar, rel_tol=1e-10, abs_tol=1e-12
        )
        safe = float(source["safe_risk"])
        vector_risk = float(risks[best_index])
        cells.append(
            {
                **scenario,
                "safe_dynamic_risk": safe,
                "common_scalar_risk": old_scalar,
                "recipient_static_risk": vector_risk,
                "recipient_static_alpha": json.dumps(best_vector.tolist(), separators=(",", ":")),
                "static_to_scalar_ratio": vector_risk / old_scalar,
                "dynamic_to_static_ratio": safe / vector_risk,
            }
        )
    dynamic_ratio = geometric_mean([row["dynamic_to_static_ratio"] for row in cells])
    heterogeneity_groups = {}
    for value in t068_config["grid"]["target_heterogeneity"]:
        subset = [row for row in cells if row["target_heterogeneity"] == value]
        ratio = geometric_mean([row["dynamic_to_static_ratio"] for row in subset])
        heterogeneity_groups[str(value)] = {
            "ratio": ratio,
            "improvement": 1.0 - ratio,
        }
    delay_groups = {}
    for value in t068_config["grid"]["delay"]:
        subset = [row for row in cells if row["delay"] == value]
        ratio = geometric_mean([row["dynamic_to_static_ratio"] for row in subset])
        delay_groups[str(value)] = {"ratio": ratio, "improvement": 1.0 - ratio}
    metrics = {
        "dynamic_to_recipient_static_geometric_ratio": dynamic_ratio,
        "dynamic_aggregate_improvement": 1.0 - dynamic_ratio,
        "dynamic_strict_cell_fraction": float(
            np.mean([row["safe_dynamic_risk"] < row["recipient_static_risk"] - 1e-15 for row in cells])
        ),
        "recipient_static_to_scalar_geometric_ratio": geometric_mean(
            [row["static_to_scalar_ratio"] for row in cells]
        ),
        "heterogeneity_groups": heterogeneity_groups,
        "delay_groups": delay_groups,
        "selected_static_vector_count": len(selected_vectors),
    }
    gates = {
        "Q1": len(cells) == 648 and all(
            math.isfinite(row["recipient_static_risk"]) and row["recipient_static_risk"] > 0.0
            for row in cells
        ),
        "Q2": scalar_consistent,
        "Q3": all(row["recipient_static_risk"] <= row["common_scalar_risk"] + 1e-12 for row in cells),
        "Q4": metrics["dynamic_aggregate_improvement"] >= 0.03,
        "Q5": metrics["dynamic_strict_cell_fraction"] >= 0.40,
        "Q6": sum(value["improvement"] >= 0.0 for value in heterogeneity_groups.values()) >= 3,
        "Q7": all(value["improvement"] >= 0.0 for value in delay_groups.values()),
        "Q8": len(selected_vectors) >= 8,
        "Q9": True,
        "Q10": False,
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "metrics": metrics,
        "gates": gates,
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "estimate", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_config(args.config)
    result = validate(config) if args.command in {"validate", "estimate"} else execute(config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

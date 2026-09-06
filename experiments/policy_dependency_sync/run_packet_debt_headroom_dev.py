"""Development-only Pareto scan for context-adaptive packet debt.

This script uses no learning outcome.  It compares the exact context-adaptive
horizon/graph minimizer with a convexified envelope of context-independent
fixed-horizon policies under the dynamic policy's own average environment and
message costs.  Results are design information and cannot be formal evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from .factored_markov_packet import horizon_certificate
from .packet_debt import choose_horizon_and_graph, packet_debt


HORIZONS = (2, 4, 8)
MODES = {
    "slow_noisy": {
        "move_probability": 0.1,
        "innovation_variance": 1.0,
        "temporal_correlation": 0.7,
    },
    "fast_clean": {
        "move_probability": 0.8,
        "innovation_variance": 0.05,
        "temporal_correlation": 0.1,
    },
}


def contexts() -> list[dict]:
    target = np.asarray([0.7, -0.5, 0.9, -0.8, 0.4, -0.6], dtype=float)
    theta = np.asarray([-0.9, 0.8, -0.7, 1.0, -0.5, 0.9], dtype=float)
    rows = []
    for owner in range(theta.size):
        for start_offset in range(theta.size - 1):
            for mode_name, mode in MODES.items():
                for lag_scale in (0.15, 0.45, 0.8):
                    cache = target + (1.0 + lag_scale) * (theta - target)
                    cache += 0.03 * np.sin(
                        np.arange(theta.size, dtype=float) + owner + start_offset
                    )
                    cache[owner] = theta[owner]
                    rows.append(
                        {
                            "owner": owner,
                            "start_offset": start_offset,
                            "mode": mode_name,
                            "lag_scale": lag_scale,
                            "theta": theta,
                            "target": target,
                            "cache": cache,
                            **mode,
                        }
                    )
    return rows


def certificates_for_context(context: dict):
    return [
        horizon_certificate(
            owner=context["owner"],
            current_parameter=context["theta"],
            target=context["target"],
            cache_for_owner=context["cache"],
            start_offset=context["start_offset"],
            move_probability=context["move_probability"],
            horizon=horizon,
            cone_coverage=0.9,
            strong_convexity=0.4,
            coupling=0.9,
            innovation_variance=context["innovation_variance"],
            temporal_correlation=context["temporal_correlation"],
            step_cap=0.05,
        )[0]
        for horizon in HORIZONS
    ]


def fixed_action_rows(all_certificates, queues: dict[str, float]) -> list[dict]:
    rows = []
    for horizon_index, horizon in enumerate(HORIZONS):
        for graph_rule in ("priced_graph", "no_refresh"):
            debt_values = []
            message_values = []
            for certificates in all_certificates:
                item = certificates[horizon_index]
                if graph_rule == "priced_graph":
                    choice = choose_horizon_and_graph([item], queues, 1.0)
                    debt_values.append(choice.packet_debt)
                    message_values.append(choice.resource_costs["message"])
                else:
                    _, debt = packet_debt(item, ())
                    debt_values.append(debt)
                    message_values.append(0.0)
            rows.append(
                {
                    "name": f"H{horizon}_{graph_rule}",
                    "debt": float(np.mean(debt_values)),
                    "environment": float(horizon),
                    "message": float(np.mean(message_values)),
                }
            )
    return rows


def convex_fixed_envelope(actions: list[dict], budget: dict[str, float]) -> dict:
    objective = np.asarray([row["debt"] for row in actions], dtype=float)
    resources = np.asarray(
        [
            [row["environment"] for row in actions],
            [row["message"] for row in actions],
        ],
        dtype=float,
    )
    result = linprog(
        objective,
        A_ub=resources,
        b_ub=np.asarray([budget["environment"], budget["message"]]) + 1e-10,
        A_eq=np.ones((1, len(actions))),
        b_eq=np.ones(1),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    weights = {
        row["name"]: float(weight)
        for row, weight in zip(actions, result.x, strict=True)
        if weight > 1e-8
    }
    return {"debt": float(result.fun), "weights": weights}


def evaluate() -> dict:
    context_rows = contexts()
    all_certificates = [certificates_for_context(row) for row in context_rows]
    cells = []
    for message_price in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05):
        for environment_price in (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01):
            queues = {
                "message": message_price,
                "environment": environment_price,
            }
            dynamic = [
                choose_horizon_and_graph(certificates, queues, 1.0)
                for certificates in all_certificates
            ]
            dynamic_debt = float(np.mean([choice.packet_debt for choice in dynamic]))
            dynamic_budget = {
                "environment": float(np.mean([choice.horizon for choice in dynamic])),
                "message": float(
                    np.mean([choice.resource_costs["message"] for choice in dynamic])
                ),
            }
            actions = fixed_action_rows(all_certificates, queues)
            envelope = convex_fixed_envelope(actions, dynamic_budget)
            gain = (envelope["debt"] - dynamic_debt) / envelope["debt"]
            horizon_counts = Counter(choice.horizon for choice in dynamic)
            mode_horizons = {
                mode: Counter(
                    choice.horizon
                    for context, choice in zip(context_rows, dynamic, strict=True)
                    if context["mode"] == mode
                )
                for mode in MODES
            }
            cells.append(
                {
                    "message_price": message_price,
                    "environment_price": environment_price,
                    "dynamic_debt": dynamic_debt,
                    "dynamic_budget": dynamic_budget,
                    "fixed_envelope_debt": envelope["debt"],
                    "fixed_envelope_weights": envelope["weights"],
                    "dynamic_gain": float(gain),
                    "horizon_counts": dict(sorted(horizon_counts.items())),
                    "mode_horizon_counts": {
                        mode: dict(sorted(counts.items()))
                        for mode, counts in mode_horizons.items()
                    },
                    "graph_changes_across_contexts": len(
                        {choice.refreshed_edges for choice in dynamic}
                    ),
                }
            )
    gains = np.asarray([row["dynamic_gain"] for row in cells], dtype=float)
    nontrivial = [
        row
        for row in cells
        if len(row["horizon_counts"]) >= 2
        and row["graph_changes_across_contexts"] >= 2
    ]
    return {
        "status": "development_only",
        "scientific_outcome_authorized": False,
        "contexts": len(context_rows),
        "price_cells": len(cells),
        "metrics": {
            "median_dynamic_gain": float(np.median(gains)),
            "maximum_dynamic_gain": float(np.max(gains)),
            "fraction_gain_at_least_0_05": float(np.mean(gains >= 0.05)),
            "fraction_gain_at_least_0_10": float(np.mean(gains >= 0.10)),
            "nontrivial_cells": len(nontrivial),
            "nontrivial_median_gain": (
                float(np.median([row["dynamic_gain"] for row in nontrivial]))
                if nontrivial
                else None
            ),
        },
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


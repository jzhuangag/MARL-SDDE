"""Optimistic nonvacuity audit of Theorem 4 on exact Blackjack constants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.nonlinear_markov_td.exp019a_blackjack_config import CONFIG
from experiments.nonlinear_markov_td.run_exp019a_blackjack_pilot import (
    exact_value,
    usable_updates,
)
from experiments.nonlinear_markov_td.t029_blackjack_static_scan import (
    continuing_transition_matrix,
)


def exact_euclidean_constants() -> dict[str, object]:
    _value, stationary, states, _reset = exact_value()
    transition = continuing_transition_matrix()[0]
    gamma = float(CONFIG["gamma"])
    diagonal_stationary = np.diag(stationary)
    mean = diagonal_stationary.dot(
        np.eye(len(states)) - gamma * transition
    )
    diagonal = (
        (1.0 + gamma * gamma) * diagonal_stationary
        - gamma
        * (
            diagonal_stationary.dot(transition)
            + transition.T.dot(diagonal_stationary)
        )
    )
    monotonicity = float(
        np.linalg.eigvalsh(0.5 * (mean + mean.T)).min()
    )
    rows = []
    for q in CONFIG["q_values"]:
        for rho in CONFIG["rho_values"]:
            vq = float(rho + (1.0 - rho) / q)
            aggregate = vq * diagonal + (1.0 - vq) * mean.T.dot(mean)
            curvature = float(np.linalg.eigvalsh(aggregate).max())
            rows.append(
                {
                    "q": int(q),
                    "rho": float(rho),
                    "variance_factor": vq,
                    "curvature": curvature,
                }
            )
    return {
        "dimension": len(states),
        "stationary_minimum": float(stationary.min()),
        "monotonicity": monotonicity,
        "actions": rows,
    }


def optimistic_audit() -> dict[str, object]:
    constants = exact_euclidean_constants()
    monotonicity = float(constants["monotonicity"])
    minimum_curvature = min(
        float(row["curvature"]) for row in constants["actions"]
    )
    maximum_updates = max(
        usable_updates(int(q), int(horizon), str(ray), float(delay))[0]
        for q in CONFIG["q_values"]
        for horizon in CONFIG["target_horizons"]
        for ray in CONFIG["budget_rays"]
        for delay in CONFIG["delay_fractions"]
    )
    # Theorem 4 with delta=G=Omega=D=0 has
    # a(eta)=1-eta*mu+2*K*eta^2.  Its exact minimum is below.
    eta = monotonicity / (4.0 * minimum_curvature)
    one_step = 1.0 - monotonicity * monotonicity / (8.0 * minimum_curvature)
    ratio = float(one_step**maximum_updates)
    improvement = 1.0 - ratio
    return {
        **constants,
        "optimistic_assumptions": {
            "mixing_tv_delta": 0.0,
            "innovation_bound": 0.0,
            "innovation_second_moment": 0.0,
            "delay": 0,
            "combines_minimum_curvature_with_maximum_updates": True,
        },
        "minimum_curvature": minimum_curvature,
        "maximum_updates": maximum_updates,
        "optimistic_eta": eta,
        "optimistic_one_step_contraction": one_step,
        "optimistic_terminal_ratio": ratio,
        "optimistic_maximum_improvement": improvement,
        "five_percent_nonvacuity_gate": improvement >= 0.05,
        "decision": "stop_euclidean_theorem4_as_practical_selector",
        "next_theory_requirement": "stationary-weighted MSVE finite-time certificate",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = optimistic_audit()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "monotonicity",
                    "minimum_curvature",
                    "maximum_updates",
                    "optimistic_eta",
                    "optimistic_terminal_ratio",
                    "optimistic_maximum_improvement",
                    "five_percent_nonvacuity_gate",
                    "decision",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Frozen W0 diagonal-weighted ideal audit for T-030."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh

from experiments.nonlinear_markov_td.exp019a_blackjack_config import CONFIG
from experiments.nonlinear_markov_td.run_exp019a_blackjack_pilot import (
    exact_value,
    usable_updates,
)
from experiments.nonlinear_markov_td.t029_blackjack_static_scan import (
    continuing_transition_matrix,
)


AUDIT_CONFIG = {
    "metric_family": "diag_stationary_power",
    "theta_values": [0.0, 0.25, 0.5, 0.75, 1.0],
    "q_values": CONFIG["q_values"],
    "rho_values": CONFIG["rho_values"],
    "optimistic_mixing_tv_delta": 0.0,
    "optimistic_innovation": 0.0,
    "optimistic_delay": 0,
    "nonvacuity_gate": 0.05,
    "rule": "if_no_metric_passes_W0_stop_W1_W2_and_new_experiments",
}


def config_sha256() -> str:
    encoded = json.dumps(
        AUDIT_CONFIG, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def transformed_second_jacobian(
    transition: np.ndarray, stationary: np.ndarray, weight: np.ndarray, gamma: float
) -> tuple[np.ndarray, float]:
    root = np.sqrt(weight)
    cross = (
        (stationary * root)[:, None]
        * transition
        / root[None, :]
    )
    following_diagonal = transition.T.dot(stationary * weight) / weight
    result = (
        np.diag(stationary)
        - gamma * (cross + cross.T)
        + gamma * gamma * np.diag(following_diagonal)
    )
    ratio = weight[:, None] / weight[None, :]
    norm_squared = 1.0 + gamma * gamma * ratio
    diagonal = np.diag_indices_from(norm_squared)
    norm_squared[diagonal] = (1.0 - gamma) ** 2
    maximum_norm_squared = float(np.max(norm_squared[transition > 0.0]))
    return result, float(np.sqrt(maximum_norm_squared))


def run_audit() -> dict[str, object]:
    _value, stationary, states, _reset = exact_value()
    transition = continuing_transition_matrix()[0]
    gamma = float(CONFIG["gamma"])
    mean = np.diag(stationary).dot(
        np.eye(len(states)) - gamma * transition
    )
    maximum_updates = max(
        usable_updates(int(q), int(horizon), str(ray), float(delay))[0]
        for q in CONFIG["q_values"]
        for horizon in CONFIG["target_horizons"]
        for ray in CONFIG["budget_rays"]
        for delay in CONFIG["delay_fractions"]
    )
    metrics = []
    for theta in AUDIT_CONFIG["theta_values"]:
        weight = stationary ** float(theta)
        root = np.sqrt(weight)
        transformed_mean = (
            root[:, None] * mean / root[None, :]
        )
        monotonicity = float(
            eigvalsh(
                0.5 * (transformed_mean + transformed_mean.T),
                subset_by_index=[0, 0],
                check_finite=False,
            )[0]
        )
        diagonal, lipschitz = transformed_second_jacobian(
            transition, stationary, weight, gamma
        )
        mean_square = transformed_mean.T.dot(transformed_mean)
        # E[H^T H] - E[H]^T E[H] is PSD, hence the aggregate
        # curvature is Loewner-monotone in v_q.  The smallest registered
        # factor is attained at q=32,rho=0, so only one largest-eigenvalue
        # solve is needed for the deliberately optimistic W0 bound.
        covariance_minimum = float(
            eigvalsh(
                diagonal - mean_square,
                subset_by_index=[0, 0],
                check_finite=False,
            )[0]
        )
        if covariance_minimum < -1e-10:
            raise RuntimeError("second-Jacobian covariance lost PSD")
        minimum_vq = min(
            float(rho + (1.0 - rho) / q)
            for q in CONFIG["q_values"]
            for rho in CONFIG["rho_values"]
        )
        aggregate = minimum_vq * diagonal + (1.0 - minimum_vq) * mean_square
        minimum_curvature = float(
            eigvalsh(
                aggregate,
                subset_by_index=[len(states) - 1, len(states) - 1],
                check_finite=False,
            )[0]
        )
        if monotonicity > 0.0:
            eta = monotonicity / (4.0 * minimum_curvature)
            one_step = 1.0 - monotonicity**2 / (8.0 * minimum_curvature)
            ratio = float(one_step**maximum_updates)
            improvement = 1.0 - ratio
        else:
            eta = None
            one_step = None
            ratio = 1.0
            improvement = 0.0
        conversion = stationary / weight
        metrics.append(
            {
                "theta": float(theta),
                "monotonicity": monotonicity,
                "lipschitz": lipschitz,
                "minimum_curvature": minimum_curvature,
                "minimum_variance_factor": minimum_vq,
                "second_jacobian_covariance_min_eigenvalue": covariance_minimum,
                "optimistic_eta": eta,
                "optimistic_one_step_contraction": one_step,
                "optimistic_terminal_ratio": ratio,
                "optimistic_maximum_improvement": improvement,
                "msve_over_weight_min": float(conversion.min()),
                "msve_over_weight_max": float(conversion.max()),
                "passes_W0": improvement >= float(AUDIT_CONFIG["nonvacuity_gate"]),
            }
        )
    any_pass = any(row["passes_W0"] for row in metrics)
    return {
        "task": "T-030-W0",
        "config_sha256": config_sha256(),
        "dimension": len(states),
        "maximum_updates": maximum_updates,
        "metrics": metrics,
        "W0_any_metric_passes": any_pass,
        "W1_W2_authorized": any_pass,
        "new_sampled_experiment_authorized": False,
        "gpu_authorized": False,
        "decision": (
            "proceed_to_W1_W2" if any_pass else "stop_diagonal_weighted_practical_route"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "run"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "validate":
        print(
            json.dumps(
                {
                    "config_sha256": config_sha256(),
                    "theta_values": AUDIT_CONFIG["theta_values"],
                    "scientific_trajectories": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = run_audit()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "config_sha256": result["config_sha256"],
                "W0_any_metric_passes": result["W0_any_metric_passes"],
                "decision": result["decision"],
                "metrics": [
                    {
                        key: row[key]
                        for key in (
                            "theta",
                            "monotonicity",
                            "minimum_curvature",
                            "optimistic_maximum_improvement",
                            "passes_W0",
                        )
                    }
                    for row in result["metrics"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

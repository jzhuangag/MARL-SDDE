"""Robust-feasibility shield with nominal finite-horizon action ranking."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from certificate_sensitivity import (
    inflated_delta,
    refit_myopic_action,
    select_minimum,
    correlation_only_q,
)
from affine_markov_certificate import (
    affine_bound_components,
    affine_candidate_actions,
    affine_finite_time_bound,
)


def certify_fixed_action(
    action: Mapping[str, float],
    *,
    curvature_factor: float,
    noise_factor: float,
    log_rate_weakening: float,
) -> Dict[str, float] | None:
    """Certify a nominal action under upper moments without changing it."""

    delta_plus = inflated_delta(
        float(action["joint_delta"]), float(log_rate_weakening)
    )
    lipschitz = float(action["lipschitz"])
    monotonicity = float(action["monotonicity"])
    constants = {
        "curvature": float(curvature_factor) * float(action["curvature"]),
        "monotonicity": monotonicity,
        "lipschitz": lipschitz,
        "effective_monotonicity": monotonicity - 2.0 * lipschitz * delta_plus,
        "effective_curvature": float(curvature_factor)
        * float(action["curvature"])
        + 2.0 * lipschitz * lipschitz * delta_plus,
        "rms_delay": float(action["rms_delay"]),
        "maximum_actual_delay": int(action["maximum_actual_delay"]),
    }
    if constants["effective_monotonicity"] <= 0.0:
        return None
    omega_plus = float(noise_factor) * float(action["omega"])
    components = affine_bound_components(
        constants,
        omega_plus,
        float(action["innovation_bound"]),
        delta_plus,
        float(action["eta"]),
    )
    if not 0.0 < float(components["contraction"]) < 1.0:
        return None
    finite = affine_finite_time_bound(
        float(action["initial_error"]),
        int(action["updates"]),
        int(action["maximum_actual_delay"]),
        components,
    )
    if not np.isfinite(float(finite["finite_time_bound"])):
        return None
    return {
        **dict(action),
        "nominal_finite_time_bound": float(action["finite_time_bound"]),
        "certificate_curvature": constants["curvature"],
        "certificate_omega": omega_plus,
        "certificate_delta": delta_plus,
        "certificate_contraction": float(components["contraction"]),
        "certificate_forcing": float(components["forcing"]),
        "certificate_bound": float(finite["finite_time_bound"]),
        "curvature_factor": float(curvature_factor),
        "noise_factor": float(noise_factor),
        "log_rate_weakening": float(log_rate_weakening),
    }


def shielded_nominal_action(
    nominal_actions: Sequence[Mapping[str, float]],
    level: Mapping[str, float],
) -> Dict[str, float]:
    """Rank by nominal finite-horizon value inside the upper-certified set."""

    safe = []
    for action in nominal_actions:
        certified = certify_fixed_action(
            action,
            curvature_factor=float(level["curvature_factor"]),
            noise_factor=float(level["noise_factor"]),
            log_rate_weakening=float(level["log_rate_weakening"]),
        )
        if certified is not None:
            safe.append(certified)
    return select_minimum(safe, "nominal_finite_time_bound")


def build_shield_policy_actions(
    model: Mapping[str, np.ndarray],
    rho: float,
    maximum_delay: int,
    *,
    resource_budget: int,
    agent_counts: Sequence[int],
    sensitivity_levels: Mapping[str, Mapping[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], Tuple[Dict[str, float], ...]]:
    nominal = affine_candidate_actions(
        dict(model),
        float(rho),
        int(maximum_delay),
        resource_budget=int(resource_budget),
        agent_counts=tuple(int(q) for q in agent_counts),
    )
    policies: Dict[str, Dict[str, float]] = {
        "finite_horizon": select_minimum(nominal, "finite_time_bound")
    }
    for name, level in sensitivity_levels.items():
        policies[name] = shielded_nominal_action(nominal, level)
    myopic = tuple(refit_myopic_action(action) for action in nominal)
    policies["one_step_myopic"] = select_minimum(myopic, "myopic_score")
    selected_q = correlation_only_q(float(rho), tuple(agent_counts))
    policies["correlation_only"] = select_minimum(
        (row for row in nominal if int(row["num_agents"]) == selected_q),
        "finite_time_bound",
    )
    return policies, nominal


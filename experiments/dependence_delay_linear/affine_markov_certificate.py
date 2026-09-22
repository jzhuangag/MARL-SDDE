"""Affine finite-time certificate for predictably decorrelated Markov TD."""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from budget_participation import AGENT_COUNTS
from linear_model import make_agent_delays
from multistate_certificate import (
    RESOURCE_BUDGET,
    SERVER_OVERHEAD,
    TARGET_FRACTIONS,
    aggregate_td_noise,
    certificate_constants,
    minimum_joint_gap,
)


def innovation_bound(model: Dict[str, np.ndarray]) -> float:
    """Return the uniform per-agent TD-innovation norm bound."""

    return float(
        np.max(np.linalg.norm(model["td_noise"], axis=1))
    )


def affine_bound_components(
    constants: Dict[str, float],
    omega: float,
    innovation_norm_bound: float,
    delta: float,
    eta: float,
) -> Dict[str, float]:
    """Evaluate Theorem 4's contraction and forcing coefficients."""

    if eta <= 0.0:
        raise ValueError("eta must be positive")
    if omega < 0.0 or innovation_norm_bound < 0.0:
        raise ValueError("noise constants must be nonnegative")
    mu_delta = float(constants["effective_monotonicity"])
    if mu_delta <= 0.0:
        raise ValueError("effective monotonicity must be positive")
    curvature = float(constants["curvature"])
    lipschitz = float(constants["lipschitz"])
    rms_delay = float(constants["rms_delay"])
    innovation = float(innovation_norm_bound)
    a_delta = (
        1.0
        - eta * mu_delta
        + eta * eta
        * (2.0 * curvature + 4.0 * lipschitz * lipschitz * delta)
    )
    beta_delta = (
        4.0 * eta * innovation * innovation * delta * delta / mu_delta
        + eta
        * eta
        * (2.0 * omega + 4.0 * innovation * innovation * delta)
    )
    h_delay = (
        2.0
        * eta ** 4
        * lipschitz ** 4
        * rms_delay ** 2
    )
    g_delay = (
        2.0
        * eta ** 4
        * lipschitz ** 2
        * innovation ** 2
        * rms_delay ** 2
    )
    if a_delta <= 0.0:
        return {
            "a_delta": float(a_delta),
            "beta_delta": float(beta_delta),
            "h_delay": float(h_delay),
            "g_delay": float(g_delay),
            "young_lambda": float("nan"),
            "contraction": float("inf"),
            "forcing": float("inf"),
            "residual": float("inf"),
        }
    if rms_delay == 0.0:
        young = 0.0
        contraction = a_delta
        forcing = beta_delta
    else:
        young = float(np.sqrt(h_delay / a_delta))
        contraction = (
            (1.0 + young) * a_delta
            + (1.0 + 1.0 / young) * h_delay
        )
        forcing = (
            (1.0 + young) * beta_delta
            + (1.0 + 1.0 / young) * g_delay
        )
    residual = (
        forcing / (1.0 - contraction)
        if contraction < 1.0
        else float("inf")
    )
    return {
        "a_delta": float(a_delta),
        "beta_delta": float(beta_delta),
        "h_delay": float(h_delay),
        "g_delay": float(g_delay),
        "young_lambda": float(young),
        "contraction": float(contraction),
        "forcing": float(forcing),
        "residual": float(residual),
    }


def affine_finite_time_bound(
    initial_error: float,
    updates: int,
    maximum_actual_delay: int,
    components: Dict[str, float],
) -> Dict[str, float]:
    """Return the sharp block-envelope bound from Theorem 4."""

    if initial_error < 0.0 or updates < 0 or maximum_actual_delay < 0:
        raise ValueError("invalid finite-time arguments")
    contraction = float(components["contraction"])
    forcing = float(components["forcing"])
    if not 0.0 <= contraction < 1.0:
        return {
            "blocks": 0,
            "residual": float("inf"),
            "finite_time_bound": float("inf"),
        }
    block_length = 2 * int(maximum_actual_delay) + 1
    blocks = int(updates) // block_length
    residual = forcing / (1.0 - contraction)
    bound = residual + contraction ** blocks * max(
        float(initial_error) - residual, 0.0
    )
    return {
        "blocks": int(blocks),
        "residual": float(residual),
        "finite_time_bound": float(bound),
    }


def first_affine_stability_boundary(
    constants: Dict[str, float],
    omega: float,
    innovation_norm_bound: float,
    delta: float,
) -> float:
    """Locate the first positive loss of Theorem 4 contraction."""

    scale = max(
        float(constants["effective_monotonicity"])
        / max(2.0 * float(constants["curvature"]), np.finfo(float).eps),
        1e-8,
    )
    lower = scale * 1e-8
    lower_value = affine_bound_components(
        constants, omega, innovation_norm_bound, delta, lower
    )["contraction"]
    if lower_value >= 1.0:
        raise RuntimeError("failed to locate a small stable affine step")
    upper = scale
    upper_value = affine_bound_components(
        constants, omega, innovation_norm_bound, delta, upper
    )["contraction"]
    while upper_value < 1.0:
        lower = upper
        upper *= 1.5
        upper_value = affine_bound_components(
            constants, omega, innovation_norm_bound, delta, upper
        )["contraction"]
        if upper > 100.0:
            raise RuntimeError("failed to bracket affine boundary")
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        value = affine_bound_components(
            constants, omega, innovation_norm_bound, delta, midpoint
        )["contraction"]
        if value < 1.0:
            lower = midpoint
        else:
            upper = midpoint
    return float(0.5 * (lower + upper))


def optimize_affine_step(
    constants: Dict[str, float],
    omega: float,
    innovation_norm_bound: float,
    delta: float,
    initial_error: float,
    updates: int,
) -> Dict[str, float]:
    """Minimize the proved finite-time bound over the stable scalar interval."""

    boundary = first_affine_stability_boundary(
        constants, omega, innovation_norm_bound, delta
    )
    maximum_delay = int(constants["maximum_actual_delay"])

    def objective(log_eta: float) -> float:
        eta = float(np.exp(log_eta))
        components = affine_bound_components(
            constants, omega, innovation_norm_bound, delta, eta
        )
        return affine_finite_time_bound(
            initial_error, updates, maximum_delay, components
        )["finite_time_bound"]

    minimum = max(boundary * 1e-8, np.finfo(float).tiny)
    maximum = boundary * (1.0 - 1e-9)
    grid = np.geomspace(minimum, maximum, 81)
    values = np.asarray([objective(np.log(value)) for value in grid])
    index = int(np.argmin(values))
    left = minimum if index == 0 else float(grid[index - 1])
    right = maximum if index == len(grid) - 1 else float(grid[index + 1])
    optimized = minimize_scalar(
        objective,
        bounds=(np.log(left), np.log(right)),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 200},
    )
    eta = float(np.exp(optimized.x))
    components = affine_bound_components(
        constants, omega, innovation_norm_bound, delta, eta
    )
    finite = affine_finite_time_bound(
        initial_error, updates, maximum_delay, components
    )
    return {
        "eta": eta,
        "affine_boundary": boundary,
        **components,
        **finite,
    }


def affine_candidate_actions(
    model: Dict[str, np.ndarray],
    rho: float,
    maximum_delay: int,
    resource_budget: int = RESOURCE_BUDGET,
    agent_counts: Iterable[int] = AGENT_COUNTS,
) -> Tuple[Dict[str, float], ...]:
    """Enumerate theorem-certified q, gap, and finite-horizon step actions."""

    actions = []
    full_delays = make_agent_delays(32, maximum_delay)
    initial_error = float(model["theta_star"].dot(model["theta_star"]))
    innovation = innovation_bound(model)
    for num_agents in agent_counts:
        delays = full_delays[: int(num_agents)]
        zero_delta = certificate_constants(
            model, int(num_agents), rho, delays, delta=0.0
        )
        admissible = (
            zero_delta["monotonicity"]
            / (2.0 * zero_delta["lipschitz"])
        )
        omega = aggregate_td_noise(model, int(num_agents), rho)
        for target_fraction in TARGET_FRACTIONS:
            target = float(target_fraction) * admissible
            mixing = minimum_joint_gap(model, int(num_agents), target)
            constants = certificate_constants(
                model,
                int(num_agents),
                rho,
                delays,
                mixing["joint_delta"],
            )
            cost = SERVER_OVERHEAD + int(num_agents) + int(mixing["gap"])
            updates = int(resource_budget) // cost
            step = optimize_affine_step(
                constants,
                omega,
                innovation,
                mixing["joint_delta"],
                initial_error,
                updates,
            )
            actions.append(
                {
                    "num_agents": int(num_agents),
                    "rho": float(rho),
                    "maximum_delay": int(maximum_delay),
                    "target_fraction": float(target_fraction),
                    **mixing,
                    **constants,
                    "omega": float(omega),
                    "innovation_bound": float(innovation),
                    "initial_error": initial_error,
                    "update_cost": int(cost),
                    "updates": int(updates),
                    "resource_budget": int(resource_budget),
                    **step,
                }
            )
    return tuple(actions)


def select_affine_action(
    actions: Iterable[Dict[str, float]],
    restricted_q: Optional[int] = None,
) -> Dict[str, float]:
    """Choose the minimum proved finite-time bound, including a no-op check."""

    eligible = [
        row
        for row in actions
        if restricted_q is None or int(row["num_agents"]) == restricted_q
    ]
    if not eligible:
        raise ValueError("no eligible affine action")
    selected = min(
        eligible,
        key=lambda row: (
            row["finite_time_bound"],
            row["num_agents"],
            row["gap"],
            row["eta"],
        ),
    ).copy()
    selected["beats_no_update_bound"] = bool(
        selected["finite_time_bound"] < selected["initial_error"]
    )
    return selected

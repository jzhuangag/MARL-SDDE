"""Certificate sensitivity and finite-horizon controller ablations."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "experiments" / "dependence_delay_linear"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from affine_markov_certificate import (  # noqa: E402
    affine_bound_components,
    affine_candidate_actions,
    affine_finite_time_bound,
    first_affine_stability_boundary,
    optimize_affine_step,
)
from multistate_certificate import SERVER_OVERHEAD  # noqa: E402


def inflated_delta(delta: float, log_rate_weakening: float) -> float:
    """Inflate a mixing remainder by weakening its certified log-decay rate."""

    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must lie in [0,1]")
    if not 0.0 <= log_rate_weakening < 1.0:
        raise ValueError("log-rate weakening must lie in [0,1)")
    if delta in (0.0, 1.0):
        return float(delta)
    return float(delta ** (1.0 - log_rate_weakening))


def refit_conservative_action(
    action: Mapping[str, float],
    *,
    curvature_factor: float,
    noise_factor: float,
    log_rate_weakening: float,
) -> Dict[str, float]:
    """Re-optimize eta under simultaneous conservative upper certificates."""

    if curvature_factor < 1.0 or noise_factor < 1.0:
        raise ValueError("moment factors must be at least one")
    delta_plus = inflated_delta(
        float(action["joint_delta"]), float(log_rate_weakening)
    )
    lipschitz = float(action["lipschitz"])
    monotonicity = float(action["monotonicity"])
    constants = {
        "curvature": curvature_factor * float(action["curvature"]),
        "monotonicity": monotonicity,
        "lipschitz": lipschitz,
        "effective_monotonicity": monotonicity - 2.0 * lipschitz * delta_plus,
        "effective_curvature": curvature_factor * float(action["curvature"])
        + 2.0 * lipschitz * lipschitz * delta_plus,
        "rms_delay": float(action["rms_delay"]),
        "maximum_actual_delay": int(action["maximum_actual_delay"]),
    }
    if constants["effective_monotonicity"] <= 0.0:
        raise ValueError("conservative mixing bound removes monotonicity")
    omega_plus = noise_factor * float(action["omega"])
    step = optimize_affine_step(
        constants,
        omega_plus,
        float(action["innovation_bound"]),
        delta_plus,
        float(action["initial_error"]),
        int(action["updates"]),
    )
    return {
        **dict(action),
        **constants,
        "joint_delta": delta_plus,
        "omega": omega_plus,
        "curvature_factor": float(curvature_factor),
        "noise_factor": float(noise_factor),
        "log_rate_weakening": float(log_rate_weakening),
        **step,
    }


def refit_myopic_action(action: Mapping[str, float]) -> Dict[str, float]:
    """Optimize one replacement-block drift c R0 + d, ignoring horizon."""

    constants = {
        key: action[key]
        for key in (
            "curvature",
            "monotonicity",
            "lipschitz",
            "effective_monotonicity",
            "effective_curvature",
            "rms_delay",
            "maximum_actual_delay",
        )
    }
    omega = float(action["omega"])
    innovation = float(action["innovation_bound"])
    delta = float(action["joint_delta"])
    initial = float(action["initial_error"])
    boundary = first_affine_stability_boundary(constants, omega, innovation, delta)
    minimum = max(boundary * 1e-8, np.finfo(float).tiny)
    maximum = boundary * (1.0 - 1e-9)

    def objective(log_eta: float) -> float:
        components = affine_bound_components(
            constants, omega, innovation, delta, float(np.exp(log_eta))
        )
        return float(components["contraction"] * initial + components["forcing"])

    grid = np.geomspace(minimum, maximum, 81)
    values = np.asarray([objective(float(np.log(value))) for value in grid])
    index = int(np.argmin(values))
    left = minimum if index == 0 else float(grid[index - 1])
    right = maximum if index == len(grid) - 1 else float(grid[index + 1])
    optimized = minimize_scalar(
        objective,
        bounds=(float(np.log(left)), float(np.log(right))),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 200},
    )
    eta = float(np.exp(optimized.x))
    components = affine_bound_components(constants, omega, innovation, delta, eta)
    finite = affine_finite_time_bound(
        initial,
        int(action["updates"]),
        int(action["maximum_actual_delay"]),
        components,
    )
    return {
        **dict(action),
        "eta": eta,
        "affine_boundary": float(boundary),
        "myopic_score": objective(float(optimized.x)),
        **components,
        **finite,
    }


def select_minimum(
    actions: Iterable[Mapping[str, float]], score: str
) -> Dict[str, float]:
    eligible = [row for row in actions if math.isfinite(float(row[score]))]
    if not eligible:
        raise ValueError("no finite eligible action")
    return dict(
        min(
            eligible,
            key=lambda row: (
                float(row[score]),
                int(row["num_agents"]),
                int(row["gap"]),
                float(row["eta"]),
            ),
        )
    )


def correlation_only_q(rho: float, counts: Sequence[int]) -> int:
    """Choose q from the message-only correlation phase objective."""

    return int(
        min(
            counts,
            key=lambda q: (
                (SERVER_OVERHEAD + q) * (rho + (1.0 - rho) / q),
                q,
            ),
        )
    )


def build_policy_actions(
    model: Mapping[str, np.ndarray],
    rho: float,
    maximum_delay: int,
    *,
    resource_budget: int,
    agent_counts: Sequence[int],
    sensitivity_levels: Mapping[str, Mapping[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], Tuple[Dict[str, float], ...]]:
    """Construct full, conservative, myopic, and correlation-only actions."""

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
        conservative = []
        for action in nominal:
            try:
                conservative.append(
                    refit_conservative_action(
                        action,
                        curvature_factor=float(level["curvature_factor"]),
                        noise_factor=float(level["noise_factor"]),
                        log_rate_weakening=float(level["log_rate_weakening"]),
                    )
                )
            except (RuntimeError, ValueError):
                continue
        policies[name] = select_minimum(conservative, "finite_time_bound")

    myopic = tuple(refit_myopic_action(action) for action in nominal)
    policies["one_step_myopic"] = select_minimum(myopic, "myopic_score")

    selected_q = correlation_only_q(float(rho), tuple(agent_counts))
    policies["correlation_only"] = select_minimum(
        (row for row in nominal if int(row["num_agents"]) == selected_q),
        "finite_time_bound",
    )
    return policies, nominal


def coefficient_perturbations(
    nominal: Mapping[str, float], conservative: Mapping[str, float]
) -> Dict[str, float]:
    """Return the exact coefficient changes used by the robustness corollary."""

    a = float(nominal["a_delta"])
    a_plus = float(conservative["a_delta"])
    beta = float(nominal["beta_delta"])
    beta_plus = float(conservative["beta_delta"])
    h = float(nominal["h_delay"])
    g = float(nominal["g_delay"])
    if a <= 0.0 or a_plus <= 0.0:
        raise ValueError("positive a coefficients are required")
    delta_c = (math.sqrt(a_plus) + math.sqrt(h)) ** 2 - (
        math.sqrt(a) + math.sqrt(h)
    ) ** 2
    if h == 0.0:
        delta_d_bound = abs(beta_plus - beta)
    else:
        delta_d_bound = (
            (1.0 + math.sqrt(h / a_plus)) * abs(beta_plus - beta)
            + math.sqrt(h) * abs(1.0 / math.sqrt(a_plus) - 1.0 / math.sqrt(a)) * beta
            + abs(math.sqrt(a_plus) - math.sqrt(a)) * g / math.sqrt(h)
        )
    return {
        "delta_a": a_plus - a,
        "delta_beta": beta_plus - beta,
        "delta_c": delta_c,
        "delta_d_bound": delta_d_bound,
    }


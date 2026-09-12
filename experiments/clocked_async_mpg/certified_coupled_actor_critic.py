"""Observable conditional Lyapunov action for coupled actor--critic updates.

The inputs are predictable or packet-observable certificates.  This module
does not estimate those certificates; their Markov-data construction is a
separate theorem obligation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .coupled_actor_critic_drift import (
    box_qp_objective,
    solve_two_dimensional_box_qp,
)


@dataclass(frozen=True)
class CertifiedCoupledDecision:
    alpha: float
    beta: float
    certified_drift: float
    gradient_alignment_lower: float
    actor_error_radius: float
    critic_radius: float
    next_critic_radius: float
    critic_response: float
    effective_actor_margin: float
    linear: np.ndarray
    quadratic: np.ndarray


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def certified_drift_coefficients(
    *,
    gradient_norm: float,
    actor_error_radius: float,
    critic_radius: float,
    critic_gradient_radius: float,
    critic_contraction: float,
    target_sensitivity: float,
    block_smoothness: float,
    history_curvature: float,
    critic_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Construct the exact quadratic upper bound for the certificate drift.

    On a simultaneous confidence event assume

    ``||true block gradient - packet gradient|| <= actor_error_radius``

    and one critic correction obeys

    ``||e - beta * critic_gradient||
       <= (1-mu*beta)*critic_radius
          + beta*critic_gradient_radius``.

    Applying an actor step moves the critic fixed point by at most
    ``target_sensitivity * alpha * gradient_norm``.
    """

    gradient = _nonnegative("gradient_norm", gradient_norm)
    actor_radius = _nonnegative("actor_error_radius", actor_error_radius)
    critic = _nonnegative("critic_radius", critic_radius)
    critic_gradient = _nonnegative("critic_gradient_radius", critic_gradient_radius)
    contraction = _nonnegative("critic_contraction", critic_contraction)
    target = _nonnegative("target_sensitivity", target_sensitivity)
    smoothness = _nonnegative("block_smoothness", block_smoothness)
    history = _nonnegative("history_curvature", history_curvature)
    weight = _nonnegative("critic_weight", critic_weight)
    if contraction == 0.0 or smoothness == 0.0 or weight == 0.0:
        raise ValueError("contraction, smoothness and critic weight must be positive")

    alignment = gradient * (gradient - actor_radius)
    critic_response = critic_gradient - contraction * critic
    target_response = target * gradient
    actor_curvature = (smoothness + history) * gradient**2
    linear = np.asarray(
        [
            -alignment + 2.0 * weight * critic * target_response,
            2.0 * weight * critic * critic_response,
        ],
        dtype=float,
    )
    response = np.asarray([target_response, critic_response], dtype=float)
    quadratic = np.diag([actor_curvature, 0.0]) + 2.0 * weight * np.outer(
        response, response
    )
    diagnostics = {
        "alignment": alignment,
        "critic_response": critic_response,
        "effective_actor_margin": actor_radius + 2.0 * weight * critic * target,
    }
    return linear, quadratic, diagnostics


def next_critic_radius(
    *,
    alpha: float,
    beta: float,
    gradient_norm: float,
    critic_radius: float,
    critic_gradient_radius: float,
    critic_contraction: float,
    critic_smoothness: float,
    target_sensitivity: float,
) -> float:
    """Propagate the pathwise critic tracking certificate."""

    alpha = _nonnegative("alpha", alpha)
    beta = _nonnegative("beta", beta)
    gradient = _nonnegative("gradient_norm", gradient_norm)
    critic = _nonnegative("critic_radius", critic_radius)
    critic_gradient = _nonnegative("critic_gradient_radius", critic_gradient_radius)
    contraction = _nonnegative("critic_contraction", critic_contraction)
    smoothness = _nonnegative("critic_smoothness", critic_smoothness)
    target = _nonnegative("target_sensitivity", target_sensitivity)
    if contraction == 0.0 or smoothness == 0.0:
        raise ValueError("critic_contraction and critic_smoothness must be positive")
    if contraction > smoothness + 1e-12:
        raise ValueError("critic_contraction cannot exceed critic_smoothness")
    if beta * smoothness > 1.0 + 1e-12:
        raise ValueError("beta exceeds the SPD contraction interval")
    result = (
        (1.0 - contraction * beta) * critic
        + beta * critic_gradient
        + target * alpha * gradient
    )
    if result < -1e-12:
        raise AssertionError("critic radius recursion became negative")
    return float(max(0.0, result))


def certified_drift_upper(
    *,
    alpha: float,
    beta: float,
    gradient_norm: float,
    actor_error_radius: float,
    critic_radius: float,
    critic_gradient_radius: float,
    critic_contraction: float,
    target_sensitivity: float,
    block_smoothness: float,
    history_curvature: float,
    critic_weight: float,
) -> float:
    """Evaluate the smooth actor plus squared-certificate drift bound."""

    linear, quadratic, _ = certified_drift_coefficients(
        gradient_norm=gradient_norm,
        actor_error_radius=actor_error_radius,
        critic_radius=critic_radius,
        critic_gradient_radius=critic_gradient_radius,
        critic_contraction=critic_contraction,
        target_sensitivity=target_sensitivity,
        block_smoothness=block_smoothness,
        history_curvature=history_curvature,
        critic_weight=critic_weight,
    )
    return box_qp_objective(
        np.asarray([float(alpha), float(beta)]), linear, quadratic
    )


def choose_certified_coupled_scales(
    *,
    alpha_cap: float,
    beta_cap: float,
    critic_smoothness: float,
    **certificate_inputs: float,
) -> CertifiedCoupledDecision:
    """Minimize the observable two-action Lyapunov certificate."""

    alpha_cap = _nonnegative("alpha_cap", alpha_cap)
    beta_cap = _nonnegative("beta_cap", beta_cap)
    contraction = float(certificate_inputs["critic_contraction"])
    smoothness = _nonnegative("critic_smoothness", critic_smoothness)
    if contraction <= 0.0:
        raise ValueError("critic_contraction must be positive")
    if contraction > smoothness + 1e-12:
        raise ValueError("critic_contraction cannot exceed critic_smoothness")
    if beta_cap * smoothness > 1.0 + 1e-12:
        raise ValueError("beta_cap exceeds the SPD contraction interval")
    linear, quadratic, diagnostics = certified_drift_coefficients(
        **certificate_inputs
    )
    decision = solve_two_dimensional_box_qp(
        linear=linear,
        quadratic=quadratic,
        lower=np.zeros(2),
        upper=np.asarray([alpha_cap, beta_cap]),
    )
    alpha, beta = map(float, decision.action)
    propagated = next_critic_radius(
        alpha=alpha,
        beta=beta,
        gradient_norm=certificate_inputs["gradient_norm"],
        critic_radius=certificate_inputs["critic_radius"],
        critic_gradient_radius=certificate_inputs["critic_gradient_radius"],
        critic_contraction=certificate_inputs["critic_contraction"],
        critic_smoothness=smoothness,
        target_sensitivity=certificate_inputs["target_sensitivity"],
    )
    return CertifiedCoupledDecision(
        alpha=alpha,
        beta=beta,
        certified_drift=float(decision.objective),
        gradient_alignment_lower=diagnostics["alignment"],
        actor_error_radius=float(certificate_inputs["actor_error_radius"]),
        critic_radius=float(certificate_inputs["critic_radius"]),
        next_critic_radius=propagated,
        critic_response=diagnostics["critic_response"],
        effective_actor_margin=diagnostics["effective_actor_margin"],
        linear=linear.copy(),
        quadratic=quadratic.copy(),
    )


def clipped_margin_progress_lower(
    *,
    gradient_norm: float,
    actor_error_radius: float,
    critic_radius: float,
    target_sensitivity: float,
    block_smoothness: float,
    history_curvature: float,
    critic_weight: float,
    alpha_cap: float,
) -> float:
    """Lower-bound improvement available from the beta=0 comparator.

    This is the finite-time bridge used after telescoping.  It is zero when
    the observed packet signal does not exceed statistical, strategic and
    critic-target debt.
    """

    gradient = _nonnegative("gradient_norm", gradient_norm)
    actor_radius = _nonnegative("actor_error_radius", actor_error_radius)
    critic = _nonnegative("critic_radius", critic_radius)
    target = _nonnegative("target_sensitivity", target_sensitivity)
    smoothness = _nonnegative("block_smoothness", block_smoothness)
    history = _nonnegative("history_curvature", history_curvature)
    weight = _nonnegative("critic_weight", critic_weight)
    cap = _nonnegative("alpha_cap", alpha_cap)
    curvature = smoothness + history + 2.0 * weight * target**2
    margin = max(0.0, gradient - actor_radius - 2.0 * weight * critic * target)
    clipped_scale = min(cap, 1.0 / curvature)
    return float(0.5 * clipped_scale * margin**2)

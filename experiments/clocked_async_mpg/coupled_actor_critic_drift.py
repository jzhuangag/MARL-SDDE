"""Exact two-action Lyapunov drift for a delayed multi-agent actor--critic toy.

This module is an outcome-free algebraic feasibility interface.  It is not a
neural MARL implementation and it never treats latent actor/critic error as an
observable certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BoxQPDecision:
    action: np.ndarray
    objective: float
    candidates_checked: int


@dataclass(frozen=True)
class CoupledDriftDecision:
    alpha: float
    beta: float
    expected_drift: float
    packet_gradient: float
    current_gradient: float
    strategic_staleness: float
    critic_bias: float
    quadratic: np.ndarray
    linear: np.ndarray
    cross_curvature: float
    candidates_checked: int


def _finite_vector(name: str, value: np.ndarray, size: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if size is not None and result.size != size:
        raise ValueError(f"{name} has incompatible dimension")
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_symmetric(name: str, value: np.ndarray, size: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if size is not None and result.shape != (size, size):
        raise ValueError(f"{name} has incompatible dimension")
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(result, result.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return result


def box_qp_objective(action: np.ndarray, linear: np.ndarray, quadratic: np.ndarray) -> float:
    """Evaluate ``linear @ u + 0.5 * u.T @ quadratic @ u``."""

    linear = _finite_vector("linear", linear)
    action = _finite_vector("action", action, linear.size)
    quadratic = _finite_symmetric("quadratic", quadratic, linear.size)
    return float(linear @ action + 0.5 * action @ quadratic @ action)


def solve_two_dimensional_box_qp(
    *,
    linear: np.ndarray,
    quadratic: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    tolerance: float = 1e-11,
) -> BoxQPDecision:
    """Globally solve a convex two-variable box QP by active-set enumeration."""

    linear = _finite_vector("linear", linear, 2)
    quadratic = _finite_symmetric("quadratic", quadratic, 2)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    eigenvalues = np.linalg.eigvalsh(quadratic)
    if eigenvalues[0] < -tolerance:
        raise ValueError("quadratic must be positive semidefinite")
    lo = np.zeros(2) if lower is None else _finite_vector("lower", lower, 2)
    hi = np.ones(2) if upper is None else _finite_vector("upper", upper, 2)
    if np.any(lo > hi):
        raise ValueError("lower must not exceed upper")

    candidates: list[np.ndarray] = []

    def add(value: np.ndarray) -> None:
        candidate = np.asarray(value, dtype=float)
        if np.all(candidate >= lo - tolerance) and np.all(candidate <= hi + tolerance):
            candidate = np.clip(candidate, lo, hi)
            if not any(np.allclose(candidate, old, atol=tolerance, rtol=0.0) for old in candidates):
                candidates.append(candidate)

    stationary = -np.linalg.pinv(quadratic, rcond=tolerance) @ linear
    residual = quadratic @ stationary + linear
    if np.linalg.norm(residual) <= 10.0 * tolerance:
        add(stationary)

    for first in (lo[0], hi[0]):
        slope = linear[1] + quadratic[1, 0] * first
        if quadratic[1, 1] > tolerance:
            second = float(np.clip(-slope / quadratic[1, 1], lo[1], hi[1]))
            add(np.asarray([first, second]))
        else:
            add(np.asarray([first, lo[1] if slope >= 0.0 else hi[1]]))
    for second in (lo[1], hi[1]):
        slope = linear[0] + quadratic[0, 1] * second
        if quadratic[0, 0] > tolerance:
            first = float(np.clip(-slope / quadratic[0, 0], lo[0], hi[0]))
            add(np.asarray([first, second]))
        else:
            add(np.asarray([lo[0] if slope >= 0.0 else hi[0], second]))
    for first in (lo[0], hi[0]):
        for second in (lo[1], hi[1]):
            add(np.asarray([first, second]))

    if not candidates:
        raise RuntimeError("box QP candidate enumeration failed")
    values = [box_qp_objective(value, linear, quadratic) for value in candidates]
    index = int(np.argmin(values))
    return BoxQPDecision(
        action=candidates[index].copy(),
        objective=float(values[index]),
        candidates_checked=len(candidates),
    )


def coupled_drift_coefficients(
    *,
    actor_error: np.ndarray,
    birth_actor_snapshot: np.ndarray,
    game_hessian: np.ndarray,
    owner: int,
    critic_error: float,
    critic_bias_sensitivity: float,
    critic_target_sensitivity: float,
    critic_contraction: float,
    critic_weight: float,
    actor_noise_variance: float,
    critic_noise_variance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return the exact conditional quadratic drift coefficients.

    The actor noise moves actor ``owner`` and, through the moving critic target,
    the critic error.  The critic innovation moves only the critic error.
    """

    actor_error = _finite_vector("actor_error", actor_error)
    birth = _finite_vector("birth_actor_snapshot", birth_actor_snapshot, actor_error.size)
    hessian = _finite_symmetric("game_hessian", game_hessian, actor_error.size)
    if owner < 0 or owner >= actor_error.size:
        raise ValueError("owner is outside the actor block range")
    if np.linalg.eigvalsh(hessian)[0] <= 0.0:
        raise ValueError("game_hessian must be positive definite")
    scalars = (
        critic_error,
        critic_bias_sensitivity,
        critic_target_sensitivity,
        critic_contraction,
        critic_weight,
        actor_noise_variance,
        critic_noise_variance,
    )
    if any(not math.isfinite(value) for value in scalars):
        raise ValueError("all scalar inputs must be finite")
    if critic_contraction <= 0.0 or critic_weight <= 0.0:
        raise ValueError("critic contraction and weight must be positive")
    if actor_noise_variance < 0.0 or critic_noise_variance < 0.0:
        raise ValueError("noise variances must be nonnegative")

    current_gradient = float(hessian[owner] @ actor_error)
    packet_gradient = float(
        hessian[owner, owner] * actor_error[owner]
        + np.delete(hessian[owner], owner) @ np.delete(birth, owner)
        + critic_bias_sensitivity * critic_error
    )
    strategic_staleness = float(
        np.delete(hessian[owner], owner)
        @ np.delete(birth - actor_error, owner)
    )
    critic_bias = float(critic_bias_sensitivity * critic_error)

    dimension = actor_error.size + 1
    state = np.concatenate([actor_error, np.asarray([critic_error])])
    metric = np.zeros((dimension, dimension), dtype=float)
    metric[:-1, :-1] = hessian
    metric[-1, -1] = critic_weight

    actor_response = np.zeros(dimension, dtype=float)
    actor_response[owner] = -packet_gradient
    actor_response[-1] = critic_target_sensitivity * packet_gradient
    critic_response = np.zeros(dimension, dtype=float)
    critic_response[-1] = -critic_contraction * critic_error
    responses = np.column_stack([actor_response, critic_response])

    actor_noise_response = np.zeros(dimension, dtype=float)
    actor_noise_response[owner] = -1.0
    actor_noise_response[-1] = critic_target_sensitivity
    critic_noise_response = np.zeros(dimension, dtype=float)
    critic_noise_response[-1] = 1.0
    noise_curvature = np.diag(
        [
            actor_noise_variance * float(actor_noise_response @ metric @ actor_noise_response),
            critic_noise_variance * float(critic_noise_response @ metric @ critic_noise_response),
        ]
    )

    linear = 2.0 * responses.T @ metric @ state
    quadratic = 2.0 * (responses.T @ metric @ responses + noise_curvature)
    diagnostics = {
        "packet_gradient": packet_gradient,
        "current_gradient": current_gradient,
        "strategic_staleness": strategic_staleness,
        "critic_bias": critic_bias,
        "initial_lyapunov": float(state @ metric @ state),
    }
    return np.asarray(linear), np.asarray(quadratic), diagnostics


def choose_coupled_actor_critic_scales(
    *,
    alpha_cap: float,
    beta_cap: float,
    **coefficient_inputs: object,
) -> CoupledDriftDecision:
    """Choose the exact actor/critic scales from the composite drift QP."""

    if not math.isfinite(alpha_cap) or not math.isfinite(beta_cap):
        raise ValueError("action caps must be finite")
    if alpha_cap < 0.0 or beta_cap < 0.0:
        raise ValueError("action caps must be nonnegative")
    linear, quadratic, diagnostics = coupled_drift_coefficients(**coefficient_inputs)
    decision = solve_two_dimensional_box_qp(
        linear=linear,
        quadratic=quadratic,
        lower=np.zeros(2),
        upper=np.asarray([alpha_cap, beta_cap]),
    )
    return CoupledDriftDecision(
        alpha=float(decision.action[0]),
        beta=float(decision.action[1]),
        expected_drift=float(decision.objective),
        packet_gradient=diagnostics["packet_gradient"],
        current_gradient=diagnostics["current_gradient"],
        strategic_staleness=diagnostics["strategic_staleness"],
        critic_bias=diagnostics["critic_bias"],
        quadratic=quadratic.copy(),
        linear=linear.copy(),
        cross_curvature=float(quadratic[0, 1]),
        candidates_checked=decision.candidates_checked,
    )


def direct_expected_drift(
    *,
    alpha: float,
    beta: float,
    **coefficient_inputs: object,
) -> float:
    """Evaluate the conditional drift from the state transition itself."""

    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError("actions must be finite")
    actor_error = _finite_vector("actor_error", np.asarray(coefficient_inputs["actor_error"]))
    birth = _finite_vector(
        "birth_actor_snapshot",
        np.asarray(coefficient_inputs["birth_actor_snapshot"]),
        actor_error.size,
    )
    hessian = _finite_symmetric(
        "game_hessian", np.asarray(coefficient_inputs["game_hessian"]), actor_error.size
    )
    owner = int(coefficient_inputs["owner"])
    if owner < 0 or owner >= actor_error.size:
        raise ValueError("owner is outside the actor block range")
    critic_error = float(coefficient_inputs["critic_error"])
    bias = float(coefficient_inputs["critic_bias_sensitivity"])
    target = float(coefficient_inputs["critic_target_sensitivity"])
    contraction = float(coefficient_inputs["critic_contraction"])
    weight = float(coefficient_inputs["critic_weight"])
    actor_variance = float(coefficient_inputs["actor_noise_variance"])
    critic_variance = float(coefficient_inputs["critic_noise_variance"])

    packet_gradient = float(
        hessian[owner, owner] * actor_error[owner]
        + np.delete(hessian[owner], owner) @ np.delete(birth, owner)
        + bias * critic_error
    )
    next_actor = actor_error.copy()
    next_actor[owner] -= alpha * packet_gradient
    next_critic = (
        critic_error
        - beta * contraction * critic_error
        + alpha * target * packet_gradient
    )
    initial = float(actor_error @ hessian @ actor_error + weight * critic_error**2)
    deterministic_next = float(
        next_actor @ hessian @ next_actor + weight * next_critic**2
    )
    actor_noise_curvature = float(
        hessian[owner, owner] + weight * target**2
    )
    noise = (
        alpha**2 * actor_variance * actor_noise_curvature
        + beta**2 * critic_variance * weight
    )
    return deterministic_next + noise - initial

"""Algebraic prototype for online joint participation--gain drift control.

The coefficients passed to this module are assumed to come from a valid
Lyapunov/Markov certificate.  This file proves no such certificate; it tests
the exact low-dimensional optimization structure that the theorem would use.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class JointDriftParameters:
    """Nonnegative coefficients of the robust one-block drift bound."""

    contraction: float
    state_signal: float
    delay_curvature: float
    noise_coefficient: float
    noise_scale: float
    rho_upper: float
    message_price: float
    environment_price: float
    overhead: float
    eta_min: float
    eta_max: float

    def validate(self) -> None:
        values = (
            self.contraction,
            self.state_signal,
            self.delay_curvature,
            self.noise_coefficient,
            self.noise_scale,
            self.message_price,
            self.environment_price,
            self.overhead,
            self.eta_min,
            self.eta_max,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("drift coefficients and prices must be finite and nonnegative")
        if not 0.0 <= self.rho_upper <= 1.0:
            raise ValueError("rho_upper must lie in [0, 1]")
        if self.eta_min > self.eta_max:
            raise ValueError("eta_min cannot exceed eta_max")


@dataclass(frozen=True)
class JointAction:
    participation: int
    gain: float
    drift_score: float
    continuous_participation: float
    integer_candidates: tuple[int, ...]


def _reduced_coefficients(parameters: JointDriftParameters) -> tuple[float, float, float, float]:
    """Return r, u, v, and the aggregate marginal resource price.

    The score is

        (u + v / q) eta^2 - r eta + price q + message_price overhead.
    """

    parameters.validate()
    r = parameters.contraction * parameters.state_signal
    u = (
        parameters.delay_curvature * parameters.state_signal
        + parameters.noise_coefficient
        * parameters.noise_scale
        * parameters.rho_upper
    )
    v = (
        parameters.noise_coefficient
        * parameters.noise_scale
        * (1.0 - parameters.rho_upper)
    )
    price = parameters.message_price + parameters.environment_price
    return r, u, v, price


def optimal_gain_for_participation(
    participation: float, parameters: JointDriftParameters
) -> float:
    """Exactly minimize the certified quadratic score for a fixed positive q."""

    if not math.isfinite(participation) or participation <= 0.0:
        raise ValueError("participation must be finite and positive")
    r, u, v, _ = _reduced_coefficients(parameters)
    curvature = u + v / participation
    if curvature == 0.0:
        unconstrained = parameters.eta_max if r > 0.0 else parameters.eta_min
    else:
        unconstrained = r / (2.0 * curvature)
    return min(parameters.eta_max, max(parameters.eta_min, unconstrained))


def joint_drift_score(
    participation: float, gain: float, parameters: JointDriftParameters
) -> float:
    """Evaluate the robust composite-Lyapunov drift score."""

    if not math.isfinite(participation) or participation <= 0.0:
        raise ValueError("participation must be finite and positive")
    if not parameters.eta_min <= gain <= parameters.eta_max:
        raise ValueError("gain lies outside the certified interval")
    r, u, v, price = _reduced_coefficients(parameters)
    return float(
        (u + v / participation) * gain * gain
        - r * gain
        + price * participation
        + parameters.message_price * parameters.overhead
    )


def _profile_derivative(participation: float, parameters: JointDriftParameters) -> float:
    """Envelope derivative after exactly minimizing over eta."""

    _, _, v, price = _reduced_coefficients(parameters)
    gain = optimal_gain_for_participation(participation, parameters)
    return price - v * gain * gain / (participation * participation)


def continuous_joint_action(
    *, q_min: int, q_max: int, parameters: JointDriftParameters
) -> tuple[float, float]:
    """Globally minimize the continuous convex relaxation by monotone bisection."""

    if int(q_min) != q_min or int(q_max) != q_max or q_min < 1 or q_min > q_max:
        raise ValueError("q_min and q_max must be ordered positive integers")
    lower = float(q_min)
    upper = float(q_max)
    if _profile_derivative(lower, parameters) >= 0.0:
        q_star = lower
    elif _profile_derivative(upper, parameters) <= 0.0:
        q_star = upper
    else:
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            if _profile_derivative(middle, parameters) < 0.0:
                lower = middle
            else:
                upper = middle
        q_star = 0.5 * (lower + upper)
    eta_star = optimal_gain_for_participation(q_star, parameters)
    return q_star, eta_star


def exact_integer_joint_action(
    *, q_min: int, q_max: int, parameters: JointDriftParameters
) -> JointAction:
    """Recover the exact integer action from the continuous convex optimum.

    The profiled objective is convex in q.  Therefore an integer minimizer over
    the contiguous feasible set lies at floor(q*) or ceil(q*), after clipping
    to the feasible endpoints.
    """

    q_continuous, _ = continuous_joint_action(
        q_min=q_min, q_max=q_max, parameters=parameters
    )
    candidates = tuple(
        sorted(
            {
                max(q_min, min(q_max, int(math.floor(q_continuous)))),
                max(q_min, min(q_max, int(math.ceil(q_continuous)))),
            }
        )
    )
    evaluated = []
    for q in candidates:
        eta = optimal_gain_for_participation(float(q), parameters)
        evaluated.append((joint_drift_score(float(q), eta, parameters), q, eta))
    score, q_star, eta_star = min(evaluated, key=lambda row: (row[0], row[1], row[2]))
    return JointAction(
        participation=int(q_star),
        gain=float(eta_star),
        drift_score=float(score),
        continuous_participation=float(q_continuous),
        integer_candidates=candidates,
    )


def brute_force_integer_joint_action(
    *, q_min: int, q_max: int, parameters: JointDriftParameters
) -> JointAction:
    """Audit-only exhaustive solution over every feasible integer q."""

    if int(q_min) != q_min or int(q_max) != q_max or q_min < 1 or q_min > q_max:
        raise ValueError("q_min and q_max must be ordered positive integers")
    evaluated = []
    for q in range(q_min, q_max + 1):
        eta = optimal_gain_for_participation(float(q), parameters)
        evaluated.append((joint_drift_score(float(q), eta, parameters), q, eta))
    score, q_star, eta_star = min(evaluated, key=lambda row: (row[0], row[1], row[2]))
    return JointAction(
        participation=int(q_star),
        gain=float(eta_star),
        drift_score=float(score),
        continuous_participation=float(q_star),
        integer_candidates=tuple(range(q_min, q_max + 1)),
    )

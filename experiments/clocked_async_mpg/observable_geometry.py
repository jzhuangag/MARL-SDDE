"""Observable operator-error interface for clocked optimism certificates.

This module does not estimate a Markov-game Jacobian.  It performs the next
deterministic step: converting a valid spectral-norm operator confidence ball
into a conservative lower bound on the log-energy value of a fresh
extra-gradient oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .clocked_optimism_phase import expected_quadratic_multiplier


@dataclass(frozen=True)
class OperatorLogGainCertificate:
    plain_multiplier_estimate: float
    fresh_multiplier_estimate: float
    plain_multiplier_radius: float
    fresh_multiplier_radius: float
    plain_multiplier_lower: float
    fresh_multiplier_upper: float
    log_gain_lower: float

    @property
    def certifies_positive_gain(self) -> bool:
        return self.log_gain_lower > 0.0


def coordinate_game_transitions(
    operator: np.ndarray,
    *,
    step: float,
    use_fresh_anchor: bool,
) -> tuple[np.ndarray, ...]:
    """Return one current-parameter coordinate transition per agent block."""

    operator = np.asarray(operator, dtype=float)
    if (
        operator.ndim != 2
        or operator.shape[0] != operator.shape[1]
        or operator.shape[0] == 0
        or not np.all(np.isfinite(operator))
    ):
        raise ValueError("operator must be a finite nonempty square matrix")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive")
    identity = np.eye(operator.shape[0])
    lookahead = identity - step * operator
    transitions = []
    for agent in range(operator.shape[0]):
        selector = np.zeros_like(operator)
        selector[agent, agent] = 1.0
        direction = operator @ lookahead if use_fresh_anchor else operator
        transitions.append(identity - step * selector @ direction)
    return tuple(transitions)


def certified_operator_log_gain(
    operator_estimate: np.ndarray,
    *,
    operator_error_radius: float,
    operator_norm_bound: float,
    metric: np.ndarray,
    arrival_probabilities: tuple[float, ...],
    step: float,
) -> OperatorLogGainCertificate:
    """Lower-bound ``log(q_plain) - log(q_fresh)`` over an operator ball.

    The caller must establish both ``||A - A_hat||_2 <= epsilon`` and
    ``max(||A||_2, ||A_hat||_2) <= L`` on the event where the certificate is
    claimed.  This function then applies a deterministic matrix perturbation
    bound; it does not assume how that confidence event was obtained.
    """

    estimate = np.asarray(operator_estimate, dtype=float)
    metric = np.asarray(metric, dtype=float)
    if (
        estimate.ndim != 2
        or estimate.shape[0] != estimate.shape[1]
        or estimate.shape[0] == 0
        or not np.all(np.isfinite(estimate))
    ):
        raise ValueError("operator estimate must be a finite square matrix")
    if metric.shape != estimate.shape:
        raise ValueError("metric must match the operator")
    if len(arrival_probabilities) != estimate.shape[0]:
        raise ValueError("one arrival probability is required per coordinate")
    if (
        not math.isfinite(operator_error_radius)
        or not math.isfinite(operator_norm_bound)
        or not math.isfinite(step)
        or operator_error_radius < 0.0
        or operator_norm_bound <= 0.0
        or step <= 0.0
    ):
        raise ValueError("invalid radius, norm bound, or step")
    if np.linalg.norm(estimate, ord=2) > operator_norm_bound + 1e-12:
        raise ValueError("operator estimate exceeds the declared norm bound")

    metric_eigenvalues = np.linalg.eigvalsh(0.5 * (metric + metric.T))
    if np.min(metric_eigenvalues) <= 0.0:
        raise ValueError("metric must be positive definite")
    condition = float(np.max(metric_eigenvalues) / np.min(metric_eigenvalues))
    probabilities = tuple(float(value) for value in arrival_probabilities)
    plain = expected_quadratic_multiplier(
        metric,
        coordinate_game_transitions(estimate, step=step, use_fresh_anchor=False),
        probabilities,
    )
    fresh = expected_quadratic_multiplier(
        metric,
        coordinate_game_transitions(estimate, step=step, use_fresh_anchor=True),
        probabilities,
    )

    plain_transition_norm = 1.0 + step * operator_norm_bound
    plain_transition_error = step * operator_error_radius
    fresh_transition_norm = (
        1.0
        + step * operator_norm_bound
        + step * step * operator_norm_bound * operator_norm_bound
    )
    fresh_transition_error = step * operator_error_radius * (
        1.0 + 2.0 * step * operator_norm_bound
    )
    plain_radius = (
        2.0 * condition * plain_transition_norm * plain_transition_error
    )
    fresh_radius = (
        2.0 * condition * fresh_transition_norm * fresh_transition_error
    )
    plain_lower = plain - plain_radius
    fresh_upper = fresh + fresh_radius
    lower = (
        math.log(plain_lower) - math.log(fresh_upper)
        if plain_lower > 0.0 and fresh_upper > 0.0
        else -math.inf
    )
    return OperatorLogGainCertificate(
        plain_multiplier_estimate=plain,
        fresh_multiplier_estimate=fresh,
        plain_multiplier_radius=plain_radius,
        fresh_multiplier_radius=fresh_radius,
        plain_multiplier_lower=plain_lower,
        fresh_multiplier_upper=fresh_upper,
        log_gain_lower=lower,
    )

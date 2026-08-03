"""Exact finite-horizon risk for time-varying agent participation.

The learning recursion is the delayed additive linear stochastic approximation

    e[t+1] = e[t] - eta A e[t-D] + eta xi_bar[t].

At time ``t`` the learner averages the prefix set ``{1, ..., q[t]}``.  Every
agent innovation is the sum of a common stationary Markov component and an
independent private component with the same lag-covariance sequence.  This
module keeps the resulting nonstationary cross-time covariance exactly; it
does not replace a time-varying schedule by an effective sample size.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from experiments.dependence_delay_linear.t037_vector_markov_phase import (
    delayed_vector_companion,
)


@dataclass(frozen=True)
class AffineRisk:
    """Coefficients of ``risk(rho) = intercept + slope * rho``."""

    intercept: float
    slope: float

    def evaluate(self, rho: float) -> float:
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [0, 1]")
        return float(self.intercept + self.slope * rho)


def prefix_overlap_factor(q_left: int, q_right: int, rho: float) -> float:
    """Cross-time covariance multiplier for two prefix participation sets."""

    if q_left < 1 or q_right < 1:
        raise ValueError("participation counts must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    private_overlap = min(q_left, q_right) / float(q_left * q_right)
    return float(rho + (1.0 - rho) * private_overlap)


def schedule_budget_summary(
    q_schedule: Sequence[int],
    *,
    message_overhead: float,
    per_agent_message: float,
    stride: int,
    delay: int,
) -> dict[str, float | int]:
    """Return the exact resources charged by a completed update schedule."""

    schedule = tuple(int(q) for q in q_schedule)
    if any(q < 1 for q in schedule):
        raise ValueError("participation counts must be positive")
    if message_overhead < 0.0 or per_agent_message <= 0.0:
        raise ValueError("message costs must be nonnegative with positive payload")
    if stride < 1 or delay < 0:
        raise ValueError("stride must be positive and delay nonnegative")
    message = sum(message_overhead + per_agent_message * q for q in schedule)
    environment = len(schedule) * stride + delay
    return {
        "updates": len(schedule),
        "message": float(message),
        "environment": int(environment),
    }


def schedule_is_feasible(
    q_schedule: Sequence[int],
    *,
    message_budget: float,
    environment_budget: int,
    message_overhead: float,
    per_agent_message: float,
    stride: int,
    delay: int,
) -> bool:
    """Check both resource budgets without rounding fractional message costs."""

    if message_budget < 0.0 or environment_budget < 0:
        raise ValueError("budgets must be nonnegative")
    used = schedule_budget_summary(
        q_schedule,
        message_overhead=message_overhead,
        per_agent_message=per_agent_message,
        stride=stride,
        delay=delay,
    )
    return bool(
        float(used["message"]) <= message_budget
        and int(used["environment"]) <= environment_budget
    )


def exact_scheduled_vector_risk(
    *,
    initial_history: np.ndarray,
    drift: np.ndarray,
    step_size: float,
    delay: int,
    q_schedule: Sequence[int],
    rho: float,
    base_lag_covariances: np.ndarray | None,
    risk_matrix: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Compute exact risk for a deterministic time-varying prefix schedule.

    ``base_lag_covariances[T-1+k]`` stores the covariance of one common or
    private component at lag ``k``.  For schedule entries ``q[s]`` and
    ``q[r]``, the aggregate covariance is exactly

        [rho + (1-rho) min(q[s],q[r])/(q[s]q[r])] K[s-r].

    The schedule must be selected independently of the learning innovations.
    A probe-selected schedule satisfies this condition when probes and learning
    streams are sample-split.
    """

    matrix = np.asarray(drift, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("drift must be square")
    dimension = matrix.shape[0]
    history = np.asarray(initial_history, dtype=float)
    if history.shape != (delay + 1, dimension):
        raise ValueError("initial_history must have shape (delay+1, dimension)")
    if step_size <= 0.0 or delay < 0:
        raise ValueError("step size must be positive and delay nonnegative")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    schedule = tuple(int(q) for q in q_schedule)
    if any(q < 1 for q in schedule):
        raise ValueError("participation counts must be positive")
    updates = len(schedule)
    weight = np.eye(dimension) if risk_matrix is None else np.asarray(risk_matrix, dtype=float)
    if weight.shape != (dimension, dimension) or not np.allclose(weight, weight.T):
        raise ValueError("risk_matrix must be symmetric with drift dimension")

    companion = delayed_vector_companion(matrix, step_size, delay)
    selector = np.zeros((dimension, dimension * (delay + 1)))
    selector[:, :dimension] = np.eye(dimension)
    injector = selector.T
    lifted_initial = history.reshape(-1)
    terminal_mean = selector @ np.linalg.matrix_power(companion, updates) @ lifted_initial

    terminal_covariance = np.zeros((dimension, dimension))
    if updates:
        if base_lag_covariances is None:
            raise ValueError("base_lag_covariances are required for a nonempty schedule")
        lags = np.asarray(base_lag_covariances, dtype=float)
        expected_shape = (2 * updates - 1, dimension, dimension)
        if lags.shape != expected_shape:
            raise ValueError(f"base_lag_covariances must have shape {expected_shape}")
        impulses = [
            selector @ np.linalg.matrix_power(companion, updates - 1 - time) @ injector
            for time in range(updates)
        ]
        center = updates - 1
        for left in range(updates):
            for right in range(updates):
                factor = prefix_overlap_factor(schedule[left], schedule[right], rho)
                terminal_covariance += (
                    step_size**2
                    * factor
                    * impulses[left]
                    @ lags[center + left - right]
                    @ impulses[right].T
                )
        terminal_covariance = (terminal_covariance + terminal_covariance.T) / 2.0

    bias_risk = float(terminal_mean @ weight @ terminal_mean)
    noise_risk = float(np.trace(weight @ terminal_covariance))
    return {
        "mean": terminal_mean,
        "covariance": terminal_covariance,
        "bias_risk": bias_risk,
        "noise_risk": noise_risk,
        "risk": bias_risk + noise_risk,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(companion)))),
    }


def scheduled_risk_affine_coefficients(
    *,
    initial_history: np.ndarray,
    drift: np.ndarray,
    step_size: float,
    delay: int,
    q_schedule: Sequence[int],
    base_lag_covariances: np.ndarray,
    risk_matrix: np.ndarray | None = None,
) -> AffineRisk:
    """Return the exact affine dependence of a fixed schedule on ``rho``."""

    at_zero = exact_scheduled_vector_risk(
        initial_history=initial_history,
        drift=drift,
        step_size=step_size,
        delay=delay,
        q_schedule=q_schedule,
        rho=0.0,
        base_lag_covariances=base_lag_covariances,
        risk_matrix=risk_matrix,
    )["risk"]
    at_one = exact_scheduled_vector_risk(
        initial_history=initial_history,
        drift=drift,
        step_size=step_size,
        delay=delay,
        q_schedule=q_schedule,
        rho=1.0,
        base_lag_covariances=base_lag_covariances,
        risk_matrix=risk_matrix,
    )["risk"]
    return AffineRisk(float(at_zero), float(at_one) - float(at_zero))


def robust_post_probe_choice(
    risks: Mapping[str, AffineRisk],
    *,
    fallback: str,
    rho_lower: float,
    rho_upper: float,
    improvement_margin: float = 0.0,
) -> str:
    """Choose a schedule only if it uniformly improves on the fallback.

    The guarantee is conditional on the supplied correlation interval and on
    resources remaining after the probe phase.  It does not erase the probe's
    opportunity cost relative to a no-probe policy.
    """

    if fallback not in risks:
        raise ValueError("fallback must be present in risks")
    if not 0.0 <= rho_lower <= rho_upper <= 1.0:
        raise ValueError("invalid correlation interval")
    if improvement_margin < 0.0:
        raise ValueError("improvement_margin must be nonnegative")

    def worst_risk(item: tuple[str, AffineRisk]) -> tuple[float, str]:
        name, risk = item
        return (
            max(risk.evaluate(rho_lower), risk.evaluate(rho_upper)),
            name,
        )

    candidate = min(risks.items(), key=worst_risk)[0]
    if candidate == fallback:
        return fallback
    candidate_risk = risks[candidate]
    fallback_risk = risks[fallback]
    worst_difference = max(
        candidate_risk.evaluate(rho_lower) - fallback_risk.evaluate(rho_lower),
        candidate_risk.evaluate(rho_upper) - fallback_risk.evaluate(rho_upper),
    )
    if worst_difference <= -improvement_margin:
        return candidate
    return fallback


def hoeffding_interval(
    estimate: float, *, trials: int, alpha: float
) -> tuple[float, float]:
    """Anytime-agnostic fixed-sample interval for bounded probe statistics."""

    if not 0.0 <= estimate <= 1.0:
        raise ValueError("estimate must lie in [0, 1]")
    if trials < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("trials must be positive and alpha in (0, 1)")
    radius = math.sqrt(math.log(2.0 / alpha) / (2.0 * trials))
    return max(0.0, estimate - radius), min(1.0, estimate + radius)

"""Scheduled Polyak--Ruppert risk and probe-cost oracle certificates.

The module has three theorem-facing components.

1. An exact finite-horizon risk identity for a tail Polyak--Ruppert average
   under a deterministic time-varying prefix-participation schedule.
2. A mixing-corrected intraclass-correlation interval from a fully charged,
   frozen-parameter probe block.
3. A finite-table robust selector and an exact conditional excess-risk
   certificate relative to a no-probe full-budget oracle or fixed baseline.

The probe block and learning stream must be independent.  Unknown mixing is
not covered: the confidence calculation requires a valid beta-mixing envelope.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from experiments.dependence_delay_linear.t037_vector_markov_phase import (
    delayed_vector_companion,
)
from experiments.dependence_delay_linear.t047_scheduled_participation import (
    AffineRisk,
    prefix_overlap_factor,
)


@dataclass(frozen=True)
class CorrelationCertificate:
    """A fixed-block confidence interval for intraclass correlation."""

    lower: float
    upper: float
    radius: float
    mixing_penalty: float
    alpha: float
    trials: int
    stride: int
    informative: bool
    mean_interval: tuple[float, float]
    variance_interval: tuple[float, float]
    covariance_interval: tuple[float, float]


@dataclass(frozen=True)
class OracleCertificate:
    """Uniform risk bounds on the event covered by a correlation interval."""

    selected: str
    conditional_full_oracle_excess: float
    expected_full_oracle_excess: float
    conditional_baseline_difference: float | None
    expected_baseline_difference: float | None
    failure_probability: float
    risk_cap: float


def exact_scheduled_pr_averaged_vector_risk(
    *,
    initial_history: np.ndarray,
    drift: np.ndarray,
    step_size: float,
    delay: int,
    q_schedule: Sequence[int],
    burn_in: int,
    rho: float,
    base_lag_covariances: np.ndarray | None,
    risk_matrix: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Exact tail-averaged risk for a deterministic prefix schedule.

    The readout is the average of ``e[burn_in+1], ..., e[updates]``.  The
    lag-covariance array stores the covariance of one common or private
    component, before participation aggregation.
    """

    matrix = np.asarray(drift, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("drift must be square")
    dimension = matrix.shape[0]
    history = np.asarray(initial_history, dtype=float)
    if delay < 0 or history.shape != (delay + 1, dimension):
        raise ValueError("initial_history must have shape (delay+1, dimension)")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    schedule = tuple(int(q) for q in q_schedule)
    updates = len(schedule)
    if updates < 1 or any(q < 1 for q in schedule):
        raise ValueError("q_schedule must be nonempty and positive")
    if burn_in < 0 or burn_in >= updates:
        raise ValueError("require 0 <= burn_in < len(q_schedule)")
    weight = (
        np.eye(dimension)
        if risk_matrix is None
        else np.asarray(risk_matrix, dtype=float)
    )
    if weight.shape != (dimension, dimension) or not np.allclose(weight, weight.T):
        raise ValueError("risk_matrix must be symmetric with drift dimension")

    lags = np.asarray(base_lag_covariances, dtype=float)
    expected_shape = (2 * updates - 1, dimension, dimension)
    if base_lag_covariances is None or lags.shape != expected_shape:
        raise ValueError(f"base_lag_covariances must have shape {expected_shape}")

    companion = delayed_vector_companion(matrix, step_size, delay)
    selector = np.zeros((dimension, dimension * (delay + 1)))
    selector[:, :dimension] = np.eye(dimension)
    injector = selector.T
    lifted_initial = history.reshape(-1)
    averaged_count = updates - burn_in

    mean = np.zeros(dimension)
    for iterate_time in range(burn_in + 1, updates + 1):
        mean += (
            selector
            @ np.linalg.matrix_power(companion, iterate_time)
            @ lifted_initial
        )
    mean /= averaged_count

    impulses: list[np.ndarray] = []
    for innovation_time in range(updates):
        impulse = np.zeros((dimension, dimension))
        first_affected = max(burn_in + 1, innovation_time + 1)
        for iterate_time in range(first_affected, updates + 1):
            impulse += (
                step_size
                * selector
                @ np.linalg.matrix_power(
                    companion, iterate_time - 1 - innovation_time
                )
                @ injector
            )
        impulses.append(impulse / averaged_count)

    covariance = np.zeros((dimension, dimension))
    center = updates - 1
    for left in range(updates):
        for right in range(updates):
            covariance += (
                prefix_overlap_factor(schedule[left], schedule[right], rho)
                * impulses[left]
                @ lags[center + left - right]
                @ impulses[right].T
            )
    covariance = (covariance + covariance.T) / 2.0
    bias_risk = float(mean @ weight @ mean)
    noise_risk = float(np.trace(weight @ covariance))
    return {
        "mean": mean,
        "covariance": covariance,
        "bias_risk": bias_risk,
        "noise_risk": noise_risk,
        "risk": bias_risk + noise_risk,
        "averaged_count": averaged_count,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(companion)))),
    }


def scheduled_pr_risk_affine_coefficients(
    *,
    initial_history: np.ndarray,
    drift: np.ndarray,
    step_size: float,
    delay: int,
    q_schedule: Sequence[int],
    burn_in: int,
    base_lag_covariances: np.ndarray,
    risk_matrix: np.ndarray | None = None,
) -> AffineRisk:
    """Return exact coefficients of the scheduled PR risk in ``rho``."""

    matrix = np.asarray(drift, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("drift must be square")
    dimension = matrix.shape[0]
    history = np.asarray(initial_history, dtype=float)
    if delay < 0 or history.shape != (delay + 1, dimension):
        raise ValueError("initial_history must have shape (delay+1, dimension)")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    schedule = tuple(int(q) for q in q_schedule)
    updates = len(schedule)
    if updates < 1 or any(q < 1 for q in schedule):
        raise ValueError("q_schedule must be nonempty and positive")
    if burn_in < 0 or burn_in >= updates:
        raise ValueError("require 0 <= burn_in < len(q_schedule)")
    weight = (
        np.eye(dimension)
        if risk_matrix is None
        else np.asarray(risk_matrix, dtype=float)
    )
    if weight.shape != (dimension, dimension) or not np.allclose(weight, weight.T):
        raise ValueError("risk_matrix must be symmetric with drift dimension")
    lags = np.asarray(base_lag_covariances, dtype=float)
    expected_shape = (2 * updates - 1, dimension, dimension)
    if lags.shape != expected_shape:
        raise ValueError(f"base_lag_covariances must have shape {expected_shape}")

    companion = delayed_vector_companion(matrix, step_size, delay)
    selector = np.zeros((dimension, dimension * (delay + 1)))
    selector[:, :dimension] = np.eye(dimension)
    injector = selector.T
    lifted_initial = history.reshape(-1)
    averaged_count = updates - burn_in
    power = np.eye(companion.shape[0])
    state_values = [selector @ power @ lifted_initial]
    responses = []
    for _ in range(updates):
        responses.append(selector @ power @ injector)
        power = power @ companion
        state_values.append(selector @ power @ lifted_initial)
    mean = np.mean(state_values[burn_in + 1 :], axis=0)

    response_prefix = np.zeros((updates + 1, dimension, dimension))
    for index, response in enumerate(responses):
        response_prefix[index + 1] = response_prefix[index] + response
    impulses = []
    for innovation_time in range(updates):
        lower_lag = max(burn_in - innovation_time, 0)
        upper_lag = updates - 1 - innovation_time
        impulses.append(
            step_size
            * (response_prefix[upper_lag + 1] - response_prefix[lower_lag])
            / averaged_count
        )
    impulse_array = np.asarray(impulses)
    weighted_impulses = np.einsum(
        "ab,nbc->nac", weight, impulse_array, optimize=True
    )
    schedule_array = np.asarray(schedule, dtype=float)

    intercept = float(mean @ weight @ mean)
    slope = 0.0
    center = updates - 1
    for lag in range(-(updates - 1), updates):
        if lag >= 0:
            left_indices = np.arange(lag, updates)
            right_indices = np.arange(0, updates - lag)
        else:
            left_indices = np.arange(0, updates + lag)
            right_indices = np.arange(-lag, updates)
        transformed = (
            weighted_impulses[left_indices] @ lags[center + lag]
        )
        pair_risks = np.einsum(
            "nij,nij->n",
            transformed,
            impulse_array[right_indices],
            optimize=True,
        )
        private_overlap = 1.0 / np.maximum(
            schedule_array[left_indices], schedule_array[right_indices]
        )
        intercept += float(private_overlap @ pair_risks)
        slope += float((1.0 - private_overlap) @ pair_risks)
    return AffineRisk(float(intercept), float(slope))


def _squared_interval(interval: tuple[float, float]) -> tuple[float, float]:
    lower, upper = interval
    maximum = max(lower * lower, upper * upper)
    minimum = 0.0 if lower <= 0.0 <= upper else min(lower * lower, upper * upper)
    return minimum, maximum


def mixing_corrected_icc_interval(
    probes: np.ndarray,
    *,
    alpha: float,
    beta_constant: float,
    beta_rate: float,
    stride: int,
    variance_lower: float = 0.0,
) -> CorrelationCertificate:
    """Certify the intraclass correlation of bounded scalar probes.

    ``probes[t, i]`` must lie in ``[-1, 1]`` and have a common stationary
    mean, marginal variance, and pairwise covariance.  The joint probe state
    at physical lag ``k`` must satisfy

        beta(k) <= beta_constant * beta_rate**k.

    The proof uses a total-variation coupling of the stride-spaced block and
    three bounded moment summaries.  If the declared mixing penalty exhausts
    the error budget, the returned interval is deliberately ``[0, 1]``.
    """

    values = np.asarray(probes, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("probes must have shape (trials>=2, agents>=2)")
    if not np.all(np.isfinite(values)) or np.max(np.abs(values)) > 1.0 + 1e-12:
        raise ValueError("probe values must be finite and lie in [-1, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if beta_constant < 0.0 or not 0.0 <= beta_rate < 1.0 or stride < 1:
        raise ValueError("invalid beta-mixing envelope or stride")
    if not 0.0 <= variance_lower <= 1.0:
        raise ValueError("variance_lower must lie in [0, 1]")

    trials, agents = values.shape
    mixing_penalty = (trials - 1) * beta_constant * beta_rate**stride
    per_summary_budget = alpha / 3.0
    concentration_budget = per_summary_budget - mixing_penalty
    if concentration_budget <= 0.0:
        return CorrelationCertificate(
            lower=0.0,
            upper=1.0,
            radius=math.inf,
            mixing_penalty=float(mixing_penalty),
            alpha=alpha,
            trials=trials,
            stride=stride,
            informative=False,
            mean_interval=(-1.0, 1.0),
            variance_interval=(variance_lower, 1.0),
            covariance_interval=(0.0, 1.0),
        )

    radius = math.sqrt(2.0 / trials * math.log(2.0 / concentration_budget))
    sums = np.sum(values, axis=1)
    sums_of_squares = np.sum(values**2, axis=1)
    time_mean = sums / agents
    time_second_moment = sums_of_squares / agents
    time_cross_moment = (
        sums**2 - sums_of_squares
    ) / float(agents * (agents - 1))

    mean_estimate = float(np.mean(time_mean))
    second_estimate = float(np.mean(time_second_moment))
    cross_estimate = float(np.mean(time_cross_moment))
    mean_interval = (
        max(-1.0, mean_estimate - radius),
        min(1.0, mean_estimate + radius),
    )
    second_interval = (
        max(0.0, second_estimate - radius),
        min(1.0, second_estimate + radius),
    )
    cross_interval = (
        max(-1.0, cross_estimate - radius),
        min(1.0, cross_estimate + radius),
    )
    mean_square_min, mean_square_max = _squared_interval(mean_interval)
    variance_interval = (
        max(variance_lower, 0.0, second_interval[0] - mean_square_max),
        min(1.0, max(0.0, second_interval[1] - mean_square_min)),
    )
    covariance_interval = (
        max(0.0, cross_interval[0] - mean_square_max),
        min(1.0, max(0.0, cross_interval[1] - mean_square_min)),
    )
    variance_low, variance_high = variance_interval
    covariance_low, covariance_high = covariance_interval
    covariance_high = min(covariance_high, variance_high)
    covariance_interval = (covariance_low, covariance_high)
    if variance_low > variance_high or covariance_low > covariance_high:
        return CorrelationCertificate(
            lower=0.0,
            upper=1.0,
            radius=radius,
            mixing_penalty=float(mixing_penalty),
            alpha=alpha,
            trials=trials,
            stride=stride,
            informative=False,
            mean_interval=mean_interval,
            variance_interval=(variance_lower, 1.0),
            covariance_interval=(0.0, 1.0),
        )
    rho_lower = (
        0.0
        if covariance_low <= 0.0 or variance_high <= 0.0
        else covariance_low / variance_high
    )
    rho_upper = (
        1.0
        if variance_low <= 0.0
        else min(1.0, covariance_high / variance_low)
    )
    rho_lower = min(1.0, max(0.0, rho_lower))
    rho_upper = min(1.0, max(rho_lower, rho_upper))
    return CorrelationCertificate(
        lower=rho_lower,
        upper=rho_upper,
        radius=radius,
        mixing_penalty=float(mixing_penalty),
        alpha=alpha,
        trials=trials,
        stride=stride,
        informative=rho_upper - rho_lower < 1.0 - 1e-15,
        mean_interval=mean_interval,
        variance_interval=variance_interval,
        covariance_interval=covariance_interval,
    )


def robust_minimax_choice(
    risks: Mapping[str, AffineRisk], *, rho_lower: float, rho_upper: float
) -> str:
    """Minimize the worst exact schedule risk over a correlation interval."""

    if not risks:
        raise ValueError("risks must be nonempty")
    if not 0.0 <= rho_lower <= rho_upper <= 1.0:
        raise ValueError("invalid correlation interval")
    return min(
        risks,
        key=lambda name: (
            max(
                risks[name].evaluate(rho_lower),
                risks[name].evaluate(rho_upper),
            ),
            name,
        ),
    )


def probe_cost_oracle_certificate(
    *,
    full_budget_risks: Mapping[str, AffineRisk],
    post_probe_risks: Mapping[str, AffineRisk],
    selected: str,
    rho_lower: float,
    rho_upper: float,
    failure_probability: float,
    risk_cap: float,
    baseline: str | None = None,
) -> OracleCertificate:
    """Bound total excess risk, including the already-paid probe cost.

    The full-budget and post-probe tables may contain different schedules or
    horizons.  On the correlation-coverage event, the bound is exact over the
    interval because a selected affine risk minus the lower envelope of affine
    oracle risks is the maximum of finitely many affine functions.
    ``risk_cap`` must upper-bound the relevant excess on certificate failure.
    """

    if not full_budget_risks or selected not in post_probe_risks:
        raise ValueError("risk tables must be nonempty and contain selected")
    if not 0.0 <= rho_lower <= rho_upper <= 1.0:
        raise ValueError("invalid correlation interval")
    if not 0.0 <= failure_probability < 1.0 or risk_cap < 0.0:
        raise ValueError("invalid failure_probability or risk_cap")
    if baseline is not None and baseline not in full_budget_risks:
        raise ValueError("baseline must be present in the full-budget table")

    selected_risk = post_probe_risks[selected]
    endpoints = (rho_lower, rho_upper)
    conditional_oracle_excess = max(
        selected_risk.evaluate(rho) - risk.evaluate(rho)
        for rho in endpoints
        for risk in full_budget_risks.values()
    )
    expected_oracle_excess = conditional_oracle_excess + failure_probability * max(
        risk_cap - conditional_oracle_excess, 0.0
    )

    conditional_baseline_difference: float | None = None
    expected_baseline_difference: float | None = None
    if baseline is not None:
        baseline_risk = full_budget_risks[baseline]
        conditional_baseline_difference = max(
            selected_risk.evaluate(rho) - baseline_risk.evaluate(rho)
            for rho in endpoints
        )
        expected_baseline_difference = (
            conditional_baseline_difference
            + failure_probability
            * max(risk_cap - conditional_baseline_difference, 0.0)
        )

    return OracleCertificate(
        selected=selected,
        conditional_full_oracle_excess=float(conditional_oracle_excess),
        expected_full_oracle_excess=float(expected_oracle_excess),
        conditional_baseline_difference=(
            None
            if conditional_baseline_difference is None
            else float(conditional_baseline_difference)
        ),
        expected_baseline_difference=(
            None
            if expected_baseline_difference is None
            else float(expected_baseline_difference)
        ),
        failure_probability=failure_probability,
        risk_cap=risk_cap,
    )


def generic_post_probe_oracle_bound(
    risks: Mapping[str, AffineRisk], *, interval_width: float
) -> float:
    """Return ``L * width`` for robust minimax selection on an affine table."""

    if not risks or not 0.0 <= interval_width <= 1.0:
        raise ValueError("invalid risks or interval width")
    slope_bound = max(abs(risk.slope) for risk in risks.values())
    return float(slope_bound * interval_width)

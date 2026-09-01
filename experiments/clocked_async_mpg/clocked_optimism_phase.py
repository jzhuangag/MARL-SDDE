"""Exact phase factors for asynchronous coordinate game updates.

The two-dimensional models isolate the paper candidate's endogenous tension:
plain coordinate gradient is cheaper and faster in a potential phase, whereas
coordinate extragradient is necessary to dissipate a rotational phase.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PhaseFactors:
    plain: float
    optimistic: float

    @property
    def optimistic_gain(self) -> float:
        return self.plain - self.optimistic


@dataclass(frozen=True)
class ClockedOptimismStep:
    use_optimism: bool
    energy_after: float
    resource_debt_after: float
    plain_factor: float
    optimistic_factor: float


@dataclass(frozen=True)
class LogDriftDecision:
    use_fresh_anchor: bool
    selected_log_multiplier: float
    resource_debt_after: float


@dataclass(frozen=True)
class FiniteHorizonBudgetReserve:
    """Nominal queue budget that implies an exact finite call budget."""

    maximum_calls: int
    horizon: int
    maximum_debt: float
    nominal_average_budget: float


def finite_horizon_budget_reserve(
    *,
    maximum_calls: int,
    horizon: int,
    lyapunov_tradeoff: float,
    maximum_log_gain: float,
) -> FiniteHorizonBudgetReserve:
    """Reserve the worst-case terminal debt before running the controller.

    If every certified log gain obeys
    ``log(q_plain) - log(q_fresh) <= maximum_log_gain``, a controller starting
    at zero debt has debt strictly below ``V * maximum_log_gain + 1``.  Running
    it with the returned nominal average budget therefore gives

    ``sum(use_fresh) <= nominal_budget * horizon + terminal_debt``
    ``                    <= maximum_calls``.

    The construction removes the proof mismatch caused by a post-hoc hard
    guard changing the controller's feasible action set near the horizon.
    """

    if isinstance(maximum_calls, bool) or not isinstance(maximum_calls, int):
        raise ValueError("maximum_calls must be an integer")
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError("horizon must be an integer")
    if maximum_calls < 0 or horizon <= 0 or maximum_calls > horizon:
        raise ValueError("maximum_calls must lie in [0, horizon]")
    if (
        not math.isfinite(lyapunov_tradeoff)
        or not math.isfinite(maximum_log_gain)
        or lyapunov_tradeoff <= 0.0
        or maximum_log_gain < 0.0
    ):
        raise ValueError("tradeoff must be positive and maximum gain nonnegative")
    maximum_debt = lyapunov_tradeoff * maximum_log_gain + 1.0
    nominal = (maximum_calls - maximum_debt) / horizon
    if nominal < 0.0:
        raise ValueError("finite call budget is smaller than the debt reserve")
    return FiniteHorizonBudgetReserve(
        maximum_calls=maximum_calls,
        horizon=horizon,
        maximum_debt=maximum_debt,
        nominal_average_budget=nominal,
    )


def rotational_coordinate_factors(normalized_step: float) -> PhaseFactors:
    """Mean-square factors for a uniform asynchronous bilinear coordinate."""

    if not math.isfinite(normalized_step) or not 0.0 < normalized_step < 1.0:
        raise ValueError("rotational normalized step must lie in (0, 1)")
    square = normalized_step * normalized_step
    return PhaseFactors(
        plain=1.0 + 0.5 * square,
        optimistic=1.0 - 0.5 * square + 0.5 * square * square,
    )


def potential_coordinate_factors(normalized_step: float) -> PhaseFactors:
    """Mean-square factors for a uniform asynchronous quadratic coordinate."""

    if not math.isfinite(normalized_step) or not 0.0 < normalized_step < 1.0:
        raise ValueError("potential normalized step must lie in (0, 1)")
    plain_updated = 1.0 - normalized_step
    optimistic_updated = 1.0 - normalized_step + normalized_step * normalized_step
    return PhaseFactors(
        plain=0.5 * (1.0 + plain_updated * plain_updated),
        optimistic=0.5 * (1.0 + optimistic_updated * optimistic_updated),
    )


def rotational_optimism_threshold(normalized_step: float) -> float:
    """Minimum randomized optimism probability for strict MS contraction."""

    factors = rotational_coordinate_factors(normalized_step)
    return (factors.plain - 1.0) / factors.optimistic_gain


def randomized_factor(factors: PhaseFactors, optimism_probability: float) -> float:
    if (
        not math.isfinite(optimism_probability)
        or not 0.0 <= optimism_probability <= 1.0
    ):
        raise ValueError("optimism_probability must lie in [0, 1]")
    return (
        (1.0 - optimism_probability) * factors.plain
        + optimism_probability * factors.optimistic
    )


def heterogeneous_clock_metric(first_agent_probability: float) -> tuple[float, float]:
    """Diagonal Lyapunov metric that balances two asynchronous arrival clocks."""

    if (
        not math.isfinite(first_agent_probability)
        or not 0.0 < first_agent_probability < 1.0
    ):
        raise ValueError("arrival probability must lie in (0, 1)")
    return ((1.0 - first_agent_probability) / first_agent_probability, 1.0)


def heterogeneous_rotational_drift_coefficient(
    normalized_step: float,
    *,
    first_agent_probability: float,
    optimism_probability: float,
) -> float:
    """Exact coefficient in E[V+ - V] for the clock-balanced metric.

    The drift equals this coefficient times the Euclidean state energy.
    Its sign boundary is independent of the nonzero arrival probability.
    """

    rotational_coordinate_factors(normalized_step)
    heterogeneous_clock_metric(first_agent_probability)
    if (
        not math.isfinite(optimism_probability)
        or not 0.0 <= optimism_probability <= 1.0
    ):
        raise ValueError("optimism_probability must lie in [0, 1]")
    square = normalized_step * normalized_step
    return (
        (1.0 - first_agent_probability)
        * square
        * (1.0 - optimism_probability * (2.0 - square))
    )


def heterogeneous_potential_drift_coefficient(
    normalized_step: float,
    *,
    first_agent_probability: float,
    use_optimism: bool,
) -> float:
    """Exact quadratic-potential drift coefficient in the balanced metric."""

    potential_coordinate_factors(normalized_step)
    heterogeneous_clock_metric(first_agent_probability)
    updated = (
        1.0 - normalized_step + normalized_step * normalized_step
        if use_optimism
        else 1.0 - normalized_step
    )
    return (1.0 - first_agent_probability) * (updated * updated - 1.0)


def lifted_rotational_transition(
    normalized_step: float,
    *,
    delay: int,
    agent: int,
    fresh_optimistic_anchor: bool,
) -> np.ndarray:
    """Lifted transition for a stale plain or arrival-fresh EG coordinate."""

    rotational_coordinate_factors(normalized_step)
    if delay < 0 or agent not in (0, 1):
        raise ValueError("delay must be nonnegative and agent must be 0 or 1")
    dimension = 2 * (delay + 1)
    transition = np.zeros((dimension, dimension), dtype=float)
    transition[:2, :2] = np.eye(2)
    rotation = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    selector = np.zeros((2, 2), dtype=float)
    selector[agent, agent] = 1.0
    if fresh_optimistic_anchor:
        transition[:2, :2] -= (
            normalized_step
            * selector
            @ rotation
            @ (np.eye(2) - normalized_step * rotation)
        )
    else:
        delayed_slice = slice(2 * delay, 2 * delay + 2)
        transition[:2, delayed_slice] -= normalized_step * selector @ rotation
    for lag in range(1, delay + 1):
        transition[2 * lag : 2 * lag + 2, 2 * (lag - 1) : 2 * lag] = np.eye(2)
    return transition


def lifted_mean_square_spectral_radius(
    normalized_step: float,
    *,
    delay: int,
    first_agent_probability: float,
    fresh_optimism_probability: float,
) -> float:
    """Exact iid-switch second-moment spectral radius for the lifted system."""

    heterogeneous_clock_metric(first_agent_probability)
    if (
        not math.isfinite(fresh_optimism_probability)
        or not 0.0 <= fresh_optimism_probability <= 1.0
    ):
        raise ValueError("fresh_optimism_probability must lie in [0, 1]")
    dimension = 2 * (delay + 1)
    operator = np.zeros((dimension * dimension, dimension * dimension))
    for agent_probability, agent in (
        (first_agent_probability, 0),
        (1.0 - first_agent_probability, 1),
    ):
        for anchor_probability, fresh in (
            (1.0 - fresh_optimism_probability, False),
            (fresh_optimism_probability, True),
        ):
            transition = lifted_rotational_transition(
                normalized_step,
                delay=delay,
                agent=agent,
                fresh_optimistic_anchor=fresh,
            )
            operator += (
                agent_probability
                * anchor_probability
                * np.kron(transition, transition)
            )
    return float(np.max(np.abs(np.linalg.eigvals(operator))))


def stale_optimistic_lifted_spectral_radius(
    normalized_step: float,
    *,
    delay: int,
    first_agent_probability: float,
) -> float:
    """Diagnostic radius when the nominal EG oracle is itself delayed."""

    rotational_coordinate_factors(normalized_step)
    heterogeneous_clock_metric(first_agent_probability)
    if delay < 0:
        raise ValueError("delay must be nonnegative")
    dimension = 2 * (delay + 1)
    rotation = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    operator = np.zeros((dimension * dimension, dimension * dimension))
    for probability, agent in (
        (first_agent_probability, 0),
        (1.0 - first_agent_probability, 1),
    ):
        transition = np.zeros((dimension, dimension))
        transition[:2, :2] = np.eye(2)
        selector = np.zeros((2, 2))
        selector[agent, agent] = 1.0
        delayed_slice = slice(2 * delay, 2 * delay + 2)
        transition[:2, delayed_slice] -= (
            normalized_step
            * selector
            @ rotation
            @ (np.eye(2) - normalized_step * rotation)
        )
        for lag in range(1, delay + 1):
            transition[2 * lag : 2 * lag + 2, 2 * (lag - 1) : 2 * lag] = np.eye(2)
        operator += probability * np.kron(transition, transition)
    return float(np.max(np.abs(np.linalg.eigvals(operator))))


def expected_quadratic_multiplier(
    metric: np.ndarray,
    transitions: tuple[np.ndarray, ...],
    probabilities: tuple[float, ...],
) -> float:
    """Smallest q satisfying sum p M^T P M <= q P."""

    metric = np.asarray(metric, dtype=float)
    if (
        metric.ndim != 2
        or metric.shape[0] != metric.shape[1]
        or not np.allclose(metric, metric.T, atol=1e-12)
    ):
        raise ValueError("metric must be square and symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError("metric must be positive definite")
    if len(transitions) == 0 or len(transitions) != len(probabilities):
        raise ValueError("transitions and probabilities must be nonempty and match")
    if any(matrix.shape != metric.shape for matrix in transitions):
        raise ValueError("every transition must match the metric")
    if any(probability < 0.0 for probability in probabilities) or not math.isclose(
        sum(probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("probabilities must be nonnegative and sum to one")
    expected = sum(
        probability * matrix.T @ metric @ matrix
        for probability, matrix in zip(probabilities, transitions)
    )
    inverse_root = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )
    normalized = inverse_root @ expected @ inverse_root
    return float(np.max(np.linalg.eigvalsh(0.5 * (normalized + normalized.T))))


def fresh_anchor_lyapunov_metric(
    normalized_step: float,
    *,
    delay: int,
    first_agent_probability: float,
    tolerance: float = 1e-12,
    maximum_iterations: int = 200_000,
) -> np.ndarray:
    """Solve P = I + E[M_fresh^T P M_fresh] by fixed-point iteration."""

    heterogeneous_clock_metric(first_agent_probability)
    transitions = tuple(
        lifted_rotational_transition(
            normalized_step,
            delay=delay,
            agent=agent,
            fresh_optimistic_anchor=True,
        )
        for agent in (0, 1)
    )
    probabilities = (first_agent_probability, 1.0 - first_agent_probability)
    dimension = transitions[0].shape[0]
    metric = np.eye(dimension)
    identity = np.eye(dimension)
    for _ in range(maximum_iterations):
        updated = identity + sum(
            probability * matrix.T @ metric @ matrix
            for probability, matrix in zip(probabilities, transitions)
        )
        relative = np.linalg.norm(updated - metric) / max(
            1.0, np.linalg.norm(updated)
        )
        metric = updated
        if relative <= tolerance:
            return 0.5 * (metric + metric.T)
    raise RuntimeError("fresh-anchor Lyapunov fixed point did not converge")


def choose_log_drift_anchor(
    *,
    plain_multiplier: float,
    fresh_multiplier: float,
    resource_debt: float,
    average_anchor_budget: float,
    lyapunov_tradeoff: float,
    hard_feasible: bool = True,
) -> LogDriftDecision:
    """Drift-plus-penalty choice using certified log energy multipliers."""

    values = (
        plain_multiplier,
        fresh_multiplier,
        resource_debt,
        average_anchor_budget,
        lyapunov_tradeoff,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("all log-drift inputs must be finite")
    if plain_multiplier <= 0.0 or fresh_multiplier <= 0.0:
        raise ValueError("multipliers must be positive")
    if resource_debt < 0.0 or not 0.0 <= average_anchor_budget <= 1.0:
        raise ValueError("invalid debt or anchor budget")
    if lyapunov_tradeoff <= 0.0:
        raise ValueError("lyapunov_tradeoff must be positive")
    plain_log = math.log(plain_multiplier)
    fresh_log = math.log(fresh_multiplier)
    use_fresh = bool(
        hard_feasible
        and lyapunov_tradeoff * (plain_log - fresh_log) > resource_debt
    )
    selected = fresh_log if use_fresh else plain_log
    debt_after = max(
        0.0,
        resource_debt + float(use_fresh) - average_anchor_budget,
    )
    return LogDriftDecision(use_fresh, selected, debt_after)


def choose_clocked_optimism(
    *,
    energy: float,
    factors: PhaseFactors,
    resource_debt: float,
    average_optimism_budget: float,
    lyapunov_tradeoff: float,
    hard_feasible: bool = True,
) -> ClockedOptimismStep:
    """One drift-plus-penalty decision with fully charged optimism cost."""

    values = (
        energy,
        resource_debt,
        average_optimism_budget,
        lyapunov_tradeoff,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("all controller inputs must be finite")
    if energy < 0.0 or resource_debt < 0.0 or lyapunov_tradeoff <= 0.0:
        raise ValueError("energy/debt must be nonnegative and tradeoff positive")
    if not 0.0 <= average_optimism_budget <= 1.0:
        raise ValueError("average_optimism_budget must lie in [0, 1]")
    drift_gain = energy * factors.optimistic_gain
    use_optimism = bool(
        hard_feasible and lyapunov_tradeoff * drift_gain > resource_debt
    )
    factor = factors.optimistic if use_optimism else factors.plain
    debt_after = max(
        0.0,
        resource_debt + float(use_optimism) - average_optimism_budget,
    )
    return ClockedOptimismStep(
        use_optimism=use_optimism,
        energy_after=energy * factor,
        resource_debt_after=debt_after,
        plain_factor=factors.plain,
        optimistic_factor=factors.optimistic,
    )

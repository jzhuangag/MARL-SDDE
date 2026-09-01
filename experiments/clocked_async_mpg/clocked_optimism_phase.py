"""Exact phase factors for asynchronous coordinate game updates.

The two-dimensional models isolate the paper candidate's endogenous tension:
plain coordinate gradient is cheaper and faster in a potential phase, whereas
coordinate extragradient is necessary to dissipate a rotational phase.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


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

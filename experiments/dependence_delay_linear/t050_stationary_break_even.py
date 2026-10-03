"""Stationary participation limits and probe amortization thresholds.

The results in this module are deliberately split into two levels.

* Independent Gaussian rounds admit an exact information-per-cost theorem for
  every deterministic or predictable participation schedule.
* Stable delayed linear stochastic approximation admits a leading-order
  Polyak--Ruppert coefficient.  This second result characterizes fixed-q
  participation asymptotically; it does not claim finite-horizon dominance of
  fixed schedules for arbitrary Markov covariance sequences.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math

import numpy as np


def aggregate_variance(
    q: float, *, common_variance: float, private_variance: float
) -> float:
    """Variance of one q-agent average in the common/private model."""

    if q <= 0.0:
        raise ValueError("q must be positive")
    if common_variance < 0.0 or private_variance <= 0.0:
        raise ValueError("require common variance >= 0 and private variance > 0")
    return float(common_variance + private_variance / q)


def round_information(
    q: float, *, common_variance: float, private_variance: float
) -> float:
    """Fisher information for a shared scalar mean in one independent round."""

    return 1.0 / aggregate_variance(
        q, common_variance=common_variance, private_variance=private_variance
    )


def information_efficiency(
    q: float,
    *,
    overhead: float,
    common_variance: float,
    private_variance: float,
) -> float:
    """Independent-round Fisher information per message-budget unit."""

    if overhead < 0.0:
        raise ValueError("overhead must be nonnegative")
    return round_information(
        q, common_variance=common_variance, private_variance=private_variance
    ) / (overhead + q)


def continuous_information_optimum(
    *,
    minimum_q: float,
    maximum_q: float,
    overhead: float,
    common_variance: float,
    private_variance: float,
) -> float:
    """Clipped continuous maximizer of information per message cost."""

    if minimum_q <= 0.0 or maximum_q < minimum_q:
        raise ValueError("invalid participation interval")
    if overhead < 0.0 or common_variance < 0.0 or private_variance <= 0.0:
        raise ValueError("invalid cost or variance")
    if common_variance == 0.0:
        return float(maximum_q)
    if overhead == 0.0:
        return float(minimum_q)
    optimum = math.sqrt(overhead * private_variance / common_variance)
    return float(np.clip(optimum, minimum_q, maximum_q))


def sequence_information(
    q_sequence: Sequence[int],
    *,
    common_variance: float,
    private_variance: float,
) -> float:
    """Exact information of independent heterogeneous Gaussian rounds."""

    sequence = tuple(int(q) for q in q_sequence)
    if not sequence or any(q < 1 for q in sequence):
        raise ValueError("q_sequence must be nonempty and positive")
    return float(
        sum(
            round_information(
                q,
                common_variance=common_variance,
                private_variance=private_variance,
            )
            for q in sequence
        )
    )


def independent_schedule_bound(
    *,
    budget: int,
    candidates: Iterable[int],
    overhead: int,
    common_variance: float,
    private_variance: float,
) -> dict[str, float | int]:
    """Exact fixed design and universal schedule-value upper bound.

    The universal lower bound on risk is ``1 / (budget * best_efficiency)``.
    Repeating an efficiency-maximizing action gives an upper bound on the
    relative improvement that any feasible time-varying schedule can obtain
    over the best fixed-q policy solely through finite-budget packing.
    """

    actions = sorted({int(q) for q in candidates})
    if budget <= 0 or overhead < 0 or not actions or actions[0] < 1:
        raise ValueError("invalid budget, overhead, or candidates")
    feasible = [q for q in actions if overhead + q <= budget]
    if not feasible:
        raise ValueError("no candidate fits the budget")
    rows = []
    for q in feasible:
        cost = overhead + q
        info = round_information(
            q,
            common_variance=common_variance,
            private_variance=private_variance,
        )
        rows.append(
            {
                "q": q,
                "cost": cost,
                "round_information": info,
                "efficiency": info / cost,
                "rounds": budget // cost,
                "fixed_information": (budget // cost) * info,
            }
        )
    efficiency_row = max(rows, key=lambda row: (row["efficiency"], -row["q"]))
    fixed_row = max(rows, key=lambda row: (row["fixed_information"], -row["q"]))
    efficiency = float(efficiency_row["efficiency"])
    used_fraction = (
        int(efficiency_row["rounds"]) * int(efficiency_row["cost"]) / budget
    )
    return {
        "efficiency_q": int(efficiency_row["q"]),
        "best_fixed_q": int(fixed_row["q"]),
        "best_efficiency": efficiency,
        "universal_risk_lower_bound": 1.0 / (budget * efficiency),
        "best_fixed_risk": 1.0 / float(fixed_row["fixed_information"]),
        "schedule_relative_improvement_upper_bound": 1.0 - used_fraction,
        "efficiency_action_used_fraction": used_fraction,
    }


def trajectory_switch_factor(q: float, rho: float) -> float:
    """Common/private covariance multiplier for fixed prefix participation."""

    if q <= 0.0 or not 0.0 <= rho <= 1.0:
        raise ValueError("invalid q or rho")
    return float(rho + (1.0 - rho) / q)


def asymptotic_participation_coefficient(
    q: float, *, overhead: float, rho: float
) -> float:
    """Task-free multiplier of the leading PR risk under a message budget."""

    if overhead < 0.0:
        raise ValueError("overhead must be nonnegative")
    return float((overhead + q) * trajectory_switch_factor(q, rho))


def continuous_pr_optimum(
    *, minimum_q: float, maximum_q: float, overhead: float, rho: float
) -> float:
    """Clipped continuous optimum of the stationary PR coefficient."""

    if minimum_q <= 0.0 or maximum_q < minimum_q or overhead < 0.0:
        raise ValueError("invalid participation interval or overhead")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    if rho == 0.0:
        return float(maximum_q)
    if overhead == 0.0 or rho == 1.0:
        return float(minimum_q)
    optimum = math.sqrt(overhead * (1.0 - rho) / rho)
    return float(np.clip(optimum, minimum_q, maximum_q))


def optimal_catalogue_q(
    candidates: Iterable[int], *, overhead: float, rho: float
) -> dict[str, float | int]:
    """Best fixed catalogue action for the leading stationary PR risk."""

    actions = sorted({int(q) for q in candidates})
    if not actions or actions[0] < 1:
        raise ValueError("candidates must be positive")
    rows = [
        (
            asymptotic_participation_coefficient(q, overhead=overhead, rho=rho),
            q,
        )
        for q in actions
    ]
    coefficient, q = min(rows, key=lambda row: (row[0], row[1]))
    return {"q": int(q), "coefficient": float(coefficient)}


def probe_break_even_budget(
    *,
    probe_message_cost: float,
    baseline_coefficient: float,
    oracle_coefficient: float,
    wrong_action_coefficient: float,
    error_probability: float,
) -> float:
    """Leading-order budget at which a paid classify-then-commit rule wins.

    The returned threshold solves

    ``K_adapt / (B-C_probe) < K_baseline / B``, where
    ``K_adapt=(1-alpha)K_oracle+alpha K_wrong``.

    An infinite threshold means that the certified expected post-probe action
    is not better than the baseline even before charging the probe.
    """

    if probe_message_cost < 0.0:
        raise ValueError("probe cost must be nonnegative")
    coefficients = (
        baseline_coefficient,
        oracle_coefficient,
        wrong_action_coefficient,
    )
    if any(value <= 0.0 for value in coefficients):
        raise ValueError("risk coefficients must be positive")
    if not 0.0 <= error_probability <= 1.0:
        raise ValueError("error_probability must lie in [0, 1]")
    adaptive = (
        (1.0 - error_probability) * oracle_coefficient
        + error_probability * wrong_action_coefficient
    )
    gap = baseline_coefficient - adaptive
    if gap <= 0.0:
        return math.inf
    return float(probe_message_cost * baseline_coefficient / gap)


def controller_beats_baseline_leading_order(
    *,
    total_budget: float,
    probe_message_cost: float,
    baseline_coefficient: float,
    oracle_coefficient: float,
    wrong_action_coefficient: float,
    error_probability: float,
) -> bool:
    """Evaluate the same paid-probe comparison without rearranging it."""

    if total_budget <= probe_message_cost:
        return False
    adaptive = (
        (1.0 - error_probability) * oracle_coefficient
        + error_probability * wrong_action_coefficient
    )
    return adaptive / (total_budget - probe_message_cost) < (
        baseline_coefficient / total_budget
    )


def exact_edge_long_run_covariance(
    *,
    transition: np.ndarray,
    stationary: np.ndarray,
    edge_gradient_sum: np.ndarray,
    conditional_gradient: np.ndarray,
    second_moment: np.ndarray,
) -> np.ndarray:
    """Exact long-run covariance of the centered finite-state edge process."""

    probability = np.asarray(transition, dtype=float)
    invariant = np.asarray(stationary, dtype=float)
    edge = np.asarray(edge_gradient_sum, dtype=float)
    conditional = np.asarray(conditional_gradient, dtype=float)
    second = np.asarray(second_moment, dtype=float)
    states = probability.shape[0]
    if probability.shape != (states, states):
        raise ValueError("transition must be square")
    if invariant.shape != (states,) or edge.shape[:2] != (states, states):
        raise ValueError("stationary or edge shape mismatch")
    dimension = edge.shape[2]
    if conditional.shape != (states, dimension) or second.shape != (
        dimension,
        dimension,
    ):
        raise ValueError("gradient moment shape mismatch")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-11):
        raise ValueError("transition must be stochastic")
    if not np.allclose(invariant @ probability, invariant, atol=1e-10):
        raise ValueError("stationary distribution must be invariant")
    if not np.allclose(invariant @ conditional, 0.0, atol=1e-9):
        raise ValueError("conditional gradient must be centered")
    fundamental = np.linalg.inv(
        np.eye(states) - probability + np.ones((states, 1)) @ invariant[None, :]
    )
    future_sum = fundamental @ conditional
    positive_sum = np.einsum(
        "s,sud,ue->de",
        invariant,
        edge,
        future_sum,
        optimize=True,
    ).T
    covariance = second + positive_sum + positive_sum.T
    covariance = (covariance + covariance.T) / 2.0
    return covariance


def pr_task_constant(
    *, drift: np.ndarray, long_run_covariance: np.ndarray, risk_matrix: np.ndarray | None = None
) -> float:
    """Trace constant in the zero-frequency PR covariance ``A^-1 Gamma A^-T``."""

    matrix = np.asarray(drift, dtype=float)
    covariance = np.asarray(long_run_covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("drift must be square")
    dimension = matrix.shape[0]
    if covariance.shape != (dimension, dimension):
        raise ValueError("long_run_covariance shape mismatch")
    weight = np.eye(dimension) if risk_matrix is None else np.asarray(risk_matrix)
    if weight.shape != (dimension, dimension):
        raise ValueError("risk_matrix shape mismatch")
    inverse = np.linalg.inv(matrix)
    value = float(np.trace(weight @ inverse @ covariance @ inverse.T))
    if value < -1e-9:
        raise ValueError("computed PR task constant is negative")
    return max(value, 0.0)


def contraction_burn_in_horizon(
    *, spectral_radius: float, target: float, averaging_fraction: float = 0.5
) -> int:
    """Smallest N whose deterministic contraction at PR burn-in is <= target."""

    if not 0.0 <= spectral_radius < 1.0:
        raise ValueError("spectral_radius must lie in [0, 1)")
    if not 0.0 < target < 1.0 or not 0.0 < averaging_fraction < 1.0:
        raise ValueError("target and averaging_fraction must lie in (0, 1)")
    if spectral_radius == 0.0:
        return 1
    required_burn = math.ceil(math.log(target) / math.log(spectral_radius))
    return max(1, math.ceil(required_burn / averaging_fraction))

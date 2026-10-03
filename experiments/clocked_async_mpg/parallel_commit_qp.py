"""Low-complexity Lyapunov commit rule for asynchronous MARL proposals.

The controller jointly selects a continuous commit scale for every ready
agent.  Its drift-plus-progress subproblem has a diagonal-plus-rank-one
quadratic term and box constraints.  The KKT system reduces to one monotone
scalar root, so the rule does not enumerate subsets or invoke a generic QP.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ParallelCommitDecision:
    scales: np.ndarray
    active_count: int
    certified_gain: float
    queue_weighted_objective: float
    interaction_load: float
    scalar_root: float
    iterations: int


def certified_joint_gain(
    *,
    scales: np.ndarray,
    gain_lower_bounds: np.ndarray,
    curvature_diagonal: np.ndarray,
    interaction_weights: np.ndarray,
    interaction_strength: float,
) -> float:
    """Evaluate the theorem-facing joint-improvement lower bound."""

    gains = _vector("gain_lower_bounds", gain_lower_bounds)
    scales = _vector("scales", scales, gains.size)
    diagonal = _vector("curvature_diagonal", curvature_diagonal, gains.size)
    weights = _vector("interaction_weights", interaction_weights, gains.size)
    if np.any(scales < 0.0) or np.any(scales > 1.0):
        raise ValueError("scales must lie in [0, 1]")
    if np.any(diagonal < 0.0) or np.any(weights < 0.0):
        raise ValueError("curvature and interaction weights must be nonnegative")
    if not math.isfinite(interaction_strength) or interaction_strength < 0.0:
        raise ValueError("interaction strength must be finite and nonnegative")
    return float(
        np.dot(gains, scales)
        - 0.5 * np.dot(diagonal, scales * scales)
        - 0.5 * interaction_strength * float(np.dot(weights, scales)) ** 2
    )


def stale_directional_lower_bounds(
    *,
    birth_directional_gains: np.ndarray,
    proposal_directions: np.ndarray,
    interaction_absolute: np.ndarray,
    policy_displacement: np.ndarray,
) -> np.ndarray:
    """Bound arrival-time directional gains from an observable policy path.

    If the absolute cross-block Hessian is bounded by ``interaction_absolute``,
    the fundamental theorem of calculus gives

    ``d_i grad_i(current) >= d_i grad_i(birth)
       - |d_i| sum_j L_ij |current_j-birth_j|``.
    """

    birth = _vector("birth_directional_gains", birth_directional_gains)
    directions = _vector("proposal_directions", proposal_directions, birth.size)
    displacement = _vector("policy_displacement", policy_displacement, birth.size)
    interaction = np.asarray(interaction_absolute, dtype=float)
    if interaction.shape != (birth.size, birth.size):
        raise ValueError("interaction_absolute has incompatible shape")
    if np.any(~np.isfinite(interaction)) or np.any(interaction < 0.0):
        raise ValueError("interaction_absolute must be finite and nonnegative")
    penalty = np.abs(directions) * (interaction @ np.abs(displacement))
    return birth - penalty


def _vector(name: str, value: np.ndarray, size: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if size is not None and result.size != size:
        raise ValueError(f"{name} has incompatible dimension")
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def solve_rank_one_box_qp(
    *,
    linear: np.ndarray,
    curvature_diagonal: np.ndarray,
    interaction_weights: np.ndarray,
    interaction_strength: float,
    maximum_scales: np.ndarray | None = None,
    tolerance: float = 1e-12,
    maximum_iterations: int = 160,
) -> ParallelCommitDecision:
    """Maximize a concave diagonal-plus-rank-one box quadratic.

    The objective is

    ``linear @ x - .5 * diagonal @ x**2
       - .5 * interaction_strength * (weights @ x)**2``

    over ``0 <= x <= maximum_scales``.  Nonnegative interaction weights and
    strictly positive diagonal curvature make the scalar KKT map monotone.
    """

    linear = _vector("linear", linear)
    diagonal = _vector("curvature_diagonal", curvature_diagonal, linear.size)
    weights = _vector("interaction_weights", interaction_weights, linear.size)
    if np.any(diagonal <= 0.0):
        raise ValueError("curvature diagonal must be strictly positive")
    if np.any(weights < 0.0):
        raise ValueError("interaction weights must be nonnegative")
    if not math.isfinite(interaction_strength) or interaction_strength < 0.0:
        raise ValueError("interaction strength must be finite and nonnegative")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    if maximum_scales is None:
        caps = np.ones_like(linear)
    else:
        caps = _vector("maximum_scales", maximum_scales, linear.size)
        if np.any(caps < 0.0) or np.any(caps > 1.0):
            raise ValueError("maximum scales must lie in [0, 1]")

    def scales_at(root: float) -> np.ndarray:
        return np.clip((linear - weights * root) / diagonal, 0.0, caps)

    if interaction_strength == 0.0 or not np.any(weights > 0.0):
        root = 0.0
        scales = scales_at(root)
        iterations = 0
    else:
        lower = 0.0
        upper = interaction_strength * float(np.dot(weights, caps))
        if upper == 0.0 or not np.any(scales_at(0.0) > 0.0):
            root = 0.0
            scales = scales_at(root)
            iterations = 0
        else:
            iterations = 0
            for iterations in range(1, maximum_iterations + 1):
                midpoint = 0.5 * (lower + upper)
                scales = scales_at(midpoint)
                residual = midpoint - interaction_strength * float(
                    np.dot(weights, scales)
                )
                if residual < 0.0:
                    lower = midpoint
                else:
                    upper = midpoint
                if upper - lower <= tolerance * max(1.0, upper):
                    break
            root = 0.5 * (lower + upper)
            scales = scales_at(root)

    interaction_load = float(np.dot(weights, scales))
    curvature_cost = 0.5 * float(np.dot(diagonal, scales * scales))
    interaction_cost = 0.5 * interaction_strength * interaction_load**2
    objective = float(np.dot(linear, scales) - curvature_cost - interaction_cost)
    return ParallelCommitDecision(
        scales=scales,
        active_count=int(np.count_nonzero(scales > tolerance)),
        certified_gain=float(
            np.dot(linear, scales) - curvature_cost - interaction_cost
        ),
        queue_weighted_objective=objective,
        interaction_load=interaction_load,
        scalar_root=float(root),
        iterations=iterations,
    )


def choose_lyapunov_parallel_commit(
    *,
    gain_lower_bounds: np.ndarray,
    service_debts: np.ndarray,
    risk_debt: float,
    risk_costs: np.ndarray,
    curvature_diagonal: np.ndarray,
    interaction_weights: np.ndarray,
    interaction_strength: float,
    tradeoff: float,
    maximum_scales: np.ndarray | None = None,
) -> ParallelCommitDecision:
    """Solve the event-time Lyapunov drift-plus-progress action.

    For service queues ``Q_i`` and a certificate-risk queue ``Z``, the
    controller maximizes

    ``(V*a + Q - Z*c)^T x - V/2*x^T H*x``.

    The nonzero coordinates are the online participant set and the scales
    multiply the base actor steps.  Thus participation and step size are one
    joint action rather than separately tuned heuristics.
    """

    gains = _vector("gain_lower_bounds", gain_lower_bounds)
    debts = _vector("service_debts", service_debts, gains.size)
    costs = _vector("risk_costs", risk_costs, gains.size)
    diagonal = _vector("curvature_diagonal", curvature_diagonal, gains.size)
    weights = _vector("interaction_weights", interaction_weights, gains.size)
    if np.any(debts < 0.0) or np.any(costs < 0.0):
        raise ValueError("queue debts and risk costs must be nonnegative")
    if not math.isfinite(risk_debt) or risk_debt < 0.0:
        raise ValueError("risk debt must be finite and nonnegative")
    if not math.isfinite(tradeoff) or tradeoff <= 0.0:
        raise ValueError("tradeoff must be finite and positive")
    linear = tradeoff * gains + debts - risk_debt * costs
    decision = solve_rank_one_box_qp(
        linear=linear,
        curvature_diagonal=tradeoff * diagonal,
        interaction_weights=weights,
        interaction_strength=tradeoff * interaction_strength,
        maximum_scales=maximum_scales,
    )
    gain = certified_joint_gain(
        scales=decision.scales,
        gain_lower_bounds=gains,
        curvature_diagonal=diagonal,
        interaction_weights=weights,
        interaction_strength=interaction_strength,
    )
    return ParallelCommitDecision(
        scales=decision.scales,
        active_count=decision.active_count,
        certified_gain=gain,
        queue_weighted_objective=decision.queue_weighted_objective,
        interaction_load=decision.interaction_load,
        scalar_root=decision.scalar_root,
        iterations=decision.iterations,
    )


def update_commit_queues(
    *,
    service_debts: np.ndarray,
    arrivals: np.ndarray,
    scales: np.ndarray,
    risk_debt: float,
    incurred_risk: float,
    risk_budget: float,
) -> tuple[np.ndarray, float]:
    """Apply projected service- and certificate-debt recursions."""

    debts = _vector("service_debts", service_debts)
    arrivals = _vector("arrivals", arrivals, debts.size)
    scales = _vector("scales", scales, debts.size)
    if np.any(debts < 0.0) or np.any(arrivals < 0.0):
        raise ValueError("debts and arrivals must be nonnegative")
    if np.any(scales < 0.0) or np.any(scales > 1.0):
        raise ValueError("scales must lie in [0, 1]")
    scalar_values = (risk_debt, incurred_risk, risk_budget)
    if any(not math.isfinite(value) or value < 0.0 for value in scalar_values):
        raise ValueError("risk queue inputs must be finite and nonnegative")
    next_service = np.maximum(0.0, debts + arrivals - scales)
    next_risk = max(0.0, risk_debt + incurred_risk - risk_budget)
    return next_service, float(next_risk)

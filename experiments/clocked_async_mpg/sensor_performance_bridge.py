"""Performance bridge from causal sensor decisions to Lyapunov certificates.

The functions in this module do not estimate game geometry.  They translate a
sequence of already certified, predictable one-step log multipliers into an
anytime energy envelope and quantify the price of disagreeing with a
counterfactual schedule.  This is the algebraic interface needed between the
dual-use sensor and a last-iterate stability theorem.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


Array = NDArray[np.float64]


def _log_multiplier_table(values: ArrayLike) -> Array:
    table = np.asarray(values, dtype=float)
    if table.ndim != 2 or table.shape[1] != 2 or table.shape[0] == 0:
        raise ValueError("log multipliers must have shape (horizon, 2)")
    if not np.all(np.isfinite(table)):
        raise ValueError("log multipliers must be finite")
    return table


def _binary_actions(values: ArrayLike, horizon: int, *, name: str) -> NDArray[np.int_]:
    actions = np.asarray(values, dtype=int)
    if actions.shape != (horizon,) or not np.all((actions == 0) | (actions == 1)):
        raise ValueError(f"{name} must be a binary horizon-length vector")
    return actions


def certified_schedule_log_cost(
    log_multipliers: ArrayLike,
    actions: ArrayLike,
) -> float:
    """Return the cumulative certified log multiplier of one schedule."""

    table = _log_multiplier_table(log_multipliers)
    selected = _binary_actions(actions, table.shape[0], name="actions")
    return float(np.sum(table[np.arange(table.shape[0]), selected]))


def schedule_excess_and_mismatch_bound(
    log_multipliers: ArrayLike,
    actions: ArrayLike,
    comparator_actions: ArrayLike,
) -> dict[str, float | int]:
    """Compare two schedules on the same exogenous certificate contexts.

    The exact log-cost excess is bounded by the sum of the absolute per-event
    action gaps over events at which the schedules disagree.  The bound is
    deterministic and does not require iid contexts or independent actions.
    """

    table = _log_multiplier_table(log_multipliers)
    horizon = table.shape[0]
    selected = _binary_actions(actions, horizon, name="actions")
    comparator = _binary_actions(
        comparator_actions, horizon, name="comparator_actions"
    )
    indices = np.arange(horizon)
    exact_excess = float(
        np.sum(table[indices, selected] - table[indices, comparator])
    )
    mismatches = selected != comparator
    absolute_gaps = np.abs(table[:, 0] - table[:, 1])
    mismatch_bound = float(np.sum(absolute_gaps[mismatches]))
    if exact_excess > mismatch_bound + 1e-12:
        raise RuntimeError("internal mismatch bound violation")
    return {
        "action_mismatches": int(np.sum(mismatches)),
        "exact_log_cost_excess": exact_excess,
        "mismatch_penalty_bound": mismatch_bound,
    }


def anytime_energy_log_envelope(
    *,
    initial_energy: float,
    cumulative_log_multipliers: ArrayLike,
    failure_probability: float,
) -> Array:
    """Return the Ville envelope for a multiplicative Lyapunov certificate.

    Suppose ``E[V_(k+1) | F_k] <= q_k V_k`` with positive predictable
    multipliers and ``cumulative_log_multipliers[k] = sum_(j<k) log(q_j)``.
    Then ``V_k / (V_0 prod_(j<k) q_j)`` is a nonnegative supermartingale.
    Ville's inequality gives, simultaneously for every supplied time,

    ``log(V_k) <= log(V_0) + cumulative_log_multipliers[k] + log(1/delta)``

    with probability at least ``1-delta``.
    """

    cumulative = np.asarray(cumulative_log_multipliers, dtype=float)
    if cumulative.ndim != 1 or not np.all(np.isfinite(cumulative)):
        raise ValueError("cumulative log multipliers must be a finite vector")
    if not math.isfinite(initial_energy) or initial_energy <= 0.0:
        raise ValueError("initial energy must be finite and positive")
    if (
        not math.isfinite(failure_probability)
        or not 0.0 < failure_probability < 1.0
    ):
        raise ValueError("failure probability must lie strictly between zero and one")
    return (
        math.log(initial_energy)
        + cumulative
        + math.log(1.0 / failure_probability)
    )


def certified_contraction_after_mismatches(
    *,
    comparator_log_cost: float,
    mismatch_penalty_bound: float,
    horizon: int,
) -> float:
    """Return the remaining per-event certified contraction margin.

    A positive value ``kappa`` means the sensor schedule has certified log
    cost at most ``-kappa * horizon`` whenever the comparator cost and the
    supplied mismatch penalty bounds are valid.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not math.isfinite(comparator_log_cost):
        raise ValueError("comparator log cost must be finite")
    if not math.isfinite(mismatch_penalty_bound) or mismatch_penalty_bound < 0.0:
        raise ValueError("mismatch penalty must be finite and nonnegative")
    return float(-(comparator_log_cost + mismatch_penalty_bound) / horizon)

"""Perfect-observation upper bound for action-dependent geometry sensing."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .clocked_optimism_phase import (
    expected_quadratic_multiplier,
    heterogeneous_clock_metric,
)
from .run_dual_use_sensor_development import _transition_table


@dataclass(frozen=True)
class ControlledSensingBound:
    optimal_log_cost: float
    call_rate: float
    stationary_rotation_probability: float
    flow_residual: float
    normalization_residual: float
    calibration_residual: float
    solver_status: int
    state_count: int


def phase_log_multiplier_table(step: float, arrival: float) -> np.ndarray:
    """Return rows potential/rotation and columns plain/optimistic."""

    transitions = _transition_table(step)
    metric = np.diag(heterogeneous_clock_metric(arrival))
    probabilities = (arrival, 1.0 - arrival)
    table = np.empty((2, 2), dtype=float)
    for phase in (0, 1):
        for action in (0, 1):
            multiplier = expected_quadratic_multiplier(
                metric,
                tuple(transitions[phase, action, agent] for agent in (0, 1)),
                probabilities,
            )
            table[phase, action] = math.log(multiplier)
    return table


def belief_state_probabilities(
    *, persistence: float, rotation_fraction: float, maximum_age: int
) -> np.ndarray:
    """Reachable beliefs after a perfect paid observation, plus no observation."""

    if not 0.0 <= persistence < 1.0:
        raise ValueError("persistence must lie in [0, 1)")
    if not 0.0 <= rotation_fraction <= 1.0:
        raise ValueError("rotation fraction must lie in [0, 1]")
    if maximum_age < 1:
        raise ValueError("maximum age must be positive")
    ages = np.arange(1, maximum_age + 1, dtype=float)
    decay = persistence**ages
    after_potential = rotation_fraction * (1.0 - decay)
    after_rotation = rotation_fraction + (1.0 - rotation_fraction) * decay
    return np.concatenate(
        (np.asarray([rotation_fraction]), after_potential, after_rotation)
    )


def _transitions(probabilities: np.ndarray, maximum_age: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    states = probabilities.size
    if states != 1 + 2 * maximum_age:
        raise ValueError("belief vector and maximum age disagree")
    no_call_rows: list[int] = []
    no_call_cols: list[int] = []
    no_call_values: list[float] = []
    # State zero is the stationary no-observation state.
    no_call_rows.append(0)
    no_call_cols.append(0)
    no_call_values.append(1.0)
    for phase in (0, 1):
        offset = 1 + phase * maximum_age
        for age_index in range(maximum_age):
            source = offset + age_index
            target = 0 if age_index == maximum_age - 1 else offset + age_index + 1
            no_call_rows.append(source)
            no_call_cols.append(target)
            no_call_values.append(1.0)
    no_call = sparse.csr_matrix(
        (no_call_values, (no_call_rows, no_call_cols)), shape=(states, states)
    )

    call_rows: list[int] = []
    call_cols: list[int] = []
    call_values: list[float] = []
    potential_next = 1
    rotation_next = 1 + maximum_age
    for source, probability in enumerate(probabilities):
        call_rows.extend((source, source))
        call_cols.extend((potential_next, rotation_next))
        call_values.extend((1.0 - float(probability), float(probability)))
    call = sparse.csr_matrix(
        (call_values, (call_rows, call_cols)), shape=(states, states)
    )
    return no_call, call


def solve_perfect_observation_bound(
    log_multipliers: np.ndarray,
    *,
    persistence: float,
    rotation_fraction: float,
    optimism_budget: float,
    maximum_age: int,
) -> ControlledSensingBound:
    """Solve the average-cost constrained belief MDP by occupation measures.

    A paid optimistic action reveals the current hidden phase perfectly only
    after the current action.  A plain action produces no observation.  This
    is more informative than the noisy fingerprint controller and therefore
    gives an upper bound on its attainable improvement, while charging every
    observation-producing action against the same budget.
    """

    table = np.asarray(log_multipliers, dtype=float)
    if table.shape != (2, 2) or not np.all(np.isfinite(table)):
        raise ValueError("log multipliers must be a finite 2 by 2 table")
    if not 0.0 <= optimism_budget <= 1.0:
        raise ValueError("optimism budget must lie in [0, 1]")
    beliefs = belief_state_probabilities(
        persistence=persistence,
        rotation_fraction=rotation_fraction,
        maximum_age=maximum_age,
    )
    no_call, call = _transitions(beliefs, maximum_age)
    transitions = (no_call, call)
    state_count = beliefs.size
    variable_count = 2 * state_count
    costs = np.empty((state_count, 2), dtype=float)
    for action in (0, 1):
        costs[:, action] = (
            (1.0 - beliefs) * table[0, action] + beliefs * table[1, action]
        )

    flow = sparse.lil_matrix((state_count, variable_count), dtype=float)
    for state in range(state_count):
        for action in (0, 1):
            variable = 2 * state + action
            flow[state, variable] += 1.0
            row = transitions[action].getrow(state)
            for target, probability in zip(row.indices, row.data):
                flow[target, variable] -= float(probability)
    normalization = sparse.csr_matrix(np.ones((1, variable_count)))
    equality = sparse.vstack((flow[:-1].tocsr(), normalization), format="csr")
    equality_rhs = np.concatenate((np.zeros(state_count - 1), np.ones(1)))
    call_indicator = np.tile(np.asarray([0.0, 1.0]), state_count)
    result = linprog(
        costs.ravel(),
        A_ub=sparse.csr_matrix(call_indicator[None, :]),
        b_ub=np.asarray([optimism_budget]),
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"controlled-sensing LP failed: {result.message}")
    occupation = np.asarray(result.x, dtype=float).reshape(state_count, 2)
    state_occupation = np.sum(occupation, axis=1)
    full_flow = np.asarray(flow.tocsr() @ result.x).ravel()
    return ControlledSensingBound(
        optimal_log_cost=float(result.fun),
        call_rate=float(np.sum(occupation[:, 1])),
        stationary_rotation_probability=float(state_occupation @ beliefs),
        flow_residual=float(np.max(np.abs(full_flow))),
        normalization_residual=abs(float(np.sum(result.x)) - 1.0),
        calibration_residual=abs(
            float(state_occupation @ beliefs) - rotation_fraction
        ),
        solver_status=int(result.status),
        state_count=state_count,
    )


def best_periodic_fixed_cost(
    log_multipliers: np.ndarray,
    *,
    rotation_fraction: float,
    optimism_budget: float,
    period: int = 4,
) -> tuple[float, float]:
    """Return the best stationary cost and call fraction on a period grid."""

    table = np.asarray(log_multipliers, dtype=float)
    if table.shape != (2, 2):
        raise ValueError("log multipliers must have shape (2, 2)")
    feasible = [count / period for count in range(period + 1) if count / period <= optimism_budget + 1e-12]
    phase_costs = (1.0 - rotation_fraction) * table[0] + rotation_fraction * table[1]
    costs = [float((1.0 - fraction) * phase_costs[0] + fraction * phase_costs[1]) for fraction in feasible]
    index = int(np.argmin(costs))
    return costs[index], feasible[index]


def exact_phase_cost(
    log_multipliers: np.ndarray,
    *,
    rotation_fraction: float,
    optimism_budget: float,
) -> tuple[float, float]:
    """Return the phase-observed optimum under an average call budget."""

    table = np.asarray(log_multipliers, dtype=float)
    if table.shape != (2, 2):
        raise ValueError("log multipliers must have shape (2, 2)")
    gains = table[:, 0] - table[:, 1]
    masses = np.asarray([1.0 - rotation_fraction, rotation_fraction])
    order = np.argsort(-gains)
    remaining = optimism_budget
    calls = np.zeros(2)
    for phase in order:
        if gains[phase] <= 0.0 or masses[phase] == 0.0:
            continue
        calls[phase] = min(1.0, remaining / masses[phase])
        remaining -= masses[phase] * calls[phase]
        if remaining <= 1e-15:
            break
    cost = float(
        np.sum(
            masses
            * ((1.0 - calls) * table[:, 0] + calls * table[:, 1])
        )
    )
    return cost, float(masses @ calls)

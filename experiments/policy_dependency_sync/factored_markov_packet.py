"""Exact moving-dependency Markov packet model.

The model is a cooperative quadratic potential whose owner-specific local
interaction partner follows a lazy directed Markov chain.  A finite rollout
averages those local interactions, so its policy-dependency cone expands with
the horizon while its Markov/noise error contracts.  It is used only as a
theorem-facing CPU model, not as a standard MARL benchmark.
"""

from __future__ import annotations

from math import comb
from typing import Mapping, Sequence

import numpy as np

from .packet_debt import HorizonCertificate, bias_square_upper


def donor_order(owner: int, agents: int) -> tuple[int, ...]:
    if agents < 2 or not 0 <= owner < agents:
        raise ValueError("invalid owner or agent count")
    return tuple((owner + offset) % agents for offset in range(1, agents))


def offset_transition(donors: int, move_probability: float) -> np.ndarray:
    """Lazy directed-cycle transition over relative donor offsets."""

    if donors <= 0 or not 0.0 <= move_probability <= 1.0:
        raise ValueError("invalid donor count or move probability")
    matrix = np.zeros((donors, donors), dtype=float)
    for state in range(donors):
        matrix[state, state] += 1.0 - move_probability
        matrix[state, (state + 1) % donors] += move_probability
    return matrix


def average_offset_occupancy(
    donors: int,
    start_offset: int,
    move_probability: float,
    horizon: int,
) -> np.ndarray:
    """Expected fraction of the rollout spent with each interaction donor."""

    if horizon <= 0 or not 0 <= start_offset < donors:
        raise ValueError("invalid horizon or start offset")
    transition = offset_transition(donors, move_probability)
    distribution = np.zeros(donors, dtype=float)
    distribution[start_offset] = 1.0
    occupancy = np.zeros(donors, dtype=float)
    for _ in range(horizon):
        occupancy += distribution
        distribution = distribution @ transition
    return occupancy / horizon


def binomial_quantile(trials: int, probability: float, coverage: float) -> int:
    """Smallest integer ``q`` with ``P(Binomial(trials,p)<=q)>=coverage``."""

    if trials < 0 or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid binomial parameters")
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must lie in (0,1]")
    cumulative = 0.0
    for value in range(trials + 1):
        cumulative += (
            comb(trials, value)
            * probability**value
            * (1.0 - probability) ** (trials - value)
        )
        if cumulative + 1e-15 >= coverage:
            return value
    return trials


def prospective_offsets(
    donors: int,
    start_offset: int,
    move_probability: float,
    horizon: int,
    coverage: float,
) -> tuple[int, ...]:
    """Launch-time high-probability cone for the monotone offset path."""

    moves = binomial_quantile(horizon - 1, move_probability, coverage)
    count = min(donors, moves + 1)
    return tuple((start_offset + move) % donors for move in range(count))


def global_hessian(agents: int, strong_convexity: float, coupling: float) -> np.ndarray:
    """Stationary cooperative potential Hessian."""

    if agents < 2 or strong_convexity <= 0.0 or coupling < 0.0:
        raise ValueError("invalid potential parameters")
    matrix = np.full((agents, agents), -coupling / (agents - 1), dtype=float)
    np.fill_diagonal(matrix, strong_convexity + coupling)
    return matrix


def conditional_owner_row(
    owner: int,
    agents: int,
    strong_convexity: float,
    coupling: float,
    start_offset: int,
    move_probability: float,
    horizon: int,
) -> np.ndarray:
    """Expected finite-horizon gradient row conditional on launch context."""

    order = donor_order(owner, agents)
    occupancy = average_offset_occupancy(
        len(order), start_offset, move_probability, horizon
    )
    row = np.zeros(agents, dtype=float)
    row[owner] = strong_convexity + coupling
    for probability, donor in zip(occupancy, order, strict=True):
        row[donor] = -coupling * probability
    return row


def ar1_mean_variance(innovation_variance: float, correlation: float, horizon: int) -> float:
    """Variance of an AR(1)-correlated trajectory average."""

    if innovation_variance < 0.0 or not 0.0 <= correlation < 1.0 or horizon <= 0:
        raise ValueError("invalid variance, correlation, or horizon")
    inflation = 1.0 + 2.0 * sum(
        (1.0 - lag / horizon) * correlation**lag
        for lag in range(1, horizon)
    )
    return float(innovation_variance * inflation / horizon)


def horizon_certificate(
    *,
    owner: int,
    current_parameter: Sequence[float],
    target: Sequence[float],
    cache_for_owner: Sequence[float],
    start_offset: int,
    move_probability: float,
    horizon: int,
    cone_coverage: float,
    strong_convexity: float,
    coupling: float,
    innovation_variance: float,
    temporal_correlation: float,
    step_cap: float,
    delivery_motion_bound: Mapping[int, float] | None = None,
    message_cost: float = 1.0,
) -> tuple[HorizonCertificate, np.ndarray]:
    """Build an exact-model certificate and return its conditional row."""

    theta = np.asarray(current_parameter, dtype=float)
    target_array = np.asarray(target, dtype=float)
    cache = np.asarray(cache_for_owner, dtype=float).copy()
    if theta.ndim != 1 or theta.shape != target_array.shape or theta.shape != cache.shape:
        raise ValueError("parameter, target, and cache must be equal-length vectors")
    agents = theta.size
    if not 0 <= owner < agents:
        raise ValueError("invalid owner")
    if message_cost < 0.0:
        raise ValueError("message cost must be nonnegative")
    cache[owner] = theta[owner]
    delivery_motion_bound = {} if delivery_motion_bound is None else delivery_motion_bound

    row = conditional_owner_row(
        owner,
        agents,
        strong_convexity,
        coupling,
        start_offset,
        move_probability,
        horizon,
    )
    stationary_row = global_hessian(agents, strong_convexity, coupling)[owner]
    error = theta - target_array
    markov_component = abs(float((row - stationary_row) @ error))
    order = donor_order(owner, agents)
    cone = set(
        prospective_offsets(
            len(order),
            start_offset,
            move_probability,
            horizon,
            cone_coverage,
        )
    )
    base_components = [markov_component]
    stale: dict[tuple[int, int], float] = {}
    fresh: dict[tuple[int, int], float] = {}
    costs: dict[tuple[int, int], dict[str, float]] = {}
    for offset, donor in enumerate(order):
        coefficient = abs(float(row[donor]))
        stale_radius = coefficient * abs(float(theta[donor] - cache[donor]))
        if offset in cone:
            edge = (donor, owner)
            stale[edge] = stale_radius
            fresh[edge] = coefficient * float(delivery_motion_bound.get(donor, 0.0))
            costs[edge] = {"message": float(message_cost)}
        else:
            base_components.append(stale_radius)
    certificate = HorizonCertificate(
        horizon=horizon,
        base_bias_components=tuple(base_components),
        gradient_variance=ar1_mean_variance(
            innovation_variance, temporal_correlation, horizon
        ),
        smoothness=float(strong_convexity + coupling),
        step_cap=float(step_cap),
        base_cost_by_resource={"environment": float(horizon)},
        stale_radius_by_edge=stale,
        fresh_radius_by_edge=fresh,
        cost_by_edge=costs,
    )
    return certificate, row


def actual_conditional_bias(
    owner: int,
    current_parameter: Sequence[float],
    target: Sequence[float],
    cache_for_owner: Sequence[float],
    conditional_row: Sequence[float],
    refreshed_edges: Sequence[tuple[int, int]],
    strong_convexity: float,
    coupling: float,
) -> float:
    """Exact scalar packet-mean bias at birth for certificate tests."""

    theta = np.asarray(current_parameter, dtype=float)
    target_array = np.asarray(target, dtype=float)
    cache = np.asarray(cache_for_owner, dtype=float).copy()
    row = np.asarray(conditional_row, dtype=float)
    for donor, recipient in refreshed_edges:
        if recipient != owner:
            raise ValueError("edge recipient does not match owner")
        cache[donor] = theta[donor]
    cache[owner] = theta[owner]
    packet_mean = float(row @ (cache - target_array))
    current_gradient = float(
        global_hessian(theta.size, strong_convexity, coupling)[owner]
        @ (theta - target_array)
    )
    return packet_mean - current_gradient


def certificate_dominates_actual_bias(
    certificate: HorizonCertificate,
    actual_bias: float,
    refreshed_edges: Sequence[tuple[int, int]],
) -> bool:
    return bool(
        actual_bias * actual_bias
        <= bias_square_upper(certificate, refreshed_edges) + 1e-12
    )


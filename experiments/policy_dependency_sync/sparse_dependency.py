"""Bounds and masks for sparse policy-gradient dependencies."""

from __future__ import annotations

from collections.abc import Mapping, Set


def validate_neighborhood(
    agents: int,
    owner: int,
    neighbors: Set[int],
) -> None:
    if agents < 2:
        raise ValueError("at least two agents are required")
    if owner < 0 or owner >= agents:
        raise ValueError("owner is outside the agent set")
    if owner in neighbors:
        raise ValueError("the owner cannot be its own dependency neighbor")
    if any(neighbor < 0 or neighbor >= agents for neighbor in neighbors):
        raise ValueError("neighbor is outside the agent set")


def omitted_dependency_tail(
    cross_sensitivities: Mapping[int, float],
    version_mismatches: Mapping[int, float],
    neighbors: Set[int],
) -> float:
    """Return ``sum_(j outside N) L_ij ||theta_j-chi_ji||``."""
    if set(cross_sensitivities) != set(version_mismatches):
        raise ValueError("sensitivity and mismatch donors must match")
    if min(cross_sensitivities.values(), default=0.0) < 0.0:
        raise ValueError("cross sensitivities must be nonnegative")
    if min(version_mismatches.values(), default=0.0) < 0.0:
        raise ValueError("version mismatches must be nonnegative")
    return float(
        sum(
            cross_sensitivities[donor] * version_mismatches[donor]
            for donor in cross_sensitivities
            if donor not in neighbors
        )
    )


def sparse_score_error_bound(
    maximum_step: float,
    objective_gradient_norm: float,
    local_candidate_gradient_norm: float,
    omitted_gradient_radius: float,
    smoothness: float,
) -> float:
    """Uniform smooth drift-score error caused by an omitted gradient tail."""
    values = (
        maximum_step,
        objective_gradient_norm,
        local_candidate_gradient_norm,
        omitted_gradient_radius,
    )
    if min(values) < 0.0:
        raise ValueError("steps, norms, and radii must be nonnegative")
    if smoothness <= 0.0:
        raise ValueError("smoothness must be positive")
    alignment = maximum_step * objective_gradient_norm * omitted_gradient_radius
    squared_norm = omitted_gradient_radius * (
        2.0 * local_candidate_gradient_norm + omitted_gradient_radius
    )
    curvature = 0.5 * smoothness * maximum_step**2 * squared_norm
    return float(alignment + curvature)

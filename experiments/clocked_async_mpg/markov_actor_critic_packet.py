"""Finite-schedule certificates for asynchronous actor--critic packets.

Each independent reset trajectory is treated as one bounded vector sample.
No independence between transitions within a trajectory is assumed.  The
module only combines declared deterministic envelopes; estimating or fitting
an envelope from the same outcomes is deliberately outside this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ActorCriticPacketCertificate:
    actor_statistical_radius: float
    actor_truncation_radius: float
    actor_critic_bias_radius: float
    actor_version_radius: float
    actor_total_radius: float
    critic_statistical_radius: float
    critic_policy_version_radius: float
    critic_parameter_version_radius: float
    critic_total_radius: float


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def simultaneous_vector_mean_radius(
    *,
    vector_dimension: int,
    trajectory_count: int,
    trajectory_norm_bound: float,
    scheduled_packet_count: int,
    joint_coordinate_count: int,
    failure_probability: float,
) -> float:
    """Coordinate-Hoeffding radius valid for a finite adaptive schedule.

    Conditional on each packet's birth filtration, the reset trajectories in
    that packet must be independent and their vector norms at most
    ``trajectory_norm_bound``.  A union bound over all declared packets and
    all actor/critic coordinates then gives a simultaneous Euclidean radius.
    Policies and packet birth times may be adaptive.  Completion order must
    not depend on the realized trajectory innovations.
    """

    dimension = _positive_integer("vector_dimension", vector_dimension)
    samples = _positive_integer("trajectory_count", trajectory_count)
    packets = _positive_integer("scheduled_packet_count", scheduled_packet_count)
    coordinates = _positive_integer("joint_coordinate_count", joint_coordinate_count)
    if coordinates < dimension:
        raise ValueError("joint_coordinate_count cannot be smaller than vector_dimension")
    bound = _nonnegative("trajectory_norm_bound", trajectory_norm_bound)
    delta = float(failure_probability)
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("failure_probability must lie strictly between zero and one")
    if bound == 0.0:
        return 0.0
    coordinate_radius = bound * math.sqrt(
        2.0 * math.log(2.0 * packets * coordinates / delta) / samples
    )
    return float(math.sqrt(dimension) * coordinate_radius)


def bounded_version_displacement(
    *, max_intervening_updates: int, max_update_norm: float
) -> float:
    """Triangle-inequality displacement from a bounded version delay."""

    if isinstance(max_intervening_updates, bool) or int(max_intervening_updates) != max_intervening_updates:
        raise ValueError("max_intervening_updates must be a nonnegative integer")
    delay = int(max_intervening_updates)
    if delay < 0:
        raise ValueError("max_intervening_updates must be a nonnegative integer")
    return float(delay * _nonnegative("max_update_norm", max_update_norm))


def build_actor_critic_packet_certificate(
    *,
    actor_statistical_radius: float,
    actor_truncation_radius: float,
    actor_critic_sensitivity: float,
    birth_critic_radius: float,
    actor_policy_lipschitz: float,
    policy_version_displacement: float,
    critic_statistical_radius: float,
    critic_policy_lipschitz: float,
    critic_parameter_lipschitz: float,
    critic_version_displacement: float,
) -> ActorCriticPacketCertificate:
    """Assemble arrival-time radii from stored birth metadata and paths.

    The actor packet mean is allowed to use the birth critic.  Its bias from
    the exact birth-policy gradient is bounded by
    ``actor_critic_sensitivity * birth_critic_radius``.  The critic packet is
    a stochastic operator evaluated at the birth policy and birth critic;
    the two Lipschitz terms translate it to the current population operator.
    """

    actor_statistical = _nonnegative(
        "actor_statistical_radius", actor_statistical_radius
    )
    actor_truncation = _nonnegative(
        "actor_truncation_radius", actor_truncation_radius
    )
    actor_critic = _nonnegative(
        "actor_critic_sensitivity", actor_critic_sensitivity
    ) * _nonnegative("birth_critic_radius", birth_critic_radius)
    policy_lipschitz = _nonnegative(
        "actor_policy_lipschitz", actor_policy_lipschitz
    )
    policy_displacement = _nonnegative(
        "policy_version_displacement", policy_version_displacement
    )
    actor_version = policy_lipschitz * policy_displacement

    critic_statistical = _nonnegative(
        "critic_statistical_radius", critic_statistical_radius
    )
    critic_policy = _nonnegative(
        "critic_policy_lipschitz", critic_policy_lipschitz
    ) * policy_displacement
    critic_parameter = _nonnegative(
        "critic_parameter_lipschitz", critic_parameter_lipschitz
    ) * _nonnegative(
        "critic_version_displacement", critic_version_displacement
    )

    return ActorCriticPacketCertificate(
        actor_statistical_radius=actor_statistical,
        actor_truncation_radius=actor_truncation,
        actor_critic_bias_radius=actor_critic,
        actor_version_radius=actor_version,
        actor_total_radius=(
            actor_statistical + actor_truncation + actor_critic + actor_version
        ),
        critic_statistical_radius=critic_statistical,
        critic_policy_version_radius=critic_policy,
        critic_parameter_version_radius=critic_parameter,
        critic_total_radius=(critic_statistical + critic_policy + critic_parameter),
    )


def contracted_critic_radius(
    *,
    critic_radius: float,
    critic_gradient_radius: float,
    strong_convexity: float,
    smoothness: float,
    step_size: float,
) -> float:
    """Radius after one inexact SPD critic correction.

    For ``F(w)=A(w-w*)`` with ``mu I <= A <= L I`` and
    ``||F_hat-F(w)|| <= epsilon``, beta at most ``1/L`` gives
    ``||(w-w*)-beta F_hat|| <= (1-mu beta)c + beta epsilon``.
    """

    radius = _nonnegative("critic_radius", critic_radius)
    error = _nonnegative("critic_gradient_radius", critic_gradient_radius)
    mu = _nonnegative("strong_convexity", strong_convexity)
    lipschitz = _nonnegative("smoothness", smoothness)
    beta = _nonnegative("step_size", step_size)
    if mu == 0.0 or lipschitz == 0.0:
        raise ValueError("strong_convexity and smoothness must be positive")
    if mu > lipschitz + 1e-12:
        raise ValueError("strong_convexity cannot exceed smoothness")
    if beta * lipschitz > 1.0 + 1e-12:
        raise ValueError("step_size exceeds the SPD contraction interval")
    return float((1.0 - mu * beta) * radius + beta * error)

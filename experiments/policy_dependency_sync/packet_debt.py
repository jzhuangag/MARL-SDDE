"""Packet-debt Lyapunov decisions for asynchronous policy learning.

The launch rule prices the certified bias and variance carried by a newly
born rollout packet.  The receipt lemma shows why the same packet energy is
the amount needed to cancel the stale-gradient terms when that packet is
eventually consumed.  These are deterministic pieces of the theorem; this
module does not construct a critic confidence set.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Hashable, Mapping, Sequence


Edge = tuple[Hashable, Hashable]
Resource = Hashable


@dataclass(frozen=True)
class HorizonCertificate:
    """Predictable launch certificate for one rollout horizon.

    ``base_bias_components`` contains non-version contributions such as
    Markov-start, critic, truncation, and causal-cone errors.  Each edge has a
    stale radius and a certified post-refresh radius.  Cauchy--Schwarz turns
    the sum of all vector bias components into the additive squared bound used
    by the launch minimizer.
    """

    horizon: int
    base_bias_components: tuple[float, ...]
    gradient_variance: float
    smoothness: float
    step_cap: float
    base_cost_by_resource: Mapping[Resource, float]
    stale_radius_by_edge: Mapping[Edge, float]
    fresh_radius_by_edge: Mapping[Edge, float]
    cost_by_edge: Mapping[Edge, Mapping[Resource, float]]
    max_edges: int | None = None


@dataclass(frozen=True)
class LaunchChoice:
    horizon: int
    refreshed_edges: tuple[Edge, ...]
    bias_square_upper: float
    packet_debt: float
    resource_costs: Mapping[Resource, float]
    drift_plus_penalty_index: float


@dataclass(frozen=True)
class ReceiptComparatorBound:
    descent: float
    interference_remainder: float
    drift_upper: float


def _validate_certificate(certificate: HorizonCertificate) -> None:
    if certificate.horizon <= 0:
        raise ValueError("horizon must be positive")
    if certificate.gradient_variance < 0.0:
        raise ValueError("gradient variance must be nonnegative")
    if certificate.smoothness <= 0.0 or certificate.step_cap <= 0.0:
        raise ValueError("smoothness and step cap must be positive")
    if min(certificate.base_bias_components, default=0.0) < 0.0:
        raise ValueError("base bias components must be nonnegative")
    keys = set(certificate.stale_radius_by_edge)
    if keys != set(certificate.fresh_radius_by_edge) or keys != set(
        certificate.cost_by_edge
    ):
        raise ValueError("all edge maps must have identical keys")
    if min(certificate.stale_radius_by_edge.values(), default=0.0) < 0.0:
        raise ValueError("stale radii must be nonnegative")
    if min(certificate.fresh_radius_by_edge.values(), default=0.0) < 0.0:
        raise ValueError("fresh radii must be nonnegative")
    if min(certificate.base_cost_by_resource.values(), default=0.0) < 0.0:
        raise ValueError("base resource costs must be nonnegative")
    for costs in certificate.cost_by_edge.values():
        if min(costs.values(), default=0.0) < 0.0:
            raise ValueError("edge resource costs must be nonnegative")
    if certificate.max_edges is not None and certificate.max_edges < 0:
        raise ValueError("max_edges must be nonnegative")


def bias_square_upper(
    certificate: HorizonCertificate,
    refreshed_edges: Sequence[Edge],
) -> float:
    """Additive Cauchy upper bound on the squared conditional mean bias."""

    _validate_certificate(certificate)
    selected = set(refreshed_edges)
    all_edges = set(certificate.stale_radius_by_edge)
    if not selected <= all_edges:
        raise ValueError("refreshed edge is outside the certificate")
    component_count = max(
        1,
        len(certificate.base_bias_components) + len(all_edges),
    )
    square_sum = sum(value * value for value in certificate.base_bias_components)
    for edge in all_edges:
        radius = (
            certificate.fresh_radius_by_edge[edge]
            if edge in selected
            else certificate.stale_radius_by_edge[edge]
        )
        square_sum += radius * radius
    return float(component_count * square_sum)


def packet_history_energy(
    step_cap: float,
    bias_square: float,
    smoothness: float,
    gradient_variance: float,
) -> float:
    """Risk debt attached to a packet from birth until receipt.

    The coefficients ``5/4`` and ``1/2`` are those required by the receipt
    cancellation lemma for ``smoothness * step_cap <= 1/4``.
    """

    if step_cap <= 0.0 or bias_square < 0.0:
        raise ValueError("step cap must be positive and bias square nonnegative")
    if smoothness <= 0.0 or gradient_variance < 0.0:
        raise ValueError("invalid smoothness or variance")
    return float(
        1.25 * step_cap * bias_square
        + 0.5 * smoothness * step_cap * step_cap * gradient_variance
    )


def packet_debt(
    certificate: HorizonCertificate,
    refreshed_edges: Sequence[Edge],
) -> tuple[float, float]:
    """Return ``(bias_square_upper, packet_history_energy)``."""

    square = bias_square_upper(certificate, refreshed_edges)
    debt = packet_history_energy(
        certificate.step_cap,
        square,
        certificate.smoothness,
        certificate.gradient_variance,
    )
    return square, debt


def choose_horizon_and_graph(
    certificates: Sequence[HorizonCertificate],
    resource_queues: Mapping[Resource, float],
    learning_weight: float,
) -> LaunchChoice:
    """Exactly minimize launch packet debt plus virtual-queue price.

    For a fixed horizon the Cauchy certificate is additive over edges, hence
    every edge has a signed net debt reduction.  With a cardinality cap, the
    largest positive reductions are exactly optimal.  The final scan over the
    declared integer horizons is also exact.
    """

    if not certificates:
        raise ValueError("at least one horizon certificate is required")
    if min(resource_queues.values(), default=0.0) < 0.0 or learning_weight <= 0.0:
        raise ValueError("queues must be nonnegative and learning weight positive")

    choices: list[LaunchChoice] = []
    for certificate in certificates:
        _validate_certificate(certificate)
        used_resources = set(certificate.base_cost_by_resource)
        for edge_costs in certificate.cost_by_edge.values():
            used_resources.update(edge_costs)
        if not used_resources <= set(resource_queues):
            raise ValueError("every charged resource requires a queue price")
        all_edges = tuple(certificate.stale_radius_by_edge)
        component_count = max(
            1,
            len(certificate.base_bias_components) + len(all_edges),
        )
        coefficient = 1.25 * certificate.step_cap * component_count
        scored: list[tuple[float, Edge]] = []
        for edge in all_edges:
            stale = certificate.stale_radius_by_edge[edge]
            fresh = certificate.fresh_radius_by_edge[edge]
            debt_reduction = coefficient * (stale * stale - fresh * fresh)
            price = sum(
                resource_queues[resource] * cost
                for resource, cost in certificate.cost_by_edge[edge].items()
            )
            net = (
                learning_weight * debt_reduction
                - price
            )
            if net > 0.0:
                scored.append((float(net), edge))
        scored.sort(key=lambda item: (-item[0], repr(item[1])))
        if certificate.max_edges is not None:
            scored = scored[: certificate.max_edges]
        selected = tuple(edge for _, edge in scored)
        square, debt = packet_debt(certificate, selected)
        total_costs = {
            resource: float(certificate.base_cost_by_resource.get(resource, 0.0))
            for resource in resource_queues
        }
        for edge in selected:
            for resource, cost in certificate.cost_by_edge[edge].items():
                total_costs[resource] = total_costs.get(resource, 0.0) + cost
        price = sum(
            resource_queues[resource] * cost
            for resource, cost in total_costs.items()
        )
        choices.append(
            LaunchChoice(
                horizon=certificate.horizon,
                refreshed_edges=selected,
                bias_square_upper=square,
                packet_debt=debt,
                resource_costs={
                    resource: float(cost) for resource, cost in total_costs.items()
                },
                drift_plus_penalty_index=float(
                    learning_weight * debt + price
                ),
            )
        )
    return min(
        choices,
        key=lambda choice: (
            choice.drift_plus_penalty_index,
            choice.horizon,
            tuple(map(repr, choice.refreshed_edges)),
        ),
    )


def remaining_packet_coefficients(
    step_caps: Sequence[float],
    component_multipliers: Sequence[float],
    cross_sensitivities: Sequence[float],
    version_distances: Sequence[float],
) -> tuple[float, float]:
    """Return the linear and quadratic history cost of one block update.

    For the remaining packet ``q``, its contribution is based on
    ``m_q L_qi^2 ||theta_i-zeta_qi||^2``.  Moving block ``i`` by norm ``u``
    increases total history by at most ``h*u + kappa*u^2/2``.
    """

    lengths = {
        len(step_caps),
        len(component_multipliers),
        len(cross_sensitivities),
        len(version_distances),
    }
    if len(lengths) != 1:
        raise ValueError("remaining-packet sequences must have equal lengths")
    if min(step_caps, default=0.0) < 0.0:
        raise ValueError("step caps must be nonnegative")
    if min(component_multipliers, default=0.0) < 0.0:
        raise ValueError("component multipliers must be nonnegative")
    if min(cross_sensitivities, default=0.0) < 0.0:
        raise ValueError("cross sensitivities must be nonnegative")
    if min(version_distances, default=0.0) < 0.0:
        raise ValueError("version distances must be nonnegative")
    linear = 0.0
    curvature = 0.0
    for cap, multiplier, sensitivity, distance in zip(
        step_caps,
        component_multipliers,
        cross_sensitivities,
        version_distances,
        strict=True,
    ):
        squared_sensitivity = sensitivity * sensitivity
        linear += 2.5 * cap * multiplier * squared_sensitivity * distance
        curvature += 2.5 * cap * multiplier * squared_sensitivity
    return float(linear), float(curvature)


def full_cap_receipt_bound(
    gradient_norm: float,
    bias_radius: float,
    noise_std: float,
    smoothness: float,
    step_cap: float,
    remaining_linear: float = 0.0,
    remaining_curvature: float = 0.0,
) -> ReceiptComparatorBound:
    """Stationarity bound for consuming a packet at its full certified cap.

    The returned upper bound includes removal of the completing packet's
    history energy and the motion imposed on every other pending packet.
    """

    if min(
        gradient_norm,
        bias_radius,
        noise_std,
        remaining_linear,
        remaining_curvature,
    ) < 0.0:
        raise ValueError("receipt-bound inputs must be nonnegative")
    if smoothness <= 0.0 or step_cap <= 0.0:
        raise ValueError("smoothness and step cap must be positive")
    if smoothness * step_cap > 0.25 + 1e-15:
        raise ValueError("receipt cancellation requires L * step_cap <= 1/4")
    if remaining_curvature * step_cap > 0.125 + 1e-15:
        raise ValueError("remaining history requires kappa * step_cap <= 1/8")

    descent = -0.25 * step_cap * gradient_norm * gradient_norm
    remainder = (
        2.0 * step_cap * remaining_linear * remaining_linear
        + step_cap * remaining_linear * (bias_radius + noise_std)
        + remaining_curvature * step_cap * step_cap * bias_radius * bias_radius
        + 0.5
        * remaining_curvature
        * step_cap
        * step_cap
        * noise_std
        * noise_std
    )
    return ReceiptComparatorBound(
        descent=float(descent),
        interference_remainder=float(remainder),
        drift_upper=float(descent + remainder),
    )


def direct_full_cap_drift_upper(
    gradient: float,
    mean_bias: float,
    noise_std: float,
    smoothness: float,
    step_cap: float,
    remaining_linear: float = 0.0,
    remaining_curvature: float = 0.0,
) -> float:
    """Direct one-dimensional smoothness bound used to test the lemma."""

    mean = gradient + mean_bias
    second_moment = mean * mean + noise_std * noise_std
    objective = (
        -step_cap * gradient * mean
        + 0.5 * smoothness * step_cap * step_cap * second_moment
    )
    completing_energy = packet_history_energy(
        step_cap,
        mean_bias * mean_bias,
        smoothness,
        noise_std * noise_std,
    )
    remaining = remaining_linear * step_cap * sqrt(second_moment)
    remaining += 0.5 * remaining_curvature * step_cap * step_cap * second_moment
    return float(objective - completing_energy + remaining)

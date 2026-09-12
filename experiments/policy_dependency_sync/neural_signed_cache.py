"""Low-complexity signed policy-cache scoring for differentiable CTDE critics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


TensorGroup = Sequence[torch.Tensor]


@dataclass(frozen=True)
class NeuralCacheChoice:
    donor: int | None
    index: float
    estimated_alignment: float
    cache_reset_benefit: float
    candidate_count: int
    vjp_calls: int
    null_index: float
    best_edge_donor: int | None
    best_edge_index: float | None
    best_edge_learning_index_delta: float | None
    best_edge_cache_reset_benefit: float | None
    best_edge_queue_price: float | None


def _gradient_tuple(
    loss: torch.Tensor, parameters: TensorGroup, *, create_graph: bool
) -> tuple[torch.Tensor, ...]:
    parameters = tuple(parameters)
    if not parameters:
        raise ValueError("parameter group must be nonempty")
    return tuple(
        torch.autograd.grad(
            loss,
            parameters,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=False,
        )
    )


def signed_cache_choice(
    *,
    current_owner_loss: torch.Tensor,
    cached_owner_loss: torch.Tensor,
    owner_parameters: TensorGroup,
    donor_parameters: Mapping[int, TensorGroup],
    donor_displacements: Mapping[int, TensorGroup],
    taylor_remainder_by_donor: Mapping[int, float],
    cache_reset_benefit_by_donor: Mapping[int, float] | None = None,
    step: float,
    smoothness: float,
    packet_second_moment_upper: float,
    receipt_motion_upper: float,
    communication_queue: float,
    learning_weight: float,
    message_cost: float = 1.0,
) -> NeuralCacheChoice:
    """Choose null or one refresh with one cross-policy reverse VJP.

    The action-dependent statistic is the first-order change in alignment
    between the current owner gradient and the gradient produced by the
    worker's cached joint policy.  Curvature and receipt-motion inputs are
    common certified upper bounds, so they cannot create a spurious edge
    preference.
    """

    scalars = (
        step,
        smoothness,
        packet_second_moment_upper,
        receipt_motion_upper,
        communication_queue,
        learning_weight,
        message_cost,
    )
    if any(value < 0.0 for value in scalars) or step == 0.0:
        raise ValueError("invalid nonnegative scoring constants")
    donors = tuple(sorted(donor_parameters))
    if set(donors) != set(donor_displacements) or set(donors) != set(
        taylor_remainder_by_donor
    ):
        raise ValueError("donor maps must have identical keys")
    if min(taylor_remainder_by_donor.values(), default=0.0) < 0.0:
        raise ValueError("Taylor remainders must be nonnegative")
    reset_benefits = (
        {donor: 0.0 for donor in donors}
        if cache_reset_benefit_by_donor is None
        else {
            int(donor): float(value)
            for donor, value in cache_reset_benefit_by_donor.items()
        }
    )
    if set(reset_benefits) != set(donors) or min(
        reset_benefits.values(), default=0.0
    ) < 0.0:
        raise ValueError("cache reset benefits must be nonnegative and match donors")

    current_gradient = tuple(
        value.detach()
        for value in _gradient_tuple(
            current_owner_loss, owner_parameters, create_graph=False
        )
    )
    cached_gradient = _gradient_tuple(
        cached_owner_loss, owner_parameters, create_graph=True
    )
    if len(current_gradient) != len(cached_gradient):
        raise ValueError("owner gradient structures differ")
    base_alignment_tensor = sum(
        (current * cached).sum()
        for current, cached in zip(current_gradient, cached_gradient)
    )

    flat_parameters: list[torch.Tensor] = []
    group_slices: dict[int, slice] = {}
    for donor in donors:
        group = tuple(donor_parameters[donor])
        displacement = tuple(donor_displacements[donor])
        if len(group) != len(displacement) or not group:
            raise ValueError("invalid donor parameter/displacement group")
        start = len(flat_parameters)
        flat_parameters.extend(group)
        group_slices[donor] = slice(start, len(flat_parameters))

    if flat_parameters:
        flat_vjp = torch.autograd.grad(
            base_alignment_tensor,
            tuple(flat_parameters),
            retain_graph=True,
            allow_unused=True,
        )
    else:
        flat_vjp = ()

    base_alignment = float(base_alignment_tensor.detach().cpu())
    common_drift = (
        0.5 * smoothness * step * step * packet_second_moment_upper
        + step * receipt_motion_upper
    )
    candidates: list[tuple[float, int, int | None, float, float]] = [
        (
            learning_weight * (-step * base_alignment + common_drift),
            0,
            None,
            base_alignment,
            0.0,
        )
    ]
    edge_diagnostics: list[tuple[float, int, float, float, float]] = []
    for donor in donors:
        selected_vjp = flat_vjp[group_slices[donor]]
        displacement = tuple(donor_displacements[donor])
        change = 0.0
        for derivative, delta in zip(selected_vjp, displacement):
            if derivative is not None:
                if derivative.shape != delta.shape:
                    raise ValueError("donor displacement shape mismatch")
                change += float((derivative.detach() * delta.detach()).sum().cpu())
        lower_alignment = (
            base_alignment
            + change
            - float(taylor_remainder_by_donor[donor])
        )
        drift = -step * lower_alignment + common_drift
        reset_benefit = reset_benefits[donor]
        index = (
            learning_weight * drift
            - reset_benefit
            + communication_queue * message_cost
        )
        edge_diagnostics.append(
            (
                float(index),
                int(donor),
                float(
                    learning_weight
                    * (drift - (-step * base_alignment + common_drift))
                ),
                float(reset_benefit),
                float(communication_queue * message_cost),
            )
        )
        candidates.append(
            (float(index), 1, donor, lower_alignment, reset_benefit)
        )

    best = min(
        candidates,
        key=lambda row: (row[0], row[1], -1 if row[2] is None else row[2]),
    )
    best_edge = min(edge_diagnostics, default=None, key=lambda row: (row[0], row[1]))
    return NeuralCacheChoice(
        donor=best[2],
        index=best[0],
        estimated_alignment=best[3],
        cache_reset_benefit=best[4],
        candidate_count=len(candidates),
        vjp_calls=1 if donors else 0,
        null_index=float(candidates[0][0]),
        best_edge_donor=None if best_edge is None else int(best_edge[1]),
        best_edge_index=None if best_edge is None else float(best_edge[0]),
        best_edge_learning_index_delta=(
            None if best_edge is None else float(best_edge[2])
        ),
        best_edge_cache_reset_benefit=(
            None if best_edge is None else float(best_edge[3])
        ),
        best_edge_queue_price=None if best_edge is None else float(best_edge[4]),
    )


def parameter_bytes(parameters: TensorGroup) -> int:
    return int(
        sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    )

"""Exact deterministic-policy alignment for continuous local factor critics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ContinuousFactorAlignment:
    alignment_by_candidate: dict[Hashable | None, float]
    owner_action: np.ndarray
    owner_action_direction: np.ndarray
    actor_reverse_calls: int
    critic_action_gradient_calls: int


def _flat_gradient(
    scalar: torch.Tensor,
    parameters: Sequence[torch.Tensor],
    *,
    retain_graph: bool,
) -> np.ndarray:
    gradient = torch.autograd.grad(
        scalar,
        tuple(parameters),
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )
    return np.concatenate(
        [value.detach().cpu().numpy().ravel() for value in gradient]
    ).astype(float, copy=False)


def deterministic_action_direction(
    *,
    actor: nn.Module,
    observation: np.ndarray,
    reference_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute ``mu`` and ``J_theta mu @ v`` once per launch."""

    parameters = tuple(actor.parameters())
    if not parameters:
        raise ValueError("continuous actor must expose trainable parameters")
    actor_dtype = parameters[0].dtype
    actor_device = parameters[0].device
    if any(
        parameter.dtype != actor_dtype or parameter.device != actor_device
        for parameter in parameters
    ):
        raise ValueError("continuous actor parameters must share dtype and device")
    tensor = torch.as_tensor(
        observation, dtype=actor_dtype, device=actor_device
    ).unsqueeze(0)
    action = actor(tensor).squeeze(0)
    if action.ndim != 1:
        raise ValueError("continuous actor must return one action vector")
    parameter_count = sum(parameter.numel() for parameter in parameters)
    direction = np.asarray(reference_direction, dtype=float).ravel()
    if direction.shape != (parameter_count,) or not np.all(np.isfinite(direction)):
        raise ValueError("reference direction has the wrong parameter dimension")
    action_direction: list[float] = []
    for coordinate in range(action.numel()):
        gradient = _flat_gradient(
            action[coordinate],
            parameters,
            retain_graph=coordinate + 1 < action.numel(),
        )
        action_direction.append(float(gradient @ direction))
    return (
        action.detach().cpu().numpy().astype(float, copy=False),
        np.asarray(action_direction, dtype=float),
        int(action.numel()),
    )


def _action_gradient(
    value: torch.Tensor,
    action: torch.Tensor,
) -> np.ndarray:
    gradient = torch.autograd.grad(
        value,
        action,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]
    return gradient.detach().cpu().numpy().ravel().astype(float, copy=False)


def continuous_factor_alignment_candidates(
    *,
    actor: nn.Module,
    owner_observation: np.ndarray,
    reference_direction: np.ndarray,
    self_factor: nn.Module,
    pair_factor_by_donor: Mapping[Hashable, nn.Module],
    donor_observation_by_donor: Mapping[Hashable, np.ndarray],
    cached_profile_by_donor: Mapping[Hashable, np.ndarray],
    current_profile_by_donor: Mapping[Hashable, np.ndarray],
) -> ContinuousFactorAlignment:
    """Score null and one-edge profiles under a continuous factor critic.

    ``self_factor(owner_observation, owner_action)`` and
    ``pair_factor(owner_observation, donor_observation, owner_action, profile)``
    must each return one scalar per batch element.
    """

    donors = tuple(sorted(pair_factor_by_donor, key=str))
    mappings = (
        donor_observation_by_donor,
        cached_profile_by_donor,
        current_profile_by_donor,
    )
    if any(set(donors) != set(mapping) for mapping in mappings):
        raise ValueError("continuous donor mappings must have identical keys")
    owner_action, action_direction, actor_calls = deterministic_action_direction(
        actor=actor,
        observation=owner_observation,
        reference_direction=reference_direction,
    )
    parameters = tuple(actor.parameters())
    actor_dtype = parameters[0].dtype
    actor_device = parameters[0].device
    owner_tensor = torch.as_tensor(
        owner_observation, dtype=actor_dtype, device=actor_device
    ).unsqueeze(0)

    def new_action() -> torch.Tensor:
        return torch.as_tensor(
            owner_action, dtype=actor_dtype, device=actor_device
        ).unsqueeze(0).requires_grad_(True)

    action = new_action()
    self_value = self_factor(owner_tensor, action).reshape(-1)
    if self_value.numel() != 1:
        raise ValueError("self factor must return one scalar")
    null_gradient = _action_gradient(self_value[0], action)
    cached_gradient: dict[Hashable, np.ndarray] = {}
    current_gradient: dict[Hashable, np.ndarray] = {}
    for donor in donors:
        donor_tensor = torch.as_tensor(
            donor_observation_by_donor[donor],
            dtype=actor_dtype,
            device=actor_device,
        ).unsqueeze(0)
        cached = torch.as_tensor(
            cached_profile_by_donor[donor],
            dtype=actor_dtype,
            device=actor_device,
        ).unsqueeze(0)
        current = torch.as_tensor(
            current_profile_by_donor[donor],
            dtype=actor_dtype,
            device=actor_device,
        ).unsqueeze(0)
        for profile, destination in (
            (cached, cached_gradient),
            (current, current_gradient),
        ):
            action = new_action()
            value = pair_factor_by_donor[donor](
                owner_tensor,
                donor_tensor,
                action,
                profile,
            ).reshape(-1)
            if value.numel() != 1:
                raise ValueError("pair factor must return one scalar")
            destination[donor] = _action_gradient(value[0], action)
        null_gradient = null_gradient + cached_gradient[donor]
    alignments: dict[Hashable | None, float] = {
        None: float(action_direction @ null_gradient)
    }
    for donor in donors:
        refreshed = null_gradient + current_gradient[donor] - cached_gradient[donor]
        alignments[donor] = float(action_direction @ refreshed)
    return ContinuousFactorAlignment(
        alignment_by_candidate=alignments,
        owner_action=owner_action,
        owner_action_direction=action_direction,
        actor_reverse_calls=int(actor_calls),
        critic_action_gradient_calls=1 + 2 * len(donors),
    )

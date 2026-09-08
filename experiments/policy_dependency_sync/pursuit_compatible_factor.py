"""Compatible pair-factor alignment for low-degree asynchronous MARL.

For a discrete owner actor, the directional derivatives of its action
probabilities are computed once.  A pair-factor critic then evaluates the null
cache profile and every one-edge refresh without an additional reverse pass per
edge.  With a fixed action count, candidate scoring is linear in local degree.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


Action = Hashable


@dataclass(frozen=True)
class CompatibleFactorAlignment:
    alignment_by_candidate: dict[Action | None, float]
    owner_probabilities: np.ndarray
    owner_probability_direction: np.ndarray
    owner_reverse_calls: int
    pair_factor_evaluations: int


def _flat_parameter_gradient(
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


def owner_probability_direction(
    *,
    actor: nn.Module,
    observation: np.ndarray,
    reference_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return probabilities and ``<v, grad_theta pi(u)>`` for every action."""

    tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    probabilities = torch.softmax(actor(tensor), dim=-1).squeeze(0)
    if probabilities.ndim != 1:
        raise ValueError("owner actor must return one categorical logit vector")
    parameters = tuple(actor.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    direction = np.asarray(reference_direction, dtype=float).ravel()
    if direction.shape != (parameter_count,) or not np.all(np.isfinite(direction)):
        raise ValueError("reference direction has the wrong parameter dimension")
    derivatives: list[float] = []
    for action in range(probabilities.numel()):
        gradient = _flat_parameter_gradient(
            probabilities[action],
            parameters,
            retain_graph=action + 1 < probabilities.numel(),
        )
        derivatives.append(float(direction @ gradient))
    return (
        probabilities.detach().cpu().numpy().astype(float, copy=False),
        np.asarray(derivatives, dtype=float),
        int(probabilities.numel()),
    )


def factorized_alignment_candidates(
    *,
    actor: nn.Module,
    owner_observation: np.ndarray,
    reference_direction: np.ndarray,
    self_factor: np.ndarray,
    pair_factor_by_donor: Mapping[Action, np.ndarray],
    cached_probability_by_donor: Mapping[Action, np.ndarray],
    current_probability_by_donor: Mapping[Action, np.ndarray],
) -> CompatibleFactorAlignment:
    """Score null and one-edge refreshes from a pair-factor critic."""

    donors = tuple(sorted(pair_factor_by_donor, key=str))
    if set(donors) != set(cached_probability_by_donor) or set(donors) != set(
        current_probability_by_donor
    ):
        raise ValueError("all donor mappings must have identical keys")
    owner_probability, owner_direction, reverse_calls = owner_probability_direction(
        actor=actor,
        observation=owner_observation,
        reference_direction=reference_direction,
    )
    actions = owner_probability.size
    owner_values = np.asarray(self_factor, dtype=float)
    if owner_values.shape != (actions,) or not np.all(np.isfinite(owner_values)):
        raise ValueError("self factor has the wrong shape")
    null_values = owner_values.copy()
    pair_matrices: dict[Action, np.ndarray] = {}
    cached_probabilities: dict[Action, np.ndarray] = {}
    current_probabilities: dict[Action, np.ndarray] = {}
    for donor in donors:
        matrix = np.asarray(pair_factor_by_donor[donor], dtype=float)
        cached = np.asarray(cached_probability_by_donor[donor], dtype=float)
        current = np.asarray(current_probability_by_donor[donor], dtype=float)
        if matrix.shape != (actions, actions):
            raise ValueError("pair factors must be action-by-action matrices")
        if cached.shape != (actions,) or current.shape != (actions,):
            raise ValueError("donor probabilities have the wrong shape")
        if not all(
            np.all(np.isfinite(value)) for value in (matrix, cached, current)
        ):
            raise ValueError("factor inputs must be finite")
        if abs(float(np.sum(cached)) - 1.0) > 1e-6 or abs(
            float(np.sum(current)) - 1.0
        ) > 1e-6:
            raise ValueError("donor probabilities must sum to one")
        pair_matrices[donor] = matrix
        cached_probabilities[donor] = cached
        current_probabilities[donor] = current
        null_values += matrix @ cached

    alignments: dict[Action | None, float] = {
        None: float(owner_direction @ null_values)
    }
    for donor in donors:
        refreshed_values = null_values + pair_matrices[donor] @ (
            current_probabilities[donor] - cached_probabilities[donor]
        )
        alignments[donor] = float(owner_direction @ refreshed_values)
    return CompatibleFactorAlignment(
        alignment_by_candidate=alignments,
        owner_probabilities=owner_probability,
        owner_probability_direction=owner_direction,
        owner_reverse_calls=reverse_calls,
        pair_factor_evaluations=len(donors),
    )


def brute_force_factor_alignment(
    *,
    owner_probability_direction: np.ndarray,
    self_factor: np.ndarray,
    pair_factor_by_donor: Mapping[Action, np.ndarray],
    probability_by_donor: Mapping[Action, np.ndarray],
) -> float:
    """Enumerate neighbor joint actions for a regression oracle."""

    donors = tuple(sorted(pair_factor_by_donor, key=str))
    direction = np.asarray(owner_probability_direction, dtype=float)
    actions = direction.size
    result = 0.0
    for owner_action in range(actions):
        for neighbor_actions in itertools.product(range(actions), repeat=len(donors)):
            probability = 1.0
            value = float(np.asarray(self_factor)[owner_action])
            for donor, donor_action in zip(donors, neighbor_actions):
                probability *= float(probability_by_donor[donor][donor_action])
                value += float(
                    pair_factor_by_donor[donor][owner_action, donor_action]
                )
            result += direction[owner_action] * probability * value
    return float(result)


from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .continuous_factor_alignment import continuous_factor_alignment_candidates


class _Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(observation))


class _SelfFactor(nn.Module):
    def forward(
        self, owner_observation: torch.Tensor, owner_action: torch.Tensor
    ) -> torch.Tensor:
        return torch.sum(owner_action**2, dim=-1) + 0.1 * torch.sum(
            owner_observation, dim=-1
        )


class _PairFactor(nn.Module):
    def __init__(self, matrix: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("matrix", torch.as_tensor(matrix, dtype=torch.float32))

    def forward(
        self,
        owner_observation: torch.Tensor,
        donor_observation: torch.Tensor,
        owner_action: torch.Tensor,
        donor_profile: torch.Tensor,
    ) -> torch.Tensor:
        del owner_observation, donor_observation
        return torch.sum((owner_action @ self.matrix) * donor_profile, dim=-1)


def _flat_parameters(actor: nn.Module) -> np.ndarray:
    return np.concatenate(
        [parameter.detach().numpy().ravel() for parameter in actor.parameters()]
    )


def _shift(actor: nn.Module, direction: np.ndarray, amount: float) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in actor.parameters():
            size = parameter.numel()
            increment = torch.as_tensor(
                direction[offset : offset + size].reshape(parameter.shape),
                dtype=parameter.dtype,
            )
            parameter.add_(float(amount) * increment)
            offset += size


def test_continuous_alignment_matches_directional_finite_difference() -> None:
    rng = np.random.default_rng(95500)
    torch.manual_seed(95501)
    actor = _Actor().double()
    observation = rng.normal(size=3)
    direction = rng.normal(size=_flat_parameters(actor).size)
    direction /= np.linalg.norm(direction)
    donors = (1, 2)
    matrices = {donor: rng.normal(size=(4, 4)) for donor in donors}
    pair = {donor: _PairFactor(matrices[donor]).double() for donor in donors}
    donor_observation = {donor: rng.normal(size=2) for donor in donors}
    cached = {donor: rng.normal(size=4) for donor in donors}
    current = {donor: rng.normal(size=4) for donor in donors}
    result = continuous_factor_alignment_candidates(
        actor=actor,
        owner_observation=observation,
        reference_direction=direction,
        self_factor=_SelfFactor().double(),
        pair_factor_by_donor=pair,
        donor_observation_by_donor=donor_observation,
        cached_profile_by_donor=cached,
        current_profile_by_donor=current,
    )

    def objective(module: nn.Module, candidate: int | None) -> float:
        action = module(torch.as_tensor(observation).unsqueeze(0)).squeeze(0)
        value = torch.sum(action**2) + 0.1 * float(np.sum(observation))
        for donor in donors:
            profile = current[donor] if donor == candidate else cached[donor]
            value = value + torch.sum(
                (action @ torch.as_tensor(matrices[donor], dtype=action.dtype))
                * torch.as_tensor(profile, dtype=action.dtype)
            )
        return float(value.detach())

    epsilon = 1e-5
    for candidate in (None, *donors):
        plus = copy.deepcopy(actor)
        minus = copy.deepcopy(actor)
        _shift(plus, direction, epsilon)
        _shift(minus, direction, -epsilon)
        finite_difference = (
            objective(plus, candidate) - objective(minus, candidate)
        ) / (2.0 * epsilon)
        assert abs(result.alignment_by_candidate[candidate] - finite_difference) < 5e-8


def test_continuous_alignment_cost_is_linear_in_degree() -> None:
    rng = np.random.default_rng(95502)
    actor = _Actor()
    observation = np.zeros(3, dtype=np.float32)
    direction = rng.normal(size=_flat_parameters(actor).size)
    for degree in (1, 2, 4):
        donors = tuple(range(degree))
        result = continuous_factor_alignment_candidates(
            actor=actor,
            owner_observation=observation,
            reference_direction=direction,
            self_factor=_SelfFactor(),
            pair_factor_by_donor={
                donor: _PairFactor(np.eye(4)) for donor in donors
            },
            donor_observation_by_donor={
                donor: np.zeros(2) for donor in donors
            },
            cached_profile_by_donor={
                donor: np.zeros(4) for donor in donors
            },
            current_profile_by_donor={
                donor: np.ones(4) for donor in donors
            },
        )
        assert result.actor_reverse_calls == 4
        assert result.critic_action_gradient_calls == 1 + 2 * degree

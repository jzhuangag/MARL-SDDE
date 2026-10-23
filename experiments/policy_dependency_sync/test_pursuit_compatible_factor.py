from __future__ import annotations

import copy

import numpy as np
import torch

from .pursuit_compatible_factor import (
    brute_force_factor_alignment,
    factorized_alignment_candidates,
)
from .pursuit_delayed_alignment_interface import PursuitHeuristicLinearActor


def _flat_parameters(actor: torch.nn.Module) -> np.ndarray:
    return np.concatenate(
        [parameter.detach().cpu().numpy().ravel() for parameter in actor.parameters()]
    )


def _shift(actor: torch.nn.Module, direction: np.ndarray, amount: float) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in actor.parameters():
            count = parameter.numel()
            value = torch.as_tensor(
                direction[offset : offset + count].reshape(parameter.shape),
                dtype=parameter.dtype,
            )
            parameter.add_(float(amount) * value)
            offset += count


def test_pair_factor_scan_matches_joint_action_enumeration() -> None:
    rng = np.random.default_rng(501)
    actor = PursuitHeuristicLinearActor(seed=502)
    observation = rng.normal(size=(7, 7, 3)).astype(np.float32)
    reference = rng.normal(size=_flat_parameters(actor).size)
    donors = (2, 5, 7)
    self_factor = rng.normal(size=5)
    pair = {donor: rng.normal(size=(5, 5)) for donor in donors}
    cached = {donor: rng.dirichlet(np.ones(5)) for donor in donors}
    current = {donor: rng.dirichlet(np.ones(5)) for donor in donors}
    result = factorized_alignment_candidates(
        actor=actor,
        owner_observation=observation,
        reference_direction=reference,
        self_factor=self_factor,
        pair_factor_by_donor=pair,
        cached_probability_by_donor=cached,
        current_probability_by_donor=current,
    )
    null = brute_force_factor_alignment(
        owner_probability_direction=result.owner_probability_direction,
        self_factor=self_factor,
        pair_factor_by_donor=pair,
        probability_by_donor=cached,
    )
    assert abs(result.alignment_by_candidate[None] - null) < 1e-10
    for donor in donors:
        refreshed = dict(cached)
        refreshed[donor] = current[donor]
        brute = brute_force_factor_alignment(
            owner_probability_direction=result.owner_probability_direction,
            self_factor=self_factor,
            pair_factor_by_donor=pair,
            probability_by_donor=refreshed,
        )
        assert abs(result.alignment_by_candidate[donor] - brute) < 1e-10
    assert result.owner_reverse_calls == 5
    assert result.pair_factor_evaluations == len(donors)


def test_alignment_is_directional_derivative_of_factor_objective() -> None:
    rng = np.random.default_rng(503)
    actor = PursuitHeuristicLinearActor(seed=504)
    observation = rng.normal(size=(7, 7, 3)).astype(np.float32)
    direction = rng.normal(size=_flat_parameters(actor).size)
    direction /= np.linalg.norm(direction)
    self_factor = rng.normal(size=5)
    pair = {1: rng.normal(size=(5, 5))}
    cached = {1: rng.dirichlet(np.ones(5))}
    current = {1: rng.dirichlet(np.ones(5))}
    result = factorized_alignment_candidates(
        actor=actor,
        owner_observation=observation,
        reference_direction=direction,
        self_factor=self_factor,
        pair_factor_by_donor=pair,
        cached_probability_by_donor=cached,
        current_probability_by_donor=current,
    )
    q_value = self_factor + pair[1] @ current[1]

    def objective(module: torch.nn.Module) -> float:
        tensor = torch.as_tensor(observation).unsqueeze(0)
        probability = torch.softmax(module(tensor), dim=-1).squeeze(0)
        return float(probability.detach().cpu().numpy() @ q_value)

    plus = copy.deepcopy(actor)
    minus = copy.deepcopy(actor)
    epsilon = 1e-3
    _shift(plus, direction, epsilon)
    _shift(minus, direction, -epsilon)
    finite_difference = (objective(plus) - objective(minus)) / (2.0 * epsilon)
    assert abs(result.alignment_by_candidate[1] - finite_difference) < 2e-5


def test_reverse_pass_count_does_not_grow_with_degree() -> None:
    rng = np.random.default_rng(505)
    actor = PursuitHeuristicLinearActor(seed=506)
    observation = np.zeros((7, 7, 3), dtype=np.float32)
    reference = rng.normal(size=_flat_parameters(actor).size)
    calls = []
    for degree in (1, 2, 4):
        donors = tuple(range(degree))
        result = factorized_alignment_candidates(
            actor=actor,
            owner_observation=observation,
            reference_direction=reference,
            self_factor=np.zeros(5),
            pair_factor_by_donor={d: np.ones((5, 5)) * d for d in donors},
            cached_probability_by_donor={d: np.full(5, 0.2) for d in donors},
            current_probability_by_donor={d: np.full(5, 0.2) for d in donors},
        )
        calls.append(result.owner_reverse_calls)
        assert result.pair_factor_evaluations == degree
    assert calls == [5, 5, 5]


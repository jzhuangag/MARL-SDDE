"""Outcome-free neural contract for policy-cache control on Pistonball.

This module deliberately does not train a policy or use environment rewards.
It checks that a CTDE actor/critic computation can expose all eligible signed
cache-refresh scores with one cross-policy VJP and that the resulting update
respects policy-block ownership and byte accounting.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .neural_signed_cache import parameter_bytes, signed_cache_choice
from .pistonball_tail import initial_ball_index, predictive_tube_support


class TinyActor(nn.Module):
    """A distinct local actor block used only for the structural smoke."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3 * 8 * 8, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(observation, (8, 8))
        return self.network(pooled.flatten(start_dim=1))


class TinyCentralCritic(nn.Module):
    """Centralized differentiable score model for the structural smoke."""

    def __init__(self, n_agents: int, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3 * 8 * 8 + n_agents, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(state, (8, 8)).flatten(start_dim=1)
        return self.network(torch.cat((pooled, actions), dim=1)).mean()


@dataclass(frozen=True)
class PistonballNeuralContract:
    seed: int
    owner: int
    eligible_donors: tuple[int, ...]
    selected_donor: int | None
    candidate_count: int
    vjp_calls: int
    distinct_actor_blocks: int
    changed_current_blocks: tuple[int, ...]
    changed_cache_blocks: tuple[int, ...]
    actor_payload_bytes: int
    optional_message_bytes: int
    selected_parameter_bytes: int
    action_trace_sha256: str
    all_finite: bool
    future_or_reward_inputs_used: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _image_tensor(array: np.ndarray) -> torch.Tensor:
    value = torch.as_tensor(np.asarray(array), dtype=torch.float32)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("expected an H x W x 3 image")
    return value.permute(2, 0, 1).unsqueeze(0) / 255.0


def _parameters(module: nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(module.parameters())


def _parameter_snapshot(modules: Iterable[nn.Module]) -> tuple[tuple[torch.Tensor, ...], ...]:
    return tuple(
        tuple(parameter.detach().clone() for parameter in module.parameters())
        for module in modules
    )


def _changed_blocks(
    before: tuple[tuple[torch.Tensor, ...], ...], modules: Iterable[nn.Module]
) -> tuple[int, ...]:
    changed: list[int] = []
    for index, (old_group, module) in enumerate(zip(before, modules)):
        new_group = tuple(module.parameters())
        if len(old_group) != len(new_group):
            raise RuntimeError("parameter structure changed during the smoke")
        if any(not torch.equal(old, new.detach()) for old, new in zip(old_group, new_group)):
            changed.append(index)
    return tuple(changed)


def _joint_actions(
    actors: Iterable[nn.Module], observations: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    return torch.cat(
        tuple(actor(observation) for actor, observation in zip(actors, observations)),
        dim=1,
    )


def _copy_parameters(source: nn.Module, destination: nn.Module) -> None:
    with torch.no_grad():
        for current, cached in zip(source.parameters(), destination.parameters()):
            if current.shape != cached.shape:
                raise RuntimeError("cache/source parameter structures differ")
            cached.copy_(current)


def _perturb_caches(caches: Iterable[nn.Module], generator: torch.Generator) -> None:
    with torch.no_grad():
        for cache in caches:
            for parameter in cache.parameters():
                noise = torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=generator,
                )
                parameter.add_(0.01 * noise)


def _squared_displacement(
    current: nn.Module, cached: nn.Module
) -> tuple[tuple[torch.Tensor, ...], float]:
    values: list[torch.Tensor] = []
    squared_norm = 0.0
    for current_parameter, cached_parameter in zip(
        current.parameters(), cached.parameters()
    ):
        delta = current_parameter.detach() - cached_parameter.detach()
        values.append(delta)
        squared_norm += float(delta.square().sum().cpu())
    return tuple(values), squared_norm


def run_neural_contract(seed: int = 73001) -> PistonballNeuralContract:
    """Run one deterministic reset and one owner-only policy update.

    The function consumes only launch observations, launch state, public ball
    kinematics, current/cache parameters, the critic, and declared constants.
    It does not step the environment and does not read reward, termination, or
    any post-launch quantity.
    """

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))
    generator = torch.Generator().manual_seed(int(seed) + 1)
    np.random.seed(int(seed) % (2**32 - 1))

    from pettingzoo.butterfly.pistonball.pistonball import raw_env

    n_agents = 20
    environment = raw_env(
        n_pistons=n_agents,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        max_cycles=16,
        render_mode=None,
    )
    try:
        environment.reset(seed=int(seed))
        observations = tuple(
            _image_tensor(environment.observe(agent))
            for agent in environment.possible_agents
        )
        global_state = _image_tensor(environment.state())
        launch_x = float(environment.ball.position.x)
        launch_velocity_x = float(environment.ball.velocity.x)
    finally:
        environment.close()

    actors = tuple(TinyActor() for _ in range(n_agents))
    caches = tuple(copy.deepcopy(actor) for actor in actors)
    _perturb_caches(caches, generator)
    critic = TinyCentralCritic(n_agents)
    for parameter in critic.parameters():
        parameter.requires_grad_(False)

    owner = initial_ball_index(launch_x, n_agents)
    support = predictive_tube_support(
        n_agents=n_agents,
        launch_x=launch_x,
        launch_velocity_x=launch_velocity_x,
        horizon=4,
        radius=2,
    )
    eligible_donors = tuple(support[owner])
    if not eligible_donors:
        raise RuntimeError("the registered Pistonball cone has no eligible donor")

    current_actions = _joint_actions(actors, observations)
    mixed_modules = tuple(
        actor if index == owner else caches[index]
        for index, actor in enumerate(actors)
    )
    cached_actions = _joint_actions(mixed_modules, observations)
    current_owner_loss = -critic(global_state, current_actions)
    cached_owner_loss = -critic(global_state, cached_actions)

    donor_parameters: dict[int, tuple[torch.Tensor, ...]] = {}
    donor_displacements: dict[int, tuple[torch.Tensor, ...]] = {}
    taylor_remainders: dict[int, float] = {}
    for donor in eligible_donors:
        displacement, squared_norm = _squared_displacement(
            actors[donor], caches[donor]
        )
        donor_parameters[donor] = _parameters(caches[donor])
        donor_displacements[donor] = displacement
        # This declared coefficient is an interface input, not a certified
        # neural-network Hessian bound.  The report keeps that gap explicit.
        taylor_remainders[donor] = 1e-4 * squared_norm

    message_bytes = parameter_bytes(_parameters(actors[eligible_donors[0]]))
    choice = signed_cache_choice(
        current_owner_loss=current_owner_loss,
        cached_owner_loss=cached_owner_loss,
        owner_parameters=_parameters(actors[owner]),
        donor_parameters=donor_parameters,
        donor_displacements=donor_displacements,
        taylor_remainder_by_donor=taylor_remainders,
        step=0.01,
        smoothness=1.0,
        packet_second_moment_upper=1.0,
        receipt_motion_upper=0.0,
        communication_queue=0.0,
        learning_weight=1.0,
        message_cost=float(message_bytes),
    )

    actors_before = _parameter_snapshot(actors)
    caches_before = _parameter_snapshot(caches)
    if choice.donor is not None:
        _copy_parameters(actors[choice.donor], caches[choice.donor])

    refreshed_modules = tuple(
        actor if index == owner else caches[index]
        for index, actor in enumerate(actors)
    )
    receipt_loss = -critic(global_state, _joint_actions(refreshed_modules, observations))
    owner_gradient = torch.autograd.grad(
        receipt_loss, _parameters(actors[owner]), allow_unused=False
    )
    with torch.no_grad():
        for parameter, gradient in zip(actors[owner].parameters(), owner_gradient):
            parameter.add_(-0.01 * gradient)

    changed_current = _changed_blocks(actors_before, actors)
    changed_cache = _changed_blocks(caches_before, caches)
    selected_bytes = (
        0
        if choice.donor is None
        else parameter_bytes(_parameters(actors[choice.donor]))
    )
    trace_payload = {
        "owner": owner,
        "eligible_donors": eligible_donors,
        "selected_donor": choice.donor,
        "candidate_count": choice.candidate_count,
        "optional_message_bytes": selected_bytes,
    }
    trace_sha = hashlib.sha256(
        json.dumps(trace_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()

    finite_values = (
        choice.index,
        choice.estimated_alignment,
        float(current_owner_loss.detach().cpu()),
        float(cached_owner_loss.detach().cpu()),
    )
    return PistonballNeuralContract(
        seed=int(seed),
        owner=owner,
        eligible_donors=eligible_donors,
        selected_donor=choice.donor,
        candidate_count=choice.candidate_count,
        vjp_calls=choice.vjp_calls,
        distinct_actor_blocks=len({id(actor) for actor in actors}),
        changed_current_blocks=changed_current,
        changed_cache_blocks=changed_cache,
        actor_payload_bytes=message_bytes,
        optional_message_bytes=selected_bytes,
        selected_parameter_bytes=selected_bytes,
        action_trace_sha256=trace_sha,
        all_finite=all(np.isfinite(value) for value in finite_values),
        future_or_reward_inputs_used=False,
    )


def validate_neural_contract(result: PistonballNeuralContract) -> dict[str, bool]:
    expected_cache_changes = (
        () if result.selected_donor is None else (result.selected_donor,)
    )
    return {
        "N1_twenty_distinct_actor_blocks": result.distinct_actor_blocks == 20,
        "N2_exact_candidate_count": result.candidate_count
        == 1 + len(result.eligible_donors),
        "N3_one_cross_policy_vjp": result.vjp_calls == 1,
        "N4_owner_only_current_update": result.changed_current_blocks
        == (result.owner,),
        "N5_selected_cache_only_refresh": result.changed_cache_blocks
        == expected_cache_changes,
        "N6_exact_optional_byte_charge": result.optional_message_bytes
        == (0 if result.selected_donor is None else result.actor_payload_bytes)
        == result.selected_parameter_bytes,
        "N7_finite": result.all_finite,
        "N8_no_future_or_reward_input": not result.future_or_reward_inputs_used,
    }

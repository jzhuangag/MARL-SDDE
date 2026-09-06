"""Core state machine for asynchronous policy-cache CTDE on Pistonball.

The module separates three events that are often conflated in asynchronous
MARL implementations: a policy-cache refresh at rollout launch, trajectory
generation under the resulting mixed joint policy, and a delayed owner-only
gradient receipt.  It contains no experiment gates or outcome selection.
"""

from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .neural_signed_cache import NeuralCacheChoice, parameter_bytes, signed_cache_choice


class PistonActor(nn.Module):
    """A compact but fully distinct continuous Pistonball actor block."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 2, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(64, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        value = observation.float()
        if observation.dtype == torch.uint8:
            value = value / 255.0
        return torch.tanh(self.action_head(self.encoder(value)))


class PistonCentralCritic(nn.Module):
    """Centralized action-value model; actors remain local at execution."""

    def __init__(self, n_agents: int) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.value = nn.Sequential(
            nn.Linear(32 * 4 * 4 + int(n_agents), 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        value = state.float()
        if state.dtype == torch.uint8:
            value = value / 255.0
        encoded = self.state_encoder(value)
        if actions.ndim != 2 or actions.shape[0] != encoded.shape[0]:
            raise ValueError("joint actions must be a batch-by-agent matrix")
        return self.value(torch.cat((encoded, actions), dim=1))


@dataclass(frozen=True)
class CompressedTransition:
    observations: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    reward: float
    next_observations: np.ndarray
    next_state: np.ndarray
    done: bool


@dataclass(frozen=True)
class ReplayTensorBatch:
    observations: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    next_states: torch.Tensor
    done: torch.Tensor


class CompressedReplay:
    """Bounded transition replay with deterministic caller-owned sampling."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._rows: list[CompressedTransition] = []
        self._next = 0

    def add(self, transition: CompressedTransition) -> None:
        if len(self._rows) < self.capacity:
            self._rows.append(transition)
        else:
            self._rows[self._next] = transition
        self._next = (self._next + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> ReplayTensorBatch:
        if batch_size <= 0 or batch_size > len(self._rows):
            raise ValueError("batch_size must be positive and no larger than replay")
        indices = rng.choice(len(self._rows), size=int(batch_size), replace=False)
        selected = [self._rows[int(index)] for index in indices]
        return ReplayTensorBatch(
            observations=torch.as_tensor(
                np.stack([row.observations for row in selected]), device=device
            ),
            states=torch.as_tensor(
                np.stack([row.state for row in selected]), device=device
            ),
            actions=torch.as_tensor(
                np.stack([row.actions for row in selected]),
                dtype=torch.float32,
                device=device,
            ),
            rewards=torch.as_tensor(
                [row.reward for row in selected], dtype=torch.float32, device=device
            ),
            next_observations=torch.as_tensor(
                np.stack([row.next_observations for row in selected]), device=device
            ),
            next_states=torch.as_tensor(
                np.stack([row.next_state for row in selected]), device=device
            ),
            done=torch.as_tensor(
                [row.done for row in selected], dtype=torch.float32, device=device
            ),
        )

    def __len__(self) -> int:
        return len(self._rows)


def resize_uint8_images(
    images: np.ndarray, output_height: int, output_width: int
) -> np.ndarray:
    """Deterministically subsample one image or batch without float workspaces."""

    array = np.asarray(images)
    single = array.ndim == 3
    if single:
        array = array[None]
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("images must have shape HxWx3 or NxHxWx3")
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output dimensions must be positive")
    source_height, source_width = array.shape[1:3]
    row_index = np.minimum(
        ((np.arange(int(output_height)) + 0.5) * source_height / int(output_height)).astype(int),
        source_height - 1,
    )
    column_index = np.minimum(
        ((np.arange(int(output_width)) + 0.5) * source_width / int(output_width)).astype(int),
        source_width - 1,
    )
    resized = array[:, row_index, :, :][:, :, column_index, :]
    result = np.ascontiguousarray(resized.transpose(0, 3, 1, 2), dtype=np.uint8)
    return result[0] if single else result


def compress_parallel_snapshot(
    *,
    observations: Mapping[str, np.ndarray],
    possible_agents: Sequence[str],
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.stack([np.asarray(observations[agent]) for agent in possible_agents])
    return (
        resize_uint8_images(ordered, 64, 32),
        resize_uint8_images(np.asarray(state), 64, 64),
    )


def collect_parallel_segment(
    *,
    environment,
    owner: int,
    actors: Sequence[nn.Module],
    caches: "PolicyCacheBank",
    observations: Mapping[str, np.ndarray],
    horizon: int,
    exploration_std: float,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[list[CompressedTransition], Mapping[str, np.ndarray], bool]:
    """Collect one fully charged mixed-policy segment from a parallel env."""

    if horizon <= 0 or exploration_std < 0.0:
        raise ValueError("horizon must be positive and exploration nonnegative")
    transitions: list[CompressedTransition] = []
    current_observations = observations
    terminated = False
    for _ in range(int(horizon)):
        compressed_observations, compressed_state = compress_parallel_snapshot(
            observations=current_observations,
            possible_agents=environment.possible_agents,
            state=environment.state(),
        )
        tensor_observations = torch.as_tensor(
            compressed_observations[None], device=device
        )
        with torch.no_grad():
            means = joint_policy_actions(
                actors=actors,
                observations=tensor_observations,
                recipient=int(owner),
                caches=caches,
            )[0].cpu().numpy()
        actions = np.clip(
            means + float(exploration_std) * rng.normal(size=len(actors)),
            -1.0,
            1.0,
        ).astype(np.float32)
        action_dict = {
            agent: np.asarray([actions[index]], dtype=np.float32)
            for index, agent in enumerate(environment.possible_agents)
        }
        next_observations, rewards, terminations, truncations, _ = environment.step(
            action_dict
        )
        done = not environment.agents or all(
            bool(terminations[agent]) or bool(truncations[agent])
            for agent in environment.possible_agents
        )
        if done:
            next_compressed_observations = np.zeros_like(compressed_observations)
            next_compressed_state = np.zeros_like(compressed_state)
        else:
            (
                next_compressed_observations,
                next_compressed_state,
            ) = compress_parallel_snapshot(
                observations=next_observations,
                possible_agents=environment.possible_agents,
                state=environment.state(),
            )
        reward_values = [float(rewards[agent]) for agent in environment.possible_agents]
        if max(reward_values) - min(reward_values) > 1e-9:
            raise RuntimeError("Pistonball contract expects a shared team reward")
        transitions.append(
            CompressedTransition(
                observations=compressed_observations,
                state=compressed_state,
                actions=actions,
                reward=reward_values[0],
                next_observations=next_compressed_observations,
                next_state=next_compressed_state,
                done=done,
            )
        )
        current_observations = next_observations
        terminated = done
        if done:
            break
    return transitions, current_observations, terminated


def module_parameters(module: nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(module.parameters())


def copy_module_parameters(source: nn.Module, destination: nn.Module) -> None:
    with torch.no_grad():
        source_parameters = tuple(source.parameters())
        destination_parameters = tuple(destination.parameters())
        if len(source_parameters) != len(destination_parameters):
            raise ValueError("source and destination structures differ")
        for source_parameter, destination_parameter in zip(
            source_parameters, destination_parameters
        ):
            if source_parameter.shape != destination_parameter.shape:
                raise ValueError("source and destination shapes differ")
            destination_parameter.copy_(source_parameter)


class PolicyCacheBank:
    """Recipient-indexed teammate policy versions used only during training."""

    def __init__(self, actors: Sequence[nn.Module]) -> None:
        if len(actors) < 2:
            raise ValueError("at least two actors are required")
        self.n_agents = len(actors)
        self._caches: tuple[tuple[nn.Module, ...], ...] = tuple(
            tuple(copy.deepcopy(actor) for actor in actors)
            for _ in range(self.n_agents)
        )
        self.current_versions = [0 for _ in actors]
        self.cache_versions = [
            [0 for _ in actors] for _ in range(self.n_agents)
        ]

    def cached_actor(self, recipient: int, donor: int) -> nn.Module:
        return self._caches[int(recipient)][int(donor)]

    def rollout_actor(
        self, recipient: int, donor: int, current_actors: Sequence[nn.Module]
    ) -> nn.Module:
        if int(recipient) == int(donor):
            return current_actors[int(donor)]
        return self.cached_actor(recipient, donor)

    def refresh(
        self, recipient: int, donor: int, current_actors: Sequence[nn.Module]
    ) -> int:
        recipient = int(recipient)
        donor = int(donor)
        if recipient == donor:
            raise ValueError("owner self-policy is already fresh")
        copy_module_parameters(
            current_actors[donor], self.cached_actor(recipient, donor)
        )
        self.cache_versions[recipient][donor] = self.current_versions[donor]
        return parameter_bytes(module_parameters(current_actors[donor]))

    def mark_owner_update(self, owner: int) -> int:
        self.current_versions[int(owner)] += 1
        return self.current_versions[int(owner)]

    def age(self, recipient: int, donor: int) -> int:
        return int(
            self.current_versions[int(donor)]
            - self.cache_versions[int(recipient)][int(donor)]
        )

    def displacement_norm(self, recipient: int, donor: int, current_actor: nn.Module) -> float:
        squared = 0.0
        for current, cached in zip(
            current_actor.parameters(),
            self.cached_actor(recipient, donor).parameters(),
        ):
            squared += float((current.detach() - cached.detach()).square().sum().cpu())
        return float(np.sqrt(squared))

    def distinct_cache_modules(self) -> int:
        return len({id(module) for row in self._caches for module in row})


@dataclass(order=True)
class InFlightOwnerGradient:
    receipt_event: int
    launch_event: int
    owner: int
    gradients: tuple[torch.Tensor, ...] = field(compare=False)


class OwnerGradientQueue:
    """Deterministic receipt ordering for asynchronously returned gradients."""

    def __init__(self) -> None:
        self._heap: list[InFlightOwnerGradient] = []

    def launch(
        self,
        *,
        owner: int,
        launch_event: int,
        delay: int,
        gradients: Sequence[torch.Tensor],
    ) -> None:
        if delay < 0:
            raise ValueError("delay must be nonnegative")
        detached = tuple(gradient.detach().clone() for gradient in gradients)
        heapq.heappush(
            self._heap,
            InFlightOwnerGradient(
                receipt_event=int(launch_event) + int(delay),
                launch_event=int(launch_event),
                owner=int(owner),
                gradients=detached,
            ),
        )

    def apply_due(
        self,
        *,
        event: int,
        actors: Sequence[nn.Module],
        caches: PolicyCacheBank,
        step: float,
    ) -> tuple[InFlightOwnerGradient, ...]:
        if step < 0.0:
            raise ValueError("step must be nonnegative")
        applied: list[InFlightOwnerGradient] = []
        while self._heap and self._heap[0].receipt_event <= int(event):
            packet = heapq.heappop(self._heap)
            parameters = tuple(actors[packet.owner].parameters())
            if len(parameters) != len(packet.gradients):
                raise RuntimeError("packet gradient structure differs from owner")
            with torch.no_grad():
                for parameter, gradient in zip(parameters, packet.gradients):
                    if parameter.shape != gradient.shape:
                        raise RuntimeError("packet gradient shape differs from owner")
                    parameter.add_(-float(step) * gradient.to(parameter.device))
            caches.mark_owner_update(packet.owner)
            applied.append(packet)
        return tuple(applied)

    def drain_event(self) -> int | None:
        return None if not self._heap else max(packet.receipt_event for packet in self._heap)

    def __len__(self) -> int:
        return len(self._heap)


@dataclass(frozen=True)
class RefreshAction:
    donors: tuple[int, ...]
    refresh_units: int
    optional_policy_bytes: int


def apply_refresh_action(
    *,
    recipient: int,
    donors: Iterable[int],
    actors: Sequence[nn.Module],
    caches: PolicyCacheBank,
) -> RefreshAction:
    selected = tuple(sorted({int(donor) for donor in donors}))
    if int(recipient) in selected:
        raise ValueError("refresh set cannot contain the owner")
    charged = sum(caches.refresh(recipient, donor, actors) for donor in selected)
    return RefreshAction(
        donors=selected,
        refresh_units=len(selected),
        optional_policy_bytes=int(charged),
    )


def hard_budget_remaining(
    *, launches_after_action: int, budget_rate: float, spent_units: int
) -> int:
    if launches_after_action < 0 or budget_rate < 0.0 or spent_units < 0:
        raise ValueError("budget arguments must be nonnegative")
    allowance = int(np.floor(float(launches_after_action) * float(budget_rate) + 1e-12))
    return max(allowance - int(spent_units), 0)


def select_age_refresh(
    *,
    recipient: int,
    eligible_donors: Iterable[int],
    caches: PolicyCacheBank,
    maximum_edges: int,
) -> tuple[int, ...]:
    if maximum_edges < 0:
        raise ValueError("maximum_edges must be nonnegative")
    candidates = sorted(
        (
            int(donor)
            for donor in eligible_donors
            if int(donor) != int(recipient)
            and caches.age(recipient, int(donor)) > 0
        ),
        key=lambda donor: (-caches.age(recipient, donor), donor),
    )
    return tuple(candidates[: int(maximum_edges)])


def select_mismatch_refresh(
    *,
    recipient: int,
    eligible_donors: Iterable[int],
    actors: Sequence[nn.Module],
    caches: PolicyCacheBank,
    maximum_edges: int,
) -> tuple[int, ...]:
    if maximum_edges < 0:
        raise ValueError("maximum_edges must be nonnegative")
    candidates = sorted(
        (
            int(donor)
            for donor in eligible_donors
            if int(donor) != int(recipient)
            and caches.displacement_norm(recipient, int(donor), actors[int(donor)])
            > 0.0
        ),
        key=lambda donor: (
            -caches.displacement_norm(recipient, donor, actors[donor]),
            donor,
        ),
    )
    return tuple(candidates[: int(maximum_edges)])


def clone_parameter_groups(modules: Sequence[nn.Module]) -> tuple[tuple[torch.Tensor, ...], ...]:
    return tuple(
        tuple(parameter.detach().clone() for parameter in module.parameters())
        for module in modules
    )


def changed_parameter_groups(
    before: Sequence[Sequence[torch.Tensor]], modules: Sequence[nn.Module]
) -> tuple[int, ...]:
    if len(before) != len(modules):
        raise ValueError("snapshot and module counts differ")
    changed: list[int] = []
    for index, (old_group, module) in enumerate(zip(before, modules)):
        current_group = tuple(module.parameters())
        if len(old_group) != len(current_group):
            raise ValueError("parameter structure changed")
        if any(
            not torch.equal(old, current.detach())
            for old, current in zip(old_group, current_group)
        ):
            changed.append(index)
    return tuple(changed)


def joint_policy_actions(
    *,
    actors: Sequence[nn.Module],
    observations: torch.Tensor,
    recipient: int | None = None,
    caches: PolicyCacheBank | None = None,
) -> torch.Tensor:
    """Evaluate a current or recipient-cache joint deterministic policy."""

    if observations.ndim != 5 or observations.shape[1] != len(actors):
        raise ValueError("observations must be batch x agent x channel x H x W")
    if (recipient is None) != (caches is None):
        raise ValueError("recipient and caches must be supplied together")
    outputs: list[torch.Tensor] = []
    for donor, actor in enumerate(actors):
        selected = (
            actor
            if recipient is None
            else caches.rollout_actor(int(recipient), donor, actors)  # type: ignore[union-attr]
        )
        outputs.append(selected(observations[:, donor]))
    return torch.cat(outputs, dim=1)


def owner_critic_gradient(
    *,
    owner: int,
    actors: Sequence[nn.Module],
    caches: PolicyCacheBank,
    critic: nn.Module,
    observations: torch.Tensor,
    states: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Form one delayed deterministic-policy-gradient packet for its owner."""

    actions = joint_policy_actions(
        actors=actors,
        observations=observations,
        recipient=int(owner),
        caches=caches,
    )
    loss = -critic(states, actions).mean()
    return tuple(
        gradient.detach().clone()
        for gradient in torch.autograd.grad(
            loss, module_parameters(actors[int(owner)]), allow_unused=False
        )
    )


def signed_refresh_for_batch(
    *,
    owner: int,
    eligible_donors: Iterable[int],
    actors: Sequence[nn.Module],
    caches: PolicyCacheBank,
    critic: nn.Module,
    observations: torch.Tensor,
    states: torch.Tensor,
    step: float,
    communication_queue: float,
    learning_weight: float,
    smoothness: float,
    packet_second_moment_upper: float,
    receipt_motion_upper: float,
    taylor_coefficient: float,
    cache_debt_weight: float = 0.0,
    can_refresh: bool = True,
) -> NeuralCacheChoice:
    """Evaluate the signed Lyapunov cache action on completed replay only."""

    donors = tuple(
        sorted({int(donor) for donor in eligible_donors if int(donor) != int(owner)})
    )
    current_actions = joint_policy_actions(
        actors=actors, observations=observations
    )
    cached_actions = joint_policy_actions(
        actors=actors,
        observations=observations,
        recipient=int(owner),
        caches=caches,
    )
    current_loss = -critic(states, current_actions).mean()
    cached_loss = -critic(states, cached_actions).mean()
    if not can_refresh or not donors:
        owner_parameters = module_parameters(actors[int(owner)])
        current_gradient = torch.autograd.grad(
            current_loss, owner_parameters, allow_unused=False
        )
        cached_gradient = torch.autograd.grad(
            cached_loss, owner_parameters, allow_unused=False
        )
        alignment = float(
            sum(
                (current.detach() * cached.detach()).sum()
                for current, cached in zip(current_gradient, cached_gradient)
            ).cpu()
        )
        common = 0.5 * smoothness * step * step * packet_second_moment_upper
        common += step * receipt_motion_upper
        return NeuralCacheChoice(
            donor=None,
            index=float(learning_weight * (-step * alignment + common)),
            estimated_alignment=alignment,
            cache_reset_benefit=0.0,
            candidate_count=1,
            vjp_calls=0,
        )

    donor_parameters: dict[int, tuple[torch.Tensor, ...]] = {}
    donor_displacements: dict[int, tuple[torch.Tensor, ...]] = {}
    remainders: dict[int, float] = {}
    reset_benefits: dict[int, float] = {}
    for donor in donors:
        cache_parameters = module_parameters(caches.cached_actor(owner, donor))
        displacement = tuple(
            current.detach() - cached.detach()
            for current, cached in zip(actors[donor].parameters(), cache_parameters)
        )
        squared_norm = sum(float(delta.square().sum().cpu()) for delta in displacement)
        donor_parameters[donor] = cache_parameters
        donor_displacements[donor] = displacement
        remainders[donor] = 0.5 * float(taylor_coefficient) * squared_norm
        reset_benefits[donor] = 0.5 * float(cache_debt_weight) * squared_norm
    return signed_cache_choice(
        current_owner_loss=current_loss,
        cached_owner_loss=cached_loss,
        owner_parameters=module_parameters(actors[int(owner)]),
        donor_parameters=donor_parameters,
        donor_displacements=donor_displacements,
        taylor_remainder_by_donor=remainders,
        cache_reset_benefit_by_donor=reset_benefits,
        step=float(step),
        smoothness=float(smoothness),
        packet_second_moment_upper=float(packet_second_moment_upper),
        receipt_motion_upper=float(receipt_motion_upper),
        communication_queue=float(communication_queue),
        learning_weight=float(learning_weight),
        message_cost=1.0,
    )


def critic_td_loss(
    *,
    critic: nn.Module,
    target_critic: nn.Module,
    target_actors: Sequence[nn.Module],
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    next_observations: torch.Tensor,
    done: torch.Tensor,
    discount: float,
) -> torch.Tensor:
    """One centralized DDPG-style temporal-difference objective."""

    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must lie in [0,1]")
    prediction = critic(states, actions).squeeze(-1)
    with torch.no_grad():
        next_actions = joint_policy_actions(
            actors=target_actors, observations=next_observations
        )
        bootstrap = target_critic(next_states, next_actions).squeeze(-1)
        target = rewards + float(discount) * (1.0 - done) * bootstrap
    return F.mse_loss(prediction, target)


def polyak_update(
    source_modules: Sequence[nn.Module],
    target_modules: Sequence[nn.Module],
    coefficient: float,
) -> None:
    if len(source_modules) != len(target_modules):
        raise ValueError("source and target module counts differ")
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("coefficient must lie in [0,1]")
    with torch.no_grad():
        for source, target in zip(source_modules, target_modules):
            source_parameters = tuple(source.parameters())
            target_parameters = tuple(target.parameters())
            if len(source_parameters) != len(target_parameters):
                raise ValueError("source and target structures differ")
            for source_parameter, target_parameter in zip(
                source_parameters, target_parameters
            ):
                target_parameter.mul_(1.0 - float(coefficient))
                target_parameter.add_(float(coefficient) * source_parameter)

"""Causal delayed-alignment data interface for cached-policy Pursuit rollouts.

The module is an implementation qualification, not a learning benchmark.  A
launch context is built before the selected mixed-policy rollout is stepped.
The response is the normalized alignment between a reference gradient already
available at launch and the owner packet gradient produced by that rollout.
The response becomes available to the estimator only at its declared receipt
event.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .async_pistonball_ctde import PolicyCacheBank, apply_refresh_action
from .audit_pursuit_cache_semantics import PursuitInterfaceActor, _pursuit_model
from .audit_pursuit_local_factorization import bounded_local_graph
from .neural_signed_cache import parameter_bytes


GRADIENT_PROJECTION_DIMENSION = 8
INTERFACE_AGENTS = 8


FEATURE_NAMES = (
    "intercept",
    "nonnull",
    "reference_score_projection",
    "owner_evader_density",
    "owner_pursuer_density",
    "donor_evader_density",
    "distance",
    "cache_age",
    "policy_tv",
) + tuple(
    f"{role}_{channel}_{action}"
    for role in ("owner", "donor")
    for channel in ("wall", "pursuer", "evader")
    for action in ("left", "right", "up", "down", "stay")
) + tuple(
    f"reference_projection_{index}"
    for index in range(GRADIENT_PROJECTION_DIMENSION)
) + tuple(
    f"score_projection_{index}"
    for index in range(GRADIENT_PROJECTION_DIMENSION)
) + tuple(
    f"gradient_product_{index}"
    for index in range(GRADIENT_PROJECTION_DIMENSION)
) + tuple(
    f"donor_{kind}_probability_{action}"
    for kind in ("current", "cached", "difference")
    for action in range(5)
) + tuple(
    f"owner_identity_{index}" for index in range(INTERFACE_AGENTS)
) + tuple(
    f"donor_identity_{index}" for index in range(INTERFACE_AGENTS)
) + (
    "event_sine",
    "event_cosine",
    "reference_norm",
    "score_norm",
)


class PursuitHeuristicLinearActor(nn.Module):
    """Differentiable policy path approaching a public local chase rule.

    The path is used only for outcome-free interface/headroom qualification.
    It is not presented as a trained RL policy.  Distinct random initial actor
    blocks move predictably toward the same observation-local chase geometry.
    """

    def __init__(self, *, seed: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(7 * 7 * 3, 5))
        generator = torch.Generator().manual_seed(int(seed))
        layer = self.network[-1]
        if not isinstance(layer, nn.Linear):
            raise AssertionError("linear actor construction failed")
        with torch.no_grad():
            layer.weight.copy_(
                0.02
                * torch.randn(
                    layer.weight.shape,
                    dtype=layer.weight.dtype,
                    generator=generator,
                )
            )
            layer.bias.copy_(
                0.01
                * torch.randn(
                    layer.bias.shape,
                    dtype=layer.bias.dtype,
                    generator=generator,
                )
            )
        target_weight = torch.zeros_like(layer.weight)
        target_bias = torch.zeros_like(layer.bias)
        motions = ((-1, 0), (1, 0), (0, 1), (0, -1), (0, 0))
        for action, motion in enumerate(motions):
            for row in range(7):
                for column in range(7):
                    relative = (row - 3, column - 3)
                    distance = abs(relative[0] - motion[0]) + abs(
                        relative[1] - motion[1]
                    )
                    evader_index = (row * 7 + column) * 3 + 2
                    pursuer_index = (row * 7 + column) * 3 + 1
                    target_weight[action, evader_index] = math.exp(-0.65 * distance)
                    target_weight[action, pursuer_index] = -0.08 * math.exp(
                        -0.65 * distance
                    )
            if action == 4:
                target_bias[action] = -0.05
        self.register_buffer("target_weight", target_weight)
        self.register_buffer("target_bias", target_bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation.float())

    def advance(self, fraction: float) -> None:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("heuristic path fraction must lie in [0, 1]")
        layer = self.network[-1]
        if not isinstance(layer, nn.Linear):
            raise AssertionError("linear actor structure changed")
        with torch.no_grad():
            layer.weight.add_(fraction * (self.target_weight - layer.weight))
            layer.bias.add_(fraction * (self.target_bias - layer.bias))


class AlignmentRepresentation(nn.Module):
    """Small frozen feature encoder fitted only at predictable epoch breaks."""

    def __init__(self, input_dimension: int, embedding_dimension: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(int(input_dimension), 64),
            nn.SiLU(),
            nn.Linear(64, int(embedding_dimension)),
            nn.SiLU(),
        )
        self.head = nn.Linear(int(embedding_dimension), 1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(context)).squeeze(-1)


@dataclass(frozen=True)
class AlignmentLaunch:
    seed: int
    event: int
    owner: int
    donor: int | None
    packet_id: int
    receipt_event: int
    context: tuple[float, ...]
    feedback: float
    policy_bytes: int
    candidate_count: int
    reference_available: bool
    candidate_contexts: tuple[tuple[int | None, tuple[float, ...]], ...]
    candidate_feedback: tuple[tuple[int | None, float], ...]
    candidate_mean_feedback: tuple[tuple[int | None, float], ...]
    owner_probability_direction: tuple[float, ...]
    candidate_probability_difference: tuple[
        tuple[int | None, tuple[float, ...]], ...
    ]
    candidate_raw_mean_alignment: tuple[tuple[int | None, float], ...]
    candidate_factor_mean_alignment: tuple[tuple[int | None, float], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class _PendingGradient:
    receipt_event: int
    owner: int
    gradient: np.ndarray


def _flatten_gradients(
    loss: torch.Tensor, parameters: Sequence[torch.Tensor]
) -> np.ndarray:
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    return np.concatenate(
        [gradient.detach().cpu().numpy().ravel() for gradient in gradients]
    ).astype(np.float64, copy=False)


def _directional_projection(reference: np.ndarray, value: np.ndarray) -> float:
    norm = float(np.linalg.norm(reference))
    if norm <= 1e-15:
        return 0.0
    return float(np.dot(reference, value) / norm)


def _policy_probabilities(actor: nn.Module, observation: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    return torch.softmax(actor(tensor), dim=-1).squeeze(0)


def _categorical_from_uniform(probabilities: torch.Tensor, uniform: float) -> int:
    cumulative = torch.cumsum(probabilities.detach(), dim=0).cpu().numpy()
    return int(np.searchsorted(cumulative, min(float(uniform), 1.0 - 1e-12)))


def _profile_actions_for_candidate(
    *,
    candidate: int | None,
    owner: int,
    observations: Mapping[str, np.ndarray],
    possible_agents: Sequence[str],
    actors: Sequence[nn.Module],
    caches: PolicyCacheBank,
    uniforms: Mapping[str, float],
) -> dict[str, int]:
    actions: dict[str, int] = {}
    for donor, agent in enumerate(possible_agents):
        if agent not in observations:
            continue
        actor = (
            actors[donor]
            if donor == owner or donor == candidate
            else caches.cached_actor(owner, donor)
        )
        probability = _policy_probabilities(actor, observations[agent])
        actions[agent] = _categorical_from_uniform(probability, uniforms[agent])
    return actions


def _observation_density(observation: np.ndarray, channel: int) -> float:
    array = np.asarray(observation, dtype=float)
    if array.shape != (7, 7, 3):
        raise ValueError("Pursuit observation must have shape (7, 7, 3)")
    return float(np.mean(array[:, :, channel]))


def _spatial_features(observation: np.ndarray | None) -> tuple[float, ...]:
    if observation is None:
        return (0.0,) * 15
    array = np.asarray(observation, dtype=float)
    if array.shape != (7, 7, 3):
        raise ValueError("Pursuit observation must have shape (7, 7, 3)")
    motions = ((-1, 0), (1, 0), (0, 1), (0, -1), (0, 0))
    values: list[float] = []
    for channel in range(3):
        layer = array[:, :, channel]
        for motion in motions:
            kernel = np.empty((7, 7), dtype=float)
            for row in range(7):
                for column in range(7):
                    distance = abs((row - 3) - motion[0]) + abs(
                        (column - 3) - motion[1]
                    )
                    kernel[row, column] = math.exp(-0.65 * distance)
            values.append(float(np.sum(layer * kernel) / np.sum(kernel)))
    return tuple(values)


def _fixed_gradient_projection(
    value: np.ndarray | None,
    *,
    dimension: int = GRADIENT_PROJECTION_DIMENSION,
) -> tuple[float, ...]:
    if value is None:
        return (0.0,) * int(dimension)
    vector = np.asarray(value, dtype=float).ravel()
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-15:
        return (0.0,) * int(dimension)
    projection = _gradient_projection_matrix(vector.size, int(dimension))
    return tuple((projection @ (vector / norm)).tolist())


@lru_cache(maxsize=8)
def _gradient_projection_matrix(size: int, dimension: int) -> np.ndarray:
    generator = np.random.default_rng(71993 + 104729 * int(size))
    return generator.choice(
        (-1.0, 1.0),
        size=(int(dimension), int(size)),
    ) / math.sqrt(float(dimension))


def _launch_context(
    *,
    owner: int,
    donor: int | None,
    observations: Mapping[str, np.ndarray],
    possible_agents: Sequence[str],
    actors: Sequence[nn.Module],
    caches: PolicyCacheBank,
    positions: Sequence[Sequence[int]],
    score_projection: float,
    reference_gradient: np.ndarray | None,
    score_gradient: np.ndarray,
    event: int,
) -> tuple[float, ...]:
    owner_observation = observations[possible_agents[owner]]
    if donor is None:
        donor_observation: np.ndarray | None = None
        donor_evader_density = 0.0
        distance = 0.0
        cache_age = 0.0
        tv = 0.0
        current_probability = np.zeros(5, dtype=float)
        cached_probability = np.zeros(5, dtype=float)
    else:
        donor_observation = observations[possible_agents[donor]]
        donor_evader_density = _observation_density(donor_observation, 2)
        displacement = np.abs(
            np.asarray(positions[owner], dtype=float)
            - np.asarray(positions[donor], dtype=float)
        )
        distance = float(np.max(displacement) / 3.0)
        cache_age = float(
            min(20, caches.age(owner, donor)) / 20.0
        )
        current_probability = (
            _policy_probabilities(actors[donor], donor_observation)
            .detach()
            .cpu()
            .numpy()
        )
        cached_probability = (
            _policy_probabilities(
                caches.cached_actor(owner, donor), donor_observation
            )
            .detach()
            .cpu()
            .numpy()
        )
        tv = float(0.5 * np.sum(np.abs(current_probability - cached_probability)))
    reference_projection = _fixed_gradient_projection(reference_gradient)
    score_vector_projection = _fixed_gradient_projection(score_gradient)
    gradient_product = tuple(
        left * right
        for left, right in zip(reference_projection, score_vector_projection)
    )
    policy_features = tuple(current_probability) + tuple(cached_probability) + tuple(
        current_probability - cached_probability
    )
    owner_identity = tuple(float(index == owner) for index in range(INTERFACE_AGENTS))
    donor_identity = tuple(
        float(donor is not None and index == donor)
        for index in range(INTERFACE_AGENTS)
    )
    phase = 2.0 * math.pi * (int(event) % 16) / 16.0
    return (
        1.0,
        float(donor is not None),
        score_projection,
        _observation_density(owner_observation, 2),
        _observation_density(owner_observation, 1),
        donor_evader_density,
        distance,
        cache_age,
        tv,
    ) + _spatial_features(owner_observation) + _spatial_features(
        donor_observation
    ) + reference_projection + score_vector_projection + gradient_product + (
        policy_features
        + owner_identity
        + donor_identity
        + (
            math.sin(phase),
            math.cos(phase),
            0.0
            if reference_gradient is None
            else math.log1p(float(np.linalg.norm(reference_gradient))),
            math.log1p(float(np.linalg.norm(score_gradient))),
        )
    )


def _deliver_gradients(
    *,
    event: int,
    pending: list[_PendingGradient],
    references: list[np.ndarray | None],
    reference_decay: float,
) -> int:
    delivered = 0
    remaining: list[_PendingGradient] = []
    for packet in pending:
        if packet.receipt_event > event:
            remaining.append(packet)
            continue
        previous = references[packet.owner]
        references[packet.owner] = (
            packet.gradient.copy()
            if previous is None
            else reference_decay * previous + (1.0 - reference_decay) * packet.gradient
        )
        delivered += 1
    pending[:] = remaining
    return delivered


def _drift_actor(
    actor: nn.Module,
    *,
    event: int,
    owner: int,
    generator: torch.Generator,
    drift_scale: float,
) -> None:
    """Apply predictable reward-free version drift to exercise stale caches."""

    if isinstance(actor, PursuitHeuristicLinearActor):
        actor.advance(drift_scale)
        return
    scale = drift_scale * (1.0 + 0.25 * math.sin(event + owner))
    final = actor.network[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError("Pursuit interface actor requires a final linear layer")
    with torch.no_grad():
        noise = torch.randn(
            final.bias.shape,
            dtype=final.bias.dtype,
            generator=generator,
        )
        final.bias.add_(scale * noise)


def collect_alignment_launches(
    *,
    seeds: Sequence[int],
    cycles: int = 64,
    n_pursuers: int = 8,
    n_evaders: int = 30,
    observation_range: int = 7,
    max_neighbors: int = 4,
    maximum_delay: int = 3,
    actor_seed: int = 93401,
    action_seed: int = 93402,
    selection_seed: int = 93403,
    drift_seed: int = 93404,
    reference_decay: float = 0.8,
    feedback_scale: float = 0.5,
    drift_scale: float = 0.03,
    rollout_horizon: int = 4,
    discount: float = 0.99,
    audit_counterfactuals: bool = True,
    actor_mode: str = "heuristic_path",
    counterfactual_replicates: int = 1,
    counterfactual_seed: int = 93405,
) -> tuple[list[AlignmentLaunch], dict[str, float | int]]:
    """Collect launch/receipt records from actual Pursuit Markov transitions."""

    from pettingzoo.sisl import pursuit_v4

    if not 0.0 <= reference_decay < 1.0:
        raise ValueError("reference_decay must lie in [0, 1)")
    if maximum_delay < 0:
        raise ValueError("maximum_delay must be nonnegative")
    if feedback_scale <= 0.0:
        raise ValueError("feedback_scale must be positive")
    if drift_scale < 0.0:
        raise ValueError("drift_scale must be nonnegative")
    if rollout_horizon <= 0:
        raise ValueError("rollout_horizon must be positive")
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must lie in (0, 1]")
    if actor_mode not in {"heuristic_path", "random_mlp"}:
        raise ValueError("unknown actor mode")
    if n_pursuers != INTERFACE_AGENTS:
        raise ValueError("the registered interface uses exactly eight pursuers")
    if counterfactual_replicates <= 0:
        raise ValueError("counterfactual_replicates must be positive")
    started = time.perf_counter()
    records: list[AlignmentLaunch] = []
    packet_id = 0
    launches = 0
    delivered_before_launch = 0
    maximum_candidates = 0
    total_bytes = 0
    nonnull_launches = 0
    counterfactual_steps = 0
    maximum_selected_reward_error = 0.0

    for seed in seeds:
        torch.manual_seed(int(actor_seed))
        actors = tuple(
            PursuitHeuristicLinearActor(seed=actor_seed + agent)
            if actor_mode == "heuristic_path"
            else PursuitInterfaceActor()
            for agent in range(n_pursuers)
        )
        for actor in actors:
            actor.eval()
        caches = PolicyCacheBank(actors)
        references: list[np.ndarray | None] = [None] * n_pursuers
        pending: list[_PendingGradient] = []
        action_rng = np.random.default_rng(int(action_seed) + int(seed))
        selection_rng = np.random.default_rng(int(selection_seed) + int(seed))
        delay_rng = np.random.default_rng(int(selection_seed) + 10_000 + int(seed))
        drift_generator = torch.Generator().manual_seed(int(drift_seed) + int(seed))
        environment = pursuit_v4.parallel_env(
            n_pursuers=n_pursuers,
            n_evaders=n_evaders,
            max_cycles=cycles * rollout_horizon,
            obs_range=observation_range,
            n_catch=2,
            shared_reward=True,
        )
        observations, _ = environment.reset(seed=int(seed))
        try:
            for event in range(cycles):
                delivered_before_launch += _deliver_gradients(
                    event=event,
                    pending=pending,
                    references=references,
                    reference_decay=reference_decay,
                )
                model = _pursuit_model(environment)
                positions = tuple(
                    tuple(int(value) for value in model.pursuer_layer.get_position(i))
                    for i in range(n_pursuers)
                )
                graph, _ = bounded_local_graph(
                    positions,
                    radius=observation_range // 2,
                    max_neighbors=max_neighbors,
                )
                owner = event % n_pursuers
                donors = tuple(donor for edge_owner, donor in graph if edge_owner == owner)
                candidates: tuple[int | None, ...] = (None,) + donors
                maximum_candidates = max(maximum_candidates, len(candidates))

                common_uniform_path = tuple(
                    {
                        agent: float(action_rng.random())
                        for agent in environment.possible_agents
                    }
                    for _ in range(rollout_horizon)
                )
                common_uniforms = common_uniform_path[0]
                owner_probability = _policy_probabilities(
                    actors[owner], observations[environment.possible_agents[owner]]
                )
                owner_action = _categorical_from_uniform(
                    owner_probability,
                    common_uniforms[environment.possible_agents[owner]],
                )
                owner_log_probability = torch.log(owner_probability[owner_action] + 1e-12)
                score_gradient = _flatten_gradients(
                    -owner_log_probability,
                    tuple(actors[owner].parameters()),
                )
                launch_reference = references[owner]
                score_projection = (
                    0.0
                    if launch_reference is None
                    else _directional_projection(launch_reference, score_gradient)
                )
                contexts = {
                    donor: _launch_context(
                        owner=owner,
                        donor=donor,
                        observations=observations,
                        possible_agents=environment.possible_agents,
                        actors=actors,
                        caches=caches,
                        positions=positions,
                        score_projection=score_projection,
                        reference_gradient=launch_reference,
                        score_gradient=score_gradient,
                        event=event,
                    )
                    for donor in candidates
                }
                if launch_reference is None:
                    probability_direction = np.zeros(5, dtype=float)
                else:
                    from .pursuit_compatible_factor import (
                        owner_probability_direction,
                    )

                    normalized_reference = launch_reference / max(
                        float(np.linalg.norm(launch_reference)), 1e-15
                    )
                    _, probability_direction, _ = owner_probability_direction(
                        actor=actors[owner],
                        observation=observations[
                            environment.possible_agents[owner]
                        ],
                        reference_direction=normalized_reference,
                    )
                probability_differences: dict[int | None, tuple[float, ...]] = {
                    None: (0.0,) * 5
                }
                for donor in donors:
                    donor_observation = observations[
                        environment.possible_agents[donor]
                    ]
                    current_probability = (
                        _policy_probabilities(actors[donor], donor_observation)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    cached_probability = (
                        _policy_probabilities(
                            caches.cached_actor(owner, donor), donor_observation
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    probability_differences[donor] = tuple(
                        float(value)
                        for value in current_probability - cached_probability
                    )
                selected = candidates[int(selection_rng.integers(0, len(candidates)))]
                candidate_feedback: list[tuple[int | None, float]] = []
                candidate_mean_feedback: list[tuple[int | None, float]] = []
                candidate_raw_mean_alignment: list[tuple[int | None, float]] = []
                candidate_factor_mean_alignment: list[tuple[int | None, float]] = []
                selected_reward_audit: float | None = None
                selected_feedback_audit: float | None = None
                selected_raw_alignment_audit: float | None = None
                if audit_counterfactuals:
                    for candidate in candidates:
                        replicate_values: list[float] = []
                        replicate_raw_values: list[float] = []
                        replicate_factor_values: list[float] = []
                        for replicate in range(counterfactual_replicates):
                            cloned_model = copy.deepcopy(model)
                            if replicate == 0:
                                replicate_uniform_path = common_uniform_path
                            else:
                                branch_seed = (
                                    int(counterfactual_seed)
                                    + 1_000_003 * int(seed)
                                    + 10_007 * event
                                    + 101 * replicate
                                )
                                cloned_model._seed(branch_seed)
                                branch_rng = np.random.default_rng(branch_seed + 1)
                                replicate_uniform_path = tuple(
                                    {
                                        agent: float(branch_rng.random())
                                        for agent in environment.possible_agents
                                    }
                                    for _ in range(rollout_horizon)
                                )
                            try:
                                candidate_reward = 0.0
                                candidate_discounted_reward = 0.0
                                candidate_loss: torch.Tensor | None = None
                                candidate_initial_observation: np.ndarray | None = None
                                candidate_initial_action: int | None = None
                                for horizon_index in range(rollout_horizon):
                                    candidate_observations = {
                                        agent: np.swapaxes(
                                            cloned_model.safely_observe(action_agent),
                                            2,
                                            0,
                                        )
                                        for action_agent, agent in enumerate(
                                            environment.possible_agents
                                        )
                                    }
                                    candidate_actions = _profile_actions_for_candidate(
                                        candidate=candidate,
                                        owner=owner,
                                        observations=candidate_observations,
                                        possible_agents=environment.possible_agents,
                                        actors=actors,
                                        caches=caches,
                                        uniforms=replicate_uniform_path[horizon_index],
                                    )
                                    candidate_owner_probability = _policy_probabilities(
                                        actors[owner],
                                        candidate_observations[
                                            environment.possible_agents[owner]
                                        ],
                                    )
                                    candidate_owner_action = candidate_actions[
                                        environment.possible_agents[owner]
                                    ]
                                    candidate_owner_log_probability = torch.log(
                                        candidate_owner_probability[
                                            candidate_owner_action
                                        ]
                                        + 1e-12
                                    )
                                    if candidate_initial_observation is None:
                                        candidate_initial_observation = np.asarray(
                                            candidate_observations[
                                                environment.possible_agents[owner]
                                            ]
                                        ).copy()
                                        candidate_initial_action = int(
                                            candidate_owner_action
                                        )
                                    cycle_reward = 0.0
                                    for action_agent, agent in enumerate(
                                        environment.possible_agents
                                    ):
                                        cloned_model.step(
                                            candidate_actions[agent],
                                            action_agent,
                                            action_agent == n_pursuers - 1,
                                        )
                                        cycle_reward += float(
                                            cloned_model.latest_reward_state[owner]
                                        )
                                    candidate_reward += cycle_reward
                                    candidate_discounted_reward += (
                                        (discount**horizon_index) * cycle_reward
                                    )
                                    loss_term = (
                                        -(discount**horizon_index)
                                        * cycle_reward
                                        * candidate_owner_log_probability
                                    )
                                    candidate_loss = (
                                        loss_term
                                        if candidate_loss is None
                                        else candidate_loss + loss_term
                                    )
                                    if cloned_model.is_terminal:
                                        break
                            finally:
                                cloned_model.close()
                            if candidate_loss is None:
                                raise AssertionError("counterfactual packet is empty")
                            if candidate_initial_observation is None or (
                                candidate_initial_action is None
                            ):
                                raise AssertionError("counterfactual launch score is empty")
                            candidate_gradient = _flatten_gradients(
                                candidate_loss,
                                tuple(actors[owner].parameters()),
                            )
                            raw_alignment = (
                                0.0
                                if launch_reference is None
                                else _directional_projection(
                                    launch_reference,
                                    candidate_gradient,
                                )
                            )
                            candidate_value = float(
                                np.clip(
                                    raw_alignment / feedback_scale,
                                    -1.0,
                                    1.0,
                                )
                            )
                            replicate_values.append(candidate_value)
                            replicate_raw_values.append(raw_alignment)
                            factor_initial_probability = _policy_probabilities(
                                actors[owner], candidate_initial_observation
                            )
                            factor_initial_log_probability = torch.log(
                                factor_initial_probability[candidate_initial_action]
                                + 1e-12
                            )
                            factor_gradient = _flatten_gradients(
                                -candidate_discounted_reward
                                * factor_initial_log_probability,
                                tuple(actors[owner].parameters()),
                            )
                            replicate_factor_values.append(
                                0.0
                                if launch_reference is None
                                else _directional_projection(
                                    launch_reference,
                                    factor_gradient,
                                )
                            )
                            if replicate == 0 and candidate == selected:
                                selected_reward_audit = candidate_reward
                                selected_feedback_audit = candidate_value
                                selected_raw_alignment_audit = raw_alignment
                            counterfactual_steps += 1
                        candidate_feedback.append((candidate, replicate_values[0]))
                        candidate_mean_feedback.append(
                            (candidate, float(np.mean(replicate_values)))
                        )
                        candidate_raw_mean_alignment.append(
                            (candidate, float(np.mean(replicate_raw_values)))
                        )
                        candidate_factor_mean_alignment.append(
                            (candidate, float(np.mean(replicate_factor_values)))
                        )
                policy_bytes = 0
                if selected is not None:
                    action = apply_refresh_action(
                        recipient=owner,
                        donors=(selected,),
                        actors=actors,
                        caches=caches,
                    )
                    policy_bytes = int(action.optional_policy_bytes)
                    expected_bytes = parameter_bytes(tuple(actors[selected].parameters()))
                    if policy_bytes != expected_bytes:
                        raise AssertionError("refresh byte accounting is not exact")
                    nonnull_launches += 1
                    total_bytes += policy_bytes

                reward = 0.0
                packet_loss: torch.Tensor | None = None
                observations_after = observations
                terms: dict[str, bool] = {}
                truncations: dict[str, bool] = {}
                for horizon_index in range(rollout_horizon):
                    actions = _profile_actions_for_candidate(
                        candidate=None,
                        owner=owner,
                        observations=observations_after,
                        possible_agents=environment.possible_agents,
                        actors=actors,
                        caches=caches,
                        uniforms=common_uniform_path[horizon_index],
                    )
                    actual_owner_probability = _policy_probabilities(
                        actors[owner],
                        observations_after[environment.possible_agents[owner]],
                    )
                    actual_owner_action = actions[environment.possible_agents[owner]]
                    actual_owner_log_probability = torch.log(
                        actual_owner_probability[actual_owner_action] + 1e-12
                    )
                    observations_after, rewards, terms, truncations, _ = (
                        environment.step(actions)
                    )
                    cycle_reward = float(rewards[environment.possible_agents[owner]])
                    reward += cycle_reward
                    loss_term = (
                        -(discount**horizon_index)
                        * cycle_reward
                        * actual_owner_log_probability
                    )
                    packet_loss = (
                        loss_term if packet_loss is None else packet_loss + loss_term
                    )
                    if not environment.agents or any(terms.values()) or any(
                        truncations.values()
                    ):
                        break
                if selected_reward_audit is not None:
                    maximum_selected_reward_error = max(
                        maximum_selected_reward_error,
                        abs(reward - selected_reward_audit),
                    )
                if packet_loss is None:
                    raise AssertionError("selected packet is empty")
                packet_gradient = _flatten_gradients(
                    packet_loss,
                    tuple(actors[owner].parameters()),
                )
                reference = launch_reference
                feedback = (
                    0.0
                    if reference is None
                    else float(
                        np.clip(
                            _directional_projection(reference, packet_gradient)
                            / feedback_scale,
                            -1.0,
                            1.0,
                        )
                    )
                )
                raw_alignment = (
                    0.0
                    if reference is None
                    else _directional_projection(reference, packet_gradient)
                )
                if selected_feedback_audit is not None:
                    maximum_selected_reward_error = max(
                        maximum_selected_reward_error,
                        abs(feedback - selected_feedback_audit),
                    )
                if selected_raw_alignment_audit is not None:
                    maximum_selected_reward_error = max(
                        maximum_selected_reward_error,
                        abs(raw_alignment - selected_raw_alignment_audit),
                    )
                delay = int(delay_rng.integers(0, maximum_delay + 1))
                receipt_event = event + delay + 1
                pending.append(
                    _PendingGradient(
                        receipt_event=receipt_event,
                        owner=owner,
                        gradient=packet_gradient,
                    )
                )
                records.append(
                    AlignmentLaunch(
                        seed=int(seed),
                        event=event,
                        owner=owner,
                        donor=selected,
                        packet_id=packet_id,
                        receipt_event=receipt_event,
                        context=contexts[selected],
                        feedback=feedback,
                        policy_bytes=policy_bytes,
                        candidate_count=len(candidates),
                        reference_available=reference is not None,
                        candidate_contexts=tuple(contexts.items()),
                        candidate_feedback=tuple(candidate_feedback),
                        candidate_mean_feedback=tuple(candidate_mean_feedback),
                        owner_probability_direction=tuple(
                            float(value) for value in probability_direction
                        ),
                        candidate_probability_difference=tuple(
                            probability_differences.items()
                        ),
                        candidate_raw_mean_alignment=tuple(
                            candidate_raw_mean_alignment
                        ),
                        candidate_factor_mean_alignment=tuple(
                            candidate_factor_mean_alignment
                        ),
                    )
                )
                packet_id += 1
                launches += 1

                _drift_actor(
                    actors[owner],
                    event=event,
                    owner=owner,
                    generator=drift_generator,
                    drift_scale=drift_scale,
                )
                caches.mark_owner_update(owner)
                observations = observations_after
                if not environment.agents or any(terms.values()) or any(
                    truncations.values()
                ):
                    break
            delivered_before_launch += _deliver_gradients(
                event=cycles + maximum_delay + 1,
                pending=pending,
                references=references,
                reference_decay=reference_decay,
            )
            if pending:
                raise AssertionError("not all delayed packets were delivered")
        finally:
            environment.close()

    return records, {
        "launches": launches,
        "delivered_packets": delivered_before_launch,
        "maximum_candidates": maximum_candidates,
        "nonnull_launches": nonnull_launches,
        "charged_policy_bytes": total_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        "feature_dimension": len(FEATURE_NAMES),
        "feedback_scale": float(feedback_scale),
        "drift_scale": float(drift_scale),
        "rollout_horizon": int(rollout_horizon),
        "discount": float(discount),
        "actor_mode": actor_mode,
        "counterfactual_steps": counterfactual_steps,
        "counterfactual_replicates": int(counterfactual_replicates),
        "maximum_selected_reward_error": maximum_selected_reward_error,
    }


def fit_ridge(
    contexts: np.ndarray,
    responses: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a standardized finite-dimensional ridge head."""

    x = np.asarray(contexts, dtype=float)
    y = np.asarray(responses, dtype=float)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("invalid ridge design")
    location = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    location[0] = 0.0
    scale[scale < 1e-10] = 1.0
    scale[0] = 1.0
    standardized = (x - location) / scale
    penalty = float(ridge) * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    weight = np.linalg.solve(
        standardized.T @ standardized + penalty,
        standardized.T @ y,
    )
    return weight, location, scale


def predict_ridge(
    contexts: np.ndarray,
    *,
    weight: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    x = np.asarray(contexts, dtype=float)
    return ((x - location) / scale) @ weight


def train_alignment_representation(
    contexts: np.ndarray,
    responses: np.ndarray,
    *,
    seed: int,
    embedding_dimension: int = 16,
    epochs: int = 400,
    learning_rate: float = 0.005,
    weight_decay: float = 1e-4,
) -> tuple[AlignmentRepresentation, np.ndarray, np.ndarray, dict[str, float]]:
    """Fit a bounded-size nonlinear encoder on an earlier episode split."""

    x = np.asarray(contexts, dtype=np.float32)
    y = np.asarray(responses, dtype=np.float32)
    if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] == 0:
        raise ValueError("invalid representation design")
    if embedding_dimension <= 0 or epochs <= 0 or learning_rate <= 0.0:
        raise ValueError("invalid representation hyperparameters")
    location = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
    location[0] = 0.0
    scale[scale < 1e-6] = 1.0
    scale[0] = 1.0
    standardized = (x - location) / scale

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))
    model = AlignmentRepresentation(x.shape[1], embedding_dimension)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    tensor_x = torch.as_tensor(standardized, dtype=torch.float32)
    tensor_y = torch.as_tensor(y, dtype=torch.float32)
    initial_loss = float(
        torch.mean((model(tensor_x).detach() - tensor_y) ** 2).cpu()
    )
    for _ in range(int(epochs)):
        prediction = model(tensor_x)
        loss = torch.mean((prediction - tensor_y) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final_loss = float(
        torch.mean((model(tensor_x).detach() - tensor_y) ** 2).cpu()
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, location, scale, {
        "initial_mse": initial_loss,
        "final_mse": final_loss,
    }


def encode_alignment_contexts(
    model: AlignmentRepresentation,
    contexts: np.ndarray,
    *,
    location: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Return an intercept plus the frozen encoder output for a linear head."""

    x = np.asarray(contexts, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("contexts must be a matrix")
    standardized = (x - np.asarray(location, dtype=np.float32)) / np.asarray(
        scale, dtype=np.float32
    )
    with torch.no_grad():
        embedding = model.encoder(
            torch.as_tensor(standardized, dtype=torch.float32)
        ).cpu().numpy()
    return np.concatenate(
        (np.ones((embedding.shape[0], 1), dtype=float), embedding.astype(float)),
        axis=1,
    )


def episode_block_conformal_radius(
    residuals: np.ndarray,
    seeds: np.ndarray,
    *,
    miscoverage: float,
) -> float:
    """Use exchangeable episode maxima, never individual Markov steps."""

    if not 0.0 < miscoverage < 1.0:
        raise ValueError("miscoverage must lie in (0, 1)")
    maxima = np.asarray(
        [
            np.max(np.abs(residuals[seeds == seed]))
            for seed in np.unique(seeds)
        ],
        dtype=float,
    )
    order = int(math.ceil((maxima.size + 1) * (1.0 - miscoverage)))
    order = min(max(1, order), maxima.size)
    return float(np.partition(maxima, order - 1)[order - 1])

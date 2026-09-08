"""Privileged CPU gate for Multiwalker Lyapunov-alignment headroom.

This module is a development diagnostic, not a deployable controller.  It
uses exact simulator replay and finite-difference deterministic policy
gradients to ask whether cache refreshes can change the packet direction in a
way that a joint edge/packet-weight Lyapunov controller could exploit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Hashable, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

from .audit_multiwalker_cache_contract import (
    MultiwalkerContractActor,
    _actor_action,
    chain_neighbors,
)
from .multiwalker_oracle_headroom_dev import _drift_actor, _environment


POLICIES = (
    "myopic_alignment",
    "action_gap",
    "parameter_gap",
    "age",
    "round_robin",
    "random",
    "fixed_left",
    "fixed_right",
    "no_refresh",
)


@dataclass(frozen=True)
class AlignmentScore:
    alignment: float
    packet_norm_squared: float
    lyapunov_decrease: float
    packet_weight: float
    diagnostic_transitions: int


@dataclass(frozen=True)
class AlignmentLocalOption:
    owner: int
    launch_events: tuple[int, ...]
    cost_sequence: tuple[int, ...]
    action_sequence: tuple[tuple[int, ...], ...]
    total_lyapunov_decrease: float


@dataclass(frozen=True)
class AlignmentOracleRow:
    seed: int
    drift_scale: float
    budget_rate: float
    exact_oracle_decrease: float
    exact_oracle_spent_edges: int
    exact_oracle_nonzero_weights: int
    strong_online_policy: str
    strong_online_decrease: float
    no_refresh_decrease: float
    ideal_reference_decrease: float
    oracle_gain_over_no_refresh: float
    oracle_headroom_over_strong: float
    oracle_recovery_gap: float
    normalized_absolute_headroom: float
    optimizer_success: bool
    optimizer_status: int
    optimizer_mip_gap: float
    selected_owner_count: int
    maximum_prefix_excess: int
    charged_diagnostic_transitions: int
    public_trace_transitions: int
    reference_replay_error: float
    minimum_owner_launches: int


@dataclass(frozen=True)
class _Trace:
    prefix_actions: tuple[dict[str, np.ndarray], ...]
    observations: tuple[dict[str, np.ndarray], ...]
    actor_snapshots: tuple[tuple[MultiwalkerContractActor, ...], ...]
    initial_actors: tuple[MultiwalkerContractActor, ...]
    owner_schedule: tuple[int, ...]


def randomized_epoch_owner_schedule(
    *, seed: int, events: int, walkers: int, schedule_seed: int
) -> tuple[int, ...]:
    """Randomize each complete owner epoch while preserving full coverage."""

    if events <= 0 or walkers <= 1:
        raise ValueError("invalid owner schedule dimensions")
    rng = np.random.default_rng(int(schedule_seed) + 1009 * int(seed))
    schedule: list[int] = []
    while len(schedule) < events:
        schedule.extend(int(value) for value in rng.permutation(walkers))
    return tuple(schedule[:events])


def _public_trace(
    *,
    seed: int,
    drift_scale: float,
    walkers: int,
    events: int,
    horizon: int,
    actor_seed: int,
    schedule_seed: int,
) -> _Trace:
    schedule = randomized_epoch_owner_schedule(
        seed=seed, events=events, walkers=walkers, schedule_seed=schedule_seed
    )
    actors = [
        MultiwalkerContractActor(seed=actor_seed + index)
        for index in range(walkers)
    ]
    initial = tuple(copy.deepcopy(actor) for actor in actors)
    prefix: list[dict[str, np.ndarray]] = []
    observations_by_event: list[dict[str, np.ndarray]] = []
    snapshots: list[tuple[MultiwalkerContractActor, ...]] = []
    environment = _environment(
        walkers=walkers, maximum_cycles=events + horizon + 2
    )
    observations, _ = environment.reset(seed=int(seed))
    try:
        for event, owner in enumerate(schedule):
            if not environment.agents:
                break
            observations_by_event.append(
                {
                    agent: np.asarray(value, dtype=np.float32).copy()
                    for agent, value in observations.items()
                }
            )
            snapshots.append(tuple(copy.deepcopy(actor) for actor in actors))
            actions = {
                agent: _actor_action(actors[index], observations[agent])
                for index, agent in enumerate(environment.possible_agents)
            }
            prefix.append(actions)
            observations, _, _, _, _ = environment.step(
                {agent: actions[agent] for agent in environment.agents}
            )
            _drift_actor(actors[owner], event, drift_scale)
    finally:
        environment.close()
    if len(prefix) != events or len(snapshots) != events:
        raise RuntimeError("public Multiwalker trace terminated before the gate horizon")
    return _Trace(
        prefix_actions=tuple(prefix),
        observations=tuple(observations_by_event),
        actor_snapshots=tuple(snapshots),
        initial_actors=initial,
        owner_schedule=schedule,
    )


def _branch_return(
    *,
    seed: int,
    prefix_actions: Sequence[Mapping[str, np.ndarray]],
    actors: Sequence[MultiwalkerContractActor],
    recipient_cache: Sequence[MultiwalkerContractActor],
    owner: int,
    refresh_donors: Sequence[int],
    forced_owner_action: np.ndarray,
    horizon: int,
    discount: float,
    walkers: int,
    maximum_cycles: int,
) -> tuple[float, int]:
    environment = _environment(walkers=walkers, maximum_cycles=maximum_cycles)
    observations, _ = environment.reset(seed=int(seed))
    transitions = 0
    for actions in prefix_actions:
        if not environment.agents:
            break
        observations, _, _, _, _ = environment.step(
            {agent: actions[agent] for agent in environment.agents}
        )
        transitions += 1
    refreshed = frozenset(int(donor) for donor in refresh_donors)
    total = 0.0
    try:
        for step in range(horizon):
            if not environment.agents:
                break
            actions: dict[str, np.ndarray] = {}
            for donor, agent in enumerate(environment.possible_agents):
                if step == 0 and donor == owner:
                    action = np.asarray(forced_owner_action, dtype=np.float32)
                else:
                    actor = (
                        actors[donor]
                        if donor == owner or donor in refreshed
                        else recipient_cache[donor]
                    )
                    action = _actor_action(actor, observations[agent])
                actions[agent] = np.clip(action, -1.0, 1.0).astype(np.float32)
            observations, rewards, terms, truncations, _ = environment.step(
                {agent: actions[agent] for agent in environment.agents}
            )
            transitions += 1
            reward = float(np.mean(tuple(rewards.values()))) if rewards else 0.0
            total += discount**step * reward
            if not environment.agents or any(terms.values()) or any(truncations.values()):
                break
    finally:
        environment.close()
    return float(total), int(transitions)


def _actor_action_jacobian(
    actor: MultiwalkerContractActor, observation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    parameters = tuple(actor.parameters())
    tensor = torch.as_tensor(observation, dtype=parameters[0].dtype).unsqueeze(0)
    action = actor(tensor).squeeze(0)
    rows: list[np.ndarray] = []
    for coordinate in range(action.numel()):
        gradient = torch.autograd.grad(
            action[coordinate],
            parameters,
            retain_graph=coordinate + 1 < action.numel(),
            create_graph=False,
        )
        rows.append(
            np.concatenate(
                [value.detach().cpu().numpy().ravel() for value in gradient]
            ).astype(float, copy=False)
        )
    return action.detach().cpu().numpy().astype(float), np.stack(rows, axis=1)


def deterministic_policy_gradient_packet(
    *,
    seed: int,
    prefix_actions: Sequence[Mapping[str, np.ndarray]],
    actors: Sequence[MultiwalkerContractActor],
    recipient_cache: Sequence[MultiwalkerContractActor],
    owner: int,
    refresh_donors: Sequence[int],
    owner_observation: np.ndarray,
    horizon: int,
    discount: float,
    action_difference: float,
    walkers: int,
    maximum_cycles: int,
) -> tuple[np.ndarray, int]:
    """Central-difference deterministic actor-loss gradient packet."""

    if action_difference <= 0.0 or horizon <= 0:
        raise ValueError("invalid finite-difference packet configuration")
    action, jacobian = _actor_action_jacobian(actors[owner], owner_observation)
    q_gradient = np.empty_like(action, dtype=float)
    transitions = 0
    for coordinate in range(action.size):
        plus = action.copy()
        minus = action.copy()
        plus[coordinate] += action_difference
        minus[coordinate] -= action_difference
        plus = np.clip(plus, -1.0, 1.0)
        minus = np.clip(minus, -1.0, 1.0)
        plus_value, plus_transitions = _branch_return(
            seed=seed,
            prefix_actions=prefix_actions,
            actors=actors,
            recipient_cache=recipient_cache,
            owner=owner,
            refresh_donors=refresh_donors,
            forced_owner_action=plus,
            horizon=horizon,
            discount=discount,
            walkers=walkers,
            maximum_cycles=maximum_cycles,
        )
        minus_value, minus_transitions = _branch_return(
            seed=seed,
            prefix_actions=prefix_actions,
            actors=actors,
            recipient_cache=recipient_cache,
            owner=owner,
            refresh_donors=refresh_donors,
            forced_owner_action=minus,
            horizon=horizon,
            discount=discount,
            walkers=walkers,
            maximum_cycles=maximum_cycles,
        )
        denominator = float(plus[coordinate] - minus[coordinate])
        q_gradient[coordinate] = (
            0.0 if denominator <= 1e-12 else (plus_value - minus_value) / denominator
        )
        transitions += plus_transitions + minus_transitions
    # F = -J, hence the actor-loss gradient is -J_theta(mu) grad_a Q.
    return -(jacobian @ q_gradient), int(transitions)


def joint_weight_decrease(
    *,
    reference_gradient: np.ndarray,
    packet_gradient: np.ndarray,
    learning_smoothness: float,
    maximum_packet_weight: float,
) -> AlignmentScore:
    """Optimize the scalar packet weight in the normalized smoothness bound."""

    reference = np.asarray(reference_gradient, dtype=float).ravel()
    packet = np.asarray(packet_gradient, dtype=float).ravel()
    if reference.shape != packet.shape or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(packet)):
        raise ValueError("gradient packets must be equally shaped and finite")
    if learning_smoothness <= 0.0 or maximum_packet_weight < 0.0:
        raise ValueError("invalid Lyapunov weight domain")
    alignment = float(reference @ packet)
    norm_squared = float(packet @ packet)
    if norm_squared <= 1e-20 or alignment <= 0.0:
        weight = 0.0
    else:
        weight = min(
            float(maximum_packet_weight),
            alignment / (float(learning_smoothness) * norm_squared),
        )
    decrease = (
        weight * alignment
        - 0.5 * float(learning_smoothness) * weight**2 * norm_squared
    )
    return AlignmentScore(
        alignment=alignment,
        packet_norm_squared=norm_squared,
        lyapunov_decrease=float(max(decrease, 0.0)),
        packet_weight=float(weight),
        diagnostic_transitions=0,
    )


def _cache_for_state(
    *,
    owner: int,
    neighbors: tuple[int, ...],
    state: tuple[int, ...],
    launch_events: Sequence[int],
    global_event: int,
    actor_snapshots: Sequence[Sequence[MultiwalkerContractActor]],
    initial_actors: Sequence[MultiwalkerContractActor],
) -> list[MultiwalkerContractActor]:
    # Non-neighbor actors are current by construction: the gate tests the
    # declared sparse policy-dependency graph rather than charging unreachable
    # long-range cache errors to every local decision.
    cache = list(actor_snapshots[global_event])
    for position, last_local_refresh in enumerate(state):
        donor = neighbors[position]
        cache[donor] = (
            initial_actors[donor]
            if last_local_refresh < 0
            else actor_snapshots[launch_events[last_local_refresh]][donor]
        )
    cache[owner] = actor_snapshots[global_event][owner]
    return cache


def _next_state(
    state: tuple[int, ...],
    neighbors: tuple[int, ...],
    candidate: tuple[int, ...],
    local_index: int,
) -> tuple[int, ...]:
    value = list(state)
    for donor in candidate:
        value[neighbors.index(donor)] = local_index
    return tuple(value)


def build_alignment_score_table(
    *,
    trace: _Trace,
    seed: int,
    horizon: int,
    discount: float,
    action_difference: float,
    learning_smoothness: float,
    maximum_packet_weight: float,
    walkers: int,
) -> tuple[
    dict[tuple[int, int, tuple[int, ...], tuple[int, ...]], AlignmentScore],
    dict[int, float],
    int,
    float,
]:
    maximum_cycles = len(trace.prefix_actions) + horizon + 2
    reference_by_event: dict[int, np.ndarray] = {}
    ideal_by_event: dict[int, float] = {}
    diagnostic_transitions = len(trace.prefix_actions)
    for event, owner in enumerate(trace.owner_schedule):
        actors = trace.actor_snapshots[event]
        current_cache = list(actors)
        owner_name = f"walker_{owner}"
        reference, charged = deterministic_policy_gradient_packet(
            seed=seed,
            prefix_actions=trace.prefix_actions[:event],
            actors=actors,
            recipient_cache=current_cache,
            owner=owner,
            refresh_donors=(),
            owner_observation=trace.observations[event][owner_name],
            horizon=horizon,
            discount=discount,
            action_difference=action_difference,
            walkers=walkers,
            maximum_cycles=maximum_cycles,
        )
        reference_by_event[event] = reference
        ideal = joint_weight_decrease(
            reference_gradient=reference,
            packet_gradient=reference,
            learning_smoothness=learning_smoothness,
            maximum_packet_weight=maximum_packet_weight,
        )
        ideal_by_event[event] = ideal.lyapunov_decrease
        diagnostic_transitions += charged

    # A single repeated all-current packet is a deterministic replay sentinel.
    sentinel_event = 0
    sentinel_owner = trace.owner_schedule[sentinel_event]
    sentinel_repeat, charged = deterministic_policy_gradient_packet(
        seed=seed,
        prefix_actions=(),
        actors=trace.actor_snapshots[sentinel_event],
        recipient_cache=list(trace.actor_snapshots[sentinel_event]),
        owner=sentinel_owner,
        refresh_donors=(),
        owner_observation=trace.observations[sentinel_event][f"walker_{sentinel_owner}"],
        horizon=horizon,
        discount=discount,
        action_difference=action_difference,
        walkers=walkers,
        maximum_cycles=maximum_cycles,
    )
    diagnostic_transitions += charged
    reference_replay_error = float(
        np.max(np.abs(sentinel_repeat - reference_by_event[sentinel_event]))
    )

    table: dict[
        tuple[int, int, tuple[int, ...], tuple[int, ...]], AlignmentScore
    ] = {}
    for owner in range(walkers):
        launch_events = tuple(
            event
            for event, selected_owner in enumerate(trace.owner_schedule)
            if selected_owner == owner
        )
        neighbors = chain_neighbors(owner, walkers)
        states: set[tuple[int, ...]] = {(-1,) * len(neighbors)}
        for local_index, event in enumerate(launch_events):
            next_states: set[tuple[int, ...]] = set()
            actors = trace.actor_snapshots[event]
            for state in sorted(states):
                cache = _cache_for_state(
                    owner=owner,
                    neighbors=neighbors,
                    state=state,
                    launch_events=launch_events,
                    global_event=event,
                    actor_snapshots=trace.actor_snapshots,
                    initial_actors=trace.initial_actors,
                )
                candidates = ((),) + tuple((donor,) for donor in neighbors)
                for candidate in candidates:
                    packet, charged = deterministic_policy_gradient_packet(
                        seed=seed,
                        prefix_actions=trace.prefix_actions[:event],
                        actors=actors,
                        recipient_cache=cache,
                        owner=owner,
                        refresh_donors=candidate,
                        owner_observation=trace.observations[event][f"walker_{owner}"],
                        horizon=horizon,
                        discount=discount,
                        action_difference=action_difference,
                        walkers=walkers,
                        maximum_cycles=maximum_cycles,
                    )
                    score = joint_weight_decrease(
                        reference_gradient=reference_by_event[event],
                        packet_gradient=packet,
                        learning_smoothness=learning_smoothness,
                        maximum_packet_weight=maximum_packet_weight,
                    )
                    table[(owner, local_index, state, candidate)] = AlignmentScore(
                        alignment=score.alignment,
                        packet_norm_squared=score.packet_norm_squared,
                        lyapunov_decrease=score.lyapunov_decrease,
                        packet_weight=score.packet_weight,
                        diagnostic_transitions=charged,
                    )
                    diagnostic_transitions += charged
                    next_states.add(
                        _next_state(state, neighbors, candidate, local_index)
                    )
            states = next_states
    return table, ideal_by_event, int(diagnostic_transitions), reference_replay_error


def local_alignment_options(
    *,
    owner: int,
    owner_schedule: Sequence[int],
    walkers: int,
    score_table: Mapping[
        tuple[int, int, tuple[int, ...], tuple[int, ...]], AlignmentScore
    ],
) -> list[AlignmentLocalOption]:
    neighbors = chain_neighbors(owner, walkers)
    launch_events = tuple(
        event for event, selected in enumerate(owner_schedule) if selected == owner
    )
    actions = ((),) + tuple((donor,) for donor in neighbors)
    dynamic: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        tuple[float, tuple[tuple[int, ...], ...]],
    ] = {(((-1,) * len(neighbors)), ()): (0.0, ())}
    for local_index, _ in enumerate(launch_events):
        updated: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[float, tuple[tuple[int, ...], ...]],
        ] = {}
        for (state, costs), (total, sequence) in dynamic.items():
            for action in actions:
                score = score_table[(owner, local_index, state, action)]
                next_state = _next_state(state, neighbors, action, local_index)
                next_costs = costs + (len(action),)
                key = (next_state, next_costs)
                proposal = (total + score.lyapunov_decrease, sequence + (action,))
                incumbent = updated.get(key)
                if incumbent is None or proposal[0] > incumbent[0]:
                    updated[key] = proposal
        dynamic = updated
    by_cost: dict[tuple[int, ...], AlignmentLocalOption] = {}
    for (_, costs), (total, actions) in dynamic.items():
        option = AlignmentLocalOption(
            owner=owner,
            launch_events=launch_events,
            cost_sequence=costs,
            action_sequence=actions,
            total_lyapunov_decrease=float(total),
        )
        incumbent = by_cost.get(costs)
        if incumbent is None or option.total_lyapunov_decrease > incumbent.total_lyapunov_decrease:
            by_cost[costs] = option
    return [by_cost[key] for key in sorted(by_cost)]


def solve_prefix_alignment_oracle(
    *,
    options_by_owner: Sequence[Sequence[AlignmentLocalOption]],
    budget_rate: float,
    events: int,
) -> tuple[float, int, bool, int, float, tuple[AlignmentLocalOption, ...]]:
    flat = [option for options in options_by_owner for option in options]
    offsets = np.cumsum([0] + [len(options) for options in options_by_owner])
    rows = len(options_by_owner) + events
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower = np.full(rows, -np.inf)
    upper = np.empty(rows)
    for owner in range(len(options_by_owner)):
        lower[owner] = upper[owner] = 1.0
        for column in range(offsets[owner], offsets[owner + 1]):
            row_indices.append(owner)
            column_indices.append(column)
            values.append(1.0)
    for event in range(events):
        row = len(options_by_owner) + event
        upper[row] = math.floor(float(budget_rate) * (event + 1))
        for column, option in enumerate(flat):
            cumulative = sum(
                cost
                for launch, cost in zip(option.launch_events, option.cost_sequence)
                if launch <= event
            )
            if cumulative:
                row_indices.append(row)
                column_indices.append(column)
                values.append(float(cumulative))
    constraint = LinearConstraint(
        csc_matrix((values, (row_indices, column_indices)), shape=(rows, len(flat))),
        lower,
        upper,
    )
    result = milp(
        c=-np.asarray([option.total_lyapunov_decrease for option in flat]),
        integrality=np.ones(len(flat), dtype=np.int8),
        bounds=Bounds(np.zeros(len(flat)), np.ones(len(flat))),
        constraints=constraint,
        options={"presolve": True, "time_limit": 300.0, "mip_rel_gap": 0.0},
    )
    if result.x is None:
        return -math.inf, 0, False, int(result.status), math.inf, ()
    selected = tuple(flat[index] for index in np.flatnonzero(result.x > 0.5))
    gap = getattr(result, "mip_gap", None)
    return (
        float(sum(option.total_lyapunov_decrease for option in selected)),
        int(sum(sum(option.cost_sequence) for option in selected)),
        bool(result.success),
        int(result.status),
        float(gap) if gap is not None else math.inf,
        selected,
    )


def _parameter_gap(current: torch.nn.Module, cached: torch.nn.Module) -> float:
    return float(
        math.sqrt(
            sum(
                float(torch.sum((left.detach() - right.detach()) ** 2))
                for left, right in zip(current.parameters(), cached.parameters())
            )
        )
    )


def _baseline_total(
    *,
    policy: str,
    budget_rate: float,
    trace: _Trace,
    score_table: Mapping[
        tuple[int, int, tuple[int, ...], tuple[int, ...]], AlignmentScore
    ],
    seed: int,
    drift_scale: float,
    walkers: int,
    random_policy_seed: int,
) -> tuple[float, int, int]:
    if policy not in POLICIES:
        raise ValueError("unknown alignment baseline")
    state_by_owner = {
        owner: (-1,) * len(chain_neighbors(owner, walkers))
        for owner in range(walkers)
    }
    local_index_by_owner = {owner: 0 for owner in range(walkers)}
    spent = 0
    total = 0.0
    nonzero = 0
    rng = np.random.default_rng(
        int(random_policy_seed)
        + 101 * int(seed)
        + 10007 * int(round(1000.0 * drift_scale))
        + POLICIES.index(policy)
    )
    for event, owner in enumerate(trace.owner_schedule):
        local_index = local_index_by_owner[owner]
        state = state_by_owner[owner]
        neighbors = chain_neighbors(owner, walkers)
        capacity = math.floor(float(budget_rate) * (event + 1)) - spent
        actions = ((),) + tuple((donor,) for donor in neighbors)
        if capacity <= 0 or policy == "no_refresh":
            selected: tuple[int, ...] = ()
        elif policy == "myopic_alignment":
            selected = max(
                actions,
                key=lambda action: (
                    score_table[(owner, local_index, state, action)].lyapunov_decrease,
                    -len(action),
                    tuple(-value for value in action),
                ),
            )
            if score_table[(owner, local_index, state, selected)].lyapunov_decrease <= score_table[(owner, local_index, state, ())].lyapunov_decrease:
                selected = ()
        elif policy == "round_robin":
            selected = (neighbors[local_index % len(neighbors)],)
        elif policy == "random":
            selected = (neighbors[int(rng.integers(0, len(neighbors)))],)
        elif policy == "fixed_left":
            selected = (min(neighbors),)
        elif policy == "fixed_right":
            selected = (max(neighbors),)
        elif policy == "age":
            selected = (
                max(
                    neighbors,
                    key=lambda donor: (
                        local_index - state[neighbors.index(donor)],
                        -donor,
                    ),
                ),
            )
        elif policy in {"parameter_gap", "action_gap"}:
            event_actors = trace.actor_snapshots[event]
            cache = _cache_for_state(
                owner=owner,
                neighbors=neighbors,
                state=state,
                launch_events=tuple(
                    index
                    for index, value in enumerate(trace.owner_schedule)
                    if value == owner
                ),
                global_event=event,
                actor_snapshots=trace.actor_snapshots,
                initial_actors=trace.initial_actors,
            )
            if policy == "parameter_gap":
                selected = (
                    max(
                        neighbors,
                        key=lambda donor: (
                            _parameter_gap(event_actors[donor], cache[donor]),
                            -donor,
                        ),
                    ),
                )
            else:
                selected = (
                    max(
                        neighbors,
                        key=lambda donor: (
                            float(
                                np.linalg.norm(
                                    _actor_action(
                                        event_actors[donor],
                                        trace.observations[event][f"walker_{donor}"],
                                    )
                                    - _actor_action(
                                        cache[donor],
                                        trace.observations[event][f"walker_{donor}"],
                                    )
                                )
                            ),
                            -donor,
                        ),
                    ),
                )
        else:
            raise AssertionError("unreachable baseline")
        score = score_table[(owner, local_index, state, selected)]
        total += score.lyapunov_decrease
        nonzero += int(score.packet_weight > 0.0)
        spent += len(selected)
        state_by_owner[owner] = _next_state(
            state, neighbors, selected, local_index
        )
        local_index_by_owner[owner] += 1
    return float(total), int(spent), int(nonzero)


def run_alignment_scenario(
    *,
    seed: int,
    drift_scale: float,
    budget_rates: Sequence[float],
    walkers: int = 5,
    events: int = 20,
    horizon: int = 4,
    discount: float = 0.99,
    action_difference: float = 0.02,
    learning_smoothness: float = 1.0,
    maximum_packet_weight: float = 1.0,
    actor_seed: int = 95200,
    schedule_seed: int = 96700,
    random_policy_seed: int = 96800,
) -> list[AlignmentOracleRow]:
    trace = _public_trace(
        seed=seed,
        drift_scale=drift_scale,
        walkers=walkers,
        events=events,
        horizon=horizon,
        actor_seed=actor_seed,
        schedule_seed=schedule_seed,
    )
    table, ideal_by_event, charged, replay_error = build_alignment_score_table(
        trace=trace,
        seed=seed,
        horizon=horizon,
        discount=discount,
        action_difference=action_difference,
        learning_smoothness=learning_smoothness,
        maximum_packet_weight=maximum_packet_weight,
        walkers=walkers,
    )
    options = [
        local_alignment_options(
            owner=owner,
            owner_schedule=trace.owner_schedule,
            walkers=walkers,
            score_table=table,
        )
        for owner in range(walkers)
    ]
    owner_counts = [trace.owner_schedule.count(owner) for owner in range(walkers)]
    output: list[AlignmentOracleRow] = []
    for budget in budget_rates:
        exact, spent, success, status, gap, selected = solve_prefix_alignment_oracle(
            options_by_owner=options,
            budget_rate=budget,
            events=events,
        )
        baseline = {
            policy: _baseline_total(
                policy=policy,
                budget_rate=budget,
                trace=trace,
                score_table=table,
                seed=seed,
                drift_scale=drift_scale,
                walkers=walkers,
                random_policy_seed=random_policy_seed,
            )
            for policy in POLICIES
        }
        strong_name = max(
            POLICIES,
            key=lambda policy: (baseline[policy][0], policy),
        )
        strong = baseline[strong_name][0]
        no_refresh = baseline["no_refresh"][0]
        headroom = exact - strong
        gain = exact - no_refresh
        maximum_prefix_excess = max(
            sum(
                cost
                for option in selected
                for launch, cost in zip(option.launch_events, option.cost_sequence)
                if launch <= event
            )
            - math.floor(float(budget) * (event + 1))
            for event in range(events)
        )
        nonzero_weights = sum(
            int(
                table[(option.owner, local_index, state, action)].packet_weight
                > 0.0
            )
            for option in selected
            for local_index, (state, action) in enumerate(
                _states_and_actions(option, walkers=walkers)
            )
        )
        ideal = float(sum(ideal_by_event.values()))
        output.append(
            AlignmentOracleRow(
                seed=int(seed),
                drift_scale=float(drift_scale),
                budget_rate=float(budget),
                exact_oracle_decrease=float(exact),
                exact_oracle_spent_edges=int(spent),
                exact_oracle_nonzero_weights=int(nonzero_weights),
                strong_online_policy=strong_name,
                strong_online_decrease=float(strong),
                no_refresh_decrease=float(no_refresh),
                ideal_reference_decrease=ideal,
                oracle_gain_over_no_refresh=float(gain),
                oracle_headroom_over_strong=float(headroom),
                oracle_recovery_gap=float(headroom / abs(gain)) if abs(gain) > 1e-20 else 0.0,
                normalized_absolute_headroom=float(headroom / max(ideal, 1e-20)),
                optimizer_success=bool(success),
                optimizer_status=int(status),
                optimizer_mip_gap=float(gap),
                selected_owner_count=len(selected),
                maximum_prefix_excess=int(maximum_prefix_excess),
                charged_diagnostic_transitions=int(charged),
                public_trace_transitions=events,
                reference_replay_error=float(replay_error),
                minimum_owner_launches=min(owner_counts),
            )
        )
    return output


def _states_and_actions(
    option: AlignmentLocalOption, *, walkers: int
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    neighbors = chain_neighbors(option.owner, walkers)
    state = (-1,) * len(neighbors)
    output: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for local_index, action in enumerate(option.action_sequence):
        output.append((state, action))
        state = _next_state(state, neighbors, action, local_index)
    return output


def analyze_alignment_rows(rows: Sequence[AlignmentOracleRow]) -> dict[str, object]:
    if not rows:
        raise ValueError("empty alignment-headroom table")
    active = [row for row in rows if row.drift_scale == 0.04]
    gain = sum(row.oracle_gain_over_no_refresh for row in active)
    headroom = sum(row.oracle_headroom_over_strong for row in active)
    recovery = headroom / abs(gain) if abs(gain) > 1e-20 else 0.0
    direction = float(np.mean([row.oracle_headroom_over_strong > 1e-12 for row in active]))
    normalized = float(np.median([row.normalized_absolute_headroom for row in active]))
    diagnostic_by_scenario = {
        (row.seed, row.drift_scale): row.charged_diagnostic_transitions
        for row in rows
    }
    gates = {
        "H1_complete_finite": all(
            math.isfinite(value)
            for row in rows
            for value in asdict(row).values()
            if isinstance(value, float)
        ),
        "H2_exact_milp_optimum": all(
            row.optimizer_success
            and row.optimizer_status == 0
            and row.optimizer_mip_gap <= 1e-9
            and row.selected_owner_count == 5
            for row in rows
        ),
        "H3_prefix_budget": all(row.maximum_prefix_excess <= 0 for row in rows),
        "H4_deterministic_replay": all(row.reference_replay_error <= 1e-12 for row in rows),
        "H5_owner_coverage": all(row.minimum_owner_launches >= 4 for row in rows),
        "H6_fully_charged_diagnostic": all(
            row.charged_diagnostic_transitions > row.public_trace_transitions
            for row in rows
        ),
        "H7_reference_signal": all(row.ideal_reference_decrease > 1e-12 for row in rows),
        "H8_active_oracle_gain": gain > 0.0,
        "H9_active_recovery": recovery >= 0.10,
        "H10_active_direction": direction >= 0.75,
        "H11_active_normalized_headroom": normalized >= 0.005,
        "H12_joint_weight_nontrivial": all(
            0 < row.exact_oracle_nonzero_weights <= 20 for row in active
        ),
    }
    return {
        "status": "multiwalker_privileged_alignment_headroom_development",
        "rows": len(rows),
        "active_oracle_gain_over_no_refresh": float(gain),
        "active_oracle_headroom_over_strong": float(headroom),
        "active_oracle_recovery_gap": float(recovery),
        "active_direction_rate": float(direction),
        "active_median_normalized_absolute_headroom": float(normalized),
        "charged_diagnostic_transitions": int(sum(diagnostic_by_scenario.values())),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "cell_metrics": [asdict(row) for row in rows],
    }


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def merge_alignment_rows(paths: Sequence[Path]) -> list[AlignmentOracleRow]:
    rows: list[AlignmentOracleRow] = []
    keys: set[tuple[int, float, float]] = set()
    for supplied in paths:
        path = supplied / "alignment_rows.json" if supplied.is_dir() else supplied
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("alignment chunk must contain a row list")
        for record in payload:
            row = AlignmentOracleRow(**record)
            key = (row.seed, row.drift_scale, row.budget_rate)
            if key in keys:
                raise ValueError("duplicate alignment-headroom cell")
            keys.add(key)
            rows.append(row)
    return sorted(rows, key=lambda row: (row.seed, row.drift_scale, row.budget_rate))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seeds", type=int, nargs="+")
    source.add_argument("--merge-inputs", type=Path, nargs="+")
    parser.add_argument("--drift-scales", type=float, nargs="+", default=(0.01, 0.04))
    parser.add_argument("--budget-rates", type=float, nargs="+", default=(0.25, 0.5))
    parser.add_argument("--events", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=4)
    args = parser.parse_args()
    if args.merge_inputs:
        rows = merge_alignment_rows(args.merge_inputs)
    else:
        rows = list(
            itertools.chain.from_iterable(
                run_alignment_scenario(
                    seed=seed,
                    drift_scale=drift,
                    budget_rates=args.budget_rates,
                    events=args.events,
                    horizon=args.horizon,
                )
                for seed, drift in itertools.product(args.seeds, args.drift_scales)
            )
        )
    rows = sorted(rows, key=lambda row: (row.seed, row.drift_scale, row.budget_rate))
    summary = analyze_alignment_rows(rows)
    rows_path = args.output_dir / "alignment_rows.json"
    summary_path = args.output_dir / "summary.json"
    _write(rows_path, [asdict(row) for row in rows])
    _write(summary_path, summary)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key != "cell_metrics"
            }
            | {
                "rows_sha256": _hash(rows_path),
                "summary_sha256": _hash(summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

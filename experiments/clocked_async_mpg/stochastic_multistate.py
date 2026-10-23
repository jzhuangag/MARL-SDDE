"""Development-only stochastic packet simulator for the clocked MPG line."""

from __future__ import annotations

import heapq
import math

import numpy as np
from numpy.typing import NDArray

from .exact_multistate_confirmation import (
    DISCOUNT,
    INITIAL_LOGITS,
    LOCAL_WEIGHTS,
    START,
    TRANSITION,
    Game,
    _duration_rng,
    _service_duration,
    derived_seed,
    make_game,
    maximum_event_delay,
    potential_and_gradient,
)
from .finite_time_drift import (
    rate_balanced_steps,
    single_flight_constant_step,
    single_flight_local_steps,
    single_flight_pathwise_constant_step,
)


Array = NDArray[np.float64]


def exact_truncated_gradient(logits: Array, game: Game, horizon: int) -> Array:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    logits = np.asarray(logits, dtype=float)
    probability = 1.0/(1.0+np.exp(-logits))
    signed = 2.0*probability-1.0
    distribution = START.copy()
    occupancy = np.zeros(TRANSITION.shape[0], dtype=float)
    for time in range(horizon):
        occupancy += DISCOUNT**time*distribution
        distribution = distribution@TRANSITION
    gradient = np.zeros_like(logits)
    gradient[0] = (
        2.0
        *occupancy
        *probability[0]
        *(1.0-probability[0])
        *(LOCAL_WEIGHTS[0]+game.coupling*signed[1])
    )
    gradient[1] = (
        2.0
        *occupancy
        *probability[1]
        *(1.0-probability[1])
        *(LOCAL_WEIGHTS[1]+game.coupling*signed[0])
    )
    return gradient


def sample_reinforce_packet(
    logits: Array,
    game: Game,
    agent: int,
    horizon: int,
    batch_size: int,
    generator: np.random.Generator,
) -> Array:
    """Sample one fixed-cost joint Markov REINFORCE block packet."""

    logits = np.asarray(logits, dtype=float)
    if logits.shape != INITIAL_LOGITS.shape:
        raise ValueError("logits have the wrong shape")
    if agent not in (0, 1):
        raise ValueError("agent must be zero or one")
    if horizon <= 0 or batch_size <= 0:
        raise ValueError("horizon and batch_size must be positive")
    probability = 1.0/(1.0+np.exp(-logits))
    states = generator.choice(TRANSITION.shape[0], size=batch_size, p=START)
    visited = np.empty((horizon, batch_size), dtype=np.int64)
    scores = np.empty((horizon, batch_size), dtype=float)
    rewards = np.empty((horizon, batch_size), dtype=float)
    for time in range(horizon):
        visited[time] = states
        action_probability = probability[:, states]
        actions = generator.random((2, batch_size)) < action_probability
        signed = 2.0*actions.astype(float)-1.0
        rewards[time] = (
            LOCAL_WEIGHTS[0, states]*signed[0]
            +LOCAL_WEIGHTS[1, states]*signed[1]
            +game.coupling*signed[0]*signed[1]
        )
        scores[time] = actions[agent].astype(float)-action_probability[agent]
        uniforms = generator.random(batch_size)
        next_states = np.empty(batch_size, dtype=np.int64)
        for state in range(TRANSITION.shape[0]):
            mask = states == state
            if np.any(mask):
                next_states[mask] = np.searchsorted(
                    np.cumsum(TRANSITION[state]), uniforms[mask], side="right"
                )
        states = next_states

    absolute_discounted_reward = rewards*DISCOUNT**np.arange(horizon)[:, None]
    returns = np.cumsum(absolute_discounted_reward[::-1], axis=0)[::-1]
    estimator = np.zeros(TRANSITION.shape[0], dtype=float)
    for time in range(horizon):
        estimator += np.bincount(
            visited[time],
            weights=scores[time]*returns[time],
            minlength=TRANSITION.shape[0],
        )
    return estimator/batch_size


def _trajectory_rng(
    namespace: str, seed_index: int, policy_stream: int, agent: int
) -> np.random.Generator:
    stream = 1000+100*policy_stream+agent
    return np.random.default_rng(derived_seed(namespace, seed_index, stream))


def _endpoint(logits: Array, game: Game, initial_gap: float) -> tuple[float, float]:
    potential, gradient = potential_and_gradient(logits, game)
    return (
        max(0.0, game.optimum-potential)/initial_gap,
        float(np.linalg.norm(gradient)),
    )


def simulate_stochastic_asynchronous(
    coupling: float,
    service_ratio: float,
    seed_index: int,
    namespace: str,
    maximum_time: float,
    horizon: int,
    batch_size: int,
    step_fraction: float,
    target_normalized_gap: float,
    step_rule: str = "single_flight_local",
    history_inflation: float = 1.0,
) -> dict[str, float | int | None]:
    game = make_game(coupling)
    delay = maximum_event_delay(service_ratio)
    rates = np.asarray([1.0, 1.0/service_ratio])
    probabilities = rates/np.sum(rates)
    if step_rule == "single_flight_local":
        allocation = single_flight_local_steps(
            game.lipschitz, probabilities, delay, history_inflation
        )
        step_sizes = step_fraction*np.asarray(allocation["step_sizes"])
    elif step_rule == "single_flight_constant":
        allocation = single_flight_constant_step(
            game.lipschitz, probabilities, delay, history_inflation
        )
        step_sizes = step_fraction*np.asarray(allocation["step_sizes"])
    elif step_rule == "single_flight_pathwise_constant":
        allocation = single_flight_pathwise_constant_step(
            game.lipschitz, delay, history_inflation
        )
        step_sizes = step_fraction*np.asarray(allocation["step_sizes"])
    elif step_rule == "generic_rate_balanced":
        allocation = rate_balanced_steps(
            game.lipschitz, probabilities, delay, history_inflation
        )
        step_sizes = step_fraction*np.asarray(allocation["step_sizes"])
    elif step_rule == "common_global":
        global_step = step_fraction/float(
            np.max(np.sum(game.lipschitz, axis=1))
        )
        step_sizes = np.full(2, global_step, dtype=float)
    else:
        raise ValueError("unknown asynchronous step_rule")
    logits = INITIAL_LOGITS.copy()
    initial_gap = game.optimum-potential_and_gradient(logits, game)[0]
    service_rng = [
        _duration_rng(namespace, seed_index, agent) for agent in range(2)
    ]
    trajectory_rng = [
        _trajectory_rng(namespace, seed_index, 0, agent) for agent in range(2)
    ]
    queue: list[tuple[float, int, int, float, Array]] = []
    for agent in range(2):
        duration = _service_duration(service_rng[agent], agent, service_ratio)
        packet = sample_reinforce_packet(
            logits, game, agent, horizon, batch_size, trajectory_rng[agent]
        )
        heapq.heappush(queue, (duration, agent, 0, duration, packet))
    applied = 0
    completed_packets = 0
    packet_transition_cost = float(batch_size*horizon)
    time_to_target: float | None = None
    work_at_target: float | None = None
    maximum_realized_delay = 0
    while queue:
        completion, agent, birth_version, duration, packet = heapq.heappop(queue)
        if completion > maximum_time:
            break
        event_delay = applied-birth_version
        maximum_realized_delay = max(maximum_realized_delay, event_delay)
        if event_delay > delay:
            raise AssertionError("registered deterministic delay bound was violated")
        logits[agent] += step_sizes[agent]*packet
        applied += 1
        completed_packets += 1
        normalized_gap, _ = _endpoint(logits, game, initial_gap)
        if time_to_target is None and normalized_gap <= target_normalized_gap:
            time_to_target = completion
            partial_work = 0.0
            for other_completion, _, _, other_duration, _ in queue:
                other_start = other_completion-other_duration
                fraction = min(
                    1.0,
                    max(0.0, (completion-other_start)/other_duration),
                )
                partial_work += packet_transition_cost*fraction
            work_at_target = completed_packets*packet_transition_cost+partial_work
        next_duration = _service_duration(
            service_rng[agent], agent, service_ratio
        )
        next_packet = sample_reinforce_packet(
            logits, game, agent, horizon, batch_size, trajectory_rng[agent]
        )
        heapq.heappush(
            queue,
            (
                completion+next_duration,
                agent,
                applied,
                next_duration,
                next_packet,
            ),
        )
    normalized_gap, gradient_norm = _endpoint(logits, game, initial_gap)
    partial_at_horizon = 0.0
    for other_completion, _, _, other_duration, _ in queue:
        other_start = other_completion-other_duration
        fraction = min(
            1.0,
            max(0.0, (maximum_time-other_start)/other_duration),
        )
        partial_at_horizon += packet_transition_cost*fraction
    return {
        "applied_updates": applied,
        "cancelled_transition_work": 0.0,
        "completed_packets": completed_packets,
        "completed_transition_work": completed_packets*packet_transition_cost,
        "final_gradient_norm": gradient_norm,
        "final_normalized_gap": normalized_gap,
        "max_realized_delay": maximum_realized_delay,
        "registered_delay": delay,
        "time_to_target": time_to_target,
        "total_transition_work": (
            completed_packets*packet_transition_cost+partial_at_horizon
        ),
        "transition_work_at_target": work_at_target,
    }


def simulate_stochastic_shadow_barrier(
    coupling: float,
    service_ratio: float,
    seed_index: int,
    namespace: str,
    maximum_time: float,
    horizon: int,
    batch_size: int,
    step_fraction: float,
    target_normalized_gap: float,
) -> dict[str, float | int | None]:
    game = make_game(coupling)
    logits = INITIAL_LOGITS.copy()
    initial_gap = game.optimum-potential_and_gradient(logits, game)[0]
    global_step = step_fraction/float(np.max(np.sum(game.lipschitz, axis=1)))
    service_rng = [
        _duration_rng(namespace, seed_index, agent) for agent in range(2)
    ]
    trajectory_rng = [
        _trajectory_rng(namespace, seed_index, 0, agent) for agent in range(2)
    ]
    time = 0.0
    rounds = 0
    completed_packets = 0
    cancelled_transition_work = 0.0
    time_to_target: float | None = None
    work_at_target: float | None = None
    while True:
        first_duration = np.asarray(
            [
                _service_duration(service_rng[agent], agent, service_ratio)
                for agent in range(2)
            ]
        )
        barrier = float(np.max(first_duration))
        if time+barrier > maximum_time:
            terminal_window = maximum_time-time
            for agent in range(2):
                duration = float(first_duration[agent])
                elapsed = 0.0
                while elapsed+duration <= terminal_window:
                    elapsed += duration
                    completed_packets += 1
                    duration = _service_duration(
                        service_rng[agent], agent, service_ratio
                    )
                cancelled_transition_work += (
                    batch_size
                    *horizon
                    *(terminal_window-elapsed)
                    /duration
                )
            break
        packet_gradients: list[list[Array]] = [[], []]
        for agent in range(2):
            packet_gradients[agent].append(
                sample_reinforce_packet(
                    logits,
                    game,
                    agent,
                    horizon,
                    batch_size,
                    trajectory_rng[agent],
                )
            )
            elapsed = float(first_duration[agent])
            while True:
                duration = _service_duration(
                    service_rng[agent], agent, service_ratio
                )
                remaining = barrier-elapsed
                if duration > remaining:
                    cancelled_transition_work += (
                        batch_size*horizon*remaining/duration
                    )
                    break
                elapsed += duration
                packet_gradients[agent].append(
                    sample_reinforce_packet(
                        logits,
                        game,
                        agent,
                        horizon,
                        batch_size,
                        trajectory_rng[agent],
                    )
                )
        gradient = np.vstack(
            [np.mean(packet_gradients[agent], axis=0) for agent in range(2)]
        )
        logits += global_step*gradient
        time += barrier
        rounds += 1
        completed_packets += sum(len(packets) for packets in packet_gradients)
        normalized_gap, _ = _endpoint(logits, game, initial_gap)
        if time_to_target is None and normalized_gap <= target_normalized_gap:
            time_to_target = time
            work_at_target = (
                completed_packets*batch_size*horizon
                +cancelled_transition_work
            )
    normalized_gap, gradient_norm = _endpoint(logits, game, initial_gap)
    return {
        "applied_updates": 2*rounds,
        "cancelled_transition_work": cancelled_transition_work,
        "completed_packets": completed_packets,
        "completed_transition_work": completed_packets*batch_size*horizon,
        "final_gradient_norm": gradient_norm,
        "final_normalized_gap": normalized_gap,
        "time_to_target": time_to_target,
        "total_transition_work": (
            completed_packets*batch_size*horizon+cancelled_transition_work
        ),
        "transition_work_at_target": work_at_target,
    }

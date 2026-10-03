"""End-to-end asynchronous factor-switch controller with charged identification."""

from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass
from math import floor
from typing import Literal, Sequence

import numpy as np

from .joint_factor_lyapunov import (
    choose_joint_factor_action_variable_bounds,
    projected_quadratic_minimizer,
)
from .markov_alignment_certificate import (
    empirical_conditional_markov_alignment_lower_bound,
    finite_horizon_markov_alignment,
    summable_event_failure_probability,
)


PolicyName = Literal[
    "certified_joint",
    "plugin_joint",
    "exact_joint",
    "fixed_edge_0",
    "fixed_edge_1",
    "fixed_edge_2",
    "fixed_initial",
    "state_myopic",
    "round_robin",
    "random_edge",
    "no_refresh",
]


@dataclass(frozen=True)
class AsyncFactorConfig:
    transition: tuple[tuple[float, float], tuple[float, float]] = (
        (0.9, 0.1),
        (0.1, 0.9),
    )
    score_by_action: tuple[tuple[float, float], tuple[float, float]] = (
        (0.25, -0.25),
        (-0.25, 0.25),
    )
    packet_horizon: int = 4
    total_events: int = 5120
    communication_budget: float = 0.5
    maximum_delay: int = 4
    maximum_packet_weight: float = 0.02
    queue_step: float = 0.02
    learning_weight: float = 100.0
    total_failure_probability: float = 0.05
    certificate_refresh_period: int = 64
    initial_parameter: float = 1.0
    base_curvature: float = 0.02
    parameter_bound: float = 2.0


@dataclass(frozen=True)
class AsyncFactorResult:
    policy: str
    seed: int
    terminal_potential: float
    average_potential: float
    messages: int
    environment_transitions: int
    applied_packets: int
    positive_weight_packets: int
    warmup_events: int
    total_events: int
    maximum_queue: float
    final_queue: float
    certified_action_accuracy: float
    certificate_refreshes: int
    final_parameter: float


def forecast_reversal_config(
    *,
    cycle_probability: float,
    total_events: int = 4096,
    communication_budget: float = 0.5,
    maximum_delay: int = 4,
) -> AsyncFactorConfig:
    """Three-factor model where trajectory value can reverse myopic ranking.

    Every factor has zero stationary mean under the uniform stationary law.
    As the context cycles faster, the two-step trajectory value anticipates
    the next state and differs from the instantaneous-score ranking.
    """

    if not 0.0 <= cycle_probability <= 1.0:
        raise ValueError("cycle probability must lie in [0,1]")
    stay = 1.0 - cycle_probability
    return AsyncFactorConfig(
        transition=(
            (stay, cycle_probability, 0.0),
            (0.0, stay, cycle_probability),
            (cycle_probability, 0.0, stay),
        ),
        score_by_action=(
            (0.05, 0.11, -0.16),
            (-0.16, 0.10, 0.06),
            (0.16, -0.23, 0.07),
        ),
        packet_horizon=2,
        total_events=total_events,
        communication_budget=communication_budget,
        maximum_delay=maximum_delay,
    )


def _validate_config(config: AsyncFactorConfig) -> None:
    transition = np.asarray(config.transition, dtype=float)
    scores = np.asarray(config.score_by_action, dtype=float)
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition must be square")
    if scores.ndim != 2 or scores.shape[1] != transition.shape[0]:
        raise ValueError("scores must have one column per Markov state")
    if scores.shape[0] < 1:
        raise ValueError("at least one candidate edge is required")
    if np.any(transition < 0.0) or not np.allclose(transition.sum(axis=1), 1.0):
        raise ValueError("transition must be row stochastic")
    if min(
        config.packet_horizon,
        config.total_events,
        config.maximum_delay,
        config.certificate_refresh_period,
    ) <= 0:
        raise ValueError("horizons, counts, delay and refresh period must be positive")
    if not 0.0 < config.communication_budget <= 1.0:
        raise ValueError("communication budget must lie in (0,1]")
    if config.maximum_packet_weight <= 0.0 or config.queue_step <= 0.0:
        raise ValueError("packet and queue steps must be positive")
    if config.learning_weight <= 0.0:
        raise ValueError("learning weight must be positive")
    if config.base_curvature <= 0.0 or config.parameter_bound <= 0.0:
        raise ValueError("base curvature and parameter bound must be positive")
    if not 0.0 < config.total_failure_probability < 1.0:
        raise ValueError("failure probability must lie in (0,1)")


def periodic_budget_slot(event: int, budget: float) -> bool:
    if event < 0 or not 0.0 < budget <= 1.0:
        raise ValueError("invalid event or budget")
    return floor((event + 1) * budget) > floor(event * budget)


def generate_common_path(
    *, config: AsyncFactorConfig, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate CRN Markov states, service delays and random-edge choices."""

    _validate_config(config)
    total_events = config.total_events
    transition = np.asarray(config.transition, dtype=float)
    action_count = len(config.score_by_action)
    state_rng = np.random.default_rng(seed * 1009 + 17)
    delay_rng = np.random.default_rng(seed * 1013 + 23)
    edge_rng = np.random.default_rng(seed * 1019 + 31)
    states = np.empty(total_events * config.packet_horizon + 1, dtype=int)
    states[0] = int(state_rng.integers(0, transition.shape[0]))
    uniforms = state_rng.random(states.size - 1)
    cumulative = np.cumsum(transition, axis=1)
    for index, uniform in enumerate(uniforms):
        states[index + 1] = int(
            np.searchsorted(cumulative[states[index]], uniform, side="right")
        )
    delays = delay_rng.integers(
        1, config.maximum_delay + 1, size=total_events, endpoint=False
    )
    random_edges = edge_rng.integers(0, action_count, size=total_events)
    return states, delays, random_edges


def _packet_states(states: np.ndarray, event: int, horizon: int) -> np.ndarray:
    start = event * horizon
    return states[start : start + horizon]


def _add_full_information_observations(
    *,
    counts: np.ndarray,
    score_sums: np.ndarray,
    score_counts: np.ndarray,
    states: np.ndarray,
    event: int,
    horizon: int,
    score_by_action: np.ndarray,
) -> None:
    start = event * horizon
    for offset in range(horizon):
        source = int(states[start + offset])
        target = int(states[start + offset + 1])
        for action in range(score_by_action.shape[0]):
            counts[action, source, target] += 1
            score_sums[action, source] += float(score_by_action[action, source])
            score_counts[action, source] += 1


def _certified_score_intervals(
    *,
    counts: np.ndarray,
    score_sums: np.ndarray,
    score_counts: np.ndarray,
    config: AsyncFactorConfig,
    certificate_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    action_count, state_count = score_sums.shape
    lower = np.empty((action_count, state_count), dtype=float)
    upper = np.empty((action_count, state_count), dtype=float)
    event_delta = summable_event_failure_probability(
        total_failure_probability=config.total_failure_probability,
        event_index=certificate_index,
    )
    score_values = np.asarray(config.score_by_action, dtype=float)
    score_range = float(np.max(score_values) - np.min(score_values))
    for action in range(action_count):
        item = empirical_conditional_markov_alignment_lower_bound(
            transition_counts=counts[action],
            score_sum_by_state=score_sums[action],
            score_count_by_state=score_counts[action],
            score_range=score_range,
            initial_state=0,
            future_horizon=config.packet_horizon,
            total_action_count=action_count + 1,
            event_failure_probability=event_delta,
        )
        negative = empirical_conditional_markov_alignment_lower_bound(
            transition_counts=counts[action],
            score_sum_by_state=-score_sums[action],
            score_count_by_state=score_counts[action],
            score_range=score_range,
            initial_state=0,
            future_horizon=config.packet_horizon,
            total_action_count=action_count + 1,
            event_failure_probability=event_delta,
        )
        lower[action] = np.asarray(item.robust_value_by_state)
        upper[action] = -np.asarray(negative.robust_value_by_state)
    return lower, upper


def _plugin_score_values(
    *,
    counts: np.ndarray,
    score_sums: np.ndarray,
    score_counts: np.ndarray,
    config: AsyncFactorConfig,
) -> np.ndarray:
    """Plug-in finite-horizon score, unidentified until every state is seen."""

    action_count, state_count = score_sums.shape
    values = np.full((action_count, state_count), np.nan, dtype=float)
    for action in range(action_count):
        row_counts = counts[action].sum(axis=1)
        if np.any(row_counts <= 0) or np.any(score_counts[action] <= 0):
            continue
        empirical_transition = counts[action] / row_counts[:, None]
        empirical_score = score_sums[action] / score_counts[action]
        for state in range(state_count):
            values[action, state] = finite_horizon_markov_alignment(
                transition=empirical_transition,
                score_by_state=empirical_score,
                initial_state=state,
                horizon=config.packet_horizon,
            )
    return values


def _exact_packet_weight(
    *, alignment: float, action: int, config: AsyncFactorConfig
) -> float:
    edge_bound = float(
        config.base_curvature * config.parameter_bound
        + np.max(np.abs(np.asarray(config.score_by_action)))
    )
    null_action = len(config.score_by_action)
    gradient_bound = (
        float(config.base_curvature * config.parameter_bound)
        if action == null_action
        else edge_bound
    )
    receipt_motion = (
        2.0
        * config.maximum_delay
        * config.maximum_packet_weight
        * edge_bound
    )
    effective_alignment = alignment - gradient_bound * receipt_motion
    weight, _ = projected_quadratic_minimizer(
        linear_gain=effective_alignment,
        quadratic_curvature=gradient_bound**2,
        maximum_weight=config.maximum_packet_weight,
    )
    return weight


def simulate_async_factor_policy(
    *, policy: PolicyName, seed: int, config: AsyncFactorConfig
) -> AsyncFactorResult:
    """Run one policy with a common Markov path and delayed receipt heap."""

    _validate_config(config)
    states, delays, random_edges = generate_common_path(config=config, seed=seed)
    scores = np.asarray(config.score_by_action, dtype=float)
    transition = np.asarray(config.transition, dtype=float)
    action_count, state_count = scores.shape
    null_action = action_count
    exact_edge_means = np.asarray(
        [
            [
                finite_horizon_markov_alignment(
                    transition=transition,
                    score_by_state=scores[action_index],
                    initial_state=state_index,
                    horizon=config.packet_horizon,
                )
                for state_index in range(state_count)
            ]
            for action_index in range(action_count)
        ],
        dtype=float,
    )
    # Certification is sequential: the controller learns on every trajectory,
    # initially falls back to the local/null action, and starts refreshing only
    # when past charged data make an edge beneficial.  There is no fixed idle
    # identification phase.
    warmup_events = 0
    total_events = config.total_events
    parameter = float(config.initial_parameter)
    queue = 0.0
    maximum_queue = 0.0
    messages = applied = positive_packets = 0
    cumulative_potential = 0.0
    receipt_heap: list[tuple[int, int, float, float]] = []
    counts = np.zeros((action_count, state_count, state_count), dtype=int)
    score_sums = np.zeros((action_count, state_count), dtype=float)
    score_counts = np.zeros((action_count, state_count), dtype=int)
    certificate_lower = np.full((action_count, state_count), -np.inf)
    certificate_upper = np.full((action_count, state_count), np.inf)
    certificate_refreshes = 0
    correct_certified = certified_decisions = 0
    scheduled_message_index = 0
    initial_state = int(states[0])
    sequence = 0

    gradient_bound = float(
        config.base_curvature * config.parameter_bound + np.max(np.abs(scores))
    )
    receipt_motion = (
        2.0
        * config.maximum_delay
        * config.maximum_packet_weight
        * gradient_bound
    )

    for event in range(total_events):
        while receipt_heap and receipt_heap[0][0] <= event:
            _, _, weight, gradient = heapq.heappop(receipt_heap)
            parameter = float(
                np.clip(
                    parameter - weight * gradient,
                    -config.parameter_bound,
                    config.parameter_bound,
                )
            )
            applied += 1
        cumulative_potential += 0.5 * parameter**2
        start_state = int(states[event * config.packet_horizon])
        block_states = _packet_states(states, event, config.packet_horizon)
        action = null_action
        weight = 0.0

        if policy in ("certified_joint", "plugin_joint"):
            if event % config.certificate_refresh_period == 0:
                if policy == "certified_joint":
                    certificate_lower, certificate_upper = _certified_score_intervals(
                        counts=counts,
                        score_sums=score_sums,
                        score_counts=score_counts,
                        config=config,
                        certificate_index=certificate_refreshes,
                    )
                else:
                    plugin = _plugin_score_values(
                        counts=counts,
                        score_sums=score_sums,
                        score_counts=score_counts,
                        config=config,
                    )
                    certificate_lower = np.where(np.isnan(plugin), -np.inf, plugin)
                    certificate_upper = np.where(np.isnan(plugin), np.inf, plugin)
                certificate_refreshes += 1
            alignments = {null_action: float(config.base_curvature * parameter**2)}
            for candidate in range(action_count):
                mean_bound = (
                    certificate_lower[candidate, start_state]
                    if parameter >= 0.0
                    else certificate_upper[candidate, start_state]
                )
                alignments[candidate] = float(
                    parameter * (config.base_curvature * parameter + mean_bound)
                )
            choice = choose_joint_factor_action_variable_bounds(
                alignment_lower_by_action=alignments,
                reset_benefit_by_action={
                    candidate: 0.0 for candidate in range(action_count + 1)
                },
                communication_cost_by_action={
                    **{candidate: 1.0 for candidate in range(action_count)},
                    null_action: 0.0,
                },
                gradient_norm_upper_by_action={
                    **{candidate: gradient_bound for candidate in range(action_count)},
                    null_action: config.base_curvature * config.parameter_bound,
                },
                communication_queue=queue,
                learning_weight=config.learning_weight,
                learning_smoothness=1.0,
                receipt_motion_upper=receipt_motion,
                receipt_cache_linear_upper=0.0,
                outgoing_cache_weight=0.0,
                maximum_packet_weight=config.maximum_packet_weight,
            )
            action = int(choice.action)
            weight = float(choice.packet_weight)
            if action < action_count:
                certified_decisions += 1
                exact_edge_alignments = []
                for candidate in range(action_count):
                    exact_edge_mean = exact_edge_means[candidate, start_state]
                    exact_edge_alignments.append(
                        parameter
                        * (config.base_curvature * parameter + exact_edge_mean)
                    )
                correct_certified += int(action == int(np.argmax(exact_edge_alignments)))
        elif policy == "exact_joint":
            alignments = {null_action: float(config.base_curvature * parameter**2)}
            for candidate in range(action_count):
                expected_gradient = exact_edge_means[candidate, start_state]
                alignments[candidate] = float(
                    parameter
                    * (config.base_curvature * parameter + expected_gradient)
                )
            choice = choose_joint_factor_action_variable_bounds(
                alignment_lower_by_action=alignments,
                reset_benefit_by_action={
                    candidate: 0.0 for candidate in range(action_count + 1)
                },
                communication_cost_by_action={
                    **{candidate: 1.0 for candidate in range(action_count)},
                    null_action: 0.0,
                },
                gradient_norm_upper_by_action={
                    **{candidate: gradient_bound for candidate in range(action_count)},
                    null_action: config.base_curvature * config.parameter_bound,
                },
                communication_queue=queue,
                learning_weight=config.learning_weight,
                learning_smoothness=1.0,
                receipt_motion_upper=receipt_motion,
                receipt_cache_linear_upper=0.0,
                outgoing_cache_weight=0.0,
                maximum_packet_weight=config.maximum_packet_weight,
            )
            action = int(choice.action)
            weight = float(choice.packet_weight)
        else:
            if policy != "no_refresh" and periodic_budget_slot(
                event, config.communication_budget
            ):
                if policy == "fixed_edge_0":
                    action = 0
                elif policy == "fixed_edge_1":
                    action = 1
                elif policy == "fixed_edge_2":
                    if action_count < 3:
                        raise ValueError("fixed_edge_2 requires at least three edges")
                    action = 2
                elif policy == "fixed_initial":
                    action = int(np.argmax(scores[:, initial_state]))
                elif policy == "state_myopic":
                    action = int(np.argmax(scores[:, start_state]))
                elif policy == "round_robin":
                    action = scheduled_message_index % action_count
                elif policy == "random_edge":
                    action = int(random_edges[event])
                else:
                    raise ValueError(f"unknown policy {policy}")
                scheduled_message_index += 1
                edge_mean = exact_edge_means[action, start_state]
            else:
                action = null_action
                edge_mean = 0.0
            weight = _exact_packet_weight(
                alignment=float(
                    parameter * (config.base_curvature * parameter + edge_mean)
                ),
                action=action,
                config=config,
            )

        # A refresh candidate with zero packet weight is a no-op and therefore
        # transmits no cache packet.  Charging it would make fixed baselines
        # artificially weak and would not match the implemented update.
        cost = float(action < action_count and weight > 0.0)
        if policy in ("certified_joint", "plugin_joint", "exact_joint"):
            queue = max(0.0, queue + config.queue_step * (cost - config.communication_budget))
            maximum_queue = max(maximum_queue, queue)
        if cost > 0.0:
            messages += 1
        if policy in ("certified_joint", "plugin_joint"):
            _add_full_information_observations(
                counts=counts,
                score_sums=score_sums,
                score_counts=score_counts,
                states=states,
                event=event,
                horizon=config.packet_horizon,
                score_by_action=scores,
            )
        edge_gradient = (
            float(np.mean(scores[action, block_states]))
            if action < action_count
            else 0.0
        )
        gradient = config.base_curvature * parameter + edge_gradient
        if weight > 0.0:
            positive_packets += 1
            heapq.heappush(
                receipt_heap,
                (event + int(delays[event]), sequence, weight, gradient),
            )
            sequence += 1

    while receipt_heap:
        _, _, weight, gradient = heapq.heappop(receipt_heap)
        parameter = float(
            np.clip(
                parameter - weight * gradient,
                -config.parameter_bound,
                config.parameter_bound,
            )
        )
        applied += 1
    return AsyncFactorResult(
        policy=policy,
        seed=seed,
        terminal_potential=float(0.5 * parameter**2),
        average_potential=float(cumulative_potential / total_events),
        messages=messages,
        environment_transitions=total_events * config.packet_horizon,
        applied_packets=applied,
        positive_weight_packets=positive_packets,
        warmup_events=warmup_events if policy == "certified_joint" else 0,
        total_events=total_events,
        maximum_queue=float(maximum_queue),
        final_queue=float(queue),
        certified_action_accuracy=(
            float(correct_certified / certified_decisions)
            if certified_decisions
            else float("nan")
        ),
        certificate_refreshes=certificate_refreshes,
        final_parameter=float(parameter),
    )


def result_dict(result: AsyncFactorResult) -> dict[str, float | int | str]:
    return asdict(result)

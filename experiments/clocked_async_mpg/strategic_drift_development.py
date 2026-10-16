"""Development-only oracle-headroom simulator for strategic drift scaling.

This module is deliberately separate from the frozen stochastic-confirmation
runner.  It uses exact birth-policy gradients to ask whether the proposed
arrival-time scaling architecture has headroom before a sampled confidence
certificate is implemented.  Its outcomes are design evidence, never formal
algorithm evidence.
"""

from __future__ import annotations

import heapq

import numpy as np

from .exact_multistate_confirmation import (
    INITIAL_LOGITS,
    _duration_rng,
    _service_duration,
    make_game,
    maximum_event_delay,
    potential_and_gradient,
)
from .stochastic_multistate import (
    _endpoint,
    _trajectory_rng,
    sample_reinforce_packet,
)
from .strategic_drift_controller import choose_strategic_drift_scale


def simulate_oracle_strategic_drift(
    *,
    coupling: float,
    service_ratio: float,
    seed_index: int,
    namespace: str,
    maximum_time: float,
    horizon: int,
    batch_size: int,
    step_fraction: float,
    target_normalized_gap: float,
    risk_budget: float,
    tradeoff: float,
    hard_no_harm: bool,
) -> dict[str, float | int | None]:
    """Run the oracle birth-gradient version of the online scalar controller."""

    game = make_game(coupling)
    delay = maximum_event_delay(service_ratio)
    base_step = step_fraction/float(np.max(np.sum(game.lipschitz, axis=1)))
    logits = INITIAL_LOGITS.copy()
    initial_gap = game.optimum-potential_and_gradient(logits, game)[0]
    service_rng = [
        _duration_rng(namespace, seed_index, agent) for agent in range(2)
    ]
    trajectory_rng = [
        _trajectory_rng(namespace, seed_index, 0, agent) for agent in range(2)
    ]
    # completion, agent, birth version, duration, packet, birth logits
    queue: list[tuple[float, int, int, float, np.ndarray, np.ndarray]] = []
    for agent in range(2):
        duration = _service_duration(service_rng[agent], agent, service_ratio)
        packet = sample_reinforce_packet(
            logits, game, agent, horizon, batch_size, trajectory_rng[agent]
        )
        heapq.heappush(
            queue, (duration, agent, 0, duration, packet, logits.copy())
        )

    applied = 0
    completed_packets = 0
    rejected = 0
    debt = 0.0
    scales: list[float] = []
    lower_bounds: list[float] = []
    packet_transition_cost = float(batch_size*horizon)
    time_to_target: float | None = None
    work_at_target: float | None = None
    maximum_realized_delay = 0
    while queue:
        (
            completion,
            agent,
            birth_version,
            duration,
            packet,
            birth_logits,
        ) = heapq.heappop(queue)
        if completion > maximum_time:
            break
        event_delay = applied-birth_version
        maximum_realized_delay = max(maximum_realized_delay, event_delay)
        if event_delay > delay:
            raise AssertionError("registered deterministic delay bound was violated")

        _, birth_gradient = potential_and_gradient(birth_logits, game)
        direction = base_step*packet
        directional_gain = float(birth_gradient[agent]@direction)
        curvature_penalty = (
            0.5
            *float(game.lipschitz[agent, agent])
            *float(direction@direction)
        )
        teammate_drift = 0.0
        for teammate in range(logits.shape[0]):
            if teammate != agent:
                teammate_drift += (
                    float(game.lipschitz[agent, teammate])
                    *float(np.linalg.norm(logits[teammate]-birth_logits[teammate]))
                )
        stale_penalty = float(np.linalg.norm(direction))*teammate_drift
        decision = choose_strategic_drift_scale(
            directional_gain=directional_gain,
            curvature_penalty=curvature_penalty,
            stale_penalty=stale_penalty,
            debt=debt,
            risk_budget=risk_budget,
            tradeoff=tradeoff,
            hard_no_harm=hard_no_harm,
        )
        logits[agent] += decision.scale*direction
        debt = decision.debt_after
        scales.append(decision.scale)
        lower_bounds.append(decision.improvement_lower_bound)
        rejected += int(decision.scale == 0.0)
        applied += 1
        completed_packets += 1

        normalized_gap, _ = _endpoint(logits, game, initial_gap)
        if time_to_target is None and normalized_gap <= target_normalized_gap:
            time_to_target = completion
            partial_work = 0.0
            for other in queue:
                other_completion, _, _, other_duration = other[:4]
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
                logits.copy(),
            ),
        )

    normalized_gap, gradient_norm = _endpoint(logits, game, initial_gap)
    partial_at_horizon = 0.0
    for other in queue:
        other_completion, _, _, other_duration = other[:4]
        other_start = other_completion-other_duration
        fraction = min(
            1.0,
            max(0.0, (maximum_time-other_start)/other_duration),
        )
        partial_at_horizon += packet_transition_cost*fraction
    return {
        "applied_updates": applied,
        "completed_packets": completed_packets,
        "completed_transition_work": completed_packets*packet_transition_cost,
        "debt": debt,
        "final_gradient_norm": gradient_norm,
        "final_normalized_gap": normalized_gap,
        "max_realized_delay": maximum_realized_delay,
        "mean_certified_lower_bound": float(np.mean(lower_bounds)),
        "mean_scale": float(np.mean(scales)),
        "registered_delay": delay,
        "rejected_updates": rejected,
        "time_to_target": time_to_target,
        "total_transition_work": (
            completed_packets*packet_transition_cost+partial_at_horizon
        ),
        "transition_work_at_target": work_at_target,
    }

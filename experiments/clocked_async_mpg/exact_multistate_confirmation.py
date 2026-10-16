"""Exact multi-state CPU confirmation for clocked asynchronous MPG theory."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import math

import numpy as np
from numpy.typing import NDArray

from .finite_time_drift import single_flight_local_steps


Array = NDArray[np.float64]


TRANSITION = np.asarray(
    [[0.84, 0.16, 0.0], [0.0, 0.79, 0.21], [0.11, 0.0, 0.89]],
    dtype=float,
)
START = np.asarray([1.0, 0.0, 0.0], dtype=float)
LOCAL_WEIGHTS = np.asarray(
    [[1.0, 0.8, 1.2], [0.32, 0.26, 0.38]], dtype=float
)
INITIAL_LOGITS = np.asarray(
    [[-1.2, -1.0, -1.4], [1.4, 1.2, 1.5]], dtype=float
)
DISCOUNT = 0.9


@dataclass(frozen=True)
class Game:
    coupling: float
    occupancy: Array
    lipschitz: Array
    optimum: float


def make_game(coupling: float) -> Game:
    if coupling < 0.0 or not math.isfinite(coupling):
        raise ValueError("coupling must be finite and nonnegative")
    occupancy = np.linalg.solve(
        (np.eye(TRANSITION.shape[0])-DISCOUNT*TRANSITION).T, START
    )
    own = np.zeros(2, dtype=float)
    for agent in range(2):
        own[agent] = float(
            np.max(
                2.0
                *occupancy
                *(np.abs(LOCAL_WEIGHTS[agent])+coupling)
                /(6.0*math.sqrt(3.0))
            )
        )
    cross = float(np.max(occupancy)*coupling/4.0)
    lipschitz = np.asarray([[own[0], cross], [cross, own[1]]])
    optimum = float(
        occupancy@(
            LOCAL_WEIGHTS[0]+LOCAL_WEIGHTS[1]+coupling
        )
    )
    return Game(coupling, occupancy, lipschitz, optimum)


def potential_and_gradient(logits: Array, game: Game) -> tuple[float, Array]:
    logits = np.asarray(logits, dtype=float)
    if logits.shape != (2, TRANSITION.shape[0]) or not np.isfinite(logits).all():
        raise ValueError("logits must be a finite (2, states) array")
    probability = 1.0/(1.0+np.exp(-logits))
    signed = 2.0*probability-1.0
    state_reward = (
        LOCAL_WEIGHTS[0]*signed[0]
        +LOCAL_WEIGHTS[1]*signed[1]
        +game.coupling*signed[0]*signed[1]
    )
    potential = float(game.occupancy@state_reward)
    gradient = np.zeros_like(logits)
    gradient[0] = (
        2.0
        *game.occupancy
        *probability[0]
        *(1.0-probability[0])
        *(LOCAL_WEIGHTS[0]+game.coupling*signed[1])
    )
    gradient[1] = (
        2.0
        *game.occupancy
        *probability[1]
        *(1.0-probability[1])
        *(LOCAL_WEIGHTS[1]+game.coupling*signed[0])
    )
    return potential, gradient


def maximum_event_delay(service_ratio: float) -> int:
    if service_ratio < 1.0 or not math.isfinite(service_ratio):
        raise ValueError("service_ratio must be finite and at least one")
    return int(math.ceil((1.1*service_ratio)/0.9))+2


def derived_seed(namespace: str, seed_index: int, stream: int) -> int:
    digest = hashlib.sha256(
        f"{namespace}|{seed_index}|{stream}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little")%(2**63-1)


def _duration_rng(namespace: str, seed_index: int, agent: int) -> np.random.Generator:
    return np.random.default_rng(derived_seed(namespace, seed_index, agent))


def _service_duration(
    generator: np.random.Generator, agent: int, service_ratio: float
) -> float:
    base = 1.0 if agent == 0 else service_ratio
    return float(base*generator.uniform(0.9, 1.1))


def _record(
    trajectory: list[dict[str, float]],
    time: float,
    logits: Array,
    game: Game,
    updates: int,
    packets: int,
) -> None:
    potential, gradient = potential_and_gradient(logits, game)
    trajectory.append(
        {
            "gradient_norm": float(np.linalg.norm(gradient)),
            "normalized_gap": float(
                max(0.0, game.optimum-potential)
                /max(1e-15, game.optimum-potential_and_gradient(INITIAL_LOGITS, game)[0])
            ),
            "packets": float(packets),
            "potential": potential,
            "time": float(time),
            "updates": float(updates),
        }
    )


def simulate_asynchronous(
    coupling: float,
    service_ratio: float,
    seed_index: int,
    namespace: str,
    maximum_time: float,
) -> dict[str, object]:
    game = make_game(coupling)
    delay = maximum_event_delay(service_ratio)
    rates = np.asarray([1.0, 1.0/service_ratio])
    probabilities = rates/np.sum(rates)
    allocation = single_flight_local_steps(
        game.lipschitz,
        probabilities,
        delay,
        history_inflation=1.0,
    )
    step_sizes = 0.8*np.asarray(allocation["step_sizes"])
    logits = INITIAL_LOGITS.copy()
    generators = [
        _duration_rng(namespace, seed_index, agent) for agent in range(2)
    ]
    queue: list[tuple[float, int, int, Array]] = []
    for agent in range(2):
        gradient = potential_and_gradient(logits, game)[1][agent].copy()
        heapq.heappush(
            queue,
            (
                _service_duration(generators[agent], agent, service_ratio),
                agent,
                0,
                gradient,
            ),
        )
    applied = 0
    packets = 0
    maximum_realized_delay = 0
    trajectory: list[dict[str, float]] = []
    _record(trajectory, 0.0, logits, game, applied, packets)
    while queue:
        completion, agent, birth_version, packet_gradient = heapq.heappop(queue)
        if completion > maximum_time:
            break
        event_delay = applied-birth_version
        maximum_realized_delay = max(maximum_realized_delay, event_delay)
        if event_delay > delay:
            raise AssertionError("registered deterministic delay bound was violated")
        logits[agent] += step_sizes[agent]*packet_gradient
        applied += 1
        packets += 1
        _record(trajectory, completion, logits, game, applied, packets)
        next_gradient = potential_and_gradient(logits, game)[1][agent].copy()
        heapq.heappush(
            queue,
            (
                completion+_service_duration(
                    generators[agent], agent, service_ratio
                ),
                agent,
                applied,
                next_gradient,
            ),
        )
    return {
        "max_realized_delay": maximum_realized_delay,
        "registered_delay": delay,
        "step_sizes": step_sizes.tolist(),
        "trajectory": trajectory,
    }


def simulate_shadow_barrier(
    coupling: float,
    service_ratio: float,
    seed_index: int,
    namespace: str,
    maximum_time: float,
) -> dict[str, object]:
    game = make_game(coupling)
    logits = INITIAL_LOGITS.copy()
    generators = [
        _duration_rng(namespace, seed_index, agent) for agent in range(2)
    ]
    global_step = 0.8/float(np.max(np.sum(game.lipschitz, axis=1)))
    time = 0.0
    rounds = 0
    packets = 0
    trajectory: list[dict[str, float]] = []
    _record(trajectory, time, logits, game, rounds, packets)
    while True:
        first = np.asarray(
            [
                _service_duration(generators[agent], agent, service_ratio)
                for agent in range(2)
            ]
        )
        barrier = float(np.max(first))
        if time+barrier > maximum_time:
            break
        counts = np.ones(2, dtype=int)
        for agent in range(2):
            elapsed = float(first[agent])
            while True:
                duration = _service_duration(
                    generators[agent], agent, service_ratio
                )
                if elapsed+duration > barrier:
                    break
                elapsed += duration
                counts[agent] += 1
        gradient = potential_and_gradient(logits, game)[1]
        logits += global_step*gradient
        time += barrier
        rounds += 1
        packets += int(np.sum(counts))
        _record(trajectory, time, logits, game, rounds, packets)
    return {
        "global_step": global_step,
        "trajectory": trajectory,
    }


def summarize_trajectory(
    trajectory: list[dict[str, float]], target_normalized_gap: float
) -> dict[str, float | None]:
    if not 0.0 < target_normalized_gap < 1.0:
        raise ValueError("target_normalized_gap must lie between zero and one")
    first_time: float | None = None
    for row in trajectory:
        if row["normalized_gap"] <= target_normalized_gap:
            first_time = float(row["time"])
            break
    final = trajectory[-1]
    return {
        "final_gradient_norm": float(final["gradient_norm"]),
        "final_normalized_gap": float(final["normalized_gap"]),
        "packets": float(final["packets"]),
        "time_to_target": first_time,
        "updates": float(final["updates"]),
    }

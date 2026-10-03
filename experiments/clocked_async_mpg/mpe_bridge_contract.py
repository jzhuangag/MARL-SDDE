"""Outcome-free execution contracts for the Two Clocks MPE bridge.

The functions in this module contain no rewards or learning outcomes.  They
define the public task shapes, deterministic service clocks, common trajectory
keys, and method-independent work accounting used before a pilot is frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math


METHODS = (
    "offdiag_async",
    "raw_async",
    "delay_scaled_async",
    "frozen_barrier",
)

TASK_AGENTS = {
    "simple_spread_v2": 3,
    "simple_reference_v2": 2,
}


@dataclass(frozen=True)
class WorkCharge:
    environment_steps: int
    actor_transitions: int
    completed_packets: int
    cancelled_environment_steps: int = 0
    cancelled_actor_transitions: int = 0

    def __post_init__(self) -> None:
        values = (
            self.environment_steps,
            self.actor_transitions,
            self.completed_packets,
            self.cancelled_environment_steps,
            self.cancelled_actor_transitions,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("work charges must be integer counts")
        if any(value < 0 for value in values):
            raise ValueError("work charges must be nonnegative")


def service_bases(task: str, profile: str) -> tuple[float, ...]:
    """Return a declared, outcome-independent service profile."""

    if task not in TASK_AGENTS:
        raise ValueError("unknown public MPE task")
    agents = TASK_AGENTS[task]
    if profile == "balanced":
        return tuple(1.0 for _ in range(agents))
    if profile == "heterogeneous":
        return (1.0, 4.0) if agents == 2 else (1.0, 1.55, 4.0)
    raise ValueError("unknown service profile")


def trajectory_seed(
    base_seed: int,
    *,
    task: str,
    owner: int,
    owner_packet_index: int,
    replicate: int,
) -> int:
    """Map a logical packet to a method-independent common-random-number seed."""

    if task not in TASK_AGENTS:
        raise ValueError("unknown public MPE task")
    if not 0 <= owner < TASK_AGENTS[task]:
        raise ValueError("owner is outside the task")
    if owner_packet_index < 0 or replicate not in (0, 1):
        raise ValueError("invalid packet index or replicate")
    payload = f"{base_seed}|{task}|{owner}|{owner_packet_index}|{replicate}".encode()
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    return 1 + (offset % 2_000_000_000)


def async_completed_packets(
    bases: tuple[float, ...], horizon: float
) -> tuple[int, ...]:
    if horizon <= 0.0 or not math.isfinite(horizon):
        raise ValueError("horizon must be finite and positive")
    if not bases or any(value <= 0.0 or not math.isfinite(value) for value in bases):
        raise ValueError("service bases must be finite and positive")
    return tuple(int(math.floor((horizon + 1e-12) / value)) for value in bases)


def barrier_schedule(
    bases: tuple[float, ...], horizon: float, episode_length: int
) -> dict[str, object]:
    """Fully utilize workers inside frozen-policy rounds and charge tail work.

    Each round lasts as long as the slowest worker.  Complete trajectories are
    used in the round update.  A worker that begins another trajectory before
    the barrier but cannot finish it contributes no gradient; its executed
    fraction is nevertheless charged as cancelled environment work.
    """

    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    completed = [0 for _ in bases]
    cancelled_steps = [0 for _ in bases]
    round_length = max(bases)
    start = 0.0
    rounds = 0
    while start < horizon - 1e-12:
        duration = min(round_length, horizon - start)
        for owner, service in enumerate(bases):
            whole = int(math.floor((duration + 1e-12) / service))
            completed[owner] += whole
            remainder = max(0.0, duration - whole * service)
            if remainder > 1e-12:
                cancelled_steps[owner] += min(
                    2 * episode_length,
                    int(math.floor(2 * episode_length * remainder / service + 1e-12)),
                )
        rounds += 1
        start += duration
    return {
        "rounds": rounds,
        "round_length": round_length,
        "completed_by_owner": tuple(completed),
        "cancelled_environment_steps_by_owner": tuple(cancelled_steps),
    }


def packet_scale(method: str, *, event_delay: int, offdiag_scale: float) -> float:
    """Return the predeclared method scale without reading a learning outcome."""

    if method not in METHODS or method == "frozen_barrier":
        raise ValueError("packet_scale applies only to asynchronous methods")
    if event_delay < 0 or not 0.0 < offdiag_scale <= 1.0:
        raise ValueError("invalid event delay or off-diagonal scale")
    if method == "offdiag_async":
        return offdiag_scale
    if method == "delay_scaled_async":
        return 1.0 / (1.0 + event_delay)
    return 1.0


def charge_completed_packets(
    completed_by_owner: tuple[int, ...], episode_length: int
) -> WorkCharge:
    if episode_length <= 0 or not completed_by_owner:
        raise ValueError("invalid episode length or packet counts")
    if any(value < 0 for value in completed_by_owner):
        raise ValueError("packet counts must be nonnegative")
    agents = len(completed_by_owner)
    packets = sum(completed_by_owner)
    # Each packet uses two independent trajectories.
    environment_steps = 2 * episode_length * packets
    return WorkCharge(
        environment_steps=environment_steps,
        actor_transitions=environment_steps * agents,
        completed_packets=packets,
    )

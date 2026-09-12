"""Framework-neutral packet overlay intended for a pinned HARL checkout.

The module contains no HARL source and imports no HARL internals.  A thin
runner adapter can feed it actor log probabilities, flat proposal/validation
gradients and transition counts while upstream HARL remains an external pinned
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .strategic_drift_controller import (
    StrategicDriftDecision,
    choose_strategic_drift_scale,
)


Array = NDArray[np.float64]


@dataclass(frozen=True)
class PacketCompletion:
    agent_id: int
    birth_event: int
    completion_event: int
    event_delay: int
    charged_transitions: int


@dataclass
class _InFlight:
    birth_event: int
    charged_transitions: int


class SingleFlightRegistry:
    """Enforce one in-flight packet per actor and exact transition charging."""

    def __init__(self, num_agents: int):
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        self._in_flight: list[_InFlight | None] = [None]*num_agents
        self._completed_transitions = 0

    @property
    def num_agents(self) -> int:
        return len(self._in_flight)

    @property
    def completed_transitions(self) -> int:
        return self._completed_transitions

    @property
    def active_agents(self) -> tuple[int, ...]:
        return tuple(
            agent for agent, packet in enumerate(self._in_flight)
            if packet is not None
        )

    def launch(
        self, agent_id: int, *, birth_event: int, charged_transitions: int
    ) -> None:
        self._validate_agent(agent_id)
        if birth_event < 0 or charged_transitions <= 0:
            raise ValueError("birth_event and charged_transitions are invalid")
        if self._in_flight[agent_id] is not None:
            raise RuntimeError("actor already has a packet in flight")
        self._in_flight[agent_id] = _InFlight(
            birth_event=birth_event,
            charged_transitions=charged_transitions,
        )

    def complete(self, agent_id: int, *, completion_event: int) -> PacketCompletion:
        self._validate_agent(agent_id)
        packet = self._in_flight[agent_id]
        if packet is None:
            raise RuntimeError("actor has no packet in flight")
        if completion_event < packet.birth_event:
            raise ValueError("completion precedes packet birth")
        self._in_flight[agent_id] = None
        self._completed_transitions += packet.charged_transitions
        return PacketCompletion(
            agent_id=agent_id,
            birth_event=packet.birth_event,
            completion_event=completion_event,
            event_delay=completion_event-packet.birth_event,
            charged_transitions=packet.charged_transitions,
        )

    def _validate_agent(self, agent_id: int) -> None:
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError("agent_id is invalid")


def categorical_mean_kl(birth_log_probs: Array, current_log_probs: Array) -> float:
    """Mean KL(birth || current) over a fixed reference-state batch."""

    birth = np.asarray(birth_log_probs, dtype=float)
    current = np.asarray(current_log_probs, dtype=float)
    if birth.shape != current.shape or birth.ndim != 2 or birth.shape[1] < 2:
        raise ValueError("categorical log probabilities have incompatible shape")
    if not np.isfinite(birth).all() or not np.isfinite(current).all():
        raise ValueError("categorical log probabilities must be finite")
    birth_probabilities = np.exp(birth)
    current_probabilities = np.exp(current)
    if not np.allclose(np.sum(birth_probabilities, axis=1), 1.0, atol=1e-7):
        raise ValueError("birth_log_probs are not normalized")
    if not np.allclose(np.sum(current_probabilities, axis=1), 1.0, atol=1e-7):
        raise ValueError("current_log_probs are not normalized")
    values = np.sum(birth_probabilities*(birth-current), axis=1)
    return float(max(0.0, np.mean(values)))


def diagonal_gaussian_mean_kl(
    birth_mean: Array,
    birth_log_std: Array,
    current_mean: Array,
    current_log_std: Array,
) -> float:
    """Mean KL between diagonal Gaussian policies on reference observations."""

    bm = np.asarray(birth_mean, dtype=float)
    bls = np.asarray(birth_log_std, dtype=float)
    cm = np.asarray(current_mean, dtype=float)
    cls = np.asarray(current_log_std, dtype=float)
    if bm.shape != cm.shape or bm.ndim != 2:
        raise ValueError("Gaussian means have incompatible shape")
    try:
        bls = np.broadcast_to(bls, bm.shape)
        cls = np.broadcast_to(cls, bm.shape)
    except ValueError as error:
        raise ValueError("Gaussian log standard deviations cannot broadcast") from error
    if not all(np.isfinite(value).all() for value in (bm, bls, cm, cls)):
        raise ValueError("Gaussian parameters must be finite")
    variance_ratio = np.exp(2.0*(bls-cls))
    squared_shift = (cm-bm)**2/np.exp(2.0*cls)
    per_coordinate = cls-bls+0.5*(variance_ratio+squared_shift-1.0)
    return float(max(0.0, np.mean(np.sum(per_coordinate, axis=1))))


def teammate_tv_drift_upper(mean_kls: Array) -> float:
    """Sum of Pinsker upper bounds for teammate policy drift."""

    values = np.asarray(mean_kls, dtype=float)
    if values.ndim != 1 or (values < -1e-12).any() or not np.isfinite(values).all():
        raise ValueError("mean_kls must be a finite nonnegative vector")
    return float(np.sum(np.sqrt(0.5*np.maximum(values, 0.0))))


def sample_split_directional_value(
    proposal_step: Array, validation_gradient: Array
) -> float:
    proposal = np.asarray(proposal_step, dtype=float)
    validation = np.asarray(validation_gradient, dtype=float)
    if proposal.shape != validation.shape or proposal.ndim != 1:
        raise ValueError("flat proposal and validation vectors must match")
    if not np.isfinite(proposal).all() or not np.isfinite(validation).all():
        raise ValueError("flat proposal and validation vectors must be finite")
    return float(validation@proposal)


def decide_harl_packet_scale(
    *,
    proposal_step: Array,
    validation_gradient: Array,
    teammate_mean_kls: Array,
    curvature_upper: float,
    mixed_drift_coefficient: float,
    debt: float,
    risk_budget: float,
    tradeoff: float,
    maximum_scale: float = 1.0,
) -> StrategicDriftDecision:
    """Map framework-level packet statistics to the O(1) scalar controller."""

    proposal = np.asarray(proposal_step, dtype=float)
    directional_gain = sample_split_directional_value(
        proposal, validation_gradient
    )
    curvature_upper = float(curvature_upper)
    mixed_drift_coefficient = float(mixed_drift_coefficient)
    if (
        curvature_upper < 0.0
        or mixed_drift_coefficient < 0.0
        or not math.isfinite(curvature_upper)
        or not math.isfinite(mixed_drift_coefficient)
    ):
        raise ValueError("curvature and mixed-drift coefficients are invalid")
    norm = float(np.linalg.norm(proposal))
    curvature_penalty = 0.5*curvature_upper*norm*norm
    stale_penalty = (
        mixed_drift_coefficient*norm*teammate_tv_drift_upper(teammate_mean_kls)
    )
    return choose_strategic_drift_scale(
        directional_gain=directional_gain,
        curvature_penalty=curvature_penalty,
        stale_penalty=stale_penalty,
        debt=debt,
        risk_budget=risk_budget,
        tradeoff=tradeoff,
        maximum_scale=maximum_scale,
    )

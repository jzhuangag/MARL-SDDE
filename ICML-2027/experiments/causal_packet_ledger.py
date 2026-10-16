"""Causal transport interface for the successor asynchronous MARL design.

This module implements resource and packet semantics, not a return guarantee.
Launch decisions create unique jobs. Completion charges transmitted packets
whether or not a scheduler accepts them. Rejected packets cannot refund work.
It intentionally does not copy a worker's old Adam state into the server.
"""

from dataclasses import dataclass
import hashlib
import math

import numpy as np


@dataclass(frozen=True)
class PacketJob:
    job_id: int
    agent: int
    source_versions: tuple[int, ...]
    launch_tick: int
    due_tick: int
    optimizer_rows: int
    message_bytes: int


@dataclass(frozen=True)
class GradientPacket:
    job: PacketJob
    gradient: np.ndarray


def public_delay(seed, agent, launch_ordinal, *, minimum=1, maximum=8):
    """Exogenous service delay independent of policy choices and returns.

    The stream is indexed by per-agent launch ordinal, rather than global tick
    or the cyclic selection order; reversing a launch order cannot change a
    given agent's registered service sequence.
    """
    if not all(isinstance(x, int) for x in (seed, agent, launch_ordinal, minimum, maximum)):
        raise TypeError("integer delay-stream coordinates required")
    if min(seed, agent, launch_ordinal) < 0 or minimum < 1 or maximum < minimum:
        raise ValueError("invalid delay stream")
    word = hashlib.sha256(f"{seed}:{agent}:{launch_ordinal}".encode()).digest()
    return minimum + int.from_bytes(word[:8], "big") % (maximum - minimum + 1)


class CausalPacketLedger:
    """One in-flight job per actor, single-use packets, explicit launch budget."""

    def __init__(self, *, agents, optimizer_row_budget, message_byte_budget):
        if agents < 1 or optimizer_row_budget < 0 or message_byte_budget < 0:
            raise ValueError("nonnegative budgets and positive team size required")
        self.agents = agents
        self.optimizer_row_budget = optimizer_row_budget
        self.message_byte_budget = message_byte_budget
        self.versions = [0] * agents
        self.tick = 0
        self.jobs = {}
        self.pending = {}
        self.ready = {}
        self.consumed = set()
        self.optimizer_rows = 0
        self.reserved_message_bytes = 0
        self.transmitted_message_bytes = 0
        self.transmitted_packets = 0
        self.joint_transitions = 0
        self.actor_transitions = 0
        self.launch_ordinals = [0] * agents
        self.events = []

    def advance(self, tick):
        if not isinstance(tick, int) or tick < self.tick:
            raise ValueError("time cannot move backwards")
        self.tick = tick

    def charge_joint_rollout(self, transitions):
        if not isinstance(transitions, int) or transitions < 0:
            raise ValueError("nonnegative integer joint transitions required")
        self.joint_transitions += transitions
        self.actor_transitions += self.agents * transitions

    def launch(self, *, agent, service_ticks, optimizer_rows, message_bytes):
        if not 0 <= agent < self.agents or agent in self.pending:
            raise ValueError("actor invalid or already has an in-flight job")
        if any(not isinstance(v, int) for v in (service_ticks, optimizer_rows, message_bytes)):
            raise TypeError("integer work and service charges required")
        if service_ticks < 1 or min(optimizer_rows, message_bytes) < 0:
            raise ValueError("invalid job charges")
        if self.optimizer_rows + optimizer_rows > self.optimizer_row_budget:
            raise ValueError("optimizer budget exhausted")
        if self.reserved_message_bytes + message_bytes > self.message_byte_budget:
            raise ValueError("communication budget exhausted")
        job_id = len(self.jobs)
        job = PacketJob(job_id, agent, tuple(self.versions), self.tick,
                        self.tick + service_ticks, optimizer_rows, message_bytes)
        self.jobs[job_id] = job
        self.pending[agent] = job_id
        self.optimizer_rows += optimizer_rows
        self.reserved_message_bytes += message_bytes
        self.launch_ordinals[agent] += 1
        self.events.append({"event": "launch", "tick": self.tick, "job_id": job_id,
                            "agent": agent, "source_versions": list(job.source_versions)})
        return job

    def complete(self, job_id, gradient):
        if job_id not in self.jobs:
            raise ValueError("unknown job")
        job = self.jobs[job_id]
        if self.pending.get(job.agent) != job_id or self.tick < job.due_tick:
            raise ValueError("job has not completed or already returned")
        gradient = np.asarray(gradient, dtype=float).copy()
        if gradient.ndim != 1 or gradient.size == 0 or not np.isfinite(gradient).all():
            raise ValueError("finite raw gradient vector required")
        gradient.flags.writeable = False
        self.ready[job_id] = GradientPacket(job, gradient)
        del self.pending[job.agent]
        self.transmitted_packets += 1
        self.transmitted_message_bytes += job.message_bytes
        self.events.append({"event": "complete", "tick": self.tick, "job_id": job_id})
        return self.ready[job_id]

    def consume(self, job_id, *, alpha):
        """Consume exactly once; the caller applies this scaled raw gradient.

        Alpha zero rejects the packet. Launch work and transmission charges are
        retained. Positive alpha changes the server's actor version. Server
        optimizer moments remain owned by the caller and are never replaced
        by a source-version optimizer snapshot.
        """
        if not math.isfinite(alpha) or alpha < 0 or job_id not in self.ready:
            raise ValueError("nonnegative finite step and ready unused packet required")
        packet = self.ready.pop(job_id)
        self.consumed.add(job_id)
        stale_versions = tuple(v - old for v, old in zip(self.versions, packet.job.source_versions))
        if alpha > 0:
            self.versions[packet.job.agent] += 1
        self.events.append({"event": "commit" if alpha > 0 else "reject", "tick": self.tick,
                            "job_id": job_id, "alpha": float(alpha),
                            "stale_versions": list(stale_versions)})
        return alpha * packet.gradient, stale_versions

    def accounting(self):
        return {"jobs_launched": len(self.jobs), "packets_transmitted": self.transmitted_packets,
                "packets_consumed": len(self.consumed), "pending_jobs": len(self.pending),
                "ready_packets": len(self.ready), "optimizer_rows": self.optimizer_rows,
                "reserved_message_bytes": self.reserved_message_bytes,
                "transmitted_message_bytes": self.transmitted_message_bytes,
                "joint_transitions": self.joint_transitions, "actor_transitions": self.actor_transitions}

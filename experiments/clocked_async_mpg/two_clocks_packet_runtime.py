"""Outcome-free runtime contracts for Two Clocks packet execution.

The runtime is deliberately framework-neutral.  It records policy-block
versions and charged work, but it never accepts rewards, gradients, returns or
task outcomes.  A HARL adapter may attach a packet payload outside this class.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple


@dataclass(frozen=True)
class PacketWork:
    """Fixed work declared before a packet is launched."""

    environment_steps: int
    actor_transitions: int
    optimizer_units: int

    def __post_init__(self) -> None:
        values = (
            self.environment_steps,
            self.actor_transitions,
            self.optimizer_units,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("packet work must contain integer counts")
        if any(value < 0 for value in values):
            raise ValueError("packet work counts must be nonnegative")


@dataclass(frozen=True)
class PacketTicket:
    ticket_id: int
    owner: int
    birth_event: int
    birth_versions: Tuple[int, ...]
    launch_time: float
    scheduled_completion_time: float
    declared_work: PacketWork


@dataclass(frozen=True)
class PacketCompletion:
    ticket_id: int
    owner: int
    birth_event: int
    completion_event: int
    event_delay: int
    birth_versions: Tuple[int, ...]
    arrival_versions: Tuple[int, ...]
    teammate_version_increments: Tuple[int, ...]
    charged_work: PacketWork


@dataclass(frozen=True)
class CancelledPacket:
    ticket_id: int
    owner: int
    reason: str
    charged_work: PacketWork


def _zero_work() -> Tuple[int, int, int]:
    return (0, 0, 0)


class TwoClocksPacketLedger:
    """Enforce single-writer, single-flight execution and exact work charging.

    Completion time is supplied at launch and the API has no reward or packet
    payload argument.  This makes the service schedule structurally separate
    from trajectory innovations; a concrete adapter must preserve that split.
    """

    def __init__(self, num_agents: int):
        if isinstance(num_agents, bool) or not isinstance(num_agents, int):
            raise TypeError("num_agents must be an integer")
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        self._versions = [0 for _ in range(num_agents)]
        self._in_flight: list[Optional[PacketTicket]] = [None for _ in range(num_agents)]
        self._pending: list[Optional[PacketCompletion]] = [None for _ in range(num_agents)]
        self._next_ticket = 0
        self._event = 0
        self._completed_packets = 0
        self._applied_updates = 0
        self._cancelled_packets = 0
        self._completed_work = _zero_work()
        self._cancelled_work = _zero_work()
        self._trace: list[dict[str, object]] = []

    @property
    def num_agents(self) -> int:
        return len(self._versions)

    @property
    def event(self) -> int:
        return self._event

    @property
    def versions(self) -> Tuple[int, ...]:
        return tuple(self._versions)

    @property
    def active_owners(self) -> Tuple[int, ...]:
        return tuple(index for index, ticket in enumerate(self._in_flight) if ticket)

    @property
    def pending_owners(self) -> Tuple[int, ...]:
        return tuple(index for index, item in enumerate(self._pending) if item)

    @property
    def trace(self) -> Tuple[dict[str, object], ...]:
        return tuple(dict(item) for item in self._trace)

    def launch(
        self,
        owner: int,
        *,
        launch_time: float,
        scheduled_completion_time: float,
        declared_work: PacketWork,
    ) -> PacketTicket:
        self._validate_owner(owner)
        if self._in_flight[owner] is not None or self._pending[owner] is not None:
            raise RuntimeError("owner already has an in-flight or unapplied packet")
        if (
            not math.isfinite(launch_time)
            or not math.isfinite(scheduled_completion_time)
            or launch_time < 0.0
            or scheduled_completion_time <= launch_time
        ):
            raise ValueError("packet service times are invalid")
        if not isinstance(declared_work, PacketWork):
            raise TypeError("declared_work must be PacketWork")
        if any(value <= 0 for value in self._work_to_tuple(declared_work)):
            raise ValueError("declared packet work counts must be positive")
        ticket = PacketTicket(
            ticket_id=self._next_ticket,
            owner=owner,
            birth_event=self._event,
            birth_versions=self.versions,
            launch_time=float(launch_time),
            scheduled_completion_time=float(scheduled_completion_time),
            declared_work=declared_work,
        )
        self._next_ticket += 1
        self._in_flight[owner] = ticket
        self._trace.append(
            {
                "kind": "launch",
                "ticket_id": ticket.ticket_id,
                "owner": owner,
                "birth_event": ticket.birth_event,
                "birth_versions": ticket.birth_versions,
                "launch_time": ticket.launch_time,
                "scheduled_completion_time": ticket.scheduled_completion_time,
            }
        )
        return ticket

    def complete(
        self, owner: int, *, ticket_id: int, completion_time: float
    ) -> PacketCompletion:
        self._validate_owner(owner)
        ticket = self._in_flight[owner]
        if ticket is None or ticket.ticket_id != ticket_id:
            raise RuntimeError("completion does not match the owner's in-flight packet")
        if completion_time != ticket.scheduled_completion_time:
            raise ValueError("completion_time differs from the predeclared service time")
        if self._versions[owner] != ticket.birth_versions[owner]:
            raise RuntimeError("single-flight owner block is not self-fresh")
        self._event += 1
        arrival_versions = self.versions
        increments = tuple(
            current - birth
            for current, birth in zip(arrival_versions, ticket.birth_versions)
        )
        if any(value < 0 for value in increments) or increments[owner] != 0:
            raise RuntimeError("policy-version path violates single-writer ownership")
        completion = PacketCompletion(
            ticket_id=ticket.ticket_id,
            owner=owner,
            birth_event=ticket.birth_event,
            completion_event=self._event,
            event_delay=self._event - ticket.birth_event,
            birth_versions=ticket.birth_versions,
            arrival_versions=arrival_versions,
            teammate_version_increments=increments,
            charged_work=ticket.declared_work,
        )
        self._in_flight[owner] = None
        self._pending[owner] = completion
        self._completed_packets += 1
        self._completed_work = self._work_tuple_add(
            self._completed_work, ticket.declared_work
        )
        self._trace.append(
            {
                "kind": "complete",
                "ticket_id": ticket.ticket_id,
                "owner": owner,
                "completion_event": self._event,
                "arrival_versions": arrival_versions,
                "teammate_version_increments": increments,
                "completion_time": completion_time,
            }
        )
        return completion

    def apply(self, owner: int, *, ticket_id: int) -> int:
        """Record the sole legal write after a matching owner packet completes."""

        self._validate_owner(owner)
        completion = self._pending[owner]
        if completion is None or completion.ticket_id != ticket_id:
            raise RuntimeError("apply does not match the owner's completed packet")
        self._versions[owner] += 1
        self._pending[owner] = None
        self._applied_updates += 1
        self._trace.append(
            {
                "kind": "apply",
                "ticket_id": ticket_id,
                "owner": owner,
                "new_owner_version": self._versions[owner],
            }
        )
        return self._versions[owner]

    def cancel(
        self,
        owner: int,
        *,
        ticket_id: int,
        charged_work: PacketWork,
        reason: str,
    ) -> CancelledPacket:
        self._validate_owner(owner)
        ticket = self._in_flight[owner]
        if ticket is None or ticket.ticket_id != ticket_id:
            raise RuntimeError("cancellation does not match an in-flight packet")
        if not isinstance(charged_work, PacketWork):
            raise TypeError("charged_work must be PacketWork")
        declared = ticket.declared_work
        if (
            charged_work.environment_steps > declared.environment_steps
            or charged_work.actor_transitions > declared.actor_transitions
            or charged_work.optimizer_units > declared.optimizer_units
        ):
            raise ValueError("cancelled work cannot exceed declared packet work")
        if not reason.strip():
            raise ValueError("cancellation reason must be nonempty")
        self._in_flight[owner] = None
        self._cancelled_packets += 1
        self._cancelled_work = self._work_tuple_add(
            self._cancelled_work, charged_work
        )
        cancelled = CancelledPacket(
            ticket_id=ticket_id,
            owner=owner,
            reason=reason,
            charged_work=charged_work,
        )
        self._trace.append(
            {
                "kind": "cancel",
                "ticket_id": ticket_id,
                "owner": owner,
                "reason": reason,
                "charged_work": self._work_to_tuple(charged_work),
            }
        )
        return cancelled

    def accounting(self) -> dict[str, object]:
        completed = self._completed_work
        cancelled = self._cancelled_work
        return {
            "completed_packets": self._completed_packets,
            "applied_updates": self._applied_updates,
            "cancelled_packets": self._cancelled_packets,
            "completed_work": completed,
            "cancelled_work": cancelled,
            "total_charged_work": tuple(
                left + right for left, right in zip(completed, cancelled)
            ),
        }

    def assert_quiescent(self) -> None:
        if self.active_owners or self.pending_owners:
            raise RuntimeError("runtime teardown left active or unapplied packets")
        if self._completed_packets != self._applied_updates:
            raise RuntimeError("completed packet count differs from applied updates")

    def _validate_owner(self, owner: int) -> None:
        if isinstance(owner, bool) or not isinstance(owner, int):
            raise TypeError("owner must be an integer")
        if owner < 0 or owner >= self.num_agents:
            raise ValueError("owner is outside the policy-block range")

    @staticmethod
    def _work_to_tuple(work: PacketWork) -> Tuple[int, int, int]:
        return (work.environment_steps, work.actor_transitions, work.optimizer_units)

    @classmethod
    def _work_tuple_add(
        cls, accumulator: Tuple[int, int, int], work: PacketWork
    ) -> Tuple[int, int, int]:
        values = cls._work_to_tuple(work)
        return tuple(left + right for left, right in zip(accumulator, values))

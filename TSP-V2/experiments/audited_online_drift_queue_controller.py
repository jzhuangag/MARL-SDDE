"""Audit-preserving extension of the frozen sliding-window controller."""

from __future__ import annotations

from online_drift_queue_controller import (
    OnlineObservation,
    SlidingWindowDriftQueueController,
)


class AuditedSlidingWindowDriftQueueController(SlidingWindowDriftQueueController):
    """Keep cumulative accounting separate from statistical window state.

    The inherited histories remain window-truncated and therefore preserve the
    frozen decision rule and memory complexity.  The additional counters and
    ledger are deterministic observers of completed actions and never enter a
    score, feasibility check, queue update, or tie break.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.selection_counts = {q: 0 for q in self.actions}
        self.decision_ledger: list[dict] = []

    def observe(self, *, risk_before: float, risk_after: float) -> OnlineObservation:
        if self._pending is None:
            raise RuntimeError("decide must precede observe")
        action = self._pending.action
        decision_index = self.step + 1
        observation = super().observe(
            risk_before=risk_before,
            risk_after=risk_after,
        )
        cost = self.costs[action]
        self.selection_counts[action] += 1
        self.decision_ledger.append(
            {
                "decision": decision_index,
                "q": action,
                "messages": cost.messages,
                "environment_ticks": cost.environment_ticks,
                "message_remaining": observation.message_remaining_after,
                "environment_remaining": observation.environment_remaining_after,
                "message_queue": observation.message_queue,
                "environment_queue": observation.environment_queue,
                "risk_before": float(risk_before),
                "risk_after": float(risk_after),
                "progress": observation.progress,
            }
        )
        return observation

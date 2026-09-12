"""Outcome-free static design primitives for a possible EXP-017B.

This is not a runner or preregistration.  It encodes the anti-starvation
invariants that a later independent preregistration would have to preserve.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from exp017a_nonlinear_config import (
    B_CANDIDATES,
    FLOAT_BYTES,
    Q_CANDIDATES,
    SERVER_OVERHEAD_BYTES,
)


PROBE_Q = 4
PROBE_B = 1
INITIAL_PROBE_BLOCKS = 8
PERIODIC_PROBE_INTERVAL = 32
FORMAL_SEEDS = None
PUBLIC_STRONG_FIXED_Q = {
    ("acrobot", "message_binding"): 4,
    ("cartpole", "message_binding"): 4,
    ("acrobot", "environment_binding"): 16,
    ("cartpole", "environment_binding"): 32,
}


@dataclass(frozen=True)
class ResourceCharge:
    message_bytes: int
    environment_steps: int
    agent_transitions: int


def probe_due(block: int) -> bool:
    """Public schedule independent of losses, selected q, and outcomes."""

    if block < 0:
        raise ValueError("block must be nonnegative")
    return block < INITIAL_PROBE_BLOCKS or block % PERIODIC_PROBE_INTERVAL == 0


def pairwise_trials(q: int) -> int:
    if q < 1:
        raise ValueError("q must be positive")
    return q * (q - 1) // 2


def action_charge(q: int, b: int, parameters: int) -> ResourceCharge:
    if q < 1 or b < 1 or parameters < 1:
        raise ValueError("q, b, and parameters must be positive")
    return ResourceCharge(
        message_bytes=SERVER_OVERHEAD_BYTES + q * parameters * FLOAT_BYTES,
        environment_steps=b,
        agent_transitions=q * b,
    )


def public_fallback_q(task: str, budget: str) -> int:
    """Return the frozen nontrivial strong baseline; never silently q=1."""

    return PUBLIC_STRONG_FIXED_Q[(task, budget)]


def learning_q(task: str, budget: str, evidence_ready: bool, proposed_q: int) -> int:
    """Decouple no-evidence fallback from a later evidence-based proposal."""

    if proposed_q not in Q_CANDIDATES:
        raise ValueError("unregistered q candidate")
    if not evidence_ready:
        return public_fallback_q(task, budget)
    return proposed_q


def probe_trials_after_blocks(blocks: int, learning_q_value: int) -> int:
    """Show that probe information grows even when learning participation is one."""

    if learning_q_value not in Q_CANDIDATES:
        raise ValueError("unregistered learning q")
    return sum(pairwise_trials(PROBE_Q) for block in range(blocks) if probe_due(block))


@lru_cache(maxsize=None)
def cached_candidate_features(parameters: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache vectorizable action features; no matrix inverse or preconditioner."""

    actions = np.asarray(
        [(q, b) for q in Q_CANDIDATES for b in B_CANDIDATES], dtype=np.int64
    )
    message_cost = SERVER_OVERHEAD_BYTES + actions[:, 0] * parameters * FLOAT_BYTES
    inverse_q = 1.0 / actions[:, 0].astype(np.float64)
    return actions, message_cost, inverse_q

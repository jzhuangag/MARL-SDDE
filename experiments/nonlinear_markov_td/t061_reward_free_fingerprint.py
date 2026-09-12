"""Reward-free MinAtar trajectory fingerprint for T-061."""

from __future__ import annotations

import hashlib
import struct

import numpy as np

from experiments.nonlinear_markov_td.run_t060a_minatar_fixed_q_pilot import (
    derived_seed,
)
from experiments.nonlinear_markov_td.t059_minatar_fixed_encoder import (
    FULL_ACTIONS,
    GAMES,
    legacy_numpy_seed,
)


def fingerprint_collision_upper_bound(length: int) -> float:
    """Independent full fingerprints can match only if all requests match."""

    if length < 1:
        raise ValueError("fingerprint length must be positive")
    return float(FULL_ACTIONS ** (-length))


def phase_action(
    *, overhead: int, rho_estimate: float, actions: tuple[int, ...] = (1, 4, 16)
) -> int:
    """Closed-form catalogue action with deterministic smaller-q tie break."""

    if overhead <= 0 or not 0.0 <= rho_estimate <= 1.0 or not actions:
        raise ValueError("invalid phase-action input")
    return min(
        sorted(set(int(q) for q in actions)),
        key=lambda q: (
            (overhead + q)
            * (rho_estimate + (1.0 - rho_estimate) / q),
            q,
        ),
    )


def action_from_match_count(*, matches: int, blocks: int, overhead: int) -> int:
    if blocks < 1 or matches < 0 or matches > blocks:
        raise ValueError("invalid match count")
    return phase_action(overhead=overhead, rho_estimate=matches / blocks)


def controller_updates(
    *,
    overhead: int,
    q: int,
    delay: int,
    target_qmax_updates: int,
    probe_blocks: int,
    probe_q: int,
    fingerprint_length: int,
    q_max: int = 16,
) -> dict[str, int]:
    """Charge probe, learning, and delay against both frozen budgets."""

    if min(overhead, q, target_qmax_updates, probe_blocks, probe_q, fingerprint_length) < 1:
        raise ValueError("cost inputs must be positive")
    if delay < 0 or q > q_max:
        raise ValueError("invalid delay or participation")
    message_budget = (overhead + q_max) * target_qmax_updates
    environment_budget = q_max * target_qmax_updates
    probe_message = probe_blocks * (overhead + probe_q)
    probe_environment = probe_blocks * probe_q * fingerprint_length
    remaining_message = message_budget - probe_message
    remaining_environment = environment_budget - probe_environment
    synchronized = min(
        remaining_message // (overhead + q),
        remaining_environment // q,
    )
    updates = max(0, synchronized - delay)
    return {
        "updates": int(updates),
        "message_budget": int(message_budget),
        "environment_budget": int(environment_budget),
        "probe_message": int(probe_message),
        "probe_environment": int(probe_environment),
        "learning_message": int((updates + delay) * (overhead + q)),
        "learning_environment": int((updates + delay) * q),
    }


def _single_fingerprint(
    environment,
    *,
    environment_seed: int,
    policy_seed: int,
    length: int,
) -> bytes:
    environment.seed(legacy_numpy_seed(environment_seed))
    environment.last_action = 0
    environment.reset()
    policy = np.random.RandomState(legacy_numpy_seed(policy_seed))
    digest = hashlib.sha256()
    for _ in range(length):
        digest.update(np.ascontiguousarray(environment.state()).tobytes())
        digest.update(struct.pack("<b", int(environment.last_action)))
        requested = int(policy.randint(FULL_ACTIONS))
        digest.update(struct.pack("<b", requested))
        reward, terminal = environment.act(requested)
        digest.update(struct.pack("<d?", float(reward), bool(terminal)))
        if terminal:
            environment.last_action = 0
            environment.reset()
    return digest.digest()


def probe_match_count(
    *,
    game: str,
    rho: float,
    blocks: int,
    length: int,
    master_seed: int,
    sticky_action_probability: float = 0.1,
    difficulty_ramping: bool = False,
) -> int:
    """Observe two coupled actor fingerprints on independent reset blocks."""

    if game not in GAMES or not 0.0 <= rho <= 1.0 or blocks < 1 or length < 1:
        raise ValueError("invalid probe design")
    from minatar import Environment

    environments = [
        Environment(
            game,
            sticky_action_prob=sticky_action_probability,
            difficulty_ramping=difficulty_ramping,
        )
        for _ in range(3)
    ]
    matches = 0
    for block in range(blocks):
        fingerprints = []
        for path, environment in enumerate(environments):
            fingerprints.append(
                _single_fingerprint(
                    environment,
                    environment_seed=derived_seed(
                        master_seed, game, "probe-environment", block, path
                    ),
                    policy_seed=derived_seed(
                        master_seed, game, "probe-policy", block, path
                    ),
                    length=length,
                )
            )
        random = np.random.default_rng(
            derived_seed(master_seed, game, "probe-switch", rho, block)
        )
        shared = random.random(2) < np.sqrt(rho)
        first = fingerprints[0] if shared[0] else fingerprints[1]
        second = fingerprints[0] if shared[1] else fingerprints[2]
        matches += int(first == second)
    return matches


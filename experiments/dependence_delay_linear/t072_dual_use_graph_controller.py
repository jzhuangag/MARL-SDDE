"""Predictable dual-use graph control with a Lyapunov safety-debt queue.

Every environment transition is used by learning.  At a decision block, the
first part of the block is also compressed into a recipient fingerprint.  The
resulting graph proposal is therefore measurable before the second part of the
block.  The second part supplies both ordinary learning updates and a fresh
conditional loss comparison against a same-data local shadow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from experiments.dependence_delay_linear.t070_nonstationary_graph import (
    recipient_actions,
)
from experiments.dependence_delay_linear.t071_sampled_graph_controller import (
    advance_history,
    candidate_parameter,
    local_updates,
    stale_snapshot,
)


@dataclass(frozen=True)
class DualUseTrajectory:
    auc_risk: float
    terminal_risk: float
    risk_path: np.ndarray
    proposed_actions: np.ndarray
    accepted_actions: np.ndarray
    debt_path: np.ndarray
    rollback_states: np.ndarray
    learning_transitions: int
    fingerprint_transitions: int
    message_units: int
    candidate_scores: int


def _select_predictable_actions(
    *,
    local_pre: np.ndarray,
    shadow_pre: np.ndarray,
    donor_snapshot: np.ndarray,
    selection_samples: np.ndarray,
    alpha_grid: Sequence[float],
    debt: np.ndarray,
    drift_weight: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Select one proposal per recipient before validation samples arrive."""

    samples = np.asarray(selection_samples, dtype=float)
    agents = local_pre.size
    if samples.ndim != 2 or samples.shape[1] != agents or samples.shape[0] < 1:
        raise ValueError("selection samples have incompatible shape")
    if debt.shape != (agents,) or np.any(debt < 0.0):
        raise ValueError("debt must be a nonnegative agent vector")
    if drift_weight < 0.0:
        raise ValueError("drift weight must be nonnegative")
    proposals = np.empty(agents, dtype=float)
    selected = np.zeros(agents, dtype=np.int16)
    scores = 0
    for recipient in range(agents):
        catalogue = recipient_actions(agents, recipient, alpha_grid)
        candidates = np.asarray([
            candidate_parameter(local_pre, donor_snapshot, recipient, action)
            for action in catalogue
        ])
        losses = np.mean(
            np.square(candidates[:, None] - samples[None, :, recipient]), axis=1
        )
        shadow_loss = float(np.mean(
            np.square(shadow_pre[recipient] - samples[:, recipient])
        ))
        estimated_excess = losses - shadow_loss
        # Squared mixing displacement is the observable action-dependent term
        # in the one-step quadratic Lyapunov upper bound.  Multiplying it by
        # the virtual safety debt is the standard drift-plus-penalty coupling:
        # accumulated validation harm progressively suppresses aggressive
        # transfers even when their in-sample loss looks favorable.
        exposure = np.square(candidates - local_pre[recipient])
        drift_penalty = drift_weight * debt[recipient] * exposure
        index = int(np.argmin(losses + drift_penalty))
        proposals[recipient] = candidates[index]
        selected[recipient] = index
        scores += len(catalogue)
    return proposals, selected, scores


def simulate_dual_use_controller(
    *,
    observations: np.ndarray,
    targets: np.ndarray,
    initial_parameter: float,
    gain: float,
    delay: int,
    decision_blocks: Sequence[int],
    pre_steps: int,
    selection_steps: int,
    alpha_grid: Sequence[float],
    drift_weight: float,
    safety_slack: float,
    rollback_margin: float,
    fingerprint_message_units: int = 1,
    mixing_message_units: int = 1,
) -> DualUseTrajectory:
    """Run the zero-extra-transition controller and its local shadow.

    The proposal depends only on the first ``pre_steps`` observations.  Its
    empirical no-harm test uses the disjoint remaining observations.  Both
    portions update the learner, so fingerprints consume no additional actor
    transition.  ``fingerprint_transitions`` records reuse, not extra cost.
    """

    samples = np.asarray(observations, dtype=float)
    target = np.asarray(targets, dtype=float)
    if samples.ndim != 3 or target.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError("observations and targets have incompatible shapes")
    blocks, steps_per_block, agents = samples.shape
    if not 0 < selection_steps <= pre_steps < steps_per_block:
        raise ValueError("invalid dual-use split")
    if drift_weight < 0.0 or safety_slack < 0.0:
        raise ValueError("invalid Lyapunov queue parameters")
    decisions = set(int(value) for value in decision_blocks)
    parameters = np.repeat(float(initial_parameter), agents)
    shadow = parameters.copy()
    history = [parameters.copy() for _ in range(delay)]
    debt = np.zeros(agents, dtype=float)
    risks: list[float] = []
    proposals_path: list[np.ndarray] = []
    accepted_path: list[np.ndarray] = []
    debt_path: list[np.ndarray] = []
    rollback_path: list[np.ndarray] = []
    messages = 0
    candidate_scores = 0
    reused = 0
    for block in range(blocks):
        pre = samples[block, :pre_steps]
        post = samples[block, pre_steps:]
        local_pre = local_updates(parameters, pre, gain)
        shadow_pre = local_updates(shadow, pre, gain)
        donor = stale_snapshot(history, local_pre, delay)
        proposed = np.zeros(agents, dtype=np.int16)
        accepted = np.zeros(agents, dtype=np.int16)
        rolled_back = np.zeros(agents, dtype=bool)
        if block in decisions:
            proposal, proposed, scored = _select_predictable_actions(
                local_pre=local_pre,
                shadow_pre=shadow_pre,
                donor_snapshot=donor,
                selection_samples=pre[:selection_steps],
                alpha_grid=alpha_grid,
                debt=debt,
                drift_weight=drift_weight,
            )
            candidate_scores += scored
            reused += pre_steps
            messages += fingerprint_message_units
            if np.any(proposed > 0):
                messages += mixing_message_units
            validation_excess = np.mean(
                np.square(proposal[None, :] - post)
                - np.square(shadow_pre[None, :] - post),
                axis=0,
            )
            candidate_post = local_updates(proposal, post, gain)
            shadow_post = local_updates(shadow_pre, post, gain)
            accept = validation_excess <= rollback_margin
            parameters = np.where(accept, candidate_post, shadow_post)
            accepted = np.where(accept, proposed, 0).astype(np.int16)
            rolled_back = ~accept
            debt = np.maximum(0.0, debt + validation_excess - safety_slack)
        else:
            parameters = local_updates(local_pre, post, gain)
            shadow_post = local_updates(shadow_pre, post, gain)
        shadow = shadow_post
        advance_history(history, parameters, delay)
        risks.append(float(np.mean(np.square(parameters - target[block]))))
        if block in decisions:
            proposals_path.append(proposed.copy())
            accepted_path.append(accepted.copy())
            debt_path.append(debt.copy())
            rollback_path.append(rolled_back.copy())
    risk_path = np.asarray(risks, dtype=float)
    return DualUseTrajectory(
        auc_risk=float(np.mean(risk_path)),
        terminal_risk=float(risk_path[-1]),
        risk_path=risk_path,
        proposed_actions=np.asarray(proposals_path, dtype=np.int16),
        accepted_actions=np.asarray(accepted_path, dtype=np.int16),
        debt_path=np.asarray(debt_path, dtype=float),
        rollback_states=np.asarray(rollback_path, dtype=bool),
        learning_transitions=blocks * steps_per_block,
        fingerprint_transitions=reused,
        message_units=messages,
        candidate_scores=candidate_scores,
    )

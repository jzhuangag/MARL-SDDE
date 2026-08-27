"""Sampled observable controller for changing delayed collaboration graphs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Literal, Sequence

import numpy as np

from experiments.dependence_delay_linear.t070_nonstationary_graph import (
    RecipientAction,
    recipient_actions,
)


@dataclass(frozen=True)
class PolicyTrajectory:
    auc_risk: float
    terminal_risk: float
    risk_path: np.ndarray
    selected_actions: np.ndarray
    used_shadow: np.ndarray
    checkpoint_violations: int
    learning_transitions: int
    probe_transitions: int
    message_units: int
    candidate_scores: int


def stable_seed(master_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(master_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def sample_markov_observations(
    *,
    targets: np.ndarray,
    steps_per_block: int,
    noise_scale: float,
    spatial_correlation: float,
    temporal_correlation: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample continuous vector AR(1) observation noise with fixed marginals."""

    target = np.asarray(targets, dtype=float)
    if target.ndim != 2 or target.shape[0] < 1 or target.shape[1] < 2:
        raise ValueError("targets must be a blocks-by-agents matrix")
    if steps_per_block < 1 or noise_scale < 0.0:
        raise ValueError("invalid horizon or noise scale")
    if not 0.0 <= spatial_correlation <= 1.0:
        raise ValueError("spatial correlation must lie in [0,1]")
    if not 0.0 <= temporal_correlation < 1.0:
        raise ValueError("temporal correlation must lie in [0,1)")
    blocks, agents = target.shape
    rng = np.random.default_rng(seed)
    covariance = noise_scale * (
        spatial_correlation * np.ones((agents, agents))
        + (1.0 - spatial_correlation) * np.eye(agents)
    )
    jitter = 1e-14 * np.eye(agents)
    root = np.linalg.cholesky(covariance + jitter)
    noise = np.empty((blocks, steps_per_block, agents), dtype=float)
    state = root @ rng.standard_normal(agents)
    innovation_scale = np.sqrt(1.0 - temporal_correlation**2)
    for block in range(blocks):
        for step in range(steps_per_block):
            if block != 0 or step != 0:
                state = (
                    temporal_correlation * state
                    + innovation_scale * (root @ rng.standard_normal(agents))
                )
            noise[block, step] = state
    return target[:, None, :] + noise, noise


def local_updates(parameters: np.ndarray, observations: np.ndarray, gain: float) -> np.ndarray:
    value = np.asarray(parameters, dtype=float).copy()
    samples = np.asarray(observations, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != value.size:
        raise ValueError("observations have incompatible shape")
    if not 0.0 < gain < 1.0:
        raise ValueError("gain must lie in (0,1)")
    for sample in samples:
        value += gain * (sample - value)
    return value


def stale_snapshot(history: list[np.ndarray], local_post: np.ndarray, delay: int) -> np.ndarray:
    if delay == 0:
        return np.asarray(local_post, dtype=float).copy()
    if len(history) != delay:
        raise ValueError("history length does not equal delay")
    return np.asarray(history[0], dtype=float).copy()


def advance_history(history: list[np.ndarray], value: np.ndarray, delay: int) -> None:
    if delay == 0:
        return
    history.pop(0)
    history.append(np.asarray(value, dtype=float).copy())


def candidate_parameter(
    local_post: np.ndarray,
    donor_snapshot: np.ndarray,
    recipient: int,
    action: RecipientAction,
) -> float:
    return float(
        (1.0 - action.alpha) * local_post[recipient]
        + action.alpha * donor_snapshot[action.donor]
    )


def choose_observable_actions(
    *,
    local_post: np.ndarray,
    shadow_post: np.ndarray,
    donor_snapshot: np.ndarray,
    selection_probe: np.ndarray,
    validation_probe: np.ndarray,
    alpha_grid: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Cross-fit graph selection with an independent empirical no-harm screen."""

    selection = np.asarray(selection_probe, dtype=float)
    validation = np.asarray(validation_probe, dtype=float)
    agents = local_post.size
    if selection.ndim != 2 or validation.ndim != 2:
        raise ValueError("probe halves must be matrices")
    if selection.shape[1] != agents or validation.shape[1] != agents:
        raise ValueError("probe halves have incompatible agent count")
    if selection.shape[0] < 1 or validation.shape[0] < 1:
        raise ValueError("both probe halves must be nonempty")
    target_hat = np.mean(selection, axis=0)
    selected = np.zeros(agents, dtype=np.int16)
    used_shadow = np.zeros(agents, dtype=bool)
    output = np.empty(agents, dtype=float)
    scores = 0
    for recipient in range(agents):
        catalogue = recipient_actions(agents, recipient, alpha_grid)
        candidates = np.asarray(
            [
                candidate_parameter(local_post, donor_snapshot, recipient, action)
                for action in catalogue
            ]
        )
        selection_loss = np.square(candidates - target_hat[recipient])
        index = int(np.argmin(selection_loss))
        proposal = float(candidates[index])
        validation_difference = np.mean(
            np.square(proposal - validation[:, recipient])
            - np.square(shadow_post[recipient] - validation[:, recipient])
        )
        if validation_difference <= 0.0:
            output[recipient] = proposal
            selected[recipient] = index
        else:
            output[recipient] = shadow_post[recipient]
            selected[recipient] = 0
            used_shadow[recipient] = True
        scores += len(catalogue)
    return output, selected, used_shadow, scores


def choose_clairvoyant_actions(
    *,
    local_post: np.ndarray,
    shadow_post: np.ndarray,
    donor_snapshot: np.ndarray,
    targets: np.ndarray,
    alpha_grid: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    agents = local_post.size
    output = np.empty(agents, dtype=float)
    selected = np.zeros(agents, dtype=np.int16)
    used_shadow = np.zeros(agents, dtype=bool)
    scores = 0
    for recipient in range(agents):
        catalogue = recipient_actions(agents, recipient, alpha_grid)
        candidates = np.asarray(
            [
                candidate_parameter(local_post, donor_snapshot, recipient, action)
                for action in catalogue
            ]
        )
        risks = np.square(candidates - targets[recipient])
        index = int(np.argmin(risks))
        if risks[index] <= (shadow_post[recipient] - targets[recipient]) ** 2 + 1e-15:
            output[recipient] = candidates[index]
            selected[recipient] = index
        else:
            output[recipient] = shadow_post[recipient]
            used_shadow[recipient] = True
        scores += len(catalogue)
    return output, selected, used_shadow, scores


def simulate_policy(
    *,
    observations: np.ndarray,
    targets: np.ndarray,
    initial_parameter: float,
    gain: float,
    delay: int,
    decision_blocks: Sequence[int],
    probe_steps: int,
    selection_steps: int,
    alpha_grid: Sequence[float],
    policy: Literal[
        "local_no_probe",
        "charged_shadow",
        "static_graph",
        "observable",
        "clairvoyant",
        "full_sharing",
    ],
    static_graph: Sequence[int] | None = None,
    probe_message_units: int = 2,
    mixing_message_units: int = 1,
) -> PolicyTrajectory:
    samples = np.asarray(observations, dtype=float)
    target = np.asarray(targets, dtype=float)
    if samples.ndim != 3 or target.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError("observations and targets have incompatible shapes")
    blocks, steps_per_block, agents = samples.shape
    if not 0 < selection_steps < probe_steps < steps_per_block:
        raise ValueError("invalid split-probe design")
    decisions = set(int(value) for value in decision_blocks)
    parameters = np.repeat(float(initial_parameter), agents)
    shadow = parameters.copy()
    history = [parameters.copy() for _ in range(delay)]
    risks = []
    actions = []
    fallbacks = []
    violations = 0
    learning_transitions = 0
    charged_probes = 0
    messages = 0
    candidate_scores = 0
    graph = None if static_graph is None else np.asarray(static_graph, dtype=int)
    if policy == "static_graph" and graph.shape != (agents,):
        raise ValueError("static_graph has incompatible shape")
    for block in range(blocks):
        is_decision = block in decisions
        if is_decision and policy in {"observable", "clairvoyant", "charged_shadow"}:
            probe = samples[block, :probe_steps]
            learning = samples[block, probe_steps:]
            charged_probes += probe_steps
            messages += probe_message_units
        else:
            probe = None
            learning = samples[block]
        local_post = local_updates(parameters, learning, gain)
        shadow_post = local_updates(shadow, learning, gain)
        donor = stale_snapshot(history, local_post, delay)
        selected = np.zeros(agents, dtype=np.int16)
        used_shadow = np.zeros(agents, dtype=bool)
        if is_decision and policy == "observable":
            parameters, selected, used_shadow, scored = choose_observable_actions(
                local_post=local_post,
                shadow_post=shadow_post,
                donor_snapshot=donor,
                selection_probe=probe[:selection_steps],
                validation_probe=probe[selection_steps:],
                alpha_grid=alpha_grid,
            )
            candidate_scores += scored
            if np.any(selected > 0):
                messages += mixing_message_units
        elif is_decision and policy == "clairvoyant":
            parameters, selected, used_shadow, scored = choose_clairvoyant_actions(
                local_post=local_post,
                shadow_post=shadow_post,
                donor_snapshot=donor,
                targets=target[block],
                alpha_grid=alpha_grid,
            )
            candidate_scores += scored
            if np.any(selected > 0):
                messages += mixing_message_units
        elif is_decision and policy == "static_graph":
            output = np.empty(agents, dtype=float)
            for recipient in range(agents):
                catalogue = recipient_actions(agents, recipient, alpha_grid)
                selected[recipient] = int(graph[recipient])
                output[recipient] = candidate_parameter(
                    local_post, donor, recipient, catalogue[selected[recipient]]
                )
            parameters = output
            if np.any(selected > 0):
                messages += mixing_message_units
        elif is_decision and policy == "full_sharing":
            parameters = np.repeat(float(np.mean(donor)), agents)
            selected[:] = -1
            messages += mixing_message_units
        elif policy == "charged_shadow":
            parameters = shadow_post.copy()
            used_shadow[:] = is_decision
        else:
            parameters = local_post
        shadow = shadow_post
        if is_decision and policy in {"observable", "clairvoyant"}:
            violations += int(
                np.sum(
                    np.square(parameters - target[block])
                    > np.square(shadow - target[block]) + 1e-15
                )
            )
        advance_history(history, parameters, delay)
        learning_transitions += learning.shape[0]
        risks.append(float(np.mean(np.square(parameters - target[block]))))
        if is_decision:
            actions.append(selected.copy())
            fallbacks.append(used_shadow.copy())
    risk_path = np.asarray(risks)
    return PolicyTrajectory(
        auc_risk=float(np.mean(risk_path)),
        terminal_risk=float(risk_path[-1]),
        risk_path=risk_path,
        selected_actions=np.asarray(actions, dtype=np.int16),
        used_shadow=np.asarray(fallbacks, dtype=bool),
        checkpoint_violations=violations,
        learning_transitions=learning_transitions,
        probe_transitions=charged_probes,
        message_units=messages,
        candidate_scores=candidate_scores,
    )


def action_change_statistics(actions: np.ndarray, shift_decisions: Sequence[int]) -> dict[str, int]:
    values = np.asarray(actions, dtype=int)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("actions must contain at least two decisions")
    shifts = set(int(value) for value in shift_decisions)
    shift_changes = 0
    shift_total = 0
    other_changes = 0
    other_total = 0
    for index in range(1, values.shape[0]):
        changed = bool(np.any(values[index] != values[index - 1]))
        if index in shifts:
            shift_total += 1
            shift_changes += int(changed)
        else:
            other_total += 1
            other_changes += int(changed)
    return {
        "shift_changes": shift_changes,
        "shift_total": shift_total,
        "other_changes": other_changes,
        "other_total": other_total,
    }

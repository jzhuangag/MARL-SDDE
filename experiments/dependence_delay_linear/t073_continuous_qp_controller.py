"""Continuous covariance-aware collaboration weights on the probability simplex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from experiments.dependence_delay_linear.t071_sampled_graph_controller import (
    advance_history,
    local_updates,
    stale_snapshot,
)


@dataclass(frozen=True)
class ContinuousTrajectory:
    auc_risk: float
    terminal_risk: float
    risk_path: np.ndarray
    proposed_weights: np.ndarray
    accepted_weights: np.ndarray
    debt_path: np.ndarray
    rollback_states: np.ndarray
    learning_transitions: int
    fingerprint_transitions: int
    message_units: int
    qp_iterations: int


def project_simplex(vector: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the unit probability simplex."""

    value = np.asarray(vector, dtype=float)
    if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("simplex input must be a nonempty finite vector")
    ordered = np.sort(value)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, value.size + 1)
    active = ordered - cumulative / indices > 0.0
    rho = int(np.flatnonzero(active)[-1])
    threshold = cumulative[rho] / float(rho + 1)
    return np.maximum(value - threshold, 0.0)


def pooled_lag_one(samples: np.ndarray) -> float:
    """Conservative observable temporal-correlation estimate."""

    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3:
        return 0.0
    centered = values - np.mean(values, axis=0, keepdims=True)
    numerator = float(np.sum(centered[1:] * centered[:-1]))
    denominator = float(np.sum(np.square(centered[:-1])))
    if denominator <= 1e-15:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 0.95))


def fingerprint_covariance(samples: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a PSD covariance-of-mean proxy and its effective sample size."""

    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("at least two fingerprint samples are required")
    correlation = pooled_lag_one(values)
    effective = max(1.0, values.shape[0] * (1.0 - correlation) / (1.0 + correlation))
    covariance = np.cov(values, rowvar=False, ddof=1) / effective
    covariance = np.atleast_2d(covariance)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return covariance, effective


def solve_recipient_qp(
    *,
    model_values: np.ndarray,
    recipient_target: float,
    covariance_of_mean: np.ndarray,
    recipient: int,
    debt: float,
    drift_weight: float,
    variance_weight: float,
    max_iterations: int = 50,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, int]:
    """Solve the convex simplex QP by projected gradient.

    The objective is estimated squared bias plus covariance-aware estimation
    variance plus the Lyapunov debt-weighted deviation from local learning.
    """

    model = np.asarray(model_values, dtype=float)
    covariance = np.asarray(covariance_of_mean, dtype=float)
    agents = model.size
    if covariance.shape != (agents, agents) or not 0 <= recipient < agents:
        raise ValueError("QP inputs have incompatible shapes")
    if debt < 0.0 or drift_weight < 0.0 or variance_weight < 0.0:
        raise ValueError("QP weights must be nonnegative")
    local = np.zeros(agents, dtype=float)
    local[recipient] = 1.0
    hessian_half = (
        np.outer(model, model)
        + variance_weight * covariance
        + drift_weight * debt * np.eye(agents)
    )
    linear = -float(recipient_target) * model - drift_weight * debt * local
    lipschitz = max(2.0 * float(np.linalg.eigvalsh(hessian_half)[-1]), 1e-12)
    weights = local.copy()
    for iteration in range(1, max_iterations + 1):
        gradient = 2.0 * (hessian_half @ weights + linear)
        updated = project_simplex(weights - gradient / lipschitz)
        if np.linalg.norm(updated - weights) <= tolerance:
            return updated, iteration
        weights = updated
    return weights, max_iterations


def choose_continuous_weights(
    *,
    local_pre: np.ndarray,
    donor_snapshot: np.ndarray,
    fingerprint_samples: np.ndarray,
    debt: np.ndarray,
    drift_weight: float,
    variance_weight: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    samples = np.asarray(fingerprint_samples, dtype=float)
    agents = local_pre.size
    if samples.ndim != 2 or samples.shape[1] != agents:
        raise ValueError("fingerprints have incompatible shape")
    covariance, effective = fingerprint_covariance(samples)
    target_hat = np.mean(samples, axis=0)
    proposals = np.empty(agents, dtype=float)
    matrix = np.empty((agents, agents), dtype=float)
    total_iterations = 0
    for recipient in range(agents):
        models = np.asarray(donor_snapshot, dtype=float).copy()
        models[recipient] = local_pre[recipient]
        weights, iterations = solve_recipient_qp(
            model_values=models,
            recipient_target=float(target_hat[recipient]),
            covariance_of_mean=covariance,
            recipient=recipient,
            debt=float(debt[recipient]),
            drift_weight=drift_weight,
            variance_weight=variance_weight,
        )
        matrix[recipient] = weights
        proposals[recipient] = float(weights @ models)
        total_iterations += iterations
    return proposals, matrix, total_iterations, effective


def simulate_continuous_qp_controller(
    *,
    observations: np.ndarray,
    targets: np.ndarray,
    initial_parameter: float,
    gain: float,
    delay: int,
    decision_blocks: Sequence[int],
    pre_steps: int,
    drift_weight: float,
    variance_weight: float,
    safety_slack: float,
    rollback_margin: float,
    fingerprint_message_units: int = 1,
    mixing_message_units: int = 1,
) -> ContinuousTrajectory:
    samples = np.asarray(observations, dtype=float)
    target = np.asarray(targets, dtype=float)
    if samples.ndim != 3 or target.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError("observations and targets have incompatible shapes")
    blocks, steps_per_block, agents = samples.shape
    if not 2 <= pre_steps < steps_per_block:
        raise ValueError("invalid dual-use split")
    if min(drift_weight, variance_weight, safety_slack) < 0.0:
        raise ValueError("invalid controller parameter")
    decisions = set(int(value) for value in decision_blocks)
    parameters = np.repeat(float(initial_parameter), agents)
    shadow = parameters.copy()
    history = [parameters.copy() for _ in range(delay)]
    debt = np.zeros(agents, dtype=float)
    identity = np.eye(agents)
    risks: list[float] = []
    proposals_path: list[np.ndarray] = []
    accepted_path: list[np.ndarray] = []
    debt_path: list[np.ndarray] = []
    rollback_path: list[np.ndarray] = []
    messages = 0
    reused = 0
    iterations = 0
    for block in range(blocks):
        pre = samples[block, :pre_steps]
        post = samples[block, pre_steps:]
        local_pre = local_updates(parameters, pre, gain)
        shadow_pre = local_updates(shadow, pre, gain)
        donor = stale_snapshot(history, local_pre, delay)
        proposed_weights = identity.copy()
        accepted_weights = identity.copy()
        rolled_back = np.zeros(agents, dtype=bool)
        if block in decisions:
            proposal, proposed_weights, solved, _ = choose_continuous_weights(
                local_pre=local_pre,
                donor_snapshot=donor,
                fingerprint_samples=pre,
                debt=debt,
                drift_weight=drift_weight,
                variance_weight=variance_weight,
            )
            iterations += solved
            reused += pre_steps
            messages += fingerprint_message_units
            nonlocal_proposal = np.max(np.abs(proposed_weights - identity), axis=1) > 1e-8
            if np.any(nonlocal_proposal):
                messages += mixing_message_units
            validation_excess = np.mean(
                np.square(proposal[None, :] - post)
                - np.square(shadow_pre[None, :] - post), axis=0
            )
            candidate_post = local_updates(proposal, post, gain)
            shadow_post = local_updates(shadow_pre, post, gain)
            accept = validation_excess <= rollback_margin
            parameters = np.where(accept, candidate_post, shadow_post)
            accepted_weights = proposed_weights.copy()
            accepted_weights[~accept] = identity[~accept]
            rolled_back = ~accept
            debt = np.maximum(0.0, debt + validation_excess - safety_slack)
        else:
            parameters = local_updates(local_pre, post, gain)
            shadow_post = local_updates(shadow_pre, post, gain)
        shadow = shadow_post
        advance_history(history, parameters, delay)
        risks.append(float(np.mean(np.square(parameters - target[block]))))
        if block in decisions:
            proposals_path.append(proposed_weights.copy())
            accepted_path.append(accepted_weights.copy())
            debt_path.append(debt.copy())
            rollback_path.append(rolled_back.copy())
    risk_path = np.asarray(risks, dtype=float)
    return ContinuousTrajectory(
        auc_risk=float(np.mean(risk_path)),
        terminal_risk=float(risk_path[-1]),
        risk_path=risk_path,
        proposed_weights=np.asarray(proposals_path),
        accepted_weights=np.asarray(accepted_path),
        debt_path=np.asarray(debt_path),
        rollback_states=np.asarray(rollback_path),
        learning_transitions=blocks * steps_per_block,
        fingerprint_transitions=reused,
        message_units=messages,
        qp_iterations=iterations,
    )

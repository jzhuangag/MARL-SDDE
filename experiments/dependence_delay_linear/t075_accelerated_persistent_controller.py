"""T-074 persistent controller with warm-started accelerated simplex QPs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from experiments.dependence_delay_linear.t071_sampled_graph_controller import (
    advance_history, local_updates, stale_snapshot,
)
from experiments.dependence_delay_linear.t074_persistent_certificate_controller import (
    NoiseCertificate,
)
from experiments.dependence_delay_linear.t075_accelerated_simplex_qp import (
    solve_accelerated_qp,
)


@dataclass(frozen=True)
class AcceleratedPersistentTrajectory:
    auc_risk: float
    terminal_risk: float
    risk_path: np.ndarray
    accepted_weights: np.ndarray
    debt_path: np.ndarray
    rollback_states: np.ndarray
    rho_upper_path: np.ndarray
    effective_samples_path: np.ndarray
    qp_residual_path: np.ndarray
    learning_transitions: int
    fingerprint_transitions: int
    message_units: int
    qp_iterations: int


def simulate_accelerated_persistent_controller(
    *, observations: np.ndarray, targets: np.ndarray, initial_parameter: float,
    gain: float, delay: int, decision_blocks: Sequence[int], pre_steps: int,
    drift_weight: float, variance_weight: float, safety_slack: float,
    rollback_margin: float, certificate_delta: float, rho_cap: float,
    fingerprint_message_units: int = 1, mixing_message_units: int = 1,
) -> AcceleratedPersistentTrajectory:
    samples = np.asarray(observations, dtype=float)
    target = np.asarray(targets, dtype=float)
    if samples.ndim != 3 or target.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError("observations and targets have incompatible shapes")
    blocks, steps_per_block, agents = samples.shape
    if not 2 <= pre_steps < steps_per_block:
        raise ValueError("invalid dual-use split")
    decisions = set(int(value) for value in decision_blocks)
    parameters = np.repeat(float(initial_parameter), agents)
    shadow = parameters.copy()
    history = [parameters.copy() for _ in range(delay)]
    debt = np.zeros(agents, dtype=float)
    identity = np.eye(agents)
    warm_weights = identity.copy()
    certificate = NoiseCertificate.empty(agents)
    risks: list[float] = []
    accepted_path: list[np.ndarray] = []
    debt_path: list[np.ndarray] = []
    rollback_path: list[np.ndarray] = []
    rho_path: list[float] = []
    effective_path: list[float] = []
    residual_path: list[np.ndarray] = []
    messages = reused = iterations = 0
    for block in range(blocks):
        pre = samples[block, :pre_steps]
        post = samples[block, pre_steps:]
        local_pre = local_updates(parameters, pre, gain)
        shadow_pre = local_updates(shadow, pre, gain)
        donor = stale_snapshot(history, local_pre, delay)
        accepted_weights = identity.copy()
        rolled_back = np.zeros(agents, dtype=bool)
        if block in decisions:
            covariance, rho_upper, effective, _ = certificate.estimate_for_current_prefix(
                pre, delta=certificate_delta, rho_cap=rho_cap)
            target_hat = np.mean(pre, axis=0)
            proposals = np.empty(agents, dtype=float)
            proposed_weights = np.empty((agents, agents), dtype=float)
            residuals = np.empty(agents, dtype=float)
            for recipient in range(agents):
                models = donor.copy()
                models[recipient] = local_pre[recipient]
                weights, solved, residual = solve_accelerated_qp(
                    model_values=models, recipient_target=float(target_hat[recipient]),
                    covariance_of_mean=covariance, recipient=recipient,
                    debt=float(debt[recipient]), drift_weight=drift_weight,
                    variance_weight=variance_weight,
                    initial_weights=warm_weights[recipient])
                proposed_weights[recipient] = weights
                proposals[recipient] = float(weights @ models)
                residuals[recipient] = residual
                iterations += solved
            warm_weights = proposed_weights.copy()
            reused += pre_steps
            messages += fingerprint_message_units
            if np.any(np.max(np.abs(proposed_weights - identity), axis=1) > 1e-8):
                messages += mixing_message_units
            validation_excess = np.mean(
                np.square(proposals[None, :] - post)
                - np.square(shadow_pre[None, :] - post), axis=0)
            candidate_post = local_updates(proposals, post, gain)
            shadow_post = local_updates(shadow_pre, post, gain)
            accept = validation_excess <= rollback_margin
            parameters = np.where(accept, candidate_post, shadow_post)
            accepted_weights = proposed_weights.copy()
            accepted_weights[~accept] = identity[~accept]
            rolled_back = ~accept
            debt = np.maximum(0.0, debt + validation_excess - safety_slack)
            rho_path.append(rho_upper)
            effective_path.append(effective)
            residual_path.append(residuals)
        else:
            parameters = local_updates(local_pre, post, gain)
            shadow_post = local_updates(shadow_pre, post, gain)
        shadow = shadow_post
        advance_history(history, parameters, delay)
        risks.append(float(np.mean(np.square(parameters - target[block]))))
        certificate.update_completed_block(samples[block])
        if block in decisions:
            accepted_path.append(accepted_weights.copy())
            debt_path.append(debt.copy())
            rollback_path.append(rolled_back.copy())
    risk_path = np.asarray(risks, dtype=float)
    return AcceleratedPersistentTrajectory(
        auc_risk=float(np.mean(risk_path)), terminal_risk=float(risk_path[-1]),
        risk_path=risk_path, accepted_weights=np.asarray(accepted_path),
        debt_path=np.asarray(debt_path), rollback_states=np.asarray(rollback_path),
        rho_upper_path=np.asarray(rho_path), effective_samples_path=np.asarray(effective_path),
        qp_residual_path=np.asarray(residual_path),
        learning_transitions=blocks * steps_per_block, fingerprint_transitions=reused,
        message_units=messages, qp_iterations=iterations)

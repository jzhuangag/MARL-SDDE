"""Causal block-end continuous collaboration with delayed Lyapunov debt.

Every observation is first used by the local learner.  At registered block
ends, the completed trajectory fingerprint defines a covariance-aware simplex
QP.  The resulting mix affects only the next block, so no future observation
enters the decision.  A local shadow supplies observable delayed excess loss
for the virtual safety queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from experiments.dependence_delay_linear.t071_sampled_graph_controller import (
    advance_history,
    local_updates,
    stale_snapshot,
)
from experiments.dependence_delay_linear.t073_continuous_qp_controller import (
    solve_recipient_qp,
)
from experiments.dependence_delay_linear.t074_persistent_certificate_controller import (
    NoiseCertificate,
)


@dataclass(frozen=True)
class EndBlockTrajectory:
    auc_risk: float
    terminal_risk: float
    risk_path: np.ndarray
    accepted_weights: np.ndarray
    debt_path: np.ndarray
    rho_upper_path: np.ndarray
    effective_samples_path: np.ndarray
    learning_transitions: int
    extra_probe_transitions: int
    message_units: int
    qp_iterations: int


def _validate_fixed_weights(weights: np.ndarray, agents: int) -> np.ndarray:
    matrix = np.asarray(weights, dtype=float)
    if matrix.shape != (agents, agents) or not np.all(np.isfinite(matrix)):
        raise ValueError("fixed weights have incompatible shape")
    if np.min(matrix) < -1e-10 or not np.allclose(
        np.sum(matrix, axis=1), 1.0, atol=1e-9
    ):
        raise ValueError("fixed weights must be row stochastic")
    matrix = np.maximum(matrix, 0.0)
    return matrix / np.sum(matrix, axis=1, keepdims=True)


def simulate_end_block_controller(
    *,
    observations: np.ndarray,
    targets: np.ndarray,
    initial_parameter: float,
    gain: float,
    delay: int,
    decision_blocks: Sequence[int],
    drift_weight: float,
    variance_weight: float,
    safety_slack: float,
    certificate_delta: float,
    rho_cap: float,
    fixed_weights: np.ndarray | None = None,
    fingerprint_message_units: int = 1,
    mixing_message_units: int = 1,
) -> EndBlockTrajectory:
    """Run the observable controller or a fixed continuous block-end graph."""

    samples = np.asarray(observations, dtype=float)
    target = np.asarray(targets, dtype=float)
    if samples.ndim != 3 or target.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError("observations and targets have incompatible shapes")
    if delay < 0 or int(delay) != delay:
        raise ValueError("delay must be a nonnegative integer")
    if min(drift_weight, variance_weight, safety_slack) < 0.0:
        raise ValueError("controller weights must be nonnegative")
    blocks, steps_per_block, agents = samples.shape
    fixed = None if fixed_weights is None else _validate_fixed_weights(fixed_weights, agents)
    decisions = set(int(value) for value in decision_blocks)
    if any(value < 0 or value >= blocks for value in decisions):
        raise ValueError("decision block lies outside the trajectory")

    parameters = np.repeat(float(initial_parameter), agents)
    shadow = parameters.copy()
    history = [parameters.copy() for _ in range(delay)]
    certificate = NoiseCertificate.empty(agents)
    debt = np.zeros(agents, dtype=float)
    identity = np.eye(agents)
    risks: list[float] = []
    weights_path: list[np.ndarray] = []
    debt_path: list[np.ndarray] = []
    rho_path: list[float] = []
    effective_path: list[float] = []
    messages = 0
    iterations = 0

    for block in range(blocks):
        block_samples = samples[block]
        local = local_updates(parameters, block_samples, gain)
        local_shadow = local_updates(shadow, block_samples, gain)
        target_hat = np.mean(block_samples, axis=0)

        # This paired excess is observed before the next collaboration choice.
        # It charges previous mixing through a predictable virtual queue.
        excess = np.square(local - target_hat) - np.square(local_shadow - target_hat)
        debt = np.maximum(0.0, debt + excess - safety_slack)
        certificate.update_completed_block(block_samples)

        selected = identity.copy()
        if block in decisions:
            donor = stale_snapshot(history, local, delay)
            if fixed is not None:
                selected = fixed.copy()
            else:
                covariance, rho_upper, effective, _ = certificate.estimate_for_current_prefix(
                    block_samples, delta=certificate_delta, rho_cap=rho_cap
                )
                for recipient in range(agents):
                    models = donor.copy()
                    models[recipient] = local[recipient]
                    selected[recipient], solved = solve_recipient_qp(
                        model_values=models,
                        recipient_target=float(target_hat[recipient]),
                        covariance_of_mean=covariance,
                        recipient=recipient,
                        debt=float(debt[recipient]),
                        drift_weight=drift_weight,
                        variance_weight=variance_weight,
                    )
                    iterations += solved
                rho_path.append(rho_upper)
                effective_path.append(effective)
                messages += int(fingerprint_message_units)
            if np.any(np.max(np.abs(selected - identity), axis=1) > 1e-8):
                messages += int(mixing_message_units)
            mixed = np.empty(agents, dtype=float)
            for recipient in range(agents):
                models = donor.copy()
                models[recipient] = local[recipient]
                mixed[recipient] = float(selected[recipient] @ models)
            parameters = mixed
            weights_path.append(selected.copy())
            debt_path.append(debt.copy())
        else:
            parameters = local

        shadow = local_shadow
        advance_history(history, parameters, delay)
        risks.append(float(np.mean(np.square(parameters - target[block]))))

    risk_path = np.asarray(risks, dtype=float)
    return EndBlockTrajectory(
        auc_risk=float(np.mean(risk_path)),
        terminal_risk=float(risk_path[-1]),
        risk_path=risk_path,
        accepted_weights=np.asarray(weights_path),
        debt_path=np.asarray(debt_path),
        rho_upper_path=np.asarray(rho_path),
        effective_samples_path=np.asarray(effective_path),
        learning_transitions=blocks * steps_per_block,
        extra_probe_transitions=0,
        message_units=messages,
        qp_iterations=iterations,
    )

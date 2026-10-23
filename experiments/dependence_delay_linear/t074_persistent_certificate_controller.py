"""Persistent predictable mixing certificate for continuous collaboration QPs."""

from __future__ import annotations

from dataclasses import dataclass
import math
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


@dataclass
class NoiseCertificate:
    agents: int
    scatter: np.ndarray
    degrees: int = 0
    lag_numerator: float = 0.0
    lag_denominator: float = 0.0
    lag_pairs: int = 0

    @classmethod
    def empty(cls, agents: int) -> "NoiseCertificate":
        return cls(agents=agents, scatter=np.zeros((agents, agents), dtype=float))

    @staticmethod
    def _summary(samples: np.ndarray) -> tuple[np.ndarray, int, float, float, int]:
        values = np.asarray(samples, dtype=float)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("a certificate block needs at least two observations")
        centered = values - np.mean(values, axis=0, keepdims=True)
        scatter = centered.T @ centered
        numerator = float(np.sum(centered[1:] * centered[:-1]))
        denominator = float(np.sum(np.square(centered[:-1])))
        return scatter, values.shape[0] - 1, numerator, denominator, values.shape[0] - 1

    def update_completed_block(self, samples: np.ndarray) -> None:
        scatter, degrees, numerator, denominator, pairs = self._summary(samples)
        if scatter.shape != (self.agents, self.agents):
            raise ValueError("certificate agent count mismatch")
        self.scatter += scatter
        self.degrees += degrees
        self.lag_numerator += numerator
        self.lag_denominator += denominator
        self.lag_pairs += pairs

    def estimate_for_current_prefix(
        self, samples: np.ndarray, *, delta: float, rho_cap: float
    ) -> tuple[np.ndarray, float, float, int]:
        """Return covariance-of-current-mean using only predictable evidence."""

        if not 0.0 < delta < 1.0 or not 0.0 < rho_cap < 1.0:
            raise ValueError("invalid certificate constants")
        scatter, degrees, numerator, denominator, pairs = self._summary(samples)
        total_degrees = self.degrees + degrees
        covariance = (self.scatter + scatter) / max(total_degrees, 1)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        total_denominator = self.lag_denominator + denominator
        rho_hat = 0.0 if total_denominator <= 1e-15 else max(
            0.0, (self.lag_numerator + numerator) / total_denominator
        )
        total_pairs = self.lag_pairs + pairs
        radius = math.sqrt(math.log(2.0 / delta) / (2.0 * max(total_pairs, 1)))
        rho_upper = float(min(rho_cap, rho_hat + radius))
        current_count = np.asarray(samples).shape[0]
        effective = max(1.0, current_count * (1.0 - rho_upper) / (1.0 + rho_upper))
        return covariance / effective, rho_upper, effective, total_pairs


@dataclass(frozen=True)
class PersistentTrajectory:
    auc_risk: float
    terminal_risk: float
    risk_path: np.ndarray
    accepted_weights: np.ndarray
    debt_path: np.ndarray
    rollback_states: np.ndarray
    rho_upper_path: np.ndarray
    effective_samples_path: np.ndarray
    learning_transitions: int
    fingerprint_transitions: int
    message_units: int
    qp_iterations: int


def simulate_persistent_qp_controller(
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
    certificate_delta: float,
    rho_cap: float,
    fingerprint_message_units: int = 1,
    mixing_message_units: int = 1,
) -> PersistentTrajectory:
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
    certificate = NoiseCertificate.empty(agents)
    risks: list[float] = []
    accepted_path: list[np.ndarray] = []
    debt_path: list[np.ndarray] = []
    rollback_path: list[np.ndarray] = []
    rho_path: list[float] = []
    effective_path: list[float] = []
    messages = 0
    reused = 0
    iterations = 0
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
                pre, delta=certificate_delta, rho_cap=rho_cap
            )
            target_hat = np.mean(pre, axis=0)
            proposals = np.empty(agents, dtype=float)
            proposed_weights = np.empty((agents, agents), dtype=float)
            for recipient in range(agents):
                models = donor.copy()
                models[recipient] = local_pre[recipient]
                weights, solved = solve_recipient_qp(
                    model_values=models,
                    recipient_target=float(target_hat[recipient]),
                    covariance_of_mean=covariance,
                    recipient=recipient,
                    debt=float(debt[recipient]),
                    drift_weight=drift_weight,
                    variance_weight=variance_weight,
                )
                proposed_weights[recipient] = weights
                proposals[recipient] = float(weights @ models)
                iterations += solved
            reused += pre_steps
            messages += fingerprint_message_units
            if np.any(np.max(np.abs(proposed_weights - identity), axis=1) > 1e-8):
                messages += mixing_message_units
            validation_excess = np.mean(
                np.square(proposals[None, :] - post)
                - np.square(shadow_pre[None, :] - post), axis=0
            )
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
    return PersistentTrajectory(
        auc_risk=float(np.mean(risk_path)), terminal_risk=float(risk_path[-1]),
        risk_path=risk_path, accepted_weights=np.asarray(accepted_path),
        debt_path=np.asarray(debt_path), rollback_states=np.asarray(rollback_path),
        rho_upper_path=np.asarray(rho_path), effective_samples_path=np.asarray(effective_path),
        learning_transitions=blocks * steps_per_block, fingerprint_transitions=reused,
        message_units=messages, qp_iterations=iterations,
    )

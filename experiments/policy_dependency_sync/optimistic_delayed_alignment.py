"""Predictable ridge-UCB scores for delayed local alignment feedback."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Hashable, Mapping

import numpy as np

from .joint_factor_lyapunov import (
    JointFactorChoice,
    choose_joint_factor_action_variable_bounds,
)
from .core_factor_lyapunov import choose_core_factor_action_variable_bounds


@dataclass(frozen=True)
class AlignmentConfidence:
    mean: float
    radius: float
    upper: float
    leverage: float


class DelayedRidgeAlignment:
    """Ridge state updated only when a launched packet actually returns."""

    def __init__(
        self,
        *,
        dimension: int,
        ridge: float,
        noise_subgaussian: float,
        parameter_norm_upper: float,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        if noise_subgaussian < 0.0 or parameter_norm_upper < 0.0:
            raise ValueError("noise and parameter bounds must be nonnegative")
        self.dimension = int(dimension)
        self.ridge = float(ridge)
        self.noise_subgaussian = float(noise_subgaussian)
        self.parameter_norm_upper = float(parameter_norm_upper)
        self.gram = self.ridge * np.eye(self.dimension)
        self.inverse_gram = np.eye(self.dimension) / self.ridge
        self.response = np.zeros(self.dimension, dtype=float)
        self.historical_bias_squared = 0.0
        self.log_determinant_ratio = 0.0
        self.pending: dict[int, np.ndarray] = {}
        self._next_packet_id = 0

    def launch(self, selected_context: np.ndarray) -> int:
        """Record a predictable selected context without updating the fit."""

        context = self._validate_context(selected_context)
        packet_id = self._next_packet_id
        self._next_packet_id += 1
        self.pending[packet_id] = context.copy()
        return packet_id

    def receive(
        self,
        packet_id: int,
        feedback: float,
        *,
        approximation_bound: float = 0.0,
    ) -> None:
        """Add one completed observation and its declared model-bias bound."""

        if packet_id not in self.pending:
            raise KeyError(f"unknown or already received packet {packet_id}")
        if not np.isfinite(feedback):
            raise ValueError("feedback must be finite")
        if not np.isfinite(approximation_bound) or approximation_bound < 0.0:
            raise ValueError("approximation bound must be finite and nonnegative")
        context = self.pending.pop(packet_id)
        inverse_context = self.inverse_gram @ context
        determinant_multiplier = 1.0 + float(context @ inverse_context)
        if determinant_multiplier <= 0.0 or not np.isfinite(determinant_multiplier):
            raise ValueError("invalid rank-one covariance update")
        self.inverse_gram -= np.outer(inverse_context, inverse_context) / (
            determinant_multiplier
        )
        self.inverse_gram = 0.5 * (self.inverse_gram + self.inverse_gram.T)
        self.log_determinant_ratio += log(determinant_multiplier)
        self.gram += np.outer(context, context)
        self.response += context * float(feedback)
        self.historical_bias_squared += float(approximation_bound) ** 2

    def confidence(
        self,
        contexts: Mapping[Hashable, np.ndarray],
        *,
        current_approximation_bound: Mapping[Hashable, float],
        delta: float,
        alignment_upper: float | None = None,
    ) -> dict[Hashable, AlignmentConfidence]:
        """Return simultaneous linear confidence values for local candidates."""

        if not contexts or set(contexts) != set(current_approximation_bound):
            raise ValueError("context and approximation mappings must share keys")
        if not np.isfinite(delta) or not 0.0 < delta < 1.0:
            raise ValueError("delta must lie strictly between zero and one")
        if alignment_upper is not None and not np.isfinite(alignment_upper):
            raise ValueError("alignment upper bound must be finite when supplied")
        information = 0.5 * self.log_determinant_ratio
        beta = (
            self.noise_subgaussian
            * sqrt(max(0.0, 2.0 * (information + log(1.0 / delta))))
            + sqrt(self.ridge) * self.parameter_norm_upper
            + sqrt(self.historical_bias_squared)
        )
        inverse_response = self.inverse_gram @ self.response
        results: dict[Hashable, AlignmentConfidence] = {}
        for action, raw_context in contexts.items():
            context = self._validate_context(raw_context)
            current_bias = float(current_approximation_bound[action])
            if not np.isfinite(current_bias) or current_bias < 0.0:
                raise ValueError("current approximation bounds must be nonnegative")
            solved = self.inverse_gram @ context
            leverage = sqrt(max(0.0, float(context @ solved)))
            mean = float(context @ inverse_response)
            radius = float(beta * leverage + current_bias)
            upper = mean + radius
            if alignment_upper is not None:
                upper = min(float(alignment_upper), upper)
            results[action] = AlignmentConfidence(
                mean=mean,
                radius=radius,
                upper=upper,
                leverage=leverage,
            )
        return results

    def _validate_context(self, context: np.ndarray) -> np.ndarray:
        array = np.asarray(context, dtype=float)
        if array.shape != (self.dimension,) or not np.all(np.isfinite(array)):
            raise ValueError(f"context must be a finite vector of length {self.dimension}")
        return array


def choose_optimistic_joint_factor_action(
    *,
    confidence_by_action: Mapping[Hashable, AlignmentConfidence],
    reset_benefit_by_action: Mapping[Hashable, float],
    communication_cost_by_action: Mapping[Hashable, float],
    gradient_norm_upper_by_action: Mapping[Hashable, float],
    communication_queue: float,
    learning_weight: float,
    learning_smoothness: float,
    receipt_motion_upper: float,
    receipt_cache_linear_upper: float,
    outgoing_cache_weight: float,
    maximum_packet_weight: float,
) -> JointFactorChoice:
    """Insert optimistic alignments into the exact joint Lyapunov minimizer."""

    return choose_joint_factor_action_variable_bounds(
        alignment_lower_by_action={
            action: confidence.upper
            for action, confidence in confidence_by_action.items()
        },
        reset_benefit_by_action=reset_benefit_by_action,
        communication_cost_by_action=communication_cost_by_action,
        gradient_norm_upper_by_action=gradient_norm_upper_by_action,
        communication_queue=communication_queue,
        learning_weight=learning_weight,
        learning_smoothness=learning_smoothness,
        receipt_motion_upper=receipt_motion_upper,
        receipt_cache_linear_upper=receipt_cache_linear_upper,
        outgoing_cache_weight=outgoing_cache_weight,
        maximum_packet_weight=maximum_packet_weight,
    )


def choose_optimistic_core_factor_action(
    *,
    confidence_by_action: Mapping[Hashable, AlignmentConfidence],
    communication_cost_by_action: Mapping[Hashable, float],
    gradient_norm_upper_by_action: Mapping[Hashable, float],
    communication_queue: float,
    learning_weight: float,
    learning_smoothness: float,
    receipt_motion_upper: float,
    maximum_packet_weight: float,
) -> JointFactorChoice:
    """Insert optimism into the topology-robust core Lyapunov controller.

    ``AlignmentConfidence.upper`` is deliberately optimistic, not a safety
    lower bound.  The selected-action regret lemma pays at most twice its
    radius; this wrapper must therefore be used with that lemma rather than
    with a claim of per-launch certified descent.
    """

    return choose_core_factor_action_variable_bounds(
        alignment_lower_by_action={
            action: confidence.upper
            for action, confidence in confidence_by_action.items()
        },
        communication_cost_by_action=communication_cost_by_action,
        gradient_norm_upper_by_action=gradient_norm_upper_by_action,
        communication_queue=communication_queue,
        learning_weight=learning_weight,
        learning_smoothness=learning_smoothness,
        receipt_motion_upper=receipt_motion_upper,
        maximum_packet_weight=maximum_packet_weight,
    )


def delayed_leverage_sum_bound(
    *,
    launches: int,
    maximum_feedback_delay: int,
    dimension: int,
    context_norm_upper: float,
    ridge: float,
) -> float:
    """Return the residue-class elliptical-potential bound from Theorem 2."""

    if launches < 0 or maximum_feedback_delay < 0 or dimension <= 0:
        raise ValueError("launches/delay must be nonnegative and dimension positive")
    if context_norm_upper < 0.0 or ridge <= 0.0:
        raise ValueError("context bound must be nonnegative and ridge positive")
    if ridge + 1e-15 < context_norm_upper**2:
        raise ValueError("the stated bound requires ridge >= context_norm_upper^2")
    return float(
        sqrt(
            2.0
            * launches
            * (maximum_feedback_delay + 1)
            * dimension
            * log(
                1.0
                + launches * context_norm_upper**2 / (ridge * dimension)
            )
        )
    )

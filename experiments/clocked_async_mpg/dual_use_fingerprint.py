"""Directional game-geometry fingerprints from an already-paid EG oracle."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DirectionalGeometryFingerprint:
    informative: bool
    gradient_energy: float
    jacobian_action_energy_ratio: float
    symmetric_alignment: float
    rotational_residual: float


@dataclass(frozen=True)
class BinaryGeometryBelief:
    rotation_probability: float


def predict_binary_geometry(
    belief: BinaryGeometryBelief,
    *,
    potential_to_rotation: float,
    rotation_to_potential: float,
) -> BinaryGeometryBelief:
    """One causal hidden-phase prediction without a new oracle."""

    values = (
        belief.rotation_probability,
        potential_to_rotation,
        rotation_to_potential,
    )
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("belief and transition probabilities must lie in [0, 1]")
    predicted = (
        belief.rotation_probability * (1.0 - rotation_to_potential)
        + (1.0 - belief.rotation_probability) * potential_to_rotation
    )
    return BinaryGeometryBelief(rotation_probability=predicted)


def update_binary_geometry(
    predicted: BinaryGeometryBelief,
    *,
    observed_score: float,
    observation_standard_deviation: float,
) -> BinaryGeometryBelief:
    """Bayes-update a predicted phase using a paid directional fingerprint.

    The development model uses score means -1 for potential and +1 for
    rotation.  This likelihood is an explicit modeling assumption, not a
    Markov-noise confidence theorem.
    """

    probability = predicted.rotation_probability
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("predicted probability must lie in [0, 1]")
    if not math.isfinite(observed_score):
        raise ValueError("observed score must be finite")
    if (
        not math.isfinite(observation_standard_deviation)
        or observation_standard_deviation <= 0.0
    ):
        raise ValueError("observation standard deviation must be positive")
    variance = observation_standard_deviation**2
    log_rotation = (
        math.log(max(probability, np.finfo(float).tiny))
        - 0.5 * (observed_score - 1.0) ** 2 / variance
    )
    log_potential = (
        math.log(max(1.0 - probability, np.finfo(float).tiny))
        - 0.5 * (observed_score + 1.0) ** 2 / variance
    )
    maximum = max(log_rotation, log_potential)
    rotation_weight = math.exp(log_rotation - maximum)
    potential_weight = math.exp(log_potential - maximum)
    posterior = rotation_weight / (rotation_weight + potential_weight)
    return BinaryGeometryBelief(rotation_probability=posterior)


def expected_binary_log_gain(
    belief: BinaryGeometryBelief,
    *,
    potential_log_gain: float,
    rotational_log_gain: float,
) -> float:
    """Posterior mean log-drift value for the next optimism decision."""

    values = (
        belief.rotation_probability,
        potential_log_gain,
        rotational_log_gain,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("belief and log gains must be finite")
    if not 0.0 <= belief.rotation_probability <= 1.0:
        raise ValueError("belief probability must lie in [0, 1]")
    return (
        (1.0 - belief.rotation_probability) * potential_log_gain
        + belief.rotation_probability * rotational_log_gain
    )


def directional_geometry_fingerprint(
    current_gradient: np.ndarray,
    lookahead_gradient: np.ndarray,
    *,
    lookahead_step: float,
    minimum_gradient_energy: float = 0.0,
) -> DirectionalGeometryFingerprint:
    """Estimate symmetric/rotational geometry along the current gradient.

    For a linear pseudo-gradient ``F(x)=Ax`` and the full virtual lookahead
    ``x_plus=x-eta F(x)``, the already-paid pair obeys

    ``(F(x)-F(x_plus))/eta = A F(x)``.

    The returned alignment is the Rayleigh quotient of the symmetric part of
    ``A`` along ``F(x)``.  The residual is the norm of the component of
    ``A F(x)`` orthogonal to ``F(x)``.  No matrix is formed.
    """

    current = np.asarray(current_gradient, dtype=float)
    lookahead = np.asarray(lookahead_gradient, dtype=float)
    if (
        current.ndim != 1
        or lookahead.shape != current.shape
        or current.size == 0
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(lookahead))
    ):
        raise ValueError("gradients must be finite matching nonempty vectors")
    if not math.isfinite(lookahead_step) or lookahead_step <= 0.0:
        raise ValueError("lookahead_step must be finite and positive")
    if (
        not math.isfinite(minimum_gradient_energy)
        or minimum_gradient_energy < 0.0
    ):
        raise ValueError("minimum_gradient_energy must be finite and nonnegative")
    energy = float(current @ current)
    if energy <= minimum_gradient_energy:
        return DirectionalGeometryFingerprint(
            informative=False,
            gradient_energy=energy,
            jacobian_action_energy_ratio=math.nan,
            symmetric_alignment=math.nan,
            rotational_residual=math.nan,
        )
    action = (current - lookahead) / lookahead_step
    alignment = float(current @ action / energy)
    action_ratio = float(action @ action / energy)
    rotational_square = max(0.0, action_ratio - alignment * alignment)
    return DirectionalGeometryFingerprint(
        informative=True,
        gradient_energy=energy,
        jacobian_action_energy_ratio=action_ratio,
        symmetric_alignment=alignment,
        rotational_residual=math.sqrt(rotational_square),
    )

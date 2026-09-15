"""Algebra for the observable affine packet-reference estimator."""

from __future__ import annotations

from math import sqrt
from typing import Sequence

import numpy as np


def nlms_expected_squared_error(
    error: Sequence[float],
    row: Sequence[float],
    gain: float,
    noise_variance: float,
) -> float:
    """Conditional expected squared error after one normalized-LMS update."""

    error_array = np.asarray(error, dtype=float)
    row_array = np.asarray(row, dtype=float)
    denominator = float(row_array @ row_array)
    if error_array.shape != row_array.shape or denominator <= 0.0:
        raise ValueError("error and nonzero row must have equal shapes")
    if not 0.0 < gain < 2.0 or noise_variance < 0.0:
        raise ValueError("invalid gain or variance")
    projected = float(row_array @ error_array)
    mean_next = error_array - gain * row_array * projected / denominator
    return float(
        mean_next @ mean_next + gain * gain * noise_variance / denominator
    )


def cyclic_normalized_pe_lower_bound(
    agents: int, strong_convexity: float, coupling: float
) -> float:
    """Conservative six-owner normalized Gram lower bound.

    Each owner row is strictly diagonally dominant by ``strong_convexity``;
    off-diagonal absolute coefficients sum to ``coupling``.
    """

    if agents <= 0 or strong_convexity <= 0.0 or coupling < 0.0:
        raise ValueError("invalid model constants")
    row_norm_upper = sqrt((strong_convexity + coupling) ** 2 + coupling**2)
    return float(
        strong_convexity**2 / (agents * row_norm_upper * row_norm_upper)
    )


def noiseless_block_contraction(
    gain: float, block_length: int, normalized_gram_lower: float
) -> float:
    """A conservative decrease coefficient for one persistently exciting block."""

    if not 0.0 < gain < 2.0 or block_length <= 0:
        raise ValueError("invalid gain or block length")
    if not 0.0 < normalized_gram_lower <= block_length:
        raise ValueError("invalid Gram lower bound")
    return float(
        gain
        * (2.0 - gain)
        * normalized_gram_lower
        / (2.0 * (1.0 + gain * gain * block_length * block_length))
    )


def signed_score_reference_error_bound(
    *,
    reference_error_norm: float,
    current_row_norm: float,
    packet_row_norm: float,
    current_gradient_bound: float,
    packet_mean_bound: float,
    step: float,
    smoothness: float,
    receipt_motion_bound: float,
) -> float:
    """Uniform drift-score perturbation induced by a reference error."""

    values = (
        reference_error_norm,
        current_row_norm,
        packet_row_norm,
        current_gradient_bound,
        packet_mean_bound,
        step,
        smoothness,
        receipt_motion_bound,
    )
    if any(value < 0.0 for value in values):
        raise ValueError("score-bound inputs must be nonnegative")
    delta_current = current_row_norm * reference_error_norm
    delta_packet = packet_row_norm * reference_error_norm
    bilinear = (
        current_gradient_bound * delta_packet
        + packet_mean_bound * delta_current
        + delta_current * delta_packet
    )
    quadratic = delta_packet * (2.0 * packet_mean_bound + delta_packet)
    return float(
        step * bilinear
        + 0.5 * smoothness * step * step * quadratic
        + step * receipt_motion_bound * delta_packet
    )

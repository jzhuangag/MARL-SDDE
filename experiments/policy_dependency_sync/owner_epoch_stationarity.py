"""Exact epoch-motion conversion from selected blocks to full stationarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class OwnerEpochLowerBound:
    selected_weighted_norm_squared: float
    full_gradient_term: float
    motion_remainder: float
    lower_bound: float


def owner_epoch_stationarity_lower_bound(
    *,
    epoch_start_block_gradients: Sequence[np.ndarray],
    selected_launch_block_gradients: Sequence[np.ndarray],
    block_smoothness: Sequence[float],
    launch_path_motion_upper: Sequence[float],
    stationarity_weights: Sequence[float],
) -> OwnerEpochLowerBound:
    """Evaluate the permutation-epoch stationarity inequality.

    Entry ``i`` in ``selected_launch_block_gradients`` is the gradient of
    block ``i`` at the launch where owner ``i`` appears.  The order inside the
    epoch is irrelevant as long as every owner appears exactly once.
    """

    count = len(epoch_start_block_gradients)
    collections = (
        selected_launch_block_gradients,
        block_smoothness,
        launch_path_motion_upper,
        stationarity_weights,
    )
    if count == 0 or any(len(values) != count for values in collections):
        raise ValueError("epoch arrays must have the same positive length")
    start = [np.asarray(value, dtype=float).ravel() for value in epoch_start_block_gradients]
    selected = [
        np.asarray(value, dtype=float).ravel()
        for value in selected_launch_block_gradients
    ]
    if any(
        left.shape != right.shape
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        for left, right in zip(start, selected)
    ):
        raise ValueError("paired block gradients must be equally shaped and finite")
    smoothness = np.asarray(block_smoothness, dtype=float)
    motion = np.asarray(launch_path_motion_upper, dtype=float)
    weights = np.asarray(stationarity_weights, dtype=float)
    if (
        not np.all(np.isfinite(smoothness))
        or not np.all(np.isfinite(motion))
        or not np.all(np.isfinite(weights))
        or np.min(smoothness) < 0.0
        or np.min(motion) < 0.0
        or np.min(weights) <= 0.0
    ):
        raise ValueError("epoch bounds must be finite with positive weights")
    kappa_min = float(np.min(weights))
    kappa_max = float(np.max(weights))
    selected_value = float(
        sum(weight * float(gradient @ gradient) for weight, gradient in zip(weights, selected))
    )
    full_term = float(
        0.5 * kappa_min * sum(float(gradient @ gradient) for gradient in start)
    )
    remainder = float(kappa_max * np.sum((smoothness * motion) ** 2))
    return OwnerEpochLowerBound(
        selected_weighted_norm_squared=selected_value,
        full_gradient_term=full_term,
        motion_remainder=remainder,
        lower_bound=full_term - remainder,
    )

"""Development-only structured pair-factor sufficiency audit.

This module never supplies counterfactual labels to an online controller.  It
uses them only to test whether a compatible bilinear pair-factor class can
represent held-out one-edge conditional-mean differences before implementing
a critic trained from selected completed packets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .pursuit_delayed_alignment_interface import (
    FEATURE_NAMES,
    AlignmentLaunch,
)


PAIR_FEATURE_NAMES = (
    "intercept",
    "owner_evader_density",
    "owner_pursuer_density",
    "donor_evader_density",
    "distance",
    "cache_age",
    "policy_tv",
) + tuple(
    f"{role}_evader_{action}"
    for role in ("owner", "donor")
    for action in ("left", "right", "up", "down", "stay")
) + tuple(f"owner_identity_{index}" for index in range(8)) + tuple(
    f"donor_identity_{index}" for index in range(8)
) + (
    "event_sine",
    "event_cosine",
)

_PAIR_FEATURE_INDEX = tuple(FEATURE_NAMES.index(name) for name in PAIR_FEATURE_NAMES)


@dataclass(frozen=True)
class PairFactorDesign:
    matrix: np.ndarray
    response: np.ndarray
    seed: np.ndarray
    packet_id: np.ndarray
    donor: np.ndarray


@dataclass(frozen=True)
class PairFactorRidge:
    weight: np.ndarray
    scale: np.ndarray
    ridge: float


def pair_feature_vector(context: Sequence[float]) -> np.ndarray:
    """Extract launch-measurable state features, excluding the reference."""

    values = np.asarray(context, dtype=float)
    if values.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError("invalid Pursuit launch context")
    return values[np.asarray(_PAIR_FEATURE_INDEX, dtype=int)]


def pair_factor_design_row(
    *,
    owner_probability_direction: Sequence[float],
    donor_probability_difference: Sequence[float],
    pair_features: Sequence[float],
) -> np.ndarray:
    """Return coefficients for ``d.T @ Q(phi) @ delta``."""

    direction = np.asarray(owner_probability_direction, dtype=float)
    difference = np.asarray(donor_probability_difference, dtype=float)
    features = np.asarray(pair_features, dtype=float)
    if direction.shape != (5,) or difference.shape != (5,):
        raise ValueError("Pursuit factor rows require five-action vectors")
    if features.ndim != 1 or features.size == 0:
        raise ValueError("pair features must be a nonempty vector")
    if not all(
        np.all(np.isfinite(value)) for value in (direction, difference, features)
    ):
        raise ValueError("pair-factor inputs must be finite")
    return np.einsum("i,j,k->ijk", direction, difference, features).ravel()


def build_counterfactual_pair_design(
    rows: Sequence[AlignmentLaunch],
) -> PairFactorDesign:
    """Build a privileged offline design from conditional-mean edge effects."""

    matrix: list[np.ndarray] = []
    response: list[float] = []
    seeds: list[int] = []
    packet_ids: list[int] = []
    donors: list[int] = []
    for row in rows:
        if not row.reference_available:
            continue
        contexts = dict(row.candidate_contexts)
        differences = dict(row.candidate_probability_difference)
        means = dict(row.candidate_factor_mean_alignment)
        if None not in means:
            raise ValueError("counterfactual audit is missing the null candidate")
        for donor in sorted(value for value in means if value is not None):
            matrix.append(
                pair_factor_design_row(
                    owner_probability_direction=row.owner_probability_direction,
                    donor_probability_difference=differences[donor],
                    pair_features=pair_feature_vector(contexts[donor]),
                )
            )
            response.append(float(means[donor] - means[None]))
            seeds.append(int(row.seed))
            packet_ids.append(int(row.packet_id))
            donors.append(int(donor))
    if not matrix:
        raise ValueError("no reference-available nonnull candidates")
    return PairFactorDesign(
        matrix=np.vstack(matrix),
        response=np.asarray(response, dtype=float),
        seed=np.asarray(seeds, dtype=int),
        packet_id=np.asarray(packet_ids, dtype=int),
        donor=np.asarray(donors, dtype=int),
    )


def fit_pair_factor_ridge(
    matrix: np.ndarray,
    response: np.ndarray,
    *,
    ridge: float,
) -> PairFactorRidge:
    """Fit a zero-preserving ridge model for edge-value differences."""

    design = np.asarray(matrix, dtype=float)
    target = np.asarray(response, dtype=float)
    if (
        design.ndim != 2
        or target.shape != (design.shape[0],)
        or design.shape[0] == 0
        or ridge <= 0.0
    ):
        raise ValueError("invalid pair-factor ridge problem")
    scale = np.sqrt(np.mean(design * design, axis=0))
    scale[scale < 1e-12] = 1.0
    standardized = design / scale
    if standardized.shape[0] < standardized.shape[1]:
        gram = standardized @ standardized.T
        dual = np.linalg.solve(
            gram + float(ridge) * np.eye(gram.shape[0]),
            target,
        )
        standardized_weight = standardized.T @ dual
    else:
        standardized_weight = np.linalg.solve(
            standardized.T @ standardized
            + float(ridge) * np.eye(standardized.shape[1]),
            standardized.T @ target,
        )
    return PairFactorRidge(
        weight=standardized_weight / scale,
        scale=scale,
        ridge=float(ridge),
    )


def predict_pair_factor_ridge(
    model: PairFactorRidge,
    matrix: np.ndarray,
) -> np.ndarray:
    design = np.asarray(matrix, dtype=float)
    if design.ndim != 2 or design.shape[1] != model.weight.size:
        raise ValueError("pair-factor prediction dimension mismatch")
    return design @ model.weight


def heldout_pair_metrics(
    *,
    truth: np.ndarray,
    prediction: np.ndarray,
    packet_id: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate effect prediction and per-launch refresh ranking."""

    actual = np.asarray(truth, dtype=float)
    forecast = np.asarray(prediction, dtype=float)
    packets = np.asarray(packet_id, dtype=int)
    if actual.shape != forecast.shape or actual.shape != packets.shape:
        raise ValueError("held-out arrays must have identical shapes")
    centered = actual - float(np.mean(actual))
    denominator = float(centered @ centered)
    r_squared = (
        float("nan")
        if denominator <= 1e-15
        else 1.0 - float(np.sum((forecast - actual) ** 2)) / denominator
    )
    signs = np.sign(actual)
    nonzero = np.abs(actual) > 1e-10
    sign_accuracy = (
        float("nan")
        if not np.any(nonzero)
        else float(np.mean(np.sign(forecast[nonzero]) == signs[nonzero]))
    )
    correct_best = 0
    strict_oracle_packets = 0
    for packet in np.unique(packets):
        mask = packets == packet
        actual_with_null = np.concatenate(([0.0], actual[mask]))
        forecast_with_null = np.concatenate(([0.0], forecast[mask]))
        actual_best = int(np.argmax(actual_with_null))
        forecast_best = int(np.argmax(forecast_with_null))
        correct_best += int(actual_best == forecast_best)
        strict_oracle_packets += int(float(np.max(actual[mask])) > 1e-10)
    return {
        "rows": int(actual.size),
        "packets": int(np.unique(packets).size),
        "r_squared": float(r_squared),
        "mean_absolute_error": float(np.mean(np.abs(forecast - actual))),
        "target_standard_deviation": float(np.std(actual)),
        "sign_accuracy_nonzero": float(sign_accuracy),
        "best_action_accuracy": float(
            correct_best / max(1, int(np.unique(packets).size))
        ),
        "strict_oracle_packet_fraction": float(
            strict_oracle_packets / max(1, int(np.unique(packets).size))
        ),
    }

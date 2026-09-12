"""Geometry fingerprints from consecutive mandatory gradient evaluations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PassiveSecantFingerprint:
    informative: bool
    displacement_energy: float
    secant_energy_ratio: float
    symmetric_alignment: float
    rotational_residual: float


def passive_secant_fingerprint(
    previous_state: np.ndarray,
    previous_gradient: np.ndarray,
    current_state: np.ndarray,
    current_gradient: np.ndarray,
    *,
    minimum_displacement_energy: float = 0.0,
) -> PassiveSecantFingerprint:
    """Estimate local symmetric and orthogonal responses from a secant pair.

    For a stationary local linear game ``F(x)=A x``, consecutive mandatory
    gradients obey ``delta_g=A delta_x``.  Alignment of ``delta_g`` with
    ``delta_x`` measures the directional symmetric response; the orthogonal
    residual measures rotation.  The routine forms no matrix and costs O(d).
    """

    x0 = np.asarray(previous_state, dtype=float)
    g0 = np.asarray(previous_gradient, dtype=float)
    x1 = np.asarray(current_state, dtype=float)
    g1 = np.asarray(current_gradient, dtype=float)
    if (
        x0.ndim != 1
        or x0.size == 0
        or x1.shape != x0.shape
        or g0.shape != x0.shape
        or g1.shape != x0.shape
        or not all(np.all(np.isfinite(value)) for value in (x0, g0, x1, g1))
    ):
        raise ValueError("states and gradients must be finite matching vectors")
    if (
        not math.isfinite(minimum_displacement_energy)
        or minimum_displacement_energy < 0.0
    ):
        raise ValueError("minimum displacement energy must be finite and nonnegative")
    displacement = x1 - x0
    gradient_change = g1 - g0
    energy = float(displacement @ displacement)
    if energy <= minimum_displacement_energy:
        return PassiveSecantFingerprint(
            informative=False,
            displacement_energy=energy,
            secant_energy_ratio=math.nan,
            symmetric_alignment=math.nan,
            rotational_residual=math.nan,
        )
    alignment = float(displacement @ gradient_change / energy)
    ratio = float(gradient_change @ gradient_change / energy)
    rotational_square = max(0.0, ratio - alignment * alignment)
    return PassiveSecantFingerprint(
        informative=True,
        displacement_energy=energy,
        secant_energy_ratio=ratio,
        symmetric_alignment=alignment,
        rotational_residual=math.sqrt(rotational_square),
    )

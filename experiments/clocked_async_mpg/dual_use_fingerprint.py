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

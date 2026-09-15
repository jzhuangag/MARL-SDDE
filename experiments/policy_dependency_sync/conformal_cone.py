"""Distribution-free trajectory-tube certificates and policy-shift correction."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def split_conformal_radius(residuals: Sequence[float], miscoverage: float) -> float:
    """Return the finite-sample split-conformal upper radius.

    The rank is ``ceil((n+1)(1-alpha))``.  If it equals ``n+1``, the finite
    calibration set cannot certify the requested level and infinity is
    returned rather than silently using the sample maximum.
    """

    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("residuals must be a nonempty finite vector")
    if not 0.0 < miscoverage < 1.0:
        raise ValueError("miscoverage must lie strictly between zero and one")
    rank = int(math.ceil((len(values) + 1) * (1.0 - miscoverage)))
    if rank > len(values):
        return float("inf")
    return float(np.partition(values, rank - 1)[rank - 1])


def gaussian_trajectory_kl_bound(
    horizon: int,
    per_step_joint_mean_l2_shift: float,
    policy_noise_std: float,
) -> float:
    """Chain-rule KL bound for equal-covariance transformed Gaussians."""

    if horizon < 0 or per_step_joint_mean_l2_shift < 0.0 or policy_noise_std <= 0.0:
        raise ValueError("invalid policy-shift parameters")
    return (
        float(horizon)
        * float(per_step_joint_mean_l2_shift) ** 2
        / (2.0 * float(policy_noise_std) ** 2)
    )


def total_variation_from_kl(kl_upper_bound: float) -> float:
    if kl_upper_bound < 0.0:
        raise ValueError("KL upper bound must be nonnegative")
    return min(1.0, math.sqrt(0.5 * float(kl_upper_bound)))


def shifted_escape_probability(
    calibration_miscoverage: float,
    total_variation_upper_bound: float,
) -> float:
    if not 0.0 <= calibration_miscoverage <= 1.0:
        raise ValueError("miscoverage must be in [0,1]")
    if not 0.0 <= total_variation_upper_bound <= 1.0:
        raise ValueError("total variation must be in [0,1]")
    return min(1.0, calibration_miscoverage + total_variation_upper_bound)


def cone_gradient_error_bound(
    return_bound: float,
    owner_score_bound: float,
    escape_probability: float,
) -> float:
    if return_bound < 0.0 or owner_score_bound < 0.0:
        raise ValueError("bounds must be nonnegative")
    if not 0.0 <= escape_probability <= 1.0:
        raise ValueError("escape_probability must be in [0,1]")
    return 4.0 * return_bound * owner_score_bound * escape_probability

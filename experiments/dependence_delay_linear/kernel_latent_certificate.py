"""Anytime hidden-sharing certificate from bounded kernel similarities."""

from typing import Dict, Optional

import numpy as np
from scipy.special import i0e

from latent_collision_certificate import time_uniform_hoeffding_radius


def periodic_rbf(first: float, second: float, lengthscale: float) -> float:
    """Return a translation-invariant RBF kernel on the unit circle."""

    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be positive")
    sine = np.sin(np.pi * (float(first) - float(second)))
    return float(np.exp(-2.0 * sine * sine / lengthscale ** 2))


def periodic_rbf_independent_mean(lengthscale: float) -> float:
    """Return E[k(U,V)] for independent uniform unit-circle samples."""

    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be positive")
    inverse_square = 1.0 / lengthscale ** 2
    return float(i0e(inverse_square))


def kernel_latent_rho_upper(
    similarity_sum: float,
    similarity_trials: int,
    similarity_bias_sum: float,
    control_sum: float,
    control_trials: int,
    control_bias_sum: float,
    alpha_similarity: float,
    alpha_control: float,
) -> Dict[str, float]:
    """Certify hidden sharing while estimating the kernel baseline."""

    if similarity_trials < 1 or control_trials < 1:
        raise ValueError("both streams require at least one observation")
    if not 0.0 <= similarity_sum <= similarity_trials:
        raise ValueError("invalid similarity_sum")
    if not 0.0 <= control_sum <= control_trials:
        raise ValueError("invalid control_sum")
    if not 0.0 <= similarity_bias_sum <= similarity_trials:
        raise ValueError("invalid similarity_bias_sum")
    if not 0.0 <= control_bias_sum <= control_trials:
        raise ValueError("invalid control_bias_sum")
    similarity_mean = similarity_sum / float(similarity_trials)
    control_mean = control_sum / float(control_trials)
    similarity_radius = time_uniform_hoeffding_radius(
        similarity_trials, alpha_similarity
    )
    control_radius = time_uniform_hoeffding_radius(
        control_trials, alpha_control
    )
    similarity_upper = min(
        1.0,
        similarity_mean
        + similarity_radius
        + similarity_bias_sum / float(similarity_trials),
    )
    baseline_lower = max(
        0.0,
        control_mean
        - control_radius
        - control_bias_sum / float(control_trials),
    )
    rho_upper = (
        similarity_upper - baseline_lower
    ) / (1.0 - baseline_lower)
    return {
        "similarity_mean": float(similarity_mean),
        "control_mean": float(control_mean),
        "similarity_radius": float(similarity_radius),
        "control_radius": float(control_radius),
        "similarity_upper": float(similarity_upper),
        "baseline_lower": float(baseline_lower),
        "rho_upper": float(np.clip(rho_upper, 0.0, 1.0)),
    }


def lazy_joint_tv_upper(
    persistence_upper: float,
    gap: int,
    num_hidden_chains: int = 3,
) -> float:
    """Bound joint TV for lazy-refresh continuous hidden chains."""

    if not 0.0 <= persistence_upper <= 1.0:
        raise ValueError("persistence_upper must lie in [0, 1]")
    if gap < 1 or num_hidden_chains < 1:
        raise ValueError("gap and num_hidden_chains must be positive")
    return float(
        min(
            1.0,
            num_hidden_chains * persistence_upper ** int(gap),
        )
    )


def minimum_kernel_gap(
    persistence_upper: float,
    target_tv: float,
    num_hidden_chains: int = 3,
    maximum_gap: int = 100000,
) -> int:
    """Return the smallest predictable gap meeting the kernel TV target."""

    if not 0.0 < target_tv < 1.0:
        raise ValueError("target_tv must lie in (0, 1)")
    if persistence_upper >= 1.0:
        return int(maximum_gap + 1)
    for gap in range(1, maximum_gap + 1):
        if (
            lazy_joint_tv_upper(
                persistence_upper, gap, num_hidden_chains
            )
            <= target_tv
        ):
            return int(gap)
    return int(maximum_gap + 1)


def sample_kernel_probe(
    rng: np.random.RandomState,
    states: np.ndarray,
    persistence: float,
    rho: float,
    gap: int,
    lengthscale: float,
    previous_first: Optional[float],
) -> Dict[str, object]:
    """Advance hidden chains and reveal same-time/control similarities."""

    if states.shape != (3,):
        raise ValueError("states must contain common and two private chains")
    retained = rng.random_sample(3) < persistence ** int(gap)
    redraws = rng.random_sample(3)
    next_states = np.where(retained, states, redraws)
    share_probability = np.sqrt(rho)
    first_shared = bool(rng.random_sample() < share_probability)
    second_shared = bool(rng.random_sample() < share_probability)
    first = next_states[0] if first_shared else next_states[1]
    second = next_states[0] if second_shared else next_states[2]
    control = (
        None
        if previous_first is None
        else periodic_rbf(previous_first, second, lengthscale)
    )
    return {
        "states": next_states.astype(float),
        "first": float(first),
        "second": float(second),
        "similarity": periodic_rbf(first, second, lengthscale),
        "control": control,
    }

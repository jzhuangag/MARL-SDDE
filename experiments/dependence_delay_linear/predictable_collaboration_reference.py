"""Causal readout reference, NOT a replacement for any frozen experiment.

Independent local learners supply bounded, possibly stale donor values. A
recipient commits its mixture before seeing its block. Within-block AR(1)
innovations give a conditionally centered learning statistic. See
docs/causal_collaboration_closure_decision.md for the deliberately narrow model.
No target, oracle risk, formal endpoint, or estimated correlation is an input.
"""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


def project_simplex(vector):
    """Euclidean simplex projection; sorting costs O(K log K)."""
    vector = np.asarray(vector, dtype=float)
    if vector.ndim != 1 or vector.size < 1 or not np.isfinite(vector).all():
        raise ValueError("expected a finite nonempty vector")
    ordered = np.sort(vector)[::-1]
    offsets = (np.cumsum(ordered) - 1.0) / np.arange(1, vector.size + 1)
    active = np.flatnonzero(ordered > offsets)
    return np.maximum(vector - offsets[active[-1]], 0.0)


def innovation_mean(observations, mixing):
    """Use only within-block pairs, never a pair crossing a target change."""
    observations = np.asarray(observations, dtype=float)
    if (observations.ndim != 2 or observations.shape[1] < 2
            or not np.isfinite(observations).all()):
        raise ValueError("expected finite agent x block observations, block >= 2")
    if not 0 <= mixing < 1:
        raise ValueError("known mixing must lie in [0, 1)")
    return np.mean(observations[:, 1:] - mixing * observations[:, :-1], axis=1) / (1 - mixing)


def innovation_variance(mixing, block_size, marginal_variance):
    if (not 0 <= mixing < 1 or block_size < 2
            or not math.isfinite(marginal_variance) or marginal_variance < 0):
        raise ValueError("invalid public variance/mixing specification")
    return marginal_variance * (1 + mixing) / ((block_size - 1) * (1 - mixing))


def mixture_boundary(variance_process, delta, mixture_scale=1.0):
    """Two-sided Gaussian-mixture, all-prefix martingale boundary."""
    value = np.asarray(variance_process, dtype=float)
    if (np.any(value < 0) or not np.isfinite(value).all()
            or not 0 < delta < 1 or not math.isfinite(mixture_scale)
            or mixture_scale <= 0):
        raise ValueError("invalid martingale boundary specification")
    return np.sqrt((value + mixture_scale)
                   * (2 * math.log(1 / delta) + np.log1p(value / mixture_scale)))


@dataclass(frozen=True)
class ReferenceConfig:
    agents: int = 4
    horizon: int = 256
    block_size: int = 8
    delay: int = 0
    mixing: float = 0.5
    marginal_variance_bound: float = 1.0
    radius: float = 2.0
    retention: float = 0.8
    delta: float = 0.05

    def __post_init__(self):
        for name in ("agents", "horizon", "block_size", "delay"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.agents < 1 or self.horizon < 1 or self.block_size < 2 or self.delay < 0:
            raise ValueError("invalid dimensions")
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be positive and finite")
        if not 0 < self.retention < 1 or not 0 < self.delta < 1:
            raise ValueError("retention/delta must lie in (0, 1)")
        innovation_variance(self.mixing, self.block_size, self.marginal_variance_bound)

    @property
    def statistic_variance_bound(self):
        return innovation_variance(self.mixing, self.block_size, self.marginal_variance_bound)

    @property
    def gradient_second_moment_bound(self):
        return 4 * self.agents * self.radius**2 * (4 * self.radius**2 + self.statistic_variance_bound)

    @property
    def weight_step(self):
        # Horizon/public-bound choice, not performance tuned.
        return math.sqrt(2 / (self.horizon * self.gradient_second_moment_bound))


class PredictableReadout:
    """Two-phase interface enforces decide-before-observe.

    Column zero is the current recipient's independent local model. Other
    columns are the other local learners, at the specified additional delay.
    Personalized outputs do NOT feed back into the local donor bank.
    Communication counts directed scalar donor payloads, not protocol headers.
    """

    def __init__(self, config, initial_local=None):
        self.config = config
        n = config.agents
        local = np.zeros(n) if initial_local is None else np.asarray(initial_local, dtype=float)
        if local.shape != (n,) or not np.isfinite(local).all() or np.any(np.abs(local) > config.radius):
            raise ValueError("initial local models must be inside the public radius")
        self._history = deque([local.copy()], maxlen=config.delay + 1)
        self._weights = np.zeros((n, n))
        self._weights[:, 0] = 1.0
        self._pending = None
        self.blocks = 0
        self.actor_transitions = 0
        self.donor_scalars = 0
        self.gradient_norm_sum = np.zeros(n)
        self.excess_noise_variance = np.zeros(n)

    @property
    def local_models(self):
        return self._history[-1].copy()

    @property
    def weights(self):
        return self._weights.copy()

    def begin_block(self):
        if self._pending is not None:
            raise RuntimeError("finish the pending block before choosing another action")
        if self.blocks >= self.config.horizon:
            raise RuntimeError("the public horizon is exhausted")
        n = self.config.agents
        current = self._history[-1]
        stale = self._history[0]  # initial value is held until delay history fills
        candidates = np.empty((n, n))
        for i in range(n):
            candidates[i, 0] = current[i]
            candidates[i, 1:] = stale[np.arange(n) != i]
        prediction = np.sum(self._weights * candidates, axis=1)
        self._pending = (candidates, prediction)
        # Count every directed retrieval, even when the chosen weight is zero.
        self.donor_scalars += n * (n - 1)
        return {"candidates": candidates.copy(), "weights": self.weights,
                "prediction": prediction.copy()}

    def finish_block(self, observations):
        if self._pending is None:
            raise RuntimeError("choose the action before supplying observations")
        cfg = self.config
        observations = np.asarray(observations, dtype=float)
        if observations.shape != (cfg.agents, cfg.block_size):
            raise ValueError("observation shape does not match the charged block")
        statistic = innovation_mean(observations, cfg.mixing)
        candidates, prediction = self._pending
        a = cfg.retention
        output = a * prediction + (1 - a) * statistic
        local_output = a * candidates[:, 0] + (1 - a) * statistic
        gradient = 2 * (prediction - statistic)[:, None] * candidates
        next_weights = np.vstack([project_simplex(w - cfg.weight_step * g)
                                  for w, g in zip(self._weights, gradient)])
        grad_norm = np.sum(gradient**2, axis=1)
        displacement = prediction - candidates[:, 0]
        next_local = np.clip(local_output, -cfg.radius, cfg.radius)
        if not all(np.isfinite(x).all() for x in (output, local_output, grad_norm, next_local)):
            raise FloatingPointError("numeric overflow; block was not committed")
        self._weights = next_weights
        self._history.append(next_local)
        self.gradient_norm_sum += grad_norm
        self.excess_noise_variance += 4 * a**2 * cfg.statistic_variance_bound * displacement**2
        self.actor_transitions += cfg.agents * cfg.block_size
        self.blocks += 1
        self._pending = None
        # Initial weights equal the local vertex, so its initial potential is 0.
        excess_bound = (a**2 * cfg.weight_step * self.gradient_norm_sum / 2
                        + mixture_boundary(self.excess_noise_variance,
                                           cfg.delta / cfg.agents))
        return {"output": output, "local_output": local_output,
                "statistic": statistic, "gradient": gradient,
                "pre_prediction": prediction.copy(),
                "local_pre_prediction": candidates[:, 0].copy(),
                "observable_local_excess_bound": excess_bound}

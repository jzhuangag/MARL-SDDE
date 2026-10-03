"""Outcome-free MinAtar bridge for a theorem-aligned fixed nonlinear encoder.

The environment stream is a standard fixed-policy Markov reward process.  The
state representation is nonlinear, but the trainable TD head is linear, so the
delayed update remains an affine Markov stochastic approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


GAMES = ("asterix", "breakout", "seaquest")
FULL_ACTIONS = 6


def legacy_numpy_seed(seed: int) -> int:
    """Map a provenance seed deterministically into RandomState's uint32 domain."""

    return int(int(seed) % (2**32))


@dataclass(frozen=True)
class StreamBatch:
    states: np.ndarray
    previous_actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    next_previous_actions: np.ndarray
    terminals: np.ndarray

    def __post_init__(self) -> None:
        rows = self.states.shape[0]
        if self.states.ndim != 4 or self.next_states.shape != self.states.shape:
            raise ValueError("states must be matching NHWC arrays")
        for value in (
            self.previous_actions,
            self.rewards,
            self.next_previous_actions,
            self.terminals,
        ):
            if value.shape != (rows,):
                raise ValueError("transition fields must have one row per state")


@dataclass(frozen=True)
class ReferenceMoments:
    drift: np.ndarray
    reward_vector: np.ndarray
    feature_covariance: np.ndarray
    fixed_point: np.ndarray
    symmetric_min_eigenvalue: float
    spectral_norm: float


class FrozenConvEncoder(nn.Module):
    """Seeded convolutional random features with no trainable parameter."""

    def __init__(
        self,
        channels: int,
        *,
        seed: int,
        filters: int = 8,
        output_features: int = 32,
    ) -> None:
        super().__init__()
        if channels < 1 or filters < 1 or output_features < 1:
            raise ValueError("encoder dimensions must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        convolution = torch.randn(
            filters, channels, 3, 3, generator=generator, dtype=torch.float64
        ) / np.sqrt(9.0 * channels)
        projection = torch.randn(
            output_features,
            4 * filters + FULL_ACTIONS,
            generator=generator,
            dtype=torch.float64,
        ) / np.sqrt(4.0 * filters + FULL_ACTIONS)
        self.register_buffer("convolution", convolution)
        self.register_buffer("projection", projection)

    @property
    def feature_dimension(self) -> int:
        return int(self.projection.shape[0] + 1)

    def forward(
        self, states_nhwc: torch.Tensor, previous_actions: torch.Tensor
    ) -> torch.Tensor:
        if states_nhwc.ndim != 4:
            raise ValueError("states must be NHWC")
        states = states_nhwc.permute(0, 3, 1, 2).to(torch.float64)
        hidden = F.relu(F.conv2d(states, self.convolution, padding=1))
        pooled = F.adaptive_avg_pool2d(hidden, (2, 2)).flatten(1)
        action = F.one_hot(
            previous_actions.to(torch.int64), num_classes=FULL_ACTIONS
        ).to(torch.float64)
        nonlinear = torch.tanh(F.linear(torch.cat((pooled, action), dim=1), self.projection))
        features = torch.cat(
            (torch.ones((nonlinear.shape[0], 1), dtype=torch.float64), nonlinear),
            dim=1,
        )
        return features / torch.linalg.vector_norm(features, dim=1, keepdim=True)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(np.ascontiguousarray(value.cpu().numpy()).tobytes())
        return digest.hexdigest()


def encode_numpy(
    encoder: FrozenConvEncoder,
    states: np.ndarray,
    previous_actions: np.ndarray,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    """Encode an NHWC state array deterministically on CPU."""

    values = np.asarray(states)
    actions = np.asarray(previous_actions)
    if values.ndim != 4 or actions.shape != (values.shape[0],):
        raise ValueError("invalid state/action arrays")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows: list[np.ndarray] = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, values.shape[0], batch_size):
            stop = min(start + batch_size, values.shape[0])
            result = encoder(
                torch.from_numpy(values[start:stop]),
                torch.from_numpy(actions[start:stop]),
            )
            rows.append(result.cpu().numpy())
    return np.concatenate(rows, axis=0)


def sample_stream(
    game: str,
    *,
    transitions: int,
    environment_seed: int,
    policy_seed: int,
    sticky_action_probability: float = 0.1,
    difficulty_ramping: bool = False,
) -> StreamBatch:
    """Generate a regenerative uniform-six-action MinAtar trajectory."""

    if game not in GAMES or transitions < 1:
        raise ValueError("unknown game or invalid horizon")
    if not 0.0 <= sticky_action_probability <= 1.0:
        raise ValueError("invalid sticky-action probability")
    from minatar import Environment

    environment = Environment(
        game,
        sticky_action_prob=sticky_action_probability,
        difficulty_ramping=difficulty_ramping,
    )
    environment.seed(legacy_numpy_seed(environment_seed))
    policy = np.random.RandomState(legacy_numpy_seed(policy_seed))
    environment.reset()
    shape = environment.state_shape()
    states = np.empty((transitions, *shape), dtype=np.bool_)
    next_states = np.zeros_like(states)
    previous = np.empty(transitions, dtype=np.int8)
    next_previous = np.zeros(transitions, dtype=np.int8)
    rewards = np.empty(transitions, dtype=np.float64)
    terminals = np.empty(transitions, dtype=np.bool_)
    for time in range(transitions):
        states[time] = environment.state()
        previous[time] = int(environment.last_action)
        requested_action = int(policy.randint(FULL_ACTIONS))
        reward, terminal = environment.act(requested_action)
        rewards[time] = float(reward)
        terminals[time] = bool(terminal)
        if terminal:
            environment.reset()
        else:
            next_states[time] = environment.state()
            next_previous[time] = int(environment.last_action)
    return StreamBatch(
        states=states,
        previous_actions=previous,
        rewards=rewards,
        next_states=next_states,
        next_previous_actions=next_previous,
        terminals=terminals,
    )


def encoded_stream(
    encoder: FrozenConvEncoder,
    stream: StreamBatch,
    *,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return current features, terminal-zeroed successors, and rewards."""

    features = encode_numpy(
        encoder, stream.states, stream.previous_actions, batch_size=batch_size
    )
    successors = encode_numpy(
        encoder,
        stream.next_states,
        stream.next_previous_actions,
        batch_size=batch_size,
    )
    successors[stream.terminals] = 0.0
    return features, successors, stream.rewards.copy()


def reference_moments(
    feature_batches: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    discount: float,
    regularization: float,
) -> ReferenceMoments:
    """Accumulate empirical regularized TD moments without storing all rows."""

    if not 0.0 <= discount < 1.0 or regularization <= 0.0:
        raise ValueError("invalid discount or regularization")
    drift = reward_vector = covariance = None
    total = 0
    for features, successors, rewards in feature_batches:
        phi = np.asarray(features, dtype=np.float64)
        phi_next = np.asarray(successors, dtype=np.float64)
        reward = np.asarray(rewards, dtype=np.float64)
        if phi.ndim != 2 or phi_next.shape != phi.shape or reward.shape != (phi.shape[0],):
            raise ValueError("invalid feature batch")
        if drift is None:
            dimension = phi.shape[1]
            drift = np.zeros((dimension, dimension), dtype=np.float64)
            reward_vector = np.zeros(dimension, dtype=np.float64)
            covariance = np.zeros((dimension, dimension), dtype=np.float64)
        drift += phi.T @ (phi - discount * phi_next)
        reward_vector += phi.T @ reward
        covariance += phi.T @ phi
        total += phi.shape[0]
    if total < 1 or drift is None or reward_vector is None or covariance is None:
        raise ValueError("at least one nonempty feature batch is required")
    drift = drift / total + regularization * np.eye(drift.shape[0])
    reward_vector = reward_vector / total
    covariance = covariance / total
    fixed_point = np.linalg.solve(drift, reward_vector)
    symmetric = 0.5 * (drift + drift.T)
    return ReferenceMoments(
        drift=drift,
        reward_vector=reward_vector,
        feature_covariance=covariance,
        fixed_point=fixed_point,
        symmetric_min_eigenvalue=float(np.linalg.eigvalsh(symmetric)[0]),
        spectral_norm=float(np.linalg.svd(drift, compute_uv=False)[0]),
    )


def coupled_prefix_indices(*, rho: float, q_max: int, seed: int) -> np.ndarray:
    """Select a common path (0) or actor-private path (actor+1)."""

    if not 0.0 <= rho <= 1.0 or q_max < 1:
        raise ValueError("invalid coupling parameters")
    random = np.random.default_rng(int(seed))
    shared = random.random(q_max) < np.sqrt(rho)
    private = np.arange(1, q_max + 1, dtype=np.int64)
    return np.where(shared, 0, private)


def stationary_cost_coefficient(*, overhead: float, q: int, rho: float) -> float:
    """Leading fully charged PR coefficient, up to a task constant."""

    if overhead <= 0.0 or q < 1 or not 0.0 <= rho <= 1.0:
        raise ValueError("invalid phase-law arguments")
    return float((overhead + q) * (rho + (1.0 - rho) / q))

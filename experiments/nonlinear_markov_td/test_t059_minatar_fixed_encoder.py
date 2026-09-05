from __future__ import annotations

import numpy as np
import pytest

from experiments.nonlinear_markov_td.t059_minatar_fixed_encoder import (
    FrozenConvEncoder,
    GAMES,
    coupled_prefix_indices,
    encode_numpy,
    encoded_stream,
    legacy_numpy_seed,
    reference_moments,
    sample_stream,
    stationary_cost_coefficient,
)


@pytest.mark.parametrize("game,channels", tuple(zip(GAMES, (4, 4, 10), strict=True)))
def test_minatar_stream_shape_and_terminal_zeroing(game: str, channels: int) -> None:
    stream = sample_stream(
        game, transitions=128, environment_seed=59001, policy_seed=59002
    )
    assert stream.states.shape == (128, 10, 10, channels)
    assert np.all((0 <= stream.previous_actions) & (stream.previous_actions < 6))
    encoder = FrozenConvEncoder(channels, seed=59003)
    phi, phi_next, reward = encoded_stream(encoder, stream, batch_size=31)
    assert phi.shape == (128, 33)
    assert reward.shape == (128,)
    assert np.allclose(np.linalg.norm(phi, axis=1), 1.0)
    assert np.all(phi_next[stream.terminals] == 0.0)


def test_stream_and_encoder_are_exactly_reproducible() -> None:
    first = sample_stream(
        "asterix", transitions=64, environment_seed=11, policy_seed=12
    )
    second = sample_stream(
        "asterix", transitions=64, environment_seed=11, policy_seed=12
    )
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.rewards, second.rewards)
    encoder_a = FrozenConvEncoder(4, seed=13)
    encoder_b = FrozenConvEncoder(4, seed=13)
    assert encoder_a.fingerprint() == encoder_b.fingerprint()
    assert np.array_equal(
        encode_numpy(encoder_a, first.states, first.previous_actions),
        encode_numpy(encoder_b, second.states, second.previous_actions),
    )


def test_large_provenance_seed_has_a_reproducible_uint32_mapping() -> None:
    seed = 202608050101
    assert 0 <= legacy_numpy_seed(seed) < 2**32
    first = sample_stream(
        "breakout", transitions=32, environment_seed=seed, policy_seed=seed + 1
    )
    second = sample_stream(
        "breakout", transitions=32, environment_seed=seed, policy_seed=seed + 1
    )
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.rewards, second.rewards)


def test_reference_moments_recover_a_constructed_fixed_point() -> None:
    rng = np.random.default_rng(19)
    phi = rng.normal(size=(2000, 5))
    phi /= np.linalg.norm(phi, axis=1, keepdims=True)
    phi_next = np.zeros_like(phi)
    target = rng.normal(size=5)
    regularization = 0.1
    reward = phi @ target + regularization * target @ target / np.maximum(
        np.sum(phi * target, axis=1), 1e-8
    )
    # Use the exact empirical b implied by a chosen fixed point; this avoids a
    # distributional assumption in the algebra test.
    drift = phi.T @ phi / phi.shape[0] + regularization * np.eye(5)
    b = drift @ target
    reward = np.linalg.lstsq(phi.T, b * phi.shape[0], rcond=None)[0]
    result = reference_moments(
        [(phi, phi_next, reward)], discount=0.97, regularization=regularization
    )
    assert np.allclose(result.fixed_point, target, atol=1e-10)
    assert result.symmetric_min_eigenvalue > 0.0


def test_coupling_endpoints_and_pairwise_probability() -> None:
    assert np.all(coupled_prefix_indices(rho=1.0, q_max=16, seed=7) == 0)
    assert np.array_equal(
        coupled_prefix_indices(rho=0.0, q_max=4, seed=7), np.arange(1, 5)
    )
    trials = 20000
    shared = 0
    for seed in range(trials):
        indices = coupled_prefix_indices(rho=0.49, q_max=2, seed=seed)
        shared += int(indices[0] == 0 and indices[1] == 0)
    assert abs(shared / trials - 0.49) < 0.015


def test_phase_has_an_interior_participation_region() -> None:
    actions = (1, 4, 16)
    low = min(actions, key=lambda q: stationary_cost_coefficient(overhead=32, q=q, rho=0.1))
    high = min(actions, key=lambda q: stationary_cost_coefficient(overhead=32, q=q, rho=0.9))
    assert low == 16
    assert high == 1


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        coupled_prefix_indices(rho=-0.1, q_max=4, seed=1)
    with pytest.raises(ValueError):
        stationary_cost_coefficient(overhead=0.0, q=1, rho=0.0)

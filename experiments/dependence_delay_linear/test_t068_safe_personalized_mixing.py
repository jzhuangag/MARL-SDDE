import numpy as np
import pytest

from experiments.dependence_delay_linear.t068_safe_personalized_mixing import (
    block_noise_multiplier,
    initial_moment_state,
    propagate_personalized_block,
    residual_transfer_quadratic,
    simulate_personalized_mixing,
    spatial_noise_covariance,
)


def test_transfer_quadratic_is_exact():
    drift = np.asarray([[1.0, 0.2], [0.1, 0.8]])
    residual = np.asarray([0.7, -0.4])
    directions = np.asarray([[0.2, -0.3], [0.5, 0.1]])
    weights = np.asarray([0.4, 0.2])
    linear, hessian = residual_transfer_quadratic(
        drift=drift, residual=residual, directions=directions
    )
    before = 0.5 * residual @ residual
    after_residual = residual + drift @ directions @ weights
    after = 0.5 * after_residual @ after_residual
    assert after - before == pytest.approx(
        linear @ weights + 0.5 * weights @ hessian @ weights
    )
    assert np.min(np.linalg.eigvalsh(hessian)) >= -1e-12


def test_block_noise_multiplier_matches_direct_double_sum():
    contraction = 0.93
    steps = 7
    temporal = 0.6
    weights = contraction ** np.arange(steps - 1, -1, -1)
    expected = sum(
        weights[i] * weights[j] * temporal ** abs(i - j)
        for i in range(steps)
        for j in range(steps)
    )
    assert block_noise_multiplier(contraction, steps, temporal) == pytest.approx(expected)


def test_spatial_covariance_endpoints():
    independent = spatial_noise_covariance(4, 2.0, 0.0)
    common = spatial_noise_covariance(4, 2.0, 1.0)
    assert np.allclose(independent, 2.0 * np.eye(4))
    assert np.allclose(common, 2.0 * np.ones((4, 4)))


def test_zero_mixing_matches_same_data_shadow_exactly():
    targets = [-0.2, -0.1, 0.1, 0.2]
    result = simulate_personalized_mixing(
        targets=targets,
        initial_parameters=[1.0] * 4,
        delay=2,
        blocks=8,
        gain=0.05,
        curvature=1.0,
        local_steps=5,
        noise_scale=0.4,
        spatial_correlation=0.3,
        temporal_correlation=0.6,
        policy="fixed",
        alpha=0.0,
    )
    assert np.allclose(result.risk_path, result.shadow_risk_path)
    assert result.terminal_risk == pytest.approx(result.terminal_shadow_risk)


def test_safe_oracle_is_checkpointwise_no_worse_than_shadow():
    result = simulate_personalized_mixing(
        targets=[-0.2, -0.1, 0.1, 0.2],
        initial_parameters=[1.5] * 4,
        delay=1,
        blocks=12,
        gain=0.04,
        curvature=1.0,
        local_steps=6,
        noise_scale=0.8,
        spatial_correlation=0.2,
        temporal_correlation=0.7,
        policy="safe_oracle",
        safe_alpha_grid=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    )
    assert np.all(result.risk_path <= result.shadow_risk_path + 1e-12)


def test_perfectly_common_noise_removes_variance_reduction_when_targets_equal():
    kwargs = dict(
        targets=[0.0] * 4,
        initial_parameters=[1.0] * 4,
        delay=0,
        blocks=10,
        gain=0.05,
        curvature=1.0,
        local_steps=5,
        noise_scale=1.0,
        spatial_correlation=1.0,
        temporal_correlation=0.5,
        policy="fixed",
    )
    local = simulate_personalized_mixing(**kwargs, alpha=0.0)
    shared = simulate_personalized_mixing(**kwargs, alpha=1.0)
    assert shared.terminal_risk == pytest.approx(local.terminal_risk)


def test_initial_state_rejects_mismatched_agent_vectors():
    with pytest.raises(ValueError):
        initial_moment_state([0.0, 1.0], [0.0], delay=0)


def test_fixed_policy_rejects_invalid_alpha():
    state = initial_moment_state([0.0, 0.0], [1.0, 1.0], delay=0)
    with pytest.raises(ValueError):
        propagate_personalized_block(
            state,
            targets=[0.0, 0.0],
            gain=0.05,
            curvature=1.0,
            local_steps=2,
            noise_scale=1.0,
            spatial_correlation=0.0,
            temporal_correlation=0.0,
            alpha=1.1,
        )

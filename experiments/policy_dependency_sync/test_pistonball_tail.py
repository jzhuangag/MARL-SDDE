import numpy as np

from .pistonball_tail import (
    active_window_support,
    chain_neighbors,
    directional_influence_matrix,
    initial_ball_index,
    predictive_tube_support,
    score_cross_influence_matrix,
    spsa_hessian_from_objectives,
    summarize_piston_influence,
)


def test_chain_and_ball_index_geometry():
    assert chain_neighbors(5) == {0: (1,), 1: (0, 2), 2: (1, 3), 3: (2, 4), 4: (3,)}
    assert initial_ball_index(45.0, 20) == 0
    assert initial_ball_index(799.0, 20) == 18


def test_active_window_is_state_dependent_and_directed_complete():
    support = active_window_support(8, ball_index=3, radius=1)
    assert support[2] == (3, 4)
    assert support[3] == (2, 4)
    assert support[4] == (2, 3)
    assert support[0] == ()


def test_directional_influence_recovers_mixed_quadratic():
    theta = np.zeros(4)
    weights = np.arange(16, dtype=np.float64).reshape(4, 4)
    weights = np.triu(weights, 1)

    def objective(candidate):
        return float(sum(weights[i, j] * candidate[i] * candidate[j] for i in range(4) for j in range(4)))

    influence = directional_influence_matrix(objective, theta, step=0.03)
    assert np.isclose(influence[0, 3], weights[0, 3])
    assert np.isclose(influence[3, 0], weights[0, 3])


def test_active_summary_retains_constructed_local_mass():
    influence = np.zeros((8, 8))
    support = active_window_support(8, ball_index=3, radius=1)
    for owner, donors in support.items():
        for donor in donors:
            influence[owner, donor] = 1.0
    summary = summarize_piston_influence(influence, 3, 1, graph_seed=4, random_graphs=256)
    assert summary.active_edge_fraction == 1.0
    assert summary.active_edge_cost_fraction == 6 / 56
    assert summary.active_minus_random_median > 0.5


def test_action_noise_shape_is_checked_without_environment_rollout():
    # The actual environment integration is covered by the development smoke;
    # this unit test keeps the pure geometry suite dependency-light.
    assert active_window_support(20, ball_index=19, radius=3)[19] == (16, 17, 18)


def test_predictive_tube_follows_left_moving_ball():
    support = predictive_tube_support(
        20, launch_x=528.0, launch_velocity_x=-35.0, horizon=6, radius=1
    )
    active = [owner for owner, donors in support.items() if donors]
    assert min(active) <= 6
    assert max(active) >= 12
    assert 19 not in active


class _QuadraticScoreFixture:
    n_pistons = 3


def test_score_cross_influence_rejects_nonpositive_scales():
    fixture = _QuadraticScoreFixture()
    with np.testing.assert_raises(ValueError):
        score_cross_influence_matrix(
            fixture,
            np.zeros(3),
            reset_seed=1,
            burn_in=0,
            horizon=2,
            donor_step=0.0,
            policy_noise_std=0.2,
            trajectory_seeds=[1, 2],
        )


def test_spsa_hessian_has_correct_shape_and_zero_diagonal():
    theta = np.zeros(4)
    hessian = np.asarray(
        [[2.0, 0.8, 0.0, 0.0], [0.8, 1.0, 0.3, 0.0], [0.0, 0.3, 1.5, 0.4], [0.0, 0.0, 0.4, 0.5]]
    )

    def builder(_seed):
        return lambda candidate: float(0.5 * candidate @ hessian @ candidate)

    estimate, standard_error = spsa_hessian_from_objectives(
        theta,
        step=0.05,
        sample_seeds=range(1, 257),
        objective_builder=builder,
    )
    assert estimate.shape == (4, 4)
    assert standard_error.shape == (4, 4)
    assert np.allclose(np.diag(estimate), 0.0)
    assert np.allclose(estimate, estimate.T)
    assert np.linalg.norm(estimate - np.diag(np.diag(estimate))) > 0.0

import numpy as np

from experiments.policy_backpressure.policy_backpressure_feasibility import (
    bound_audit,
    joint_tv_max,
    lookahead_beam_oracle,
    make_interference_game,
    policy_path_length,
    simulate_async,
    wallclock_mean_return,
)


def test_evaluation_and_occupancy_are_finite():
    game = make_interference_game(0.8, 0.75)
    policy = np.array([[0.2, 0.7], [0.6, 0.4]])
    ret, value, q, occupancy = game.evaluate(policy)
    assert np.isfinite(ret)
    assert np.all(np.isfinite(value))
    assert np.all(np.isfinite(q))
    assert np.isclose(occupancy.sum(), 1.0)


def test_path_length_dominates_direct_joint_tv():
    game = make_interference_game(0.7, 0.8)
    p0 = np.array([[0.1, 0.2], [0.3, 0.4]])
    p1 = np.array([[0.2, 0.35], [0.5, 0.55]])
    p2 = np.array([[0.7, 0.6], [0.4, 0.2]])
    assert joint_tv_max(game, p2, p0) <= policy_path_length(game, [p0, p1, p2], 0)+1e-12


def test_stale_performance_bound_grid():
    result = bound_audit()
    assert result["bound_checks"] > 100
    assert result["minimum_actual_minus_lower_bound"] >= -1e-10
    assert result["maximum_direct_tv_over_path_tv"] <= 1+1e-12


def test_oracle_never_accepts_a_harmful_grid_step():
    game = make_interference_game(0.85, 0.9)
    policy = np.array([[0.2, 0.8], [0.8, 0.2]])
    result = simulate_async(game, policy, (1, 6), 30, "oracle")
    assert result["harmful"] == 0
    assert result["final_return"] >= result["initial_return"]-1e-12


def test_wallclock_mean_return_uses_piecewise_constant_values():
    assert np.isclose(
        wallclock_mean_return([(0, 1.0), (2, 3.0), (5, 2.0)], 6), 13/6
    )


def test_lookahead_beam_is_feasible_and_finite():
    game = make_interference_game(0.7, 0.8)
    policy = np.array([[0.2, 0.8], [0.8, 0.2]])
    result = lookahead_beam_oracle(game, policy, (1, 3), 8, beam_width=16)
    assert np.isfinite(result["wallclock_mean_return"])
    assert result["final_return"] >= 0
    assert result["status"].startswith("feasible_nonmyopic")

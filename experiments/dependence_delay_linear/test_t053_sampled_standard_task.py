import numpy as np
from functools import lru_cache

from experiments.dependence_delay_linear.run_t049a_exact_schedule_static import (
    build_tasks,
    load_config,
)
from experiments.dependence_delay_linear.t053_sampled_standard_task import (
    build_task_sampling_table,
    delayed_pr_risk,
    prefix_aggregate_innovations,
    sample_fingerprint_matches,
    sample_gradient_paths,
    verify_sampling_table,
)


@lru_cache(maxsize=1)
def _tasks():
    return build_tasks(load_config())


def test_sampling_tables_exactly_reproduce_public_task_moments():
    tasks = _tasks()
    for task in tasks.values():
        verification = verify_sampling_table(task, build_task_sampling_table(task))
        assert verification["transition_max_error"] <= 1e-14
        assert verification["mean_norm"] <= 1e-8
        assert verification["second_moment_max_error"] <= 1e-10


def test_gradient_path_sampler_is_deterministic_and_finite():
    task = _tasks()["frozenlake8x8"]
    table = build_task_sampling_table(task)
    first = sample_gradient_paths(table, paths=3, horizon=20, seed=123)
    second = sample_gradient_paths(table, paths=3, horizon=20, seed=123)
    assert np.array_equal(first, second)
    assert first.shape == (3, 20, task["features"].shape[1])
    assert np.all(np.isfinite(first))


def test_fingerprint_sampler_matches_at_rho_one():
    task = _tasks()["cliffwalking"]
    matches = sample_fingerprint_matches(
        transition=task["continuing_transition"],
        stationary=task["stationary"],
        transitions=12,
        blocks=100,
        rho=1.0,
        seed=7,
    )
    assert np.all(matches == 1)


def test_prefix_aggregate_reuses_common_and_private_trajectories():
    bank = np.arange(17 * 5, dtype=float).reshape(17, 5, 1)
    at_zero = prefix_aggregate_innovations(
        bank, rho=0.0, candidates=(1, 4, 16), seed=9
    )
    np.testing.assert_allclose(at_zero[1], bank[1])
    np.testing.assert_allclose(at_zero[4], np.mean(bank[1:5], axis=0))
    at_one = prefix_aggregate_innovations(
        bank, rho=1.0, candidates=(1, 4, 16), seed=9
    )
    for value in at_one.values():
        np.testing.assert_allclose(value, bank[0])


def test_delayed_pr_risk_matches_direct_scalar_recursion():
    innovations = np.array([[1.0], [-0.5], [0.25], [0.0], [0.75], [-0.25]])
    drift = np.array([[0.8]])
    step = 0.1
    delay = 2
    initial = np.array([-1.0])
    errors = [-1.0] * (delay + 1)
    for noise in innovations[:, 0]:
        errors.append(errors[-1] - step * 0.8 * errors[-1 - delay] + step * noise)
    burn = len(innovations) // 2
    direct = float(np.mean(errors[delay + burn + 1 :]) ** 2)
    computed = delayed_pr_risk(
        innovations,
        drift=drift,
        step_size=step,
        delay=delay,
        initial_error=initial,
    )
    assert abs(computed - direct) < 1e-14

from experiments.dependence_delay_linear.run_t050_stationary_audit import (
    execute,
    load_plan,
)


def test_t050_audit_is_outcome_free_and_complete():
    plan = load_plan()
    result = execute(plan)
    assert result["sampled_trajectories"] == 0
    assert sorted(result["task_results"]) == sorted(plan["tasks"])
    assert len(result["stationary_phase"]) == len(plan["message_overheads"])
    for task in result["task_results"].values():
        assert task["pr_task_constant"] > 0.0
        assert task["long_run_covariance_minimum_eigenvalue"] >= -1e-9
        for delay in task["delays"].values():
            assert 0.0 <= delay["spectral_radius"] < 1.0
            assert set(delay["contraction_horizons"]) == {"0.001", "0.0001"}


def test_t050_phase_has_a_nontrivial_fixed_action_transition():
    result = execute(load_plan())
    for phase in result["stationary_phase"]:
        assert phase["oracle_geometric_improvement"] >= 0.05
        assert phase["strict_improvement_fraction"] >= 0.5
        assert len(set(phase["oracle_support"])) >= 2

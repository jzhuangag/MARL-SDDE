from experiments.dependence_delay_linear.run_t055a_validation import execute


def test_t055a_validation_replays_frozen_artifacts():
    result = execute(bootstrap_replicates=100, bootstrap_seed=13)
    assert result["endpoint_rows"] == 5376
    assert result["unique_cells"] == 84
    assert result["unique_seeds"] == 64
    assert result["summary_replay_pass"]
    assert result["summary_replay_max_abs_error"] < 1e-14
    assert result["provenance_pass"]
    assert result["reproduction_pass"]
    assert result["all_12_gates_pass"]


def test_t055a_validation_preserves_authorization_boundary():
    result = execute(bootstrap_replicates=50, bootstrap_seed=17)
    assert result["formal_preregistration_authorized"]
    assert not result["formal_execution_authorized"]
    assert not result["gpu_authorized"]
    assert not result["hpc4_authorized"]

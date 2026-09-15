from experiments.dependence_delay_linear.run_t076_parallel_persistent_calibration import (
    load_config, scientific_config, validate,
)


def test_parallel_amendment_preserves_frozen_scientific_config() -> None:
    config = load_config()
    result = validate(config)
    assert result["scientific_config_unchanged"] is True
    assert result["ordered_endpoints"] == 13_824
    assert scientific_config(config)["experiment_id"].startswith("T-074-")


def test_execution_does_not_authorize_escalation() -> None:
    authorization = load_config()["authorization"]
    assert authorization["run_local_CPU"] is True
    assert all(authorization[key] is False for key in (
        "formal", "new_seed_pilot", "nonlinear", "gpu", "hpc4"))

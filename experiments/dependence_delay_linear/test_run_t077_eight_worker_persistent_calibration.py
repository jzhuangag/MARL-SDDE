from experiments.dependence_delay_linear.run_t077_eight_worker_persistent_calibration import (
    load_config, validate,
)


def test_eight_worker_contract_and_scientific_hash() -> None:
    result = validate(load_config())
    assert result["workers"] == 8
    assert result["ordered_endpoints"] == 13_824
    assert result["scientific_config_unchanged"] is True


def test_no_compute_escalation_is_authorized() -> None:
    authorization = load_config()["authorization"]
    assert authorization["run_local_CPU"] is True
    assert all(authorization[key] is False for key in (
        "new_seed_pilot", "formal", "nonlinear", "gpu", "hpc4"))

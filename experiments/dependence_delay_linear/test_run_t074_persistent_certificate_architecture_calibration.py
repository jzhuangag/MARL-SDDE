from experiments.dependence_delay_linear.run_t074_persistent_certificate_architecture_calibration import (
    load_config, validate,
)


def test_manifest_is_valid_and_explicitly_tainted() -> None:
    result = validate(load_config())
    assert result["cells"] == 432
    assert result["source_seeds"] == 32
    assert result["tainted_design_evidence_only"] is True


def test_no_new_seed_or_compute_escalation_is_authorized() -> None:
    authorization = load_config()["authorization"]
    assert authorization["run_local_cpu_calibration"] is True
    assert all(authorization[key] is False for key in (
        "claim_formal_evidence", "preregister_new_seed_pilot", "formal", "nonlinear", "gpu", "hpc4"))

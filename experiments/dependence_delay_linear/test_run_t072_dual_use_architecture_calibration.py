import json

from experiments.dependence_delay_linear.run_t072_dual_use_architecture_calibration import (
    load_config,
    validate,
)


def test_calibration_manifest_is_valid_and_explicitly_tainted() -> None:
    result = validate(load_config())
    assert result["cells"] == 432
    assert result["source_seeds"] == 32
    assert result["tainted_design_evidence_only"] is True


def test_calibration_does_not_authorize_scientific_escalation() -> None:
    config = load_config()
    assert all(config["authorization"][key] is False for key in (
        "claim_formal_evidence", "reuse_results_as_new_pilot", "formal",
        "nonlinear", "gpu", "hpc4",
    ))
    assert json.loads(json.dumps(config))["controller"]["extra_probe_transitions"] == 0

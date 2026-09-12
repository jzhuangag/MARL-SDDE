from experiments.dependence_delay_linear.run_t073_continuous_qp_architecture_calibration import (
    load_config,
    validate,
)


def test_manifest_is_valid_and_tainted() -> None:
    result = validate(load_config())
    assert result == {
        "experiment_id": "T-073-continuous-QP-collaboration-architecture-calibration",
        "cells": 432, "source_seeds": 32, "endpoints": 13_824,
        "tainted_design_evidence_only": True,
    }


def test_no_scientific_escalation_is_authorized() -> None:
    authorization = load_config()["authorization"]
    assert authorization["run_local_cpu_calibration"] is True
    assert all(authorization[key] is False for key in (
        "claim_formal_evidence", "preregister_T073A", "formal", "nonlinear", "gpu", "hpc4"
    ))

from __future__ import annotations

from .run_markov_certificate_nonvacuity import (
    CONFIG,
    canonical_config_hash,
    run_rows,
    summarize,
)


def test_frozen_config_shape_and_hash_are_stable() -> None:
    assert CONFIG["audit_id"] == "MCERT-DEV-001"
    assert len(CONFIG["seeds"]) == 64
    assert len(CONFIG["sample_sizes_per_positive_action"]) == 6
    assert canonical_config_hash() == "2e9f28a600f40b3fdc4b612f6d5217e8b6070cdf2c4452c9876d90c2e55c03fb"


def test_small_deterministic_prefix_is_finite_and_fully_charged() -> None:
    rows = run_rows()
    assert len(rows) == 64 * 6 * 2
    assert all(
        row["charged_transitions"]
        == 2 * row["sample_size_per_positive_action"]
        for row in rows
    )


def test_summary_has_exact_frozen_gate_keys() -> None:
    summary = summarize(run_rows())
    assert set(summary["gates"]) == {
        "N1_correct_edge_rate_at_4096",
        "N2_median_value_recovery_at_4096",
        "N3_p05_value_recovery_at_4096",
        "N4_wrong_edge_positive_rate_at_4096",
        "N5_median_recovery_monotone",
        "N6_exact_transition_charging",
        "N7_local_robust_dp_scalar_work",
        "N8_all_rows_finite",
    }

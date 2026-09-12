import math
from pathlib import Path

import pytest

from t021_fixed_q_feasibility import (
    MAX_PLANNED_REPLICATIONS,
    achieved_power,
    build_summary,
    effective_speedup,
    required_replications,
    validate_summary,
    variance_factor,
)


ROOT = Path(__file__).resolve().parents[2]


def test_variance_factor_endpoints_and_speedup() -> None:
    assert variance_factor(32, 0.0) == pytest.approx(1.0 / 32.0)
    assert variance_factor(32, 1.0) == pytest.approx(1.0)
    assert effective_speedup(16, 0.0) == pytest.approx(16.0)


def test_variance_saturates_with_correlation() -> None:
    assert variance_factor(32, 0.9) > variance_factor(4, 0.0)
    assert effective_speedup(32, 0.9) < 1.11


def test_invalid_variance_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        variance_factor(0, 0.5)
    with pytest.raises(ValueError):
        variance_factor(4, 1.1)


def test_power_requirement_monotone_in_noise() -> None:
    effect = math.log(1.05)
    low = required_replications(effect, 0.10, 0.90, 0.05 / 3.0)
    high = required_replications(effect, 0.20, 0.90, 0.05 / 3.0)
    assert low < high


def test_power_requirement_monotone_in_target() -> None:
    effect = math.log(1.05)
    n80 = required_replications(effect, 0.15, 0.80, 0.05 / 3.0)
    n90 = required_replications(effect, 0.15, 0.90, 0.05 / 3.0)
    assert n80 < n90


def test_achieved_power_increases_with_replications() -> None:
    effect = math.log(1.05)
    assert achieved_power(192, effect, 0.15, 0.05 / 3.0) > achieved_power(
        64, effect, 0.15, 0.05 / 3.0
    )


def test_static_sources_and_stop_decision() -> None:
    summary = build_summary(ROOT)
    assert summary["source_artifacts"]["t019_phase_rows"] == 432
    assert summary["source_artifacts"]["t019_phase_cells"] == 72
    assert summary["source_artifacts"]["t020_exp017b_permanently_stopped"]


def test_phase_directions_are_descriptive_only() -> None:
    checks = build_summary(ROOT)["descriptive_design_checks"]
    assert checks["rho_direction_fraction"] == pytest.approx(22.0 / 24.0)
    assert checks["delay_direction_fraction"] == pytest.approx(21.0 / 24.0)
    assert checks["budget_direction_fraction"] == pytest.approx(1.0)
    assert not checks["adaptive_controller_value_gate_passed"]


def test_maximum_power_design_covers_moderate_sd() -> None:
    summary = build_summary(ROOT)
    rows = summary["power_design"]["normal_approximation_rows"]
    moderate = next(row for row in rows if row["paired_log_ratio_sd"] == 0.15)
    assert moderate["n_for_90pct_power"] <= MAX_PLANNED_REPLICATIONS
    assert moderate["power_at_192"] >= 0.90


def test_claim_boundary_and_validation() -> None:
    summary = build_summary(ROOT)
    validate_summary(summary)
    decision = summary["claim_decision"]
    assert decision["online_adaptive_controller_main_claim"] == "exclude"
    assert decision["fixed_q_correlation_delay_mainline"] == "retain"
    assert not decision["future_gpu_preregistration_authorized"]


def test_t021_generates_no_outcome() -> None:
    summary = build_summary(ROOT)
    assert summary["scientific_trajectories_generated"] == 0
    assert summary["gpu_jobs_submitted"] == 0


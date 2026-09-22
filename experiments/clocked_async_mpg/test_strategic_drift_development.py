from __future__ import annotations

import pytest

from .strategic_drift_development import simulate_oracle_strategic_drift


def test_oracle_development_simulator_is_finite_and_fully_charged() -> None:
    result = simulate_oracle_strategic_drift(
        coupling=0.12,
        service_ratio=3.0,
        seed_index=0,
        namespace="strategic-drift-unit",
        maximum_time=8.0,
        horizon=4,
        batch_size=3,
        step_fraction=1.0,
        target_normalized_gap=0.5,
        risk_budget=1e-4,
        tradeoff=1.0,
        hard_no_harm=False,
    )
    assert result["completed_packets"] == result["applied_updates"]
    assert result["completed_transition_work"] == pytest.approx(
        12.0*float(result["completed_packets"])
    )
    assert float(result["total_transition_work"]) >= float(
        result["completed_transition_work"]
    )
    assert 0.0 <= float(result["mean_scale"]) <= 1.0
    assert float(result["debt"]) >= 0.0
    assert int(result["max_realized_delay"]) <= int(result["registered_delay"])


def test_hard_oracle_shield_has_nonnegative_mean_certificate() -> None:
    result = simulate_oracle_strategic_drift(
        coupling=0.2,
        service_ratio=4.0,
        seed_index=1,
        namespace="strategic-drift-unit-hard",
        maximum_time=8.0,
        horizon=4,
        batch_size=3,
        step_fraction=1.0,
        target_normalized_gap=0.5,
        risk_budget=0.0,
        tradeoff=1.0,
        hard_no_harm=True,
    )
    assert float(result["mean_certified_lower_bound"]) >= -1e-12

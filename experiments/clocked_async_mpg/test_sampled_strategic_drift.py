from __future__ import annotations

import pytest

from .sampled_strategic_drift import simulate_sample_split_strategic_drift


def test_sample_split_packet_is_fully_charged_and_finite() -> None:
    result = simulate_sample_split_strategic_drift(
        coupling=0.16,
        service_ratio=3.0,
        seed_index=0,
        namespace="sample-split-controller-unit",
        maximum_time=8.0,
        horizon=4,
        batch_size=4,
        step_fraction=1.0,
        target_normalized_gap=0.5,
        risk_budget=0.001,
        tradeoff=10.0,
    )
    assert result["completed_packets"] == result["applied_updates"]
    assert result["completed_transition_work"] == pytest.approx(
        16.0*float(result["completed_packets"])
    )
    assert float(result["total_transition_work"]) >= float(
        result["completed_transition_work"]
    )
    assert 0.0 <= float(result["mean_scale"]) <= 1.0
    assert int(result["max_realized_delay"]) <= int(result["registered_delay"])


@pytest.mark.parametrize("batch_size", [1, 3])
def test_sample_split_requires_an_even_total_batch(batch_size: int) -> None:
    with pytest.raises(ValueError):
        simulate_sample_split_strategic_drift(
            coupling=0.1,
            service_ratio=2.0,
            seed_index=0,
            namespace="sample-split-invalid",
            maximum_time=1.0,
            horizon=2,
            batch_size=batch_size,
            step_fraction=1.0,
            target_normalized_gap=0.5,
            risk_budget=0.0,
            tradeoff=1.0,
        )

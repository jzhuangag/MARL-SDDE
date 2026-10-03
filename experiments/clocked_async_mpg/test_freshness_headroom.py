from __future__ import annotations

import numpy as np
import pytest

from .freshness_headroom import (
    balanced_periodic_indicator,
    best_periodic_refresh_value,
    causal_resource_schedule,
    causal_resource_schedule_fast,
    equal_cost_headroom,
    markov_regime_path,
    oracle_refresh_value,
    refresh_value,
)


def test_markov_path_is_reproducible_and_boolean() -> None:
    first = markov_regime_path(
        horizon=128, high_prevalence=0.25, persistence=0.8, seed=11
    )
    second = markov_regime_path(
        horizon=128, high_prevalence=0.25, persistence=0.8, seed=11
    )
    assert first.dtype == np.bool_
    assert np.array_equal(first, second)


def test_refresh_value_matches_fusion_reduction() -> None:
    values = refresh_value(np.asarray([1.0, 3.0]), fresh_variance=1.0)
    assert np.allclose(values, np.asarray([0.5, 2.25]))


def test_periodic_indicator_has_exact_even_charge() -> None:
    indicator = balanced_periodic_indicator(12, 3)
    assert indicator.sum() == 3
    assert np.flatnonzero(indicator).tolist() == [2, 6, 10]


def test_fft_periodic_value_matches_brute_force() -> None:
    values = np.asarray([0.1, 3.0, 0.2, 2.0, 0.4, 1.0])
    indicator = balanced_periodic_indicator(len(values), 2)
    brute = max(
        float(np.dot(np.roll(indicator, shift), values))
        for shift in range(len(values))
    )
    assert best_periodic_refresh_value(values, 2) == pytest.approx(brute)


def test_oracle_selects_largest_values() -> None:
    values = np.asarray([0.1, 3.0, 0.2, 2.0])
    assert oracle_refresh_value(values, 2) == pytest.approx(5.0)


def test_dynamic_risk_has_positive_equal_cost_headroom() -> None:
    risk = np.asarray([8.0, 8.0, 8.0, 8.0, 1.0, 1.0, 1.0, 1.0])
    result = equal_cost_headroom(risk, fresh_variance=1.0, refresh_count=2)
    assert result.oracle_risk < result.periodic_risk
    assert result.refresh_count == 2


def test_stationary_risk_has_no_schedule_headroom() -> None:
    result = equal_cost_headroom(
        np.ones(32), fresh_variance=1.0, refresh_count=8
    )
    assert result.oracle_risk == pytest.approx(result.periodic_risk)
    assert result.relative_oracle_improvement == pytest.approx(0.0, abs=1e-14)


def test_causal_schedule_respects_hard_refresh_cap() -> None:
    result = causal_resource_schedule(
        np.asarray([8.0, 1.0, 8.0, 1.0, 8.0, 1.0]),
        fresh_variance=1.0,
        maximum_refresh_count=2,
        average_refresh_budget=2 / 6,
        risk_tradeoff=2.0,
    )
    assert result.refresh_count <= 2
    assert result.incurred_risk > 0.0
    assert result.maximum_resource_debt >= result.final_resource_debt


def test_causal_schedule_uses_no_future_values() -> None:
    prefix = np.asarray([1.0, 8.0, 1.0, 8.0])
    left = causal_resource_schedule(
        np.concatenate([prefix, np.ones(4)]),
        fresh_variance=1.0,
        maximum_refresh_count=4,
        average_refresh_budget=0.25,
        risk_tradeoff=1.0,
    )
    right = causal_resource_schedule(
        np.concatenate([prefix, np.full(4, 100.0)]),
        fresh_variance=1.0,
        maximum_refresh_count=4,
        average_refresh_budget=0.25,
        risk_tradeoff=1.0,
    )
    # Future values change final totals but cannot alter prefix actions.
    common = causal_resource_schedule(
        prefix,
        fresh_variance=1.0,
        maximum_refresh_count=4,
        average_refresh_budget=0.25,
        risk_tradeoff=1.0,
    )
    assert left.refresh_events[: len(prefix)] == common.refresh_events
    assert right.refresh_events[: len(prefix)] == common.refresh_events


def test_fast_causal_schedule_is_fieldwise_identical() -> None:
    risk = np.asarray([1.0, 8.0, 2.0, 1.0, 4.0, 8.0, 1.0])
    kwargs = {
        "fresh_variance": 1.3,
        "maximum_refresh_count": 3,
        "average_refresh_budget": 3 / 7,
        "risk_tradeoff": 2.0,
    }
    reference = causal_resource_schedule(risk, **kwargs)
    fast = causal_resource_schedule_fast(risk, **kwargs)
    assert fast == reference

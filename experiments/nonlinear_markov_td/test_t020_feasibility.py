from pathlib import Path

import numpy as np
import pytest

from t020_adaptation_value_audit import (
    EXPECTED_FALLBACK,
    analyze,
    probe_count,
)
from t020_learning_value_surrogate import (
    ObservableState,
    continuous_message_optimum,
    learning_value_lcb,
    shielded_choice,
    usable_horizon,
    variance_factor,
)


ENDPOINTS = Path(__file__).resolve().parents[2] / "tmp" / "t020" / "endpoints.csv"
requires_endpoints = pytest.mark.skipif(
    not ENDPOINTS.exists(), reason="read-only HPC4 endpoint artifact is unavailable"
)


@requires_endpoints
def test_task_budget_fallback_includes_q1_when_it_is_strongest() -> None:
    result = analyze(ENDPOINTS)
    observed = {
        (row["task"], row["budget"]): row["q"]
        for row in result["strong_task_budget_fallback"]
    }
    assert observed == EXPECTED_FALLBACK
    assert observed[("acrobot", "message_binding")] == 1
    assert observed[("cartpole", "message_binding")] == 1


@requires_endpoints
def test_current_oracle_ceiling_fails_both_new_gates() -> None:
    result = analyze(ENDPOINTS)
    ceiling = result["cellwise_fixed_q_oracle"]
    assert abs(ceiling["relative_geometric_improvement"] - 0.003845573166280647) < 1e-12
    assert ceiling["cells_at_least_2_percent"] == 6
    assert ceiling["strictly_improved_cells"] == 21
    gates = result["gate_audit"]
    assert not gates["aggregate_gate_pass"]
    assert not gates["directional_gate_pass"]
    assert gates["exp017b_permanently_stopped"]


@requires_endpoints
def test_full_probe_cost_reduces_usable_learning_updates() -> None:
    result = analyze(ENDPOINTS)
    ceiling = result["full_probe_cost_optimistic_ceiling"]
    for arm in ("fallback", "oracle"):
        metrics = ceiling[arm]
        assert metrics["mean_usable_learning_updates"] < metrics["mean_no_probe_learning_updates"]
        assert 0.0 < metrics["usable_over_no_probe_update_fraction"] < 1.0
        assert metrics["mean_probe_message_bytes"] > 0.0
        assert metrics["mean_probe_environment_steps"] > 0.0


def test_probe_schedule_is_complete_and_persistent() -> None:
    assert probe_count(1) == 1
    assert probe_count(16 * 8) == 8
    assert probe_count(16 * 33) == 9


def test_variance_benefit_saturates_at_high_correlation() -> None:
    q = np.asarray([1, 4, 16, 32])
    low = variance_factor(q, 0.0)
    high = variance_factor(q, 0.9)
    assert low[-1] < low[1] < low[0]
    assert (high[0] - high[-1]) < (low[0] - low[-1])
    assert high[-1] >= 0.9


def test_public_cost_ratio_can_place_low_rho_message_optimum_inside_grid() -> None:
    # Outcome-free candidate models: one-hot FrozenLake MLP (3169 parameters)
    # and small globally pooled MinAtar CNN (1521 parameters).
    for parameters in (3169, 1521):
        optimum = continuous_message_optimum(
            rho=0.1,
            server_overhead=65_536,
            per_agent_payload=4 * parameters,
        )
        assert 4.0 <= optimum <= 16.0


def test_delay_changes_public_usable_horizon() -> None:
    kwargs = dict(
        message_budget=134_217_728,
        environment_budget=4096,
        q=4,
        b=2,
        parameters=3169,
    )
    zero = usable_horizon(delay_p90=0.0, **kwargs)
    delayed = usable_horizon(delay_p90=16.0, **kwargs)
    assert delayed < zero


def test_surrogate_is_vectorized_and_uses_only_scalar_observables() -> None:
    state = ObservableState(
        signal_sq_lcb=1.0,
        noise_variance_ucb=2.0,
        rho_upper=0.5,
        mixing_time_upper=4.0,
        delay_bias_ucb=0.1,
        smoothness_upper=1.0,
        learning_rate=0.01,
        message_price=1e-8,
        environment_price=1e-3,
    )
    values = learning_value_lcb(
        q=np.asarray([1, 4, 16, 32]),
        b=np.asarray([1, 1, 2, 4]),
        message_cost=np.asarray([1e5, 2e5, 4e5, 8e5]),
        state=state,
        delay_p90=12.0,
        confidence_radius=np.full(4, 0.01),
    )
    assert values.shape == (4,)
    assert np.isfinite(values).all()


def test_safety_shield_falls_back_when_certified_surplus_is_insufficient() -> None:
    values = np.asarray([0.0, 0.1, 0.3])
    choice, wealth = shielded_choice(values, fallback_index=2, safety_wealth=0.0)
    assert choice == 2
    assert wealth == 0.0


def test_safety_shield_preserves_nonnegative_certified_wealth() -> None:
    wealth = 0.2
    values = np.asarray([0.5, 0.2, 0.1])
    choice, wealth = shielded_choice(values, fallback_index=2, safety_wealth=wealth)
    assert choice == 0
    assert wealth >= 0.0

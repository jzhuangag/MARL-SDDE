from __future__ import annotations

import numpy as np
import pytest

from .async_packet_game import (
    AsyncGameCell,
    _signed_h4_choice,
    make_budget_cells,
    simulate,
)
from .run_online_model_dev_audit import audit


def small_cell(coupling: float = 0.9) -> AsyncGameCell:
    return AsyncGameCell(
        coupling=coupling,
        maximum_extra_delay=2,
        message_budget=1.5,
        environment_budget=12.0,
        mode_switch_probability=0.08,
    )


def test_simulation_is_deterministic_and_finite() -> None:
    first = simulate(small_cell(), "packet_debt", 41001, launches=40)
    second = simulate(small_cell(), "packet_debt", 41001, launches=40)
    assert first == second
    assert first["finite"]


def test_receipt_control_and_update_trajectory_charge_is_exact() -> None:
    row = simulate(small_cell(), "fixed_h4", 41002, launches=40)
    assert row["environment_per_launch"] == 8.0


def test_fixed_receipt_cap_uses_one_fully_charged_trajectory() -> None:
    row = simulate(small_cell(), "signed_oracle_graph_h4", 41002, launches=40)
    assert row["environment_per_launch"] == 4.0
    assert row["messages_per_launch"] <= 1.0


def test_online_model_score_is_deterministic_and_fully_charged() -> None:
    first = simulate(small_cell(), "signed_online_model_graph_h4", 41005, launches=40)
    second = simulate(small_cell(), "signed_online_model_graph_h4", 41005, launches=40)
    assert first == second
    assert first["environment_per_launch"] == 4.0
    assert first["messages_per_launch"] <= 1.0
    assert first["target_estimation_error"] >= 0.0
    assert len(first["graph_trace_sha256"]) == 64


def test_signed_choice_interface_has_no_environment_target_argument() -> None:
    assert "target" not in _signed_h4_choice.__annotations__


def test_signed_choice_is_translation_equivariant() -> None:
    theta = np.asarray([0.6, -0.2, 0.9])
    reference = np.asarray([0.1, 0.3, -0.4])
    cache = np.asarray(
        [[0.6, -0.5, 0.9], [0.1, -0.2, 0.7], [0.4, -0.1, 0.9]]
    )
    hessian = np.asarray(
        [[1.0, -0.2, -0.1], [-0.2, 1.0, -0.2], [-0.1, -0.2, 1.0]]
    )
    kwargs = {
        "eligible_edges": ((0, 1), (2, 1)),
        "conditional_row": hessian[1],
        "owner": 1,
        "step_cap": 0.04,
        "smoothness": 1.4,
        "gradient_variance": 0.2,
        "message_queue": 0.01,
        "learning_weight": 10.0,
    }
    original = _signed_h4_choice(
        theta=theta,
        score_reference=reference,
        cache=cache,
        hessian=hessian,
        **kwargs,
    )
    shift = np.full(3, 7.0)
    shifted = _signed_h4_choice(
        theta=theta + shift,
        score_reference=reference + shift,
        cache=cache + shift,
        hessian=hessian,
        **kwargs,
    )
    assert original[0] == shifted[0]
    assert original[1] == pytest.approx(shifted[1], abs=1e-12)


def test_online_oracle_audit_is_complete() -> None:
    result = audit([41006], launches=12)
    assert len(result["rows"]) == 24
    assert result["metrics"]["active_comparisons"] == 16


def test_fixed_pair_is_fully_charged() -> None:
    cell = AsyncGameCell(0.9, 2, 2.0, 12.0, 0.08)
    row = simulate(cell, "fixed_offsets01_h4", 41002, launches=40)
    assert row["environment_per_launch"] == 4.0
    assert row["messages_per_launch"] == 2.0


def test_fixed_pair_is_rate_limited_by_lower_budget() -> None:
    row = simulate(small_cell(), "fixed_offsets01_h4", 41002, launches=40)
    assert row["messages_per_launch"] <= small_cell().message_budget


def test_uncoupled_packet_debt_sends_no_messages() -> None:
    row = simulate(small_cell(coupling=0.0), "packet_debt", 41003, launches=40)
    assert row["messages_per_launch"] == 0.0


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        simulate(small_cell(), "not-a-policy", 1, launches=2)


def test_budget_grid_contains_binding_and_uncoupled_controls() -> None:
    cells = make_budget_cells()
    assert len(cells) == 24
    assert {cell.message_budget for cell in cells} == {0.5, 1.0}
    assert sum(cell.coupling == 0.0 for cell in cells) == 8

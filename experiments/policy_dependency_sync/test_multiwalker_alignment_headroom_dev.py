from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pytest

from .multiwalker_alignment_headroom_dev import (
    AlignmentLocalOption,
    AlignmentOracleRow,
    POLICIES,
    analyze_alignment_rows,
    joint_weight_decrease,
    randomized_epoch_owner_schedule,
    run_alignment_scenario,
    solve_prefix_alignment_oracle,
)


def test_low_drift_chunk_summary_is_finite_and_not_a_gate_population() -> None:
    row = AlignmentOracleRow(
        seed=1,
        drift_scale=0.01,
        budget_rate=0.25,
        exact_oracle_decrease=1.1,
        exact_oracle_spent_edges=1,
        exact_oracle_nonzero_weights=1,
        strong_online_policy="no_refresh",
        strong_online_decrease=1.0,
        no_refresh_decrease=1.0,
        ideal_reference_decrease=2.0,
        oracle_gain_over_no_refresh=0.1,
        oracle_headroom_over_strong=0.1,
        oracle_recovery_gap=1.0,
        normalized_absolute_headroom=0.05,
        optimizer_success=True,
        optimizer_status=0,
        optimizer_mip_gap=0.0,
        selected_owner_count=5,
        maximum_prefix_excess=0,
        charged_diagnostic_transitions=10,
        public_trace_transitions=1,
        reference_replay_error=0.0,
        minimum_owner_launches=4,
    )
    summary = analyze_alignment_rows([row])
    assert math.isfinite(summary["active_direction_rate"])
    assert math.isfinite(summary["active_median_normalized_absolute_headroom"])
    assert not summary["gates"]["H8_active_oracle_gain"]
    assert not summary["gates"]["H12_joint_weight_nontrivial"]
    assert not summary["all_gates_pass"]


def test_randomized_epoch_owner_schedule_is_balanced_and_reproducible() -> None:
    first = randomized_epoch_owner_schedule(
        seed=11, events=20, walkers=5, schedule_seed=23
    )
    second = randomized_epoch_owner_schedule(
        seed=11, events=20, walkers=5, schedule_seed=23
    )
    assert first == second
    assert first != tuple(index % 5 for index in range(20))
    assert all(first.count(owner) == 4 for owner in range(5))
    assert all(set(first[start : start + 5]) == set(range(5)) for start in range(0, 20, 5))


def test_frozen_alignment_gate_configuration_matches_runner_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "docs" / "multiwalker_alignment_headroom_dev_config_20260908.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["experiment_id"] == "MW-AH-DEV-001"
    assert config["development_seeds"] == [96300, 96301, 96302, 96303]
    assert config["events"] == 20 and config["horizon"] == 4
    assert config["policies"] == list(POLICIES)
    assert set(config["mandatory_gates"]) == {
        f"H{index}_{suffix}"
        for index, suffix in enumerate(
            (
                "complete_finite",
                "exact_milp_optimum",
                "prefix_budget",
                "deterministic_replay",
                "owner_coverage",
                "fully_charged_diagnostic",
                "reference_signal",
                "active_oracle_gain",
                "active_recovery",
                "active_direction",
                "active_normalized_headroom",
                "joint_weight_nontrivial",
            ),
            start=1,
        )
    }


def test_joint_weight_decrease_solves_the_scalar_lyapunov_problem() -> None:
    reference = np.asarray([2.0, 0.0])
    packet = np.asarray([1.0, 1.0])
    score = joint_weight_decrease(
        reference_gradient=reference,
        packet_gradient=packet,
        learning_smoothness=2.0,
        maximum_packet_weight=1.0,
    )
    assert score.alignment == pytest.approx(2.0)
    assert score.packet_norm_squared == pytest.approx(2.0)
    assert score.packet_weight == pytest.approx(0.5)
    assert score.lyapunov_decrease == pytest.approx(0.5)
    for weight in np.linspace(0.0, 1.0, 101):
        trial = weight * 2.0 - 0.5 * 2.0 * weight**2 * 2.0
        assert trial <= score.lyapunov_decrease + 1e-12


def test_joint_weight_shuts_off_antialigned_packet() -> None:
    score = joint_weight_decrease(
        reference_gradient=np.asarray([1.0, 0.0]),
        packet_gradient=np.asarray([-1.0, 0.0]),
        learning_smoothness=1.0,
        maximum_packet_weight=1.0,
    )
    assert score.alignment == -1.0
    assert score.packet_weight == 0.0
    assert score.lyapunov_decrease == 0.0


def test_prefix_milp_respects_global_budget_with_irregular_owner_events() -> None:
    options = [
        [
            AlignmentLocalOption(0, (0, 3), (0, 0), ((), ()), 0.0),
            AlignmentLocalOption(0, (0, 3), (0, 1), ((), (1,)), 3.0),
            AlignmentLocalOption(0, (0, 3), (1, 0), ((1,), ()), 4.0),
        ],
        [
            AlignmentLocalOption(1, (1, 2), (0, 0), ((), ()), 0.0),
            AlignmentLocalOption(1, (1, 2), (0, 1), ((), (0,)), 2.0),
            AlignmentLocalOption(1, (1, 2), (1, 0), ((0,), ()), 5.0),
        ],
    ]
    value, spent, success, status, gap, selected = solve_prefix_alignment_oracle(
        options_by_owner=options,
        budget_rate=0.5,
        events=4,
    )
    assert success and status == 0 and gap <= 1e-9
    assert value == pytest.approx(8.0)
    assert spent == 2
    assert len(selected) == 2


def test_small_multiwalker_alignment_scenario_is_finite_and_charged() -> None:
    rows = run_alignment_scenario(
        seed=96991,
        drift_scale=0.01,
        budget_rates=(0.5,),
        events=5,
        horizon=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.optimizer_success and row.optimizer_status == 0
    assert row.optimizer_mip_gap <= 1e-9
    assert row.maximum_prefix_excess <= 0
    assert row.reference_replay_error <= 1e-12
    assert row.minimum_owner_launches == 1
    assert row.charged_diagnostic_transitions > row.public_trace_transitions
    assert all(
        math.isfinite(value)
        for value in (
            row.exact_oracle_decrease,
            row.strong_online_decrease,
            row.no_refresh_decrease,
            row.ideal_reference_decrease,
        )
    )

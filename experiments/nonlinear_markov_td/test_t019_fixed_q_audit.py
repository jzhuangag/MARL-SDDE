from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pytest

from t019_fixed_q_audit import (
    AUDIT_POLICIES,
    B_CANDIDATES,
    BUDGETS,
    DELAY_TRACES,
    MIXING_PROFILES,
    Q_CANDIDATES,
    TASKS,
    absorbing_state_grid_audit,
    analyze_endpoints,
    frozen_score,
    parameter_count,
)


ENDPOINTS = Path(__file__).resolve().parents[2] / "tmp" / "t019" / "endpoints.csv"
FROZEN_RUNNER = Path(__file__).with_name("run_exp017a_nonlinear_pilot.py")
requires_pilot_endpoints = pytest.mark.skipif(
    not ENDPOINTS.exists(), reason="read-only HPC4 pilot artifact is not in a clean clone"
)


def test_parameter_counts_match_frozen_network() -> None:
    assert parameter_count(4) == 4545
    assert parameter_count(6) == 4673


def test_proof_targets_the_exact_frozen_runner_and_control_flow() -> None:
    source_bytes = FROZEN_RUNNER.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == (
        "0713b8ff2937f4d7c383acb2ef1b9114d1d9a17e82fb68cf79d3607b0237e452"
    )
    source = source_bytes.decode("utf-8")
    assert "if state.collision_trials == 0:" in source
    assert "return 0.0, 1.0, 1.0" in source
    assert "planning_rho = rho_upper" in source
    assert "rho + (1.0 - rho) / float(action.q)" in source
    assert "if len(states) < 2:" in source
    assert "return 0, 0" in source
    assert "state.collision_trials += trials" in source


def test_q1_weakly_dominates_for_rho_upper_one_at_every_registered_cell() -> None:
    rows = absorbing_state_grid_audit()
    assert len(rows) == len(TASKS) * len(MIXING_PROFILES) * len(DELAY_TRACES) * len(BUDGETS)
    assert all(row["same_b_q1_weak_domination"] for row in rows)
    assert all(row["selected_q"] == 1 for row in rows)


def test_q1_domination_holds_for_arbitrary_positive_learning_summaries() -> None:
    parameters = parameter_count(6)
    for loss, gradient, progress in ((1.0, 1.0, 0.0), (0.1, 25.0, 0.2), (8.0, 0.01, -0.5)):
        for budget in BUDGETS.values():
            for mixing in MIXING_PROFILES.values():
                for delay_p90 in (0.0, 3.0, 12.0):
                    for b in B_CANDIDATES:
                        baseline = frozen_score(
                            1,
                            b,
                            1.0,
                            float(mixing["lambda_upper"]),
                            delay_p90,
                            int(budget["message_bytes"]),
                            int(budget["environment_steps"]),
                            parameters,
                            loss,
                            gradient,
                            progress,
                        )
                        assert all(
                            baseline
                            <= frozen_score(
                                q,
                                b,
                                1.0,
                                float(mixing["lambda_upper"]),
                                delay_p90,
                                int(budget["message_bytes"]),
                                int(budget["environment_steps"]),
                                parameters,
                                loss,
                                gradient,
                                progress,
                            )
                            for q in Q_CANDIDATES
                        )


@requires_pilot_endpoints
def test_phase_analyzer_uses_only_requested_arms_and_complete_cells() -> None:
    cells, long, summary = analyze_endpoints(ENDPOINTS)
    assert set(long["policy"]) == set(AUDIT_POLICIES)
    assert len(cells) == 72
    assert len(long) == 72 * len(AUDIT_POLICIES)
    assert summary["input"]["endpoint_rows_used"] == 72 * 2 * len(AUDIT_POLICIES)
    assert summary["pilot_seed_count_per_cell_arm"] == 2


@requires_pilot_endpoints
def test_best_fixed_and_ratios_are_internally_consistent() -> None:
    cells, long, _summary = analyze_endpoints(ENDPOINTS)
    assert set(cells["best_fixed_q"]).issubset({1, 4, 16, 32})
    best_rows = long[long["policy"] == long["best_fixed_policy"]]
    assert np.allclose(
        best_rows["geometric_terminal_error_ratio_to_best_fixed"], 1.0
    )
    non_oracle = long[long["policy"] != "oracle_evaluation_only"]
    assert (
        non_oracle["geometric_terminal_error_ratio_to_best_fixed"] >= 1.0 - 1e-12
    ).all()


@requires_pilot_endpoints
def test_always_all_is_a_registered_duplicate_of_fixed_q32_for_terminal_error() -> None:
    _cells, _long, summary = analyze_endpoints(ENDPOINTS)
    assert summary["always_all_terminal_error_exactly_matches_fixed_q32"]


@requires_pilot_endpoints
def test_cvar90_is_two_seed_maximum() -> None:
    _cells, long, _summary = analyze_endpoints(ENDPOINTS)
    endpoints = pd.read_csv(ENDPOINTS)
    key = long.iloc[0]
    raw = endpoints[
        (endpoints["task"] == key["task"])
        & (endpoints["mixing"] == key["mixing"])
        & (endpoints["rho"] == key["rho"])
        & (endpoints["delay_trace"] == key["delay_trace"])
        & (endpoints["budget"] == key["budget"])
        & (endpoints["policy"] == key["policy"])
    ]["terminal_prediction_mse"]
    assert len(raw) == 2
    assert key["cvar90_terminal_prediction_mse"] == raw.max()

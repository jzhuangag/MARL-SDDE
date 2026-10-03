from __future__ import annotations

import json
import math

from .multiwalker_budget_oracle_amendment import (
    LocalOption,
    ExactOracleRow,
    local_owner_options,
    merge_exact_rows,
    solve_global_prefix_oracle,
    _public_trace,
)


def test_exact_local_options_and_prefix_milp_dominate_feasible_sequences() -> None:
    prefix, snapshots, initial = _public_trace(
        seed=95910, drift_scale=0.04, walkers=3, events=6, horizon=2
    )
    options = [
        local_owner_options(
            seed=95910,
            drift_scale=0.04,
            owner=owner,
            prefix_actions=prefix,
            actor_snapshots=snapshots,
            initial_actors=initial,
            walkers=3,
            events=6,
            horizon=2,
            discount=0.99,
        )
        for owner in range(3)
    ]
    value, spent, success, status, gap, selected = solve_global_prefix_oracle(
        options_by_owner=options,
        budget_rate=0.5,
        events=6,
        walkers=3,
    )
    assert success and status == 0 and gap <= 1e-9
    assert len(selected) == 3
    assert spent <= math.floor(0.5 * 6)
    assert value == sum(option.total_h_return for option in selected)
    all_zero = sum(
        next(option.total_h_return for option in owner if sum(option.cost_sequence) == 0)
        for owner in options
    )
    assert value >= all_zero - 1e-12


def test_global_oracle_respects_every_prefix_not_only_terminal_budget() -> None:
    options = [
        [
            LocalOption(0, (0, 0), ((), ()), 0.0),
            LocalOption(0, (1, 0), ((1,), ()), 10.0),
            LocalOption(0, (0, 1), ((), (1,)), 4.0),
        ],
        [LocalOption(1, (0, 0), ((), ()), 0.0)],
    ]
    value, spent, success, status, gap, selected = solve_global_prefix_oracle(
        options_by_owner=options,
        budget_rate=0.5,
        events=4,
        walkers=2,
    )
    assert success and status == 0 and gap <= 1e-9
    # Owner 0 launches at global times 0 and 2. A token is unavailable at time
    # zero, so the nominally larger early option is infeasible.
    assert value == 4.0
    assert spent == 1
    assert selected[0].cost_sequence == (0, 1)


def test_chunk_merge_rejects_duplicate_cells(tmp_path) -> None:
    row = ExactOracleRow(
        seed=1,
        drift_scale=0.04,
        budget_rate=0.5,
        exact_oracle_h_return=1.0,
        exact_oracle_spent_edges=1,
        strong_online_policy="age",
        strong_online_h_return=0.0,
        no_refresh_h_return=-1.0,
        oracle_gain_over_no_refresh=2.0,
        oracle_headroom_over_strong=1.0,
        oracle_recovery_gap=0.5,
        normalized_absolute_headroom=1.0,
        optimizer_success=True,
        optimizer_status=0,
        optimizer_mip_gap=0.0,
        selected_owner_count=5,
        maximum_prefix_excess=0,
        no_refresh_replay_error=0.0,
    )
    payload = json.dumps([row.__dict__])
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(payload, encoding="utf-8")
    second.write_text(payload, encoding="utf-8")
    assert merge_exact_rows([first]) == [row]
    try:
        merge_exact_rows([first, second])
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate amended oracle cell was accepted")

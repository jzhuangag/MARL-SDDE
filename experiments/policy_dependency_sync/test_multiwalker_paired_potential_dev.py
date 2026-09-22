from __future__ import annotations

import pytest

from .multiwalker_paired_potential_dev import (
    CACHE_RADIUS_UPPER,
    DESIGN_SEEDS,
    PotentialRow,
    QUEUE_STEPS,
    RESET_BONUS_CAPS,
    STRONG_POLICIES,
    analyze,
    _select,
    cache_radius_upper,
    crossfit_configuration,
    reset_benefit,
)


def test_registered_radius_is_the_four_coordinate_drift_bound() -> None:
    assert cache_radius_upper(events=40, walkers=5, maximum_drift=0.04) == pytest.approx(0.64)
    assert CACHE_RADIUS_UPPER == pytest.approx(0.64)


def test_reset_is_exact_quadratic_and_bounded_at_radius() -> None:
    assert reset_benefit(
        parameter_gap=0.32, reset_bonus_cap=0.2, cache_radius=0.64
    ) == pytest.approx(0.05)
    assert reset_benefit(
        parameter_gap=0.64, reset_bonus_cap=0.2, cache_radius=0.64
    ) == pytest.approx(0.2)


def test_queue_can_reverse_the_unpriced_potential_choice() -> None:
    selected, unpriced = _select(
        h_values={(): 1.0, (1,): 1.05},
        reset_by_candidate={(): 0.0, (1,): 0.1},
        capacity=1,
        queue=0.2,
    )
    assert unpriced == (1,)
    assert selected == ()


def test_hard_prefix_capacity_keeps_null_feasible() -> None:
    selected, unpriced = _select(
        h_values={(): 1.0, (1,): 9.0},
        reset_by_candidate={(): 0.0, (1,): 9.0},
        capacity=0,
        queue=0.0,
    )
    assert selected == unpriced == ()


def _row(seed: int, bonus: float, step: float, value: float) -> PotentialRow:
    return PotentialRow(
        seed=seed,
        drift_scale=0.01,
        budget_rate=0.25,
        reset_bonus_cap=bonus,
        queue_step=step,
        selected_h_return=value,
        no_refresh_h_return=0.0,
        spent_edges=0,
        launches=1,
        nonnull_choices=0,
        positive_utility_choices=0,
        queue_changed_choices=0,
        maximum_queue=0.0,
        maximum_prefix_excess=0,
        replay_failures=0,
        maximum_reset_benefit=0.0,
        maximum_cache_radius=0.0,
    )


def test_crossfit_does_not_read_heldout_outcome() -> None:
    heldout = DESIGN_SEEDS[0]
    rows = []
    for seed in DESIGN_SEEDS:
        for bonus in RESET_BONUS_CAPS:
            for step in QUEUE_STEPS:
                training_best = bonus == 0.1 and step == 0.01
                heldout_best = bonus == 0.2 and step == 0.01
                value = (
                    (1000.0 if heldout_best else -1000.0)
                    if seed == heldout
                    else (1.0 if training_best else 0.0)
                )
                rows.append(_row(seed, bonus, step, value))
    assert crossfit_configuration(rows, heldout_seed=heldout) == (0.1, 0.01)


def test_analyzer_uses_authenticated_no_refresh_not_candidate_null_sum() -> None:
    rows = []
    baseline = []
    exact = []
    for seed in DESIGN_SEEDS:
        for drift in (0.01, 0.04):
            for budget in (0.25, 0.5):
                for policy in (*STRONG_POLICIES, "no_refresh"):
                    baseline.append(
                        {
                            "seed": seed,
                            "drift_scale": drift,
                            "budget_rate": budget,
                            "policy": policy,
                            "selected_h_return": 10.0 if policy != "no_refresh" else 5.0,
                        }
                    )
                exact.append(
                    {
                        "seed": seed,
                        "drift_scale": drift,
                        "budget_rate": budget,
                        "exact_oracle_h_return": 12.0,
                    }
                )
                for bonus in RESET_BONUS_CAPS:
                    for step in QUEUE_STEPS:
                        candidate = 11.0 if bonus == 0.1 and step == 0.01 else 9.0
                        row = _row(seed, bonus, step, candidate)
                        rows.append(
                            PotentialRow(
                                **{
                                    **row.__dict__,
                                    "drift_scale": drift,
                                    "budget_rate": budget,
                                    "queue_changed_choices": 1,
                                }
                            )
                        )
    summary = analyze(rows, baseline_records=baseline, exact_records=exact)
    assert summary["gates"]["P1_complete_grid"]
    assert summary["gates"]["P10_candidate_aggregate_gain_over_no_refresh"]
    assert summary["active_recovery_fraction"] == pytest.approx(0.5)

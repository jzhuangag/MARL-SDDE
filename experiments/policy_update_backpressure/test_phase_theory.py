from __future__ import annotations

import numpy as np
import pytest

from experiments.policy_update_backpressure.phase_theory import (
    EventCertificate,
    audit,
    closed_form_step,
    finite_time_event_slack,
    freshness_residual,
    freshness_lyapunov_drift_bound,
    quadratic_sign_flip_example,
    wall_clock_separation_example,
)


def test_closed_form_step_matches_dense_grid() -> None:
    cert = EventCertificate(
        proposal_norm=1.4,
        own_debt=0.2,
        markov_radius=0.1,
        smoothness=1.3,
        cross_to_pending=(0.2, 0.5),
        pending_debts=(0.4, 0.1),
        max_step=0.8,
        potential_weight=2.0,
    )
    step = closed_form_step(cert)
    grid = np.linspace(0.0, cert.max_step, 10001)
    numeric = float(grid[np.argmin([
        freshness_lyapunov_drift_bound(cert, float(x)) for x in grid
    ])])
    assert abs(step-numeric) <= cert.max_step/10000+1e-12


def test_uncertifiable_proposal_is_rejected() -> None:
    cert = EventCertificate(
        proposal_norm=0.2,
        own_debt=0.3,
        markov_radius=0.0,
        smoothness=1.0,
        cross_to_pending=(),
        pending_debts=(),
        max_step=1.0,
    )
    assert cert.freshness_ratio > 1
    assert closed_form_step(cert) == 0.0


def test_pending_debt_can_throttle_an_otherwise_fresh_update() -> None:
    clean = EventCertificate(1.0, 0.1, 0.0, 1.0, (0.8,), (0.0,), 1.0)
    loaded = EventCertificate(1.0, 0.1, 0.0, 1.0, (0.8,), (2.0,), 1.0)
    assert closed_form_step(loaded) < closed_form_step(clean)


def test_sign_flip_example_is_exactly_harmful() -> None:
    result = quadratic_sign_flip_example()
    assert result["freshness_ratio"] > 1
    assert result["gradient_sign_flipped"]
    assert result["stale_update_is_harmful"]


def test_wall_clock_separation_scales_with_slow_agent() -> None:
    short = wall_clock_separation_example(slow_clock=10)
    long = wall_clock_separation_example(slow_clock=160)
    assert 0 < short["freshness_ratio"] < 1
    assert short["pub_fast_queue_is_throttled"]
    assert long["pub_wall_clock_regret_upper_bound"] < 2.0
    assert long["barrier_wall_clock_regret_lower_bound"] == pytest.approx(
        16*short["barrier_wall_clock_regret_lower_bound"]
    )
    assert long["accept_all_wall_clock_regret"] > 8*short["accept_all_wall_clock_regret"]


def test_full_algebra_audit() -> None:
    result = audit()
    assert result["stale_bias_checks"] == 144
    assert result["gain_bound_checks"] == 576
    assert result["finite_time_event_checks"] > 0
    assert result["maximum_closed_form_grid_gap"] <= 0.00025+1e-12


def test_finite_time_residual_is_exact_linear_drift_signal() -> None:
    v = 8.0
    curvature = v*1.0+0.2**2+0.4**2
    cert = EventCertificate(
        proposal_norm=1.2,
        own_debt=0.1,
        markov_radius=0.05,
        smoothness=1.0,
        cross_to_pending=(0.2, 0.4),
        pending_debts=(0.3, 0.1),
        max_step=v/curvature,
        potential_weight=v,
    )
    expected = 1.2-0.15-(0.2*0.3+0.4*0.1)/v
    assert freshness_residual(cert) == pytest.approx(expected)
    assert finite_time_event_slack(cert) >= -1e-12


def test_finite_time_audit_rejects_clipping_cap_or_zero_v() -> None:
    wrong_cap = EventCertificate(
        proposal_norm=1.0,
        own_debt=0.0,
        markov_radius=0.0,
        smoothness=1.0,
        cross_to_pending=(0.2,),
        pending_debts=(0.0,),
        max_step=0.05,
        potential_weight=10.0,
    )
    with pytest.raises(ValueError, match="max_step"):
        finite_time_event_slack(wrong_cap)

    zero_v = EventCertificate(
        proposal_norm=1.0,
        own_debt=0.0,
        markov_radius=0.0,
        smoothness=1.0,
        cross_to_pending=(1.0,),
        pending_debts=(0.0,),
        max_step=1.0,
        potential_weight=0.0,
    )
    with pytest.raises(ValueError, match="V>0"):
        finite_time_event_slack(zero_v)

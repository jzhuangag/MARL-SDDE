import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "learning_progress_controller.py"
)
SPEC = importlib.util.spec_from_file_location("learning_progress_controller", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def certificate(lower):
    return MODULE.AdditiveDriftCertificate(
        progress_lower=lower,
        confidence=0.95,
        samples=100,
        effective_samples=50.0,
        mixing_bias=0.01,
    )


def test_bounded_certificate_penalizes_family_and_mixing():
    before = np.full(200, 0.8)
    after = np.full(200, 0.2)
    base = MODULE.fit_bounded_additive_drift_certificate(
        before,
        after,
        confidence=0.95,
        family_size=1,
        effective_samples=200,
    )
    conservative = MODULE.fit_bounded_additive_drift_certificate(
        before,
        after,
        confidence=0.95,
        family_size=4,
        effective_samples=100,
        mixing_bias=0.05,
    )
    assert 0.0 < conservative.progress_lower < base.progress_lower < 0.6


def test_bounded_certificate_rejects_unbounded_observation():
    with pytest.raises(ValueError):
        MODULE.fit_bounded_additive_drift_certificate(
            np.asarray([0.1, 1.1]),
            np.asarray([0.0, 0.2]),
            confidence=0.95,
            family_size=4,
        )


def test_selector_combines_progress_and_exact_horizon():
    budget = MODULE.ResourceBudget(message_remaining=1000, environment_remaining=100)
    costs = {
        1: MODULE.ActionCost(messages_per_update=10, environment_per_update=1),
        2: MODULE.ActionCost(messages_per_update=20, environment_per_update=1),
        4: MODULE.ActionCost(messages_per_update=50, environment_per_update=1),
    }
    decision = MODULE.select_additive_drift_action(
        budget=budget,
        costs=costs,
        certificates={1: certificate(0.01), 2: certificate(0.03), 4: certificate(0.05)},
        fallback_q=1,
    )
    assert decision.horizons == {1: 100, 2: 50, 4: 20}
    assert decision.selected_q == 2
    assert decision.used_fallback is False


def test_selector_uses_registered_fallback_without_positive_certificate():
    budget = MODULE.ResourceBudget(message_remaining=100, environment_remaining=100)
    costs = {
        1: MODULE.ActionCost(messages_per_update=10, environment_per_update=1),
        2: MODULE.ActionCost(messages_per_update=20, environment_per_update=1),
    }
    decision = MODULE.select_additive_drift_action(
        budget=budget,
        costs=costs,
        certificates={1: certificate(-0.1), 2: certificate(0.0)},
        fallback_q=1,
    )
    assert decision.selected_q == 1
    assert decision.used_fallback is True


def test_bounded_validation_risk_is_monotone_and_bounded():
    returns = np.asarray([-100.0, 0.0, 100.0])
    risk = MODULE.bounded_validation_risk(returns, center=0.0, scale=10.0)
    assert np.all((risk > 0.0) & (risk < 1.0))
    assert np.all(np.diff(risk) < 0.0)
    assert risk[1] == pytest.approx(0.5)


def test_blocking_plan_spends_coupling_failure_explicitly():
    plan = MODULE.make_beta_mixing_blocking_plan(
        raw_samples=1000,
        burn_in=10,
        stride=20,
        beta_at_stride=1e-5,
        confidence=0.95,
        family_size=4,
    )
    assert plan.selected_samples == 50
    assert plan.coupling_failure == pytest.approx(49e-5)
    assert plan.concentration_failure > 0.0
    assert plan.radius > 0.0


def test_blocking_plan_rejects_unproved_effective_sample_claim():
    with pytest.raises(ValueError, match="exhausts"):
        MODULE.make_beta_mixing_blocking_plan(
            raw_samples=100,
            burn_in=0,
            stride=2,
            beta_at_stride=0.01,
            confidence=0.95,
            family_size=4,
        )


def test_strict_certificate_charges_transfer_remainder():
    plan = MODULE.make_beta_mixing_blocking_plan(
        raw_samples=2000,
        burn_in=0,
        stride=20,
        beta_at_stride=1e-6,
        confidence=0.9,
        family_size=2,
    )
    before = np.full(2000, 0.9)
    after = np.full(2000, 0.1)
    base = MODULE.fit_blocked_additive_drift_certificate(
        before, after, plan=plan
    )
    penalized = MODULE.fit_blocked_additive_drift_certificate(
        before,
        after,
        plan=plan,
        mean_bias=0.03,
        transfer_penalty=MODULE.trust_region_transfer_penalty(
            drift_lipschitz_upper=0.5, policy_radius=0.2
        ),
    )
    assert penalized.progress_lower == pytest.approx(base.progress_lower - 0.13)


def test_validation_charge_counts_all_workers_and_ticks():
    charge = MODULE.charge_validation_block(workers=8, horizon=25)
    assert charge.messages == 200
    assert charge.environment_ticks == 200


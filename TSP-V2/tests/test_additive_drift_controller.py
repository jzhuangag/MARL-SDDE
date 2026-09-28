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


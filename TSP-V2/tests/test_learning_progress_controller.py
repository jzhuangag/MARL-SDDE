import importlib.util
import sys
from pathlib import Path

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


def test_dual_budget_horizon_uses_binding_resource():
    budget = MODULE.ResourceBudget(1000, 100)
    cost = MODULE.ActionCost(100, 30)
    assert MODULE.feasible_horizon(budget, cost) == 3


def test_affine_terminal_risk_matches_recursion():
    certificate = MODULE.DriftCertificate(0.8, 0.2, 0.95, 32)
    closed = MODULE.affine_terminal_risk(4.0, 7, certificate)
    value = 4.0
    for _ in range(7):
        value = 0.8 * value + 0.2
    assert closed == pytest.approx(value)


def test_interior_action_can_be_selected():
    budget = MODULE.ResourceBudget(10_000, 10_000)
    costs = {
        1: MODULE.ActionCost(10, 10),
        2: MODULE.ActionCost(12, 10),
        4: MODULE.ActionCost(16, 10),
        8: MODULE.ActionCost(24, 10),
    }
    certificates = {
        1: MODULE.DriftCertificate(0.998, 0.020, 0.95, 64),
        2: MODULE.DriftCertificate(0.985, 0.012, 0.95, 64),
        4: MODULE.DriftCertificate(0.990, 0.030, 0.95, 64),
        8: MODULE.DriftCertificate(0.992, 0.050, 0.95, 64),
    }
    decision = MODULE.select_learning_progress_action(
        initial_lyapunov=10.0,
        budget=budget,
        costs=costs,
        certificates=certificates,
    )
    assert decision.selected_q == 2


def test_more_parallelism_wins_when_contraction_is_strong():
    budget = MODULE.ResourceBudget(10_000, 10_000)
    costs = {
        1: MODULE.ActionCost(10, 10),
        8: MODULE.ActionCost(24, 10),
    }
    certificates = {
        1: MODULE.DriftCertificate(0.999, 0.02, 0.95, 64),
        8: MODULE.DriftCertificate(0.970, 0.01, 0.95, 64),
    }
    decision = MODULE.select_learning_progress_action(
        initial_lyapunov=10.0,
        budget=budget,
        costs=costs,
        certificates=certificates,
    )
    assert decision.selected_q == 8


def test_catalogue_mismatch_is_rejected():
    with pytest.raises(ValueError):
        MODULE.select_learning_progress_action(
            initial_lyapunov=1.0,
            budget=MODULE.ResourceBudget(100, 100),
            costs={1: MODULE.ActionCost(10, 10)},
            certificates={},
        )


def test_affine_fit_covers_low_noise_coefficients():
    import numpy as np

    rng = np.random.default_rng(7)
    x = np.linspace(2.0, 10.0, 128)
    a, c = 0.82, 0.15
    y = np.maximum(a * x + c + rng.normal(0.0, 0.01, x.size), 0.0)
    certificate = MODULE.fit_affine_drift_certificate(
        x,
        y,
        confidence=0.99,
        noise_scale=0.01,
        parameter_radius=1.5,
        ridge=0.01,
    )
    assert certificate.a_upper >= a
    assert certificate.c_upper >= c

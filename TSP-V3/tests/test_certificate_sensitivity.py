from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "TSP-V3" / "experiments" / "certificate_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("certificate_sensitivity", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_preregistration_has_disjoint_fresh_seeds() -> None:
    config = json.loads(
        (ROOT / "TSP-V3" / "experiments" / "certificate_sensitivity_preregistration.json").read_text(encoding="utf-8")
    )
    development = set(
        range(config["development_seeds"]["start"], config["development_seeds"]["start"] + config["development_seeds"]["count"])
    )
    confirmation = set(
        range(config["confirmation_seeds"]["start"], config["confirmation_seeds"]["start"] + config["confirmation_seeds"]["count"])
    )
    assert development.isdisjoint(confirmation)
    assert min(development | confirmation) > 32_000_064


def test_mixing_remainder_inflation_is_conservative() -> None:
    for delta in (1e-6, 1e-3, 0.2, 0.9):
        inflated = MODULE.inflated_delta(delta, 0.1)
        assert delta <= inflated <= 1.0


def test_correlation_only_phase_moves_from_large_to_small_q() -> None:
    counts = (1, 4, 16, 32)
    assert MODULE.correlation_only_q(0.0, counts) == 32
    assert MODULE.correlation_only_q(0.9, counts) == 1


def test_coefficient_perturbation_bound_is_nonnegative() -> None:
    nominal = {
        "a_delta": 0.81,
        "beta_delta": 0.02,
        "h_delay": 0.01,
        "g_delay": 0.003,
    }
    conservative = {
        "a_delta": 0.86,
        "beta_delta": 0.03,
        "h_delay": 0.01,
        "g_delay": 0.003,
    }
    result = MODULE.coefficient_perturbations(nominal, conservative)
    assert result["delta_a"] > 0.0
    assert result["delta_beta"] > 0.0
    assert result["delta_c"] > 0.0
    assert result["delta_d_bound"] > 0.0


def test_zero_delay_forcing_perturbation_is_exact() -> None:
    nominal = {
        "a_delta": 0.81,
        "beta_delta": 0.02,
        "h_delay": 0.0,
        "g_delay": 0.0,
    }
    conservative = {
        "a_delta": 0.86,
        "beta_delta": 0.03,
        "h_delay": 0.0,
        "g_delay": 0.0,
    }
    result = MODULE.coefficient_perturbations(nominal, conservative)
    assert np.isclose(result["delta_d_bound"], 0.01)

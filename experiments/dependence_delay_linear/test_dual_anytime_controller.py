"""Tests for dual anytime mixing/correlation confidence control."""

import numpy as np

from dual_anytime_controller import (
    ANYTIME_ALPHA,
    block_observation_counts,
    dual_confidence_bounds,
    log_beta_binomial_mixture_ratio,
    mixture_upper_confidence,
    rounded_upper,
)
from markov_jump_ms import (
    aggregate_same_time_curvature,
    registered_expanding_td_model,
)


def test_two_stream_mixture_bounds_split_alpha():
    bounds = dual_confidence_bounds(64, 128, 32, 128, 1)
    assert bounds["alpha_each"] == ANYTIME_ALPHA / 2.0


def test_mixture_upper_lies_on_eprocess_boundary():
    upper = mixture_upper_confidence(50, 128, 0.005)
    log_ratio = log_beta_binomial_mixture_ratio(
        upper, 50, 128
    )
    assert np.isclose(log_ratio, np.log(1.0 / 0.005), atol=1e-10)
    assert upper > 50.0 / 128.0


def test_one_step_mixture_ratio_has_unit_null_expectation():
    probability = 0.37
    success = np.exp(
        log_beta_binomial_mixture_ratio(probability, 1, 1)
    )
    failure = np.exp(
        log_beta_binomial_mixture_ratio(probability, 0, 1)
    )
    assert np.isclose(
        probability * success + (1.0 - probability) * failure,
        1.0,
    )


def test_registered_curvature_is_monotone_in_sharing():
    model = registered_expanding_td_model()
    for q in (1, 2, 4, 8, 16, 32):
        values = [
            aggregate_same_time_curvature(model, q, rho)[1]
            for rho in np.linspace(0.0, 1.0, 11)
        ]
        assert np.all(np.diff(values) >= -1e-12)


def test_rounding_is_upward_and_bounded():
    assert rounded_upper(0.901, 0.02) == 0.92
    assert rounded_upper(0.999, 0.02) == 1.0
    assert rounded_upper(0.0, 0.02) == 0.0


def test_dual_bounds_cover_all_success_and_zero_success_edges():
    bounds = dual_confidence_bounds(128, 128, 0, 128, 1)
    assert bounds["persistence_upper"] == 1.0
    assert 0.0 < bounds["rho_upper"] < 0.1
    assert bounds["certified_rho"] >= bounds["rho_upper"]


def test_block_observations_charge_pair_probes():
    action = {"gap": 10, "num_agents": 1}
    counts = block_observation_counts(action, 5, 99)
    assert counts == {
        "pair_probes": 9,
        "transition_trials": 59,
        "sharing_trials": 9,
    }
    action["num_agents"] = 4
    counts = block_observation_counts(action, 5, 99)
    assert counts["sharing_trials"] == 14

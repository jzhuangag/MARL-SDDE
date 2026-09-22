import math

import numpy as np

from .conformal_cone import (
    cone_gradient_error_bound,
    gaussian_trajectory_kl_bound,
    shifted_escape_probability,
    split_conformal_radius,
    total_variation_from_kl,
)


def test_conformal_rank_and_small_sample_refusal():
    residuals = np.arange(1.0, 21.0)
    assert split_conformal_radius(residuals, 0.1) == 19.0
    assert math.isinf(split_conformal_radius(np.arange(10.0), 0.05))


def test_conformal_is_permutation_invariant():
    residuals = np.asarray([0.4, 0.1, 0.9, 0.2, 0.5, 0.3, 0.8, 0.7, 0.6])
    shuffled = np.random.default_rng(3).permutation(residuals)
    assert split_conformal_radius(residuals, 0.2) == split_conformal_radius(shuffled, 0.2)


def test_gaussian_shift_chain_rule_and_pinsker():
    kl = gaussian_trajectory_kl_bound(8, 0.1, 0.4)
    assert np.isclose(kl, 0.25)
    assert np.isclose(total_variation_from_kl(kl), math.sqrt(0.125))


def test_escape_and_gradient_tail_are_capped_and_explicit():
    assert np.isclose(shifted_escape_probability(0.1, 0.2), 0.3)
    assert shifted_escape_probability(0.7, 0.5) == 1.0
    assert cone_gradient_error_bound(5.0, 3.0, 0.1) == 6.0

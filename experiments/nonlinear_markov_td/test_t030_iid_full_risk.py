import itertools

import numpy as np

from experiments.nonlinear_markov_td.t030_iid_full_risk import (
    delayed_iid_cross_step,
    delayed_iid_diagonal_step,
    variance_factor,
)


def test_variance_factor_matches_common_flag_pair_probability() -> None:
    for q in (1, 2, 4, 16):
        for rho in (0.0, 0.1, 0.5, 1.0):
            assert variance_factor(q, rho) == rho + (1.0 - rho) / q


def test_scalar_diagonal_recurrence_matches_exhaustive_batch_enumeration() -> None:
    # Two equiprobable samples (J, xi), with E[xi]=0 but J and xi correlated.
    samples = ((0.5, -1.0), (1.5, 1.0))
    errors = ((2.0, 1.0, 0.25), (-1.0, -2.0, 0.75))  # e_t, e_(t-D), p
    q = 2
    rho = 0.36
    alpha = 0.1
    flag_probability = rho**0.5

    mean_t = sum(e_t * p for e_t, _e_d, p in errors)
    mean_d = sum(e_d * p for _e_t, e_d, p in errors)
    m_tt = sum(e_t * e_t * p for e_t, _e_d, p in errors)
    m_dt = sum(e_d * e_t * p for e_t, e_d, p in errors)
    m_dd = sum(e_d * e_d * p for _e_t, e_d, p in errors)
    mean_j = sum(j for j, _xi in samples) / len(samples)
    ej2 = sum(j * j for j, _xi in samples) / len(samples)
    q_noise = sum(xi * xi for _j, xi in samples) / len(samples)
    ejxi = sum(j * xi for j, xi in samples) / len(samples)

    predicted_mean, predicted_moment = delayed_iid_diagonal_step(
        mean_current=np.array([mean_t]),
        mean_delayed=np.array([mean_d]),
        moment_current=np.array([[m_tt]]),
        moment_delayed_current=np.array([[m_dt]]),
        moment_delayed=np.array([[m_dd]]),
        mean_jacobian=np.array([[mean_j]]),
        jacobian_operator=lambda x: ej2 * x,
        noise_second_moment=np.array([[q_noise]]),
        jacobian_noise_cross=lambda m: ejxi * m.reshape(1, 1),
        alpha=alpha,
        q=q,
        rho=rho,
    )

    exact_mean = 0.0
    exact_second = 0.0
    for e_t, e_d, p_error in errors:
        for flags in itertools.product((False, True), repeat=q):
            p_flags = np.prod(
                [flag_probability if flag else 1.0 - flag_probability for flag in flags]
            )
            for common_index in range(len(samples)):
                for private_indices in itertools.product(range(len(samples)), repeat=q):
                    probability = p_error * p_flags / (len(samples) ** (q + 1))
                    chosen = [
                        samples[common_index] if flags[i] else samples[private_indices[i]]
                        for i in range(q)
                    ]
                    j_bar = sum(j for j, _xi in chosen) / q
                    xi_bar = sum(xi for _j, xi in chosen) / q
                    updated = e_t - alpha * j_bar * e_d + alpha * xi_bar
                    exact_mean += probability * updated
                    exact_second += probability * updated * updated

    np.testing.assert_allclose(predicted_mean, [exact_mean], atol=1e-13)
    np.testing.assert_allclose(predicted_moment, [[exact_second]], atol=1e-13)


def test_cross_block_recurrence() -> None:
    result = delayed_iid_cross_step(
        np.array([[3.0]]), np.array([[2.0]]), np.array([[0.5]]), 0.1
    )
    np.testing.assert_allclose(result, [[2.9]])

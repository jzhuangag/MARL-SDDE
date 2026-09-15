from __future__ import annotations

import itertools

import numpy as np
import pytest

from .stale_correction_qp import (
    choose_reuse_correct_or_refresh,
    exponential_secant_slope,
    solve_tempered_correction_qp,
    update_resource_debt,
)


def _objective(
    alpha: np.ndarray,
    d: np.ndarray,
    v: np.ndarray,
    second_moment: float,
    batch: float,
    prices: np.ndarray,
) -> float:
    slope = exponential_secant_slope(float(np.sum(v)))
    return float(
        np.dot(d, 1.0 - alpha) ** 2
        + second_moment / batch * (1.0 + slope * np.dot(v, alpha * alpha))
        + np.dot(prices, alpha)
    )


def test_secant_really_upper_bounds_exponential() -> None:
    for total in (0.0, 0.1, 1.0, 3.0):
        slope = exponential_secant_slope(total)
        for point in np.linspace(0.0, total, 101):
            assert np.exp(point) <= 1.0 + slope * point + 1e-12


def test_scalar_root_solver_matches_dense_two_dimensional_grid() -> None:
    d = np.asarray([0.3, 0.12])
    v = np.asarray([0.2, 0.8])
    prices = np.asarray([0.01, 0.03])
    decision = solve_tempered_correction_qp(
        bias_sensitivities=d,
        divergence_proxies=v,
        integrand_second_moment=0.4,
        effective_batch_size=8.0,
        correction_prices=prices,
    )
    grid = np.linspace(0.0, 1.0, 501)
    brute = min(
        _objective(np.asarray(candidate), d, v, 0.4, 8.0, prices)
        for candidate in itertools.product(grid, repeat=2)
    )
    assert decision.priced_objective <= brute + 2e-6
    assert np.all((decision.alphas >= 0.0) & (decision.alphas <= 1.0))


def test_more_batch_supports_stronger_correction() -> None:
    kwargs = {
        "bias_sensitivities": np.asarray([0.2, 0.1, 0.05]),
        "divergence_proxies": np.asarray([0.2, 0.4, 0.8]),
        "integrand_second_moment": 1.0,
        "correction_prices": np.zeros(3),
    }
    small = solve_tempered_correction_qp(**kwargs, effective_batch_size=2.0)
    large = solve_tempered_correction_qp(**kwargs, effective_batch_size=32.0)
    assert np.all(large.alphas >= small.alphas - 1e-12)
    assert large.residual_bias_upper <= small.residual_bias_upper + 1e-12


def test_expensive_noisy_factor_receives_less_correction() -> None:
    decision = solve_tempered_correction_qp(
        bias_sensitivities=np.asarray([0.2, 0.2]),
        divergence_proxies=np.asarray([0.1, 1.0]),
        integrand_second_moment=1.0,
        effective_batch_size=8.0,
        correction_prices=np.asarray([0.0, 0.1]),
    )
    assert decision.alphas[0] > decision.alphas[1]


def test_refresh_is_chosen_only_when_feasible_and_strictly_better() -> None:
    kwargs = {
        "bias_sensitivities": np.asarray([0.4, 0.3]),
        "divergence_proxies": np.asarray([0.5, 0.5]),
        "integrand_second_moment": 1.0,
        "effective_batch_size": 4.0,
        "correction_prices": np.zeros(2),
        "refresh_risk_upper": 0.01,
        "refresh_price": 0.0,
    }
    assert choose_reuse_correct_or_refresh(**kwargs, refresh_feasible=True).action == "refresh"
    assert choose_reuse_correct_or_refresh(**kwargs, refresh_feasible=False).action == "correct"


def test_no_staleness_reduces_to_uncorrected_sampling_variance() -> None:
    decision = solve_tempered_correction_qp(
        bias_sensitivities=np.zeros(3),
        divergence_proxies=np.zeros(3),
        integrand_second_moment=2.0,
        effective_batch_size=10.0,
    )
    np.testing.assert_array_equal(decision.alphas, np.zeros(3))
    assert decision.risk_upper == pytest.approx(0.2)


def test_resource_debt_is_reflected_at_zero() -> None:
    assert update_resource_debt(0.1, incurred_cost=0.0, average_budget=0.2) == 0.0
    assert update_resource_debt(0.1, incurred_cost=1.0, average_budget=0.2) == pytest.approx(0.9)


@pytest.mark.parametrize(
    "d,v",
    [
        ([-1.0], [1.0]),
        ([1.0], [0.0]),
        ([1.0, 2.0], [1.0]),
    ],
)
def test_invalid_certificates_are_rejected(d: list[float], v: list[float]) -> None:
    with pytest.raises(ValueError):
        solve_tempered_correction_qp(
            bias_sensitivities=np.asarray(d),
            divergence_proxies=np.asarray(v),
            integrand_second_moment=1.0,
            effective_batch_size=4.0,
        )

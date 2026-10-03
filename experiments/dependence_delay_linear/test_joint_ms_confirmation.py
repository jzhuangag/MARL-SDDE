"""Unit tests for EXP-007D confirmation statistics."""

import numpy as np

from run_joint_ms_confirmation import bootstrap_statistic_limit


def test_bootstrap_limit_is_deterministic() -> None:
    values = np.asarray([0.1, 0.2, 0.3, 0.4])
    first = bootstrap_statistic_limit(
        values, np.random.RandomState(123), "mean", 0.99
    )
    second = bootstrap_statistic_limit(
        values, np.random.RandomState(123), "mean", 0.99
    )
    assert first == second
    assert first >= values.mean()


def test_bootstrap_paired_lower_limit() -> None:
    ratios = np.asarray([3.0, 4.0, 5.0, 6.0])
    lower = bootstrap_statistic_limit(
        ratios, np.random.RandomState(321), "median", 0.01
    )
    assert lower >= 3.0
    assert lower <= np.median(ratios)


from __future__ import annotations

import numpy as np
import pytest

from .factored_markov_packet import conditional_owner_row
from .online_reference_theory import (
    cyclic_normalized_pe_lower_bound,
    nlms_expected_squared_error,
    noiseless_block_contraction,
    signed_score_reference_error_bound,
)


def test_normalized_lms_energy_identity_matches_noise_quadrature() -> None:
    error = np.asarray([0.4, -0.7, 0.2])
    row = np.asarray([1.1, -0.3, 0.4])
    gain = 0.25
    variance = 0.16
    predicted = nlms_expected_squared_error(error, row, gain, variance)
    standard = variance**0.5
    realized = []
    for noise in (-standard, standard):
        next_error = error - gain * row * (row @ error + noise) / (row @ row)
        realized.append(float(next_error @ next_error))
    assert float(np.mean(realized)) == pytest.approx(predicted, abs=1e-12)


@pytest.mark.parametrize("coupling", [0.0, 0.6, 0.9])
def test_cyclic_owner_rows_dominate_analytic_pe_bound(coupling: float) -> None:
    agents = 6
    rng = np.random.default_rng(7163 + int(10 * coupling))
    lower = cyclic_normalized_pe_lower_bound(agents, 0.4, coupling)
    for _ in range(32):
        rows = []
        for owner in range(agents):
            row = conditional_owner_row(
                owner,
                agents,
                0.4,
                coupling,
                start_offset=int(rng.integers(0, agents - 1)),
                move_probability=float(rng.choice([0.1, 0.8])),
                horizon=4,
            )
            rows.append(row / np.linalg.norm(row))
        gram = np.asarray(rows).T @ np.asarray(rows)
        assert float(np.linalg.eigvalsh(gram)[0]) + 1e-12 >= lower


def _score(current: float, packet: float, step: float, smoothness: float, motion: float) -> float:
    return float(
        -step * current * packet
        + 0.5 * smoothness * step * step * packet * packet
        + step * motion * abs(packet)
    )


def test_reference_error_bound_dominates_actual_score_change() -> None:
    true_reference = np.asarray([0.2, -0.1, 0.5])
    estimated_reference = np.asarray([0.35, -0.25, 0.58])
    theta = np.asarray([0.8, 0.3, -0.2])
    cache = np.asarray([0.7, -0.4, 0.1])
    current_row = np.asarray([1.0, -0.2, -0.1])
    packet_row = np.asarray([1.1, -0.4, -0.2])
    step = 0.04
    smoothness = 1.3
    motion = 0.12
    current = float(current_row @ (theta - true_reference))
    packet = float(packet_row @ (cache - true_reference))
    estimated_current = float(current_row @ (theta - estimated_reference))
    estimated_packet = float(packet_row @ (cache - estimated_reference))
    actual = abs(
        _score(estimated_current, estimated_packet, step, smoothness, motion)
        - _score(current, packet, step, smoothness, motion)
    )
    error_norm = float(np.linalg.norm(estimated_reference - true_reference))
    bound = signed_score_reference_error_bound(
        reference_error_norm=error_norm,
        current_row_norm=float(np.linalg.norm(current_row)),
        packet_row_norm=float(np.linalg.norm(packet_row)),
        current_gradient_bound=abs(current),
        packet_mean_bound=abs(packet),
        step=step,
        smoothness=smoothness,
        receipt_motion_bound=motion,
    )
    assert actual <= bound + 1e-12


def test_block_contraction_is_positive_and_conservative() -> None:
    lower = cyclic_normalized_pe_lower_bound(6, 0.4, 0.9)
    coefficient = noiseless_block_contraction(0.25, 6, lower)
    assert 0.0 < coefficient < 1.0

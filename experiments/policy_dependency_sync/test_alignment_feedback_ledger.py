from __future__ import annotations

import numpy as np
import pytest

from .alignment_feedback_ledger import (
    PredictableAlignmentLedger,
    receipt_smoothness_drift_upper,
    reference_corrected_alignment_lower,
)


def test_ledger_uses_launch_reference_and_selected_context_only() -> None:
    ledger = PredictableAlignmentLedger(context_dimension=2, gradient_dimension=3)
    context = np.asarray([0.2, -0.4])
    reference = np.asarray([1.0, 2.0, -1.0])
    packet = ledger.launch(
        launch_event=4,
        receipt_event=7,
        selected_context=context,
        reference_direction=reference,
    )
    context[:] = 99.0
    reference[:] = 99.0
    with pytest.raises(ValueError):
        ledger.receive(
            packet_id=packet,
            event=6,
            packet_gradient=np.asarray([0.5, -1.0, 0.25]),
        )
    observation = ledger.receive(
        packet_id=packet,
        event=7,
        packet_gradient=np.asarray([0.5, -1.0, 0.25]),
    )
    assert np.array_equal(observation.selected_context, np.asarray([0.2, -0.4]))
    assert observation.feedback == pytest.approx(-1.75)
    assert ledger.pending_count == 0
    with pytest.raises(KeyError):
        ledger.receive(
            packet_id=packet,
            event=8,
            packet_gradient=np.zeros(3),
        )


def test_reference_correction_lower_bounds_true_alignment() -> None:
    true_gradient = np.asarray([0.5, -1.0, 0.25])
    reference = np.asarray([0.4, -0.8, 0.2])
    packet_gradient = np.asarray([1.0, 0.5, -0.5])
    observable = float(reference @ packet_gradient)
    lower = reference_corrected_alignment_lower(
        observable_alignment=observable,
        reference_error_norm_upper=float(np.linalg.norm(true_gradient - reference)),
        packet_gradient_norm_upper=float(np.linalg.norm(packet_gradient)),
    )
    assert lower <= float(true_gradient @ packet_gradient) + 1e-12


def test_receipt_drift_bound_matches_quadratic_worst_case() -> None:
    upper = receipt_smoothness_drift_upper(
        observable_alignment=0.8,
        reference_error_norm_upper=0.1,
        packet_gradient_norm_upper=2.0,
        learning_smoothness=1.5,
        receipt_motion_upper=0.05,
        packet_weight=0.2,
    )
    expected_effective = 0.8 - 0.1 * 2.0 - 1.5 * 2.0 * 0.05
    expected = -0.2 * expected_effective + 0.5 * 1.5 * 0.2**2 * 2.0**2
    assert upper == pytest.approx(expected)


def test_alignment_ledger_invalid_inputs_fail_closed() -> None:
    ledger = PredictableAlignmentLedger(context_dimension=2, gradient_dimension=3)
    with pytest.raises(ValueError):
        ledger.launch(
            launch_event=1,
            receipt_event=1,
            selected_context=np.zeros(2),
            reference_direction=np.zeros(3),
        )
    with pytest.raises(ValueError):
        reference_corrected_alignment_lower(
            observable_alignment=0.0,
            reference_error_norm_upper=-1.0,
            packet_gradient_norm_upper=1.0,
        )

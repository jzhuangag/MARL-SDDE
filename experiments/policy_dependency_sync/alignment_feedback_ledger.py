"""Causal launch--receipt ledger for observable gradient-alignment feedback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignmentObservation:
    packet_id: int
    launch_event: int
    receipt_event: int
    selected_context: np.ndarray
    feedback: float
    reference_norm: float
    packet_gradient_norm: float


@dataclass(frozen=True)
class _PendingAlignment:
    launch_event: int
    receipt_event: int
    selected_context: np.ndarray
    reference_direction: np.ndarray


class PredictableAlignmentLedger:
    """Snapshot launch-measurable objects and reveal one selected target later.

    Candidate contexts may be used to choose an edge at launch, but only the
    selected context is registered.  At receipt the observable response is
    ``<v_launch, g_packet>``.  Neither unselected gradients nor a future
    reference direction are exposed to the estimator.
    """

    def __init__(self, *, context_dimension: int, gradient_dimension: int) -> None:
        if context_dimension <= 0 or gradient_dimension <= 0:
            raise ValueError("ledger dimensions must be positive")
        self.context_dimension = int(context_dimension)
        self.gradient_dimension = int(gradient_dimension)
        self._next_packet_id = 0
        self._pending: dict[int, _PendingAlignment] = {}

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def launch(
        self,
        *,
        launch_event: int,
        receipt_event: int,
        selected_context: np.ndarray,
        reference_direction: np.ndarray,
    ) -> int:
        if launch_event < 0 or receipt_event <= launch_event:
            raise ValueError("receipt must occur strictly after launch")
        context = self._vector(selected_context, self.context_dimension, "context")
        reference = self._vector(
            reference_direction, self.gradient_dimension, "reference"
        )
        packet_id = self._next_packet_id
        self._next_packet_id += 1
        self._pending[packet_id] = _PendingAlignment(
            launch_event=int(launch_event),
            receipt_event=int(receipt_event),
            selected_context=context.copy(),
            reference_direction=reference.copy(),
        )
        return packet_id

    def receive(
        self, *, packet_id: int, event: int, packet_gradient: np.ndarray
    ) -> AlignmentObservation:
        if packet_id not in self._pending:
            raise KeyError(f"unknown or already received packet {packet_id}")
        pending = self._pending[packet_id]
        if event < pending.receipt_event:
            raise ValueError("packet feedback was revealed before its receipt event")
        gradient = self._vector(
            packet_gradient, self.gradient_dimension, "packet gradient"
        )
        del self._pending[packet_id]
        return AlignmentObservation(
            packet_id=int(packet_id),
            launch_event=pending.launch_event,
            receipt_event=int(event),
            selected_context=pending.selected_context.copy(),
            feedback=float(pending.reference_direction @ gradient),
            reference_norm=float(np.linalg.norm(pending.reference_direction)),
            packet_gradient_norm=float(np.linalg.norm(gradient)),
        )

    @staticmethod
    def _vector(value: np.ndarray, dimension: int, label: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.shape != (dimension,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must be a finite vector of length {dimension}")
        return array


def reference_corrected_alignment_lower(
    *,
    observable_alignment: float,
    reference_error_norm_upper: float,
    packet_gradient_norm_upper: float,
) -> float:
    """Cauchy--Schwarz lower bound on true-gradient packet alignment."""

    values = (
        observable_alignment,
        reference_error_norm_upper,
        packet_gradient_norm_upper,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("alignment-bound inputs must be finite")
    if reference_error_norm_upper < 0.0 or packet_gradient_norm_upper < 0.0:
        raise ValueError("alignment error and gradient bound must be nonnegative")
    return float(
        observable_alignment
        - reference_error_norm_upper * packet_gradient_norm_upper
    )


def receipt_smoothness_drift_upper(
    *,
    observable_alignment: float,
    reference_error_norm_upper: float,
    packet_gradient_norm_upper: float,
    learning_smoothness: float,
    receipt_motion_upper: float,
    packet_weight: float,
) -> float:
    """Upper-bound the objective change from a delayed owner packet.

    The owner update is ``theta_i <- theta_i - alpha g``.  Reference error,
    launch-to-receipt motion, and smoothness are charged explicitly.
    """

    nonnegative = (
        reference_error_norm_upper,
        packet_gradient_norm_upper,
        learning_smoothness,
        receipt_motion_upper,
        packet_weight,
    )
    if not all(np.isfinite(value) for value in (observable_alignment, *nonnegative)):
        raise ValueError("drift-bound inputs must be finite")
    if min(nonnegative) < 0.0:
        raise ValueError("drift bounds and packet weight must be nonnegative")
    corrected = reference_corrected_alignment_lower(
        observable_alignment=observable_alignment,
        reference_error_norm_upper=reference_error_norm_upper,
        packet_gradient_norm_upper=packet_gradient_norm_upper,
    )
    effective = corrected - (
        learning_smoothness
        * packet_gradient_norm_upper
        * receipt_motion_upper
    )
    return float(
        -packet_weight * effective
        + 0.5
        * learning_smoothness
        * packet_weight**2
        * packet_gradient_norm_upper**2
    )

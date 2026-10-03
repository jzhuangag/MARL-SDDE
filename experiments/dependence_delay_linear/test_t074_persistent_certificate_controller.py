import numpy as np

from experiments.dependence_delay_linear.t074_persistent_certificate_controller import (
    NoiseCertificate,
    simulate_persistent_qp_controller,
)


def test_certificate_uses_only_completed_blocks_plus_current_prefix() -> None:
    certificate = NoiseCertificate.empty(2)
    completed = np.asarray([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    certificate.update_completed_block(completed)
    before = certificate.scatter.copy()
    covariance, rho_upper, effective, pairs = certificate.estimate_for_current_prefix(
        np.asarray([[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5]]),
        delta=0.1, rho_cap=0.95,
    )
    assert np.array_equal(certificate.scatter, before)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-12
    assert 0.0 <= rho_upper <= 0.95
    assert effective >= 1.0
    assert pairs == 4


def test_persistent_evidence_tightens_the_rho_radius() -> None:
    rng = np.random.default_rng(7)
    certificate = NoiseCertificate.empty(2)
    prefix = rng.standard_normal((5, 2))
    _, first_upper, _, _ = certificate.estimate_for_current_prefix(
        prefix, delta=0.1, rho_cap=0.95
    )
    for _ in range(8):
        certificate.update_completed_block(rng.standard_normal((10, 2)))
    _, later_upper, _, _ = certificate.estimate_for_current_prefix(
        prefix, delta=0.1, rho_cap=0.95
    )
    assert later_upper < first_upper


def test_controller_has_zero_extra_probe_transitions() -> None:
    targets = np.repeat(np.asarray([[-1.0, -1.0, 1.0, 1.0]]), 3, axis=0)
    observations = np.repeat(targets[:, None, :], 10, axis=1)
    result = simulate_persistent_qp_controller(
        observations=observations, targets=targets, initial_parameter=0.0,
        gain=0.1, delay=0, decision_blocks=[0, 2], pre_steps=5,
        drift_weight=4.0, variance_weight=1.0, safety_slack=0.0,
        rollback_margin=-0.05, certificate_delta=0.1, rho_cap=0.95,
    )
    assert result.learning_transitions == 30
    assert result.fingerprint_transitions == 10
    assert result.message_units <= 4
    assert result.rho_upper_path.shape == (2,)
    assert np.all(result.effective_samples_path >= 1.0)

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_tracking_recurrence_is_bounded_by_closed_form() -> None:
    c = 0.72
    d = 0.014
    v = 0.035
    chi = 0.2
    c_bar = (1.0 + chi) * c
    assert c_bar < 1.0

    z = 1.3
    z0 = z
    forcing = (1.0 + chi) * d + (1.0 + 1.0 / chi) * v**2
    for n in range(1, 51):
        z = c_bar * z + forcing
        bound = c_bar**n * z0 + forcing / (1.0 - c_bar)
        assert z <= bound + 1e-14


def test_advantage_transfer_uses_bounded_features() -> None:
    rng = np.random.default_rng(20260909)
    gamma = 0.97
    phi_bound = 2.0
    for _ in range(100):
        error = rng.normal(size=7)
        phi = rng.normal(size=7)
        phi_next = rng.normal(size=7)
        phi *= min(1.0, phi_bound / np.linalg.norm(phi))
        phi_next *= min(1.0, phi_bound / np.linalg.norm(phi_next))
        advantage_error = float((gamma * phi_next - phi) @ error)
        rhs = (1.0 + gamma) ** 2 * phi_bound**2 * float(error @ error)
        assert advantage_error**2 <= rhs + 1e-12


def test_tracking_statement_and_proof_are_linked() -> None:
    main = (ROOT / "main.tex").read_text(encoding="utf-8")
    appendix = (ROOT / "appendices.tex").read_text(encoding="utf-8")
    assert "\\label{thm:tracking}" in main
    assert "\\label{eq:tracking-bound}" in main
    assert "Proof of Theorem~\\ref{thm:tracking}" in appendix
    assert "\\eqref{eq:tracking-bound}" in appendix

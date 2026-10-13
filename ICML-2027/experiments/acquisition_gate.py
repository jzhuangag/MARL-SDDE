"""Outcome-free acquisition gate against a public-budget-wise strong fallback."""

import numpy as np

from efficient_critic import InnovationAction, efficient_true_risk
from efficient_critic import efficient_select


def public_fallback(unpaid: list[InnovationAction], rho_prior: list[float]) -> InnovationAction:
    """Strong single q for each public cost/mixing configuration.

    The prior is declared before outcomes. For a fixed q, optimal spacing is
    independent of rho because its variance multiplies every spacing equally.
    """
    fixed = [max((a for a in unpaid if a.q==q),key=lambda a:(a.information_factor,-a.spacing)) for q in sorted({a.q for a in unpaid})]
    return min(fixed,key=lambda a:(np.mean(np.log([efficient_true_risk(a,rho) for rho in rho_prior])),a.q,a.spacing))


def perfect_information_is_futile(paid: list[InnovationAction], fallback: InnovationAction) -> bool:
    """Certify that even a paid clairvoyant cannot beat the free fallback.

    Each risk difference is affine in rho in [0,1], so endpoint checks prove
    the statement over the entire continuum, not just the evaluated rho grid.
    """
    return all(efficient_true_risk(a,rho) >= efficient_true_risk(fallback,rho)-1e-14
               for a in paid for rho in (0.,1.))


def gated_select(paid, unpaid, rho_prior, calibration_supplier):
    """No observation supplier is invoked on the provably futile branch."""
    fallback=public_fallback(unpaid,rho_prior)
    if perfect_information_is_futile(paid,fallback):
        return fallback, True
    return efficient_select(paid,calibration_supplier()), False

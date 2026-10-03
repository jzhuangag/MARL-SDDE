"""Independent deterministic-map covariance audit; not a frozen runner replacement.

x_t=A_t x_(t-1)+B_t xi_t, xi_t=lambda xi_(t-1)+u_t.
Initial x is deterministic and xi is stationary. Maps must not depend on samples.
"""
import numpy as np


def covariance_path(maps, spatial_covariance, temporal_correlation):
    """Return parameter-state covariances with all noise cross terms retained."""
    maps = list(maps)
    sigma = np.asarray(spatial_covariance, dtype=float)
    lam = float(temporal_correlation)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1] or not sigma.size:
        raise ValueError('spatial covariance must be square and nonempty')
    if not np.all(np.isfinite(sigma)) or not np.allclose(sigma, sigma.T):
        raise ValueError('spatial covariance must be finite and symmetric')
    if np.linalg.eigvalsh(sigma).min() < -1e-12:
        raise ValueError('spatial covariance must be positive semidefinite')
    if not 0 <= lam < 1 or not maps:
        raise ValueError('need maps and temporal correlation in [0,1)')
    dimension = np.asarray(maps[0][0]).shape[0]
    n = sigma.shape[0]
    cov = np.zeros((dimension+n, dimension+n))
    cov[dimension:, dimension:] = sigma
    outputs = []
    for matrix, noise_map in maps:
        a, b = np.asarray(matrix, dtype=float), np.asarray(noise_map, dtype=float)
        if a.shape != (dimension, dimension) or b.shape != (dimension, n):
            raise ValueError('incompatible deterministic map dimensions')
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError('maps must be finite')
        transition = np.block([[a, lam*b], [np.zeros((n,dimension)), lam*np.eye(n)]])
        injection = np.vstack([b, np.eye(n)])
        cov = transition @ cov @ transition.T + (1-lam*lam)*injection @ sigma @ injection.T
        cov = (cov+cov.T)/2
        outputs.append(cov[:dimension,:dimension].copy())
    return np.asarray(outputs)

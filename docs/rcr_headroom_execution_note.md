# RCR-H1 pre-outcome execution correction

The first post-preregistration `run` attempt terminated before creating its
output directory.  SciPy L-BFGS-B returned `ABNORMAL` while solving one strong
static-comparator box QP.  No RCR-H1 row, cell, aggregate, or gate outcome was
written or inspected.

The frozen configuration, SHA-256, seeds, scenario generator, metrics,
comparators, thresholds, and stopping rule remain unchanged.  The only code
change adds a deterministic SLSQP fallback for the identical convex objective
and verifies the returned point with box-constrained KKT residual at most
`1e-6`.  A 64-scenario solver robustness test was added.  Fifteen targeted
tests and 134 package tests pass; the intended result directory remains absent.

Corrected runner SHA-256:
`3E7E18849191C44D253F38CFFA7006E195F592A3BFBC4D72E99CF8E67C4C9CED`.

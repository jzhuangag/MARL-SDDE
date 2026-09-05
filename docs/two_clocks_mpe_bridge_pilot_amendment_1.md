# Two Clocks public-MPE CPU pilot Amendment 1

Date: 2026-09-03.

The first execution attempt from preregistration commit `9328056` was
interrupted after approximately 15 minutes because four workers had not
completed the first four runs.  The runner writes scientific files only after
all futures succeed; no output directory, endpoint, curve, summary, return, or
other scientific result was written or inspected.

Static host inspection shows eight logical processors and sufficient memory.
This amendment changes only `workers` from 4 to 8.  Process count is not a
learning parameter, the tasks are isolated, rows are sorted before writing,
and physical runtime is written to a separate non-scientific file.  Seeds,
tasks, horizons, methods, estimator, checkpoints, network, learning rate,
step cap, Lyapunov constants, accounting, expected rows, and P1--P12 are
unchanged.

The interrupted no-artifact attempt is retained in the CPU ledger.  The next
execution must use the amended configuration and a new empty output path.

Amended configuration SHA-256:
`a06f250c4edf07b0dc7db74d9df5c6edac662a2f77386c6857cd47d99c269cda`.

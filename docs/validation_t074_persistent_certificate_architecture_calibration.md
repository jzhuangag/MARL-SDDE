# T-074 persistent-certificate calibration execution report

## Decision

T-074 reached its declared 30-minute local-CPU hard timeout and was terminated.
It is a computational-feasibility failure, not a scientific pass or failure.
The process remained responsive and CPU-bound with approximately 79 MB RSS;
the runner had not created its output directory, so there is no partial grid,
summary, endpoint table, or result eligible for interpretation.

The frozen architecture commit is `f94991416bd5497f2d0cd378a60585673b92fb57`.
At the final read-only check, accumulated CPU time was 1,749.61 seconds. No
automatic retry, solver-tolerance change, GPU/HPC4 escalation, or partial-row
salvage occurred.

## Design-only subset

Before freezing the full calibration, the declared four-old-seed check took
53.15 seconds. It showed 8.61% nonstationary improvement, 58.33% strict cell
coverage, stationary ratio 1.0738, and essentially zero high-temporal harm.
Those numbers are outcome-informed design information only. They cannot be
promoted to a pilot or substituted for the timed-out full grid.

## Diagnosis and next admissible action

The persistent certificate changes the QP conditioning. The current solver
starts projected gradient from the local vertex for every recipient and every
decision, so many instances approach the 50-iteration cap. Certificate memory
and covariance updates remain (O(n^2)) and low-memory; repeated cold-start QP
optimization is the bottleneck.

A successor requires a new identifier and an outcome-free solver-equivalence
audit. The admissible direction is warm-started accelerated projected gradient
or a scalable primal-dual simplex solver. It must compare objective values,
KKT residuals, weights, action acceptance, and complete risk paths against the
frozen T-074 solver on deterministic fixtures. Runtime/overhead becomes a hard
gate. T-074 itself may not be rerun with a changed solver.

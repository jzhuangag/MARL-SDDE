# T-072 dual-use graph architecture calibration validation

## Decision

T-072 is a verified, outcome-informed design calibration and an honest 7/8
failure. It does not authorize a T-072A pilot. The frozen strict-coverage
criterion C3 required improvement in at least 55% of nonstationary cells; the
observed fraction is 53.125%. The threshold and controller constants remain
unchanged.

The result rejects the discrete seven-action implementation, not the broader
dynamic collaboration-graph mechanism. Removing independent probe transitions
changes the aggregate nonstationary comparison from T-071A's 14.03% harm to
7.67% improvement while preserving the full 240 learning transitions. This is
strong causal design evidence that T-071A's opportunity cost was material.

## Frozen results

| Quantity | Result |
|---|---:|
| Nonstationary geometric improvement vs strong static | 7.6746% |
| Strictly improved nonstationary cells | 53.125% |
| Single-switch improvement | 4.6480% |
| Alternating improvement | 10.6052% |
| Delay 0 / 1 / 3 improvement | 8.8982% / 8.1200% / 5.9810% |
| Stationary controller/static ratio | 1.08405 |
| Accepted nonlocal action rate | 20.4388% |
| Rollback rate | 77.8724% |
| Mean / maximum safety debt | 0.08923 / 3.92016 |

C1, C2, and C4--C8 pass. C3 fails. All 13,824 endpoints are finite; each uses
exactly 240 learning transitions, zero additional probe transitions, no more
than 18 message units, and 168 candidate scores.

## Phase diagnosis

The remaining coverage failure is structured rather than random. Descriptive
nonstationary improvement is -6.2564%, 3.1030%, and 23.5640% at target scales
0.1, 0.3, and 0.6. It is -0.1595% at noise scale 0.5 and -0.4684% at temporal
correlation 0.9. Thus the discrete action catalogue transfers too often near
the low-signal/high-correlation boundary, while delivering large value when
heterogeneity is identifiable.

The next admissible mechanism receives a new identifier. It should replace
recipient-wise finite action scanning by a continuous simplex collaboration
weight computed from a covariance-aware convex QP. Its regularization must be
an observable uncertainty term, and the same Lyapunov debt must multiply
mixing exposure. This directly targets low-SNR shrinkage and gives a more
principled, jointly optimized graph than seven heuristic actions.

## Reproduction and integrity

The frozen run completed on local CPU in 86.27 seconds. An unchanged clean run
completed in 85.24 seconds. `endpoints.csv`, `cells.csv`, and `summary.json`
are byte-identical, with SHA-256 hashes recorded in the machine-readable
validation file. No GPU, HPC4, nonlinear benchmark, new seed, significance
test, formal run, or `/project` write occurred.

## Statistical-fallacy scan

Coverage is 11/11. The full grid and every source seed are retained; no
outcome-conditioned cell filtering, attrition, collider adjustment, or
individual-level inference occurs. All eight criteria, including C3, are
reported. The four-seed architecture tuning is explicitly declared, so these
results are not reused as an independent pilot. Aggregate claims are broken
down by schedule, delay, scale, noise, and temporal correlation; causal claims
remain limited to the common-random-number simulator. Target schedules precede
actions, excluding reverse causality in the simulated intervention.

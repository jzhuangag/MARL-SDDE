# LCO-S0 causal dual-use sensor development validation

## Decision

The frozen fixed-period hidden-filter architecture **does not survive** its
development gates.  D2, D3, and D4 fail.  It must not be promoted to an
independent confirmation, formal evidence, or a standard MARL/GPU benchmark.
The probe period, thresholds, seeds, and target population were not changed
after reading the outcome.

This is an architecture failure rather than an execution failure.  All 5,760
paths and 5,898,240 coordinate events completed, every reported rate is
finite, both resource-accounting checks pass, and the selected sensor remains
contractive in every separated dynamic cell.

## Frozen result

- execution commit: `2d74f47539bbb13f7bf20b3f89981717204e7cc5`;
- configuration SHA-256:
  `43fac4f2ec52b478b729c61066a66afba287b8958415994a9cdd4b2bbf0f724d`;
- result SHA-256:
  `77a0e31f1fb176952feabec97b99604764b9abc88ba62faa83c427b3749f3ec7`;
- selected probe period: 16 events;
- rows/cells: 5,760/720;
- development seeds: 83,001--83,008, permanently excluded from confirmation.

| Frozen gate | Threshold | Noise 0 | Noise 0.05 | Result |
|---|---:|---:|---:|---|
| Mean dynamic log-rate gain | at least 0.02 each | 0.015853 | 0.015974 | fail |
| Median exact-controller gain capture | at least 0.40 each | 0.316579 | 0.339145 | fail |
| Improved dynamic-cell fraction | at least 0.75 each | 0.71875 | 0.71875 | fail |
| Dynamic contraction fraction | descriptive | 1.0 | 1.0 | pass descriptively |

The two arrival groups retain positive mean gains, 0.012816 at arrival 0.1 and
0.019012 at arrival 0.5.  Stationary potential loss and maximum budget
overshoot are both zero.

## Failure diagnosis

Fingerprint perturbation is not the limiting factor: noise 0 and 0.05 produce
nearly identical aggregate results.  The failure is concentrated where
information is perishable or optimism is scarce:

| Group | Sensor gain over strong fixed | Exact-phase headroom | Improved cells |
|---|---:|---:|---:|
| persistence 0.80 | 0.006195 | 0.064615 | 0.50 |
| persistence 0.95 | 0.025632 | 0.057267 | 0.9375 |
| rotation fraction 0.25 | 0.002608 | 0.046591 | 0.55 |
| budget 0.25 | -0.001044 | 0.033946 | 0.50 |
| normalized step 0.20 | -0.000396 | 0.002295 | 0.50 |

The exact-phase controller still has substantial headroom at persistence 0.8,
rotation fraction 0.25, and budget 0.25.  Therefore the negative result does
not falsify the phase-adaptive optimism problem.  It falsifies the proposed
solution: periodic forced probes spend scarce optimistic updates in potential
phases, while the one-step hidden filter reacts too slowly when phase evidence
is short-lived.  A longer fixed period reduces probe waste but cannot solve
this information-versus-control allocation problem.

## Integrity and reanalysis

The serialized rows contain 5,760 unique frozen specifications.  Recomputing
the summary from those rows gives a JSON-semantically identical object.  There
are no call-budget, probe-count, or informative-fingerprint count violations.
The result directory contains a single 6,215,206-byte JSON artifact.  The run
was local CPU only; no GPU, HPC4, or remote storage was used.

A clean scientific reproduction was not launched because the frozen rule only
authorized reproduction after all development survival gates passed.  The
pre-outcome serial/parallel equivalence test remains relevant to deterministic
serialization, but it does not convert this failed development result into
evidence.

## Consequence for the unified story

The project should retain the single research problem but replace the failed
fixed probing mechanism.  An optimism call is simultaneously a stabilizing
control and a geometry observation.  The next candidate must price its
*value of information* together with its immediate Lyapunov drift value and
resource debt; otherwise sparse probes lag in switching games and dense probes
waste the same budget they are meant to allocate.

Before another run, that decision must be derived as a causal constrained
one-step lookahead (or a proved approximation), tested for zero-information
and zero-headroom cases, and frozen in a new development-only commit.  The
existing eight seeds may remain development data but can never become
confirmatory evidence.

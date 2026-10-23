## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-06
- Verification Status: VERIFIED
- Version Label: pdsg_cone001_validation_v1

# PDSG-CONE-001 validation: shifted causal trajectory cone

## Decision

PDSG-CONE-001 passes all frozen primary gates C1--C11 and the clean
reproduction gate C12.  On standard 20-agent Pistonball, the launch-state
position--velocity tube is empirically calibrated, nonvacuous at short
horizons, state dependent, and systematically denser at longer horizons.

This is positive certificate and phase evidence.  It authorizes **design only**
of a separately preregistered CPU synthetic Lyapunov-controller audit.  It
does not establish higher return, authorize formal controller seeds, or
authorize GPU/HPC4 training.

Overall confidence is **SOLID** for the frozen certificate experiment and
**CAUTION** for the eventual learning-performance claim.

## Provenance

- causal-cone theorem commit: `420d113`;
- shifted conformal theorem commit: `a5a8132`;
- unified spatiotemporal mainline commit: `3447ace`;
- preregistration commit: `1cf52a89542ccddb7212db38575e1ba619652f0c`;
- frozen runner commit: `ddf6f9d0feaab0f49bacefd15fadd2036eb6f1a4`;
- manifest SHA-256:
  `B46BBCE028B7416DAF065F8F8362271CB7CD5EC8673E98E5EA998435736F54BA`;
- runner SHA-256:
  `9262972777E80D1D50FDC5B2357FB2A7193293B80169C510A08C51725C027922`;
- local environment: conda `ust2`, Python 3.11.13, PettingZoo 1.26.1,
  Gymnasium 1.0.0, Pygame-ce 2.5.8, Pymunk 7.3.0, NumPy 2.2.2;
- primary runtime: 1,195.18 seconds;
- independent clean-rerun runtime: 1,181.73 seconds;
- package regression: `90 passed in 34.82s`;
- `git diff --check`: pass.

Ignored raw outputs are under
`tmp/policy_dependency_sync/pdsg_cone001_primary` and
`tmp/policy_dependency_sync/pdsg_cone001_reproduction`.

## Frozen population and estimator

The experiment contains 27 fixed-policy launch strata: three heterogeneous
policy seeds, three launch burn-ins, and horizons 4, 6, and 8.  Every stratum
uses 128 calibration and 256 disjoint holdout trajectories, giving 10,368
trajectories in total.  Reused seed numbers across strata are common random
numbers and are not treated as independent cells.

The runner reads trajectory states but not rewards or learning outcomes.  It
calibrates the maximum deviation from a launch-time constant-velocity ball
path with split conformal rank 117.  The policy-shift correction is fixed from
the declared joint mean displacement of 0.01 and policy standard deviation
0.3; it is not fitted to holdout coverage.

## Results

| Metric | Observation | Frozen requirement | Result |
|---|---:|---:|---:|
| completed finite strata | 27/27 | 27/27 | pass |
| completed trajectories | 10,368/10,368 | all | pass |
| pooled holdout coverage | 0.912182 | at least 0.88 | pass |
| minimum cell coverage | 0.855469 | at least 0.80 | pass |
| horizon-4 median edge cost | 0.347368 | at most 0.45 | pass |
| horizon-4 p90 edge cost | 0.509474 | at most 0.65 | pass |
| median horizon-8 minus horizon-4 cost | 0.205263 | at least 0.10 | pass |
| nondecreasing horizon-cost paths | 9/9 | at least 7/9 | pass |
| horizon-4 burn-in-5/25 support Jaccard | 0.000000 median | at most 0.40 | pass |
| maximum shifted escape bound | 0.147140 | at most 0.15 | pass |
| maximum reported edge count | 380 | at most 380, exact accounting | pass |

Coverage remains above the frozen threshold at every horizon.  Pooled
coverage is 0.894531, 0.922743, and 0.919271 at horizons 4, 6, and 8,
respectively.  The corresponding median conformal radii are 2.500760,
4.234528, and 6.385996 piston widths.

The edge-cost distribution reveals the intended phase rather than only a
mean improvement.  Median edge cost rises from 0.347368 at horizon 4 to
0.478947 at horizon 6 and 0.552632 at horizon 8.  The p90 rises from 0.509474
to 0.920000 and then 1.000000.  Thus a short rollout often admits a sparse
policy-version graph, whereas a long rollout can require the complete graph.
The zero median support overlap between early and late launch states confirms
that one static sparse graph cannot represent the observed spatial footprint.

## Gate ledger

| Gate | Result |
|---|---:|
| C1 completeness, finiteness, disjoint calibration/holdout | pass |
| C2 outcome-free state-path analysis | pass |
| C3 finite conformal radii | pass |
| C4 pooled holdout coverage | pass |
| C5 per-stratum coverage | pass |
| C6 short-horizon communication sparsity | pass |
| C7 horizon densification | pass |
| C8 launch-state support movement | pass |
| C9 shifted-policy escape bound | pass |
| C10 exact graph accounting | pass |
| C11 development-data exclusion | pass |
| C12 byte-identical clean reproduction | pass |

No gate, seed, horizon, policy profile, or threshold was changed after output
inspection.

## Reproducibility

The experiment was rerun into a clean output directory with the frozen
manifest, runner, seeds, and environment.  All three scientific products are
byte-identical; runtime is deliberately stored outside the scientific files.

| Artifact | Primary and reproduction SHA-256 |
|---|---|
| `cells.json` | `3F082733143B3A029F4D2773A578FCA5C83174CEE27627C6CC48EC234F46A879` |
| `summary.json` | `196C787750421330F9531A1063CDAE5B2DBD97CC3C0B0453F4CE10629AAAD946` |
| `trajectories.csv` | `3329AFFFED2E59E4B2A1FFA415C12EE07695F462CA66AE3659B3ABC39FAE1224` |

Reproducibility verdict: **REPRODUCIBLE**.

## Claim boundaries

1. Split-conformal coverage is marginal within each fixed reference-policy
   launch stratum, not an anytime simultaneous guarantee during learning.
2. The policy-shift correction assumes equal-variance transformed Gaussian
   actors and a certified per-state joint mean displacement.  A vacuous bound
   must trigger full prospective support or recalibration.
3. Pistonball's geometric tube is used under the explicit causal-model
   condition that pistons outside the contact-expanded ball tube cannot alter
   the finite-horizon reward/owner update before a tube escape.  The generic
   coupling lemma is theorem-facing; the environment-specific condition must
   remain explicit in the paper.
4. Edge savings are communication-structure evidence, not evidence of higher
   policy return.  No reward, gradient update, or controller was evaluated.
5. The result supports joint horizon--graph control because horizon changes
   the required spatial footprint.  It does not show that the eventual
   Lyapunov controller beats a strong fixed horizon/graph envelope.

## Fallacy scan

Coverage: **11/11** statistical and methodological fallacy types checked.

| Fallacy | Severity | Finding |
|---|---:|---|
| Simpson's paradox | NOTE | Horizon trends are reported by all nine matched policy--launch paths; all nine are nondecreasing. |
| Ecological fallacy | NOTE | Coverage and edge-cost claims remain at the registered stratum/task level; no claim about arbitrary MARL tasks is made. |
| Berkson's paradox | CAUTION | Pistonball was selected after a development audit; it is a declared positive task, not a random benchmark sample. |
| Collider bias | NOTE | No outcome-conditioned filtering, covariate adjustment, or reward-based selection occurs. |
| Base-rate neglect | NOTE | Coverage is reported directly over every holdout path; no diagnostic-prevalence extrapolation is made. |
| Regression to the mean | NOTE | Calibration and holdout seeds are disjoint, and no noisy extreme cell was selected for confirmation. |
| Survivorship bias | NOTE | All 27 cells and all 10,368 planned trajectories are present and finite. |
| Look-elsewhere effect | NOTE | All C1--C11 gates are reported conjunctively; no hypothesis-test search is performed. |
| Garden of forking paths | NOTE | Manifest, gates, seeds, and runner were committed before confirmation outputs and were not amended afterward. |
| Correlation versus causation | CAUTION | The observed tube is predictive; causal omission safety additionally relies on the stated coupled-cone structural condition. |
| Reverse causality | NOTE | Launch state and calibration radius precede every holdout path; no future reward or controller outcome enters selection. |

## Next admissible work

The next step is not GPU training.  It is a separately identified CPU study
of the complete Lyapunov decision rule on a factored Markov game with exact
causal cones, random launch/receipt delays, fully charged communication, and a
strong fixed horizon/graph envelope.  Before freezing that runner, the
within-cone signed drift estimator, launch-to-receipt remainder, and comparator
activation condition must be stated at the same filtration as the executable
decisions.


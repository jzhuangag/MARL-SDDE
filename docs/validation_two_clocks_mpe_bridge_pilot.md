# Validation of the Two Clocks public-MPE CPU bridge pilot

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-03
- Verification Status: VERIFIED
- Version Label: two_clocks_mpe_bridge_pilot_validation_v1

## Decision

**The CPU bridge fails and stops.**  P1, P4, P5, P7--P9, P11, and P12 pass;
P2, P3, and P6 fail.  The separate upstream HAA2C quality anchor is not run,
because it was authorized only after every CPU survival gate passed.  Formal
seeds, GPU, and HPC4 remain unauthorized.

This is not a power-only failure.  The registered aggregate heterogeneous AUC
gain is `0.00492039`, below the required `0.05`.  The task directions differ:
`simple_reference_v2` is positive (`0.0114174`), whereas
`simple_spread_v2` is negative (`-0.00157662`).  Directionality passes at
`11/16 = 0.6875`, but the effect is too small and not taskwise consistent.

## Frozen results

| Gate | Result | Evidence |
|---|---:|---|
| P1 validity, pairing, accounting | pass | 128 endpoints, 640 curve rows; all finite; exact keys/checkpoints; paired initialization and control variate |
| P2 positive learning | **fail** | three `simple_spread_v2` references have negative mean change |
| P3 heterogeneous AUC gain | **fail** | aggregate `0.00492039`; spread negative; threshold `0.05` |
| P4 directionality | pass | `0.6875`, threshold `0.60` |
| P5 asynchronous-reference safety | pass | normalized gap `-0.00608788`, limit `-0.02`; both tasks pass |
| P6 rate--coupling ordering | **fail** | reference passes, spread has heterogeneous `-0.00157662` below balanced `0.0107281` |
| P7 nontrivial Lyapunov design | pass | multipliers `0.575728` and `0.692810`; motion differs from raw async |
| P8 motion/work disclosure | pass | complete/partial work, updates, step norm, policy KL and teammate KL finite |
| P9 frozen outcome-independent setting | pass | source/config unchanged after outcome access |
| P11 no formal/GPU escalation | pass | no formal, HAA2C anchor, GPU or HPC4 run |
| P12 reproducibility | pass | both scientific CSV files byte-identical |

Mean final-minus-initial return by task and method:

| Task | offdiag async | raw async | delay-scaled async | frozen barrier |
|---|---:|---:|---:|---:|
| `simple_reference_v2` | 5.7122 | 5.8237 | 4.3229 | 4.9338 |
| `simple_spread_v2` | 0.2383 | -1.4961 | -0.1587 | -0.6819 |

Mean cumulative policy KL is `7.0616` for off-diagonal async, `16.6746` for
raw async, `1.9332` for delay-scaled async, and `11.6157` for the barrier.
Thus the candidate genuinely changes policy motion and stays close to the
stronger asynchronous reference, but this does not yield the preregistered
broad barrier advantage.

## Interpretation

The experiment supports a narrow mechanism observation: on
`simple_reference_v2`, heterogeneous clocks make the off-diagonal method
better than the barrier in AUC, and the method is not merely a disguised raw
or delay-scaled update.  It does **not** support the intended cross-task
standard-neural rate--coupling claim.  The theorem-facing finite-game evidence
remains valid within its assumptions; the failed bridge shows that the frozen
Monte Carlo neural interface does not transfer the phase broadly enough.

No setting may be retuned and rerun under this experiment identifier.  The
result therefore closes the present public-MPE bridge and leaves the current
project short of an ICML-complete experimental package.

## Fallacy scan

- Coverage: 11/11 statistical fallacy types checked.

| Fallacy | Severity | Finding |
|---|---|---|
| Simpson's paradox | CAUTION | positive aggregate gain masks a negative `simple_spread_v2` task direction; taskwise gate correctly prevents aggregation from rescuing it |
| Ecological fallacy | NOTE | no individual-agent claim is inferred from task aggregates |
| Berkson's paradox | NOTE | all preregistered seeds/tasks/methods are retained; no success-only selection |
| Collider bias | NOTE | no post-outcome covariate is conditioned on |
| Base-rate neglect | NOTE | no diagnostic conditional probability is reported |
| Regression to the mean | NOTE | seeds were not selected for extreme initial performance |
| Survivorship bias | NOTE | all 128 runs and 640 checkpoints are finite and analyzed |
| Look-elsewhere effect | NOTE | frozen gates and all failed taskwise directions are reported |
| Garden of forking paths | NOTE | scientific settings were committed before outcomes; the sole amendment changed workers before any artifact existed |
| Correlation versus causation | CAUTION | paired simulation supports comparisons within these frozen tasks, not universal MARL claims |
| Reverse causality | NOTE | temporal/cross-sectional reverse-causality interpretation is not applicable |

## Reproducibility and provenance

- Method: unchanged deterministic rerun on the same frozen commit/config/seeds.
- Verdict: REPRODUCIBLE.
- Primary runtime: `1764.41 s`; reproduction runtime: `1958.74 s`.
- Endpoint SHA-256, both runs:
  `d4bb72e5a9a28539a0533507375513f001da2d6f8a9cc6ba266288ddf3763554`.
- Curve SHA-256, both runs:
  `7a5905c001e187b5bb3d31ac3d55ee848911dab57ed728dff73ff138240b812d`.
- Manifest SHA-256, both runs:
  `0c2d8456e51188f3d69863992b7b1f7b721c1bd7745445e26472e6a24e480c10`.
- Frozen source: preregistration `9328056`, worker-only Amendment `b6aa00d`.
- Configuration SHA-256:
  `a06f250c4edf07b0dc7db74d9df5c6edac662a2f77386c6857cd47d99c269cda`.
- Frozen analyzer output SHA-256:
  `ceaceebe5babd0a37bfb82d5d294d4c24a705eedd7216657e64159507d37ff3a`.
- Tests: 18 targeted passed; full `clocked_async_mpg` package 428 passed.

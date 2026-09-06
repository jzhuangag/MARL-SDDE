# Validation: policy-dependency synchronization oracle gate

Date: 2026-09-06

## Decision

The frozen PDSG-001 gate is an **algorithm failure with a positive problem
signal**.  Gates G2, G3, and G6 establish broad equal-message-budget headroom
for the registered one-step drift oracle.  Gates G4, G5, and G8 reject the
unsigned `weighted_mismatch` rule.  Under the frozen stopping rule there is no
reproduction, sampled pilot, formal seed set, standard MARL benchmark, GPU
job, or HPC4 operation.

This is materially different from the strategic-clock result.  There, the
strong fixed scaling removed the purported problem-level value.  Here, the
strong envelope includes all 81 fixed directed donor mappings, periodic full
refresh, all-donor round robin, oldest cache, largest mismatch, active oldest,
and no refresh; the registered drift oracle still improves every active cell.

## Frozen provenance

- design commit: `3cfbc8c20b1c7b7dfa840a81d0a924592923e413`;
- preregistration commit: `660320f`;
- implementation freeze commit: `7bd2060`;
- manifest SHA-256:
  `E3CCB4AF2AF8ACDD865D380862B60F158A4FD5C0A9CB9F28D89ACA3662FED95B`;
- runner SHA-256:
  `69642B9470F9E033F631F39DDB9B9DC1E937DFDFFEBD640E9EB738D6A944E5FB`;
- result SHA-256:
  `89BDFFC2F5042A143AB20F68E16746EB3985DFF2A0493EB386003FE4018C3118`;
- workload: 252 cells, 72 active cells, and 181,440 method-seed rows
  including the fixed-mapping search;
- seeds: 92001--92008;
- targeted tests before execution: `7 passed in 0.29s`;
- local CPU only.

The ignored raw result is
`tmp/policy_dependency_sync/primary/summary.json`.  It is design-stage
evidence and was not edited after execution.

## Frozen gate ledger

| Gate | Result | Observation |
|---|---:|---|
| G1 validity and pathwise message budget | pass | all finite; equal horizons; no overrun |
| G2 median active oracle improvement >= 10% | pass | `13.9878%` |
| G3 oracle strict active-cell fraction >= 60% | pass | `72/72 = 100%` |
| G4 unsigned rule retains >= 70% and improves >= 60% | **fail** | median retention `-14.3386%`; strict `15.2778%` |
| G5 maximum static/uncoupled control loss <= 1% | **fail** | `13.7133%` |
| G6 positive by nonzero Markov persistence and every delay | pass | all registered groups positive |
| G7 graph changes in >= 50% active method-seed rows | pass | `100%` |
| G8 unsigned rule beats mismatch and active-oldest | **fail** | aggregate risk `1.28101` vs. `1.28351` and `1.27623` |
| G9 cell-level, not seed-level, fixed-map selection | pass | enforced by runner |
| G10 clean reproduction | not run | blocked by G4/G5/G8 |

The active oracle gain remains positive by communication delay:

| Delay | Median oracle improvement |
|---:|---:|
| 0 | `15.1684%` |
| 2 | `13.5500%` |
| 5 | `11.6711%` |

It is also positive at Markov stay probabilities 0.8 and 0.95, with medians
`13.1362%` and `14.5151%`.

The strongest comparator varies across active cells: outcome-aware best fixed
mapping wins 34, all-donor round robin 22, periodic full refresh 11,
active-oldest 4, and largest mismatch 1.  The result is therefore not driven
by one deliberately weak reference.

## Diagnosis

For a quadratic expected potential

\[
F(\theta)=\tfrac12(\theta-\theta^*)^\top\bar H
(\theta-\theta^*),
\]

the signed progress from an owner update depends on both the true current
block gradient and the candidate cached-policy gradient.  The rejected rule
uses only

\[
|H_{s,ij}|\,|\theta_j^{k-d}-\chi_{j\to i}|.
\]

It discards whether refreshing an edge rotates the applied gradient toward or
away from the true descent direction.  It also discards the quadratic
curvature cost.  Consequently it refreshes high-mismatch edges even when doing
so worsens the next expected-potential step, explaining both the negative
median oracle retention and the static-control losses.

This diagnosis was implied by the frozen formulas; it does not alter or rescue
PDSG-001.  The next admissible work is a symbolic derivation of the signed
drift minimizer and deterministic equality tests.  A successor outcome gate
requires a new preregistration only after its observable estimator, error
bound, computation cost, and launch-time filtration are fixed.

## Current research status

The result supports the existence of a nontrivial problem: pairwise strategic
freshness has value under a communication budget even against strong static
and simple online schedules.  It does not establish that a model-free learner
can estimate the signed edge values, that the hybrid SDDE approximates deep
MARL, or that a standard benchmark contains the same headroom.

The high-level architecture remains a candidate; the frozen unsigned
controller is permanently rejected.


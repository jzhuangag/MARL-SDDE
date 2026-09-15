## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: development experiment validation
- Origin Date: 2026-09-07
- Verification Status: DEVELOPMENT TARGETS 8/8 PASS; NOT HELD-OUT FORMAL EVIDENCE
- Version Label: MCERT-DEV-001-result-v1

# MCERT-DEV-001 validation

## Decision

The state-conditional tabular Markov certificate is nonvacuous on the frozen
development instance.  All eight development targets pass.  This authorizes a
separately frozen confirmation with untouched seeds; it does not authorize a
Pursuit efficacy run or a neural-critic coverage claim.

The originally intended `93000,...,93063` seeds were invoked by the static
test suite before a preregistration commit.  They are therefore permanently
classified as development data.  No result below is labelled formal.

## Results

At 4,096 transitions for each of two positive actions, or 8,192 fully charged
transitions in total:

| Quantity | Result | Target |
|---|---:|---:|
| Correct state-compatible edge | 1.0000 | at least 0.99 |
| Median certified/exact value | 0.545105 | at least 0.50 |
| Fifth-percentile certified/exact value | 0.495633 | at least 0.40 |
| Incompatible edge certified positive | 0.0000 | exactly 0 |
| Maximum declared local robust-DP scalar work | 32 | at most 128 |

Median value recovery increased monotonically over the frozen sample grid:

| Charged transitions per positive action | Median recovery |
|---:|---:|
| 256 | -0.449806 |
| 512 | -0.113997 |
| 1,024 | 0.177863 |
| 2,048 | 0.372891 |
| 4,096 | 0.545105 |
| 8,192 | 0.665834 |

Negative recovery at the two smallest sizes means that the lower bound is
negative, not that the estimator reverses the edge ranking.  The Lyapunov
packet-weight projection assigns zero weight in that region.  The meaningful
result is the transition from conservative rejection to a positive,
state-correct certificate as charged evidence accumulates.

## Reproduction

- configuration SHA-256:
  `2e9f28a600f40b3fdc4b612f6d5217e8b6070cdf2c4452c9876d90c2e55c03fb`;
- primary JSON SHA-256:
  `048bb60e4f2aaf36b071b7b0555bef3f9d0ab4be6443c9d78105c390124c07eb`;
- clean reproduction JSON SHA-256: identical;
- rows: 768, all finite;
- targeted tests: 3/3 before execution;
- compute: local CPU only; no GPU, HPC4, or `/project` access.

Raw primary and reproduction JSON remain under
`experiments/policy_dependency_sync/results/mcert_dev001_20260907/` and are
excluded from Git as generated artifacts.  The machine-readable aggregate is
`mcert_dev001_summary_20260907.json`.

## Scope

This result closes an important feasibility question: a mathematically valid
state-conditioned Markov shield need not collapse to the null action, and its
local dynamic program is small.  It does not yet establish the quality of a
neural state abstraction, the critic approximation radius, or return gains in
a standard MARL benchmark.  Those remain separate gates.

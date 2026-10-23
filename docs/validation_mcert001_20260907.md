## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: independent experiment validation
- Origin Date: 2026-09-07
- Verification Status: 10/10 CONFIRMATION GATES PASS; STANDARD MARL EFFICACY NOT TESTED
- Version Label: MCERT-001-result-v1

# MCERT-001 independent confirmation

## Decision

The count-uniform state-conditional tabular Markov certificate passes its
independently frozen confirmation.  It may now be used as the theorem-facing
alignment interface in a CPU end-to-end factored asynchronous controller.
The result does not certify a neural critic or authorize a positive Pursuit
return claim.

Preregistration commit `d4874a7` fixed 128 untouched seeds, the scientific
model, all thresholds, exact charging, and the runner before execution.

## Results

All 1,536 rows are finite.  At the frozen primary point of 4,096 transitions
per positive action, or 8,192 charged transitions in total:

| Quantity | Confirmation | Frozen gate |
|---|---:|---:|
| Correct state-compatible edge | 1.0000 | at least 0.99 |
| Median certified/exact value | 0.539842 | at least 0.50 |
| Fifth-percentile certified/exact value | 0.496488 | at least 0.40 |
| Incompatible edge certified positive | 0.0000 | exactly 0 |
| Maximum local robust-DP scalar work | 32 | at most 128 |

Median value recovery is nondecreasing on the complete registered grid:

| Transitions per positive action | Median recovery |
|---:|---:|
| 256 | -0.454022 |
| 512 | -0.106084 |
| 1,024 | 0.169537 |
| 2,048 | 0.379477 |
| 4,096 | 0.539842 |
| 8,192 | 0.666351 |

This reproduces the development phase transition without reusing its seeds.
The negative small-sample lower bounds cause the packet-weight projection to
select zero, as required by the safety interface.

## Gate ledger

The eight numerical gates pass.  Exact transition charging passes.  Primary
and clean reproduction are byte-identical with SHA-256
`82f0b2f08747c116b4d8ece0f1cc3130bc34c88ca3de4eaeaa983861d8d5d49c`.
The static preregistration tests passed 6/6, and the maintained
`policy_dependency_sync` package passed 560/560 in 200.05 seconds.  Thus all
ten registered gates pass without threshold or seed changes.

## Scope and next gate

MCERT-001 proves that the strict Markov shield is not merely formal: after a
finite, fully charged local evidence budget, it selects the correct switching
edge with a nontrivial fraction of the exact value.  The next CPU experiment
must place this certificate inside the actual joint Lyapunov edge/packet-weight
controller and include the identification phase in the comparator's total
resource ledger.  That test must beat strong fixed, age, and round-robin
schedulers after amortizing identification; otherwise the tabular mechanism
does not justify a standard benchmark.

Pursuit still requires an outcome-free neural/local-state interface with an
explicit approximation treatment.  No GPU job is authorized by MCERT-001.

Raw primary and reproduction JSON remain in the ignored directory
`experiments/policy_dependency_sync/results/mcert001_20260907/`.

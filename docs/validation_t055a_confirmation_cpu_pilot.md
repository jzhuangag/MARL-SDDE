# T-055A independent CPU confirmation pilot validation

## Decision

T-055A passes all twelve frozen gates. It authorizes an independent formal
preregistration, but not formal execution, GPU work, HPC4 work, or a nonlinear
benchmark. T-053A remains an honest 11/12 failure and none of its eight seeds
was reused.

## Frozen scientific results

The controller's aggregate geometric risk ratio relative to the strong
overhead-specific fixed-q baseline is 0.851201, a 14.8799% reduction. The
corresponding ratio to the full-information true-rho fixed-q oracle is 1.01310.
The probe is fully charged and the strong fixed comparator receives the full
no-probe message budget.

| Slice | Controller / strong fixed-q | Risk reduction |
|---|---:|---:|
| Aggregate | 0.851201 | 14.8799% |
| CliffWalking | 0.847295 | 15.2705% |
| FrozenLake 8x8 | 0.839117 | 16.0883% |
| Taxi | 0.867439 | 13.2561% |
| Delay 0 | 0.853088 | 14.6912% |
| Delay 8 | 0.849318 | 15.0682% |

The controller improves 48 of 54 oracle-active cells (88.89%). Across the 30
inactive cells its geometric ratio is 1.01539, within the frozen 1.05 no-harm
gate. The standardized fingerprint residual RMSE is 0.92409, below 1.5, and
the selected participation remains nonincreasing over the registered
correlation grid.

## Seed-cluster uncertainty

The master seed is the independent unit. A deterministic 50,000-replicate
bootstrap resamples complete seed columns and preserves all within-seed
dependence across cells and comparators. These intervals are validation
diagnostics rather than additional preregistered gates.

| Slice | Ratio | Seed-cluster bootstrap 95% interval |
|---|---:|---:|
| Aggregate | 0.851201 | [0.830921, 0.872945] |
| CliffWalking | 0.847295 | [0.812908, 0.884733] |
| FrozenLake 8x8 | 0.839117 | [0.798990, 0.881801] |
| Taxi | 0.867439 | [0.838244, 0.897574] |

All 50,000 diagnostic replicates lie below the registered aggregate/task
gates. This does not turn the pilot into formal paper evidence; it supports
the prospective decision to register a new formal seed family.

## Gate ledger

P1--P11 pass in the frozen summary. P12 passes because `endpoints.csv`,
`cells.csv`, and `summary.json` from an independent clean rerun have exactly
the same SHA-256 hashes as the primary run. The clean rerun required 599.88
seconds on the local CPU. Its verified duplicate directory was removed after
comparison; the primary artifacts were preserved.

| Artifact | SHA-256 |
|---|---|
| `endpoints.csv` | `c424c2a51778405c6fa6928039172ee562f944ca3418c26eeca48312cffba717` |
| `cells.csv` | `284d503db46c5fa807f7e07d6b2881ddeb2e9f67149d49d7bffbe44f214b5f92` |
| `summary.json` | `2878deedc829ce4028344a7e8b3a47d907d03f9546d034022828e109ecdcebf8` |

The stored summary replays from the endpoint CSV with maximum absolute
floating-point difference `6.89e-15`, below the frozen validation tolerance
`1e-12`. The validation JSON itself reruns byte-identically with SHA-256
`e7e50a6f1575e69e7372476e5a03eb4c8840a6f064deeb26230c48bfaf5e5621`.
The complete `experiments/dependence_delay_linear` test suite reports 368
passed, zero failures, and zero errors in 662.57 seconds.

## Scope boundary

This is evidence for correlation-adaptive participation in delayed,
multi-agent fixed-policy Markov TD on exact public Gymnasium kernels. It is not
evidence for arbitrary actor--critic, control, nonlinear function
approximation, or general deep MARL. Those claims require separate theory and
benchmarks and cannot be inferred from T-055A.

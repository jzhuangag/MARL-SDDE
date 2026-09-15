# Multi-baseline q-grid infrastructure amendment 2

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run and validate
- Origin Date: 2026-09-15
- Verification Status: VERIFIED-INFRASTRUCTURE-AMENDMENT
- Version Label: qgrid_infrastructure_amendment_2

## Scope

The fail-closed continuation process for the frozen MPE q-grid stopped after
Slurm array `1861812` because its accounting parser requested `JobIDRaw` and
then expected values of the form `1861812_0` through `1861812_7`.
On HPC4, `JobIDRaw` exposed the underlying allocation identifiers for seven
array tasks, whereas the display field `JobID` retained the array identifiers.
Consequently, the monitor observed zero matching rows and correctly withheld
all later submissions.

This was a monitoring failure, not a training failure.
The eight frozen offset-zero cells all completed with exit code `0:0`.
They produced exactly eight unique bridge metadata files, eight progress
records, and eight checksum manifests.
Every checksum manifest passed, no fatal log signature was present, every
message and environment charge respected its frozen budget, and the cells
covered fixed `q=2`, independent coupling, and seeds `96001--96008` exactly.
No cell is retried.

## Amendment

Both continuation scripts now parse Slurm's `JobID` display field rather than
`JobIDRaw` while retaining the same exact eight-task, `COMPLETED`, and `0:0`
requirements.
This amendment changes no environment, method, hyperparameter, budget, seed,
gate, analyzer, or scientific result.
Execution may continue at the next untouched frozen batch, offset 8.

The detached login-node waiters are not treated as durable orchestration.
Subsequent batches are submitted only after an explicit read-only audit of the
preceding batch; failed scientific cells remain non-retriable without a new
user-authorized amendment.

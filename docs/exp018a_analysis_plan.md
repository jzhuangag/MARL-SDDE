# EXP-018A frozen analysis plan

## Population and unit of analysis

The inferential cluster is the registered pilot seed. A gradient projection is
a repeated measurement inside a seed and is never counted as an independent
replication. Cells are fixed by task, mixing profile, network initialization,
rho, and q.

## Primary estimand

For each projection and `(task,mixing,checkpoint,rho,q)` cell, estimate the
sample variance across the 64 seeds. For `q>1`, divide by the variance of the
same projection and cell at `q=1`. The theoretical target is
`rho+(1-rho)/q`. Relative calibration error is
`abs(empirical/theoretical-1)`.

The pilot reports the median, p90, and maximum error over the frozen cells and
projections. Only the median and p90 enter G4. The maximum is descriptive.

## Secondary diagnostics

- pairwise common-source fraction at q=32 for each rho;
- `q=1` variance spread across rho;
- fraction of projection paths with nonincreasing variance in q;
- finite values, unique keys, manifest hash, and parameter immutability.

No p-value, confidence interval, favorable-cell selection, environment
selection, or post-pilot threshold modification is permitted. A formal study,
if authorized, requires a new seed registry and a separate power analysis.

## Missingness and invalidity

Missing rows, duplicate seed-cell keys, nonfinite projections, mismatched
manifest hashes, or changed parameter hashes fail the corresponding mandatory
gate. Rows are never imputed. A failed gate stops formal preregistration.

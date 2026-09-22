# T-057A formal CPU preregistration

## Scope and evidence status

T-057A is the first experiment authorized to supply formal empirical evidence
for the fixed-policy delayed Markov-TD claim. T-053A and T-055A remain pilot
design evidence and none of their seeds is reused. The controller, public
Gymnasium kernels, coupling, horizons, probe, costs, comparator catalogue,
and 84 cells are unchanged from T-055A.

T-056 supplies the end-to-end finite-budget identity that conditions the
exact delayed PR risk on the independent Binomial fingerprint count. Formal
experiments estimate the corresponding expected risks using independent
master-seed clusters.

## Formal replication and inference

The seed registry is every integer in the closed interval 202608036001--
202608036256. This gives 256 master-seed clusters, 21,504 endpoints, and
365,568 sequentially generated long paths. Endpoint rows are not treated as
independent observations.

The primary statistic is the geometric mean, across cells, of the ratio of
controller and strong-fixed arithmetic seed means. A 50,000-replicate paired
cluster bootstrap resamples complete seed columns with RNG seed 57001.

- Aggregate: point estimate and one-sided 95% upper bound must be at most 0.95.
- Tasks: each point estimate and Bonferroni one-sided 98.333% upper bound must
  be at most 0.97.
- Delays: each point estimate and Bonferroni one-sided 97.5% upper bound must
  be at most 0.97.
- Active breadth: point estimate and one-sided 95% lower bound must be at
  least 0.60.
- Inactive no-harm: point estimate and one-sided 95% upper bound must be at
  most 1.05.
- True-rho oracle proximity: point estimate and one-sided 95% upper bound must
  be at most 1.20.

All budget, finite-value, direction, fingerprint, provenance, seed-isolation,
byte-reproduction, and full-test gates are also mandatory. Any failure is a
formal failure; no threshold, seed, task, cell, or analysis method may change.

## Authorization

The preregistration commit authorizes one primary local-CPU run and one clean
local-CPU reproducibility run. It does not authorize GPU, HPC4, `/project`, a
nonlinear benchmark, or a broader actor--critic claim. The expected primary
runtime is approximately 40 minutes based on the 599.88-second 64-seed clean
run; actual runtime must be reported.

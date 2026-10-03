# T-062 T-061A seed-cluster power audit

## Decision

T-062 is a read-only, post-pilot design audit.  It creates no new MinAtar
trajectory and does not reinterpret T-061A as formal evidence.  It recommends
512 entirely new master-seed clusters for an independently committed formal
CPU experiment.  It does not authorize that execution, GPU, HPC4, or
`/project` output.

The binding requirement is Breakout.  After shrinking its observed log-effect
by 50% toward the null and using a one-sided 97.5% upper confidence bound on
the complete-seed-cluster influence standard deviation, its simultaneous
taskwise gate requires 421 seeds.  The next registered power-of-two design is
512.  The nonsmooth strict-cell breadth projection independently requires at
least 256 seeds, so it does not bind the final design.

## Provenance and estimand

The audit reads only the exactly reproduced T-061A endpoint table, SHA-256
`28795ccad0bf09ffeed1606df2358bccb0eacdaf998bda266adf8cd1d3d2892e`.
It uses 32 pilot seed clusters and all 84 registered cells.  A master seed is
the resampling unit because controller and comparators share the same MinAtar
learning bank within every cell and the same seed appears across tasks,
correlations, overheads, and delays.  Endpoint rows are never treated as
independent observations.

For any registered cell set, the statistic is

\[
 R=\exp\left\{\frac1J\sum_{j=1}^J
 \log\frac{\bar C_j}{\bar F_j}\right\},
\]

where each bar is the arithmetic mean across complete master-seed clusters.
The delta-method influence retains all within-seed, across-cell dependence.

## Conservative ratio design

For each aggregate, task, and delay statistic, T-062 first replaces the pilot
ratio by `sqrt(R_pilot)`.  This halves the observed log-effect toward the null.
It then inflates the cluster influence standard deviation to its one-sided
97.5% chi-square upper bound and solves the one-sided normal design equation
at the exact formal multiplicity-adjusted quantile.

| Statistic | Pilot ratio | Planning ratio | Formal upper gate | Required seeds |
|---|---:|---:|---:|---:|
| Aggregate | 0.723812 | 0.850771 | 0.95 | 52 |
| Asterix | 0.670200 | 0.818658 | 0.98 | 92 |
| Breakout | 0.854287 | 0.924276 | 0.98 | **421** |
| Seaquest | 0.662321 | 0.813831 | 0.98 | 71 |
| Delay 0 | 0.724536 | 0.851197 | 0.97 | 53 |
| Delay 8 | 0.723089 | 0.850346 | 0.97 | 53 |

The calculation plans for simultaneous one-sided task bounds at
`1-0.05/3`, simultaneous delay bounds at `1-0.05/2`, and an aggregate 0.95
upper quantile.  It is deliberately more conservative than plugging the
pilot point estimate directly into the power equation.

## Strict-cell breadth projection

The fraction of cells with a smaller controller seed-mean risk is nonsmooth,
so T-062 separately performs 50,000 complete-cluster resamples for each
prospective sample size.  The 2.5% projected lower fractions are 0.5833,
0.5952, 0.6071, 0.6190, and 0.6429 for 64, 128, 256, 512, and 1,024 seeds.
Thus 256 is the first candidate whose projected 2.5% lower fraction is at
least 0.60.  At 512 seeds, the projected probability that the point breadth
is at least 0.60 is 0.99382.

This projection is a design diagnostic, not a pilot p-value or a future
formal confidence interval.  The formal analyzer must recompute its one-sided
cluster-bootstrap lower bound using only the new formal seeds.

## Reproducibility and permitted next step

The audit uses 50,000 resamples with RNG seed 62001.  Four targeted tests
cover seed-column preservation, the influence calculation, monotone sample
size response, and deterministic breadth projection.  A clean rerun must be
byte-identical before a formal preregistration is written.

The permitted next step is to implement and statically test a deterministic
parallel CPU runner and frozen seed-cluster analyzer, then commit a separate
T-063A preregistration with 512 new seeds.  T-060A and T-061A seeds must not be
reused.  No formal execution is authorized by T-062 itself.

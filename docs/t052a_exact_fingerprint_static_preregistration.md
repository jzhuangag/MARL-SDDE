# T-052A exact-binomial fingerprint static preregistration

T-052A is a new deterministic gate. It does not alter or reanalyse the failed
T-051A gate. The tasks, correlations, delays, participation catalogue,
contraction rule, 96 blocks, two probe actors, resource accounting, and all
thresholds are unchanged. The only scientific change is prospective use of
T-052's exact Binomial action distribution instead of T-051's generic
Hoeffding upper bound.

For every cell, the runner enumerates all 97 possible match counts, applies
the public plug-in action rule, and evaluates the exact expected leading
coefficient. It charges 96 packets of (h+2) message units and every
task-specific fingerprint transition. The no-probe strong fixed baseline
uses the complete total message budget.

B1--B12 are mandatory. Any failure stops a sampled pilot, without changing a
task, gate, or block count. Passing would authorize only a separately frozen
local CPU sampled pilot. It would not authorize formal seeds, nonlinear
experiments, GPU, or HPC4.

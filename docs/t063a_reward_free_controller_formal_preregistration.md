# T-063A formal reward-free correlation-adaptive MinAtar TD preregistration

## Scope and evidence status

T-063A is authorized to provide the first formal standard-benchmark evidence
for the reward-free correlation-adaptive participation claim.  T-060A remains
a failed selector pilot and T-061A remains positive design evidence; neither
experiment contributes a row or seed to formal inference.

The claim is deliberately scoped to cooperative fixed-policy Markov learning
with a fixed nonlinear state representation and a regularized affine TD head.
It is not an actor--critic, end-to-end representation-learning, or general
multi-agent control claim.  The official MinAtar 1.0.15 Asterix, Breakout, and
Seaquest dynamics, uniform full-six-action policy, sticky-action probability
0.1, and disabled difficulty ramping are unchanged from T-061A.

## Frozen mechanism and comparators

The controller charges 96 independent two-agent, length-four trajectory
fingerprint probes.  Fingerprints include state, previous executed action,
requested action, reward, and termination, but the participation decision
uses only exact-match counts, never task performance.  It estimates
`rho_hat=matches/96` and chooses q in {1,4,16} minimizing

`(overhead+q) * (rho_hat + (1-rho_hat)/q)`.

Every probe message and actor transition is deducted before learning.  The
strong comparator uses the task-overhead fixed q frozen before T-061A and
receives no probe cost and the full message/environment budget.  The
descriptive true-rho comparator also receives no probe cost and the full
budget, making the oracle-proximity gate conservative.

The online decision requires one match counter and a three-action scan.  TD
aggregation costs O(qd) arithmetic and O((D+1)d) delayed-head memory.  The
method has no Hessian, preconditioner, covariance estimate, or matrix inverse.

## Formal seeds and workload

The formal registry is every integer in the closed interval
202608056201--202608056712: 512 complete master-seed clusters.  It is disjoint
from T-060A 202608056001--202608056032 and T-061A
202608056101--202608056132.  The frozen grid has 84 cells and 43,008
endpoints.  Endpoint rows are not independent observations.

The expected workload is 305,986,560 generated transitions, no stored full
trajectories, approximately 6 GiB peak memory with four workers, and 1.47
ideal-scaling hours per run.  One primary and one clean reproduction run are
authorized on the local CPU.  GPU, HPC4, and `/project` use are not authorized.

Parallel execution is deterministic: jobs are ordered by game and seed,
`executor.map` preserves that order, and each job emits rho, overhead, and
delay rows in frozen order.  Each worker uses deterministic Torch operations
and one Torch thread.  Parallel scheduling cannot enter any random seed or
artifact field.

## Frozen inference

The primary cell endpoint is the arithmetic risk mean over 512 formal seed
clusters.  For each registered cell set, the ratio statistic is the geometric
mean of cell controller/strong ratios.  A 50,000-replicate paired bootstrap
with RNG seed 63001 resamples complete master-seed columns across all cells
and comparators.

- Aggregate: point and one-sided 95% upper ratio must be at most 0.95.
- Tasks: every point and Bonferroni one-sided 98.333% upper ratio must be at
  most 0.98.
- Delays: both points and Bonferroni one-sided 97.5% upper ratios must be at
  most 0.97.
- Breadth: point and one-sided 95% lower strict-cell fraction must be at least
  0.60.
- True-rho proximity: point and one-sided 95% upper ratio must be at most 1.15.

The thresholds were frozen from the T-061A scientific gates before formal
seeds were executed.  T-062 selected 512 seeds by halving every pilot
log-effect toward zero and inflating the complete-cluster variance; Breakout's
421-seed taskwise requirement was binding.

## Mandatory gates and stopping rule

All F1--F13 gates in the machine-readable preregistration are mandatory:
coverage and seed isolation; finite full dual-cost accounting; aggregate,
taskwise, delay, breadth, and oracle confidence gates; all 12 participation
directions; fingerprint calibration; independent-path collision; raw-summary
replay; byte-exact reproduction; and the complete nonlinear test suite.

Any failure is a formal failure.  No seed, task, cell, threshold, bootstrap
quantile, comparator, probe, encoder, or analysis rule may change.  The formal
result will be reported even if negative, and no failed gate may be rounded or
replaced.  A later learned-representation or actor--critic benchmark, if any,
requires a new identifier and a separate theorem and preregistration.

## Frozen provenance

- formal configuration SHA-256:
  `8e90b08f18a14b777356ab3c575c738d9c8b62c5a9a0d7b2ff06ae78605d457d`;
- T-061A source configuration SHA-256:
  `479f29cce9d66f846a7dc60c26a4dae6de67f9ae14623e85e6bc1afb8b9c03f4`;
- T-062 power audit SHA-256:
  `d48189f06e7c4c8bf4e2ea404c8ae13613d3002744ceb10cba7660849cc149dd`;
- formal runner SHA-256:
  `bdc1eb4f37a11fc22a383110609cf684d5351845fcddb4c34e16e7c1f910428b`;
- formal analyzer SHA-256:
  `82f975680d2b9b1e589a75ff2a0c9853eaeb6d1467e515927f116cabbc810ab6`;
- formal tests SHA-256:
  `bd3e4343a51023ccf38c3119c4613215dc06c630a13d1816e1096f2d80677a58`.

Execution is authorized only after these files and hashes exist in an
independent Git commit.

# T-061A reward-free participation-controller CPU pilot validation

## Decision

T-061A is a positive, exactly reproducible prospective pilot.  All eleven
frozen validity, value, direction, calibration, and provenance gates pass.
The primary and clean reproduction artifacts are byte-identical, and a
separate validator replays the complete summary from raw endpoints with zero
numeric difference.  The result authorizes a separate formal preregistration
with entirely new seeds.  It is not itself formal paper evidence and does not
authorize GPU, HPC4, or `/project` use.

The controller improves aggregate geometric terminal prediction error by
27.62% relative to the frozen no-probe strong fixed-q comparator.  It improves
all three tasks and both delay layers, strictly improves 69.05% of registered
cells, and chooses participation in the preregistered theoretical direction
on all 12 task-overhead-delay paths.  Its aggregate error is within 2.29% of a
no-probe controller given the true correlation and full learning budget,
despite paying the complete fingerprint-probe message and environment cost.

## Provenance and scope

- preregistration commit: `99eab11`;
- pre-outcome comparator-adapter amendment: `b451060`;
- configuration SHA-256:
  `5ac55dc247b366fb58434bd9c4fb8d68ea0baec7d9e7ceec429a3f5f48a61afb`;
- 32 new seeds with no overlap with T-060A;
- 84 cells and 2,688 endpoints;
- official `MinAtar==1.0.15` Asterix, Breakout, and Seaquest;
- fixed nonlinear random encoder and regularized affine TD head;
- local CPU only; no GPU, HPC4, `/project`, or external artifact write.

The first attempted process produced no result directory or accessible
scientific outcome.  It failed only after computing an in-memory controller
risk because a T-060 comparator helper expected a selection-seed registry.
Amendment 1 froze an empty split registry for the already-frozen comparator
and added a true mini-endpoint regression test.  It did not change a seed,
task, action, threshold, budget, probe, or scientific statistic.

## Mechanism

The controller never uses reward performance to select participation.  It
charges 96 independent two-agent, length-four trajectory-fingerprint probes,
estimates the common-path probability from exact fingerprint matches, and
then minimizes the preregistered phase coefficient over q in {1,4,16}.  The
probe and learning budgets are disjointly charged under both message and
environment constraints.  At deployment the controller requires a constant
number of match counters plus a three-action scan; it uses no Hessian,
preconditioner, covariance matrix, or matrix inverse.

## Frozen gate ledger

| Gate | Result | Value |
|---|---:|---:|
| P1 complete and unique | pass | 2,688/2,688 endpoints; 84/84 cells |
| P2 finite and full dual-cost accounting | pass | zero violations |
| P3 aggregate gain | pass | 0.723812 <= 0.95 |
| P4 taskwise gain | pass | all three ratios <= 0.98 |
| P5 delay robustness | pass | both delay ratios <= 0.97 |
| P6 directional breadth | pass | 0.690476 >= 0.60 |
| P7 true-rho proximity | pass | 1.022948 <= 1.15 |
| P8 participation direction | pass | 12/12 paths |
| P9 fingerprint calibration | pass | standardized RMSE 1.044854 <= 1.5 |
| P10 independent-path collision | pass | max 0.010417 <= 0.02 |
| P11 new-seed coverage | pass | all seeds new and complete |

Taskwise controller/strong ratios are 0.670200 for Asterix, 0.854287 for
Breakout, and 0.662321 for Seaquest.  Delay-0 and delay-8 ratios are 0.724536
and 0.723089.  These metrics aggregate all preregistered correlations,
communication overheads, and seeds; no task or unfavorable cell was removed.

## Exact reproduction and replay

The clean rerun used the same frozen configuration in a separate output
directory.  A standalone validator then compared all artifacts and recomputed
the full summary from `endpoints.csv`.  Every file is byte-identical and the
maximum replay difference is exactly zero.

| Artifact | SHA-256 |
|---|---|
| `endpoints.csv` | `28795ccad0bf09ffeed1606df2358bccb0eacdaf998bda266adf8cd1d3d2892e` |
| `cells.csv` | `9aabf164eca215cbfb8452a34476c262b1cec3cdf684f9baf1241f4dd3442727` |
| `summary.json` | `cc40420896350728726f2572073e70494380a22c400b828df1192fa6752baa8d` |

The primary scientific computation took approximately 1,324 seconds of wall
time, and the clean reproduction took approximately 1,309 seconds.  Runtime
is excluded from scientific artifacts and gates.

## Interpretation and permitted next step

T-061A resolves the two diagnosed failures of earlier nonlinear pilots.  The
probe remains informative when learning participation is q=1, so there is no
q=1 absorbing state.  Participation is selected by an outcome-independent
correlation phase rule rather than a high-variance empirical reward selector,
so it does not overfit sparse task outcomes.  The result supports the scoped
claim that reward-free correlation adaptation can improve a strong fixed-q
baseline in delayed multi-agent Markov TD with a fixed nonlinear
representation.

The only authorized next experiment is an independently preregistered formal
CPU confirmation.  Before execution, a seed-cluster power audit must freeze
the formal sample size, all-new seeds, runner/analyzer hashes, multiplicity
handling, taskwise and aggregate confidence gates, and an unconditional stop
rule.  No pilot seed may enter the formal analysis.

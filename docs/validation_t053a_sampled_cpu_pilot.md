# T-053A sampled standard-task CPU pilot: validation

## Decision

T-053A is an honest 11/12 pilot failure and does not authorize formal seeds,
GPU, or HPC4. The aggregate mechanism is positive, but the frozen per-task
gate cannot be replaced by the aggregate result.

| Metric | Result | Gate |
|---|---:|---:|
| Controller / strong fixed, aggregate | 0.890429 | P3 pass |
| Aggregate improvement | 10.9571% | threshold 5% |
| Controller / true-rho oracle | 0.997116 | P8 pass |
| Active cells improved | 37/54 = 68.5185% | P6 pass |
| Inactive-cell ratio | 1.001144 | P7 pass |
| Delay 0 ratio | 0.888945 | P5 pass |
| Delay 8 ratio | 0.891916 | P5 pass |
| Fingerprint standardized RMSE | 0.933958 | P10 pass |

The task ratios are 0.750072 on FrozenLake, 0.994533 on CliffWalking, and
0.946403 on Taxi. P4 required every task ratio to be at most 0.97, so
CliffWalking fails. Its value is not rounded or pooled away.

## Implementation integrity

Before sampling, the runner reconstructs every policy-weighted Gymnasium
action/transition/reward outcome. Across the three tasks, the regenerative
transition residual is at most (2.81\times10^{-16}), the centered
innovation norm is at most (3.09\times10^{-15}), and the full innovation
second-moment residual is at most (2.28\times10^{-13}). The controller and
all fixed-(q) comparators share one common plus 16 private trajectory banks;
probe paths use an independent stream. All probe and learning resources are
charged.

This evidence rules out the earlier absorbing-state bug and a
conditional-mean-only sampler. It does not by itself distinguish a small
CliffWalking effect from eight-seed sampling variance or a finite-horizon
remainder.

## Reproduction

The first process completed locally after the launch wrapper's short timeout;
the runner created artifacts only at completion. A clean independent rerun
took 47.2 seconds. `endpoints.csv`, `cells.csv`, and `summary.json` were
byte-identical across runs, with SHA-256 values recorded in the machine-readable
validation file. The duplicate rerun directory was removed.

## Next admissible step

Only T-054 is authorized: a read-only paired-variance and mechanism audit of
the frozen endpoints, plus a theorem-facing finite-horizon diagnosis for
CliffWalking. It may estimate how many new pilot seeds would be required to
resolve a 3% task effect, but it may not reuse the eight pilot seeds as formal
evidence or silently launch a larger run. Any later sampled run requires a
new identifier, new seeds, and an outcome-free commit.

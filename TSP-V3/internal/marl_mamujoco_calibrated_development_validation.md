# TSP-V3 calibrated-catalogue MaMuJoCo development validation

## Decision

`TSP-V3-MARL-MAMUJOCO-CAL-DEV-001` passed every frozen development gate.
The result authorizes a separate fresh-seed confirmation preregistration.
It is development evidence only and is not eligible for the manuscript.

The scientific implementation was fixed at commit
`16abe256a82a0d5f46bb2a9fad6cb03c1ba962f0`, and the preregistration was fixed
at commit `7c3894998c45cb8f5b6f64e0b6c8c816984445fc` before execution.

## Execution and payload audit

- Slurm array: `1921830`.
- Scratch root:
  `/scratch/jzhuangag/MARL-SDDE-TSP-V3-MAMUJOCO-CAL-DEV-001`.
- Eight of eight tasks completed with exit code `0:0`.
- Task elapsed times ranged from 22:49 to 38:41.
- All eight stderr files were empty, and the bounded fatal-pattern scan was
  clean.
- Exactly eight controller artifact directories were present: both coupling
  regimes crossed with training seeds `125001`--`125004`.
- Probe seeds were the disjoint registered seeds `136001`--`136004`.
- Every per-artifact `SHA256SUMS` manifest verified.
- The remote source tree was clean at the preregistration commit.
- The pinned HARL commit was
  `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2` for every run.
- Every charged progress record was finite.
- The probe did not update the model, and the controller did not use benchmark
  return when selecting participation.

The selected participation was q=8 in all four independent runs and q=2 in
all four shared runs.
The observed choices were restricted to the frozen catalogue `{2,8}`.

Exact accounting was reconstructed from every metadata file.
The probe used 16 non-learning blocks, 38,400 messages, and 3,200 environment
ticks in every run.
For q=8, training used 2,067 updates, 4,960,800 messages, and 413,400
environment ticks.
For q=2, training used 4,134 updates, 4,960,800 messages, and 826,800
environment ticks.
Thus every run used 4,999,200 messages, while total environment use was
416,600 ticks for q=8 and 830,000 ticks for q=2, both within the frozen
budgets.

## Frozen gate results

| Gate | Result |
|---|---:|
| Complete, finite, exact accounting, clean upstream | pass |
| Independent q=8 fraction at least 0.75 | 1.00, pass |
| Shared q=2 fraction at least 0.75 | 1.00, pass |
| Within 2% of the fixed-q envelope in each regime | pass |
| Equal-mixture AUC gain over strongest single fixed q at least 1% | 12.4303%, pass |
| Probe message fraction at most 1% | 0.768%, pass |
| Selection overhead at most 2% | 0.0000304% maximum, pass |
| Byte-identical analysis replay | pass |

The controller was 0.4658% below the fixed-q envelope in the independent
regime and 0.7283% below it in the shared regime.
The strongest single fixed comparator was q=8.
Against that comparator under the registered equal-regime mixture, the
controller improved normalized return AUC by 12.4303%.

## Replay and provenance

The analyzer was run twice in disjoint directories without the replay flag.
The two CSV files and two JSON files were byte-identical.
A third run with the replay flag produced the final pass decision.

- Replay/final summary SHA-256:
  `3b8d2939118080c0d8cd273af30d831dda224e2053961085998e5b3dcaaedc45`.
- Final gate JSON SHA-256:
  `5393705740cbe82fff3eafb725bc23d65bd81be7d8095fe047231f2f83e6da19`.

The next permitted stage is a separately committed confirmation
preregistration with fresh training and probe seeds, the controller, and all
fixed q in `{1,2,4,8}`.
Only a passing confirmation may be migrated into the TSP manuscript.

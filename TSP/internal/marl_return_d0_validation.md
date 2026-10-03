# TSP-MARL-DEV-001 validation report

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-10
- Verification Status: ANALYZED
- Version Label: validation_v1

## Scope and decision

TSP-MARL-DEV-001 is a development-only fixed-action headroom scan on PettingZoo MPE `simple_spread_v2` with MAPPO.
It tests whether an observable controller could have enough equal-resource return value to justify a subsequent controller-design stage.
It is not confirmatory evidence and uses one registered development seed per action and coupling regime.

The frozen decision is **STOP**.
Five of six mandatory gates passed.
The aggregate oracle return-AUC headroom was `0.0503254649`, just above the frozen `0.05` threshold, but the oracle did not strictly improve upon the global strong fixed action in both coupling regimes.
No controller-design, confirmation, additional task, or manuscript-return figure is authorized by this experiment.

## Registered design

- Repository commit: `106e9b187e954b229d1cf1298f2c379b154fce46`.
- Pinned HARL commit: `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`.
- Task: discrete PettingZoo MPE `simple_spread_v2` under centralized training and decentralized evaluation.
- Coupling regimes: marginal-preserving independent and shared rollout streams.
- Fixed actions: `q in {1,4,8,16}` and critic step size `eta in {0.00025,0.0005}`.
- Seed: `93001` for each of the 16 registered cells.
- Budgets: `B_msg=5,000,000` and `B_env=1,000,000`, with update cost `100+25q` messages and `25` environment ticks.
- Primary outcome: trapezoidal deterministic team-return AUC over 21 equally spaced fractions of each action's exactly charged feasible horizon.
- Strong baseline: the single fixed action maximizing mean AUC across both regimes.
- Oracle: the best fixed action selected separately within each coupling regime.

## Execution and integrity

- HPC4 work root: `/scratch/jzhuangag/MARL-SDDE-TSP-MARL-DEV-001`.
- Slurm submission identifiers: `1843182_[0]`, `1843190_[1-7]`, `1843198_[8-9]`, `1843327_[10-11]`, `1843353_[12-13]`, `1843408_[14]`, and `1843411_[15]`.
- All 16 allocations completed on NVIDIA A30 nodes with `ExitCode=0:0`.
- Elapsed time per cell ranged from `00:19:06` to `00:56:53`.
- Sixteen metadata files, sixteen progress files, and sixteen per-run `SHA256SUMS` files were present.
- Every per-run checksum passed; the fatal log scan for traceback, out-of-memory, missing-file, refusal, and dirty-worktree errors was empty.
- All 16 records were finite, completed, exactly within both budgets, based on the pinned HARL commit, and produced from clean source worktrees.
- The coupling audit passed: worker marginal reset laws matched across regimes while their joint coupling differed as registered.
- No scientific output was written to `/project`; run artifacts occupy approximately 67 MB under `/scratch`.

## Statistical findings

This is a deterministic analysis of an exploratory one-seed development grid; no p-value, confidence interval, or confirmatory effect-size claim is valid.

| Metric | Registered criterion | Observed | Status |
|---|---:|---:|---|
| Complete finite cells | 16/16 | 16/16 | pass |
| Exact dual-budget accounting | all cells | 16/16 | pass |
| Clean pinned upstream | all cells | 16/16 | pass |
| Oracle relative AUC headroom | at least 5% | 5.032546% | pass |
| Strict oracle improvement | both regimes | 1/2 regimes | **fail** |
| Marginal-preserving coupling audit | pass | pass | pass |

The global strong fixed action was `(q,eta)=(8,0.0005)`, with mean AUC `-88.98358284`.
The per-regime oracle mean was `-84.50544267`.
For independent streams, `(8,0.0005)` was both the global strong action and the regime oracle, with AUC `-74.23774720`, so strict improvement was false.
For shared streams, the oracle switched to `(1,0.0005)`, improving AUC from `-103.72941848` to `-94.77313814`, a relative gain of approximately 8.634% on an absolute-baseline scale.
Thus the aggregate headroom comes from adapting to the shared regime, while the independent regime calls for retaining the strong fixed action.

## Fixed-action AUC table

| q | eta | Independent | Shared |
|---:|---:|---:|---:|
| 1 | 0.00025 | -99.129413 | -98.827594 |
| 1 | 0.00050 | -97.633981 | **-94.773138** |
| 4 | 0.00025 | -80.222729 | -99.740804 |
| 4 | 0.00050 | -80.242983 | -99.755559 |
| 8 | 0.00025 | -75.225644 | -103.016107 |
| 8 | 0.00050 | **-74.237747** | -103.729418 |
| 16 | 0.00025 | -78.021212 | -109.686975 |
| 16 | 0.00050 | -74.737869 | -107.898043 |

## Reproducibility

- Method: deterministic replay of the frozen analyzer against the immutable 16-run artifact set.
- Verdict: PARTIALLY_REPRODUCIBLE.
- The replayed gate JSON, 336-row curve summary CSV, and plotted PDF were byte-identical to the first analysis.
- Full MAPPO training was not repeated; training is GPU- and environment-sensitive, and this development scan does not warrant a second 16-run allocation after its mandatory stop.
- Gate JSON SHA-256: `c3a8dc81a34c712cc26dc3a92ac4f94d10960d141f7d6ecb09dab6c411f8d209`.
- Summary CSV SHA-256: `b29c90fb2b5bac46488b1d3a9d356e5988f2d2853d0ca5c0fb984adf98d78002`.
- Curve PDF SHA-256: `f16ba3d339decbb157dd5fb3993a5867a426d19510d83ec0d3cf26850e4a8117`.
- Environment freeze SHA-256: `bd5cf97a2ffd77bca0674e5508293a35ef66ec724c90eb430b1da2c869a092c7`.

## Fallacy scan

Coverage: 11/11 types checked.

| Fallacy | Severity | Finding |
|---|---|---|
| Simpson's paradox | NOTE | Aggregate and per-regime results were both inspected; aggregate headroom is localized to shared streams and is not presented as uniform. |
| Ecological fallacy | NOTE | No inference from regime averages to individual strategic agents is made. |
| Berkson's paradox | NOTE | The complete frozen 16-cell grid was analyzed; there is no outcome-based cell inclusion. |
| Collider bias | NOTE | No post-outcome covariate conditioning is used. |
| Base-rate neglect | NOTE | No diagnostic classification probabilities are reported. |
| Regression to the mean | NOTE | No extreme-outcome cell was selected for a pre/post claim. |
| Survivorship bias | NOTE | All 16 registered cells completed and were retained. |
| Look-elsewhere effect | CAUTION | Eight actions were compared, but this is explicitly a development selection scan and not inferential evidence. |
| Garden of forking paths | NOTE | Runner, analyzer, grid, seed, gates, and hashes were frozen in commit `106e9b1` before outcomes. |
| Correlation implies causation | CAUTION | The coupling intervention is controlled, but a one-seed design is insufficient for a population-level causal performance claim. |
| Reverse causality | NOTE | The registered coupling is imposed before training; reverse temporal direction is not the relevant threat. |

## Artifact locations

- Active immutable run artifacts: `/scratch/jzhuangag/MARL-SDDE-TSP-MARL-DEV-001/artifacts`.
- Original analysis: `/scratch/jzhuangag/MARL-SDDE-TSP-MARL-DEV-001/analysis`.
- Byte-identical replay: `/scratch/jzhuangag/MARL-SDDE-TSP-MARL-DEV-001/analysis_replay`.
- Repository copies: `TSP/internal/marl_return_d0_gate.json` and `TSP/internal/marl_return_d0_summary.csv`.

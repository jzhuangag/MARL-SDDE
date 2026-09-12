# TSP-MARL-CONF-001 validation report

## Scope and decision

TSP-MARL-CONF-001 is the preregistered confirmation of the frozen observable Lyapunov probe-then-commit controller on discrete PettingZoo MPE `simple_spread_v2` with MAPPO and centralized training with decentralized execution.
All 12 mandatory gates passed, so the frozen decision is **ADMIT CONFIRMED RETURN EVIDENCE**.

The study used eight new training seeds (`96001--96008`) and eight disjoint probe seeds (`97001--97008`).
No confirmation outcome changed the controller, candidate actions, budgets, inference rule, seeds, or gate thresholds frozen at commit `c8ac978987ee65960a3b8dc6e37a8c61a898405d`.

## Main result

The return-free probe selected the dependence-matched action in every controller run:

| Rollout regime | Selected action | Selection frequency |
|---|---:|---:|
| Independent innovations | `q=8` | 8/8 |
| Shared innovations | `q=1` | 8/8 |

The preregistered team-return AUC comparisons were:

| Paired estimand | Mean relative change | One-sided 95% lower bound | Frozen gate | Status |
|---|---:|---:|---:|---:|
| Independent controller vs. fixed `q=8` | -0.3567% | -0.3932% | lower >= -2% | pass |
| Shared controller vs. fixed `q=8` | +4.9957% | +2.1890% | lower >= 0%; mean >= 4% | pass |
| Shared controller vs. fixed `q=1` | -0.8978% | -1.3446% | lower >= -2% | pass |
| Equal-weight mixture vs. fixed `q=8` | +2.3195% | +0.9172% | lower >= 0%; mean >= 2% | pass |

The probe consumed at most 0.768% of the message budget.
Maximum scalar selection overhead was `9.2391e-7` of training time.
All 48 records were finite, unique, exactly budgeted, and generated from the clean pinned HARL commit `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`.

## Execution

The experiment was split into six immutable eight-cell arrays to satisfy HPC4 QOS limits:

| Array | Tasks | Method and regime | Slurm result |
|---|---:|---|---|
| `1846514` | 0--7 | controller, independent | 8/8 `COMPLETED 0:0` |
| `1846655` | 8--15 | controller, shared | 8/8 `COMPLETED 0:0` |
| `1846941` | 16--23 | fixed `q=1`, independent | 8/8 `COMPLETED 0:0` |
| `1848265` | 24--31 | fixed `q=1`, shared | 8/8 `COMPLETED 0:0` |
| `1848550` | 32--39 | fixed `q=8`, independent | 8/8 `COMPLETED 0:0` |
| `1848644` | 40--47 | fixed `q=8`, shared | 8/8 `COMPLETED 0:0` |

Every per-cell `SHA256SUMS` file passed, exactly 48 result directories and 48 manifests were present, and the fatal-log scan was empty.
All outputs remained under `/scratch/jzhuangag/MARL-SDDE-TSP-MARL-CONF-001`; no result was written to `/project` or `/home`.

## Reproduction

The frozen analyzer ran once in `analysis` and once in the fresh `analysis_replay` directory.
The gate JSON, curve CSV, and PDF were byte-identical across the two invocations.

- Gate JSON SHA-256: `fbb931d6e7f149d72a7b9ce42cf958e01d586a8fb21d0f952cf4657763efd178`.
- Curve CSV SHA-256: `6e2c5d1af24952f72d83ee6347b06f5322ac5dd55bd2fcbf1df22ca8d39d3c64`.
- Return-curve PDF SHA-256: `ea40c05bbb4bdba346c4fb964792ce7c061232190948a5a04cfc8cf44ad1b748`.
- Complete artifact-manifest SHA-256: `5d38096aafe792b42a3adf3a79a6602013100b30c3244b2eb7f56028d68c9436`.

## Interpretation

The confirmation establishes the intended mechanism on a standard cooperative policy-learning benchmark under the registered equal-resource comparison.
The controller preserves broad participation when rollout innovations are independent and switches to local participation when shared innovations eliminate the statistical value of broader sampling.
The nonlinear benchmark uses the frozen empirical correlation upper proxy; theorem-level coverage remains attached to the affine delayed Markov model.

# T-071A sampled observable nonstationary graph validation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-27
- Verification Status: VERIFIED
- Version Label: t071a_validation_v1

## Decision

T-071A is a mandatory-gate failure. The independent split-probe observable
controller must not enter formal seeds, nonlinear benchmarks, GPU, or HPC4.

The negative result does not remove the dynamic collaboration value established
by T-070A. In the sampled model, the fully charged clairvoyant dynamic oracle
still reduces cumulative risk by 13.5291% relative to the strong static graph.
The observable controller reduces risk by 11.1663% relative to its same-budget
charged shadow, but its 24 independent probe transitions shorten the learning
horizon enough that it is 14.0315% worse than the no-probe strong static graph.

## Frozen provenance

- Preregistration commit: `7d4a4ee`
- Configuration SHA-256:
  `29cf65e9267720d27b9dfd10df9ed0845e51d75f8adc1800010e05280cbdede8`
- Endpoints SHA-256:
  `1432FDBE403A2ADA3D4C1734BAB476AE0F47F4765491D33245FD76B4A26CE446`
- Cells SHA-256:
  `CFCDBDD8799DBE0822A9630A810E143B955AA177EF09DE93A41F8DF7EF1666F6`
- Summary SHA-256:
  `29CE6F39115679822C6BEE66FDA6B490AECF280A00A702698CA2EEC89E1C033A`
- Workload: 432 cells, 32 seeds, 13,824 endpoints, 82,944 policy trajectories
- First run: 1,918.04 seconds on local CPU
- Clean reproduction: 1,803.57 seconds on local CPU

No formal seed, Slurm job, GPU job, HPC4 job, or `/project` artifact was
created.

## Primary result

| Metric | Frozen target | Observed | Result |
|---|---:|---:|---:|
| Observable improvement over strong static | >= 5% | -14.0315% | fail |
| Strictly improved nonstationary cells | >= 55% | 16.6667% | fail |
| Single-switch improvement | >= 2% | -18.2015% | fail |
| Alternating improvement | >= 2% | -10.0087% | fail |
| CVaR90 ratio to strong static | <= 0.97 | 1.15756 | fail |
| Shift-boundary graph-change rate | >= 60% | 97.2928% | pass |
| Stationary ratio to strong static | <= 1.10 | 1.35687 | fail |

All delay groups are negative: -10.9473%, -14.4391%, and -16.7841% for delays
0, 1, and 3. The degradation grows with staleness.

## Opportunity-cost diagnosis

On nonstationary cells:

- charged shadow / strong static geometric ratio: 1.283651;
- observable / charged shadow geometric ratio: 0.888337;
- observable / strong static geometric ratio: 1.140315;
- no-probe local / strong static geometric ratio: 1.038310;
- full sharing / strong static geometric ratio: 1.939068;
- clairvoyant dynamic / strong static geometric ratio: 0.864709.

Thus the controller's graph decisions have positive same-budget value, but the
independent probe architecture imposes a larger opportunity cost. Changing only
the acceptance threshold cannot repair this comparison because the charged
shadow is already 28.3651% worse than the static baseline before any observable
transfer decision.

The cross-fit screen is also not a finite-sample certificate: 17.1427% of
recipient decision checkpoints have larger realized squared error than the
same-data shadow. Aggregate observable/shadow risk is favorable, but only
84.3750% of nonstationary cells have ratio at most 1.05, below the frozen 90%
threshold.

## Descriptive phase analysis

This analysis was not a gate and cannot rescue T-071A.

| Target scale | Improvement over static | Improvement over charged shadow |
|---:|---:|---:|
| 0.1 | -27% | -1% |
| 0.3 | -20% | 5% |
| 0.6 | 3% | 26% |

The direction agrees with T-070A's separation phase: stronger target changes
increase the value of graph tracking. The observable mechanism is nevertheless
not broadly competitive under the frozen full grid.

## Gate ledger

After byte-exact reproduction, S1, S2, S10, and S12 pass. S3--S9 and S11 fail.
The final result is 4/12 mandatory gates passed. No threshold, seed, schedule,
cell, or comparator was changed.

## Reproducibility

The clean reproduction produced byte-identical `endpoints.csv`, `cells.csv`,
and `summary.json`. The targeted implementation suite passes nine tests. The
reproduction directory remains ignored.

## Fallacy scan

Coverage: 11/11.

1. Simpson's paradox: no aggregate-direction reversal across either schedule or any delay group; target-scale heterogeneity is reported explicitly.
2. Ecological fallacy: claims remain at the registered cell/policy level; no agent-level population inference is made.
3. Berkson's paradox: the full frozen grid and all seeds are retained; no outcome-conditioned filtering occurs.
4. Collider bias: no post-outcome covariate adjustment is used.
5. Base-rate neglect: not applicable; no diagnostic conditional probability is interpreted.
6. Regression to the mean: not applicable; cells were not selected by extreme pilot outcomes.
7. Survivorship bias: no failed or nonfinite endpoint was removed.
8. Look-elsewhere effect: all preregistered gates are reported, including failures.
9. Garden of forking paths: the independent preregistration fixes seeds, comparisons, and thresholds; target-scale breakdown is labeled descriptive.
10. Correlation versus causation: mechanism claims are limited to the controlled common-random-number simulator.
11. Reverse causality: target schedules precede graph decisions by construction.

## Next admissible mechanism

Any successor requires a new identifier. The allowed direction is a predictable
dual-use fingerprint controller: previous-block learning residuals select the
next graph, subsequent learning residuals verify transfer, and a Lyapunov
safety-debt queue controls rollback. It must use no extra environment
transitions for sensing, while charging every fingerprint message. T-071A may
not be rerun with fewer probes or altered thresholds.

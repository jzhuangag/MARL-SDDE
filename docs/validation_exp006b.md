# Validation report: EXP-006B observable state-correlation controller

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: validation_v1

## Experiment result

- **ID**: EXP-006B-state-correlation-controller
- **Type**: stochastic simulation with paired seeds
- **Status**: completed
- **Primary duration**: 189.9 seconds
- **Reproduction duration**: 129.6 seconds
- **Primary command**:
  `python run_state_correlation.py --output-dir results/state_correlation --num-seeds 64 --bootstrap-replications 2000`
- **Exit code**: 0
- **Anomalies**: none

All 3,584 registered policy runs completed. The two adaptive policies used only
charged gradient probes; simulator noise and true error were retained solely
as audit columns and did not enter their action selection.

## Preregistered findings

| Gate | Observed result | Verdict |
|---|---:|---|
| State value versus correlation-only | ratio 1.131; 95% upper 1.172 | FAIL |
| Improvement over best fixed \(q=4\) | ratio 1.728; 95% upper 1.822 | FAIL |
| Oracle proximity | score 2.652; 95% upper 2.827 | FAIL |
| Action agreement | 34.35% | FAIL |
| Probe budget | 768 units = 4.8% | PASS |
| Accounting/numerical validity | 3,584/3,584 valid runs | PASS |
| Overall | all six gates required | **FAIL** |

Mean normalized scores were 1.000 for the state oracle, 1.535 for fixed
\(q=4\), 1.543 for fixed \(q=8\), 2.018 for fixed \(q=1\), 2.155 for fixed
\(q=32\), 2.344 for the correlation-only controller, and 2.652 for the
state-correlation controller.

## Root-cause audit

The dependence estimator was not the dominant failure. Median estimated
global/cluster components were 0.672/0.000 in the global cell, 0.000/0.751 in
the clustered cell, 0.288/0.438 in the balanced cell, and 0.013/0.000 in the
independent cell.

The observable state proxy was poorly calibrated. Its log-scale correlation
with true error magnitude was 0.326, and its median ratio to true error was
2.89. Common Markov noise made a short-window absolute gradient mean remain
large after the optimization error had contracted. Consequently, the adaptive
controller retained median \(q=1\) in global/balanced cells and \(q=4\) in
clustered cells, while the state oracle generally returned to \(q=32\) after
one or two blocks.

This is a substantive negative result: an oracle participation phase diagram
cannot be implemented by simply substituting a short-window gradient magnitude
for the unknown optimization error.

## Reproducibility

- **Verdict**: REPRODUCIBLE
- **Method**: deterministic same-seed full rerun in an isolated output path.

All nine artifacts matched byte-for-byte by SHA-256, including all CSV files,
`summary.json`, and the three figures.

## Fallacy scan

- **Coverage**: 11/11 statistical fallacy types checked.

| Fallacy | Status | Finding |
|---|---|---|
| Simpson's paradox | NOTE | The controller is worse overall and in every dependence scenario; no aggregate reversal rescues it. |
| Ecological fallacy | NOTE | Seed-averaged scores are not interpreted as individual-agent behavior. |
| Berkson's paradox | NOTE | All registered seeds and cells were retained. |
| Collider bias | NOTE | No outcome-dependent conditioning was used. |
| Base-rate neglect | NOTE | Not applicable. |
| Regression to the mean | NOTE | Seeds were prospectively fixed and not selected by outcome. |
| Survivorship bias | NOTE | Every registered run completed. |
| Look-elsewhere effect | NOTE | Smoke output was excluded and all primary gates were preregistered. |
| Garden of forking paths | NOTE | The failed proxy and thresholds were not altered after execution. |
| Correlation versus causation | NOTE | Dependence is controlled in simulation; claims remain mechanism-specific. |
| Reverse causality | NOTE | Not applicable to the controlled update process. |

## Validated conclusion

EXP-006B rejects the raw gradient-magnitude state proxy. It does not reject the
oracle mechanism from EXP-006A. The next defensible controller should propagate
a model-based Lyapunov state bound using estimated long-run variance and the
chosen delayed transition, rather than attempting to infer optimization error
directly from a short noisy window.

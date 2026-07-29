# Validation report: EXP-005B online probe-charging controller

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-29
- Verification Status: VERIFIED
- Version Label: validation_v1

## Validation Report

- **Source**: EXP-005B-online-probe-controller
- **Overall Confidence**: RED_FLAG for the proposed full-probe controller;
  SOLID for the recorded simulator comparisons.

The controller detects the intended participation regimes and strongly
improves on full participation under correlated noise. However, it does not
beat the best registered fixed-\(q\) policy after its probe cost is charged.
The pre-registered overall result is therefore a failure.

### Statistical findings

| Metric | Method | Value | Interpretation | Confidence |
|---|---|---:|---|---|
| Correlated adaptive/all-agent MSE | seed-level paired bootstrap | 0.3445, 95% interval [0.2317, 0.5348] | Large resource-matched gain over retaining all agents | SOLID within simulator |
| Adaptive/probe-oracle MSE | seed-level paired bootstrap | 1.0833, interval [0.9375, 1.2699] | Estimated controller is close to the same-cost information oracle | SOLID within simulator |
| Adaptive/best-fixed MSE | seed-level paired bootstrap | 1.0584, interval [0.8210, 1.3573] | No demonstrated gain over fixed \(q=1\); registered threshold failed | RED_FLAG for main controller claim |
| Full-budget AUC | descriptive mean | adaptive 0.06425; fixed \(q=1\) 0.01732 | Full probing severely harms anytime performance | RED_FLAG |
| Participation response | paired action summaries | independent median 16; correlated medians 1--4 | Controller responds in the intended direction | SOLID |

The three bootstrap comparisons were pre-specified. The experiment uses an
intersection gate, so the unadjusted intervals are descriptive and cannot
rescue the failed overall decision.

### Warnings

| Type | Detail | Affected claim |
|---|---|---|
| Probe burden | The full probe costs 2,880/16,000 = 18% of budget | Anytime and final-window adaptivity |
| Cross-cell scale | Global and mixed cells have much larger raw MSE and dominate the aggregate best-fixed comparison | Interpretation of a single cross-environment ratio |
| Static environments | Each run has one stationary dependence regime | Value of adaptation versus a fixed policy chosen for the task distribution |
| Oracle scope | The oracle shares probe cost but knows exact long-run variances | Oracle proximity |
| Model scope | Scalar linear dynamics with constructed clustered factors | TD and nonlinear RL generalization |

### Fallacy scan

- **Coverage**: 11/11 statistical fallacy types checked

| Fallacy | Severity | Finding |
|---|---|---|
| Simpson's paradox | CAUTION | Subgroup directions differ and raw-MSE scales vary; no strict reversal is present, but the aggregate fixed-policy comparison is dominated by global/mixed cells. |
| Ecological fallacy | NOTE | Claims remain at simulator-policy level and are not transferred to individual agents. |
| Berkson's paradox | NOTE | Fastest-prefix selection is fixed for all policies; cluster identities and loadings are interleaved with latency. |
| Collider bias | NOTE | No post-treatment covariate adjustment is used. |
| Base-rate neglect | NOTE | No diagnostic probabilities are reported. |
| Regression to the mean | NOTE | No extreme group is selected for pre/post comparison. |
| Survivorship bias | NOTE | All 64 seeds, eight cells, and five policies are retained; no failed trajectory is dropped. |
| Look-elsewhere effect | NOTE | All registered gates and comparisons are reported, including the failed gate. |
| Garden of forking paths | NOTE | Design, thresholds, and aggregation were recorded before smoke and primary output. |
| Correlation versus causation | NOTE | Effects are controlled interventions inside the simulator; no real-system causal claim is made. |
| Reverse causality | NOTE | Not applicable to the controlled simulation. |

### Reproducibility

- **Method**: deterministic same-code, same-seed independent rerun;
- **Verdict**: REPRODUCIBLE.

`actions.csv`, `budget_trajectories.csv`, both figures,
`paired_bootstrap_ratios.csv`, `per_seed_metrics.csv`, and `summary.json` were
byte-identical by SHA-256.

### Validated conclusion

A scalar long-run-variance controller can infer whether few or many agents
should participate under heterogeneous clustered Markov noise. The current
80-round full-probe design is not resource-efficient enough to outperform the
best fixed participation baseline and should not be presented as the final
algorithm.


# Validation report: EXP-005C sparse dynamic controller

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-29
- Verification Status: VERIFIED
- Version Label: validation_v2

## Experiment result

- **ID**: EXP-005C-sparse-dynamic-controller
- **Type**: simulation
- **Status**: completed
- **Execution version**: v2 (`numba_block_v2`)
- **Command**:
  `python run_sparse_dynamic.py --output-dir results/sparse_dynamic --num-seeds 64 --bootstrap-replications 2000`
- **Working directory**: `experiments/dependence_delay_linear`
- **Duration**: 102.5 seconds
- **Exit code**: 0
- **Anomalies**: none

The v2 kernel was admitted only after all six policies matched the NumPy
reference implementation on checkpoint errors to relative tolerance
\(10^{-12}\), with identical block actions and exact budget accounting.

## Preregistered findings

| Gate | Observed result | Verdict |
|---|---:|---|
| Adaptive / best fixed | 2.619; paired-bootstrap 95% CI [1.628, 4.080] | FAIL |
| Adaptive / piecewise oracle | 8.226; paired-bootstrap 95% CI [6.215, 10.580] | FAIL |
| Correct switch response | 1/4 regimes | FAIL |
| Probe budget | 768 units = 2.4% | PASS |
| Accounting and numerical validity | all runs finite and within budget | PASS |
| Overall | all five gates required | **FAIL** |

The hindsight-best registered fixed baseline was
`fixed_q32_oracle_eta`. Mean normalized dynamic scores were 1.000 for the
piecewise oracle, 3.141 for fixed \(q=32\), 7.303 for fixed \(q=8\), 8.226 for
the sparse controller, 8.980 for fixed \(q=4\), and 24.370 for fixed \(q=1\).
Both primary confidence intervals lie wholly above one, so the two performance
failures are not borderline bootstrap outcomes.

## Diagnostic boundary

The preregistered decision is unambiguous: the present sparse controller does
not qualify as the paper's main algorithmic contribution.

This result does not establish that adaptive agent participation is generally
unhelpful. The registered switch directions are not aligned with the behavior
of the experiment's own information oracle. The piecewise oracle selected
median \(q=32\) in every regime; its regime-level frequencies assigned
\(q=32\) in 97.3% of independent, 88.1% of clustered, 88.1% of global, and
71.1% of mixed blocks. Consequently, the registered requirements
\(q\leq8\), \(q\leq4\), and \(q\leq8\) in the three correlated regimes are not
oracle-supported consequences of the finite-budget proxy used here.

The sparse moment estimator did respond to dependence: median estimated
global/cluster components were approximately 0.000/0.001 for independent,
0.000/0.451 for clustered, 0.377/0.083 for global, and 0.301/0.317 for mixed.
The main failure therefore lies in the complete estimator-to-action objective,
not simply in an inability to detect correlation. The controller's selected
participation was also highly bimodal, with \(q=32\) used in 55.9%--72.7% of
correlated-regime blocks and \(q=1\) used in 0.2%--29.5%.

## Reproducibility

- **Method**: deterministic full rerun with the same code, seeds, delays,
  policies, and 2,000 bootstrap resamples; output isolated under
  `results/reproduction/sparse_dynamic`
- **Duration**: 83.6 seconds
- **Verdict**: REPRODUCIBLE

All eight expected artifacts matched byte-for-byte by SHA-256:
`block_actions.csv`, `budget_trajectories.csv`,
`fig1_dynamic_mse.png`, `fig2_block_participation.png`,
`paired_bootstrap_ratios.csv`, `per_seed_regime_metrics.csv`,
`run_accounting.csv`, and `summary.json`.

## Fallacy scan

- **Coverage**: 11/11 statistical fallacy types checked.

| Fallacy | Status | Finding |
|---|---|---|
| Simpson's paradox | NOTE | Delay/regime-stratified results do not reverse the adaptive-versus-oracle failure. |
| Ecological fallacy | NOTE | No individual-agent inference is drawn from aggregate seed scores. |
| Berkson's paradox | NOTE | No outcome-conditioned sample selection occurred. |
| Collider bias | NOTE | No post-treatment or common-effect conditioning was used. |
| Base-rate neglect | NOTE | Diagnostic classification metrics are not involved. |
| Regression to the mean | NOTE | Seeds were fixed prospectively and not selected for extreme outcomes. |
| Survivorship bias | NOTE | All 64 paired seeds, both delays, and all policies completed. |
| Look-elsewhere effect | NOTE | Comparisons, normalization, and gates were preregistered; smoke results were excluded. |
| Garden of forking paths | NOTE | The failed gates were retained without reweighting or threshold changes. |
| Correlation versus causation | NOTE | This is a controlled simulation comparison; no observational causal claim is made. |
| Reverse causality | NOTE | Not applicable to the randomized simulation paths and policy interventions. |

## Validated conclusion

EXP-005C rejects the current sparse dynamic participation controller as a main
paper contribution. It simultaneously reveals that the current finite-budget
proxy and switch gate do not provide a valid test of the broader hypothesis
that correlation should induce fewer participating agents. Any subsequent
experiment must first establish an oracle participation surface with genuine
regime-dependent \(q\) transitions; it must be registered as a new experiment,
not treated as a repair of EXP-005C.

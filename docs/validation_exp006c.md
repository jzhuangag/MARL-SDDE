# Validation report: EXP-006C Lyapunov-surrogate controller

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: validation_v1

## Experiment result

- **ID**: EXP-006C-lyapunov-state-controller
- **Type**: preregistered stochastic simulation with paired seeds
- **Status**: completed
- **Primary duration**: approximately 771 seconds, including cold compilation
- **Reproduction duration**: approximately 156 seconds
- **Primary command**:
  `python run_lyapunov_state.py --output-dir results/lyapunov_state --num-seeds 64 --bootstrap-replications 2000`
- **Exit code**: 0
- **Anomalies**: none

All 4,096 registered policy runs completed. The three implementable adaptive
policies used only charged probe gradients. True error, true dependence
components, and simulator noise did not enter their action selection.

## Preregistered findings

| Gate | Observed result | Verdict |
|---|---:|---|
| Replacement value versus raw state | ratio 0.705; 95% upper 0.729 | PASS |
| State value versus correlation-only | ratio 0.819; 95% upper 0.833 | PASS |
| Improvement over best fixed \(q=4\) | ratio 1.204; 95% upper 1.264 | FAIL |
| Oracle proximity | score 1.887; 95% upper 1.993 | FAIL |
| Action agreement | 26.67% | FAIL |
| Probe budget | 768 units = 4.8% | PASS |
| Accounting/numerical validity | 4,096/4,096 valid runs | PASS |
| Overall | all seven gates required | **FAIL** |

Mean normalized scores were 1.000 for the state oracle, 1.567 for fixed
\(q=4\), 1.601 for fixed \(q=8\), 1.887 for the Lyapunov controller, 1.938
for fixed \(q=1\), 2.305 for correlation-only, 2.311 for fixed \(q=32\), and
2.677 for the raw-state controller.

The Lyapunov replacement is a real improvement over both failed observable
controllers, but it does not satisfy the registered scientific claim. A
statistically clear improvement over a weak predecessor is not evidence of
competitiveness against the strongest baseline.

## Root-cause audit

### Dependence estimation

The dependence estimator remained directionally accurate. Median estimated
global/cluster components were 0.655/0.000 in the global cell, 0.000/0.753 in
the clustered cell, 0.282/0.445 in the balanced cell, and 0.000/0.000 in the
independent cell. Dependence estimation is therefore not the primary failure.

### State calibration

The Lyapunov surrogate improved log-scale state correlation from 0.326 in
EXP-006B to 0.564, and reduced the median predicted/true state ratio from 2.89
to 1.81. It nevertheless remained systematically conservative in every
scenario. Median predicted/true ratios were 1.66 (balanced), 1.72
(clustered), 2.00 (global), and 1.90 (independent).

The participation decision surface is sharp near the noise-dominated regime.
This modest state-scale mismatch produced a large action mismatch. After
warm-up, the Lyapunov controller usually retained \(q=1\) in balanced/global,
\(q=4\) in clustered, and \(q=4\) or \(8\) in independent cells. The
realized-state oracle usually selected \(q=32\) from block 1 or 2 onward.

### Comparator mismatch

The deeper issue is not merely calibration. The scalar recursion predicts an
ensemble risk under estimated long-run variance, whereas the registered oracle
conditions on the current realized signed delayed-error history. Persistent
common Markov noise keeps the ensemble risk positive even when one realized
trajectory happens to be close to its optimum. An observable conservative
upper surrogate therefore cannot be expected to imitate a comparator that
knows the unobservable realized error.

This distinction matters for the theory. A theorem-quality Lyapunov controller
should compete with a bound oracle or a predictable-information oracle, not
with a clairvoyant realized-state oracle. Changing the comparator after seeing
EXP-006C would invalidate its gates, so the registered result remains FAIL.

## Scenario audit against the strongest fixed baseline

The Lyapunov/fixed-\(q=4\) mean MSE ratios were above one in all eight
scenario-delay cells. They ranged from 1.078 in global/\(D=16\) to 1.592 in
clustered/\(D=16\). There is no Simpson reversal hidden by aggregation.

## Reproducibility

- **Verdict**: REPRODUCIBLE
- **Method**: deterministic same-seed full rerun in an isolated output path.

All nine artifacts matched byte-for-byte by SHA-256:

- `block_actions.csv`;
- `budget_trajectories.csv`;
- `fig1_mse_by_scenario.png`;
- `fig2_participation_by_block.png`;
- `fig3_surrogate_calibration.png`;
- `paired_bootstrap_ratios.csv`;
- `per_seed_cell_metrics.csv`;
- `run_accounting.csv`;
- `summary.json`.

The complete test suite also passed: 30 tests, with one unrelated
`pyreadline` deprecation warning.

## Fallacy scan

- **Coverage**: 11/11 statistical fallacy types checked.

| Fallacy | Status | Finding |
|---|---|---|
| Simpson's paradox | NOTE | Lyapunov/fixed-\(q=4\) exceeds one in every scenario-delay cell. |
| Ecological fallacy | NOTE | Seed/cell averages are not interpreted as individual-agent behavior. |
| Berkson's paradox | NOTE | All registered seeds, cells, and policies were retained. |
| Collider bias | NOTE | No outcome-dependent conditioning was used. |
| Base-rate neglect | NOTE | All four scenarios and both delays retain equal registered weight. |
| Regression to the mean | NOTE | Fresh seeds were fixed before smoke and primary execution. |
| Survivorship bias | NOTE | Every registered run completed and is finite. |
| Look-elsewhere effect | NOTE | Smoke output is excluded; primary comparisons were preregistered. |
| Garden of forking paths | NOTE | No initialization, estimator, grid, gate, or cell was altered after execution. |
| Correlation versus causation | NOTE | Dependence is intervened on in simulation; generalization remains unclaimed. |
| Reverse causality | NOTE | Actions are predictable from past probes; audit-only true error is not an input. |

## Validated decision

EXP-006C rejects this exact online Lyapunov-surrogate controller as the main
algorithm. It also shows that replacing a noisy raw gradient proxy by a scalar
model recursion is useful but insufficient.

The project should now:

1. stop tuning observable state proxies on this synthetic oracle;
2. retain the correlation-limited speedup and finite-budget participation
   phase transition as the main theorem target;
3. compare future predictable controllers with an information-matched oracle;
4. move the next experiment to linear TD/Markov policy evaluation with
   controlled cross-agent common factors, rather than another scalar proxy
   variant;
5. keep delay in the Lyapunov--Krasovskii stability/rate theorem, without
   claiming that delay must directly change the optimal agent count.

# Validation report: EXP-007B active delayed-TD stability

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: validation_v1

## Experiment result

- **ID**: EXP-007B-td-delay-stability
- **Type**: exact mean-system spectral analysis plus stochastic TD simulation
- **Status**: completed
- **Primary duration**: approximately 290 seconds
- **Reproduction duration**: approximately 276 seconds
- **Executed source commit**: `7301609`
- **Primary command**:
  `python run_td_delay_stability.py --output-dir results/td_delay_stability --num-seeds 16`
- **Exit code**: 0
- **Anomalies**: none

All 1,632 registered stochastic runs completed. Divergent trajectories stopped
at the preregistered squared-error threshold \(10^{12}\), so floating-point
overflow was never used as an outcome.

## Preregistered findings

| Gate | Observed result | Verdict |
|---|---:|---|
| Active boundary | \(D=32/D=0\) critical-\(\eta\) ratios 0.307 (\(q=16\)) and 0.127 (\(q=32\)) | PASS |
| Exact spectral separation | 27/27 low-multiplier stable; 18/18 high-multiplier unstable | PASS |
| Monte Carlo boundary agreement | \(m=0.8\) has crossings; \(m=1.2\) delayed crossing rate 93.23% | FAIL |
| Delay-adaptive value | delay-blind crossing 100%, but proposed adaptive rule also crosses | FAIL |
| Correlation/stability separation | high-correlation error no smaller in 9/9 cells | PASS |
| Accounting/determinism/numerics | 1,632/1,632 valid runs | PASS |
| Overall | all six gates required | **FAIL** |

## Exact mean stability boundary

The mean-recursion critical step sizes were:

| \(q\) | \(D=0\) | \(D=8\) | \(D=32\) |
|---:|---:|---:|---:|
| 8 | 4.210 | 8.420 | 2.768 |
| 16 | 4.210 | 4.115 | 1.293 |
| 32 | 4.210 | 2.103 | 0.535 |

Delay is now unquestionably active for the larger participation levels. The
nonmonotone \(q=8,D=8\) value arises because the first eight registered agents
have selected maximum delay one; a one-step heterogeneous mean recursion can
have a different first stability boundary from the synchronous recursion.

## Root-cause audit: mean stability is insufficient

The exact companion matrix governs only
\(\mathbb E[e_t]\). Stochastic TD contains sample-dependent Jacobians, so its
homogeneous component is a Markov jump linear system. Stability of
\(\mathbb E[e_t]\) does not imply stability of
\(\mathbb E[\|e_t\|^2]\), nor almost-sure stability of the random matrix
product.

This distinction is large, not asymptotic bookkeeping. At \(m=0.8\):

- for \(q=8\), every run crossed the threshold at both correlation levels and
  all delays;
- at \(\rho=0\), \(q=16,32,D=32\) had zero crossings, although some median
  final errors remained above the initial value;
- at \(\rho=0.9\), \(q=16,D=32\) had 100% crossings and
  \(q=32,D=32\) had 75% crossings;
- the delay-blind rule had 100% crossings in the registered target cells.

Thus cross-agent correlation changes the stochastic/mean-square stability
region even though it leaves the mean companion matrix unchanged. This is the
nontrivial correlation--delay interaction that EXP-007A did not activate.

## Consequence for SDDE and theorem design

An additive-noise SDDE with a deterministic drift Jacobian is insufficient for
this TD regime. The limiting/stability model must retain state-dependent or
multiplicative diffusion, or the discrete-time theorem must directly control
the random delayed Jacobian.

A defensible one-step target is

\[
\mathbb E_k[\mathcal V_{k+1}]
\le
\left[
1-c_1\mu\eta+
c_2\eta^2\Lambda_{\rm mult}(S_k)+
c_3\eta^2L^2\Psi(\tau_k)
\right]\mathcal V_k
+c_4\eta^2\Omega_{\rm add}(S_k),
\]

where \(\Lambda_{\rm mult}\) includes cross-agent long-run covariance of
sample Jacobians. The safe step size must therefore depend on both delay and
effective participation/correlation. A mean-spectral lookup is rejected.

## Reproducibility

- **Verdict**: REPRODUCIBLE
- **Method**: deterministic same-seed full rerun in an isolated output path.

All seven artifacts matched byte-for-byte by SHA-256. The complete experiment
suite passed 38 tests; the only warning was an unrelated `pyreadline`
deprecation notice.

## Fallacy scan

- **Coverage**: 11/11 statistical fallacy types checked.

| Fallacy | Status | Finding |
|---|---|---|
| Simpson's paradox | NOTE | Correlation-specific crossing rates are reported; the failure is not inferred only from an aggregate. |
| Ecological fallacy | NOTE | Mean-system stability is explicitly not transferred to individual trajectories. |
| Berkson's paradox | NOTE | No run was conditioned on remaining finite; crossings are retained. |
| Collider bias | NOTE | No outcome-dependent cell selection was used. |
| Base-rate neglect | NOTE | Both registered correlation levels and every \(q,D,m\) cell are retained. |
| Regression to the mean | NOTE | Seeds and threshold were fixed before execution. |
| Survivorship bias | NOTE | Threshold-crossing runs are recorded, not discarded. |
| Look-elsewhere effect | NOTE | Smoke is excluded; formal gates use the preregistered multipliers. |
| Garden of forking paths | NOTE | The failed 0.8 rule was not reduced after the implementation test or smoke. |
| Correlation versus causation | NOTE | \(\rho\), delay, and multiplier are controlled interventions in the simulator. |
| Reverse causality | NOTE | Stability outcomes do not alter paths, delays, or step sizes. |

## Validated decision

Reject the mean-companion critical step size as the paper's delay-adaptive
controller. Retain the exact mean boundary only as a diagnostic and lower
layer of the proof.

Promote the new main theoretical question:

> How do cross-agent Markov correlations and heterogeneous delays jointly
> determine the mean-square stability and finite-budget effective
> participation of multi-agent TD/stochastic approximation?

This question joins the strongest positive result from EXP-007A with the
reproducible failure mechanism from EXP-007B. It is more novel and technically
harder than a delay-only or correlation-only analysis, while still admitting a
low-complexity scalar safe-step/participation rule if a conservative
Lyapunov bound can be derived.

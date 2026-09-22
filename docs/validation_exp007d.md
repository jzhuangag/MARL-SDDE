# Validation report: EXP-007D fresh mean-square confirmation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: validation_v1

## Experiment result

- **ID**: EXP-007D-joint-ms-confirmation
- **Type**: fresh-seed stochastic delayed-TD confirmation with deterministic
  bootstrap inference
- **Status**: completed; all seven preregistered gates **PASS**
- **Executed source commit**: `d55a953`
- **Formal command**:
  `python run_joint_ms_confirmation.py --output-dir results/joint_ms_confirmation --num-seeds 64 --base-seed 20270130`
- **Exit code**: 0
- **Formal runs**: 9,216
- **Bootstrap**: 20,000 resamples, fixed seed 20270730
- **Anomalies**: none

## Preregistered findings

| Gate | Observed result | Verdict |
|---|---:|---|
| Analytic participation saturation | independent \(K\) reduction 22.46%; \(\rho=0.9\) reduction 0.32%; inflation 7.07--9.08 times | PASS |
| Joint mean-square contraction | 0/512 crossings; largest 99% upper mean-error limit 0.649 | PASS |
| Correlation-awareness value | smallest 99% lower paired error-ratio limit 8.193 | PASS |
| Delay-awareness value | smallest 99% lower paired error-ratio limit 1.110 | PASS |
| Nonvacuous tightness | smallest joint/useful-grid step ratio 0.545 | PASS |
| Correlation adaptation retains speed | 4/4 cells no slower; 4/4 use at least fourfold larger step | PASS |
| Accounting/numerics/reproducibility | 9,216/9,216 rows valid; 10/10 artifacts exact | PASS |
| Overall | all seven gates required | **PASS (7/7)** |

## Mean-square contraction

The unchanged joint rule was

\[
\eta_{\rm joint}^{-1}
=
\eta_{\rm mean}^{-1}
+K(q,\rho)/(2\mu).
\]

Its final-error estimates were:

| \(\rho\) | \(q\) | \(D\) | Mean | Median | 99% bootstrap upper mean |
|---:|---:|---:|---:|---:|---:|
| 0 | 16 | 8 | 0.203 | 0.171 | 0.261 |
| 0 | 16 | 32 | 0.236 | 0.156 | 0.304 |
| 0 | 32 | 8 | 0.148 | 0.108 | 0.184 |
| 0 | 32 | 32 | 0.151 | 0.130 | 0.185 |
| 0.9 | 16 | 8 | 0.440 | 0.390 | 0.548 |
| 0.9 | 16 | 32 | 0.465 | 0.379 | 0.585 |
| 0.9 | 32 | 8 | 0.451 | 0.371 | 0.562 |
| 0.9 | 32 | 32 | 0.508 | 0.390 | 0.649 |

Every upper limit is below the initial squared error one.  This confirms
finite-horizon mean-square contraction rather than merely absence of numerical
explosion.

## Participation and correlation effect

For the registered exchangeable observation model,

\[
\mathbb E[\bar H^\top\bar H]
=
\alpha(q,\rho)B+[1-\alpha(q,\rho)]A^\top A,\qquad
\alpha(q,\rho)=\rho+\frac{1-\rho}{q}.
\]

At \(\rho=0\), doubling participation from 16 to 32 reduces the largest
second-moment eigenvalue by 22.46%.  At \(\rho=0.9\), the same reduction is
only 0.32%.  This is a direct agent-count effect, not an interpretation of a
flat MSE curve: nominal \(q\) enters the exact multiplicative-noise operator,
but common correlation makes its benefit saturate.

The paired `correlation_blind / joint_aware` median final-error ratios and
their preregistered 99% lower bootstrap limits were:

| \(q\) | \(D\) | Median ratio | 99% lower limit |
|---:|---:|---:|---:|
| 16 | 8 | 12.23 | 8.19 |
| 16 | 32 | 17.96 | 12.60 |
| 32 | 8 | 33.95 | 22.39 |
| 32 | 32 | 25.17 | 16.85 |

Thus using nominal agent count as if observations were independent is
systematically inaccurate in the high-correlation cells.

## Delay effect and tightness

At \(q=32,D=32\), the paired `delay_blind / joint_aware` median final-error
ratio was 889.90 under independence and 1.128 under \(\rho=0.9\).  Their 99%
lower limits were 431.33 and 1.110, respectively.  The smaller effect under
high correlation is expected because the multiplicative-noise term already
dominates the parallel-sum step.

The joint step was between 0.545 and 0.970 of the largest registered grid step
whose crossing rate was at most 5% and whose 99% upper mean-error limit was
below one.  It is therefore conservative but not vacuous on the tested
family.  Under independence it used a 5.1--7.9 times larger step than the
worst-correlation rule and reached error 0.5 no later in all four cells.

## Reproducibility

- **Verdict**: REPRODUCIBLE
- **Method**: deterministic same-seed full rerun in a separate output path.
- **Artifacts**: 10/10 matched byte-for-byte by SHA-256.
- **Test suite**: 46 passed; one unrelated `pyreadline` deprecation warning.

## Fallacy scan

- **Coverage**: 11/11 statistical fallacy types checked.

| Fallacy | Status | Finding |
|---|---|---|
| Simpson's paradox | NOTE | Every gate is all-cell; the aggregate pass cannot hide a reversed cell. |
| Ecological fallacy | NOTE | Claims concern the registered trajectory distribution and are not transferred to individual real systems. |
| Berkson's paradox | NOTE | Runs are never conditioned on finite or successful outcomes. |
| Collider bias | NOTE | Paired comparisons use fixed common paths and no outcome-dependent controls. |
| Base-rate neglect | NOTE | Continuous errors, confidence limits, and crossing rates are reported together. |
| Regression to the mean | NOTE | The confirmation uses a new fixed 64-seed sample and an unchanged rule. |
| Survivorship bias | NOTE | Every one of 9,216 rows enters the saved aggregates. |
| Look-elsewhere effect | NOTE | The endpoints, 99% limits, thresholds, and all-cell rules were frozen before execution. |
| Garden of forking paths | NOTE | EXP-007C remains a formal failure; EXP-007D is a separately preregistered confirmation. |
| Correlation versus causation | NOTE | \(\rho\), \(q\), delay, and policy are controlled interventions in the simulator. |
| Reverse causality | NOTE | Outcomes cannot alter paths, policies, or fixed step sizes. |

## Validated decision

Promote the joint correlation--delay scalar rule to the paper's algorithmic
prototype for linear TD.  Its empirical status is now confirmed on the
registered finite MRP, not yet universal.

The next proof target is not the empirical constant itself.  It is a
Lyapunov--Krasovskii theorem producing a sufficient condition of the form

\[
\eta^{-1}
\ge
c_D\,\eta_{\rm mean}^{-1}
+c_M\,K(q,\rho)/\mu
+c_{\rm mix}\,\tau_{\rm mix}L^2/\mu,
\]

with explicit constants and a discrete-time error bound.  The SDDE should
retain state-dependent multiplicative diffusion whose local quadratic
variation is controlled by \(K(q,\rho)\).


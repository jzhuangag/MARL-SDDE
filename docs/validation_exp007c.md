# Validation report: EXP-007C joint mean-square step size

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: validation_v1

## Experiment result

- **ID**: EXP-007C-joint-mean-square-step
- **Type**: analytic second-moment calculation plus stochastic delayed-TD
  simulation
- **Status**: completed; preregistered overall verdict **FAIL**
- **Executed source commit**: `862d15b`
- **Formal command**:
  `python run_joint_mean_square_step.py --output-dir results/joint_mean_square_step --num-seeds 32 --base-seed 20261230`
- **Exit code**: 0
- **Formal runs**: 4,608
- **Anomalies**: none

## Preregistered findings

| Gate | Observed result | Verdict |
|---|---:|---|
| Analytic correlation saturation | independent \(K\) falls 22.46% from \(q=16\) to 32; \(\rho=0.9\) \(K\) falls 0.32%; correlation inflates \(K\) by 7.07--9.08 times | PASS |
| Joint safety and contraction | 0/256 crossings; largest cell median final error 0.561 | PASS |
| Correlation-awareness crossing value | correlation-blind has 0 crossings at \(\rho=0.9\) | FAIL |
| Delay-awareness crossing value | delay-blind has 0 crossings at \(q=32,D=32\) | FAIL |
| Nonvacuous crossing-boundary tightness | 4/8 cells within one quarter of the largest non-crossing grid step; speed subgate 4/4 | FAIL |
| Accounting/determinism/numerics | 4,608/4,608 valid rows | PASS |
| Overall | all six gates required | **FAIL (3/6)** |

The formal failure is retained.  In particular, threshold crossings cannot be
claimed as evidence that either blind rule is unusable at its registered step
size.

## What the failure does and does not mean

The implementation and experimental setting are valid.  The failed gates used
crossing of squared error \(10^{12}\) as the operational definition of an
inadequate mean-square step size.  That event is too coarse: a trajectory can
be finite yet fail to contract, or can settle at an error orders of magnitude
larger than a correlation-aware trajectory.

This occurred systematically.  The correlation-blind rule had no catastrophic
crossing but, at \(\rho=0.9\), its median final error was 5.33--15.90, whereas
the joint rule's corresponding medians were 0.500--0.561.  The delayed
\(q=32,D=32,\rho=0\) mean-only rule also had no crossing, yet its median final
error was 21.68.  Therefore “no crossing” is not mean-square stability at the
finite horizon and should not be used as its sole empirical proxy.

## Registered positive evidence

The analytic identity is exact for the registered exchangeable pair-sharing
model:

\[
\mathbb E[\bar H^\top\bar H]
=
\alpha(q,\rho)\mathbb E[H^\top H]
+[1-\alpha(q,\rho)]A^\top A,\qquad
\alpha(q,\rho)=\rho+\frac{1-\rho}{q}.
\]

It exhibits the desired participation effect.  Doubling \(q\) reduces the
multiplicative curvature by 22.46% under independence but only 0.32% at
\(\rho=0.9\).  Thus agent count participates explicitly in the stochastic
stability quantity, while high common correlation removes almost all of its
benefit.

The registered joint rule also satisfied the non-catastrophic part of the
hypothesis:

- it produced zero crossings in all 256 runs;
- every cell median contracted below the initial squared error one;
- it reached squared error 0.5 at least as quickly as the worst-correlation
  safe rule in all four independent-agent cells;
- mean-only steps crossed in seven of eight cells, and the remaining cell did
  not contract in median.

## Labelled post-hoc diagnostic

The following calculations diagnose the gate, not repair the preregistered
verdict.

For each high-correlation cell, pair each correlation-blind run with the joint
run on the same transition paths.  The median final-error ratios
`correlation_blind / joint_aware`, with a 10,000-resample paired bootstrap
95% interval, are:

| \(q\) | \(D\) | Median ratio | 95% bootstrap interval |
|---:|---:|---:|---:|
| 16 | 8 | 10.58 | [5.55, 14.18] |
| 16 | 32 | 13.33 | [9.65, 28.28] |
| 32 | 8 | 34.16 | [21.95, 61.92] |
| 32 | 32 | 41.97 | [14.81, 81.63] |

For \(q=32,D=32\), the delay-blind/joint paired median final-error ratio is
2,223 under \(\rho=0\) and 1.184 under \(\rho=0.9\).  The corresponding
bootstrap intervals are [744.70, 6,234.47] and [1.149, 1.218].

If a grid step is called useful only when its crossing rate is at most 5%
**and** its cell median final error is below the initial value, the joint step
is 0.461--0.970 of the largest useful grid step in all eight cells.  The
registered crossing-only oracle instead admitted finite but noncontracting
steps and made the joint rule appear vacuously conservative.

These post-hoc results justify a fresh-seed confirmation with a
mean-square-contraction endpoint.  They do not convert EXP-007C to a pass.

## Reproducibility

- **Verdict**: REPRODUCIBLE
- **Method**: deterministic same-seed full rerun in a separate output path.
- **Artifacts**: 8/8 matched byte-for-byte by SHA-256.
- **Test suite**: 44 passed; one unrelated `pyreadline` deprecation warning.

## Fallacy scan

- **Coverage**: 11/11 statistical fallacy types checked.

| Fallacy | Status | Finding |
|---|---|---|
| Simpson's paradox | NOTE | Results are retained by \((q,D,\rho)\) cell; aggregate conclusions do not reverse all cellwise directions. |
| Ecological fallacy | NOTE | Mean and mean-square trajectory claims are explicitly separated. |
| Berkson's paradox | NOTE | Finite, noncontracting runs are retained rather than conditioning on successful trajectories. |
| Collider bias | NOTE | Policy comparisons use the same paths and no outcome-dependent covariate. |
| Base-rate neglect | NOTE | Crossing rates and continuous final errors are both reported. |
| Regression to the mean | NOTE | Initial state, horizon, and fresh seeds were fixed before the formal run. |
| Survivorship bias | NOTE | Crossed runs remain at the registered threshold in every aggregate. |
| Look-elsewhere effect | CAUTION | Post-hoc contraction diagnostics are clearly separated and require fresh-seed confirmation. |
| Garden of forking paths | NOTE | The failed crossing gates and overall FAIL are preserved. |
| Correlation versus causation | NOTE | Correlation sharing, delay, and step policy are controlled simulator interventions. |
| Reverse causality | NOTE | Outcomes cannot alter the fixed paths or registered policy step sizes. |

## Validated decision

Do not discard the joint correlation--delay idea.  Discard catastrophic
threshold crossing as the primary empirical definition of mean-square
stability.

Keep the unchanged scalar rule

\[
\eta_{\rm joint}^{-1}
=
\eta_{\rm mean}^{-1}
+K(q,\rho)/(2\mu)
\]

for one fresh-seed confirmation.  The primary endpoint must require
contraction of \(\mathbb E\|e_k\|^2\) or a finite-horizon upper confidence
bound, and must compare continuous paired error rather than only explosion.
If that confirmation fails, the parallel-sum rule should be abandoned rather
than retuned.


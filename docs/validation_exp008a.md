# Validation report: EXP-008A exact lifted boundary

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: validation_v1

## Experiment result

- **ID**: EXP-008A-exact-lifted-boundary
- **Type**: deterministic matrix-free spectral analysis
- **Status**: completed; preregistered overall verdict **FAIL (4/7)**
- **Executed source commit**: `ba05d36`
- **Formal command**:
  `python run_exact_lifted_boundary.py --output-dir results/exact_lifted_boundary`
- **Exact cells**: 12
- **Largest lifted operator dimension**: 17,424
- **Maximum eigensolver residual**: \(9.98\times10^{-11}\)
- **Anomalies**: none

## Preregistered findings

| Gate | Observed result | Verdict |
|---|---:|---|
| Independent numerical implementation | dense/matrix-free difference \(2.55\times10^{-15}\) | PASS |
| Boundary validity | 12/12 bracketed; all residuals below \(10^{-7}\) | PASS |
| Scalar-rule safety | 12/12 exact radii below one; maximum 0.9866 | PASS |
| Nonvacuous scalar tightness | 3/12 ratios at least 0.25; range 0.133--0.541 | FAIL |
| Correlation shrinks exact region | largest high/zero-correlation boundary ratio 0.586 | FAIL |
| Agent-count saturation at \(D=0\) | independent gain 1.176; high-correlation gain 1.003 | PASS |
| Mean stability insufficient | exact boundary at most half mean in 5/12 cells | FAIL |
| Overall | all seven gates required | **FAIL (4/7)** |

The formal failure is retained.  In particular, the scalar joint step cannot
be described as a uniformly tight approximation to the exact independent-time
boundary.

## Exact boundaries

| \(q\) | \(D\) | \(\rho\) | Mean boundary | Exact mean-square boundary | Joint step | Joint/exact |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0 | 0 | 4.210 | 3.156 | 0.437 | 0.139 |
| 16 | 0 | 0.9 | 4.210 | 0.511 | 0.0680 | 0.133 |
| 16 | 8 | 0 | 4.115 | 2.344 | 0.436 | 0.186 |
| 16 | 8 | 0.9 | 4.115 | 0.509 | 0.0679 | 0.134 |
| 16 | 32 | 0 | 1.293 | 1.014 | 0.354 | 0.349 |
| 16 | 32 | 0.9 | 1.293 | 0.429 | 0.0656 | 0.153 |
| 32 | 0 | 0 | 4.210 | 3.713 | 0.548 | 0.147 |
| 32 | 0 | 0.9 | 4.210 | 0.513 | 0.0682 | 0.133 |
| 32 | 8 | 0 | 2.103 | 1.694 | 0.485 | 0.286 |
| 32 | 8 | 0.9 | 2.103 | 0.473 | 0.0671 | 0.142 |
| 32 | 32 | 0 | 0.535 | 0.535 | 0.289 | 0.541 |
| 32 | 32 | 0.9 | 0.535 | 0.313 | 0.0613 | 0.196 |

## Mathematical conclusions

The exact lifted theorem is numerically confirmed.  The matrix-free operator
matches an independently materialized Kronecker matrix to machine precision,
and every first boundary has a valid below/above spectral bracket.

Three conclusions sharpen the paper claim.

First, correlation and participation interact exactly as predicted.  At zero
delay, increasing \(q\) from 16 to 32 enlarges the exact region by 17.64% under
independence and only 0.32% at \(\rho=0.9\).  Agent number therefore belongs in
the theorem, but common correlation removes almost all of its stability
benefit.

Second, delay and multiplicative noise are competing stability constraints.
At \(q=32,D=32,\rho=0\), the exact mean-square boundary is 0.53456 and the mean
boundary is 0.53458.  Delay dominates and the mean boundary is nearly exact.
At \(q=32,D=0,\rho=0.9\), the exact boundary is only 12.18% of the mean
boundary, so multiplicative common noise dominates.  The manuscript should
state a phase-dependent competition, not universal dominance by either term.

Third, the scalar joint rule is safe but conservative under temporally
independent sampling.  Its exact spectral radius remains below one in all
cells, but it uses only 13.3%--54.1% of the exact boundary.  This is acceptable
as a sufficient theorem candidate, but not as an exact stability
characterization.  The exact lifted operator and the scalar rule must be
presented as two distinct results.

## Relation to the Markov experiment

EXP-007D used temporally Markov trajectories and found the scalar step at
54.5%--97.0% of the largest fixed grid step having a 99% mean-contraction
upper limit.  EXP-008A uses independent stationary transition pairs and finds
a substantially larger exact region.  This difference is evidence that the
temporal Markov component is not captured by the same-time \(K(q,\rho)\)
alone.  A rigorous Markov theorem must retain a Markov jump operator or an
explicit mixing inflation; the independent-time exact boundary cannot be
reported as the Markov boundary.

## Reproducibility

- **Verdict**: REPRODUCIBLE
- **Method**: deterministic full rerun in a separate output path.
- **Artifacts**: 5/5 matched byte-for-byte by SHA-256.

## Fallacy scan

- **Coverage**: 11/11 statistical fallacy types checked.

| Fallacy | Status | Finding |
|---|---|---|
| Simpson's paradox | NOTE | Every \((q,D,\rho)\) cell is shown; dominance changes across delay regimes are retained. |
| Ecological fallacy | NOTE | Independent-time and Markov-data boundaries are not conflated. |
| Berkson's paradox | NOTE | No stable cell is selected or discarded based on the desired conclusion. |
| Collider bias | NOTE | Exact radii depend only on the registered operator inputs. |
| Base-rate neglect | NOTE | The audit uses exact spectral boundaries rather than rare crossing events. |
| Regression to the mean | NOTE | The experiment is deterministic and has no sampling regression. |
| Survivorship bias | NOTE | All 12 registered cells enter every all-cell gate. |
| Look-elsewhere effect | NOTE | Ratios and thresholds were fixed before the exact run. |
| Garden of forking paths | NOTE | All three failed gates and the overall failure are preserved. |
| Correlation versus causation | NOTE | \(\rho\), \(q\), and delay are explicit operator parameters. |
| Reverse causality | NOTE | Spectral outcomes do not alter registered operator construction. |

## Validated decision

Retain the main mechanism and exact lifted theorem.  Retain the scalar joint
rule only as a conservative low-complexity sufficient policy whose Markov
guarantee remains to be proved.

The central theoretical claim should be:

> Cross-agent dependence and heterogeneous delay enter different components
> of the exact mean-square operator; their competing margins determine whether
> additional agents enlarge the stable learning region.

This claim is exact for the independent-time lifted system and is directly
testable.  The next mandatory gate is the finite-state Markov jump extension,
followed by a tractable bound that does not materialize the joint mode space.


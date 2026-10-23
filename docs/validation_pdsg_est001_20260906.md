## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-06
- Verification Status: VERIFIED
- Version Label: pdsg_est001_validation_v1

# PDSG-EST-001 validation: sparse signed-alignment estimator

## Decision

The frozen local CPU audit passes E1--E8 and its clean deterministic
reproduction is byte-identical.  It establishes that the registered
expectation-level signed score is estimable and nonvacuous on the favorable
synthetic exact-sparse, sign-separated Markov class.  It authorizes only an
outcome-free standard-benchmark dependency-tail audit.  It does not authorize
a learning pilot, formal experiment, GPU job, or claim of improvement on a
standard MARL benchmark.

Overall confidence is **CAUTION** for transfer beyond the synthetic audit and
**SOLID** for the frozen gate computation itself.

## Provenance

- theorem commit: `c73b220`;
- estimator theorem commit: `c0f676c`;
- finite-action bound commit: `911a582`;
- original preregistration commit: `f3ac1cf`;
- Amendment 1 commit: `99eb49a`;
- frozen runner commit: `d5726e0137b25dcd2262e864b799aa71c522d890`;
- amended manifest SHA-256:
  `DE6A3847BEA601868B8E17837A792F324EEF9DF1BCD886C289F79602F3506FD0`;
- runner SHA-256:
  `71522F24605F0B349844E6D848BB3EBD7A5DAF470C31757076BF89436641CAB0`;
- local environment: conda `ust2`, Python 3.11.13;
- primary and reproduction summary SHA-256:
  `B09A977157742E97AD2C4B84CBE8B0EBA7B95FBA9B57E89C3530520A6BE01264`.

The ignored raw outputs are under
`tmp/policy_dependency_sync/pdsg_est001_primary` and
`tmp/policy_dependency_sync/pdsg_est001_reproduction`.

Regression was split across the environments already present on the machine.
The Python 3.11 `ust2` run produced `1467 passed, 7 skipped, 8 failed`; all
eight failures require the absent optional `minatar` package.  Four old
`cvxpy`-dependent files could not be collected in `ust2`; running exactly
those files in the existing Python 3.8 base environment, which contains
`cvxpy`, produced `26 passed`.  Thus the combined executable ledger is
`1493 passed, 7 skipped`, plus eight unresolved optional-MinAtar environment
failures.  It is not reported as an unconditional all-environment pass.

## Statistical findings

No null-hypothesis significance test or outcome-selected confidence interval
is used.  The experiment evaluates frozen deterministic thresholds over 288
model cells and 64 independent seeds per cell.

| Metric | Population | Observation | Frozen gate | Result |
|---|---|---:|---:|---:|
| finite and fully charged rows | all 18,432 seed rows | all valid; 104,448 action rows | all | pass |
| action error / registered bound | favorable action rows | `0.126439` mean | `<=1.0` | pass |
| best-action ranking | favorable | `1.000000` | `>=0.85` | pass |
| `2 e_uniform / margin` | favorable | `0.183289` median | `<=0.75` | pass |
| `2 e_uniform / margin` | adverse | `1.065730` median | descriptive phase side | -- |
| adverse/favorable ratio | frozen pair | `5.814485` | `>=1.5` | pass |
| ranking by degree | degree 2 / 4 / 8 | `1.0 / 1.0 / 1.0` | each `>=0.75` | pass |
| analytic direction | matched cells | decreases with `m`; increases with `rho` | both directions | pass |
| control/update accounting | every seed row | `m+m=2m`, disjoint IDs | exact | pass |

The empirical ranking accuracy is also 100% in the adverse and nonzero-tail
controls.  This does not invalidate E5: E5 concerns theorem-facing error
separation, not empirical failure.  It does show that the registered synthetic
action margins are easy relative to realized Gaussian noise, so the audit must
not be presented as a challenging learning benchmark.

## Gate ledger

| Gate | Result |
|---|---:|
| E1 validity, completeness, charging | pass |
| E2 empirical error below registered expectation bound | pass |
| E3 favorable ranking accuracy | pass |
| E4 favorable nonvacuity | pass |
| E5 mixing/sample-size phase separation | pass |
| E6 analytic monotonicity | pass |
| E7 every dependency degree | pass |
| E8 disjoint fully charged blocks | pass |
| E9 byte-identical reproduction | pass |

## Warnings

1. The favorable class assumes exact policy-dependency sparsity, local
   linearity, Gaussian AR(1) estimator noise, known correlation bound, and a
   true best--second-best score margin of at least 0.04.
2. The expectation bound is not a per-dispatch safety certificate.  It enters
   the cumulative finite-time theorem through the score-error sum.
3. The audit generates a fully charged independent update half, but the primary
   gates concern estimator ranking rather than accumulated policy learning.
4. PDSG-COMP-002's failed wall-clock scaling gates remain failed.  Structural
   derivative support is `O(Delta)`; degree-independent runtime is not claimed.
5. The local `ust2` environment lacks `minatar`, and the nonlinear requirements
   file does not currently declare it.  Eight legacy environment smoke tests
   therefore remain unexecuted successfully; no package was installed during
   this frozen validation.

## Fallacy scan

Coverage: **11/11** statistical and methodological fallacy types checked.

| Fallacy | Severity | Finding |
|---|---:|---|
| Simpson's paradox | NOTE | Aggregate favorable accuracy is not hiding a degree reversal; all three degrees equal 1.0. |
| Ecological fallacy | NOTE | Claims remain at synthetic scenario/action level; no individual-agent inference is made. |
| Berkson's paradox | CAUTION | The favorable class is deliberately sign-separated; conclusions are scoped to that preregistered selected class. |
| Collider bias | NOTE | No post-outcome covariate adjustment or regression control is used. |
| Base-rate neglect | NOTE | No diagnostic sensitivity or prevalence claim is made. |
| Regression to the mean | NOTE | Cells are not selected by noisy extreme pilot values. |
| Survivorship bias | NOTE | No row failed or was removed; all expected rows are present. |
| Look-elsewhere effect | NOTE | All eight mandatory gates are reported conjunctively; no p-value search is performed. |
| Garden of forking paths | NOTE | The original preregistration and pre-sample Amendment 1 are preserved with hashes; no post-sample changes occurred. |
| Correlation versus causation | NOTE | The AR correlation and score geometry are generated interventions; claims do not extend causally to benchmark return. |
| Reverse causality | NOTE | Control data precede action selection and the disjoint update block by construction. |

## Reproducibility

- Method: deterministic rerun with identical manifest, runner, seeds, and
  conda environment.
- Verdict: **REPRODUCIBLE**.
- `action_rows.csv`: byte-identical,
  SHA-256 `7F14E4896120212E720A39808C492763E88F25C4E5F784B6899DF5D8118105E0`.
- `seed_rows.csv`: byte-identical,
  SHA-256 `6D5FD1E17EE232481BCFAE5C0C22CF72A98C6864CBD3BBD7F21DA7B08551EB44`.
- `summary.json`: byte-identical,
  SHA-256 `B09A977157742E97AD2C4B84CBE8B0EBA7B95FBA9B57E89C3530520A6BE01264`.

## Next admissible work

The next gate is not another synthetic efficacy run.  It is an outcome-free
audit of candidate standard MARL tasks asking whether their true or certified
policy-gradient dependency tail is sparse/compressible and whether ordinary
critic/replay data provide a nontrivial signed margin at the measured
PDSG-COMP-002 compute cost.  Only a passing task may enter a separately
preregistered learning pilot.

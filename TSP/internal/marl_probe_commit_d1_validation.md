# TSP-MARL-DEV-002 validation report

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-10
- Verification Status: ANALYZED
- Version Label: validation_v1

## Scope and decision

TSP-MARL-DEV-002 is a development-only test of an observable Lyapunov
probe-then-commit controller on discrete PettingZoo MPE `simple_spread_v2` with
MAPPO.  It uses a fully charged, non-learning critic-residual probe to choose
`q=8` or `q=1`, then commits for the remaining dual budget.  Evaluation return
is never an input to the controller.

The frozen decision is **PASS: authorize a separate confirmation
preregistration**.  All nine mandatory development gates passed.  This report
does not turn the four development seeds into confirmatory evidence and does
not authorize placing the development curve in the manuscript.

## Design

- Preregistration commit: `208333bce5fa824c94ecec9a4628cf1aaa4c4297`.
- Pinned HARL commit: `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`.
- CTDE benchmark: parameter-shared decentralized actors and a centralized
  critic during training; deterministic decentralized evaluation.
- Coupling regimes: marginal-preserving independent and shared rollout streams.
- Training seeds: `94001--94004`; disjoint probe seeds: `95001--95004`.
- Methods: Lyapunov probe-then-commit, fixed `q=1`, and strong fixed `q=8`.
- Probe: 128 non-learning blocks, `q_probe=8`, rollout length 25.
- Budgets: 5,000,000 messages and 1,000,000 environment ticks, with server
  overhead 100 per rollout packet.
- Primary metric: deterministic team-return AUC over 21 equally spaced values
  of the exactly charged maximum dual-budget fraction.
- Registered runs: 24; observed unique runs: 24.

## Main result

The probe separated the two dependence regimes without using evaluation
return:

| Regime | `rho_hat` range | upper-proxy range | selected action |
|---|---:|---:|---:|
| Independent | -0.0338 to 0.00395 | 0 to 0.0346 | `q=8` in 4/4 seeds |
| Shared | 1.000 | 1.000 | `q=1` in 4/4 seeds |

Against the registered global strong fixed `q=8` comparator:

| Regime | Fixed-q8 mean AUC | Controller mean AUC | Relative change |
|---|---:|---:|---:|
| Independent | -73.8918 | -74.1975 | -0.4134% |
| Shared | -104.5617 | -97.8694 | +6.3388% |
| Equal-weight mixture | -- | -- | +2.9627% |

Thus the controller retained the strong participation action when rollout
innovations were independent and switched to low participation when common
randomness eliminated effective sample-size growth.  Its 0.413% independent
cost is below the registered 2% noninferiority margin; its shared and mixture
gains exceed the registered 5% and 2.5% thresholds.

The fully charged probe consumed 38,400 messages and 3,200 environment ticks
per controller run, i.e. 0.768% of the message budget.  After the probe, the
controller executed 16,538 `q=8` learning blocks in the independent regime and
39,692 `q=1` blocks in the shared regime.  Maximum scalar selection overhead
was `5.248e-7` of training time.

## Gate ledger

| Mandatory gate | Threshold | Observed | Status |
|---|---:|---:|---|
| Complete, finite, exact, clean lattice | 24/24 | 24/24 | pass |
| Independent selects q8 | at least 3/4 | 4/4 | pass |
| Shared selects q1 | at least 3/4 | 4/4 | pass |
| Independent relative AUC | at least -2% | -0.4134% | pass |
| Shared relative AUC | at least +5% | +6.3388% | pass |
| Mixture relative AUC | at least +2.5% | +2.9627% | pass |
| Probe message fraction | at most 1% | 0.768% | pass |
| Selection overhead fraction | at most 5% | `5.248e-7` | pass |
| Marginal-preserving coupling audit | pass | pass | pass |

## Execution and integrity

- Non-scientific smoke: `1845470_[0-1]`, both `COMPLETED 0:0`.
- Controller: `1845485_[0-7]`, all `COMPLETED 0:0`; elapsed 27:46--54:26.
- Fixed q1: `1845956_[8-15]`, all `COMPLETED 0:0`; elapsed 53:21--57:46.
- Fixed q8: `1846340_[16-23]`, all `COMPLETED 0:0`; elapsed 26:22--27:39.
- Every per-cell checksum passed and the fatal log scan was empty.
- The 752-file scientific manifest passed; its SHA-256 is
  `101269bd7b275593afb074c8f5e8226006af9072a523ac8ffbee45db14e02416`.
- Scientific artifacts occupy about 102 MB under the isolated scratch root.
- No experimental output was written to `/project` or `/home`.

The first analyzer invocation failed before producing outputs because pinned
NumPy 1.23 lacks `numpy.trapezoid`.  Commit `fdbd45d` added only a version
fallback to the mathematically identical `numpy.trapz`; no data, gate, metric,
seed, or aggregation changed.  The failed directory is retained and the
erratum is recorded in `marl_probe_commit_analysis_erratum.md`.

## Reproduction

The corrected analyzer was run twice in distinct directories.  Gate JSON,
curve CSV, and PDF were byte-identical:

- gate JSON: `0f12063dc9725308a2d1c24174360ece9ef6cd295a31a1bcbfdf0a710d585c04`;
- curve CSV: `2efaf908d50fcf3b5fc397a02eccfe4db10534902889a13dd22e0a9dcd81747c`;
- curve PDF: `79011665cc7d91f817cc3186ae4624ec37b87594612ef24621a4831ae4dd1779`.

The remote NumPy-1.23 targeted suite passed 13 tests.  Before execution, the
local TSP suite passed 33 tests and the repository regression passed 2,131
tests with seven skips.

## Interpretation and next action

This is the first standard policy-learning result in the TSP project showing
the intended positive mechanism against the global strong fixed action under
equal communication and environment budgets.  It supports, but does not by
itself prove, external transfer of the Lyapunov variance--resource phase rule.

The next authorized action is a separate confirmatory preregistration with new
training/probe seeds and fixed source hashes.  That preregistration must state
explicitly that the normal upper proxy is an empirical plug-in sensor for the
nonlinear benchmark, not a time-uniform Markov coverage theorem.  Confirmation
must test shared superiority and independent noninferiority against fixed
`q=8`; only a passed confirmation can supply a manuscript return figure.
